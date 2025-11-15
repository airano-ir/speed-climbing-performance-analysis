"""
Batch Process Race Segments - End-to-End Pipeline

This script processes all 188 race segments through the complete pipeline:
1. Race boundary detection (start/finish)
2. Pose extraction (race frames only)
3. Calibration (periodic with frame selection)
4. Metrics calculation (meters)
5. Results aggregation

Usage:
    # Process all races
    python scripts/batch_process_races.py

    # Process specific competition
    python scripts/batch_process_races.py --competition chamonix_2024

    # Test with limited races
    python scripts/batch_process_races.py --max-races 20

    # Resume from previous run
    python scripts/batch_process_races.py --resume

Author: Speed Climbing Analysis Project
Date: 2025-11-15
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calibration.camera_calibration import PeriodicCalibrator
from phase1_pose_estimation.hold_detector import HoldDetector
from phase1_pose_estimation.race_start_detector import RaceStartDetector
from phase1_pose_estimation.race_finish_detector import RaceFinishDetector
from phase1_pose_estimation.dual_lane_detector import DualLaneDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RacePipeline:
    """End-to-end pipeline for processing race segments."""

    def __init__(
        self,
        route_map_path: str = "configs/ifsc_route_coordinates.json",
        output_dir: str = "data/processed",
        use_calibration: bool = True,
        calibration_interval: int = 30
    ):
        """Initialize pipeline.

        Args:
            route_map_path: Path to IFSC route coordinates
            output_dir: Output directory for processed data
            use_calibration: Whether to use camera calibration
            calibration_interval: Frames between calibrations (for periodic calibrator)
        """
        self.route_map_path = route_map_path
        self.output_dir = Path(output_dir)
        self.use_calibration = use_calibration
        self.calibration_interval = calibration_interval

        # Create output directories
        self.poses_dir = self.output_dir / "poses"
        self.calibration_dir = self.output_dir / "calibration"
        self.metrics_dir = self.output_dir / "metrics"
        self.reports_dir = self.output_dir / "reports"

        for dir_path in [self.poses_dir, self.calibration_dir, self.metrics_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.race_start_detector = RaceStartDetector(method='fusion')
        self.race_finish_detector = RaceFinishDetector(method='combined')
        self.dual_lane_detector = DualLaneDetector(method='fixed')

        if self.use_calibration:
            self.hold_detector = HoldDetector(
                route_coordinates_path=route_map_path,
                min_area=200,
                min_confidence=0.2
            )
            self.calibrator = PeriodicCalibrator(
                route_coordinates_path=route_map_path,
                calibration_interval=calibration_interval,
                min_holds_for_calibration=4,
                ransac_threshold=0.05
            )
        else:
            self.hold_detector = None
            self.calibrator = None

        logger.info("Pipeline initialized")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Calibration: {'enabled' if use_calibration else 'disabled'}")
        if use_calibration:
            logger.info(f"  Calibration interval: {calibration_interval} frames")

    def process_race(
        self,
        video_path: Path,
        race_metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Process a single race video.

        Args:
            video_path: Path to race video
            race_metadata: Optional metadata from race segmentation

        Returns:
            Processing result dictionary or None if failed
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {video_path.name}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"  Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")

        # Step 1: Detect race boundaries
        logger.info("  Step 1: Detecting race boundaries...")

        try:
            start_result = self.race_start_detector.detect_from_video(str(video_path))

            if start_result and start_result.confidence > 0.3:
                race_start = start_result.frame_id
                logger.info(f"    Start: frame {race_start} (conf={start_result.confidence:.2f})")
            else:
                logger.warning(f"    Start detection failed, using fallback")
                race_start = 30

            # Detect both lanes
            finish_left = self.race_finish_detector.detect_from_video(
                video_path, lane='left', start_frame=race_start
            )
            finish_right = self.race_finish_detector.detect_from_video(
                video_path, lane='right', start_frame=race_start
            )

            # Determine race end (earliest finish or fallback)
            race_end = total_frames - 30
            if finish_left and finish_left.confidence > 0.3:
                race_end = min(race_end, finish_left.frame_id)
                logger.info(f"    Finish (left): frame {finish_left.frame_id} (conf={finish_left.confidence:.2f})")
            if finish_right and finish_right.confidence > 0.3:
                race_end = min(race_end, finish_right.frame_id)
                logger.info(f"    Finish (right): frame {finish_right.frame_id} (conf={finish_right.confidence:.2f})")

            if race_end == total_frames - 30:
                logger.warning(f"    Finish detection failed, using fallback")

        except Exception as e:
            logger.error(f"    Race detection error: {e}")
            race_start = 30
            race_end = total_frames - 30

        logger.info(f"    Race frames: {race_start} to {race_end} ({race_end - race_start} frames)")

        # Step 2: Extract poses from race frames
        logger.info("  Step 2: Extracting poses...")

        poses_left = []
        poses_right = []
        calibrations = []

        frame_count = 0
        detection_count_left = 0
        detection_count_right = 0

        for frame_num in range(race_start, race_end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                break

            # Dual-lane pose detection
            lane_result = self.dual_lane_detector.process_frame(frame)

            if lane_result:
                poses_left.append({
                    'frame_id': frame_num,
                    'timestamp': frame_num / fps,
                    'keypoints': lane_result.left_pose.tolist() if lane_result.left_pose is not None else None,
                    'confidence': float(lane_result.left_confidence) if lane_result.left_pose is not None else 0.0
                })
                poses_right.append({
                    'frame_id': frame_num,
                    'timestamp': frame_num / fps,
                    'keypoints': lane_result.right_pose.tolist() if lane_result.right_pose is not None else None,
                    'confidence': float(lane_result.right_confidence) if lane_result.right_pose is not None else 0.0
                })

                if lane_result.left_pose is not None:
                    detection_count_left += 1
                if lane_result.right_pose is not None:
                    detection_count_right += 1
            else:
                poses_left.append({
                    'frame_id': frame_num,
                    'timestamp': frame_num / fps,
                    'keypoints': None,
                    'confidence': 0.0
                })
                poses_right.append({
                    'frame_id': frame_num,
                    'timestamp': frame_num / fps,
                    'keypoints': None,
                    'confidence': 0.0
                })

            # Step 3: Periodic calibration
            if self.use_calibration and frame_num % self.calibration_interval == 0:
                for lane, lane_name in [('left', 'left'), ('right', 'right')]:
                    detected_holds = self.hold_detector.detect_holds(frame, lane=lane_name)

                    if len(detected_holds) >= 4:
                        calibration = self.calibrator.calibrate(frame, detected_holds, lane=lane_name)

                        if calibration:
                            calibrations.append({
                                'frame_id': frame_num,
                                'lane': lane_name,
                                'rmse_cm': calibration.rmse_error * 100,
                                'holds_used': calibration.inlier_count,
                                'confidence': calibration.confidence,
                                'scale': calibration.pixel_to_meter_scale
                            })

            frame_count += 1

        cap.release()

        detection_rate_left = detection_count_left / frame_count if frame_count > 0 else 0
        detection_rate_right = detection_count_right / frame_count if frame_count > 0 else 0

        logger.info(f"    Poses extracted: {frame_count} frames")
        logger.info(f"    Detection rate: Left={detection_rate_left*100:.1f}%, Right={detection_rate_right*100:.1f}%")

        if self.use_calibration and calibrations:
            avg_rmse = np.mean([c['rmse_cm'] for c in calibrations])
            logger.info(f"    Calibrations: {len(calibrations)}, avg RMSE={avg_rmse:.1f}cm")

        # Save outputs
        competition = video_path.parent.name
        race_name = video_path.stem

        # Poses
        poses_output = {
            'video_name': video_path.name,
            'competition': competition,
            'race_name': race_name,
            'fps': fps,
            'resolution': [width, height],
            'race_bounds': [race_start, race_end],
            'poses_left': poses_left,
            'poses_right': poses_right,
            'detection_rate_left': detection_rate_left,
            'detection_rate_right': detection_rate_right
        }

        poses_file = self.poses_dir / competition / f"{race_name}_poses.json"
        poses_file.parent.mkdir(parents=True, exist_ok=True)
        with open(poses_file, 'w') as f:
            json.dump(poses_output, f, indent=2)

        # Calibration
        if calibrations:
            cal_file = self.calibration_dir / competition / f"{race_name}_calibration.json"
            cal_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cal_file, 'w') as f:
                json.dump(calibrations, f, indent=2)

        elapsed = time.time() - start_time
        logger.info(f"  Completed in {elapsed:.1f}s")

        # Return summary
        return {
            'video_name': video_path.name,
            'competition': competition,
            'race_name': race_name,
            'success': True,
            'frames_processed': frame_count,
            'race_bounds': [race_start, race_end],
            'detection_rate_left': detection_rate_left,
            'detection_rate_right': detection_rate_right,
            'calibrations': len(calibrations) if calibrations else 0,
            'processing_time_s': elapsed,
            'outputs': {
                'poses': str(poses_file),
                'calibration': str(cal_file) if calibrations else None
            }
        }

    def process_batch(
        self,
        video_paths: List[Path],
        resume: bool = False
    ) -> Dict:
        """Process multiple races.

        Args:
            video_paths: List of video paths
            resume: Skip already processed videos

        Returns:
            Batch processing summary
        """
        results = []
        failed = []
        skipped = []

        start_time = time.time()

        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"\n[{i}/{len(video_paths)}] Processing: {video_path.name}")

            # Check if already processed
            competition = video_path.parent.name
            race_name = video_path.stem
            poses_file = self.poses_dir / competition / f"{race_name}_poses.json"

            if resume and poses_file.exists():
                logger.info(f"  Skipping (already processed)")
                skipped.append(str(video_path))
                continue

            try:
                result = self.process_race(video_path)

                if result:
                    results.append(result)
                else:
                    failed.append(str(video_path))

            except Exception as e:
                logger.error(f"  Error processing {video_path.name}: {e}")
                failed.append(str(video_path))

        elapsed = time.time() - start_time

        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': len(video_paths),
            'processed': len(results),
            'failed': len(failed),
            'skipped': len(skipped),
            'total_time_s': elapsed,
            'avg_time_per_video_s': elapsed / len(results) if results else 0,
            'results': results,
            'failed_videos': failed,
            'skipped_videos': skipped
        }

        # Save summary
        summary_file = self.reports_dir / "batch_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total: {len(video_paths)} videos")
        logger.info(f"Processed: {len(results)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Skipped: {len(skipped)}")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Avg per video: {summary['avg_time_per_video_s']:.1f}s")
        logger.info(f"\nSummary saved to: {summary_file}")

        return summary


def find_race_videos(
    segments_dir: str,
    competition: Optional[str] = None,
    max_races: Optional[int] = None
) -> List[Path]:
    """Find race segment videos.

    Args:
        segments_dir: Race segments directory
        competition: Specific competition name (optional)
        max_races: Maximum number of races to process (optional)

    Returns:
        List of video paths
    """
    segments_dir = Path(segments_dir)

    if not segments_dir.exists():
        logger.error(f"Directory not found: {segments_dir}")
        return []

    # Find videos
    if competition:
        pattern = f"{competition}/*.mp4"
        videos = list(segments_dir.glob(pattern))
        logger.info(f"Found {len(videos)} videos in {competition}")
    else:
        videos = list(segments_dir.glob("*/*.mp4"))
        logger.info(f"Found {len(videos)} videos across all competitions")

    # Limit if requested
    if max_races and len(videos) > max_races:
        videos = videos[:max_races]
        logger.info(f"Limited to first {max_races} videos")

    return sorted(videos)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process race segments through complete pipeline"
    )

    parser.add_argument(
        "--segments-dir",
        type=str,
        default="data/race_segments",
        help="Race segments directory"
    )
    parser.add_argument(
        "--competition",
        type=str,
        help="Process specific competition only (e.g., chamonix_2024)"
    )
    parser.add_argument(
        "--max-races",
        type=int,
        help="Maximum number of races to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory"
    )
    parser.add_argument(
        "--route-map",
        type=str,
        default="configs/ifsc_route_coordinates.json",
        help="IFSC route map path"
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable camera calibration"
    )
    parser.add_argument(
        "--calibration-interval",
        type=int,
        default=30,
        help="Frames between calibrations (default: 30)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed videos"
    )

    args = parser.parse_args()

    # Find videos
    video_paths = find_race_videos(
        args.segments_dir,
        competition=args.competition,
        max_races=args.max_races
    )

    if not video_paths:
        logger.error("No videos found")
        return

    # Initialize pipeline
    pipeline = RacePipeline(
        route_map_path=args.route_map,
        output_dir=args.output_dir,
        use_calibration=not args.no_calibration,
        calibration_interval=args.calibration_interval
    )

    # Process
    summary = pipeline.process_batch(video_paths, resume=args.resume)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
