"""
Batch Camera Calibration Script
================================
Generates calibration files for all race segments using:
- HoldDetector (HSV-based red hold detection)
- PeriodicCalibrator (efficient per-30-frames calibration)
- IFSC route map (31 standard holds)

Usage:
    python scripts/batch_calibration.py
    python scripts/batch_calibration.py --competition chamonix_2024
    python scripts/batch_calibration.py --max-races 10 --test

Author: Speed Climbing Performance Analysis Project
Date: 2025-11-15
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import List, Dict, Optional
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calibration.camera_calibration import PeriodicCalibrator
from phase1_pose_estimation.hold_detector import HoldDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchCalibrator:
    """Batch calibration for race segments."""

    def __init__(
        self,
        race_segments_dir: str = "data/race_segments",
        output_dir: str = "data/processed/calibration",
        ifsc_map_path: str = "configs/ifsc_route_coordinates.json"
    ):
        self.race_segments_dir = Path(race_segments_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ifsc_map_path = ifsc_map_path

        logger.info(f"IFSC route map: {ifsc_map_path}")

        # Hold detector
        self.hold_detector = HoldDetector(
            min_confidence=0.2,  # Lower threshold for better detection
            min_area=50
        )

    def calibrate_race(
        self,
        video_path: Path,
        output_path: Path
    ) -> Optional[Dict]:
        """Calibrate a single race video."""

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"  Video: {total_frames} frames @ {fps:.1f} fps")

        # Create periodic calibrator (every 30 frames = 1 second)
        calibrator = PeriodicCalibrator(
            route_coordinates_path=self.ifsc_map_path,
            recalibration_interval=30,
            min_holds_for_calibration=4
        )

        # Process frames
        frame_count = 0
        calibrations = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect holds in frame
            detected_holds = self.hold_detector.detect_holds(frame)

            # Calibrate (uses caching internally, processes both lanes)
            # For race segments, we calibrate using left lane by default
            result = calibrator.calibrate_frame(
                frame,
                frame_count,
                detected_holds,
                lane='left'
            )

            if result is not None:
                calibrations.append({
                    'frame_id': frame_count,
                    'rmse': result.rmse_error,
                    'holds_detected': result.total_holds,
                    'holds_used': result.inlier_count,
                    'inlier_ratio': result.inlier_ratio,
                    'confidence': result.confidence
                })

            frame_count += 1

            # Progress
            if frame_count % 30 == 0:
                progress = frame_count / total_frames * 100
                logger.info(f"    Frame {frame_count}/{total_frames} ({progress:.1f}%)")

        cap.release()

        # Get final calibration result (use last successful calibration)
        final_calibration = calibrator.last_calibration

        if final_calibration is None:
            logger.error(f"  Calibration failed for {video_path.name} - no successful calibrations")
            return None

        # Save calibration
        calibrator.save_calibration(
            final_calibration,
            output_path,
            metadata={'video_path': str(video_path), 'total_frames': total_frames}
        )

        # Statistics
        if len(calibrations) > 0:
            avg_rmse = sum(c['rmse'] for c in calibrations) / len(calibrations)
            avg_holds = sum(c['holds_used'] for c in calibrations) / len(calibrations)
        else:
            avg_rmse = 0
            avg_holds = 0

        logger.info(f"  ✓ Calibration saved: {output_path.name}")
        logger.info(f"    RMSE: {avg_rmse:.2f}cm")
        logger.info(f"    Holds used (avg): {avg_holds:.1f}")
        logger.info(f"    Calibrations performed: {len(calibrations)}")

        return {
            'video_path': str(video_path),
            'output_path': str(output_path),
            'avg_rmse': avg_rmse,
            'avg_holds_used': avg_holds,
            'calibrations_performed': len(calibrations),
            'success': True
        }

    def run(
        self,
        competition: Optional[str] = None,
        max_races: Optional[int] = None,
        force: bool = False
    ):
        """Run batch calibration.

        Args:
            competition: Specific competition (optional)
            max_races: Maximum number of races (optional)
            force: Overwrite existing calibration files
        """

        # Find race videos
        if competition:
            pattern = f"{competition}/*.mp4"
        else:
            pattern = "*/*.mp4"

        race_videos = sorted(self.race_segments_dir.glob(pattern))

        if max_races:
            race_videos = race_videos[:max_races]

        logger.info(f"Found {len(race_videos)} race videos to calibrate")

        # Process each
        results = []
        start_time = time.time()

        for idx, video_path in enumerate(race_videos, 1):
            logger.info(f"\n[{idx}/{len(race_videos)}] {video_path.name}")

            # Output path
            competition_name = video_path.parent.name
            race_name = video_path.stem
            output_dir = self.output_dir / competition_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{race_name}_calibration.json"

            # Skip if exists
            if output_path.exists() and not force:
                logger.info(f"  Skipping (already calibrated): {output_path.name}")
                continue

            # Calibrate
            result = self.calibrate_race(video_path, output_path)
            if result:
                results.append(result)

        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("BATCH CALIBRATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total races: {len(race_videos)}")
        logger.info(f"Calibrated: {len(results)}")
        logger.info(f"Time: {elapsed/60:.1f} minutes")
        if len(results) > 0:
            logger.info(f"Avg per race: {elapsed/len(results):.1f}s")

        # Save summary
        summary_path = self.output_dir / "batch_calibration_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total': len(race_videos),
                'calibrated': len(results),
                'elapsed_seconds': elapsed,
                'results': results
            }, f, indent=2)

        logger.info(f"\nSummary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch calibration for race segments")
    parser.add_argument('--competition', type=str, help="Specific competition (e.g., chamonix_2024)")
    parser.add_argument('--max-races', type=int, help="Limit number of races")
    parser.add_argument('--test', action='store_true', help="Test mode (max 5 races)")
    parser.add_argument('--force', action='store_true', help="Overwrite existing calibration files")

    args = parser.parse_args()

    if args.test:
        args.max_races = 5

    calibrator = BatchCalibrator()
    calibrator.run(
        competition=args.competition,
        max_races=args.max_races,
        force=args.force
    )


if __name__ == "__main__":
    main()
