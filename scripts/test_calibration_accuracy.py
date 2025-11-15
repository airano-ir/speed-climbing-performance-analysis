"""Test calibration accuracy across multiple videos.

This script tests the calibration system on multiple videos from different
competitions to validate RMSE and accuracy metrics.

Usage:
    python scripts/test_calibration_accuracy.py --videos-dir data/race_segments
    python scripts/test_calibration_accuracy.py --video-list videos_to_test.txt
    python scripts/test_calibration_accuracy.py --count 10  # Test 10 random videos
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict
import random

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calibration.camera_calibration import CameraCalibrator, PeriodicCalibrator
from phase1_pose_estimation.hold_detector import HoldDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalibrationTester:
    """Test calibration accuracy on multiple videos."""

    def __init__(
        self,
        route_map_path: str = "configs/ifsc_route_coordinates.json",
        test_frames: List[int] = [30, 60, 90, 120, 150],
        skip_start_frames: int = 0,
        skip_end_frames: int = 0
    ):
        """Initialize calibration tester.

        Args:
            route_map_path: Path to IFSC route coordinates
            test_frames: Frame numbers to test in each video
            skip_start_frames: Skip first N frames (pre-race section)
            skip_end_frames: Skip last N frames (post-race section)
        """
        self.route_map_path = route_map_path
        self.test_frames = test_frames
        self.skip_start_frames = skip_start_frames
        self.skip_end_frames = skip_end_frames

        # Initialize detector and calibrator
        self.hold_detector = HoldDetector(
            route_coordinates_path=route_map_path,
            min_area=200,
            min_confidence=0.2
        )

        self.calibrator = CameraCalibrator(
            route_coordinates_path=route_map_path,
            min_holds_for_calibration=4,
            ransac_threshold=0.05
        )

        self.results = []

    def test_video(
        self,
        video_path: str,
        lane: str = 'left'
    ) -> Dict:
        """Test calibration on a single video.

        Args:
            video_path: Path to video file
            lane: Lane to test ('left' or 'right')

        Returns:
            Dictionary with test results
        """
        video_path = Path(video_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {video_path.name}")
        logger.info(f"{'='*60}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Apply frame selection (skip pre/post race sections)
        effective_start = self.skip_start_frames
        effective_end = total_frames - self.skip_end_frames

        if self.skip_start_frames > 0 or self.skip_end_frames > 0:
            logger.info(
                f"  Frame selection: testing frames {effective_start} to {effective_end} "
                f"(skipping first {self.skip_start_frames} and last {self.skip_end_frames})"
            )

        # Adjust test frames based on video length and skip settings
        valid_test_frames = [
            f for f in self.test_frames
            if effective_start <= f < effective_end
        ]
        if not valid_test_frames:
            # If video too short or all frames skipped, use middle frame
            middle_frame = (effective_start + effective_end) // 2
            if effective_start < effective_end:
                valid_test_frames = [middle_frame]
            else:
                logger.warning(f"  No valid frames after skip settings")
                cap.release()
                return None

        frame_results = []

        for frame_num in valid_test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Could not read frame {frame_num}")
                continue

            # Detect holds
            detected_holds = self.hold_detector.detect_holds(frame, lane=lane)

            if len(detected_holds) == 0:
                logger.warning(f"Frame {frame_num}: No holds detected")
                continue

            # Calibrate
            calibration = self.calibrator.calibrate(frame, detected_holds, lane=lane)

            if calibration:
                frame_result = {
                    "frame_num": int(frame_num),
                    "timestamp": float(frame_num / fps if fps > 0 else 0),
                    "holds_detected": int(len(detected_holds)),
                    "holds_used": int(calibration.inlier_count),
                    "inlier_ratio": float(calibration.inlier_ratio),
                    "rmse_m": float(calibration.rmse_error),
                    "rmse_cm": float(calibration.rmse_error * 100),
                    "scale_px_per_m": float(calibration.pixel_to_meter_scale),
                    "confidence": float(calibration.confidence)
                }

                frame_results.append(frame_result)

                logger.info(
                    f"  Frame {frame_num:4d}: "
                    f"{len(detected_holds):2d} detected, "
                    f"{calibration.inlier_count:2d} used, "
                    f"RMSE={calibration.rmse_error*100:5.1f}cm, "
                    f"conf={calibration.confidence:.2f}"
                )
            else:
                logger.warning(f"  Frame {frame_num}: Calibration failed")

        cap.release()

        if not frame_results:
            logger.error(f"No successful calibrations for {video_path.name}")
            return None

        # Calculate statistics
        rmse_values = [r['rmse_m'] for r in frame_results]
        confidence_values = [r['confidence'] for r in frame_results]
        holds_detected = [r['holds_detected'] for r in frame_results]
        holds_used = [r['holds_used'] for r in frame_results]

        video_result = {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "competition": self._extract_competition(video_path.name),
            "total_frames": int(total_frames),
            "fps": float(fps),
            "lane": lane,
            "frames_tested": len(frame_results),
            "frame_results": frame_results,
            "statistics": {
                "rmse_mean_cm": float(np.mean(rmse_values) * 100),
                "rmse_std_cm": float(np.std(rmse_values) * 100),
                "rmse_min_cm": float(np.min(rmse_values) * 100),
                "rmse_max_cm": float(np.max(rmse_values) * 100),
                "rmse_median_cm": float(np.median(rmse_values) * 100),
                "confidence_mean": float(np.mean(confidence_values)),
                "confidence_std": float(np.std(confidence_values)),
                "holds_detected_mean": float(np.mean(holds_detected)),
                "holds_used_mean": float(np.mean(holds_used)),
                "pass_rate_10cm": float(sum(1 for r in rmse_values if r <= 0.10) / len(rmse_values)),
                "pass_rate_8cm": float(sum(1 for r in rmse_values if r <= 0.08) / len(rmse_values)),
                "pass_rate_5cm": float(sum(1 for r in rmse_values if r <= 0.05) / len(rmse_values)),
            }
        }

        # Log summary
        stats = video_result['statistics']
        logger.info(f"\nSummary for {video_path.name}:")
        logger.info(f"  RMSE: {stats['rmse_mean_cm']:.1f} ± {stats['rmse_std_cm']:.1f} cm")
        logger.info(f"  Range: {stats['rmse_min_cm']:.1f} - {stats['rmse_max_cm']:.1f} cm")
        logger.info(f"  Pass rate (≤10cm): {stats['pass_rate_10cm']*100:.0f}%")
        logger.info(f"  Pass rate (≤8cm):  {stats['pass_rate_8cm']*100:.0f}%")
        logger.info(f"  Pass rate (≤5cm):  {stats['pass_rate_5cm']*100:.0f}%")
        logger.info(f"  Holds detected: {stats['holds_detected_mean']:.1f} avg")
        logger.info(f"  Confidence: {stats['confidence_mean']:.2f} avg")

        return video_result

    def test_multiple_videos(
        self,
        video_paths: List[str],
        lane: str = 'left'
    ) -> List[Dict]:
        """Test calibration on multiple videos.

        Args:
            video_paths: List of paths to video files
            lane: Lane to test

        Returns:
            List of video result dictionaries
        """
        results = []

        for video_path in video_paths:
            result = self.test_video(video_path, lane=lane)
            if result:
                results.append(result)

        return results

    def generate_report(
        self,
        results: List[Dict],
        output_path: str = "data/processed/calibration/calibration_test_report.json"
    ):
        """Generate and save test report.

        Args:
            results: List of video test results
            output_path: Path to output JSON report
        """
        if not results:
            logger.error("No results to report")
            return

        # Calculate overall statistics
        all_rmse = []
        all_confidence = []
        all_holds_detected = []
        all_holds_used = []

        for result in results:
            for frame_result in result['frame_results']:
                all_rmse.append(frame_result['rmse_m'])
                all_confidence.append(frame_result['confidence'])
                all_holds_detected.append(frame_result['holds_detected'])
                all_holds_used.append(frame_result['holds_used'])

        overall_stats = {
            "total_videos_tested": len(results),
            "total_frames_tested": len(all_rmse),
            "rmse_overall_mean_cm": float(np.mean(all_rmse) * 100),
            "rmse_overall_std_cm": float(np.std(all_rmse) * 100),
            "rmse_overall_min_cm": float(np.min(all_rmse) * 100),
            "rmse_overall_max_cm": float(np.max(all_rmse) * 100),
            "rmse_overall_median_cm": float(np.median(all_rmse) * 100),
            "confidence_overall_mean": float(np.mean(all_confidence)),
            "holds_detected_overall_mean": float(np.mean(all_holds_detected)),
            "holds_used_overall_mean": float(np.mean(all_holds_used)),
            "pass_rate_10cm_overall": float(sum(1 for r in all_rmse if r <= 0.10) / len(all_rmse)),
            "pass_rate_8cm_overall": float(sum(1 for r in all_rmse if r <= 0.08) / len(all_rmse)),
            "pass_rate_5cm_overall": float(sum(1 for r in all_rmse if r <= 0.05) / len(all_rmse)),
        }

        # Build report
        report = {
            "test_info": {
                "route_map": self.route_map_path,
                "test_frames_per_video": self.test_frames,
                "min_confidence": self.hold_detector.min_confidence,
                "min_holds": self.calibrator.min_holds,
                "ransac_threshold_m": self.calibrator.ransac_threshold
            },
            "overall_statistics": overall_stats,
            "video_results": results,
            "assessment": self._assess_results(overall_stats)
        }

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"OVERALL TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Videos tested: {overall_stats['total_videos_tested']}")
        logger.info(f"Frames tested: {overall_stats['total_frames_tested']}")
        logger.info(f"")
        logger.info(f"RMSE: {overall_stats['rmse_overall_mean_cm']:.1f} ± "
                   f"{overall_stats['rmse_overall_std_cm']:.1f} cm")
        logger.info(f"Range: {overall_stats['rmse_overall_min_cm']:.1f} - "
                   f"{overall_stats['rmse_overall_max_cm']:.1f} cm")
        logger.info(f"Median: {overall_stats['rmse_overall_median_cm']:.1f} cm")
        logger.info(f"")
        logger.info(f"Pass rates:")
        logger.info(f"  ≤10cm: {overall_stats['pass_rate_10cm_overall']*100:.1f}%")
        logger.info(f"  ≤8cm:  {overall_stats['pass_rate_8cm_overall']*100:.1f}%")
        logger.info(f"  ≤5cm:  {overall_stats['pass_rate_5cm_overall']*100:.1f}%")
        logger.info(f"")
        logger.info(f"Holds detected: {overall_stats['holds_detected_overall_mean']:.1f} avg")
        logger.info(f"Holds used:     {overall_stats['holds_used_overall_mean']:.1f} avg")
        logger.info(f"Confidence:     {overall_stats['confidence_overall_mean']:.2f} avg")
        logger.info(f"")
        logger.info(f"Assessment: {report['assessment']['verdict']}")
        logger.info(f"{'='*60}")
        logger.info(f"\nReport saved to: {output_path}")

    def _extract_competition(self, filename: str) -> str:
        """Extract competition name from filename."""
        filename_lower = filename.lower()
        competitions = ['villars', 'chamonix', 'seoul', 'innsbruck', 'zilina']

        for comp in competitions:
            if comp in filename_lower:
                return comp.capitalize()

        return "Unknown"

    def _assess_results(self, stats: Dict) -> Dict:
        """Assess calibration quality based on statistics.

        Args:
            stats: Overall statistics dictionary

        Returns:
            Assessment dictionary with verdict and recommendations
        """
        mean_rmse = stats['rmse_overall_mean_cm']
        pass_rate_8cm = stats['pass_rate_8cm_overall']
        holds_used = stats['holds_used_overall_mean']

        # Verdict
        if mean_rmse <= 5.0 and pass_rate_8cm >= 0.9:
            verdict = "EXCELLENT - Production ready"
            status = "✅"
        elif mean_rmse <= 8.0 and pass_rate_8cm >= 0.7:
            verdict = "GOOD - Acceptable for most analyses"
            status = "✓"
        elif mean_rmse <= 10.0:
            verdict = "MARGINAL - Needs improvement"
            status = "⚠️"
        else:
            verdict = "POOR - Significant issues"
            status = "❌"

        # Recommendations
        recommendations = []

        if mean_rmse > 8.0:
            recommendations.append(
                "High RMSE: Increase hold detection count (lower min_confidence or improve detection algorithm)"
            )

        if holds_used < 10:
            recommendations.append(
                f"Low hold count ({holds_used:.1f} avg): Aim for 15-20 holds per frame"
            )

        if pass_rate_8cm < 0.7:
            recommendations.append(
                f"Low pass rate ({pass_rate_8cm*100:.0f}%): Improve calibration consistency"
            )

        if stats['confidence_overall_mean'] < 0.6:
            recommendations.append(
                "Low confidence: Review RANSAC parameters and hold matching algorithm"
            )

        if not recommendations:
            recommendations.append("No major issues detected. Continue monitoring performance.")

        return {
            "verdict": verdict,
            "status": status,
            "mean_rmse_cm": mean_rmse,
            "pass_rate_8cm_percent": pass_rate_8cm * 100,
            "recommendations": recommendations
        }


def find_test_videos(
    videos_dir: str,
    count: int = 10,
    pattern: str = "**/*.mp4"
) -> List[str]:
    """Find test videos from directory.

    Args:
        videos_dir: Directory to search
        count: Number of videos to select (random sampling)
        pattern: Glob pattern for video files

    Returns:
        List of video paths
    """
    videos_dir = Path(videos_dir)

    if not videos_dir.exists():
        logger.error(f"Directory not found: {videos_dir}")
        return []

    # Find all videos
    all_videos = list(videos_dir.glob(pattern))

    if not all_videos:
        logger.error(f"No videos found matching {pattern} in {videos_dir}")
        return []

    # Sample randomly
    if len(all_videos) <= count:
        selected = all_videos
    else:
        selected = random.sample(all_videos, count)

    logger.info(f"Found {len(all_videos)} videos, selected {len(selected)} for testing")

    return [str(v) for v in selected]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test calibration accuracy on multiple videos"
    )

    # Input options
    parser.add_argument(
        "--videos-dir",
        type=str,
        help="Directory containing videos (will randomly sample)"
    )
    parser.add_argument(
        "--video-list",
        type=str,
        help="Text file with list of video paths (one per line)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of videos to test (default: 10)"
    )

    # Configuration
    parser.add_argument(
        "--route-map",
        type=str,
        default="configs/ifsc_route_coordinates.json",
        help="Path to route map JSON"
    )
    parser.add_argument(
        "--lane",
        type=str,
        default="left",
        choices=['left', 'right'],
        help="Lane to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/calibration/calibration_test_report.json",
        help="Output report path"
    )

    # Frame selection (for skipping pre/post race sections)
    parser.add_argument(
        "--skip-start",
        type=int,
        default=0,
        help="Skip first N frames (pre-race section, default: 0)"
    )
    parser.add_argument(
        "--skip-end",
        type=int,
        default=0,
        help="Skip last N frames (post-race section, default: 0)"
    )

    args = parser.parse_args()

    # Get video list
    if args.video_list:
        with open(args.video_list, 'r') as f:
            video_paths = [line.strip() for line in f if line.strip()]
    elif args.videos_dir:
        video_paths = find_test_videos(args.videos_dir, count=args.count)
    else:
        # Default: test from race_segments
        default_dir = "data/race_segments"
        logger.info(f"No input specified, using default directory: {default_dir}")
        video_paths = find_test_videos(default_dir, count=args.count)

    if not video_paths:
        logger.error("No videos to test")
        return

    # Run tests
    tester = CalibrationTester(
        route_map_path=args.route_map,
        skip_start_frames=args.skip_start,
        skip_end_frames=args.skip_end
    )
    results = tester.test_multiple_videos(video_paths, lane=args.lane)

    if results:
        tester.generate_report(results, output_path=args.output)
    else:
        logger.error("No successful test results")


if __name__ == "__main__":
    main()
