"""
Run Batch Calibration Tests

Simple script to run comprehensive calibration tests on 20+ videos from different competitions.

Usage:
    # Test 20 random videos
    python scripts/run_batch_calibration_tests.py --count 20

    # Test specific competition
    python scripts/run_batch_calibration_tests.py --competition chamonix_2024 --count 10

    # Test with race detection
    python scripts/run_batch_calibration_tests.py --count 20 --use-race-detection

    # Test specific videos
    python scripts/run_batch_calibration_tests.py --video-list test_videos.txt

Author: Speed Climbing Analysis Project
Date: 2025-11-15
"""

import argparse
import logging
import sys
from pathlib import Path
import random

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the test script's main functionality
from test_calibration_accuracy import CalibrationTester, find_test_videos

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def select_diverse_videos(
    segments_dir: str,
    count: int = 20,
    competition: str = None
) -> list:
    """Select a diverse set of videos from different competitions.

    Args:
        segments_dir: Race segments directory
        count: Number of videos to select
        competition: Specific competition (optional)

    Returns:
        List of selected video paths
    """
    segments_dir = Path(segments_dir)

    if competition:
        # Get all from one competition
        videos = list((segments_dir / competition).glob("*.mp4"))
        logger.info(f"Found {len(videos)} videos in {competition}")

        if len(videos) > count:
            selected = random.sample(videos, count)
        else:
            selected = videos

    else:
        # Get diverse selection from all competitions
        all_competitions = [d for d in segments_dir.iterdir() if d.is_dir()]

        videos_per_comp = count // len(all_competitions)
        remainder = count % len(all_competitions)

        selected = []

        for i, comp_dir in enumerate(all_competitions):
            comp_videos = list(comp_dir.glob("*.mp4"))

            # Distribute remainder across first competitions
            n_from_comp = videos_per_comp + (1 if i < remainder else 0)
            n_from_comp = min(n_from_comp, len(comp_videos))

            if n_from_comp > 0:
                comp_selected = random.sample(comp_videos, n_from_comp)
                selected.extend(comp_selected)
                logger.info(f"  {comp_dir.name}: selected {n_from_comp}/{len(comp_videos)} videos")

    logger.info(f"\nTotal selected: {len(selected)} videos")
    return [str(v) for v in selected]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive calibration tests on multiple race videos"
    )

    # Input options
    parser.add_argument(
        "--segments-dir",
        type=str,
        default="data/race_segments",
        help="Race segments directory (default: data/race_segments)"
    )
    parser.add_argument(
        "--competition",
        type=str,
        help="Test specific competition only"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of videos to test (default: 20)"
    )
    parser.add_argument(
        "--video-list",
        type=str,
        help="Text file with list of video paths (one per line)"
    )

    # Test configuration
    parser.add_argument(
        "--route-map",
        type=str,
        default="configs/ifsc_route_coordinates.json",
        help="Path to IFSC route map (default: configs/ifsc_route_coordinates.json)"
    )
    parser.add_argument(
        "--lane",
        type=str,
        default="left",
        choices=['left', 'right'],
        help="Lane to test (default: left)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/calibration/batch_calibration_test_report.json",
        help="Output report path"
    )

    # Frame selection
    parser.add_argument(
        "--use-race-detection",
        action="store_true",
        help="Use race start/finish detection for frame selection (RECOMMENDED)"
    )
    parser.add_argument(
        "--race-detection-method",
        type=str,
        default="fusion",
        choices=['audio', 'motion', 'fusion'],
        help="Race start detection method (default: fusion)"
    )

    args = parser.parse_args()

    # Select videos
    if args.video_list:
        logger.info(f"Loading video list from: {args.video_list}")
        with open(args.video_list, 'r') as f:
            video_paths = [line.strip() for line in f if line.strip()]
    else:
        logger.info(f"Selecting {args.count} diverse videos from {args.segments_dir}")
        if args.competition:
            logger.info(f"  Competition filter: {args.competition}")

        video_paths = select_diverse_videos(
            args.segments_dir,
            count=args.count,
            competition=args.competition
        )

    if not video_paths:
        logger.error("No videos selected")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH CALIBRATION TEST")
    logger.info(f"{'='*60}")
    logger.info(f"Videos to test: {len(video_paths)}")
    logger.info(f"Lane: {args.lane}")
    logger.info(f"Race detection: {'enabled' if args.use_race_detection else 'disabled'}")
    if args.use_race_detection:
        logger.info(f"  Method: {args.race_detection_method}")
    logger.info(f"Output: {args.output}")
    logger.info(f"{'='*60}\n")

    # Run tests
    tester = CalibrationTester(
        route_map_path=args.route_map,
        use_race_detection=args.use_race_detection,
        race_detection_method=args.race_detection_method
    )

    results = tester.test_multiple_videos(video_paths, lane=args.lane)

    if results:
        tester.generate_report(results, output_path=args.output)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TEST COMPLETE")
        logger.info("="*60)
        logger.info(f"Tested: {len(results)}/{len(video_paths)} videos")
        logger.info(f"Report: {args.output}")
    else:
        logger.error("No successful test results")


if __name__ == "__main__":
    main()
