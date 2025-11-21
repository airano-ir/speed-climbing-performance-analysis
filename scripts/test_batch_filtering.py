#!/usr/bin/env python3
"""
Batch test spatial filtering on multiple race videos
"""

import sys
from pathlib import Path
import cv2
import json

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.vision.holds import HoldDetector
from speed_climbing.vision.calibration import CameraCalibrator


def test_race(video_path: Path, route_map_path: str):
    """Test spatial filtering on a single race video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # Initialize components
    hold_detector = HoldDetector(
        route_coordinates_path=route_map_path,
        use_spatial_filtering=False
    )
    calibrator = CameraCalibrator(route_coordinates_path=route_map_path)

    # Step 1: Detect holds
    initial_holds = hold_detector.detect_holds(frame)

    # Step 2: Calibrate
    calibration_result = calibrator.calibrate(frame, initial_holds)

    if not calibration_result or calibration_result.homography_matrix is None:
        # Try relaxed parameters
        hold_detector_relaxed = HoldDetector(
            route_coordinates_path=route_map_path,
            min_area=300,
            min_confidence=0.3,
            use_spatial_filtering=False
        )
        initial_holds = hold_detector_relaxed.detect_holds(frame)
        calibration_result = calibrator.calibrate(frame, initial_holds)

        if not calibration_result:
            return {
                'status': 'calibration_failed',
                'initial_holds': len(initial_holds),
                'filtered_holds': 0,
                'removed': 0
            }

    # Step 3: Spatial filtering
    homography = calibration_result.homography_matrix
    filtered_holds = hold_detector.filter_by_spatial_grid(
        initial_holds,
        homography,
        frame.shape[:2]
    )

    return {
        'status': 'success',
        'initial_holds': len(initial_holds),
        'filtered_holds': len(filtered_holds),
        'removed': len(initial_holds) - len(filtered_holds),
        'calibration_confidence': calibration_result.confidence,
        'inliers': calibration_result.inlier_count
    }


def main():
    route_map_path = 'configs/ifsc_route_coordinates.json'
    race_dir = Path('data/race_segments/innsbruck_2024')

    race_videos = sorted(race_dir.glob('Speed_finals_Innsbruck_2024_race*.mp4'))

    print(f"{'='*80}")
    print(f"Batch Spatial Filtering Test")
    print(f"{'='*80}\n")
    print(f"Testing {len(race_videos)} race videos...\n")

    results = []
    for video_path in race_videos:
        race_name = video_path.stem
        print(f"Testing {race_name}...", end=' ')

        result = test_race(video_path, route_map_path)

        if result is None:
            print("❌ Failed to read video")
            continue

        results.append({
            'race': race_name,
            **result
        })

        if result['status'] == 'success':
            print(f"✅ {result['initial_holds']} → {result['filtered_holds']} "
                  f"(removed {result['removed']})")
        else:
            print(f"⚠️  {result['status']}")

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}\n")

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}\n")

    if successful:
        total_initial = sum(r['initial_holds'] for r in successful)
        total_filtered = sum(r['filtered_holds'] for r in successful)
        total_removed = sum(r['removed'] for r in successful)

        print(f"Total holds detected: {total_initial}")
        print(f"Total holds after filtering: {total_filtered}")
        print(f"Total false positives removed: {total_removed}")
        print(f"False positive rate: {total_removed/total_initial*100:.1f}%")

        avg_calibration = sum(r['calibration_confidence'] for r in successful) / len(successful)
        print(f"Average calibration confidence: {avg_calibration:.2f}")


if __name__ == '__main__':
    main()
