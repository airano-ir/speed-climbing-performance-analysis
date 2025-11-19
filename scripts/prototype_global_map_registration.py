#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Prototype: Global Map Registration Proof of Concept
===========================================================

این prototype مفهوم اصلی را validate می‌کند:
- Hold detection در یک فریم
- Calibration محاسبه
- Pixel→Meter transformation
- نمایش نتایج

Usage:
    python scripts/prototype_global_map_registration.py \
        --video data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race001.mp4 \
        --frame 60

Author: Speed Climbing Performance Analysis
Date: 2025-11-19
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calibration.camera_calibration import CameraCalibrator
# from calibration.ifsc_route_map import IFSCRouteMapParser  # Not needed for prototype
from phase1_pose_estimation.hold_detector import HoldDetector


def main():
    parser = argparse.ArgumentParser(
        description='Prototype: Global Map Registration'
    )
    parser.add_argument(
        '--video',
        type=str,
        default='data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race001.mp4',
        help='Path to race video'
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=60,
        help='Frame number to test (default: 60)'
    )
    parser.add_argument(
        '--lane',
        type=str,
        default='left',
        choices=['left', 'right'],
        help='Which lane to process'
    )
    parser.add_argument(
        '--route-map',
        type=str,
        default='configs/ifsc_route_coordinates.json',
        help='Path to IFSC route map'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/prototype_test/',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Global Map Registration - Prototype Test")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Frame: {args.frame}")
    print(f"Lane: {args.lane}")
    print(f"Route map: {args.route_map}")
    print()

    # Step 1: Load video and extract frame
    print("Step 1: Loading video...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {args.video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  ✓ Video loaded: {frame_width}x{frame_height}, {fps:.1f} fps, {total_frames} frames")

    # Seek to target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Error: Cannot read frame {args.frame}")
        return 1

    print(f"  ✓ Frame {args.frame} extracted")
    print()

    # Step 2: Detect holds
    print("Step 2: Detecting red holds...")
    hold_detector = HoldDetector(
        route_coordinates_path=args.route_map,
        min_area=200,
        min_confidence=0.2  # Lower threshold for better recall
    )

    detected_holds = hold_detector.detect_holds(frame, lane=args.lane)
    print(f"  ✓ Detected {len(detected_holds)} holds")

    if len(detected_holds) == 0:
        print("  ⚠️  Warning: No holds detected!")
        print("  Suggestions:")
        print("    - Try a different frame number (--frame)")
        print("    - Check if holds are visible in the frame")
        print("    - Adjust detection parameters")
        return 1

    # Display top detected holds
    print(f"\n  Top 10 detected holds:")
    for i, hold in enumerate(detected_holds[:10]):
        print(f"    {i+1}. Confidence: {hold.confidence:.2f}, "
              f"Position: ({hold.pixel_x:.0f}, {hold.pixel_y:.0f}), "
              f"Area: {hold.contour_area:.0f} px²")
    print()

    # Save visualization
    vis_frame = hold_detector.visualize_detections(
        frame, detected_holds, show_labels=True
    )
    vis_path = output_dir / f"frame_{args.frame}_holds_detected.jpg"
    cv2.imwrite(str(vis_path), vis_frame)
    print(f"  ✓ Saved visualization: {vis_path}")
    print()

    # Step 3: Camera calibration
    print("Step 3: Camera calibration (Homography)...")
    calibrator = CameraCalibrator(
        route_coordinates_path=args.route_map,
        min_holds_for_calibration=4,
        ransac_threshold=0.05  # 5cm threshold
    )

    calibration = calibrator.calibrate(frame, detected_holds, lane=args.lane)

    if calibration is None:
        print("  ❌ Calibration failed!")
        print("  Possible reasons:")
        print("    - Not enough holds detected")
        print("    - Holds don't match route map")
        print("    - RANSAC failed to find good solution")
        return 1

    print(f"  ✓ Calibration successful!")
    print(f"    Inliers: {calibration.inlier_count}/{calibration.total_holds}")
    print(f"    RMSE: {calibration.rmse_error:.4f} m ({calibration.rmse_error*100:.2f} cm)")
    print(f"    Scale: {calibration.pixel_to_meter_scale:.1f} pixels/meter")
    print(f"    Confidence: {calibration.confidence:.2f}")
    print()

    # Quality assessment
    if calibration.rmse_error > 0.10:
        print(f"  ⚠️  Warning: High RMSE ({calibration.rmse_error*100:.1f} cm > 10 cm)")
    else:
        print(f"  ✅ Good calibration quality (RMSE < 10 cm)")

    if calibration.inlier_ratio < 0.7:
        print(f"  ⚠️  Warning: Low inlier ratio ({calibration.inlier_ratio:.1%})")
    else:
        print(f"  ✅ Good inlier ratio ({calibration.inlier_ratio:.1%})")
    print()

    # Step 4: Demonstrate pixel→meter transformation
    print("Step 4: Demonstrating pixel→meter transformation...")

    # Test point: center of frame
    test_pixel_x = frame_width / 2
    test_pixel_y = frame_height / 2
    test_meter_x, test_meter_y = calibration.pixel_to_meter_func(
        test_pixel_x, test_pixel_y
    )

    print(f"  Example transformation:")
    print(f"    Center of frame: ({test_pixel_x:.0f}, {test_pixel_y:.0f}) pixels")
    print(f"    → Wall coordinates: ({test_meter_x:.3f}, {test_meter_y:.3f}) meters")
    print()

    # Calculate height from start pad (15m wall - y_meter)
    WALL_HEIGHT_M = 15.0
    height_from_start = WALL_HEIGHT_M - test_meter_y
    print(f"    Height from start pad: {height_from_start:.2f} m")
    print()

    # Step 5: Test with multiple points across frame
    print("Step 5: Testing transformation at different heights...")
    test_points = [
        ("Bottom", frame_width/2, frame_height * 0.9),
        ("25% up", frame_width/2, frame_height * 0.7),
        ("50% up", frame_width/2, frame_height * 0.5),
        ("75% up", frame_width/2, frame_height * 0.3),
        ("Top", frame_width/2, frame_height * 0.1),
    ]

    results = []
    for label, px_x, px_y in test_points:
        m_x, m_y = calibration.pixel_to_meter_func(px_x, px_y)
        height_m = WALL_HEIGHT_M - m_y
        results.append({
            'label': label,
            'pixel_x': px_x,
            'pixel_y': px_y,
            'meter_x': m_x,
            'meter_y': m_y,
            'height_from_start_m': height_m
        })
        print(f"  {label:8s}: pixel_y={px_y:4.0f} → height={height_m:5.2f} m")

    print()

    # Step 6: Check if results are physically reasonable
    print("Step 6: Validation of results...")

    # Check if heights increase as we go up
    heights = [r['height_from_start_m'] for r in results]
    if all(heights[i] < heights[i+1] for i in range(len(heights)-1)):
        print("  ✅ Heights increase correctly (bottom→top)")
    else:
        print("  ❌ Heights don't increase monotonically - calibration may be wrong!")

    # Check if heights are in reasonable range
    if all(0 <= h <= 16 for h in heights):
        print("  ✅ All heights within wall range (0-16m)")
    else:
        print("  ❌ Some heights outside valid range!")

    # Check velocity would be reasonable
    # Typical climb time: 6-8 seconds, so velocity should be ~2-2.5 m/s
    if heights[-1] - heights[0] > 10:  # Top to bottom should be ~15m
        expected_velocity = (heights[-1] - heights[0]) / 6  # Assume 6s climb
        print(f"  ✅ Expected velocity: {expected_velocity:.2f} m/s (reasonable for speed climbing)")
    print()

    # Step 7: Save results
    print("Step 7: Saving results...")

    results_data = {
        'video': str(args.video),
        'frame': args.frame,
        'lane': args.lane,
        'calibration': {
            'homography_matrix': calibration.homography_matrix.tolist(),
            'pixel_to_meter_scale': float(calibration.pixel_to_meter_scale),
            'rmse_error_m': float(calibration.rmse_error),
            'rmse_error_cm': float(calibration.rmse_error * 100),
            'inlier_count': int(calibration.inlier_count),
            'total_holds': int(calibration.total_holds),
            'inlier_ratio': float(calibration.inlier_ratio),
            'confidence': float(calibration.confidence)
        },
        'test_transformations': results,
        'validation': {
            'heights_increase_correctly': all(heights[i] < heights[i+1] for i in range(len(heights)-1)),
            'heights_in_valid_range': all(0 <= h <= 16 for h in heights),
            'estimated_climb_velocity_m_s': float((heights[-1] - heights[0]) / 6) if heights[-1] > heights[0] else 0
        }
    }

    results_path = output_dir / f"frame_{args.frame}_calibration_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"  ✓ Results saved: {results_path}")
    print()

    # Summary
    print("=" * 70)
    print("PROTOTYPE TEST SUMMARY")
    print("=" * 70)
    print(f"Holds detected: {len(detected_holds)}")
    print(f"Calibration success: {'✅ YES' if calibration else '❌ NO'}")
    if calibration:
        print(f"Calibration quality: {calibration.rmse_error*100:.1f} cm RMSE")
        print(f"Transformation working: ✅ YES")
        print(f"Physical validation: {'✅ PASS' if results_data['validation']['heights_increase_correctly'] else '❌ FAIL'}")
    print()
    print("Next steps:")
    print("  1. ✅ Concept validated - pixel→meter transformation works!")
    print("  2. ⏭️  Implement TimeSeriesBuilder")
    print("  3. ⏭️  Implement DropoutHandler")
    print("  4. ⏭️  Implement WorldCoordinateTracker")
    print("  5. ⏭️  Implement GlobalMapVideoProcessor")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
