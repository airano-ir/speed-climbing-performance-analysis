#!/usr/bin/env python3
"""
Test Spatial Grid Filtering
============================
Demonstrates how spatial filtering using route map removes false positives.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.vision.holds import HoldDetector
from speed_climbing.vision.calibration import CameraCalibrator


def test_spatial_filtering(video_path: str, route_map_path: str, output_path: str = None):
    """
    Test spatial filtering on a video frame.

    Shows:
    1. Original detections (with false positives)
    2. After spatial filtering (only valid holds)
    3. Expected hold positions from route map
    """
    print(f"\n{'='*70}")
    print(f"Spatial Grid Filtering Test")
    print(f"{'='*70}\n")

    # Load route map
    with open(route_map_path, 'r', encoding='utf-8') as f:
        route_map = json.load(f)

    # Read first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame")
        return

    # Initialize components
    hold_detector = HoldDetector(
        route_coordinates_path=route_map_path,
        use_spatial_filtering=False  # We'll do it manually to compare
    )

    calibrator = CameraCalibrator(route_coordinates_path=route_map_path)

    print("Step 1: Initial hold detection (without spatial filtering)...")
    initial_holds = hold_detector.detect_holds(frame)
    print(f"  Detected: {len(initial_holds)} holds")

    print("\nStep 2: Camera calibration...")
    calibration_result = calibrator.calibrate(frame, initial_holds)

    if not calibration_result or calibration_result.homography_matrix is None:
        print("  ERROR: Calibration failed")
        print("  Trying with relaxed parameters...")

        # Try again with more lenient detector
        hold_detector_relaxed = HoldDetector(
            route_coordinates_path=route_map_path,
            min_area=300,
            min_confidence=0.3,
            use_spatial_filtering=False
        )
        initial_holds = hold_detector_relaxed.detect_holds(frame)
        print(f"  Re-detected: {len(initial_holds)} holds")

        calibration_result = calibrator.calibrate(frame, initial_holds)

        if not calibration_result:
            print("  ERROR: Calibration still failed, cannot test spatial filtering")
            return

    homography = calibration_result.homography_matrix
    print(f"  ✓ Calibration successful (confidence: {calibration_result.confidence:.2f})")
    print(f"  Inliers: {calibration_result.inlier_count}/{calibration_result.total_holds}")

    print("\nStep 3: Spatial filtering...")
    filtered_holds = hold_detector.filter_by_spatial_grid(
        initial_holds,
        homography,
        frame.shape[:2]
    )
    print(f"  After filtering: {len(filtered_holds)} holds")
    print(f"  Removed: {len(initial_holds) - len(filtered_holds)} false positives")

    # Create visualization
    vis = create_comparison_viz(
        frame, initial_holds, filtered_holds,
        route_map, homography
    )

    # Save or display
    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"\nSaved to: {output_path}")
    else:
        cv2.imshow('Spatial Filtering Comparison', vis)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print detailed comparison
    print(f"\n{'='*70}")
    print("Detailed Comparison")
    print(f"{'='*70}")
    print(f"\nBefore Spatial Filtering ({len(initial_holds)} detections):")
    for i, hold in enumerate(initial_holds[:10], 1):
        print(f"  {i:2d}. pos=({hold.pixel_x:6.1f}, {hold.pixel_y:6.1f}), "
              f"area={hold.contour_area:5.0f}px, conf={hold.confidence:.3f}")

    print(f"\nAfter Spatial Filtering ({len(filtered_holds)} valid holds):")
    for i, hold in enumerate(filtered_holds, 1):
        print(f"  {i:2d}. Hold #{hold.hold_num} at {hold.grid_position} "
              f"(panel {hold.panel}), conf={hold.confidence:.3f}")


def create_comparison_viz(
    frame, initial_holds, filtered_holds,
    route_map, homography
):
    """Create side-by-side comparison visualization."""
    h, w = frame.shape[:2]

    # Panel 1: Original detections
    panel1 = frame.copy()
    for i, hold in enumerate(initial_holds[:20], 1):
        x, y = int(hold.pixel_x), int(hold.pixel_y)
        cv2.circle(panel1, (x, y), 8, (0, 0, 255), 2)  # Red
        cv2.putText(panel1, str(i), (x+10, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(panel1, f"Before: {len(initial_holds)} detections", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Panel 2: Filtered detections with route map overlay
    panel2 = frame.copy()

    # Draw expected positions from route map
    route_holds = route_map.get('holds', [])
    for route_hold in route_holds:
        world_pos = np.array([[route_hold['wall_x_m'], route_hold['wall_y_m']]], dtype=np.float32)
        try:
            # Transform world to pixel
            pixel_pos = cv2.perspectiveTransform(
                world_pos.reshape(-1, 1, 2),
                np.linalg.inv(homography)
            )
            px, py = int(pixel_pos[0][0][0]), int(pixel_pos[0][0][1])

            # Draw expected position as small circle
            cv2.circle(panel2, (px, py), 4, (128, 128, 128), 1)  # Gray
        except:
            pass

    # Draw filtered detections
    for i, hold in enumerate(filtered_holds, 1):
        x, y = int(hold.pixel_x), int(hold.pixel_y)
        cv2.circle(panel2, (x, y), 10, (0, 255, 0), 2)  # Green
        label = f"#{hold.hold_num}" if hold.hold_num else str(i)
        cv2.putText(panel2, label, (x+10, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(panel2, f"After: {len(filtered_holds)} valid holds", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(panel2, "(Gray dots = expected positions)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # Resize and combine
    target_w = 700
    scale = target_w / w

    panel1 = cv2.resize(panel1, (target_w, int(h * scale)))
    panel2 = cv2.resize(panel2, (target_w, int(h * scale)))

    combined = np.hstack([panel1, panel2])

    return combined


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test Spatial Grid Filtering')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--route-map', default='configs/ifsc_route_coordinates.json',
                       help='Path to route map JSON')
    parser.add_argument('--output', '-o', help='Output image path')

    args = parser.parse_args()

    test_spatial_filtering(args.video, args.route_map, args.output)


if __name__ == '__main__':
    main()
