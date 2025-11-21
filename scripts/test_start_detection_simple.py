#!/usr/bin/env python3
"""
Simple Start Detection Test with Real Calibration
==================================================
Uses actual camera calibration for accurate COM height measurement.
"""

import sys
from pathlib import Path
import json
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.vision.pose import BlazePoseExtractor
from speed_climbing.vision.holds import HoldDetector
from speed_climbing.vision.calibration import CameraCalibrator
from speed_climbing.analysis.start_finish_detector import StartDetector

def test_with_calibration(video_path: str, route_map_path: str, max_frames: int = 150):
    """Test start detection with proper calibration."""

    print(f"\n{'='*60}")
    print(f"Start Detection Test (with calibration)")
    print(f"{'='*60}\n")

    # Load route map
    with open(route_map_path, 'r', encoding='utf-8') as f:
        route_map = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps:.2f}\n")

    # Initialize components
    pose_extractor = BlazePoseExtractor()
    hold_detector = HoldDetector()
    calibrator = CameraCalibrator(route_coordinates_path=route_map_path)
    detector = StartDetector()

    # Calibrate on first frame
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Could not read first frame")
        return

    print("Calibrating...")
    holds = hold_detector.detect_holds(first_frame)
    print(f"Detected {len(holds)} holds")

    if len(holds) < 4:
        print("ERROR: Not enough holds for calibration")
        cap.release()
        return

    # Match to route map
    route_holds = route_map.get('holds', [])
    detected_points = np.array([[h['center_x'], h['center_y']] for h in holds], dtype=np.float32)
    n = min(len(holds), len(route_holds))
    reference_points = np.array([
        [h['wall_x_m'], h['wall_y_m']] for h in route_holds[:n]
    ], dtype=np.float32)

    success, homography = calibrator.compute_homography(
        detected_points[:n], reference_points[:n]
    )

    if not success or homography is None:
        print("ERROR: Calibration failed")
        cap.release()
        return

    print("✓ Calibration successful\n")

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process frames
    frame_id = 0
    com_data = []

    print("Processing frames...\n")

    try:
        while frame_id < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_id / fps if fps > 0 else frame_id * 0.033

            # Extract pose
            pose_result = pose_extractor.process_frame(frame, frame_id, timestamp)

            if not pose_result.has_detection:
                frame_id += 1
                continue

            # Transform COM to world coordinates
            com_kp = pose_result.keypoints.get('COM')
            if com_kp:
                h, w = frame.shape[:2]
                pixel_x = int(com_kp.x * w)
                pixel_y = int(com_kp.y * h)

                point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
                world_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), homography)
                com_world_x, com_world_y = world_point[0][0]

                com_data.append({
                    'frame': frame_id,
                    'time': timestamp,
                    'height': com_world_y,
                    'x': com_world_x,
                    'confidence': pose_result.overall_confidence
                })

                # Test start detection
                start_event = detector.process_frame(
                    frame_id, timestamp, com_world_y,
                    pose_result.keypoints, pose_result.overall_confidence
                )

                if start_event:
                    print(f"\n{'='*60}")
                    print(f"✓ START DETECTED!")
                    print(f"{'='*60}")
                    print(f"Frame: {start_event.frame_id}")
                    print(f"Time: {start_event.timestamp:.3f}s")
                    print(f"COM Height: {start_event.com_height_m:.3f}m")
                    print(f"Baseline: {start_event.details.get('baseline_height', 0):.3f}m")
                    print(f"Delta: {start_event.details.get('delta_height', 0):.3f}m")
                    print(f"Method: {start_event.method}")
                    print(f"{'='*60}\n")
                    break

                # Print progress
                if frame_id % 15 == 0:
                    status = "calibrating" if len(detector.calibration_buffer) < 30 else "detecting"
                    print(f"Frame {frame_id:3d} ({status:12s}): COM height = {com_world_y:6.3f}m, "
                          f"conf = {pose_result.overall_confidence:.2f}")

            frame_id += 1

    finally:
        cap.release()
        pose_extractor.release()

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Frames processed: {frame_id}")

    if len(com_data) > 0:
        heights = [h['height'] for h in com_data]
        print(f"\nCOM Height Statistics:")
        print(f"  Min: {min(heights):.3f}m")
        print(f"  Max: {max(heights):.3f}m")
        print(f"  Mean: {np.mean(heights):.3f}m")
        print(f"  Median: {np.median(heights):.3f}m")

        if detector.baseline_com_height:
            print(f"  Baseline (calibrated): {detector.baseline_com_height:.3f}m")
        else:
            print(f"  Baseline: NOT CALIBRATED (heights may be outside ready range)")

        # Show calibration window
        print(f"\nFirst 30 frames (calibration window):")
        for h in com_data[:30]:
            marker = " ← baseline" if h['frame'] == 29 else ""
            print(f"  Frame {h['frame']:3d}: {h['height']:.3f}m{marker}")

        if detector.start_detected:
            print(f"\n✓ Start was detected!")
        else:
            print(f"\n✗ Start was NOT detected")
            print(f"  Possible reasons:")
            print(f"  - COM heights outside ready position range (0.8-1.2m)")
            print(f"  - Video starts after race has begun")
            print(f"  - Calibration error")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_start_detection_simple.py <video_path> [route_map]")
        sys.exit(1)

    video_path = sys.argv[1]
    route_map = sys.argv[2] if len(sys.argv) > 2 else 'configs/ifsc_route_coordinates.json'

    test_with_calibration(video_path, route_map)
