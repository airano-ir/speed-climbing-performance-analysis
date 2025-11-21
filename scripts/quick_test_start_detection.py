#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Detection Test
===========================
Fast test of start detection logic on first N frames.
"""

import sys
from pathlib import Path
import json
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.vision.pose import BlazePoseExtractor
from speed_climbing.analysis.start_finish_detector import StartDetector

def quick_test(video_path: str, max_frames: int = 100):
    """Quick test of start detection."""
    print(f"\n{'='*60}")
    print(f"Quick Start Detection Test")
    print(f"Video: {video_path}")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps:.2f}")

    pose_extractor = BlazePoseExtractor()
    detector = StartDetector()

    frame_id = 0
    com_heights = []

    print("\nProcessing frames...")

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

            # Get COM (normalized coordinates, we'll fake the height in meters for now)
            com_kp = pose_result.keypoints.get('COM')
            if com_kp:
                # Fake world coordinate: assume frame height ~= 15m, normalized y is from top
                # COM at bottom (y=1.0) = 0m, at top (y=0.0) = 15m
                # Approximate: com_height_m = (1.0 - com_kp.y) * 15
                # But for ready position, COM should be ~1m, so let's scale differently
                # Better: assume COM in pixel space, convert to approximate height
                # For simplicity: use normalized y, scaled to reasonable range
                # This is a hack for quick test without calibration

                # Estimate: y=0.9 (near bottom) -> ~1m, y=0.1 (near top) -> ~14m
                # Linear approximation: height = 15 - (y * 13)
                # At y=0.85: height = 15 - 0.85*13 = 15 - 11.05 = 3.95m (too high for ready)
                # Better: height = 1.0 + (1.0 - y) * 14
                # At y=0.9: height = 1.0 + 0.1*14 = 2.4m (still high)
                # Let's just use: height = (1.0 - y) * 15
                # At y=0.9: 1.5m (better!)

                com_height_m = (1.0 - com_kp.y) * 15.0

                com_heights.append({
                    'frame': frame_id,
                    'time': timestamp,
                    'height': com_height_m,
                    'confidence': pose_result.overall_confidence
                })

                # Test start detection
                start_event = detector.process_frame(
                    frame_id, timestamp, com_height_m,
                    pose_result.keypoints, pose_result.overall_confidence
                )

                if start_event:
                    print(f"\n{'='*60}")
                    print(f"START DETECTED!")
                    print(f"{'='*60}")
                    print(f"Frame: {start_event.frame_id}")
                    print(f"Time: {start_event.timestamp:.3f}s")
                    print(f"COM Height: {start_event.com_height_m:.3f}m")
                    print(f"Method: {start_event.method}")
                    print(f"Details: {json.dumps(start_event.details, indent=2)}")
                    print(f"{'='*60}\n")
                    break

                # Print progress every 10 frames
                if frame_id % 10 == 0:
                    print(f"Frame {frame_id}: COM height = {com_height_m:.3f}m, "
                          f"confidence = {pose_result.overall_confidence:.2f}")

            frame_id += 1

    finally:
        cap.release()
        pose_extractor.release()

    print(f"\nProcessed {frame_id} frames")

    if len(com_heights) > 0:
        print(f"\nCOM Height Statistics:")
        heights = [h['height'] for h in com_heights]
        print(f"  Min: {min(heights):.3f}m")
        print(f"  Max: {max(heights):.3f}m")
        print(f"  Mean: {np.mean(heights):.3f}m")
        print(f"  Median: {np.median(heights):.3f}m")

        # Check if baseline was calibrated
        if detector.baseline_com_height:
            print(f"  Baseline (calibrated): {detector.baseline_com_height:.3f}m")

        # Show first 10 heights
        print(f"\nFirst 10 COM heights:")
        for h in com_heights[:10]:
            print(f"  Frame {h['frame']:3d}: {h['height']:.3f}m (conf={h['confidence']:.2f})")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python quick_test_start_detection.py <video_path>")
        sys.exit(1)

    quick_test(sys.argv[1])
