#!/usr/bin/env python3
"""
Quick Parameter Tuning for Hold Detection
==========================================
Test different HSV ranges and thresholds quickly.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.vision.holds import HoldDetector


def test_params(video_path: str, frame_num: int = 0):
    """Test different hold detection parameters."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return

    # Read specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame")
        return

    print(f"\n{'='*70}")
    print(f"Hold Detection Parameter Test")
    print(f"{'='*70}\n")

    # Test different configurations
    configs = [
        {
            'name': 'DEFAULT (Current)',
            'min_area': 500,
            'max_area': 30000,
            'min_confidence': 0.4,
            'hsv_params': None  # Use defaults
        },
        {
            'name': 'VERY STRICT (Fewer false positives)',
            'min_area': 1000,
            'max_area': 20000,
            'min_confidence': 0.5,
            'hsv_params': ([0, 120, 120], [10, 255, 255], [170, 120, 120], [180, 255, 255])
        },
        {
            'name': 'RELAXED (More detections)',
            'min_area': 300,
            'max_area': 40000,
            'min_confidence': 0.3,
            'hsv_params': ([0, 80, 80], [15, 255, 255], [165, 80, 80], [180, 255, 255])
        },
    ]

    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 70)

        # Create detector
        detector = HoldDetector(
            min_area=config['min_area'],
            max_area=config['max_area'],
            min_confidence=config['min_confidence']
        )

        # Override HSV if specified
        if config['hsv_params']:
            detector.hsv_lower_red1 = np.array(config['hsv_params'][0])
            detector.hsv_upper_red1 = np.array(config['hsv_params'][1])
            detector.hsv_lower_red2 = np.array(config['hsv_params'][2])
            detector.hsv_upper_red2 = np.array(config['hsv_params'][3])

        # Detect holds
        holds, mask = detector.detect_holds(frame, return_mask=True)

        print(f"  Parameters:")
        print(f"    Min Area: {config['min_area']}px")
        print(f"    Max Area: {config['max_area']}px")
        print(f"    Min Confidence: {config['min_confidence']}")
        print(f"    HSV Range 1: {detector.hsv_lower_red1} - {detector.hsv_upper_red1}")
        print(f"    HSV Range 2: {detector.hsv_lower_red2} - {detector.hsv_upper_red2}")
        print(f"\n  Results:")
        print(f"    Total Holds Detected: {len(holds)}")

        if len(holds) > 0:
            areas = [h.contour_area for h in holds]
            confs = [h.confidence for h in holds]
            print(f"    Area Range: {int(min(areas))}-{int(max(areas))}px")
            print(f"    Confidence Range: {min(confs):.3f}-{max(confs):.3f}")
            print(f"\n  Top 5 Holds:")
            for i, hold in enumerate(holds[:5], 1):
                print(f"    {i}. pos=({hold.pixel_x:6.1f}, {hold.pixel_y:6.1f}), "
                      f"area={hold.contour_area:5.0f}px, conf={hold.confidence:.3f}")

    print(f"\n{'='*70}")
    print("Recommendation:")
    print("  - If too many false positives (>20 holds): use VERY STRICT")
    print("  - If missing real holds (<8 holds): use RELAXED")
    print("  - Expected: 10-15 holds for full wall view")
    print(f"{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test Hold Detection Parameters')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to test (default: 0)')

    args = parser.parse_args()

    test_params(args.video, args.frame)


if __name__ == '__main__':
    main()
