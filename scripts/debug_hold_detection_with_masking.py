#!/usr/bin/env python3
"""
Advanced Hold Detection Debugger with Climber Masking
======================================================
Shows the effect of climber masking on hold detection.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.vision.holds import HoldDetector
from speed_climbing.vision.pose import BlazePoseExtractor


def debug_with_masking(video_path: str, output_path: str = None):
    """
    Debug hold detection with and without climber masking.

    Shows 4 panels:
    1. Original frame
    2. Holds WITHOUT climber masking
    3. Frame WITH climber masked
    4. Holds WITH climber masking
    """
    print(f"\n{'='*70}")
    print(f"Hold Detection Debug: WITH vs WITHOUT Climber Masking")
    print(f"{'='*70}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return

    # Use first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame")
        return

    # Initialize
    hold_detector = HoldDetector()
    pose_extractor = BlazePoseExtractor()

    print("Processing...")

    # 1. Detect holds WITHOUT masking
    holds_no_mask, mask_no_mask = hold_detector.detect_holds(frame, return_mask=True)
    print(f"Holds detected WITHOUT masking: {len(holds_no_mask)}")

    # 2. Extract pose
    pose_result = pose_extractor.process_frame(frame)

    if not pose_result.has_detection:
        print("WARNING: No pose detected, masking will have no effect")
        masked_frame = frame
    else:
        print(f"Pose detected (confidence: {pose_result.overall_confidence:.2f})")
        masked_frame = create_climber_mask(frame, pose_result)

    # 3. Detect holds WITH masking
    holds_with_mask, mask_with_mask = hold_detector.detect_holds(masked_frame, return_mask=True)
    print(f"Holds detected WITH masking: {len(holds_with_mask)}")

    reduction = len(holds_no_mask) - len(holds_with_mask)
    print(f"\nReduction: {reduction} false positives removed")

    # Create visualization
    vis = create_comparison_visualization(
        frame, masked_frame,
        holds_no_mask, holds_with_mask,
        mask_no_mask, mask_with_mask,
        pose_result
    )

    # Save or display
    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"\nSaved to: {output_path}")
    else:
        cv2.imshow('Hold Detection: With vs Without Masking', vis)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print hold details
    print(f"\n{'='*70}")
    print("Top 10 Holds WITHOUT Masking:")
    print(f"{'='*70}")
    for i, hold in enumerate(holds_no_mask[:10], 1):
        print(f"  {i:2d}. pos=({hold.pixel_x:6.1f}, {hold.pixel_y:6.1f}), "
              f"area={hold.contour_area:6.0f}px, conf={hold.confidence:.3f}")

    print(f"\n{'='*70}")
    print("Top 10 Holds WITH Masking:")
    print(f"{'='*70}")
    for i, hold in enumerate(holds_with_mask[:10], 1):
        print(f"  {i:2d}. pos=({hold.pixel_x:6.1f}, {hold.pixel_y:6.1f}), "
              f"area={hold.contour_area:6.0f}px, conf={hold.confidence:.3f}")


def create_climber_mask(frame: np.ndarray, pose_result) -> np.ndarray:
    """Mask out the climber from the frame."""
    if not pose_result or not pose_result.has_detection:
        return frame

    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

    # Collect keypoints
    points = []
    h, w = frame.shape[:2]
    for kp in pose_result.keypoints.values():
        if kp.confidence > 0.3:
            points.append((int(kp.x * w), int(kp.y * h)))

    if points:
        # Create convex hull around climber
        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(mask, hull, 0)

        # Dilate to ensure full coverage
        kernel = np.ones((30, 30), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    # Apply mask
    masked_frame = frame.copy()
    masked_frame[mask == 0] = [0, 0, 0]

    return masked_frame


def create_comparison_visualization(
    original, masked_frame,
    holds_no_mask, holds_with_mask,
    red_mask_no_mask, red_mask_with_mask,
    pose_result
):
    """Create 4-panel comparison visualization."""
    h, w = original.shape[:2]

    # Panel 1: Original with holds (no masking)
    panel1 = original.copy()
    for i, hold in enumerate(holds_no_mask[:15], 1):
        x, y = int(hold.pixel_x), int(hold.pixel_y)
        color = (0, 0, 255)  # Red for potentially false positives
        cv2.circle(panel1, (x, y), 8, color, 2)
        cv2.putText(panel1, str(i), (x+10, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(panel1, f"WITHOUT Masking: {len(holds_no_mask)} holds", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Panel 2: Masked frame with pose overlay
    panel2 = masked_frame.copy()
    if pose_result and pose_result.has_detection:
        # Draw skeleton
        for kp in pose_result.keypoints.values():
            if kp.confidence > 0.3:
                x, y = int(kp.x * w), int(kp.y * h)
                cv2.circle(panel2, (x, y), 3, (0, 255, 0), -1)
    cv2.putText(panel2, "Climber MASKED", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Panel 3: HSV mask comparison
    # Combine both masks side by side
    red_mask_no = cv2.cvtColor(red_mask_no_mask, cv2.COLOR_GRAY2BGR)
    red_mask_with = cv2.cvtColor(red_mask_with_mask, cv2.COLOR_GRAY2BGR)

    # Add labels
    cv2.putText(red_mask_no, "Red Mask: No Climber Masking", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(red_mask_with, "Red Mask: With Climber Masking", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Side by side
    panel3 = np.hstack([red_mask_no, red_mask_with])

    # Panel 4: Masked frame with final holds
    panel4 = masked_frame.copy()
    for i, hold in enumerate(holds_with_mask[:15], 1):
        x, y = int(hold.pixel_x), int(hold.pixel_y)
        color = (0, 255, 0)  # Green for actual holds
        cv2.circle(panel4, (x, y), 8, color, 2)
        cv2.putText(panel4, str(i), (x+10, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(panel4, f"WITH Masking: {len(holds_with_mask)} holds", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Resize panels
    target_w = 600
    scale = target_w / w

    panel1 = cv2.resize(panel1, (target_w, int(h * scale)))
    panel2 = cv2.resize(panel2, (target_w, int(h * scale)))
    panel4 = cv2.resize(panel4, (target_w, int(h * scale)))

    # Panel 3 is already double width
    panel3 = cv2.resize(panel3, (target_w * 2, int(h * scale)))

    # Arrange: top row = panel1 + panel2, bottom row = panel3, third row = panel4
    top_row = np.hstack([panel1, panel2])
    bottom_half = np.hstack([panel4, np.zeros_like(panel4)])  # Pad to match width

    # Stack vertically
    combined = np.vstack([top_row, panel3, bottom_half])

    return combined


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Debug Hold Detection with Masking')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--output', '-o', help='Output image path')

    args = parser.parse_args()

    debug_with_masking(args.video, args.output)


if __name__ == '__main__':
    main()
