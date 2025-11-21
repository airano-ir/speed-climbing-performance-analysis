#!/usr/bin/env python3
"""
Hold Detection Debug Visualizer
================================
Visualize red hold detection process including HSV mask and detected blobs.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.vision.holds import HoldDetector


def visualize_hold_detection(video_path: str, output_path: str = None, max_frames: int = 5):
    """
    Debug visualization of hold detection.

    Shows:
    - Original frame
    - HSV mask (what's detected as red)
    - Detected holds with bounding boxes and confidence scores
    """
    print(f"\n{'='*60}")
    print(f"Hold Detection Debug Visualizer")
    print(f"{'='*60}\n")
    print(f"Video: {video_path}")
    print(f"Processing first {max_frames} frames...\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return

    detector = HoldDetector()

    print("HSV Thresholds:")
    print(f"  Red Range 1: {detector.hsv_lower_red1} - {detector.hsv_upper_red1}")
    print(f"  Red Range 2: {detector.hsv_lower_red2} - {detector.hsv_upper_red2}")
    print(f"  Min Area: {detector.min_area} px")
    print(f"  Max Area: {detector.max_area} px\n")

    frame_id = 0
    visualizations = []

    try:
        while frame_id < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect holds with mask
            holds, mask = detector.detect_holds(frame, return_mask=True)

            print(f"Frame {frame_id}: Detected {len(holds)} holds")

            # Create visualization
            vis = create_debug_visualization(frame, mask, holds, frame_id)
            visualizations.append(vis)

            # Print hold details
            for i, hold in enumerate(holds[:5]):  # Top 5 holds
                print(f"  Hold {i+1}: pos=({hold.pixel_x:.1f}, {hold.pixel_y:.1f}), "
                      f"area={hold.contour_area:.0f}px, conf={hold.confidence:.3f}")

            print()
            frame_id += 1

    finally:
        cap.release()

    # Save or display
    if output_path:
        save_visualization_grid(visualizations, output_path)
        print(f"Saved visualization to: {output_path}")
    else:
        print("Displaying visualizations... (close windows to continue)")
        for i, vis in enumerate(visualizations):
            cv2.imshow(f'Frame {i}', vis)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


def create_debug_visualization(frame, mask, holds, frame_id):
    """Create a 3-panel debug visualization."""
    h, w = frame.shape[:2]

    # Panel 1: Original frame
    panel1 = frame.copy()
    cv2.putText(panel1, f"Frame {frame_id}: Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 2: HSV mask (red regions)
    panel2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel2, f"Red Mask ({len(holds)} blobs)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 3: Detected holds with annotations
    panel3 = frame.copy()

    # Convert HSV to BGR and get red regions from original frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Find contours from mask for visualization
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours (even rejected ones) in gray
    cv2.drawContours(panel3, contours, -1, (128, 128, 128), 1)

    # Draw accepted holds
    for i, hold in enumerate(holds):
        x, y = int(hold.pixel_x), int(hold.pixel_y)

        # Color based on confidence
        color_value = int(255 * hold.confidence)
        color = (0, color_value, 255 - color_value)  # Blue to Green

        # Draw centroid
        cv2.circle(panel3, (x, y), 8, color, -1)
        cv2.circle(panel3, (x, y), 12, color, 2)

        # Draw label with details
        label = f"#{i+1}: {hold.confidence:.2f}"
        cv2.putText(panel3, label, (x + 15, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw area info
        area_label = f"{int(hold.contour_area)}px"
        cv2.putText(panel3, area_label, (x + 15, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.putText(panel3, f"Detected Holds: {len(holds)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Combine panels side by side
    # Resize to fit
    target_width = 400
    scale = target_width / w

    panel1 = cv2.resize(panel1, (target_width, int(h * scale)))
    panel2 = cv2.resize(panel2, (target_width, int(h * scale)))
    panel3 = cv2.resize(panel3, (target_width, int(h * scale)))

    combined = np.hstack([panel1, panel2, panel3])

    return combined


def save_visualization_grid(visualizations, output_path):
    """Save all visualizations as a vertical grid."""
    if not visualizations:
        return

    grid = np.vstack(visualizations)

    cv2.imwrite(output_path, grid)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Debug Hold Detection')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--output', '-o', help='Output image path (if not set, displays interactively)')
    parser.add_argument('--frames', '-f', type=int, default=5, help='Number of frames to process')

    args = parser.parse_args()

    visualize_hold_detection(args.video, args.output, args.frames)


if __name__ == '__main__':
    main()
