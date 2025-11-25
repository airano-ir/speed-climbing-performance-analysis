#!/usr/bin/env python3
"""
Debug script for improved hold detection with star shape analysis.

This script provides comprehensive visualization of the improved hold detection
including:
- Color mask visualization
- Star shape detection scores
- Calibration results
- Lighting condition detection
"""

import sys
from pathlib import Path
import argparse
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from speed_climbing.vision.holds import HoldDetector, LightingCondition
from speed_climbing.vision.calibration import CameraCalibrator


def create_debug_visualization(
    frame: np.ndarray,
    holds: list,
    mask: np.ndarray,
    lighting_condition: str,
    debug_info: dict
) -> np.ndarray:
    """Create a comprehensive debug visualization."""
    h, w = frame.shape[:2]

    # Create panels
    # Panel 1: Original with detections
    panel1 = frame.copy()
    for i, hold in enumerate(holds[:20]):
        x, y = int(hold.pixel_x), int(hold.pixel_y)

        # Color based on star score
        if hold.star_score > 0.6:
            color = (0, 255, 0)  # Green - high star score
        elif hold.star_score > 0.3:
            color = (0, 255, 255)  # Yellow - medium
        else:
            color = (0, 165, 255)  # Orange - low

        # Draw contour if available
        if hold.contour is not None:
            cv2.drawContours(panel1, [hold.contour], 0, color, 2)

        cv2.circle(panel1, (x, y), 5, color, -1)

        # Label
        label = f"#{i+1} s:{hold.star_score:.2f}"
        cv2.putText(panel1, label, (x + 10, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Add info text
    cv2.putText(panel1, f"Detected: {len(holds)} holds | Lighting: {lighting_condition}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(panel1, f"Candidates: {debug_info.get('candidates', 0)} | "
                       f"Filtered (area): {debug_info.get('filtered_area', 0)} | "
                       f"Filtered (conf): {debug_info.get('filtered_confidence', 0)}",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Panel 2: Color mask
    panel2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel2, "Red Color Mask", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 3: HSV channels
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Create hue visualization with color map
    h_vis = cv2.applyColorMap(h_channel, cv2.COLORMAP_HSV)
    cv2.putText(h_vis, "Hue Channel", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 4: Score distribution
    panel4 = np.zeros((h, w, 3), dtype=np.uint8)
    panel4[:] = (40, 40, 40)

    # Draw hold info as text
    y_offset = 40
    cv2.putText(panel4, "Hold Analysis:", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 30

    for i, hold in enumerate(holds[:12]):
        text = f"{i+1:2d}. star={hold.star_score:.2f} color={hold.color_score:.2f} conf={hold.confidence:.2f} area={hold.contour_area:.0f}"
        color = (0, 255, 0) if hold.star_score > 0.5 else (0, 200, 200)
        cv2.putText(panel4, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_offset += 25

    # Resize panels for combined view
    target_w = 640
    scale = target_w / w

    panel1 = cv2.resize(panel1, (target_w, int(h * scale)))
    panel2 = cv2.resize(panel2, (target_w, int(h * scale)))
    h_vis = cv2.resize(h_vis, (target_w, int(h * scale)))
    panel4 = cv2.resize(panel4, (target_w, int(h * scale)))

    # Combine panels
    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([h_vis, panel4])
    combined = np.vstack([top_row, bottom_row])

    return combined


def process_video_frame(video_path: str, frame_num: int = 0, output_path: str = None):
    """Process a single frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_num}")
        return

    process_frame(frame, output_path, f"Frame {frame_num} from {Path(video_path).name}")


def process_image(image_path: str, output_path: str = None):
    """Process an image file."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return

    process_frame(frame, output_path, f"Image: {Path(image_path).name}")


def process_frame(frame: np.ndarray, output_path: str = None, title: str = "Debug"):
    """Process a frame and create debug visualization."""
    print(f"\n{'='*60}")
    print(f"Processing: {title}")
    print(f"{'='*60}")

    # Initialize detector
    route_path = project_root / "configs" / "ifsc_route_coordinates.json"
    detector = HoldDetector(
        route_coordinates_path=str(route_path) if route_path.exists() else None,
        min_area=500,
        min_confidence=0.4,
        use_star_detection=True,
        lighting_condition=LightingCondition.AUTO
    )

    # Detect lighting condition
    lighting = detector._detect_lighting_condition(frame)
    print(f"Detected lighting: {lighting.value}")

    # Detect holds
    holds, mask = detector.detect_holds(frame, return_mask=True)
    _, debug_info = detector.detect_holds(frame, return_debug_info=True)

    print(f"\nDetection Results:")
    print(f"  Total candidates: {debug_info.get('candidates', 0)}")
    print(f"  Filtered by area: {debug_info.get('filtered_area', 0)}")
    print(f"  Filtered by confidence: {debug_info.get('filtered_confidence', 0)}")
    print(f"  Final detections: {len(holds)}")

    # Print top holds
    print(f"\nTop 10 Detected Holds:")
    for i, hold in enumerate(holds[:10]):
        print(f"  {i+1:2d}. pos=({hold.pixel_x:6.0f}, {hold.pixel_y:6.0f}), "
              f"star={hold.star_score:.2f}, color={hold.color_score:.2f}, "
              f"conf={hold.confidence:.2f}, area={hold.contour_area:.0f}")

    # Try calibration if we have enough holds
    if len(holds) >= 4:
        print(f"\nAttempting calibration...")
        calibrator = CameraCalibrator(
            str(route_path) if route_path.exists() else None
        )

        for lane in ['left', 'right']:
            result = calibrator.calibrate(frame, holds, lane=lane)
            if result:
                print(f"  {lane.upper()} lane calibration:")
                print(f"    Inliers: {result.inlier_count}/{len(holds)}")
                print(f"    RMSE: {result.rmse_error:.4f}m")
                print(f"    Confidence: {result.confidence:.2f}")
                if result.matched_hold_nums:
                    print(f"    Matched holds: {result.matched_hold_nums}")
            else:
                print(f"  {lane.upper()} lane: calibration failed")

    # Create visualization
    vis = create_debug_visualization(
        frame, holds, mask,
        lighting.value, debug_info
    )

    # Save or display
    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"\nSaved visualization to: {output_path}")
    else:
        cv2.imshow('Debug Visualization', vis)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Debug improved hold detection with star shape analysis'
    )
    parser.add_argument('input', help='Input video or image path')
    parser.add_argument('--frame', '-f', type=int, default=0,
                       help='Frame number for video (default: 0)')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--lighting', '-l',
                       choices=['auto', 'daylight', 'indoor', 'spotlight', 'low_light'],
                       default='auto', help='Lighting preset')

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Check if video or image
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    ext = input_path.suffix.lower()

    if ext in video_extensions:
        process_video_frame(str(input_path), args.frame, args.output)
    elif ext in image_extensions:
        process_image(str(input_path), args.output)
    else:
        print(f"Error: Unsupported file type: {ext}")
        sys.exit(1)


if __name__ == '__main__':
    main()
