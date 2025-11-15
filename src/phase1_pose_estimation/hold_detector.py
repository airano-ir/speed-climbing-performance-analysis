"""Detect IFSC red holds in video frames using color-based detection.

This module detects the standardized red holds used in IFSC speed climbing
competitions. The detection is designed to work with moving cameras by
processing each frame independently.

Key features:
- HSV color-based detection for red holds
- Handles camera movement (frame-by-frame detection)
- Matches detected holds to expected IFSC route positions
- Provides confidence scores for each detection
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectedHold:
    """Represents a detected hold in a video frame."""

    hold_num: Optional[int]  # Matched to IFSC route map, None if unmatched
    pixel_x: float
    pixel_y: float
    confidence: float  # 0.0 to 1.0
    contour_area: float
    panel: Optional[str] = None  # e.g., 'DX1', 'SN3'
    grid_position: Optional[str] = None  # e.g., 'F4', 'M8'


class HoldDetector:
    """Detect red IFSC holds in video frames using HSV color thresholding.

    The detector is designed to work with moving cameras by processing each
    frame independently. It can optionally match detected holds to the
    expected IFSC route map for identification.
    """

    def __init__(
        self,
        route_coordinates_path: Optional[str] = None,
        min_area: int = 100,
        max_area: int = 50000,
        min_confidence: float = 0.2
    ):
        """Initialize hold detector.

        Args:
            route_coordinates_path: Path to IFSC route coordinates JSON.
                                   If None, holds won't be matched to route.
            min_area: Minimum contour area (pixels) to consider as a hold
            max_area: Maximum contour area (pixels) to consider as a hold
            min_confidence: Minimum confidence threshold for detections (default: 0.2 for better recall)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_confidence = min_confidence
        self.route_map = None

        if route_coordinates_path:
            self.route_map = self._load_route_coordinates(route_coordinates_path)
            logger.info(f"Loaded route map with {len(self.route_map['holds'])} holds")

        # HSV range for red holds
        # Red wraps around in HSV (0° and 360° are both red)
        # So we need two ranges: low red (0-10) and high red (170-180)
        self.hsv_lower_red1 = np.array([0, 100, 100])    # Low red
        self.hsv_upper_red1 = np.array([10, 255, 255])
        self.hsv_lower_red2 = np.array([170, 100, 100])  # High red
        self.hsv_upper_red2 = np.array([180, 255, 255])

    def _load_route_coordinates(self, path: str) -> dict:
        """Load IFSC route coordinates from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def detect_holds(
        self,
        frame: np.ndarray,
        lane: Optional[str] = None
    ) -> List[DetectedHold]:
        """Detect holds in a single video frame.

        Args:
            frame: RGB or BGR video frame
            lane: Optional lane filter ('left' or 'right')
                 If specified, only detect holds in that lane

        Returns:
            List of DetectedHold objects sorted by confidence (highest first)
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided")
            return []

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for red color (two ranges)
        mask1 = cv2.inRange(hsv, self.hsv_lower_red1, self.hsv_upper_red1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_red2, self.hsv_upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours
        detected_holds = []
        frame_height, frame_width = frame.shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Get centroid
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            # Filter by lane if specified
            if lane:
                mid_x = frame_width / 2
                if lane == 'left' and cx > mid_x:
                    continue
                if lane == 'right' and cx < mid_x:
                    continue

            # Calculate confidence based on:
            # 1. Circularity (holds are roughly circular)
            # 2. Size consistency
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            circularity = min(circularity, 1.0)  # Cap at 1.0

            # Size consistency (normalized to expected range)
            size_score = 1.0 - abs(area - 5000) / 10000
            size_score = max(0.0, min(size_score, 1.0))

            # Combined confidence
            confidence = (circularity * 0.7 + size_score * 0.3)

            if confidence < self.min_confidence:
                continue

            detected_hold = DetectedHold(
                hold_num=None,  # Will be matched later if route map available
                pixel_x=cx,
                pixel_y=cy,
                confidence=confidence,
                contour_area=area
            )

            detected_holds.append(detected_hold)

        # Sort by confidence (highest first)
        detected_holds.sort(key=lambda h: h.confidence, reverse=True)

        logger.debug(f"Detected {len(detected_holds)} holds in frame")
        return detected_holds

    def match_to_route(
        self,
        detected_holds: List[DetectedHold],
        frame_shape: Tuple[int, int],
        max_distance: float = 0.1
    ) -> List[DetectedHold]:
        """Match detected holds to expected route positions.

        This uses a simple nearest-neighbor matching based on normalized
        positions. For moving cameras, this should be called per-frame.

        Args:
            detected_holds: List of detected holds from detect_holds()
            frame_shape: (height, width) of the frame
            max_distance: Maximum normalized distance for matching (0-1)

        Returns:
            List of DetectedHold objects with matched hold_num, panel, and
            grid_position filled in. Unmatched holds have hold_num=None.
        """
        if not self.route_map:
            logger.warning("No route map loaded, cannot match holds")
            return detected_holds

        frame_height, frame_width = frame_shape

        # For each detected hold, find the nearest expected hold
        for detected in detected_holds:
            # Normalize pixel coordinates to 0-1 range
            norm_x = detected.pixel_x / frame_width
            norm_y = detected.pixel_y / frame_height

            # Find nearest route hold
            best_match = None
            best_distance = float('inf')

            for route_hold in self.route_map['holds']:
                # Get expected position (normalized to wall dimensions)
                # Note: This is a simplified approach - proper calibration
                # will use homography transformation
                expected_x = route_hold['wall_x_m'] / 6.0  # 6m total width
                expected_y = route_hold['wall_y_m'] / 15.0  # 15m height

                # Calculate Euclidean distance
                distance = np.sqrt((norm_x - expected_x)**2 + (norm_y - expected_y)**2)

                if distance < best_distance:
                    best_distance = distance
                    best_match = route_hold

            # Assign match if distance is acceptable
            if best_distance < max_distance and best_match:
                detected.hold_num = best_match['hold_num']
                detected.panel = best_match['panel']
                detected.grid_position = best_match['grid_position']

        # Count matched holds
        matched_count = sum(1 for h in detected_holds if h.hold_num is not None)
        logger.debug(f"Matched {matched_count}/{len(detected_holds)} holds to route")

        return detected_holds

    def visualize_detections(
        self,
        frame: np.ndarray,
        detected_holds: List[DetectedHold],
        show_labels: bool = True
    ) -> np.ndarray:
        """Visualize detected holds on the frame.

        Args:
            frame: Input frame (will be copied, not modified)
            detected_holds: List of detected holds
            show_labels: If True, show hold numbers and confidence

        Returns:
            Annotated frame with detected holds visualized
        """
        output = frame.copy()

        for hold in detected_holds:
            x, y = int(hold.pixel_x), int(hold.pixel_y)

            # Color based on confidence (green = high, red = low)
            color_value = int(255 * hold.confidence)
            color = (0, color_value, 255 - color_value)  # BGR

            # Draw circle
            cv2.circle(output, (x, y), 10, color, 2)

            # Draw label if requested
            if show_labels:
                if hold.hold_num is not None:
                    label = f"#{hold.hold_num} ({hold.confidence:.2f})"
                else:
                    label = f"? ({hold.confidence:.2f})"

                cv2.putText(
                    output, label, (x + 15, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )

        return output


def main():
    """Test hold detector on a sample frame."""
    import argparse

    parser = argparse.ArgumentParser(description="Detect holds in video frame")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--route-map",
        type=str,
        default="configs/ifsc_route_coordinates.json",
        help="Path to route map JSON"
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        default=0,
        help="Frame number to test (default: first frame)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hold_detection_test.png",
        help="Output image path"
    )
    parser.add_argument(
        "--lane",
        type=str,
        choices=['left', 'right'],
        help="Detect holds in specific lane only"
    )

    args = parser.parse_args()

    # Load video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {args.frame_num}")
        return

    # Create detector
    detector = HoldDetector(route_coordinates_path=args.route_map)

    # Detect holds
    print(f"Detecting holds in frame {args.frame_num}...")
    detected_holds = detector.detect_holds(frame, lane=args.lane)
    print(f"Found {len(detected_holds)} holds")

    # Match to route if available
    if detector.route_map:
        detected_holds = detector.match_to_route(
            detected_holds, frame.shape[:2]
        )
        matched_count = sum(1 for h in detected_holds if h.hold_num is not None)
        print(f"Matched {matched_count} holds to route")

    # Visualize
    output = detector.visualize_detections(frame, detected_holds)

    # Save
    cv2.imwrite(args.output, output)
    print(f"\n✅ Detection results saved to {args.output}")

    # Print details
    print("\nDetected holds:")
    for hold in detected_holds[:10]:  # Top 10
        if hold.hold_num:
            print(f"  Hold #{hold.hold_num} ({hold.grid_position}) at "
                  f"({hold.pixel_x:.0f}, {hold.pixel_y:.0f}) - "
                  f"confidence: {hold.confidence:.2f}")
        else:
            print(f"  Unmatched hold at ({hold.pixel_x:.0f}, {hold.pixel_y:.0f}) - "
                  f"confidence: {hold.confidence:.2f}")


if __name__ == "__main__":
    main()
