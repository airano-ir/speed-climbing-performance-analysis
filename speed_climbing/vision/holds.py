"""
Detect IFSC red holds in video frames using color-based detection.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from speed_climbing.core.settings import DEFAULT_PROCESSING_CONFIG

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
    """Detect red IFSC holds in video frames using HSV color thresholding."""

    def __init__(
        self,
        route_coordinates_path: Optional[str] = None,
        min_area: int = 100,
        max_area: int = 50000,
        min_confidence: float = DEFAULT_PROCESSING_CONFIG["min_hold_confidence"]
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_confidence = min_confidence
        self.route_map = None

        if route_coordinates_path:
            self._load_route_coordinates(route_coordinates_path)

        # HSV range for red holds (Standard IFSC Red)
        # Range 1: 0-10 (Red)
        self.hsv_lower_red1 = np.array([0, 100, 100])
        self.hsv_upper_red1 = np.array([10, 255, 255])
        
        # Range 2: 170-180 (Red wrap-around)
        self.hsv_lower_red2 = np.array([170, 100, 100])
        self.hsv_upper_red2 = np.array([180, 255, 255])

    def _load_route_coordinates(self, path: str):
        """Load IFSC route coordinates from JSON file."""
        try:
            with open(path, 'r') as f:
                self.route_map = json.load(f)
            logger.info(f"Loaded route map with {len(self.route_map['holds'])} holds")
        except Exception as e:
            logger.error(f"Failed to load route map from {path}: {e}")
            self.route_map = None

    def detect_holds(
        self,
        frame: np.ndarray,
        lane: Optional[str] = None
    ) -> List[DetectedHold]:
        """Detect holds in a single video frame."""
        if frame is None or frame.size == 0:
            return []

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for red color
        mask1 = cv2.inRange(hsv, self.hsv_lower_red1, self.hsv_upper_red1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_red2, self.hsv_upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_holds = []
        frame_height, frame_width = frame.shape[:2]
        mid_x = frame_width / 2

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area or area > self.max_area:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            # Lane filtering
            if lane == 'left' and cx > mid_x:
                continue
            if lane == 'right' and cx < mid_x:
                continue

            # Confidence calculation
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            circularity = min(circularity, 1.0)

            size_score = 1.0 - abs(area - 5000) / 10000
            size_score = max(0.0, min(size_score, 1.0))

            confidence = (circularity * 0.7 + size_score * 0.3)

            if confidence < self.min_confidence:
                continue

            detected_holds.append(DetectedHold(
                hold_num=None,
                pixel_x=cx,
                pixel_y=cy,
                confidence=confidence,
                contour_area=area
            ))

        detected_holds.sort(key=lambda h: h.confidence, reverse=True)
        return detected_holds

    def visualize_detections(
        self,
        frame: np.ndarray,
        detected_holds: List[DetectedHold],
        show_labels: bool = True
    ) -> np.ndarray:
        """Visualize detected holds on the frame."""
        output = frame.copy()

        for hold in detected_holds:
            x, y = int(hold.pixel_x), int(hold.pixel_y)
            color_value = int(255 * hold.confidence)
            color = (0, color_value, 255 - color_value)  # BGR

            cv2.circle(output, (x, y), 10, color, 2)

            if show_labels:
                label = f"{hold.confidence:.2f}"
                if hold.hold_num:
                    label = f"#{hold.hold_num} " + label
                
                cv2.putText(
                    output, label, (x + 15, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )

        return output
