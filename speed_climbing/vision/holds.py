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
        min_area: int = 50,  # Reduced to catch smaller blobs
        max_area: int = 100000,  # Increased to catch larger regions
        min_confidence: float = DEFAULT_PROCESSING_CONFIG["min_hold_confidence"],
        use_adaptive_hsv: bool = True  # New: adaptive HSV based on lighting
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_confidence = min_confidence
        self.use_adaptive_hsv = use_adaptive_hsv
        self.route_map = None

        if route_coordinates_path:
            self._load_route_coordinates(route_coordinates_path)

        # HSV range for red holds (Expanded for better detection)
        # Range 1: 0-15 (Red) - Expanded upper bound
        self.hsv_lower_red1 = np.array([0, 70, 70])  # Reduced saturation/value thresholds
        self.hsv_upper_red1 = np.array([15, 255, 255])

        # Range 2: 165-180 (Red wrap-around) - Expanded lower bound
        self.hsv_lower_red2 = np.array([165, 70, 70])  # Reduced saturation/value thresholds
        self.hsv_upper_red2 = np.array([180, 255, 255])

    def _load_route_coordinates(self, path: str):
        """Load IFSC route coordinates from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.route_map = json.load(f)
            logger.info(f"Loaded route map with {len(self.route_map['holds'])} holds")
        except Exception as e:
            logger.error(f"Failed to load route map from {path}: {e}")
            self.route_map = None

    def detect_holds(
        self,
        frame: np.ndarray,
        lane: Optional[str] = None,
        return_mask: bool = False  # New: option to return mask for debugging
    ) -> List[DetectedHold]:
        """
        Detect holds in a single video frame using blob/region detection.

        Args:
            frame: Input BGR frame
            lane: Optional lane filter ('left' or 'right')
            return_mask: If True, return (holds, mask) instead of just holds

        Returns:
            List of DetectedHold objects, or tuple (holds, mask) if return_mask=True
        """
        if frame is None or frame.size == 0:
            return [] if not return_mask else ([], None)

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for red color with expanded range
        mask1 = cv2.inRange(hsv, self.hsv_lower_red1, self.hsv_upper_red1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_red2, self.hsv_upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Gentler morphological operations to preserve blob shape
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)

        # Remove noise (small artifacts)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # Fill small holes within blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)

        # Optional: Dilate slightly to merge nearby regions
        mask = cv2.dilate(mask, kernel_small, iterations=1)

        # Find contours (regions of red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_holds = []
        frame_height, frame_width = frame.shape[:2]
        mid_x = frame_width / 2

        for contour in contours:
            area = cv2.contourArea(contour)

            # More lenient area filtering
            if area < self.min_area or area > self.max_area:
                continue

            # Get moments for centroid
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

            # Improved confidence calculation based on blob properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # Bounding box for aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Circularity (1.0 = perfect circle)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            circularity = min(circularity, 1.0)

            # Extent (how much of bounding box is filled)
            extent = area / (w * h) if (w * h) > 0 else 0

            # Size score (prefer medium-sized blobs, but be more lenient)
            # Typical hold blob: 500-15000 pixels
            ideal_size = 3000
            size_score = 1.0 - min(abs(area - ideal_size) / ideal_size, 1.0)
            size_score = max(0.2, size_score)  # Minimum score of 0.2

            # Aspect ratio score (prefer roughly square blobs)
            aspect_score = 1.0 - min(abs(aspect_ratio - 1.0), 1.0)
            aspect_score = max(0.3, aspect_score)  # Minimum score of 0.3

            # Combined confidence
            # Weighted: size matters most, then extent (filled blob), then circularity
            confidence = (
                size_score * 0.4 +
                extent * 0.3 +
                circularity * 0.2 +
                aspect_score * 0.1
            )

            # More lenient threshold
            if confidence < max(self.min_confidence * 0.5, 0.2):
                continue

            detected_holds.append(DetectedHold(
                hold_num=None,
                pixel_x=cx,
                pixel_y=cy,
                confidence=confidence,
                contour_area=area
            ))

        detected_holds.sort(key=lambda h: h.confidence, reverse=True)

        if return_mask:
            return detected_holds, mask

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
