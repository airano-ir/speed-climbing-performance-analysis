"""
Detect IFSC red holds in video frames using color and shape-based detection.

IFSC speed climbing holds are distinctive 4-pointed red stars.
This module uses multiple detection strategies:
1. HSV color filtering with adaptive presets
2. Star shape detection via convexity analysis
3. Optional template matching
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum

import cv2
import numpy as np

from speed_climbing.core.settings import DEFAULT_PROCESSING_CONFIG

logger = logging.getLogger(__name__)


class LightingCondition(Enum):
    """Lighting condition presets for adaptive HSV filtering."""
    AUTO = "auto"
    DAYLIGHT = "daylight"
    INDOOR = "indoor"
    SPOTLIGHT = "spotlight"  # Night competitions with spotlights
    LOW_LIGHT = "low_light"


# HSV presets for different lighting conditions
HSV_PRESETS = {
    LightingCondition.DAYLIGHT: {
        "red1_lower": np.array([0, 120, 100]),
        "red1_upper": np.array([10, 255, 255]),
        "red2_lower": np.array([170, 120, 100]),
        "red2_upper": np.array([180, 255, 255]),
    },
    LightingCondition.INDOOR: {
        "red1_lower": np.array([0, 100, 80]),
        "red1_upper": np.array([12, 255, 255]),
        "red2_lower": np.array([168, 100, 80]),
        "red2_upper": np.array([180, 255, 255]),
    },
    LightingCondition.SPOTLIGHT: {
        "red1_lower": np.array([0, 80, 120]),
        "red1_upper": np.array([15, 255, 255]),
        "red2_lower": np.array([165, 80, 120]),
        "red2_upper": np.array([180, 255, 255]),
    },
    LightingCondition.LOW_LIGHT: {
        "red1_lower": np.array([0, 60, 60]),
        "red1_upper": np.array([15, 255, 255]),
        "red2_lower": np.array([165, 60, 60]),
        "red2_upper": np.array([180, 255, 255]),
    },
}


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
    star_score: float = 0.0  # How star-like the shape is (0-1)
    color_score: float = 0.0  # How red the detected region is (0-1)
    contour: Optional[np.ndarray] = field(default=None, repr=False)


class HoldDetector:
    """
    Detect red IFSC holds in video frames using color and shape analysis.

    IFSC holds are 4-pointed red stars with distinctive properties:
    - Red color (HSV hue 0-15 or 165-180)
    - Star shape with 4 points
    - Specific size range based on camera distance
    """

    def __init__(
        self,
        route_coordinates_path: Optional[str] = None,
        min_area: int = 300,  # Minimum contour area in pixels (lowered for distant holds)
        max_area: int = 20000,  # Maximum contour area
        min_confidence: float = 0.50,  # Minimum combined confidence
        lighting_condition: LightingCondition = LightingCondition.AUTO,
        use_star_detection: bool = True,  # Enable star shape analysis
        use_spatial_filtering: bool = True,  # Filter by expected positions
        spatial_tolerance_m: float = 0.20,  # Tolerance in meters
        star_weight: float = 0.40,  # Weight for star shape in confidence (increased)
        color_weight: float = 0.35,  # Weight for color match
        size_weight: float = 0.15,  # Weight for size appropriateness
        aspect_weight: float = 0.10,  # Weight for aspect ratio
        min_solidity: float = 0.45,  # Minimum solidity for star-like shapes
        max_solidity: float = 0.85,  # Maximum solidity (too high = circle/blob)
        min_aspect_ratio: float = 0.4,  # Minimum width/height ratio
        max_aspect_ratio: float = 2.5,  # Maximum width/height ratio
        min_star_score: float = 0.30,  # Minimum star shape score to be considered a hold
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_confidence = min_confidence
        self.lighting_condition = lighting_condition
        self.use_star_detection = use_star_detection
        self.use_spatial_filtering = use_spatial_filtering
        self.spatial_tolerance_m = spatial_tolerance_m
        self.route_map = None

        # Confidence weights
        self.star_weight = star_weight
        self.color_weight = color_weight
        self.size_weight = size_weight
        self.aspect_weight = aspect_weight

        # Shape filtering parameters
        self.min_solidity = min_solidity
        self.max_solidity = max_solidity
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_star_score = min_star_score

        # Will be set based on lighting condition
        self._hsv_preset = None
        self._auto_lighting = lighting_condition == LightingCondition.AUTO

        if route_coordinates_path:
            self._load_route_coordinates(route_coordinates_path)

        # Initialize HSV ranges
        self._update_hsv_preset(lighting_condition)

    def _update_hsv_preset(self, condition: LightingCondition):
        """Update HSV ranges based on lighting condition."""
        if condition == LightingCondition.AUTO:
            # Default to indoor as a reasonable middle ground
            condition = LightingCondition.INDOOR

        preset = HSV_PRESETS.get(condition, HSV_PRESETS[LightingCondition.INDOOR])
        self._hsv_preset = preset

        self.hsv_lower_red1 = preset["red1_lower"]
        self.hsv_upper_red1 = preset["red1_upper"]
        self.hsv_lower_red2 = preset["red2_lower"]
        self.hsv_upper_red2 = preset["red2_upper"]

    def _detect_lighting_condition(self, frame: np.ndarray) -> LightingCondition:
        """
        Automatically detect lighting condition from frame brightness/contrast.

        Returns appropriate LightingCondition enum value.
        """
        # Convert to grayscale and analyze
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Check for spotlight effect (high contrast, bright spots)
        if std_brightness > 70 and mean_brightness > 100:
            return LightingCondition.SPOTLIGHT

        # Daylight (bright, even lighting)
        if mean_brightness > 140:
            return LightingCondition.DAYLIGHT

        # Low light
        if mean_brightness < 80:
            return LightingCondition.LOW_LIGHT

        # Default to indoor
        return LightingCondition.INDOOR

    def _load_route_coordinates(self, path: str):
        """Load IFSC route coordinates from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.route_map = json.load(f)
            logger.info(f"Loaded route map with {len(self.route_map['holds'])} holds")
        except Exception as e:
            logger.error(f"Failed to load route map from {path}: {e}")
            self.route_map = None

    def _calculate_star_score(self, contour: np.ndarray) -> float:
        """
        Calculate how star-like a contour is.

        IFSC holds are 4-pointed stars with:
        - 4 convexity defects (valleys between points)
        - Relatively low solidity (area/convex_hull_area)
        - Specific perimeter-to-area ratio

        Returns:
            Score from 0.0 (not star-like) to 1.0 (perfect star)
        """
        if len(contour) < 5:
            return 0.0

        try:
            area = cv2.contourArea(contour)
            if area < 100:
                return 0.0

            # Get convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            if hull_area == 0:
                return 0.0

            # Solidity: ratio of contour area to convex hull area
            # Stars have lower solidity than circles (0.5-0.75 typical)
            solidity = area / hull_area

            # Ideal solidity for 4-pointed star is around 0.5-0.7
            # Too high means circle-like, too low means too irregular
            if 0.45 <= solidity <= 0.75:
                solidity_score = 1.0 - abs(solidity - 0.6) / 0.15
            else:
                solidity_score = max(0, 0.5 - abs(solidity - 0.6))

            # Count convexity defects
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if hull_indices is None or len(hull_indices) < 3:
                return solidity_score * 0.5

            try:
                defects = cv2.convexityDefects(contour, hull_indices)
            except cv2.error:
                return solidity_score * 0.5

            if defects is None:
                defect_score = 0.3
            else:
                # Count significant defects (depth > threshold)
                perimeter = cv2.arcLength(contour, True)
                depth_threshold = perimeter * 0.03  # 3% of perimeter

                significant_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth = d / 256.0  # Convert from fixed-point
                    if depth > depth_threshold:
                        significant_defects += 1

                # 4-pointed star should have 4 defects
                if significant_defects == 4:
                    defect_score = 1.0
                elif significant_defects == 3 or significant_defects == 5:
                    defect_score = 0.7
                elif 2 <= significant_defects <= 6:
                    defect_score = 0.4
                else:
                    defect_score = 0.1

            # Perimeter to area ratio check
            perimeter = cv2.arcLength(contour, True)
            # For star shapes, this ratio is higher than for circles
            pa_ratio = perimeter / np.sqrt(area)
            # Ideal for 4-pointed star is around 5-7
            if 4.5 <= pa_ratio <= 8.0:
                pa_score = 1.0 - abs(pa_ratio - 6.0) / 3.0
            else:
                pa_score = 0.2

            # Combined star score
            star_score = (
                solidity_score * 0.4 +
                defect_score * 0.4 +
                pa_score * 0.2
            )

            return min(1.0, max(0.0, star_score))

        except Exception as e:
            logger.debug(f"Error calculating star score: {e}")
            return 0.0

    def _calculate_color_score(
        self,
        frame: np.ndarray,
        contour: np.ndarray
    ) -> float:
        """
        Calculate how well the contour matches expected red color.

        Returns:
            Score from 0.0 to 1.0 based on color match
        """
        try:
            # Create mask for this contour
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)

            # Get mean color in HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mean_hsv = cv2.mean(hsv, mask=mask)[:3]

            h, s, v = mean_hsv

            # Check if hue is in red range
            if h <= 15 or h >= 165:
                hue_score = 1.0
            elif h <= 25 or h >= 155:
                hue_score = 0.5
            else:
                hue_score = 0.0

            # Check saturation (should be high for vivid red)
            sat_score = min(1.0, s / 150.0) if s > 50 else 0.3

            # Check value (should be reasonably bright)
            val_score = min(1.0, v / 180.0) if v > 40 else 0.2

            return hue_score * 0.5 + sat_score * 0.3 + val_score * 0.2

        except Exception as e:
            logger.debug(f"Error calculating color score: {e}")
            return 0.5

    def detect_holds(
        self,
        frame: np.ndarray,
        lane: Optional[str] = None,
        return_mask: bool = False,
        return_debug_info: bool = False
    ) -> List[DetectedHold]:
        """
        Detect holds in a single video frame.

        Args:
            frame: Input BGR frame
            lane: Optional lane filter ('left' or 'right')
            return_mask: If True, return (holds, mask) tuple
            return_debug_info: If True, return (holds, debug_dict)

        Returns:
            List of DetectedHold objects, or tuple if return_mask/return_debug_info
        """
        if frame is None or frame.size == 0:
            empty_result = []
            if return_mask:
                return empty_result, None
            if return_debug_info:
                return empty_result, {}
            return empty_result

        # Auto-detect lighting if enabled
        if self._auto_lighting:
            detected_condition = self._detect_lighting_condition(frame)
            self._update_hsv_preset(detected_condition)
            logger.debug(f"Auto-detected lighting: {detected_condition.value}")

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for red color using current HSV preset
        mask1 = cv2.inRange(hsv, self.hsv_lower_red1, self.hsv_upper_red1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_red2, self.hsv_upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to clean up mask
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)

        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_holds = []
        debug_info = {"candidates": 0, "filtered_area": 0, "filtered_shape": 0, "filtered_confidence": 0}
        frame_height, frame_width = frame.shape[:2]
        mid_x = frame_width / 2

        for contour in contours:
            area = cv2.contourArea(contour)
            debug_info["candidates"] += 1

            # Area filtering
            if area < self.min_area or area > self.max_area:
                debug_info["filtered_area"] += 1
                continue

            # Get centroid
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

            # Bounding box analysis
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Early shape filtering based on aspect ratio
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                debug_info["filtered_shape"] += 1
                continue

            # Calculate solidity for shape filtering
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Solidity filtering - stars have moderate solidity (not too high, not too low)
            if solidity < self.min_solidity or solidity > self.max_solidity:
                debug_info["filtered_shape"] += 1
                continue

            # Calculate shape scores
            star_score = self._calculate_star_score(contour) if self.use_star_detection else 0.5

            # Filter by minimum star score to eliminate non-star shapes (e.g., athlete clothing)
            if self.use_star_detection and star_score < self.min_star_score:
                debug_info["filtered_shape"] += 1
                continue

            color_score = self._calculate_color_score(frame, contour)

            # Aspect ratio score (stars are roughly square)
            aspect_score = 1.0 - min(abs(aspect_ratio - 1.0), 1.0)
            aspect_score = max(0.3, aspect_score)

            # Size score - adaptive based on frame size
            # For a 1024px wide frame, ideal hold is ~600-1500px area
            frame_scale = frame_width / 1024.0
            ideal_size = 800 * (frame_scale ** 2)
            size_score = 1.0 - min(abs(area - ideal_size) / (ideal_size * 2), 0.8)
            size_score = max(0.2, size_score)

            # Combined confidence
            confidence = (
                star_score * self.star_weight +
                color_score * self.color_weight +
                size_score * self.size_weight +
                aspect_score * self.aspect_weight
            )

            if confidence < self.min_confidence:
                debug_info["filtered_confidence"] += 1
                continue

            detected_holds.append(DetectedHold(
                hold_num=None,
                pixel_x=cx,
                pixel_y=cy,
                confidence=confidence,
                contour_area=area,
                star_score=star_score,
                color_score=color_score,
                contour=contour
            ))

        # Sort by confidence
        detected_holds.sort(key=lambda h: h.confidence, reverse=True)

        logger.debug(f"Detected {len(detected_holds)} holds "
                    f"(candidates: {debug_info['candidates']}, "
                    f"area_filtered: {debug_info['filtered_area']}, "
                    f"conf_filtered: {debug_info['filtered_confidence']})")

        if return_mask:
            return detected_holds, mask
        if return_debug_info:
            debug_info["detected"] = len(detected_holds)
            return detected_holds, debug_info

        return detected_holds

    def filter_by_spatial_grid(
        self,
        detected_holds: List[DetectedHold],
        homography: Optional[np.ndarray],
        frame_shape: Tuple[int, int],
        lane: Optional[str] = None
    ) -> List[DetectedHold]:
        """
        Filter detected holds by comparing to expected grid positions from route map.

        Args:
            detected_holds: List of initially detected holds
            homography: Homography matrix for pixel->world coordinate transform
            frame_shape: (height, width) of frame
            lane: 'left' or 'right' lane (filters route map by panel)

        Returns:
            Filtered list of holds that match expected positions
        """
        if not self.route_map or homography is None:
            return detected_holds

        route_holds = self.route_map.get('holds', [])
        if not route_holds:
            return detected_holds

        # Filter route map by lane if specified
        if lane:
            lane_prefix = 'SN' if lane == 'left' else 'DX'
            route_holds = [h for h in route_holds if h.get('panel', '').startswith(lane_prefix)]

        if not route_holds:
            return detected_holds

        # Extract expected world positions
        expected_positions = np.array([
            [h['wall_x_m'], h['wall_y_m']] for h in route_holds
        ], dtype=np.float32)

        filtered_holds = []
        used_route_indices = set()

        for hold in detected_holds:
            pixel_point = np.array([[hold.pixel_x, hold.pixel_y]], dtype=np.float32)

            try:
                world_point = cv2.perspectiveTransform(
                    pixel_point.reshape(-1, 1, 2),
                    homography
                )
                world_x, world_y = world_point[0][0]

                # Find closest expected position
                distances = np.sqrt(
                    (expected_positions[:, 0] - world_x) ** 2 +
                    (expected_positions[:, 1] - world_y) ** 2
                )

                # Exclude already matched positions
                for idx in used_route_indices:
                    distances[idx] = float('inf')

                min_distance = np.min(distances)
                closest_idx = np.argmin(distances)

                if min_distance <= self.spatial_tolerance_m:
                    matched_hold_info = route_holds[closest_idx]
                    hold.hold_num = matched_hold_info.get('hold_num')
                    hold.panel = matched_hold_info.get('panel')
                    hold.grid_position = matched_hold_info.get('grid_position')

                    # Boost confidence for spatially validated holds
                    hold.confidence = min(hold.confidence * 1.15, 1.0)

                    filtered_holds.append(hold)
                    used_route_indices.add(closest_idx)

            except Exception as e:
                logger.debug(f"Failed to transform hold at ({hold.pixel_x}, {hold.pixel_y}): {e}")
                continue

        logger.info(f"Spatial filtering: {len(detected_holds)} -> {len(filtered_holds)} holds")

        return filtered_holds

    def visualize_detections(
        self,
        frame: np.ndarray,
        detected_holds: List[DetectedHold],
        show_labels: bool = True,
        show_contours: bool = False,
        show_star_score: bool = False
    ) -> np.ndarray:
        """Visualize detected holds on the frame."""
        output = frame.copy()

        for hold in detected_holds:
            x, y = int(hold.pixel_x), int(hold.pixel_y)

            # Color based on confidence
            if hold.confidence > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif hold.confidence > 0.5:
                color = (0, 255, 255)  # Yellow - medium
            else:
                color = (0, 165, 255)  # Orange - low

            # Draw contour if available
            if show_contours and hold.contour is not None:
                cv2.drawContours(output, [hold.contour], 0, color, 2)
            else:
                cv2.circle(output, (x, y), 12, color, 2)

            if show_labels:
                label_parts = []
                if hold.hold_num:
                    label_parts.append(f"#{hold.hold_num}")
                label_parts.append(f"{hold.confidence:.2f}")
                if show_star_score:
                    label_parts.append(f"*{hold.star_score:.2f}")

                label = " ".join(label_parts)
                cv2.putText(
                    output, label, (x + 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )

        return output

    def set_lighting_condition(self, condition: LightingCondition):
        """Manually set lighting condition."""
        self.lighting_condition = condition
        self._auto_lighting = (condition == LightingCondition.AUTO)
        if not self._auto_lighting:
            self._update_hsv_preset(condition)
