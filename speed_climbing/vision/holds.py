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
        min_area: int = 500,  # Increased to filter small noise (typical hold: 1000-10000px)
        max_area: int = 30000,  # Reduced to avoid large false regions
        min_confidence: float = 0.4,  # Increased from default to reduce false positives
        use_adaptive_hsv: bool = True,  # New: adaptive HSV based on lighting
        use_spatial_filtering: bool = True,  # New: filter by expected grid positions
        spatial_tolerance_m: float = 0.06  # Stricter tolerance: 6cm (reduced from 15cm)
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_confidence = min_confidence
        self.use_adaptive_hsv = use_adaptive_hsv
        self.use_spatial_filtering = use_spatial_filtering
        self.spatial_tolerance_m = spatial_tolerance_m
        self.route_map = None

        if route_coordinates_path:
            self._load_route_coordinates(route_coordinates_path)

        # HSV range for red holds (More restrictive for fewer false positives)
        # Range 1: 0-12 (Pure red, tightened from 0-15)
        self.hsv_lower_red1 = np.array([0, 100, 100])  # Increased saturation/value for purer red
        self.hsv_upper_red1 = np.array([12, 255, 255])

        # Range 2: 168-180 (Red wrap-around, tightened from 165-180)
        self.hsv_lower_red2 = np.array([168, 100, 100])  # Increased saturation/value
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
        return_mask: bool = False,  # Option to return mask for debugging
        roi_mask: Optional[np.ndarray] = None,  # External ROI mask (1=valid, 0=ignore)
        climber_mask: Optional[np.ndarray] = None  # Climber mask (1=climber, 0=background)
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

            # ROI filtering: check if centroid is within valid region
            if roi_mask is not None:
                cy_int, cx_int = int(cy), int(cx)
                if 0 <= cy_int < frame_height and 0 <= cx_int < frame_width:
                    if roi_mask[cy_int, cx_int] == 0:
                        continue  # Outside ROI, skip
                else:
                    continue  # Out of bounds

            # Climber masking: check if centroid overlaps with climber
            if climber_mask is not None:
                cy_int, cx_int = int(cy), int(cx)
                if 0 <= cy_int < frame_height and 0 <= cx_int < frame_width:
                    if climber_mask[cy_int, cx_int] > 0:
                        continue  # Overlaps with climber, skip

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

            # Size score (prefer medium-sized blobs)
            # Typical hold blob: 1000-10000 pixels at typical camera distance
            ideal_size = 3000
            size_score = 1.0 - min(abs(area - ideal_size) / ideal_size, 1.0)
            size_score = max(0.1, size_score)  # Stricter minimum

            # Aspect ratio score (prefer roughly square blobs, but tolerate some variation)
            aspect_score = 1.0 - min(abs(aspect_ratio - 1.0), 1.0)
            aspect_score = max(0.2, aspect_score)  # Stricter minimum

            # Combined confidence
            # Weighted: circularity and extent matter most for holds
            confidence = (
                circularity * 0.35 +  # Holds tend to be round/circular
                extent * 0.35 +       # Holds should fill their bounding box well
                size_score * 0.2 +    # Size should be reasonable
                aspect_score * 0.1    # Aspect ratio less critical
            )

            # Stricter threshold to reduce false positives
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

        if return_mask:
            return detected_holds, mask

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

        Uses a greedy matching algorithm with uniqueness constraint:
        1. Transform all detected holds to world coordinates
        2. For each expected hold position, find the closest detection within tolerance
        3. Each expected hold can match at most one detection (uniqueness)
        4. Confidence is adjusted based on distance from expected position

        Args:
            detected_holds: List of initially detected holds
            homography: Homography matrix for pixel->world coordinate transform
            frame_shape: (height, width) of frame
            lane: 'left' or 'right' lane (filters route map by panel)

        Returns:
            Filtered list of holds that match expected positions
        """
        # Check if filtering is possible (route map and homography must exist)
        # Note: We ignore self.use_spatial_filtering here since this function
        # is explicitly called - the caller decides whether to filter or not
        if not self.route_map or homography is None:
            logger.debug("Spatial filtering not possible: " +
                        ("no route map" if not self.route_map else "no homography"))
            return detected_holds

        # Get expected hold positions from route map
        route_holds = self.route_map.get('holds', [])
        if not route_holds:
            logger.warning("No holds found in route map")
            return detected_holds

        # Filter route map by lane if specified
        if lane:
            lane_prefix = 'SN' if lane == 'left' else 'DX'
            route_holds = [h for h in route_holds if h.get('panel', '').startswith(lane_prefix)]
            logger.debug(f"Filtered to {len(route_holds)} holds for lane '{lane}' (panel prefix '{lane_prefix}')")

        if not route_holds:
            logger.warning(f"No holds found for lane '{lane}'")
            return []

        # Transform detected holds to world coordinates
        height, width = frame_shape
        detected_world_coords = []

        for hold in detected_holds:
            pixel_point = np.array([[hold.pixel_x, hold.pixel_y]], dtype=np.float32)

            try:
                world_point = cv2.perspectiveTransform(
                    pixel_point.reshape(-1, 1, 2),
                    homography
                )
                world_x, world_y = world_point[0][0]
                detected_world_coords.append((world_x, world_y, hold))
            except Exception as e:
                logger.debug(f"Failed to transform hold at ({hold.pixel_x:.1f}, {hold.pixel_y:.1f}): {e}")
                continue

        if not detected_world_coords:
            logger.warning("No detections could be transformed to world coordinates")
            return []

        # Build distance matrix: [expected_hold_idx][detected_hold_idx] -> distance
        n_expected = len(route_holds)
        n_detected = len(detected_world_coords)
        distance_matrix = np.zeros((n_expected, n_detected))

        for i, route_hold in enumerate(route_holds):
            expected_x = route_hold['wall_x_m']
            expected_y = route_hold['wall_y_m']

            for j, (world_x, world_y, _) in enumerate(detected_world_coords):
                distance = np.sqrt((expected_x - world_x)**2 + (expected_y - world_y)**2)
                distance_matrix[i, j] = distance

        # Greedy matching: for each expected hold, find best detection within tolerance
        # This ensures uniqueness: each detection matches at most one expected hold
        matched_pairs = []  # List of (expected_idx, detected_idx, distance)
        used_detections = set()

        # Sort expected holds by their best match distance (prioritize good matches)
        expected_best_distances = np.min(distance_matrix, axis=1)
        expected_order = np.argsort(expected_best_distances)

        for exp_idx in expected_order:
            # Find best unmatched detection for this expected hold
            best_det_idx = None
            best_distance = float('inf')

            for det_idx in range(n_detected):
                if det_idx in used_detections:
                    continue

                dist = distance_matrix[exp_idx, det_idx]
                if dist < best_distance and dist <= self.spatial_tolerance_m:
                    best_distance = dist
                    best_det_idx = det_idx

            if best_det_idx is not None:
                matched_pairs.append((exp_idx, best_det_idx, best_distance))
                used_detections.add(best_det_idx)

                logger.debug(f"Matched: expected hold #{route_holds[exp_idx].get('hold_num')} "
                           f"<-> detection at ({detected_world_coords[best_det_idx][0]:.2f}, "
                           f"{detected_world_coords[best_det_idx][1]:.2f}) m, "
                           f"distance={best_distance:.3f}m")

        # Build filtered holds list
        filtered_holds = []

        for exp_idx, det_idx, distance in matched_pairs:
            route_hold = route_holds[exp_idx]
            _, _, detected_hold = detected_world_coords[det_idx]

            # Update hold metadata
            detected_hold.hold_num = route_hold.get('hold_num')
            detected_hold.panel = route_hold.get('panel')
            detected_hold.grid_position = route_hold.get('grid_position')

            # Adjust confidence based on distance from expected position
            # Closer = higher confidence boost
            distance_score = max(0.0, 1.0 - (distance / self.spatial_tolerance_m))

            # Boost original confidence, weighted by distance accuracy
            # Very close matches (< 3cm) get full boost, far matches (> 6cm) get minimal boost
            confidence_boost = 1.0 + (0.3 * distance_score)
            detected_hold.confidence = min(detected_hold.confidence * confidence_boost, 1.0)

            filtered_holds.append(detected_hold)

        # Sort by confidence
        filtered_holds.sort(key=lambda h: h.confidence, reverse=True)

        logger.info(f"Spatial filtering: {len(detected_holds)} detections -> {len(filtered_holds)} valid holds "
                   f"(removed {len(detected_holds) - len(filtered_holds)} false positives, "
                   f"tolerance={self.spatial_tolerance_m*100:.1f}cm)")

        return filtered_holds

    def create_climber_mask_from_pose(
        self,
        frame_shape: Tuple[int, int],
        pose_result,  # PoseResult from BlazePoseExtractor
        expansion_factor: float = 1.5  # Expand bounding box by this factor
    ) -> np.ndarray:
        """
        Create a binary mask of the climber's body based on pose keypoints.

        Args:
            frame_shape: (height, width) of frame
            pose_result: PoseResult object with detected keypoints
            expansion_factor: Factor to expand bounding box (to ensure full coverage)

        Returns:
            Binary mask where 1=climber, 0=background
        """
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if not pose_result or not pose_result.has_detection:
            return mask

        # Collect all visible keypoints
        keypoints_pixel = []
        for kp in pose_result.keypoints.values():
            if kp.visibility > 0.5:  # Only use visible keypoints
                px, py = kp.to_pixel_coords(w, h)
                keypoints_pixel.append((px, py))

        if len(keypoints_pixel) < 4:
            return mask  # Not enough keypoints

        # Find bounding box
        xs = [p[0] for p in keypoints_pixel]
        ys = [p[1] for p in keypoints_pixel]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Expand bounding box
        box_w = max_x - min_x
        box_h = max_y - min_y
        expand_w = int(box_w * (expansion_factor - 1.0) / 2)
        expand_h = int(box_h * (expansion_factor - 1.0) / 2)

        min_x = max(0, min_x - expand_w)
        max_x = min(w, max_x + expand_w)
        min_y = max(0, min_y - expand_h)
        max_y = min(h, max_y + expand_h)

        # Fill bounding box region
        mask[min_y:max_y, min_x:max_x] = 1

        return mask

    def create_wall_roi_mask(
        self,
        frame_shape: Tuple[int, int],
        vertical_trim_top: float = 0.05,  # Trim top 5% (scoreboard, etc.)
        vertical_trim_bottom: float = 0.15,  # Trim bottom 15% (floor, ads)
        horizontal_margin: float = 0.05  # Trim 5% from left/right edges
    ) -> np.ndarray:
        """
        Create a ROI mask for the climbing wall area (excludes scoreboard, floor, ads).

        Args:
            frame_shape: (height, width) of frame
            vertical_trim_top: Fraction of height to trim from top
            vertical_trim_bottom: Fraction of height to trim from bottom
            horizontal_margin: Fraction of width to trim from left and right

        Returns:
            Binary mask where 1=valid wall region, 0=ignore
        """
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Calculate valid region
        top = int(h * vertical_trim_top)
        bottom = int(h * (1.0 - vertical_trim_bottom))
        left = int(w * horizontal_margin)
        right = int(w * (1.0 - horizontal_margin))

        # Fill valid region
        mask[top:bottom, left:right] = 1

        return mask

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
