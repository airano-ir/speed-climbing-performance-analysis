"""
Camera calibration using homography transformation with hybrid fallback strategies.

This module handles the transformation between pixel coordinates (from video frames)
and world coordinates (meters on the climbing wall) using hold detection as reference points.

Hybrid Calibration Strategy:
- ≥4 holds: Full Homography (8 DOF) - Most accurate
- 2-3 holds: Affine Transform (6 DOF) + Wall constraints - Good fallback
- <2 holds: Temporal Cache from previous frames - Emergency fallback
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict
from enum import Enum
import cv2
import numpy as np

from speed_climbing.core.settings import DEFAULT_PROCESSING_CONFIG, IFSC_STANDARDS

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Calibration method used for transformation."""
    HOMOGRAPHY = "homography"  # 8 DOF, requires ≥4 points
    AFFINE = "affine"  # 6 DOF, requires ≥3 points (or 2 with constraints)
    SIMILARITY = "similarity"  # 4 DOF, requires ≥2 points
    TEMPORAL_CACHE = "temporal_cache"  # Using cached calibration from previous frames
    FAILED = "failed"  # No calibration available


@dataclass
class CalibrationResult:
    """Result of camera calibration."""
    homography_matrix: np.ndarray
    pixel_to_meter_scale: float
    rmse_error: float
    inlier_count: int
    total_holds: int
    inlier_ratio: float
    confidence: float
    matched_hold_nums: List[int] = None
    pixel_to_meter_func: Optional[Callable[[float, float], Tuple[float, float]]] = None
    meter_to_pixel_func: Optional[Callable[[float, float], Tuple[float, float]]] = None
    # New fields for hybrid calibration
    method: CalibrationMethod = CalibrationMethod.HOMOGRAPHY
    is_cached: bool = False
    cache_age_frames: int = 0  # How old the cached calibration is
    anchor_points_used: int = 0  # Number of anchor points (holds + edges)
    edge_anchors_count: int = 0  # Number of wall edge anchor points used


class CameraCalibrator:
    """
    Calibrate camera using detected holds and IFSC route map.

    Uses homography transformation to map pixel coordinates to world coordinates.
    The calibration works per-lane, matching detected red holds to known IFSC positions.
    """

    # Lane width and wall height from IFSC standards
    LANE_WIDTH_M = 1.5
    WALL_HEIGHT_M = IFSC_STANDARDS["WALL_HEIGHT_M"]

    def __init__(
        self,
        route_coordinates_path: str,
        min_holds_for_calibration: int = 4,
        min_holds_for_affine: int = 2,  # Minimum holds for affine fallback
        ransac_threshold: float = 0.08,  # Increased for better tolerance
        min_inlier_ratio: float = 0.4,   # Reduced for partial visibility
        max_match_distance: float = 0.25,  # Max normalized distance for matching
        enable_affine_fallback: bool = True,  # Enable Affine transform for 2-3 holds
        enable_edge_anchors: bool = True,  # Enable wall edge detection as anchors
    ):
        self.min_holds = min_holds_for_calibration
        self.min_holds_affine = min_holds_for_affine
        self.ransac_threshold = ransac_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.max_match_distance = max_match_distance
        self.enable_affine_fallback = enable_affine_fallback
        self.enable_edge_anchors = enable_edge_anchors
        self.route_map = None

        try:
            with open(route_coordinates_path, 'r', encoding='utf-8') as f:
                self.route_map = json.load(f)
                # Get wall dimensions from route map or use defaults
                wall_info = self.route_map.get('wall', {})
                self.lane_width = wall_info.get('width_m', self.LANE_WIDTH_M)
                self.wall_height = wall_info.get('height_m', self.WALL_HEIGHT_M)
        except Exception as e:
            logger.error(f"Failed to load route map: {e}")
            self.route_map = None
            self.lane_width = self.LANE_WIDTH_M
            self.wall_height = self.WALL_HEIGHT_M

    def calibrate(
        self,
        frame: np.ndarray,
        detected_holds: List,
        lane: str = 'left'
    ) -> Optional[CalibrationResult]:
        """
        Calibrate camera for a single frame using hybrid strategy.

        Strategy:
        - ≥4 holds: Full Homography (8 DOF) - Most accurate
        - 2-3 holds: Affine Transform (6 DOF) + Wall constraints - Good fallback
        - <2 holds: Return None (caller should use temporal cache)

        Args:
            frame: Video frame (BGR)
            detected_holds: List of DetectedHold objects
            lane: 'left' (SN panels) or 'right' (DX panels)

        Returns:
            CalibrationResult if successful, None otherwise
        """
        if not self.route_map:
            logger.debug("No route map available for calibration")
            return None

        # Filter route holds by lane
        lane_prefix = 'SN' if lane == 'left' else 'DX'
        route_holds = [
            h for h in self.route_map['holds']
            if h.get('panel', '').startswith(lane_prefix)
        ]

        if len(route_holds) < self.min_holds_affine:
            logger.debug(f"Insufficient route holds for lane {lane}")
            return None

        # Match detected holds to route map
        pixel_points, meter_points, matched_nums = self._match_holds_to_route(
            detected_holds, route_holds, frame.shape, lane
        )

        num_matched = len(pixel_points)
        edge_anchors_count = 0

        # Strategy selection based on number of matched holds
        if num_matched >= self.min_holds:
            # Full Homography (8 DOF) - Most accurate
            result = self._calibrate_homography(
                pixel_points, meter_points, matched_nums, detected_holds, frame.shape
            )
            if result:
                result.method = CalibrationMethod.HOMOGRAPHY
                return result

        # Try Affine fallback if enabled and we have 2-3 holds
        if self.enable_affine_fallback and num_matched >= self.min_holds_affine:
            # Try to augment with edge anchors if enabled
            if self.enable_edge_anchors and num_matched < 3:
                edge_pixel, edge_meter = self._detect_wall_edge_anchors(
                    frame, lane, num_needed=3 - num_matched
                )
                if edge_pixel:
                    pixel_points.extend(edge_pixel)
                    meter_points.extend(edge_meter)
                    matched_nums.extend([-1] * len(edge_pixel))  # -1 indicates edge anchor
                    edge_anchors_count = len(edge_pixel)
                    logger.debug(f"Added {edge_anchors_count} edge anchors")

            # Affine Transform (6 DOF)
            if len(pixel_points) >= 3:
                result = self._calibrate_affine(
                    pixel_points, meter_points, matched_nums, detected_holds, frame.shape
                )
                if result:
                    result.method = CalibrationMethod.AFFINE
                    result.edge_anchors_count = edge_anchors_count
                    return result

            # Similarity Transform (4 DOF) for exactly 2 points
            if len(pixel_points) >= 2:
                result = self._calibrate_similarity(
                    pixel_points, meter_points, matched_nums, detected_holds, frame.shape
                )
                if result:
                    result.method = CalibrationMethod.SIMILARITY
                    result.edge_anchors_count = edge_anchors_count
                    return result

        logger.debug(f"Calibration failed: {num_matched} holds matched, need at least {self.min_holds_affine}")
        return None

    def _calibrate_homography(
        self,
        pixel_points: List,
        meter_points: List,
        matched_nums: List,
        detected_holds: List,
        frame_shape: Tuple
    ) -> Optional[CalibrationResult]:
        """Calibrate using full homography (8 DOF, needs ≥4 points)."""
        pixel_arr = np.array(pixel_points, dtype=np.float32)
        meter_arr = np.array(meter_points, dtype=np.float32)

        # Compute homography with RANSAC
        homography, inliers = self._compute_homography_ransac(pixel_arr, meter_arr)

        if homography is None:
            logger.debug("Homography computation failed")
            return None

        inlier_count = int(np.sum(inliers))
        inlier_ratio = inlier_count / len(pixel_arr)

        if inlier_ratio < self.min_inlier_ratio:
            logger.debug(f"Low inlier ratio: {inlier_ratio:.2f} < {self.min_inlier_ratio}")
            return None

        # Calculate RMSE for inliers
        rmse = self._calculate_rmse(
            pixel_arr[inliers == 1],
            meter_arr[inliers == 1],
            homography
        )

        # Calculate pixel-to-meter scale
        scale = self._calculate_scale(pixel_arr, meter_arr, homography, frame_shape)

        # Calculate confidence score
        confidence = self._calculate_confidence(inlier_ratio, rmse, inlier_count)

        # Create transformation functions
        pixel_to_meter_func = lambda px, py: self._transform_point(px, py, homography)

        try:
            inv_homography = np.linalg.inv(homography)
            meter_to_pixel_func = lambda mx, my: self._transform_point(mx, my, inv_homography)
        except np.linalg.LinAlgError:
            meter_to_pixel_func = None

        logger.info(f"Homography calibration: {inlier_count} inliers, "
                   f"RMSE={rmse:.4f}m, confidence={confidence:.2f}")

        return CalibrationResult(
            homography_matrix=homography,
            pixel_to_meter_scale=scale,
            rmse_error=rmse,
            inlier_count=inlier_count,
            total_holds=len(detected_holds),
            inlier_ratio=inlier_ratio,
            confidence=confidence,
            matched_hold_nums=[matched_nums[i] for i, v in enumerate(inliers) if v == 1],
            pixel_to_meter_func=pixel_to_meter_func,
            meter_to_pixel_func=meter_to_pixel_func,
            method=CalibrationMethod.HOMOGRAPHY,
            anchor_points_used=len(pixel_arr)
        )

    def _calibrate_affine(
        self,
        pixel_points: List,
        meter_points: List,
        matched_nums: List,
        detected_holds: List,
        frame_shape: Tuple
    ) -> Optional[CalibrationResult]:
        """
        Calibrate using affine transform (6 DOF, needs ≥3 points).

        Affine transform preserves:
        - Parallelism
        - Ratios of distances along a line

        Parameters: scale_x, scale_y, rotation, shear, translate_x, translate_y
        """
        pixel_arr = np.array(pixel_points, dtype=np.float32)
        meter_arr = np.array(meter_points, dtype=np.float32)

        if len(pixel_arr) < 3:
            return None

        try:
            # Compute affine transform
            # cv2.getAffineTransform needs exactly 3 points
            # For more points, use estimateAffine2D with RANSAC
            if len(pixel_arr) == 3:
                affine_2x3 = cv2.getAffineTransform(pixel_arr, meter_arr)
            else:
                affine_2x3, inliers = cv2.estimateAffine2D(
                    pixel_arr, meter_arr,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_threshold * 1.5,  # Slightly more tolerant
                    maxIters=2000,
                    confidence=0.99
                )

            if affine_2x3 is None:
                logger.debug("Affine transform computation failed")
                return None

            # Convert 2x3 affine matrix to 3x3 homography-like matrix
            homography = np.vstack([affine_2x3, [0, 0, 1]])

            # Calculate RMSE
            rmse = self._calculate_rmse(pixel_arr, meter_arr, homography)

            # Calculate scale
            scale = self._calculate_scale(pixel_arr, meter_arr, homography, frame_shape)

            # Affine confidence is lower than homography (penalty for fewer DOF)
            base_confidence = self._calculate_confidence(0.9, rmse, len(pixel_arr))
            confidence = base_confidence * 0.85  # 15% penalty for using affine

            # Create transformation functions
            pixel_to_meter_func = lambda px, py: self._transform_point(px, py, homography)

            try:
                inv_homography = np.linalg.inv(homography)
                meter_to_pixel_func = lambda mx, my: self._transform_point(mx, my, inv_homography)
            except np.linalg.LinAlgError:
                meter_to_pixel_func = None

            logger.info(f"Affine calibration: {len(pixel_arr)} points, "
                       f"RMSE={rmse:.4f}m, confidence={confidence:.2f}")

            return CalibrationResult(
                homography_matrix=homography,
                pixel_to_meter_scale=scale,
                rmse_error=rmse,
                inlier_count=len(pixel_arr),
                total_holds=len(detected_holds),
                inlier_ratio=1.0,  # All points used in affine
                confidence=confidence,
                matched_hold_nums=[n for n in matched_nums if n != -1],
                pixel_to_meter_func=pixel_to_meter_func,
                meter_to_pixel_func=meter_to_pixel_func,
                method=CalibrationMethod.AFFINE,
                anchor_points_used=len(pixel_arr)
            )

        except cv2.error as e:
            logger.debug(f"Affine transform error: {e}")
            return None

    def _calibrate_similarity(
        self,
        pixel_points: List,
        meter_points: List,
        matched_nums: List,
        detected_holds: List,
        frame_shape: Tuple
    ) -> Optional[CalibrationResult]:
        """
        Calibrate using similarity transform (4 DOF, needs ≥2 points).

        Uses wall constraints (known dimensions) to compensate for missing DOF.
        Similarity transform preserves:
        - Angles
        - Ratios of distances

        Parameters: scale, rotation, translate_x, translate_y
        """
        pixel_arr = np.array(pixel_points, dtype=np.float32)
        meter_arr = np.array(meter_points, dtype=np.float32)

        if len(pixel_arr) < 2:
            return None

        try:
            # For 2 points, compute similarity transform using known wall constraints
            # Calculate the transform that maps pixel points to meter points

            # Get vectors
            pixel_vec = pixel_arr[1] - pixel_arr[0]
            meter_vec = meter_arr[1] - meter_arr[0]

            # Calculate scale from point distances
            pixel_dist = np.linalg.norm(pixel_vec)
            meter_dist = np.linalg.norm(meter_vec)

            if pixel_dist < 1e-6:
                logger.debug("Points too close for similarity transform")
                return None

            scale = meter_dist / pixel_dist

            # Calculate rotation angle
            pixel_angle = np.arctan2(pixel_vec[1], pixel_vec[0])
            meter_angle = np.arctan2(meter_vec[1], meter_vec[0])
            rotation = meter_angle - pixel_angle

            # Build similarity matrix (2x3)
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)

            # Rotation and scale matrix
            R = scale * np.array([[cos_r, -sin_r], [sin_r, cos_r]])

            # Translation: t = meter_point - R @ pixel_point
            translation = meter_arr[0] - R @ pixel_arr[0]

            # Build 3x3 homography-like matrix
            homography = np.array([
                [R[0, 0], R[0, 1], translation[0]],
                [R[1, 0], R[1, 1], translation[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            # Calculate RMSE
            rmse = self._calculate_rmse(pixel_arr, meter_arr, homography)

            # Similarity confidence is lower than affine (more penalty)
            base_confidence = self._calculate_confidence(0.8, rmse, len(pixel_arr))
            confidence = base_confidence * 0.70  # 30% penalty for using similarity

            # Create transformation functions
            pixel_to_meter_func = lambda px, py: self._transform_point(px, py, homography)

            try:
                inv_homography = np.linalg.inv(homography)
                meter_to_pixel_func = lambda mx, my: self._transform_point(mx, my, inv_homography)
            except np.linalg.LinAlgError:
                meter_to_pixel_func = None

            logger.info(f"Similarity calibration: {len(pixel_arr)} points, "
                       f"RMSE={rmse:.4f}m, confidence={confidence:.2f}, scale={scale:.6f}")

            return CalibrationResult(
                homography_matrix=homography,
                pixel_to_meter_scale=scale,
                rmse_error=rmse,
                inlier_count=len(pixel_arr),
                total_holds=len(detected_holds),
                inlier_ratio=1.0,
                confidence=confidence,
                matched_hold_nums=[n for n in matched_nums if n != -1],
                pixel_to_meter_func=pixel_to_meter_func,
                meter_to_pixel_func=meter_to_pixel_func,
                method=CalibrationMethod.SIMILARITY,
                anchor_points_used=len(pixel_arr)
            )

        except Exception as e:
            logger.debug(f"Similarity transform error: {e}")
            return None

    def _detect_wall_edge_anchors(
        self,
        frame: np.ndarray,
        lane: str,
        num_needed: int = 1
    ) -> Tuple[List, List]:
        """
        Detect wall edge points to use as anchor points for calibration.

        Uses edge detection to find vertical wall boundaries and horizontal
        features (start pad, top edge) that can serve as known reference points.

        Args:
            frame: Video frame (BGR)
            lane: 'left' or 'right'
            num_needed: Number of anchor points to find

        Returns:
            Tuple of (pixel_points, meter_points) lists
        """
        pixel_anchors = []
        meter_anchors = []

        if num_needed <= 0:
            return pixel_anchors, meter_anchors

        frame_height, frame_width = frame.shape[:2]

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Apply morphological operations to clean up edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=frame_height // 4,
            maxLineGap=20
        )

        if lines is None:
            logger.debug("No wall edges detected")
            return pixel_anchors, meter_anchors

        # Define lane boundaries
        mid_x = frame_width / 2
        if lane == 'left':
            lane_start = 0
            lane_end = mid_x
        else:
            lane_start = mid_x
            lane_end = frame_width

        # Find vertical edges (wall boundaries)
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check if line is in the correct lane
            line_mid_x = (x1 + x2) / 2
            if not (lane_start <= line_mid_x <= lane_end):
                continue

            # Calculate line angle
            dx = x2 - x1
            dy = y2 - y1
            angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)

            line_length = np.sqrt(dx**2 + dy**2)

            # Vertical lines (wall sides): angle close to 90°
            if 80 <= angle <= 100:
                vertical_lines.append((line[0], line_length, x1))

            # Horizontal lines (start pad, top): angle close to 0° or 180°
            if angle <= 10 or angle >= 170:
                horizontal_lines.append((line[0], line_length, y1))

        # Sort by length (longest = most likely wall boundary)
        vertical_lines.sort(key=lambda x: x[1], reverse=True)
        horizontal_lines.sort(key=lambda x: x[1], reverse=True)

        anchors_found = 0

        # Add bottom horizontal line (start pad) - known position: y = 0m
        if horizontal_lines and anchors_found < num_needed:
            # Find the horizontal line closest to the bottom
            bottom_lines = [(l, length, y) for l, length, y in horizontal_lines if y > frame_height * 0.7]
            if bottom_lines:
                bottom_lines.sort(key=lambda x: x[2], reverse=True)
                line = bottom_lines[0][0]
                x1, y1, x2, y2 = line

                # Anchor point: center of the bottom line
                pixel_x = (x1 + x2) / 2
                pixel_y = (y1 + y2) / 2

                # Map to world coordinates
                # Normalize x within lane (0 to lane_width)
                if lane == 'left':
                    norm_x = pixel_x / mid_x
                else:
                    norm_x = (pixel_x - mid_x) / mid_x

                meter_x = norm_x * self.lane_width
                meter_y = 0.0  # Start pad is at y = 0

                pixel_anchors.append([pixel_x, pixel_y])
                meter_anchors.append([meter_x, meter_y])
                anchors_found += 1
                logger.debug(f"Added start pad anchor: pixel=({pixel_x:.0f}, {pixel_y:.0f}), meter=({meter_x:.2f}, {meter_y:.2f})")

        # Add top horizontal line if visible - known position: y = wall_height
        if horizontal_lines and anchors_found < num_needed:
            top_lines = [(l, length, y) for l, length, y in horizontal_lines if y < frame_height * 0.3]
            if top_lines:
                top_lines.sort(key=lambda x: x[2])  # Closest to top
                line = top_lines[0][0]
                x1, y1, x2, y2 = line

                pixel_x = (x1 + x2) / 2
                pixel_y = (y1 + y2) / 2

                if lane == 'left':
                    norm_x = pixel_x / mid_x
                else:
                    norm_x = (pixel_x - mid_x) / mid_x

                meter_x = norm_x * self.lane_width
                meter_y = self.wall_height

                pixel_anchors.append([pixel_x, pixel_y])
                meter_anchors.append([meter_x, meter_y])
                anchors_found += 1
                logger.debug(f"Added top edge anchor: pixel=({pixel_x:.0f}, {pixel_y:.0f}), meter=({meter_x:.2f}, {meter_y:.2f})")

        # Add vertical wall edge (left or right boundary of lane)
        if vertical_lines and anchors_found < num_needed:
            if lane == 'left':
                # Use leftmost vertical line as left wall boundary (x = 0)
                vertical_lines.sort(key=lambda x: x[2])  # Sort by x position
                target_meter_x = 0.0
            else:
                # Use rightmost vertical line as right wall boundary (x = lane_width)
                vertical_lines.sort(key=lambda x: x[2], reverse=True)
                target_meter_x = self.lane_width

            if vertical_lines:
                line = vertical_lines[0][0]
                x1, y1, x2, y2 = line

                # Use a point at mid-height of the line
                pixel_x = (x1 + x2) / 2
                pixel_y = (y1 + y2) / 2

                # Estimate meter_y based on normalized y position
                norm_y = 1.0 - (pixel_y / frame_height)
                meter_y = norm_y * self.wall_height

                pixel_anchors.append([pixel_x, pixel_y])
                meter_anchors.append([target_meter_x, meter_y])
                anchors_found += 1
                logger.debug(f"Added vertical edge anchor: pixel=({pixel_x:.0f}, {pixel_y:.0f}), meter=({target_meter_x:.2f}, {meter_y:.2f})")

        logger.debug(f"Detected {len(pixel_anchors)} wall edge anchors for lane {lane}")
        return pixel_anchors, meter_anchors

    def _match_holds_to_route(
        self,
        detected_holds: List,
        route_holds: List[dict],
        frame_shape: Tuple[int, int, int],
        lane: str
    ) -> Tuple[List, List, List]:
        """
        Match detected holds to route map using normalized coordinates.

        Uses a greedy best-match algorithm that prevents duplicate matches.
        """
        frame_height, frame_width = frame_shape[:2]

        # Calculate lane boundaries for detected holds
        # Left lane: x < 0.5 of frame width
        # Right lane: x > 0.5 of frame width
        mid_x = frame_width / 2

        # Filter detected holds by lane
        if lane == 'left':
            lane_holds = [h for h in detected_holds if h.pixel_x < mid_x]
            # Normalize x to 0-1 within the left half
            def normalize_x(px):
                return px / mid_x
        else:
            lane_holds = [h for h in detected_holds if h.pixel_x >= mid_x]
            # Normalize x to 0-1 within the right half
            def normalize_x(px):
                return (px - mid_x) / mid_x

        pixel_points = []
        meter_points = []
        matched_holds = []
        used_route_indices = set()

        # Sort detected holds by confidence (best first)
        sorted_detected = sorted(lane_holds, key=lambda h: h.confidence, reverse=True)

        for detected in sorted_detected:
            # Normalize pixel coordinates
            norm_x = normalize_x(detected.pixel_x)
            norm_y = 1.0 - (detected.pixel_y / frame_height)  # Y=0 at bottom

            best_match_idx = None
            best_distance = float('inf')

            for idx, route_hold in enumerate(route_holds):
                if idx in used_route_indices:
                    continue

                # Normalize route coordinates
                route_norm_x = route_hold['wall_x_m'] / self.lane_width
                route_norm_y = route_hold['wall_y_m'] / self.wall_height

                # Calculate Euclidean distance in normalized space
                # Weight Y more heavily since height variance is more important
                dist = np.sqrt(
                    (norm_x - route_norm_x) ** 2 +
                    2.0 * (norm_y - route_norm_y) ** 2  # 2x weight on Y
                )

                if dist < best_distance:
                    best_distance = dist
                    best_match_idx = idx

            # Accept match if within threshold
            if best_distance < self.max_match_distance and best_match_idx is not None:
                route_hold = route_holds[best_match_idx]
                pixel_points.append([detected.pixel_x, detected.pixel_y])
                meter_points.append([route_hold['wall_x_m'], route_hold['wall_y_m']])
                matched_holds.append(route_hold['hold_num'])
                used_route_indices.add(best_match_idx)

                logger.debug(f"Matched hold at ({detected.pixel_x:.0f}, {detected.pixel_y:.0f}) "
                           f"to #{route_hold['hold_num']} (dist={best_distance:.3f})")

        logger.debug(f"Matched {len(pixel_points)} of {len(sorted_detected)} detected holds")
        return pixel_points, meter_points, matched_holds

    def _compute_homography_ransac(
        self,
        pixel_points: np.ndarray,
        meter_points: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute homography matrix using RANSAC."""
        if len(pixel_points) < 4:
            return None, None

        try:
            homography, inliers = cv2.findHomography(
                pixel_points,
                meter_points,
                cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=3000,
                confidence=0.995
            )

            if homography is None or inliers is None:
                return None, None

            return homography, inliers.ravel()

        except cv2.error as e:
            logger.debug(f"Homography computation error: {e}")
            return None, None

    def _calculate_rmse(
        self,
        pixel_points: np.ndarray,
        meter_points: np.ndarray,
        homography: np.ndarray
    ) -> float:
        """Calculate RMSE between transformed points and target points."""
        if len(pixel_points) == 0:
            return float('inf')

        # Transform pixel points to meter coordinates
        ones = np.ones((len(pixel_points), 1))
        pixel_homogeneous = np.hstack([pixel_points, ones])
        transformed = (homography @ pixel_homogeneous.T).T

        # Normalize by homogeneous coordinate
        transformed_meters = transformed[:, :2] / transformed[:, 2:3]

        # Calculate errors
        errors = np.linalg.norm(transformed_meters - meter_points, axis=1)
        return float(np.sqrt(np.mean(errors ** 2)))

    def _calculate_scale(
        self,
        pixel_points: np.ndarray,
        meter_points: np.ndarray,
        homography: np.ndarray,
        frame_shape: Tuple[int, ...]
    ) -> float:
        """Calculate approximate pixel-to-meter scale."""
        if len(pixel_points) < 2:
            return 0.0

        try:
            # Sample two points far apart in pixel space
            pixel_dists = np.linalg.norm(
                pixel_points[:, np.newaxis] - pixel_points[np.newaxis, :],
                axis=2
            )
            np.fill_diagonal(pixel_dists, 0)

            # Find pair with maximum pixel distance
            idx = np.unravel_index(np.argmax(pixel_dists), pixel_dists.shape)
            pixel_dist = pixel_dists[idx]

            if pixel_dist == 0:
                return 0.0

            # Calculate corresponding meter distance
            meter_dist = np.linalg.norm(meter_points[idx[0]] - meter_points[idx[1]])

            return meter_dist / pixel_dist

        except Exception as e:
            logger.debug(f"Scale calculation error: {e}")
            return 0.0

    def _calculate_confidence(
        self,
        inlier_ratio: float,
        rmse: float,
        inlier_count: int
    ) -> float:
        """
        Calculate overall confidence score for calibration.

        Considers:
        - Inlier ratio (higher is better)
        - RMSE error (lower is better)
        - Number of inliers (more is better)
        """
        # Inlier ratio contribution (0-1)
        ratio_score = min(inlier_ratio * 1.2, 1.0)

        # RMSE contribution (0-1), penalize high error
        # RMSE > 0.3m is very poor, < 0.05m is excellent
        rmse_score = max(0, 1.0 - (rmse / 0.3))

        # Count contribution (0-1)
        # 4 inliers is minimum, 10+ is excellent
        count_score = min((inlier_count - 3) / 7, 1.0)

        # Weighted combination
        confidence = (
            ratio_score * 0.35 +
            rmse_score * 0.40 +
            count_score * 0.25
        )

        return min(max(confidence, 0.0), 1.0)

    def _transform_point(
        self,
        x: float,
        y: float,
        homography: np.ndarray
    ) -> Tuple[float, float]:
        """Transform a point using homography matrix."""
        point = np.array([x, y, 1.0])
        transformed = homography @ point
        return (
            transformed[0] / transformed[2],
            transformed[1] / transformed[2]
        )


class PeriodicCalibrator(CameraCalibrator):
    """
    Periodic calibrator with intelligent caching and temporal smoothing.

    Features:
    - Caches good calibrations and reuses them when recalibration fails
    - Applies confidence decay to cached calibrations over time
    - Uses hybrid calibration strategy (homography → affine → similarity → cache)
    - Tracks calibration quality metrics over time
    """

    # Confidence decay rate per frame (assuming static camera, decay is slow)
    CONFIDENCE_DECAY_PER_FRAME = 0.001  # 0.1% decay per frame
    # Minimum confidence before cache becomes unusable
    MIN_CACHE_CONFIDENCE = 0.30

    def __init__(
        self,
        route_coordinates_path: str,
        recalibration_interval: int = DEFAULT_PROCESSING_CONFIG["calibration_interval_frames"],
        min_holds_for_calibration: int = 4,
        min_holds_for_affine: int = 2,
        ransac_threshold: float = 0.08,
        min_inlier_ratio: float = 0.4,
        min_confidence_for_cache: float = DEFAULT_PROCESSING_CONFIG["min_calibration_confidence"],
        max_frames_without_recalibration: int = 300,  # Increased: static camera assumption
        enable_affine_fallback: bool = True,
        enable_edge_anchors: bool = True,
        confidence_decay_rate: float = 0.001,  # Decay per frame
    ):
        super().__init__(
            route_coordinates_path,
            min_holds_for_calibration,
            min_holds_for_affine,
            ransac_threshold,
            min_inlier_ratio,
            enable_affine_fallback=enable_affine_fallback,
            enable_edge_anchors=enable_edge_anchors,
        )
        self.recalibration_interval = recalibration_interval
        self.min_confidence_for_cache = min_confidence_for_cache
        self.max_frames_without_recalibration = max_frames_without_recalibration
        self.confidence_decay_rate = confidence_decay_rate

        # Calibration storage
        self.calibration_cache: Dict[str, CalibrationResult] = {}  # Historical cache by lane_frameId
        self.last_calibration: Dict[str, CalibrationResult] = {}  # Best recent calibration per lane
        self.frames_since_calibration: Dict[str, int] = {'left': 0, 'right': 0}

        # Cache quality tracking
        self.cache_original_confidence: Dict[str, float] = {'left': 0.0, 'right': 0.0}
        self.cache_frame_id: Dict[str, int] = {'left': -1, 'right': -1}

        # Statistics
        self.stats = {
            'total_calibrations': {'left': 0, 'right': 0},
            'cache_hits': {'left': 0, 'right': 0},
            'homography_count': {'left': 0, 'right': 0},
            'affine_count': {'left': 0, 'right': 0},
            'similarity_count': {'left': 0, 'right': 0},
            'cache_fallback_count': {'left': 0, 'right': 0},
        }

    def calibrate_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        detected_holds: List,
        lane: str = 'left',
        force_recalibration: bool = False
    ) -> Optional[CalibrationResult]:
        """
        Calibrate a frame with intelligent caching and fallback.

        Calibration Strategy (in order of preference):
        1. Try new calibration if due
        2. If new calibration succeeds with good confidence → cache it
        3. If new calibration fails → use cached with confidence decay
        4. If no cache available → return None

        Args:
            frame: Video frame
            frame_id: Frame number
            detected_holds: List of detected holds
            lane: 'left' or 'right'
            force_recalibration: Force new calibration regardless of cache

        Returns:
            CalibrationResult with is_cached and cache_age_frames set appropriately
        """
        frames_since = self.frames_since_calibration.get(lane, 0)

        # Determine if we should attempt recalibration
        should_recalibrate = (
            force_recalibration or
            frame_id % self.recalibration_interval == 0 or
            lane not in self.last_calibration or
            frames_since >= self.max_frames_without_recalibration
        )

        if should_recalibrate:
            # Attempt new calibration
            calibration = self.calibrate(frame, detected_holds, lane)

            if calibration and calibration.confidence >= self.min_confidence_for_cache:
                # Good calibration - cache it
                self._cache_calibration(calibration, lane, frame_id)
                self.stats['total_calibrations'][lane] += 1
                self._update_method_stats(calibration.method, lane)
                return calibration

            elif calibration:
                # Calibration succeeded but below cache threshold
                # Compare with cached: use better one
                cached = self._get_cached_calibration(lane, frame_id)

                if cached and cached.confidence > calibration.confidence:
                    # Cached is still better
                    self.stats['cache_hits'][lane] += 1
                    return cached
                else:
                    # New calibration is better (or no cache)
                    self.frames_since_calibration[lane] += 1
                    self.stats['total_calibrations'][lane] += 1
                    self._update_method_stats(calibration.method, lane)
                    return calibration

            else:
                # Calibration failed - use cached if available
                cached = self._get_cached_calibration(lane, frame_id)
                if cached:
                    self.stats['cache_hits'][lane] += 1
                    self.stats['cache_fallback_count'][lane] += 1
                    return cached
                return None

        else:
            # Not recalibrating - use cached
            cached = self._get_cached_calibration(lane, frame_id)
            if cached:
                self.stats['cache_hits'][lane] += 1
                return cached
            return None

    def _cache_calibration(
        self,
        calibration: CalibrationResult,
        lane: str,
        frame_id: int
    ):
        """Store calibration in cache with metadata."""
        cache_key = f"{lane}_{frame_id}"
        self.calibration_cache[cache_key] = calibration
        self.last_calibration[lane] = calibration
        self.cache_original_confidence[lane] = calibration.confidence
        self.cache_frame_id[lane] = frame_id
        self.frames_since_calibration[lane] = 0

        logger.debug(f"Cached {calibration.method.value} calibration for {lane} lane "
                    f"(confidence={calibration.confidence:.3f})")

    def _get_cached_calibration(
        self,
        lane: str,
        current_frame_id: int
    ) -> Optional[CalibrationResult]:
        """
        Get cached calibration with confidence decay applied.

        The confidence decays over time based on how many frames have passed
        since the calibration was cached. This reflects uncertainty about
        whether the camera has moved.

        Returns:
            CalibrationResult with updated confidence and cache metadata, or None
        """
        if lane not in self.last_calibration:
            return None

        cached = self.last_calibration[lane]
        cache_frame = self.cache_frame_id.get(lane, 0)
        original_confidence = self.cache_original_confidence.get(lane, cached.confidence)

        # Calculate cache age
        cache_age = current_frame_id - cache_frame
        self.frames_since_calibration[lane] = cache_age

        # Apply confidence decay
        decay_factor = 1.0 - (cache_age * self.confidence_decay_rate)
        decay_factor = max(decay_factor, 0.5)  # Don't decay below 50% of original

        decayed_confidence = original_confidence * decay_factor

        # Check if cache is still usable
        if decayed_confidence < self.MIN_CACHE_CONFIDENCE:
            logger.debug(f"Cache confidence too low ({decayed_confidence:.3f}), invalidating")
            return None

        # Create a copy with updated metadata
        # We need to create a new result since dataclass is immutable in practice
        cached_result = CalibrationResult(
            homography_matrix=cached.homography_matrix,
            pixel_to_meter_scale=cached.pixel_to_meter_scale,
            rmse_error=cached.rmse_error,
            inlier_count=cached.inlier_count,
            total_holds=cached.total_holds,
            inlier_ratio=cached.inlier_ratio,
            confidence=decayed_confidence,
            matched_hold_nums=cached.matched_hold_nums,
            pixel_to_meter_func=cached.pixel_to_meter_func,
            meter_to_pixel_func=cached.meter_to_pixel_func,
            method=CalibrationMethod.TEMPORAL_CACHE,
            is_cached=True,
            cache_age_frames=cache_age,
            anchor_points_used=cached.anchor_points_used,
            edge_anchors_count=cached.edge_anchors_count,
        )

        logger.debug(f"Using cached calibration for {lane} (age={cache_age} frames, "
                    f"confidence: {original_confidence:.3f} → {decayed_confidence:.3f})")

        return cached_result

    def _update_method_stats(self, method: CalibrationMethod, lane: str):
        """Update statistics for calibration method usage."""
        if method == CalibrationMethod.HOMOGRAPHY:
            self.stats['homography_count'][lane] += 1
        elif method == CalibrationMethod.AFFINE:
            self.stats['affine_count'][lane] += 1
        elif method == CalibrationMethod.SIMILARITY:
            self.stats['similarity_count'][lane] += 1

    def reset(self, lane: Optional[str] = None):
        """Reset calibration cache."""
        if lane:
            self.last_calibration.pop(lane, None)
            self.frames_since_calibration[lane] = 0
            self.cache_original_confidence[lane] = 0.0
            self.cache_frame_id[lane] = -1
        else:
            self.calibration_cache.clear()
            self.last_calibration.clear()
            self.frames_since_calibration = {'left': 0, 'right': 0}
            self.cache_original_confidence = {'left': 0.0, 'right': 0.0}
            self.cache_frame_id = {'left': -1, 'right': -1}

    def get_cache_stats(self) -> Dict:
        """Get comprehensive statistics about calibration cache and usage."""
        left_confidence = 0.0
        right_confidence = 0.0

        if 'left' in self.last_calibration:
            left_confidence = self.last_calibration['left'].confidence
        if 'right' in self.last_calibration:
            right_confidence = self.last_calibration['right'].confidence

        return {
            'cached_calibrations': len(self.calibration_cache),
            'frames_since_calibration': dict(self.frames_since_calibration),
            'has_left_calibration': 'left' in self.last_calibration,
            'has_right_calibration': 'right' in self.last_calibration,
            'left_confidence': left_confidence,
            'right_confidence': right_confidence,
            'left_method': self.last_calibration.get('left', CalibrationResult(
                np.eye(3), 0, 0, 0, 0, 0, 0, method=CalibrationMethod.FAILED
            )).method.value if 'left' in self.last_calibration else 'none',
            'right_method': self.last_calibration.get('right', CalibrationResult(
                np.eye(3), 0, 0, 0, 0, 0, 0, method=CalibrationMethod.FAILED
            )).method.value if 'right' in self.last_calibration else 'none',
            'statistics': dict(self.stats),
        }

    def get_calibration_summary(self) -> str:
        """Get a human-readable summary of calibration status."""
        stats = self.get_cache_stats()
        lines = [
            "=== Calibration Summary ===",
            f"Left Lane: {stats['left_method']} (confidence: {stats['left_confidence']:.2f})",
            f"Right Lane: {stats['right_method']} (confidence: {stats['right_confidence']:.2f})",
            f"Frames since recalibration: L={stats['frames_since_calibration']['left']}, "
            f"R={stats['frames_since_calibration']['right']}",
            "",
            "Method Usage:",
            f"  Homography: L={self.stats['homography_count']['left']}, "
            f"R={self.stats['homography_count']['right']}",
            f"  Affine: L={self.stats['affine_count']['left']}, "
            f"R={self.stats['affine_count']['right']}",
            f"  Similarity: L={self.stats['similarity_count']['left']}, "
            f"R={self.stats['similarity_count']['right']}",
            f"  Cache Fallbacks: L={self.stats['cache_fallback_count']['left']}, "
            f"R={self.stats['cache_fallback_count']['right']}",
        ]
        return "\n".join(lines)
