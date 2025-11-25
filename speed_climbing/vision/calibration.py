"""
Camera calibration using homography transformation.

This module handles the transformation between pixel coordinates (from video frames)
and world coordinates (meters on the climbing wall) using hold detection as reference points.
"""

import json
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
import cv2
import numpy as np

from speed_climbing.core.settings import DEFAULT_PROCESSING_CONFIG, IFSC_STANDARDS

logger = logging.getLogger(__name__)


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
        ransac_threshold: float = 0.08,  # Increased for better tolerance
        min_inlier_ratio: float = 0.4,   # Reduced for partial visibility
        max_match_distance: float = 0.25  # Max normalized distance for matching
    ):
        self.min_holds = min_holds_for_calibration
        self.ransac_threshold = ransac_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.max_match_distance = max_match_distance
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
        Calibrate camera for a single frame.

        Args:
            frame: Video frame (BGR)
            detected_holds: List of DetectedHold objects
            lane: 'left' (SN panels) or 'right' (DX panels)

        Returns:
            CalibrationResult if successful, None otherwise
        """
        if not self.route_map or len(detected_holds) < self.min_holds:
            logger.debug(f"Insufficient holds for calibration: {len(detected_holds)} < {self.min_holds}")
            return None

        # Filter route holds by lane
        lane_prefix = 'SN' if lane == 'left' else 'DX'
        route_holds = [
            h for h in self.route_map['holds']
            if h.get('panel', '').startswith(lane_prefix)
        ]

        if len(route_holds) < self.min_holds:
            logger.debug(f"Insufficient route holds for lane {lane}")
            return None

        # Match detected holds to route map
        pixel_points, meter_points, matched_nums = self._match_holds_to_route(
            detected_holds, route_holds, frame.shape, lane
        )

        if len(pixel_points) < self.min_holds:
            logger.debug(f"Only {len(pixel_points)} holds matched, need {self.min_holds}")
            return None

        pixel_points = np.array(pixel_points, dtype=np.float32)
        meter_points = np.array(meter_points, dtype=np.float32)

        # Compute homography with RANSAC
        homography, inliers = self._compute_homography_ransac(pixel_points, meter_points)

        if homography is None:
            logger.debug("Homography computation failed")
            return None

        inlier_count = int(np.sum(inliers))
        inlier_ratio = inlier_count / len(pixel_points)

        if inlier_ratio < self.min_inlier_ratio:
            logger.debug(f"Low inlier ratio: {inlier_ratio:.2f} < {self.min_inlier_ratio}")
            return None

        # Calculate RMSE for inliers
        rmse = self._calculate_rmse(
            pixel_points[inliers == 1],
            meter_points[inliers == 1],
            homography
        )

        # Calculate pixel-to-meter scale
        scale = self._calculate_scale(pixel_points, meter_points, homography, frame.shape)

        # Calculate confidence score
        confidence = self._calculate_confidence(inlier_ratio, rmse, inlier_count)

        # Create transformation functions
        pixel_to_meter_func = lambda px, py: self._transform_point(px, py, homography)

        try:
            inv_homography = np.linalg.inv(homography)
            meter_to_pixel_func = lambda mx, my: self._transform_point(mx, my, inv_homography)
        except np.linalg.LinAlgError:
            meter_to_pixel_func = None

        logger.info(f"Calibration successful: {inlier_count} inliers, "
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
            meter_to_pixel_func=meter_to_pixel_func
        )

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
    Periodic calibrator that caches calibration results.

    Useful for video processing where recalibrating every frame is expensive.
    Caches good calibrations and reuses them until a recalibration is triggered.
    """

    def __init__(
        self,
        route_coordinates_path: str,
        recalibration_interval: int = DEFAULT_PROCESSING_CONFIG["calibration_interval_frames"],
        min_holds_for_calibration: int = 4,
        ransac_threshold: float = 0.08,
        min_inlier_ratio: float = 0.4,
        min_confidence_for_cache: float = DEFAULT_PROCESSING_CONFIG["min_calibration_confidence"],
        max_frames_without_recalibration: int = 150  # Force recalibration after this many frames
    ):
        super().__init__(
            route_coordinates_path,
            min_holds_for_calibration,
            ransac_threshold,
            min_inlier_ratio
        )
        self.recalibration_interval = recalibration_interval
        self.min_confidence_for_cache = min_confidence_for_cache
        self.max_frames_without_recalibration = max_frames_without_recalibration

        self.calibration_cache: Dict[str, CalibrationResult] = {}  # Keyed by lane
        self.last_calibration: Dict[str, CalibrationResult] = {}
        self.frames_since_calibration: Dict[str, int] = {'left': 0, 'right': 0}

    def calibrate_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        detected_holds: List,
        lane: str = 'left',
        force_recalibration: bool = False
    ) -> Optional[CalibrationResult]:
        """
        Calibrate a frame with caching.

        Args:
            frame: Video frame
            frame_id: Frame number
            detected_holds: List of detected holds
            lane: 'left' or 'right'
            force_recalibration: Force new calibration regardless of cache

        Returns:
            CalibrationResult (potentially cached)
        """
        frames_since = self.frames_since_calibration.get(lane, 0)

        should_recalibrate = (
            force_recalibration or
            frame_id % self.recalibration_interval == 0 or
            lane not in self.last_calibration or
            frames_since >= self.max_frames_without_recalibration
        )

        if should_recalibrate:
            calibration = self.calibrate(frame, detected_holds, lane)

            if calibration and calibration.confidence >= self.min_confidence_for_cache:
                # Good calibration - cache it
                cache_key = f"{lane}_{frame_id}"
                self.calibration_cache[cache_key] = calibration
                self.last_calibration[lane] = calibration
                self.frames_since_calibration[lane] = 0
                return calibration

            elif calibration:
                # Calibration succeeded but below cache threshold
                # Still use it but don't cache
                self.frames_since_calibration[lane] += 1
                return calibration

            else:
                # Calibration failed - use cached if available
                self.frames_since_calibration[lane] += 1
                return self.last_calibration.get(lane)

        else:
            # Not recalibrating - use cached
            self.frames_since_calibration[lane] += 1
            return self.last_calibration.get(lane)

    def reset(self, lane: Optional[str] = None):
        """Reset calibration cache."""
        if lane:
            self.last_calibration.pop(lane, None)
            self.frames_since_calibration[lane] = 0
        else:
            self.calibration_cache.clear()
            self.last_calibration.clear()
            self.frames_since_calibration = {'left': 0, 'right': 0}

    def get_cache_stats(self) -> Dict:
        """Get statistics about the calibration cache."""
        return {
            'cached_calibrations': len(self.calibration_cache),
            'frames_since_calibration': dict(self.frames_since_calibration),
            'has_left_calibration': 'left' in self.last_calibration,
            'has_right_calibration': 'right' in self.last_calibration,
        }
