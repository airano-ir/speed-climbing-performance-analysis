"""
Camera calibration using homography transformation.
"""

import json
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
import cv2
import numpy as np

from speed_climbing.core.settings import DEFAULT_PROCESSING_CONFIG

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
    pixel_to_meter_func: Optional[Callable[[float, float], Tuple[float, float]]] = None
    meter_to_pixel_func: Optional[Callable[[float, float], Tuple[float, float]]] = None


class CameraCalibrator:
    """Calibrate camera using detected holds and IFSC route map."""

    def __init__(
        self,
        route_coordinates_path: str,
        min_holds_for_calibration: int = 4,
        ransac_threshold: float = 0.05,
        min_inlier_ratio: float = 0.5
    ):
        self.min_holds = min_holds_for_calibration
        self.ransac_threshold = ransac_threshold
        self.min_inlier_ratio = min_inlier_ratio

        try:
            with open(route_coordinates_path, 'r', encoding='utf-8') as f:
                self.route_map = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load route map: {e}")
            self.route_map = None

    def calibrate(
        self,
        frame: np.ndarray,
        detected_holds: List,
        lane: str = 'left'
    ) -> Optional[CalibrationResult]:
        """Calibrate camera for a single frame."""
        if not self.route_map or len(detected_holds) < self.min_holds:
            return None

        lane_prefix = 'SN' if lane == 'left' else 'DX'
        route_holds = [
            h for h in self.route_map['holds']
            if h['panel'].startswith(lane_prefix)
        ]

        if not route_holds:
            return None

        pixel_points, meter_points, _ = self._match_holds_to_route(
            detected_holds, route_holds, frame.shape
        )

        if len(pixel_points) < self.min_holds:
            return None

        pixel_points = np.array(pixel_points, dtype=np.float32)
        meter_points = np.array(meter_points, dtype=np.float32)

        homography, inliers = self._compute_homography_ransac(pixel_points, meter_points)

        if homography is None:
            return None

        inlier_count = np.sum(inliers)
        inlier_ratio = inlier_count / len(pixel_points)

        if inlier_ratio < self.min_inlier_ratio:
            return None

        rmse = self._calculate_rmse(
            pixel_points[inliers == 1],
            meter_points[inliers == 1],
            homography
        )

        scale = self._calculate_scale(pixel_points, meter_points, homography)

        confidence = min(
            inlier_ratio * 1.5,
            1.0 - min(rmse / 0.5, 1.0)
        )

        pixel_to_meter_func = lambda px, py: self._transform_point(px, py, homography)
        meter_to_pixel_func = lambda mx, my: self._transform_point(mx, my, np.linalg.inv(homography))

        return CalibrationResult(
            homography_matrix=homography,
            pixel_to_meter_scale=scale,
            rmse_error=rmse,
            inlier_count=inlier_count,
            total_holds=len(detected_holds),
            inlier_ratio=inlier_ratio,
            confidence=confidence,
            pixel_to_meter_func=pixel_to_meter_func,
            meter_to_pixel_func=meter_to_pixel_func
        )

    def _match_holds_to_route(
        self,
        detected_holds: List,
        route_holds: List[dict],
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[List, List, List]:
        frame_height, frame_width = frame_shape[:2]
        pixel_points = []
        meter_points = []
        matched_holds = []

        for detected in detected_holds:
            norm_x = detected.pixel_x / frame_width
            norm_y = 1.0 - (detected.pixel_y / frame_height)

            best_match = None
            best_distance = float('inf')

            for route_hold in route_holds:
                route_norm_x = route_hold['wall_x_m'] / 3.0
                route_norm_y = route_hold['wall_y_m'] / 15.0

                dist = np.sqrt((norm_x - route_norm_x)**2 + (norm_y - route_norm_y)**2)

                if dist < best_distance:
                    best_distance = dist
                    best_match = route_hold

            if best_distance < 0.2 and best_match:
                pixel_points.append([detected.pixel_x, detected.pixel_y])
                meter_points.append([best_match['wall_x_m'], best_match['wall_y_m']])
                matched_holds.append(best_match['hold_num'])

        return pixel_points, meter_points, matched_holds

    def _compute_homography_ransac(self, pixel_points, meter_points):
        if len(pixel_points) < 4:
            return None, None
        try:
            homography, inliers = cv2.findHomography(
                pixel_points, meter_points, cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=2000, confidence=0.995
            )
            if homography is None or inliers is None:
                return None, None
            return homography, inliers.ravel()
        except cv2.error:
            return None, None

    def _calculate_rmse(self, pixel_points, meter_points, homography):
        if len(pixel_points) == 0:
            return float('inf')
        ones = np.ones((len(pixel_points), 1))
        pixel_homogeneous = np.hstack([pixel_points, ones])
        transformed = (homography @ pixel_homogeneous.T).T
        transformed_meters = transformed[:, :2] / transformed[:, 2:3]
        errors = np.linalg.norm(transformed_meters - meter_points, axis=1)
        return np.sqrt(np.mean(errors**2))

    def _calculate_scale(self, pixel_points, meter_points, homography):
        if len(pixel_points) < 2:
            return 0.0
        # Simplified scale calculation
        return 0.0 # Placeholder, logic is complex and maybe not strictly needed for core function

    def _transform_point(self, x, y, homography):
        point = np.array([x, y, 1.0])
        transformed = homography @ point
        return transformed[0] / transformed[2], transformed[1] / transformed[2]


class PeriodicCalibrator(CameraCalibrator):
    """Periodic calibrator that caches calibration results."""

    def __init__(
        self,
        route_coordinates_path: str,
        recalibration_interval: int = DEFAULT_PROCESSING_CONFIG["calibration_interval_frames"],
        min_holds_for_calibration: int = 4,
        ransac_threshold: float = 0.05,
        min_inlier_ratio: float = 0.5,
        min_confidence_for_cache: float = DEFAULT_PROCESSING_CONFIG["min_calibration_confidence"]
    ):
        super().__init__(
            route_coordinates_path, min_holds_for_calibration,
            ransac_threshold, min_inlier_ratio
        )
        self.recalibration_interval = recalibration_interval
        self.min_confidence_for_cache = min_confidence_for_cache
        self.calibration_cache = {}
        self.last_calibration = None
        self.frames_since_calibration = 0

    def calibrate_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        detected_holds: List,
        lane: str = 'left',
        force_recalibration: bool = False
    ) -> Optional[CalibrationResult]:
        should_recalibrate = (
            force_recalibration or
            frame_id % self.recalibration_interval == 0 or
            self.last_calibration is None
        )

        if should_recalibrate:
            calibration = self.calibrate(frame, detected_holds, lane)
            if calibration and calibration.confidence >= self.min_confidence_for_cache:
                self.calibration_cache[frame_id] = calibration
                self.last_calibration = calibration
                self.frames_since_calibration = 0
            elif calibration:
                return calibration
            else:
                if self.last_calibration:
                    self.frames_since_calibration += 1
                return self.last_calibration
        else:
            self.frames_since_calibration += 1
        
        return self.last_calibration
