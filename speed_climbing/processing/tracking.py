"""
Track athlete position in world coordinates (meters) using per-frame camera calibration.
"""

from typing import Dict, Any, Optional
import numpy as np

from speed_climbing.core.settings import IFSC_STANDARDS
from speed_climbing.vision.holds import HoldDetector
from speed_climbing.vision.calibration import PeriodicCalibrator

class WorldCoordinateTracker:
    """
    Track athlete position in world coordinates (meters) using
    per-frame camera calibration.
    """

    def __init__(self, route_map_path: str, config: Dict[str, Any] = None):
        self.hold_detector = HoldDetector(route_map_path)
        self.calibrator = PeriodicCalibrator(route_map_path)
        self.ifsc_standards = IFSC_STANDARDS

    def process_frame(self, frame: np.ndarray, frame_id: int, lane: str) -> Dict[str, Any]:
        """
        Process single frame and return world coordinates.
        """
        # 1. Detect holds in frame
        holds = self.hold_detector.detect_holds(frame, lane=lane)

        # 2. Calibrate camera (periodic)
        calibration = self.calibrator.calibrate_frame(
            frame, frame_id, holds, lane=lane
        )

        if calibration is None or calibration.confidence < 0.6:
            return {'is_valid': False, 'reason': 'calibration_failed'}

        # Note: Pose estimation happens outside this class and is passed in
        # or we could integrate it here. For now, we return the calibration
        # so the caller can transform the pose.
        
        return {
            'is_valid': True,
            'calibration': calibration,
            'calibration_quality': calibration.confidence,
            'calibration_rmse': calibration.rmse_error
        }

    def transform_point(self, pixel_x: float, pixel_y: float, calibration) -> Optional[Dict[str, float]]:
        """Transform a pixel point to world coordinates using the calibration."""
        if not calibration or not calibration.pixel_to_meter_func:
            return None
            
        wall_x_m, wall_y_m = calibration.pixel_to_meter_func(pixel_x, pixel_y)
        
        # Transform to wall-relative coordinates (Y=0 at bottom)
        # Note: The calibration returns wall coordinates where Y=0 is bottom-left of wall
        # So wall_y_m is already correct relative to ground/start.
        
        # Calculate height from start pad
        y_from_start = wall_y_m - self.ifsc_standards['START_PAD_HEIGHT_M']
        
        return {
            'y_position_m': y_from_start,
            'x_position_m': wall_x_m,
            'wall_y_m': wall_y_m
        }
