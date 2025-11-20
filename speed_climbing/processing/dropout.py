"""
Detect and manage athlete dropout scenarios.
"""

from typing import Dict, Any, Optional
from speed_climbing.core.settings import DEFAULT_PROCESSING_CONFIG, IFSC_STANDARDS

class DropoutHandler:
    """
    Detect and manage athlete dropout scenarios:
    - Out of frame (fall/disqualification)
    - Lost tracking (occlusion)
    - Race finished
    """

    def __init__(self, timeout_frames: int = DEFAULT_PROCESSING_CONFIG["dropout_timeout_frames"]):
        self.timeout_frames = timeout_frames
        self.tracking_history = []
        self.finish_height_threshold = IFSC_STANDARDS["WALL_HEIGHT_M"] - 0.5 # Approximate finish pad

    def check_dropout(
        self, 
        frame_id: int,
        pose_valid: bool, 
        calibration_valid: bool, 
        y_position_m: Optional[float]
    ) -> Dict[str, Any]:
        """
        Check if athlete has dropped out.
        """
        # Case 1: Finished
        if y_position_m and y_position_m >= self.finish_height_threshold:
             return {
                'has_dropped_out': True,
                'status': 'finished',
                'confidence': 1.0
            }

        # Case 2: No pose detected
        if not pose_valid:
            self.tracking_history.append(False)
            
            if len(self.tracking_history) > self.timeout_frames:
                recent_detections = sum(self.tracking_history[-self.timeout_frames:])
                if recent_detections < 5:
                    return {
                        'has_dropped_out': True,
                        'status': 'out_of_frame',
                        'confidence': 0.9
                    }
        else:
            self.tracking_history.append(True)

        # Case 3: Calibration failed (often means camera panned away from wall/climber)
        if not calibration_valid:
             # We treat this similarly to pose loss, but maybe with different logic
             # For now, let's rely on pose history mostly
             pass

        return {
            'has_dropped_out': False,
            'status': 'climbing',
            'confidence': 1.0
        }
