"""
Build time-series data from frame processing results.
"""

from typing import Dict, Any, List
import json
from dataclasses import dataclass, asdict

@dataclass
class TimeSeriesData:
    timestamps: List[float]
    y_position_m: List[float]
    x_position_m: List[float]
    status: List[str]
    calibration_quality: List[float]

class TimeSeriesBuilder:
    """
    Build time-series and calculate summary statistics.
    """

    def __init__(self):
        self.timestamps = []
        self.y_position_m = []
        self.x_position_m = []
        self.status = []
        self.calibration_quality = []
        self.finished = False

    def add_frame(
        self,
        timestamp: float,
        y_position_m: float,
        x_position_m: float,
        status: str,
        calibration_quality: float
    ):
        self.timestamps.append(timestamp)
        self.y_position_m.append(y_position_m if y_position_m is not None else 0.0)
        self.x_position_m.append(x_position_m if x_position_m is not None else 0.0)
        self.status.append(status)
        self.calibration_quality.append(calibration_quality if calibration_quality is not None else 0.0)
        
        if status in ['finished', 'out_of_frame', 'DNF']:
            self.finished = True

    def is_finished(self) -> bool:
        return self.finished

    def build(self) -> Dict[str, Any]:
        """Build final dictionary output."""
        valid_y = [y for y, s in zip(self.y_position_m, self.status) if s == 'climbing']
        
        total_distance = max(valid_y) - min(valid_y) if valid_y else 0.0
        total_time = self.timestamps[-1] - self.timestamps[0] if self.timestamps else 0.0
        
        avg_velocity = total_distance / total_time if total_time > 0 else 0.0
        
        # Calculate max velocity (simple finite difference)
        velocities = []
        for i in range(1, len(self.y_position_m)):
            dt = self.timestamps[i] - self.timestamps[i-1]
            dy = self.y_position_m[i] - self.y_position_m[i-1]
            if dt > 0:
                velocities.append(dy/dt)
        max_velocity = max(velocities) if velocities else 0.0

        return {
            "time_series": {
                "timestamps": self.timestamps,
                "y_position_m": self.y_position_m,
                "x_position_m": self.x_position_m,
                "status": self.status,
                "calibration_quality": self.calibration_quality
            },
            "summary": {
                "total_time_s": total_time,
                "total_distance_m": total_distance,
                "avg_velocity_m_s": avg_velocity,
                "max_velocity_m_s": max_velocity,
                "final_height_m": self.y_position_m[-1] if self.y_position_m else 0.0
            }
        }
