"""
Absolute metrics extraction (when calibration is available).

This module provides optional absolute metrics when wall calibration succeeds.
Works alongside relative features to provide hybrid output.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class AbsoluteMetrics:
    """Absolute metrics requiring wall calibration."""
    available: bool = False
    confidence: float = 0.0

    # Distance metrics (meters)
    total_distance_m: Optional[float] = None
    max_height_m: Optional[float] = None
    start_height_m: Optional[float] = None

    # Velocity metrics (m/s)
    avg_velocity_m_s: Optional[float] = None
    max_velocity_m_s: Optional[float] = None

    # Time metrics (seconds) - if finish detected
    race_duration_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'available': self.available,
            'confidence': self.confidence,
            'total_distance_m': self.total_distance_m,
            'max_height_m': self.max_height_m,
            'start_height_m': self.start_height_m,
            'avg_velocity_m_s': self.avg_velocity_m_s,
            'max_velocity_m_s': self.max_velocity_m_s,
            'race_duration_s': self.race_duration_s,
        }


class AbsoluteMetricsExtractor:
    """
    Extract absolute metrics when calibration data is available.

    This is OPTIONAL - only used when calibration succeeds.
    Falls back gracefully when calibration is unavailable.
    """

    def __init__(
        self,
        min_calibration_confidence: float = 0.6,
        wall_height_m: float = 15.0
    ):
        self.min_calibration_confidence = min_calibration_confidence
        self.wall_height_m = wall_height_m

    def extract_from_calibrated_data(
        self,
        calibrated_positions: List[Dict[str, Any]],
        fps: float = 30.0
    ) -> AbsoluteMetrics:
        """
        Extract absolute metrics from calibrated position data.

        Args:
            calibrated_positions: List of {y_position_m, timestamp, confidence}
            fps: Frame rate

        Returns:
            AbsoluteMetrics with available=True if successful
        """
        if not calibrated_positions:
            return AbsoluteMetrics(available=False)

        # Filter by confidence
        valid = [
            p for p in calibrated_positions
            if p.get('confidence', 0) >= self.min_calibration_confidence
        ]

        if len(valid) < 10:
            return AbsoluteMetrics(available=False)

        # Extract heights
        heights = [p['y_position_m'] for p in valid]
        timestamps = [p['timestamp'] for p in valid]

        # Calculate metrics
        max_height = max(heights)
        min_height = min(heights)
        total_distance = max_height - min_height

        # Velocity calculation
        velocities = []
        for i in range(1, len(heights)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                v = (heights[i] - heights[i-1]) / dt
                velocities.append(v)

        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = max(velocities) if velocities else 0

        # Duration
        duration = timestamps[-1] - timestamps[0] if timestamps else 0

        # Average confidence
        avg_confidence = np.mean([p.get('confidence', 0) for p in valid])

        # Validate against wall constraints
        is_valid = (
            0 <= min_height <= self.wall_height_m and
            0 <= max_height <= self.wall_height_m and
            total_distance > 0 and
            total_distance <= self.wall_height_m
        )

        if not is_valid:
            return AbsoluteMetrics(
                available=False,
                confidence=avg_confidence * 0.5  # Lower confidence due to invalid values
            )

        return AbsoluteMetrics(
            available=True,
            confidence=avg_confidence,
            total_distance_m=total_distance,
            max_height_m=max_height,
            start_height_m=min_height,
            avg_velocity_m_s=avg_velocity,
            max_velocity_m_s=max_velocity,
            race_duration_s=duration
        )

    def try_extract_from_pose_with_scale(
        self,
        frames: List[Dict[str, Any]],
        lane: str,
        estimated_scale: float,  # pixels per meter
        scale_confidence: float
    ) -> AbsoluteMetrics:
        """
        Try to extract absolute metrics using estimated scale.

        This is a fallback when homography calibration fails but
        we have an athlete-based scale estimate.
        """
        if scale_confidence < self.min_calibration_confidence:
            return AbsoluteMetrics(available=False)

        # Extract COM Y positions
        com_positions = []
        for i, frame in enumerate(frames):
            climber = frame.get(f'{lane}_climber')
            if climber and climber.get('has_detection'):
                keypoints = climber.get('keypoints', {})
                com = keypoints.get('COM')
                if com and com.get('confidence', 0) > 0.5:
                    com_positions.append({
                        'y_pixel': com['y'],  # Normalized 0-1
                        'timestamp': frame.get('timestamp', i / 30.0),
                        'confidence': com['confidence']
                    })

        if len(com_positions) < 10:
            return AbsoluteMetrics(available=False)

        # Convert to meters (approximate)
        # Note: This assumes frame height represents ~camera_view_height_m
        camera_view_m = 5.0  # Typical view window

        calibrated = []
        for p in com_positions:
            # In normalized coords, y=0 is top, y=1 is bottom
            # Convert to height from bottom of view
            height_in_view = (1.0 - p['y_pixel']) * camera_view_m
            calibrated.append({
                'y_position_m': height_in_view,
                'timestamp': p['timestamp'],
                'confidence': p['confidence'] * scale_confidence
            })

        return self.extract_from_calibrated_data(calibrated)
