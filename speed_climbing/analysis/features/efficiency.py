"""
Path efficiency analysis for climbing movement.

Extracts efficiency metrics from COM trajectory without wall calibration.
"""

import numpy as np
from typing import Dict, Any, List

from .base import (
    extract_keypoint_series,
    compute_path_length,
    compute_direct_distance,
    compute_velocity,
    compute_jerk,
    interpolate_missing,
)


class EfficiencyAnalyzer:
    """
    Analyze climbing efficiency from COM trajectory.

    Extracts:
    - path_straightness: Actual path / direct path ratio
    - lateral_movement_ratio: Lateral vs vertical movement
    - vertical_progress_rate: Climbing speed in normalized units
    - com_stability_index: Smoothness of COM trajectory
    """

    def __init__(self, fps: float = 30.0, min_frames: int = 30):
        """
        Args:
            fps: Video frame rate
            min_frames: Minimum frames required for analysis
        """
        self.fps = fps
        self.min_frames = min_frames

    def analyze(self, frames: List[Dict[str, Any]], lane: str = 'left') -> Dict[str, float]:
        """
        Extract efficiency features from pose frames.

        Args:
            frames: List of frame dictionaries from pose JSON
            lane: 'left' or 'right' climber

        Returns:
            Dictionary of efficiency features
        """
        # Extract COM trajectory
        com_x, com_y, com_valid = extract_keypoint_series(frames, 'COM', lane)

        # Check minimum data
        if np.sum(com_valid) < self.min_frames:
            return self._default_features()

        # Interpolate small gaps
        com_x = interpolate_missing(com_x)
        com_y = interpolate_missing(com_y)

        # Calculate features
        features = {
            'path_straightness': self._path_straightness(com_x, com_y),
            'lateral_movement_ratio': self._lateral_ratio(com_x, com_y),
            'vertical_progress_rate': self._vertical_rate(com_y),
            'com_stability_index': self._stability_index(com_x, com_y),
            'movement_smoothness': self._smoothness(com_y),
            'acceleration_variance': self._acceleration_variance(com_y),
        }

        return features

    def _path_straightness(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate path straightness ratio.

        Returns: direct_distance / actual_path (0-1, 1 = perfectly straight)
        """
        path_length = compute_path_length(x, y)
        direct_dist = compute_direct_distance(x, y)

        if path_length < 1e-10:
            return 0.0

        # Inverse ratio since lower path length is better
        straightness = direct_dist / path_length

        return float(np.clip(straightness, 0, 1))

    def _lateral_ratio(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate lateral movement ratio.

        Returns: std(x) / delta(y) - lower is better
        """
        valid = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid]
        y_valid = y[valid]

        if len(x_valid) < 2:
            return 0.0

        # Note: In normalized coords, y decreases as climber goes UP
        # So we use min - max for vertical progress
        delta_y = abs(y_valid[0] - y_valid[-1])  # Vertical progress
        std_x = np.std(x_valid)  # Lateral variance

        if delta_y < 1e-10:
            return 1.0  # No vertical progress = bad

        return float(std_x / delta_y)

    def _vertical_rate(self, y: np.ndarray) -> float:
        """
        Calculate average vertical progress rate.

        Returns: delta_y / num_frames (normalized units per frame)
        """
        valid = ~np.isnan(y)
        y_valid = y[valid]

        if len(y_valid) < 2:
            return 0.0

        # In normalized coords, y decreases as climber goes UP
        # So vertical progress = y_start - y_end
        delta_y = y_valid[0] - y_valid[-1]
        num_frames = len(y_valid)

        # Rate per frame (positive = upward movement)
        rate = delta_y / num_frames

        return float(rate)

    def _stability_index(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate COM stability index.

        Based on jerk minimization - less jerk = more stable.
        Returns 0-1 (1 = most stable)
        """
        valid = ~(np.isnan(x) | np.isnan(y))

        if np.sum(valid) < self.min_frames:
            return 0.0

        x_valid = x[valid]
        y_valid = y[valid]

        # Calculate jerk for both axes
        jerk_x = compute_jerk(x_valid, self.fps)
        jerk_y = compute_jerk(y_valid, self.fps)

        # Combined jerk magnitude
        jerk_valid = ~(np.isnan(jerk_x) | np.isnan(jerk_y))
        if np.sum(jerk_valid) < 10:
            return 0.0

        jerk_magnitude = np.sqrt(jerk_x[jerk_valid]**2 + jerk_y[jerk_valid]**2)
        mean_jerk = np.mean(jerk_magnitude)

        # Convert to stability index (inverse of jerk, normalized)
        # Empirical scaling - typical jerk range is 0-0.1 in normalized coords
        stability = 1.0 / (1.0 + mean_jerk * 100)

        return float(np.clip(stability, 0, 1))

    def _smoothness(self, y: np.ndarray) -> float:
        """
        Calculate movement smoothness using spectral arc length.

        Based on: Balasubramanian et al., "A robust and sensitive metric
        for quantifying movement smoothness"

        Returns 0-1 (1 = smoothest)
        """
        valid = ~np.isnan(y)
        if np.sum(valid) < self.min_frames:
            return 0.0

        y_valid = y[valid]

        # Velocity
        velocity = compute_velocity(y_valid, self.fps)
        vel_valid = ~np.isnan(velocity)

        if np.sum(vel_valid) < 10:
            return 0.0

        v = velocity[vel_valid]

        # Normalize velocity
        v_max = np.max(np.abs(v))
        if v_max < 1e-10:
            return 0.0

        v_norm = v / v_max

        # FFT of velocity
        from scipy.fft import fft, fftfreq
        n = len(v_norm)
        yf = np.abs(fft(v_norm))[:n//2]
        xf = fftfreq(n, 1/self.fps)[:n//2]

        # Spectral arc length (simplified)
        # Lower = smoother
        if len(yf) < 2:
            return 0.0

        yf_norm = yf / (np.max(yf) + 1e-10)
        arc_length = np.sum(np.sqrt(np.diff(xf)**2 + np.diff(yf_norm)**2))

        # Convert to smoothness (inverse)
        smoothness = 1.0 / (1.0 + arc_length)

        return float(np.clip(smoothness, 0, 1))

    def _acceleration_variance(self, y: np.ndarray) -> float:
        """
        Calculate variance of acceleration.

        Lower variance = more consistent movement.
        Returns normalized variance (0-1 scale)
        """
        valid = ~np.isnan(y)
        if np.sum(valid) < self.min_frames:
            return 0.0

        y_valid = y[valid]

        # Velocity then acceleration
        velocity = compute_velocity(y_valid, self.fps)
        acceleration = compute_velocity(velocity, self.fps)

        acc_valid = ~np.isnan(acceleration)
        if np.sum(acc_valid) < 10:
            return 0.0

        acc = acceleration[acc_valid]

        # Variance normalized by max acceleration
        max_acc = np.max(np.abs(acc))
        if max_acc < 1e-10:
            return 0.0

        variance = np.var(acc) / (max_acc**2)

        return float(variance)

    def _default_features(self) -> Dict[str, float]:
        """Return default features when analysis is not possible."""
        return {
            'path_straightness': 0.0,
            'lateral_movement_ratio': 0.0,
            'vertical_progress_rate': 0.0,
            'com_stability_index': 0.0,
            'movement_smoothness': 0.0,
            'acceleration_variance': 0.0,
        }
