"""
Base utilities for feature extraction.

Provides common functions used across all feature analyzers.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3.

    Args:
        p1: First point (x, y)
        p2: Vertex point (x, y)
        p3: Third point (x, y)

    Returns:
        Angle in degrees (0-180)
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Handle zero vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
    angle = np.arccos(cos_angle)

    return np.degrees(angle)


def extract_keypoint_series(
    frames: List[Dict[str, Any]],
    keypoint_name: str,
    lane: str = 'left',
    min_confidence: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract x, y time series for a specific keypoint.

    Args:
        frames: List of frame dictionaries from pose JSON
        keypoint_name: Name of keypoint (e.g., 'left_wrist', 'COM')
        lane: 'left' or 'right' climber
        min_confidence: Minimum confidence threshold

    Returns:
        Tuple of (x_series, y_series, valid_mask) as numpy arrays
    """
    x_series = []
    y_series = []
    valid_mask = []

    climber_key = f'{lane}_climber'

    for frame in frames:
        climber = frame.get(climber_key)
        if climber is None or not climber.get('has_detection', False):
            x_series.append(np.nan)
            y_series.append(np.nan)
            valid_mask.append(False)
            continue

        keypoints = climber.get('keypoints', {})
        kp = keypoints.get(keypoint_name)

        if kp is None or kp.get('confidence', 0) < min_confidence:
            x_series.append(np.nan)
            y_series.append(np.nan)
            valid_mask.append(False)
            continue

        x_series.append(kp['x'])
        y_series.append(kp['y'])
        valid_mask.append(True)

    return np.array(x_series), np.array(y_series), np.array(valid_mask)


def get_keypoint_at_frame(
    frame: Dict[str, Any],
    keypoint_name: str,
    lane: str = 'left',
    min_confidence: float = 0.5
) -> Optional[Tuple[float, float]]:
    """
    Get a single keypoint (x, y) from a frame.

    Returns:
        (x, y) tuple or None if not available
    """
    climber_key = f'{lane}_climber'
    climber = frame.get(climber_key)

    if climber is None or not climber.get('has_detection', False):
        return None

    keypoints = climber.get('keypoints', {})
    kp = keypoints.get(keypoint_name)

    if kp is None or kp.get('confidence', 0) < min_confidence:
        return None

    return (kp['x'], kp['y'])


def compute_path_length(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute total path length from x, y series.

    Handles NaN values by skipping them.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) < 2:
        return 0.0

    dx = np.diff(x_valid)
    dy = np.diff(y_valid)

    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def compute_direct_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute direct (straight-line) distance from start to end.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) < 2:
        return 0.0

    dx = x_valid[-1] - x_valid[0]
    dy = y_valid[-1] - y_valid[0]

    return float(np.sqrt(dx**2 + dy**2))


def normalize_series(series: np.ndarray) -> np.ndarray:
    """
    Normalize series to 0-1 range, handling NaN values.
    """
    valid = ~np.isnan(series)
    if not np.any(valid):
        return series

    min_val = np.nanmin(series)
    max_val = np.nanmax(series)

    if max_val - min_val < 1e-10:
        return np.where(valid, 0.5, np.nan)

    return (series - min_val) / (max_val - min_val)


def interpolate_missing(series: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """
    Linear interpolation for small gaps in series.

    Args:
        series: Input series with potential NaN values
        max_gap: Maximum gap size to interpolate

    Returns:
        Interpolated series
    """
    result = series.copy()
    valid = ~np.isnan(series)

    if not np.any(valid):
        return result

    # Find valid indices
    valid_idx = np.where(valid)[0]

    # Interpolate only small gaps
    for i in range(len(valid_idx) - 1):
        start = valid_idx[i]
        end = valid_idx[i + 1]
        gap = end - start - 1

        if 0 < gap <= max_gap:
            # Linear interpolation
            result[start+1:end] = np.linspace(
                series[start], series[end], gap + 2
            )[1:-1]

    return result


def compute_velocity(y: np.ndarray, fps: float) -> np.ndarray:
    """
    Compute velocity from position series.

    Uses central difference for smoother results.
    """
    dt = 1.0 / fps
    velocity = np.full_like(y, np.nan)

    # Central difference for interior points
    for i in range(1, len(y) - 1):
        if not np.isnan(y[i-1]) and not np.isnan(y[i+1]):
            velocity[i] = (y[i+1] - y[i-1]) / (2 * dt)

    # Forward difference for first point
    if len(y) >= 2 and not np.isnan(y[0]) and not np.isnan(y[1]):
        velocity[0] = (y[1] - y[0]) / dt

    # Backward difference for last point
    if len(y) >= 2 and not np.isnan(y[-2]) and not np.isnan(y[-1]):
        velocity[-1] = (y[-1] - y[-2]) / dt

    return velocity


def compute_acceleration(velocity: np.ndarray, fps: float) -> np.ndarray:
    """
    Compute acceleration from velocity series.
    """
    return compute_velocity(velocity, fps)


def compute_jerk(y: np.ndarray, fps: float) -> np.ndarray:
    """
    Compute jerk (rate of change of acceleration) from position.

    Higher jerk = less smooth movement.
    """
    velocity = compute_velocity(y, fps)
    acceleration = compute_velocity(velocity, fps)
    jerk = compute_velocity(acceleration, fps)
    return jerk
