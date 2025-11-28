"""
Variance-based race segment detection for feature extraction.

Detects the actual climbing portion of a video using movement variance,
which works reliably even with moving cameras and varying video quality.

Key insight: During racing, limbs move rapidly. Before/after racing,
athletes are relatively still (ready position or finished).
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from .base import extract_keypoint_series


@dataclass
class RaceSegment:
    """Detected race segment boundaries."""
    start_frame: int
    end_frame: int
    total_frames: int
    confidence: float
    detection_method: str  # Which method detected this segment

    # Additional diagnostics
    variance_contrast: float = 0.0  # How clear is the transition
    duration_plausible: bool = True  # Is duration reasonable for speed climbing

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def coverage_ratio(self) -> float:
        """Fraction of video that is racing."""
        return self.duration_frames / self.total_frames if self.total_frames > 0 else 0


class RaceSegmentDetector:
    """
    Detect the racing portion of a climbing video using movement variance.

    This approach works because:
    - During climbing: limbs move rapidly (high variance)
    - Ready position: athlete is still (low variance)
    - Finished/fallen: athlete stops moving (low variance)

    The method is self-calibrating - it uses relative variance within
    each video, making it robust to different camera angles and zoom levels.
    """

    # Speed climbing duration constraints
    MIN_RACE_DURATION_S = 3.0   # Minimum realistic race time
    MAX_RACE_DURATION_S = 10.0  # Maximum (with margin for falls/incomplete)
    TYPICAL_RACE_DURATION_S = 6.0  # Average race duration

    def __init__(
        self,
        min_race_frames: int = 30,
        smoothing_window: int = 10,
        activity_threshold_ratio: float = 0.25,  # 25% of max activity
        min_sustained_frames: int = 15,  # Min frames above threshold to count
        fps: float = 30.0,
    ):
        self.min_race_frames = min_race_frames
        self.smoothing_window = smoothing_window
        self.activity_threshold_ratio = activity_threshold_ratio
        self.min_sustained_frames = min_sustained_frames
        self.fps = fps

    def detect(
        self,
        frames: List[Dict[str, Any]],
        lane: str = 'left'
    ) -> Optional[RaceSegment]:
        """
        Detect race segment using movement variance.

        Returns:
            RaceSegment or None if detection failed
        """
        n = len(frames)
        if n < self.min_race_frames:
            return None

        # Calculate movement activity (frame-to-frame variance)
        activity = self._calculate_movement_activity(frames, lane)

        if activity is None or np.all(activity == 0):
            return self._fallback_detection(frames, lane)

        # Smooth the activity curve
        smoothed = self._smooth_signal(activity)

        # Find the active racing region
        start, end, method = self._find_active_region(smoothed)

        if end - start < self.min_race_frames:
            return self._fallback_detection(frames, lane)

        # Calculate confidence and validate
        variance_contrast = self._calculate_variance_contrast(smoothed, start, end)
        duration_s = (end - start) / self.fps
        duration_plausible = self.MIN_RACE_DURATION_S <= duration_s <= self.MAX_RACE_DURATION_S

        # Confidence based on multiple factors
        confidence = self._calculate_confidence(
            variance_contrast=variance_contrast,
            duration_s=duration_s,
            coverage_ratio=(end - start) / n,
            method=method
        )

        return RaceSegment(
            start_frame=start,
            end_frame=end,
            total_frames=n,
            confidence=confidence,
            detection_method=method,
            variance_contrast=variance_contrast,
            duration_plausible=duration_plausible
        )

    def _calculate_movement_activity(
        self,
        frames: List[Dict],
        lane: str
    ) -> Optional[np.ndarray]:
        """
        Calculate frame-to-frame movement activity for key limbs.

        Uses wrists and ankles as they show the most distinctive
        movement patterns during climbing.
        """
        # Extract keypoint series for limbs
        lw_x, lw_y, _ = extract_keypoint_series(frames, 'left_wrist', lane)
        rw_x, rw_y, _ = extract_keypoint_series(frames, 'right_wrist', lane)
        la_x, la_y, _ = extract_keypoint_series(frames, 'left_ankle', lane)
        ra_x, ra_y, _ = extract_keypoint_series(frames, 'right_ankle', lane)

        n = len(frames)
        activity = np.zeros(n)

        # Calculate frame-to-frame displacement for each limb
        for series_x, series_y in [(lw_x, lw_y), (rw_x, rw_y), (la_x, la_y), (ra_x, ra_y)]:
            if len(series_x) != n or len(series_y) != n:
                continue

            for i in range(1, n):
                if not np.isnan(series_x[i]) and not np.isnan(series_x[i-1]):
                    dx = series_x[i] - series_x[i-1]
                    dy = series_y[i] - series_y[i-1]
                    displacement = np.sqrt(dx**2 + dy**2)
                    activity[i] += displacement

        # Check if we have meaningful data
        if np.max(activity) < 1e-6:
            return None

        return activity

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply sliding window smoothing to reduce noise."""
        n = len(signal)
        smoothed = np.zeros(n)
        half_win = self.smoothing_window // 2

        for i in range(n):
            start = max(0, i - half_win)
            end = min(n, i + half_win + 1)
            smoothed[i] = np.mean(signal[start:end])

        return smoothed

    def _find_active_region(
        self,
        activity: np.ndarray
    ) -> Tuple[int, int, str]:
        """
        Find the sustained high-activity region.

        Returns:
            (start_frame, end_frame, detection_method)
        """
        n = len(activity)
        max_activity = np.max(activity)

        if max_activity < 1e-6:
            return 0, n, "fallback_no_activity"

        # Adaptive threshold based on activity distribution
        threshold = max_activity * self.activity_threshold_ratio

        # Find frames above threshold
        above_threshold = activity > threshold

        # Find the first and last sustained regions above threshold
        start = self._find_first_sustained(above_threshold, self.min_sustained_frames)
        end = self._find_last_sustained(above_threshold, self.min_sustained_frames)

        if start >= end or end - start < self.min_race_frames:
            # Try with lower threshold
            threshold = max_activity * (self.activity_threshold_ratio * 0.5)
            above_threshold = activity > threshold
            start = self._find_first_sustained(above_threshold, self.min_sustained_frames // 2)
            end = self._find_last_sustained(above_threshold, self.min_sustained_frames // 2)

            if start >= end:
                return 0, n, "fallback_threshold"
            return start, end, "variance_relaxed"

        return start, end, "variance_primary"

    def _find_first_sustained(
        self,
        mask: np.ndarray,
        min_frames: int
    ) -> int:
        """Find the first index where condition is sustained for min_frames."""
        n = len(mask)
        count = 0

        for i in range(n):
            if mask[i]:
                count += 1
                if count >= min_frames:
                    return i - min_frames + 1
            else:
                count = 0

        return 0

    def _find_last_sustained(
        self,
        mask: np.ndarray,
        min_frames: int
    ) -> int:
        """Find the last index where condition was sustained for min_frames."""
        n = len(mask)
        count = 0

        for i in range(n - 1, -1, -1):
            if mask[i]:
                count += 1
                if count >= min_frames:
                    return i + min_frames
            else:
                count = 0

        return n

    def _calculate_variance_contrast(
        self,
        activity: np.ndarray,
        start: int,
        end: int
    ) -> float:
        """
        Calculate how distinct the racing period is from non-racing.

        Higher contrast = more confident detection.
        """
        if start <= 0 and end >= len(activity):
            return 0.5  # Can't compare, medium confidence

        racing_activity = activity[start:end]

        # Compare to pre-race and post-race periods
        pre_race = activity[:max(1, start)]
        post_race = activity[min(len(activity)-1, end):]

        racing_mean = np.mean(racing_activity)

        non_racing_parts = []
        if len(pre_race) > 5:
            non_racing_parts.extend(pre_race)
        if len(post_race) > 5:
            non_racing_parts.extend(post_race)

        if not non_racing_parts:
            return 0.5

        non_racing_mean = np.mean(non_racing_parts)

        # Contrast ratio
        if non_racing_mean < 1e-6:
            return 1.0 if racing_mean > 0 else 0.5

        contrast = racing_mean / (non_racing_mean + 1e-6)
        return min(1.0, contrast / 5.0)  # Normalize: 5x = perfect

    def _calculate_confidence(
        self,
        variance_contrast: float,
        duration_s: float,
        coverage_ratio: float,
        method: str
    ) -> float:
        """
        Calculate overall confidence in the detected segment.

        Factors:
        - Variance contrast: How distinct is racing from non-racing
        - Duration plausibility: Is it a realistic race duration
        - Coverage: How much of the video is racing
        - Method: Primary vs fallback detection
        """
        scores = []

        # Variance contrast score (0-1)
        scores.append(variance_contrast)

        # Duration score
        if self.MIN_RACE_DURATION_S <= duration_s <= self.MAX_RACE_DURATION_S:
            # Perfect if close to typical duration
            duration_score = 1.0 - abs(duration_s - self.TYPICAL_RACE_DURATION_S) / 5.0
            scores.append(max(0.5, min(1.0, duration_score)))
        elif duration_s < self.MIN_RACE_DURATION_S:
            scores.append(0.3)  # Too short
        else:
            scores.append(0.4)  # Too long

        # Coverage score - racing should be 40-90% of video typically
        if 0.3 <= coverage_ratio <= 0.95:
            scores.append(0.8)
        else:
            scores.append(0.5)

        # Method score
        method_scores = {
            "variance_primary": 1.0,
            "variance_relaxed": 0.7,
            "fallback_threshold": 0.4,
            "fallback_no_activity": 0.3,
            "fallback_all_valid": 0.5,
        }
        scores.append(method_scores.get(method, 0.5))

        return np.mean(scores)

    def _fallback_detection(
        self,
        frames: List[Dict],
        lane: str
    ) -> Optional[RaceSegment]:
        """Fallback: return all frames with valid pose data."""
        valid_indices = []
        for i, frame in enumerate(frames):
            climber = frame.get(f'{lane}_climber')
            if climber and climber.get('has_detection', False):
                valid_indices.append(i)

        if len(valid_indices) < self.min_race_frames:
            return None

        return RaceSegment(
            start_frame=valid_indices[0],
            end_frame=valid_indices[-1],
            total_frames=len(frames),
            confidence=0.4,  # Lower confidence for fallback
            detection_method="fallback_all_valid",
            variance_contrast=0.0,
            duration_plausible=True  # Assume valid
        )

    def filter_racing_frames(
        self,
        frames: List[Dict[str, Any]],
        lane: str = 'left'
    ) -> Tuple[List[Dict[str, Any]], Optional[RaceSegment]]:
        """
        Filter frames to only include racing portion.

        Returns:
            Tuple of (filtered_frames, segment_info)
        """
        segment = self.detect(frames, lane)

        if segment is None:
            return frames, None

        filtered = frames[segment.start_frame:segment.end_frame + 1]
        return filtered, segment

    def get_activity_curve(
        self,
        frames: List[Dict[str, Any]],
        lane: str = 'left'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the raw and smoothed activity curves for visualization.

        Returns:
            (raw_activity, smoothed_activity)
        """
        activity = self._calculate_movement_activity(frames, lane)
        if activity is None:
            return np.zeros(len(frames)), np.zeros(len(frames))

        smoothed = self._smooth_signal(activity)
        return activity, smoothed
