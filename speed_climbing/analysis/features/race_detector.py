"""
Hybrid race segment detection for feature extraction.

Uses multiple signals for accurate detection:
1. Activity burst detection for START (derivative of movement variance)
2. Wrist minimum Y detection for END (highest point = touching top device)
3. Variance-based fallback for robustness

Key insight:
- Start: Athletes make preparatory movements, but react with sudden BURST to signal
- End: Hand reaches minimum Y coordinate when touching top device
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
    start_method: str = "variance"  # How start was detected
    end_method: str = "variance"  # How end was detected

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def coverage_ratio(self) -> float:
        """Fraction of video that is racing."""
        return self.duration_frames / self.total_frames if self.total_frames > 0 else 0


class RaceSegmentDetector:
    """
    Detect the racing portion of a climbing video using hybrid approach.

    Detection Strategy:
    - START: Detect activity BURST (sharp increase in movement)
    - END: Detect when wrist reaches MINIMUM Y (top of climb = hand touches device)

    This approach works because:
    - Preparatory movement is gradual, reaction to start signal is sudden
    - Wrist Y decreases during climb, minimum Y = highest point = finish
    """

    # Speed climbing duration constraints
    MIN_RACE_DURATION_S = 4.5   # Minimum realistic race time (world record ~5s)
    MAX_RACE_DURATION_S = 8.0   # Maximum (falls/incomplete should be < 8s)
    TYPICAL_RACE_DURATION_S = 5.5  # Elite average race duration
    EXPECTED_END_BUFFER_S = 0.5  # Buffer before expected end to stop

    def __init__(
        self,
        min_race_frames: int = 30,
        smoothing_window: int = 10,
        activity_threshold_ratio: float = 0.25,  # 25% of max activity
        min_sustained_frames: int = 15,  # Min frames above threshold to count
        burst_threshold_ratio: float = 0.35,  # 35% of max derivative for burst
        wrist_top_margin: float = 1.15,  # 15% margin for wrist top detection
        fps: float = 30.0,
    ):
        self.min_race_frames = min_race_frames
        self.smoothing_window = smoothing_window
        self.activity_threshold_ratio = activity_threshold_ratio
        self.min_sustained_frames = min_sustained_frames
        self.burst_threshold_ratio = burst_threshold_ratio
        self.wrist_top_margin = wrist_top_margin
        self.fps = fps

    def detect(
        self,
        frames: List[Dict[str, Any]],
        lane: str = 'left'
    ) -> Optional[RaceSegment]:
        """
        Detect race segment using hybrid approach.

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

        # ===== START DETECTION: Activity burst =====
        start, start_method = self._detect_activity_burst(smoothed)

        # ===== END DETECTION: Duration-constrained activity analysis =====
        # Use the fact that speed climbing races are 5-7 seconds
        end, end_method = self._detect_end_by_duration_and_activity(smoothed, start)

        # Fallback to variance-based if detection failed
        if end is None or end <= start:
            _, end_variance, _ = self._find_active_region(smoothed)
            end = end_variance
            end_method = "variance_fallback"

        # Sanity checks
        if end - start < self.min_race_frames:
            return self._fallback_detection(frames, lane)

        # Calculate confidence and validate
        variance_contrast = self._calculate_variance_contrast(smoothed, start, end)
        duration_s = (end - start) / self.fps
        duration_plausible = self.MIN_RACE_DURATION_S <= duration_s <= self.MAX_RACE_DURATION_S

        # Determine overall method
        if start_method == "burst" and end_method in ["duration_peak", "duration_typical"]:
            method = "hybrid_optimal"
        elif start_method == "burst":
            method = "hybrid_burst_start"
        elif end_method in ["duration_peak", "duration_typical"]:
            method = "hybrid_duration_end"
        else:
            method = "variance_primary"

        # Confidence based on multiple factors
        confidence = self._calculate_confidence(
            variance_contrast=variance_contrast,
            duration_s=duration_s,
            coverage_ratio=(end - start) / n,
            method=method,
            start_method=start_method,
            end_method=end_method
        )

        return RaceSegment(
            start_frame=start,
            end_frame=end,
            total_frames=n,
            confidence=confidence,
            detection_method=method,
            variance_contrast=variance_contrast,
            duration_plausible=duration_plausible,
            start_method=start_method,
            end_method=end_method
        )

    def _detect_activity_burst(
        self,
        smoothed_activity: np.ndarray
    ) -> Tuple[int, str]:
        """
        Detect race start by finding the first significant activity BURST.

        A burst is a sharp increase in activity (high derivative).
        This distinguishes the reaction to start signal from preparatory movement.

        Returns:
            (start_frame, detection_method)
        """
        n = len(smoothed_activity)

        # Calculate derivative (rate of change)
        derivative = np.gradient(smoothed_activity)

        # Smooth the derivative to reduce noise
        derivative_smooth = self._smooth_signal(derivative)

        max_derivative = np.max(derivative_smooth)
        if max_derivative < 1e-6:
            # Fallback to threshold-based
            start, _, _ = self._find_active_region(smoothed_activity)
            return start, "variance_fallback"

        # Threshold for burst detection
        burst_threshold = max_derivative * self.burst_threshold_ratio

        # Find first sustained burst
        burst_mask = derivative_smooth > burst_threshold
        min_burst_frames = max(3, self.min_sustained_frames // 3)

        start = self._find_first_sustained(burst_mask, min_burst_frames)

        if start == 0:
            # No clear burst found, use threshold-based
            start, _, _ = self._find_active_region(smoothed_activity)
            return start, "variance_fallback"

        return start, "burst"

    def _detect_end_by_duration_and_activity(
        self,
        smoothed_activity: np.ndarray,
        start_frame: int
    ) -> Tuple[Optional[int], str]:
        """
        Detect race end using duration constraints and activity patterns.

        Speed climbing races are predictable in duration (5-7 seconds for elite).
        This method finds the activity peak within that expected window.

        Returns:
            (end_frame, detection_method)
        """
        n = len(smoothed_activity)

        # Define search window based on expected race duration
        min_end = start_frame + int(self.MIN_RACE_DURATION_S * self.fps)
        max_end = start_frame + int(self.MAX_RACE_DURATION_S * self.fps)
        typical_end = start_frame + int(self.TYPICAL_RACE_DURATION_S * self.fps)

        # Ensure within bounds
        min_end = min(min_end, n - 1)
        max_end = min(max_end, n - 1)
        typical_end = min(typical_end, n - 1)

        if min_end >= n - 10:
            return None, "failed"

        # Strategy: Find the last significant activity peak in the expected window
        # Athletes are most active near the end of the race (sprint to finish)

        # Look for activity peak in the typical race duration window
        search_start = start_frame + int(4.0 * self.fps)  # After 4 seconds
        search_end = min(n, start_frame + int(7.0 * self.fps))  # Before 7 seconds

        if search_start >= search_end:
            # Video too short, use typical end
            return typical_end, "duration_typical"

        window = smoothed_activity[search_start:search_end]

        # Find the peak in this window (athletes sprint at the end)
        peak_idx_local = np.argmax(window)
        peak_frame = search_start + peak_idx_local

        # The end is slightly after the peak (when they hit the button)
        # Add a small buffer (0.3-0.5 seconds)
        buffer_frames = int(self.EXPECTED_END_BUFFER_S * self.fps)
        end_frame = min(peak_frame + buffer_frames, max_end)

        # Validate: activity at end should still be reasonably high
        if end_frame < n and smoothed_activity[end_frame] > 0.3 * np.max(smoothed_activity):
            return end_frame, "duration_peak"
        else:
            # Use typical duration as fallback
            return typical_end, "duration_typical"

    def _detect_activity_drop(
        self,
        smoothed_activity: np.ndarray,
        start_frame: int
    ) -> Tuple[Optional[int], str]:
        """
        Detect race end by finding where activity DROPS sharply.

        After touching the top device, there's a characteristic drop in climbing
        activity. This works even with moving cameras because it's based on
        movement patterns, not position.

        Returns:
            (end_frame, detection_method)
        """
        n = len(smoothed_activity)

        # Only search after start + minimum race duration
        min_race_frames = int(self.MIN_RACE_DURATION_S * self.fps)
        search_start = start_frame + min_race_frames

        if search_start >= n - 10:
            return None, "failed"

        # Calculate derivative (negative = dropping)
        derivative = np.gradient(smoothed_activity)
        derivative_smooth = self._smooth_signal(derivative)

        # Find significant negative derivative (activity drop)
        min_derivative = np.min(derivative_smooth[search_start:])

        if min_derivative >= 0:
            # No significant drop found
            return None, "failed"

        # Threshold for drop detection (40% of max drop)
        drop_threshold = min_derivative * 0.4

        # Find first significant drop after minimum race time
        for i in range(search_start, n):
            if derivative_smooth[i] < drop_threshold:
                # Found a drop - the end is where activity was still high
                # Go back a few frames to before the drop started
                end_frame = max(start_frame + min_race_frames, i - 5)
                return end_frame, "activity_drop"

        return None, "failed"

    def _detect_sustained_low_activity(
        self,
        smoothed_activity: np.ndarray,
        start_frame: int
    ) -> Tuple[Optional[int], str]:
        """
        Detect race end by finding where activity becomes sustainedly LOW.

        After the race, activity drops and stays low. Find the transition point.

        Returns:
            (end_frame, detection_method)
        """
        n = len(smoothed_activity)

        # Calculate statistics for the racing period
        min_race_frames = int(self.MIN_RACE_DURATION_S * self.fps)
        max_race_frames = int(self.MAX_RACE_DURATION_S * self.fps)

        search_start = start_frame + min_race_frames
        search_end = min(n, start_frame + max_race_frames)

        if search_start >= search_end:
            return None, "failed"

        # Get racing activity level (first half of expected race)
        race_mid = start_frame + min_race_frames
        if race_mid < n:
            racing_mean = np.mean(smoothed_activity[start_frame:race_mid])
        else:
            racing_mean = np.mean(smoothed_activity[start_frame:])

        # Threshold: activity drops to 30% of racing level
        low_threshold = racing_mean * 0.30

        # Find first sustained low activity period
        low_count = 0
        required_low_frames = max(5, self.min_sustained_frames // 3)

        for i in range(search_start, search_end):
            if smoothed_activity[i] < low_threshold:
                low_count += 1
                if low_count >= required_low_frames:
                    # Found sustained low activity
                    end_frame = i - required_low_frames
                    return end_frame, "low_activity"
            else:
                low_count = 0

        return None, "failed"

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
        Find the sustained high-activity region (variance-based fallback).

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
        method: str,
        start_method: str,
        end_method: str
    ) -> float:
        """
        Calculate overall confidence in the detected segment.

        Factors:
        - Variance contrast: How distinct is racing from non-racing
        - Duration plausibility: Is it a realistic race duration
        - Coverage: How much of the video is racing
        - Method: Hybrid methods get higher confidence
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

        # Method score - hybrid methods are more reliable
        method_scores = {
            "hybrid_optimal": 1.0,
            "hybrid_burst_start": 0.85,
            "hybrid_duration_end": 0.85,
            "variance_primary": 0.75,
            "variance_relaxed": 0.6,
            "fallback_threshold": 0.4,
            "fallback_no_activity": 0.3,
            "fallback_all_valid": 0.5,
        }
        scores.append(method_scores.get(method, 0.5))

        # Bonus for specific detection methods
        if start_method == "burst":
            scores.append(0.85)
        if end_method == "duration_peak":
            scores.append(0.90)
        elif end_method == "duration_typical":
            scores.append(0.80)

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
            duration_plausible=True,  # Assume valid
            start_method="fallback",
            end_method="fallback"
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

    def get_wrist_height_curve(
        self,
        frames: List[Dict[str, Any]],
        lane: str = 'left'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the wrist Y-coordinate curves for visualization.

        Lower Y = higher position on wall.

        Returns:
            (min_wrist_y_raw, min_wrist_y_smoothed)
        """
        _, lw_y, _ = extract_keypoint_series(frames, 'left_wrist', lane)
        _, rw_y, _ = extract_keypoint_series(frames, 'right_wrist', lane)

        n = len(frames)
        if len(lw_y) != n or len(rw_y) != n:
            return np.ones(n), np.ones(n)

        # Replace NaN with 1.0 (bottom)
        lw_y_clean = np.where(np.isnan(lw_y), 1.0, lw_y)
        rw_y_clean = np.where(np.isnan(rw_y), 1.0, rw_y)

        # Minimum of both wrists
        min_wrist_y = np.minimum(lw_y_clean, rw_y_clean)
        smoothed = self._smooth_signal(min_wrist_y)

        return min_wrist_y, smoothed
