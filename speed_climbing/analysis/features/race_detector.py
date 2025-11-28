"""
Multi-criteria race segment detection for feature extraction.

Detects the actual climbing portion of a video using multiple signals
that work even with moving cameras.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from .base import extract_keypoint_series, get_keypoint_at_frame, calculate_angle


@dataclass
class RaceSegment:
    """Detected race segment boundaries."""
    start_frame: int
    end_frame: int
    total_frames: int
    confidence: float
    detection_method: str  # Which method detected this segment

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def coverage_ratio(self) -> float:
        """Fraction of video that is racing."""
        return self.duration_frames / self.total_frames if self.total_frames > 0 else 0


class RaceSegmentDetector:
    """
    Detect the racing portion of a climbing video using multiple criteria.

    Works with moving cameras by using pose configuration, not just position.

    Detection signals:
    1. Pose Configuration: Standing (pre-race) vs Climbing (during race)
    2. Movement Energy: Activity level based on limb movement variance
    3. Knee Angle: Straight legs (standing) vs bent legs (climbing)
    4. Arm Position: Arms down (standing) vs arms up (climbing)
    """

    def __init__(
        self,
        min_race_frames: int = 30,
        standing_knee_threshold: float = 155.0,  # Degrees - straighter = standing
        climbing_knee_threshold: float = 145.0,  # Degrees - more bent = climbing
        energy_window: int = 10,  # Frames for energy calculation
    ):
        self.min_race_frames = min_race_frames
        self.standing_knee_threshold = standing_knee_threshold
        self.climbing_knee_threshold = climbing_knee_threshold
        self.energy_window = energy_window

    def detect(
        self,
        frames: List[Dict[str, Any]],
        lane: str = 'left'
    ) -> Optional[RaceSegment]:
        """
        Detect race segment using multiple criteria.

        Returns:
            RaceSegment or None if detection failed
        """
        n = len(frames)
        if n < self.min_race_frames:
            return None

        # Calculate multiple signals
        knee_angles = self._extract_knee_angles(frames, lane)
        arm_heights = self._extract_arm_heights(frames, lane)
        movement_energy = self._calculate_movement_energy(frames, lane)

        # Combine signals to detect racing frames
        is_racing = self._combine_signals(knee_angles, arm_heights, movement_energy)

        # Find longest continuous racing segment
        start, end = self._find_longest_segment(is_racing)

        if end - start < self.min_race_frames:
            # Fallback: use all valid frames
            return self._fallback_detection(frames, lane)

        # Calculate confidence based on signal agreement
        confidence = self._calculate_confidence(
            knee_angles, arm_heights, movement_energy, start, end
        )

        return RaceSegment(
            start_frame=start,
            end_frame=end,
            total_frames=n,
            confidence=confidence,
            detection_method="multi_criteria"
        )

    def _extract_knee_angles(self, frames: List[Dict], lane: str) -> np.ndarray:
        """Extract average knee angle per frame."""
        angles = []
        for frame in frames:
            left_angle = self._get_knee_angle(frame, lane, 'left')
            right_angle = self._get_knee_angle(frame, lane, 'right')

            if left_angle is not None and right_angle is not None:
                angles.append((left_angle + right_angle) / 2)
            elif left_angle is not None:
                angles.append(left_angle)
            elif right_angle is not None:
                angles.append(right_angle)
            else:
                angles.append(np.nan)

        return np.array(angles)

    def _get_knee_angle(self, frame: Dict, lane: str, side: str) -> Optional[float]:
        """Calculate knee angle (hip-knee-ankle)."""
        hip = get_keypoint_at_frame(frame, f'{side}_hip', lane, 0.5)
        knee = get_keypoint_at_frame(frame, f'{side}_knee', lane, 0.5)
        ankle = get_keypoint_at_frame(frame, f'{side}_ankle', lane, 0.5)

        if hip is None or knee is None or ankle is None:
            return None

        return calculate_angle(hip, knee, ankle)

    def _extract_arm_heights(self, frames: List[Dict], lane: str) -> np.ndarray:
        """
        Extract relative arm height (wrist Y relative to shoulder Y).

        Returns: Negative = arms above shoulders (climbing)
                 Positive = arms below shoulders (standing)
        """
        heights = []
        for frame in frames:
            left_rel = self._get_arm_height(frame, lane, 'left')
            right_rel = self._get_arm_height(frame, lane, 'right')

            if left_rel is not None and right_rel is not None:
                # Use the higher arm (more negative = higher)
                heights.append(min(left_rel, right_rel))
            elif left_rel is not None:
                heights.append(left_rel)
            elif right_rel is not None:
                heights.append(right_rel)
            else:
                heights.append(np.nan)

        return np.array(heights)

    def _get_arm_height(self, frame: Dict, lane: str, side: str) -> Optional[float]:
        """Get wrist Y - shoulder Y (negative = wrist above shoulder)."""
        shoulder = get_keypoint_at_frame(frame, f'{side}_shoulder', lane, 0.5)
        wrist = get_keypoint_at_frame(frame, f'{side}_wrist', lane, 0.5)

        if shoulder is None or wrist is None:
            return None

        # In normalized coords, Y increases downward
        # So wrist.y < shoulder.y means wrist is ABOVE shoulder
        return wrist[1] - shoulder[1]

    def _calculate_movement_energy(self, frames: List[Dict], lane: str) -> np.ndarray:
        """
        Calculate movement energy based on limb position variance.

        Higher energy = more active movement (likely climbing)
        """
        # Extract multiple keypoint series
        _, lw_y, _ = extract_keypoint_series(frames, 'left_wrist', lane)
        _, rw_y, _ = extract_keypoint_series(frames, 'right_wrist', lane)
        _, la_y, _ = extract_keypoint_series(frames, 'left_ankle', lane)
        _, ra_y, _ = extract_keypoint_series(frames, 'right_ankle', lane)

        n = len(frames)
        energy = np.zeros(n)

        # Calculate sliding window variance
        half_win = self.energy_window // 2
        for i in range(n):
            start = max(0, i - half_win)
            end = min(n, i + half_win + 1)

            # Variance of each limb
            vars = []
            for series in [lw_y, rw_y, la_y, ra_y]:
                window = series[start:end]
                valid_window = window[~np.isnan(window)]
                if len(valid_window) >= 3:
                    vars.append(np.var(valid_window))

            energy[i] = np.mean(vars) if vars else 0

        return energy

    def _combine_signals(
        self,
        knee_angles: np.ndarray,
        arm_heights: np.ndarray,
        energy: np.ndarray
    ) -> np.ndarray:
        """
        Combine signals to determine if each frame is racing.

        A frame is racing if:
        - Knees are bent (< threshold) OR
        - Arms are raised (negative relative height) OR
        - Movement energy is high
        """
        n = len(knee_angles)
        is_racing = np.zeros(n, dtype=bool)

        # Normalize energy
        energy_norm = energy / (np.nanmax(energy) + 1e-10)
        energy_threshold = 0.2  # 20% of max energy

        for i in range(n):
            # Skip if no data
            if np.isnan(knee_angles[i]) and np.isnan(arm_heights[i]):
                continue

            # Check each criterion
            knees_bent = (
                not np.isnan(knee_angles[i]) and
                knee_angles[i] < self.climbing_knee_threshold
            )

            arms_raised = (
                not np.isnan(arm_heights[i]) and
                arm_heights[i] < -0.05  # Wrist at least 5% above shoulder
            )

            high_energy = energy_norm[i] > energy_threshold

            # Racing if any TWO criteria are met (more robust)
            criteria_met = sum([knees_bent, arms_raised, high_energy])
            is_racing[i] = criteria_met >= 1  # At least one criterion

        return is_racing

    def _find_longest_segment(self, is_racing: np.ndarray) -> Tuple[int, int]:
        """Find the longest continuous segment of racing frames."""
        n = len(is_racing)
        best_start, best_end = 0, 0
        best_length = 0

        current_start = 0
        in_segment = False

        for i in range(n):
            if is_racing[i]:
                if not in_segment:
                    current_start = i
                    in_segment = True
            else:
                if in_segment:
                    length = i - current_start
                    if length > best_length:
                        best_start = current_start
                        best_end = i
                        best_length = length
                    in_segment = False

        # Check if segment extends to end
        if in_segment:
            length = n - current_start
            if length > best_length:
                best_start = current_start
                best_end = n

        return best_start, best_end

    def _calculate_confidence(
        self,
        knee_angles: np.ndarray,
        arm_heights: np.ndarray,
        energy: np.ndarray,
        start: int,
        end: int
    ) -> float:
        """Calculate confidence in the detected segment."""
        segment_knee = knee_angles[start:end]
        segment_arm = arm_heights[start:end]
        segment_energy = energy[start:end]

        scores = []

        # Knee score: more bent = more confident
        valid_knee = segment_knee[~np.isnan(segment_knee)]
        if len(valid_knee) > 0:
            avg_knee = np.mean(valid_knee)
            # 180° = standing (score 0), 90° = very bent (score 1)
            knee_score = 1 - (avg_knee - 90) / 90
            scores.append(np.clip(knee_score, 0, 1))

        # Arm score: more raised = more confident
        valid_arm = segment_arm[~np.isnan(segment_arm)]
        if len(valid_arm) > 0:
            avg_arm = np.mean(valid_arm)
            # 0.2 = below shoulder (score 0), -0.2 = well above (score 1)
            arm_score = 1 - (avg_arm + 0.2) / 0.4
            scores.append(np.clip(arm_score, 0, 1))

        # Energy score
        if len(segment_energy) > 0:
            energy_ratio = np.mean(segment_energy) / (np.max(energy) + 1e-10)
            scores.append(np.clip(energy_ratio, 0, 1))

        return np.mean(scores) if scores else 0.5

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
            confidence=0.5,  # Lower confidence for fallback
            detection_method="fallback_all_valid"
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
