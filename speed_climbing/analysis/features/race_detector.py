"""
Simple race segment detection for feature extraction.

Detects the actual climbing portion of a video by analyzing COM movement.
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

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def coverage_ratio(self) -> float:
        """Fraction of video that is racing."""
        return self.duration_frames / self.total_frames if self.total_frames > 0 else 0


class RaceSegmentDetector:
    """
    Detect the racing portion of a climbing video.

    Uses COM vertical movement to identify:
    - Start: When athlete begins consistent upward movement
    - End: When upward movement stops (finish or fall)
    """

    def __init__(
        self,
        movement_threshold: float = 0.005,  # Minimum Y change per frame
        stillness_frames: int = 10,  # Frames of stillness before start
        min_race_frames: int = 30,  # Minimum frames for valid race
    ):
        self.movement_threshold = movement_threshold
        self.stillness_frames = stillness_frames
        self.min_race_frames = min_race_frames

    def detect(
        self,
        frames: List[Dict[str, Any]],
        lane: str = 'left'
    ) -> Optional[RaceSegment]:
        """
        Detect race segment in frames.

        Args:
            frames: List of frame dictionaries
            lane: 'left' or 'right' climber

        Returns:
            RaceSegment or None if detection failed
        """
        # Extract COM Y-coordinate
        _, com_y, valid = extract_keypoint_series(frames, 'COM', lane)

        if np.sum(valid) < self.min_race_frames:
            return None

        # Calculate frame-to-frame movement
        movement = np.zeros(len(com_y))
        for i in range(1, len(com_y)):
            if valid[i] and valid[i-1]:
                # Note: Y decreases as athlete climbs UP
                movement[i] = com_y[i-1] - com_y[i]  # Positive = climbing
            else:
                movement[i] = 0

        # Detect start: First sustained upward movement after stillness
        start_frame = self._detect_start(movement, valid)

        # Detect end: Last significant upward movement or frame with valid pose
        end_frame = self._detect_end(movement, valid, start_frame)

        if end_frame - start_frame < self.min_race_frames:
            # Too short, use all valid frames
            valid_indices = np.where(valid)[0]
            if len(valid_indices) < self.min_race_frames:
                return None
            start_frame = valid_indices[0]
            end_frame = valid_indices[-1]

        # Calculate confidence
        race_movement = movement[start_frame:end_frame]
        positive_movement = np.sum(race_movement > self.movement_threshold / 2)
        confidence = positive_movement / len(race_movement) if len(race_movement) > 0 else 0

        return RaceSegment(
            start_frame=start_frame,
            end_frame=end_frame,
            total_frames=len(frames),
            confidence=confidence
        )

    def _detect_start(self, movement: np.ndarray, valid: np.ndarray) -> int:
        """Find the frame where racing starts."""
        n = len(movement)

        # Look for stillness followed by movement
        for i in range(self.stillness_frames, n - self.min_race_frames):
            # Check for stillness before
            pre_window = movement[max(0, i-self.stillness_frames):i]
            is_still = np.all(np.abs(pre_window) < self.movement_threshold)

            # Check for movement after
            post_window = movement[i:min(n, i+10)]
            has_movement = np.sum(post_window > self.movement_threshold) >= 3

            if is_still and has_movement:
                return i

        # Fallback: first valid frame with upward movement
        for i in range(n):
            if valid[i] and movement[i] > self.movement_threshold:
                return max(0, i - 5)  # Include a few frames before

        # Last resort: first valid frame
        valid_indices = np.where(valid)[0]
        return valid_indices[0] if len(valid_indices) > 0 else 0

    def _detect_end(
        self,
        movement: np.ndarray,
        valid: np.ndarray,
        start_frame: int
    ) -> int:
        """Find the frame where racing ends."""
        n = len(movement)

        # Find last frame with significant upward movement
        last_movement = start_frame
        for i in range(start_frame, n):
            if valid[i] and movement[i] > self.movement_threshold / 2:
                last_movement = i

        # Add some buffer frames after last movement
        end_frame = min(n - 1, last_movement + 15)

        # But don't go past last valid frame
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 0:
            end_frame = min(end_frame, valid_indices[-1])

        return end_frame

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
            # Return all frames if detection failed
            return frames, None

        filtered = frames[segment.start_frame:segment.end_frame + 1]
        return filtered, segment
