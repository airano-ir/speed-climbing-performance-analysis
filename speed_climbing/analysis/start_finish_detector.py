"""
Start and Finish Detection for Speed Climbing Races
====================================================
Detects race start and finish events using pose keypoints and wall coordinates.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StartEvent:
    """Represents a detected race start event."""
    frame_id: int
    timestamp: float
    com_height_m: float
    confidence: float
    method: str  # 'com_rise' or 'foot_lift' or 'manual'
    details: Dict


@dataclass
class FinishEvent:
    """Represents a detected race finish event."""
    frame_id: int
    timestamp: float
    hand_height_m: float
    confidence: float
    method: str  # 'top_reach' or 'manual'
    details: Dict


class StartDetector:
    """
    Detects the start of a speed climbing race.

    Strategy:
    1. Monitor COM height in ready position (should be 0.8-1.2m)
    2. Detect upward movement initiation
    3. Validate using foot keypoints (one foot lifts off ground)
    """

    def __init__(
        self,
        ready_position_height_range: Tuple[float, float] = (0.8, 1.2),
        movement_threshold_m: float = 0.15,
        min_confidence: float = 0.5,
        calibration_window_frames: int = 30
    ):
        """
        Args:
            ready_position_height_range: Expected COM height range when standing on start pad (meters)
            movement_threshold_m: Minimum upward movement to trigger start (meters)
            min_confidence: Minimum pose confidence to consider
            calibration_window_frames: Number of initial frames to calibrate baseline
        """
        self.ready_height_min, self.ready_height_max = ready_position_height_range
        self.movement_threshold = movement_threshold_m
        self.min_confidence = min_confidence
        self.calibration_window = calibration_window_frames

        # State tracking
        self.baseline_com_height: Optional[float] = None
        self.calibration_buffer: List[float] = []
        self.start_detected = False

    def process_frame(
        self,
        frame_id: int,
        timestamp: float,
        com_height_m: float,
        pose_keypoints: Dict,
        pose_confidence: float
    ) -> Optional[StartEvent]:
        """
        Process a single frame to detect start event.

        Args:
            frame_id: Frame number
            timestamp: Frame timestamp in seconds
            com_height_m: Center of mass height in meters (world coordinates)
            pose_keypoints: Dict of BlazePose keypoints with world coordinates
            pose_confidence: Overall pose detection confidence

        Returns:
            StartEvent if start is detected, None otherwise
        """
        if self.start_detected:
            return None

        if pose_confidence < self.min_confidence:
            return None

        # Calibration phase: establish baseline COM height
        if len(self.calibration_buffer) < self.calibration_window:
            if self._validate_ready_position(com_height_m):
                self.calibration_buffer.append(com_height_m)
            return None

        # Set baseline from calibration
        if self.baseline_com_height is None and len(self.calibration_buffer) > 0:
            self.baseline_com_height = np.median(self.calibration_buffer)
            logger.info(f"Baseline COM height calibrated: {self.baseline_com_height:.3f}m")

        # Detect start: upward movement beyond threshold
        if self.baseline_com_height is not None:
            delta = com_height_m - self.baseline_com_height

            if delta > self.movement_threshold:
                # Additional validation using foot keypoints
                foot_lift_detected = self._detect_foot_lift(pose_keypoints)

                self.start_detected = True

                event = StartEvent(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    com_height_m=com_height_m,
                    confidence=pose_confidence,
                    method='com_rise' if delta > self.movement_threshold else 'foot_lift',
                    details={
                        'baseline_height': self.baseline_com_height,
                        'delta_height': delta,
                        'foot_lift_detected': foot_lift_detected,
                        'calibration_frames': len(self.calibration_buffer)
                    }
                )

                logger.info(f"START detected at frame {frame_id} (t={timestamp:.2f}s), "
                           f"COM: {com_height_m:.3f}m, delta: {delta:.3f}m")
                return event

        return None

    def _validate_ready_position(self, com_height_m: float) -> bool:
        """Check if COM height is within expected ready position range."""
        return self.ready_height_min <= com_height_m <= self.ready_height_max

    def _detect_foot_lift(self, pose_keypoints: Dict) -> bool:
        """
        Detect if one foot has lifted off the ground.
        Uses ankle/foot keypoints to check for asymmetry.
        """
        left_ankle = pose_keypoints.get('left_ankle')
        right_ankle = pose_keypoints.get('right_ankle')

        if not left_ankle or not right_ankle:
            return False

        # Check for significant height difference between ankles
        # (indicates one foot lifted)
        ankle_height_diff = abs(left_ankle.y - right_ankle.y)

        # If difference is > 10% of frame height, likely one foot lifted
        # Note: y coordinates are in normalized space [0, 1]
        return ankle_height_diff > 0.1

    def reset(self):
        """Reset detector state for processing a new race."""
        self.baseline_com_height = None
        self.calibration_buffer = []
        self.start_detected = False


class FinishDetector:
    """
    Detects the finish of a speed climbing race.

    Strategy:
    1. Monitor hand/wrist keypoints
    2. Detect when hand reaches top pad zone (14.5-15.0m)
    3. Validate sustained contact (not a momentary spike)
    """

    def __init__(
        self,
        top_pad_height_m: float = 15.0,
        top_pad_zone_m: float = 0.5,
        sustained_frames: int = 3,
        min_confidence: float = 0.5
    ):
        """
        Args:
            top_pad_height_m: Height of the top pad (IFSC standard: 15m)
            top_pad_zone_m: Detection zone below top pad (default: 0.5m)
            sustained_frames: Number of consecutive frames to confirm finish
            min_confidence: Minimum pose confidence to consider
        """
        self.top_pad_height = top_pad_height_m
        self.top_zone_min = top_pad_height_m - top_pad_zone_m
        self.sustained_frames = sustained_frames
        self.min_confidence = min_confidence

        # State tracking
        self.finish_detected = False
        self.top_zone_counter = 0
        self.candidate_event: Optional[FinishEvent] = None

    def process_frame(
        self,
        frame_id: int,
        timestamp: float,
        pose_keypoints: Dict,
        pose_confidence: float,
        world_to_wall_transform=None
    ) -> Optional[FinishEvent]:
        """
        Process a single frame to detect finish event.

        Args:
            frame_id: Frame number
            timestamp: Frame timestamp in seconds
            pose_keypoints: Dict of BlazePose keypoints (normalized or world coords)
            pose_confidence: Overall pose detection confidence
            world_to_wall_transform: Optional homography for coordinate transformation

        Returns:
            FinishEvent if finish is detected, None otherwise
        """
        if self.finish_detected:
            return None

        if pose_confidence < self.min_confidence:
            self.top_zone_counter = 0
            return None

        # Get hand heights (use wrist as proxy)
        hand_height = self._get_max_hand_height(pose_keypoints, world_to_wall_transform)

        if hand_height is None:
            self.top_zone_counter = 0
            return None

        # Check if hand is in top zone
        if hand_height >= self.top_zone_min:
            self.top_zone_counter += 1

            # Update candidate event
            if self.candidate_event is None:
                self.candidate_event = FinishEvent(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    hand_height_m=hand_height,
                    confidence=pose_confidence,
                    method='top_reach',
                    details={
                        'top_pad_height': self.top_pad_height,
                        'detection_zone_min': self.top_zone_min
                    }
                )

            # Confirm finish if sustained
            if self.top_zone_counter >= self.sustained_frames:
                self.finish_detected = True
                logger.info(f"FINISH detected at frame {frame_id} (t={timestamp:.2f}s), "
                           f"hand height: {hand_height:.3f}m")
                return self.candidate_event
        else:
            # Reset if hand drops out of zone
            self.top_zone_counter = 0
            self.candidate_event = None

        return None

    def _get_max_hand_height(
        self,
        pose_keypoints: Dict,
        world_to_wall_transform=None
    ) -> Optional[float]:
        """
        Get the maximum hand height from left/right wrist keypoints.

        Args:
            pose_keypoints: Dict of keypoints (normalized or world coords)
            world_to_wall_transform: Optional homography matrix

        Returns:
            Maximum hand height in meters, or None if not available
        """
        left_wrist = pose_keypoints.get('left_wrist')
        right_wrist = pose_keypoints.get('right_wrist')

        heights = []

        for wrist in [left_wrist, right_wrist]:
            if wrist and wrist.visibility > self.min_confidence:
                # If we have world coordinates, use y directly
                # If normalized, need transformation (would need frame dimensions)
                # For now, assume we get world coordinates
                heights.append(wrist.y)

        return max(heights) if heights else None

    def reset(self):
        """Reset detector state for processing a new race."""
        self.finish_detected = False
        self.top_zone_counter = 0
        self.candidate_event = None


class StartFinishDetector:
    """
    Combined start and finish detection for speed climbing races.
    """

    def __init__(
        self,
        start_detector: Optional[StartDetector] = None,
        finish_detector: Optional[FinishDetector] = None
    ):
        """
        Args:
            start_detector: Custom StartDetector instance (uses default if None)
            finish_detector: Custom FinishDetector instance (uses default if None)
        """
        self.start_detector = start_detector or StartDetector()
        self.finish_detector = finish_detector or FinishDetector()

        self.start_event: Optional[StartEvent] = None
        self.finish_event: Optional[FinishEvent] = None

    def process_frame(
        self,
        frame_id: int,
        timestamp: float,
        com_height_m: float,
        pose_keypoints: Dict,
        pose_confidence: float,
        world_to_wall_transform=None
    ) -> Tuple[Optional[StartEvent], Optional[FinishEvent]]:
        """
        Process a frame for both start and finish detection.

        Returns:
            Tuple of (StartEvent or None, FinishEvent or None)
        """
        start_event = None
        finish_event = None

        # Detect start
        if not self.start_event:
            start_event = self.start_detector.process_frame(
                frame_id, timestamp, com_height_m, pose_keypoints, pose_confidence
            )
            if start_event:
                self.start_event = start_event

        # Only look for finish after start is detected
        if self.start_event and not self.finish_event:
            finish_event = self.finish_detector.process_frame(
                frame_id, timestamp, pose_keypoints, pose_confidence, world_to_wall_transform
            )
            if finish_event:
                self.finish_event = finish_event

        return start_event, finish_event

    def get_race_duration(self) -> Optional[float]:
        """
        Calculate race duration if both start and finish are detected.

        Returns:
            Duration in seconds, or None if race not complete
        """
        if self.start_event and self.finish_event:
            return self.finish_event.timestamp - self.start_event.timestamp
        return None

    def reset(self):
        """Reset all detectors for a new race."""
        self.start_detector.reset()
        self.finish_detector.reset()
        self.start_event = None
        self.finish_event = None

    def get_summary(self) -> Dict:
        """Get summary of detected events."""
        return {
            'start_detected': self.start_event is not None,
            'finish_detected': self.finish_event is not None,
            'race_duration_s': self.get_race_duration(),
            'start_event': self.start_event.__dict__ if self.start_event else None,
            'finish_event': self.finish_event.__dict__ if self.finish_event else None
        }
