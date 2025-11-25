"""
Athlete-Centric Processing Pipeline for Speed Climbing.

This module implements a robust approach that:
1. Uses athlete pose as the PRIMARY reference (not holds)
2. Handles dual-lane racing (left and right climbers)
3. Detects actual race segments within video (not just video start/end)
4. Estimates wall position relative to camera
5. Calculates cumulative climbing distance
6. Uses hold-based anchoring to compensate for camera tracking

Key insight: Athletes are ALWAYS visible in the video, while holds may be
occluded by the climber or camera movement. Therefore, we track the athlete
and use occasional hold sightings to calibrate absolute position.

Camera Motion Compensation Strategy:
- When camera follows athlete UP, the athlete appears relatively stationary
- But HOLDS appear to move DOWN in the frame
- By tracking hold movement, we can estimate camera motion
- This allows us to recover the true climbing distance
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from collections import deque

from speed_climbing.core.settings import (
    IFSC_STANDARDS,
    ATHLETE_CENTRIC_CONFIG,
    DEFAULT_PROCESSING_CONFIG
)

logger = logging.getLogger(__name__)


class RacePhase(Enum):
    """Current phase of the race video."""
    PRE_RACE = "pre_race"        # Before start - athlete warming up or waiting
    READY = "ready"              # Athlete in start position, waiting for signal
    RACING = "racing"            # Active climbing
    FINISHED = "finished"        # Race completed
    FALL = "fall"                # Athlete fell during race
    POST_RACE = "post_race"      # After race finished


@dataclass
class LaneState:
    """Tracks state for a single lane (one athlete)."""
    lane: str  # 'left' or 'right'

    # Current phase
    phase: RacePhase = RacePhase.PRE_RACE

    # Calibration
    scale_pixels_per_meter: Optional[float] = None
    scale_confidence: float = 0.0
    reference_frame_id: Optional[int] = None

    # Position tracking (in meters)
    current_height_m: float = 0.0
    cumulative_distance_m: float = 0.0
    max_height_m: float = 0.0

    # Velocity (m/s)
    current_velocity_m_s: float = 0.0
    max_velocity_m_s: float = 0.0

    # Event frames
    start_frame: Optional[int] = None
    finish_frame: Optional[int] = None
    fall_frame: Optional[int] = None

    # Baseline for start detection
    baseline_com_normalized: Optional[float] = None
    stillness_counter: int = 0

    # History for smoothing and velocity calculation
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Frame-to-frame tracking
    last_com_pixel_y: Optional[float] = None
    last_frame_id: Optional[int] = None


@dataclass
class FrameResult:
    """Result of processing a single frame for one lane."""
    frame_id: int
    timestamp: float
    lane: str

    # Pose data (normalized coordinates)
    has_detection: bool
    com_normalized: Optional[Tuple[float, float]] = None
    pose_confidence: float = 0.0

    # World coordinates (if calibrated)
    height_m: Optional[float] = None
    velocity_m_s: Optional[float] = None
    cumulative_distance_m: Optional[float] = None

    # State
    phase: RacePhase = RacePhase.PRE_RACE
    is_start_frame: bool = False
    is_finish_frame: bool = False
    is_fall_frame: bool = False

    # Quality metrics
    calibration_confidence: float = 0.0


class RaceSegmentDetector:
    """
    Detects the actual race segment within a video.

    Videos often contain:
    - Pre-race footage (athletes warming up, walking around)
    - Waiting phase (athletes in start position, waiting for signal)
    - The actual race
    - Post-race footage (celebration, replays)

    This class identifies the actual racing segment.
    """

    def __init__(self, config: Dict = None):
        cfg = config or ATHLETE_CENTRIC_CONFIG
        self.stillness_threshold = cfg["stillness_threshold"]
        self.stillness_min_frames = cfg["stillness_min_frames"]
        self.com_standing_min = cfg["com_standing_height_min_m"]
        self.com_standing_max = cfg["com_standing_height_max_m"]
        self.start_movement_threshold = cfg["start_movement_threshold_m"]

    def analyze_pose_sequence(
        self,
        pose_sequence: List[Dict],
        scale_pixels_per_meter: float = None
    ) -> Dict[str, Any]:
        """
        Analyze a sequence of poses to find race boundaries.

        Args:
            pose_sequence: List of pose results with COM positions
            scale_pixels_per_meter: Optional scale factor

        Returns:
            Dict with start_frame, ready_frame, finish_frame, etc.
        """
        if not pose_sequence:
            return {"error": "Empty pose sequence"}

        n_frames = len(pose_sequence)

        # Extract COM y positions (normalized, lower y = higher on screen = higher on wall)
        # Note: In image coordinates, y=0 is top, y=1 is bottom
        # For climbing, we need to invert: climbing UP means y decreases
        com_y_values = []
        for pose in pose_sequence:
            if pose.get('has_detection') and pose.get('com'):
                com_y_values.append(pose['com']['y'])
            else:
                com_y_values.append(None)

        # Calculate frame-to-frame movement
        movements = self._calculate_movements(com_y_values)

        # Find stillness regions (potential ready positions)
        stillness_regions = self._find_stillness_regions(movements)

        # Find the race start (first significant upward movement after stillness)
        start_info = self._find_race_start(com_y_values, movements, stillness_regions)

        # Find the race finish (COM reaches top, or stops moving at top)
        finish_info = self._find_race_finish(com_y_values, movements, start_info)

        return {
            "total_frames": n_frames,
            "ready_frame": start_info.get("ready_frame"),
            "start_frame": start_info.get("start_frame"),
            "finish_frame": finish_info.get("finish_frame"),
            "fall_frame": finish_info.get("fall_frame"),
            "stillness_regions": stillness_regions,
            "is_complete_race": start_info.get("start_frame") is not None and
                               finish_info.get("finish_frame") is not None,
            "analysis_confidence": self._calculate_confidence(start_info, finish_info)
        }

    def _calculate_movements(self, com_y_values: List[Optional[float]]) -> List[float]:
        """Calculate frame-to-frame movements."""
        movements = [0.0]  # First frame has no movement
        for i in range(1, len(com_y_values)):
            if com_y_values[i] is not None and com_y_values[i-1] is not None:
                # Negative delta means moving UP (y decreases)
                movements.append(com_y_values[i-1] - com_y_values[i])
            else:
                movements.append(0.0)
        return movements

    def _find_stillness_regions(
        self,
        movements: List[float]
    ) -> List[Tuple[int, int]]:
        """Find regions where athlete is relatively still."""
        regions = []
        start_idx = None

        for i, mov in enumerate(movements):
            is_still = abs(mov) < self.stillness_threshold

            if is_still and start_idx is None:
                start_idx = i
            elif not is_still and start_idx is not None:
                if i - start_idx >= self.stillness_min_frames:
                    regions.append((start_idx, i - 1))
                start_idx = None

        # Handle region at end
        if start_idx is not None and len(movements) - start_idx >= self.stillness_min_frames:
            regions.append((start_idx, len(movements) - 1))

        return regions

    def _find_race_start(
        self,
        com_y_values: List[Optional[float]],
        movements: List[float],
        stillness_regions: List[Tuple[int, int]]
    ) -> Dict:
        """Find the frame where the race starts."""
        result = {"ready_frame": None, "start_frame": None}

        if not stillness_regions:
            # No clear stillness - look for first sustained upward movement
            for i in range(len(movements)):
                if self._is_sustained_upward_movement(movements, i):
                    result["start_frame"] = i
                    break
            return result

        # Look for stillness followed by sudden upward movement
        for start_idx, end_idx in stillness_regions:
            # Check if there's upward movement after this stillness
            search_start = end_idx + 1
            search_end = min(end_idx + 30, len(movements))  # Look 30 frames ahead

            for i in range(search_start, search_end):
                if self._is_sustained_upward_movement(movements, i):
                    result["ready_frame"] = start_idx
                    result["start_frame"] = i
                    return result

        return result

    def _is_sustained_upward_movement(
        self,
        movements: List[float],
        start_idx: int,
        window: int = 5
    ) -> bool:
        """Check if there's sustained upward movement starting at index."""
        if start_idx + window > len(movements):
            return False

        total_movement = sum(movements[start_idx:start_idx + window])
        # Upward movement = positive (since we calculated y_prev - y_curr)
        return total_movement > self.stillness_threshold * window * 2

    def _find_race_finish(
        self,
        com_y_values: List[Optional[float]],
        movements: List[float],
        start_info: Dict
    ) -> Dict:
        """Find the frame where the race finishes."""
        result = {"finish_frame": None, "fall_frame": None}

        start_frame = start_info.get("start_frame", 0)
        if start_frame is None:
            start_frame = 0

        # Look for:
        # 1. COM reaches very low y (top of wall) and stops
        # 2. OR sudden downward movement (fall)

        min_y = 1.0
        min_y_frame = start_frame

        for i in range(start_frame, len(com_y_values)):
            if com_y_values[i] is not None:
                if com_y_values[i] < min_y:
                    min_y = com_y_values[i]
                    min_y_frame = i

                # Check for fall: sudden large downward movement
                if i > 0 and movements[i] < -0.1:  # Large downward movement
                    # Verify it's sustained
                    if self._is_sustained_downward_movement(movements, i):
                        result["fall_frame"] = i
                        return result

        # If we reached a very low y position, that's likely the finish
        # "Low y" in normalized coords means high on screen = high on wall
        if min_y < 0.15:  # Top 15% of frame
            result["finish_frame"] = min_y_frame

        return result

    def _is_sustained_downward_movement(
        self,
        movements: List[float],
        start_idx: int,
        window: int = 5
    ) -> bool:
        """Check for sustained downward movement (fall)."""
        if start_idx + window > len(movements):
            return False

        total_movement = sum(movements[start_idx:start_idx + window])
        return total_movement < -self.stillness_threshold * window * 3

    def _calculate_confidence(self, start_info: Dict, finish_info: Dict) -> float:
        """Calculate confidence in the detection."""
        confidence = 0.5  # Base confidence

        if start_info.get("ready_frame") is not None:
            confidence += 0.2
        if start_info.get("start_frame") is not None:
            confidence += 0.15
        if finish_info.get("finish_frame") is not None:
            confidence += 0.15

        return min(1.0, confidence)


class WallReferenceEstimator:
    """
    Estimates wall position and scale relative to camera.

    Key insight: The camera follows the athlete, so the visible portion
    of the wall changes. We need to track:
    1. Which part of the wall is currently visible
    2. The pixel-to-meter scale factor
    """

    def __init__(self, route_map: Dict = None, config: Dict = None):
        self.route_map = route_map
        cfg = config or ATHLETE_CENTRIC_CONFIG
        self.wall_height = IFSC_STANDARDS["WALL_HEIGHT_M"]
        self.wall_width = IFSC_STANDARDS["WALL_WIDTH_M"]
        self.camera_view_height = cfg["camera_view_height_m"]

        # Reference heights for scale estimation
        self.athlete_heights = cfg["athlete_reference_heights"]

    def estimate_scale_from_athlete(
        self,
        pose_result: Dict,
        frame_height: int,
        gender: str = "male"
    ) -> Optional[Dict]:
        """
        Estimate scale using athlete body proportions.

        When athlete is standing on start pad (COM at ~1m height),
        we can estimate scale from their body size.

        Args:
            pose_result: Pose detection result
            frame_height: Frame height in pixels
            gender: 'male' or 'female' for reference height

        Returns:
            Dict with scale_pixels_per_meter, confidence, method
        """
        if not pose_result.get('has_detection'):
            return None

        keypoints = pose_result.get('keypoints', {})

        # Method 1: Use shoulder-to-hip distance (torso length)
        scale_torso = self._estimate_from_torso(keypoints, frame_height, gender)

        # Method 2: Use full body height if visible
        scale_body = self._estimate_from_full_body(keypoints, frame_height, gender)

        # Method 3: Use arm span if visible
        scale_arms = self._estimate_from_arm_span(keypoints, frame_height, gender)

        # Combine estimates with confidence weighting
        scales = [s for s in [scale_torso, scale_body, scale_arms] if s is not None]

        if not scales:
            return None

        # Weight by confidence and average
        total_weight = sum(s['confidence'] for s in scales)
        if total_weight == 0:
            return None

        weighted_scale = sum(
            s['scale'] * s['confidence'] for s in scales
        ) / total_weight

        avg_confidence = total_weight / len(scales)

        return {
            'scale_pixels_per_meter': weighted_scale,
            'confidence': avg_confidence,
            'method': 'athlete_proportions',
            'components': scales
        }

    def _estimate_from_torso(
        self,
        keypoints: Dict,
        frame_height: int,
        gender: str
    ) -> Optional[Dict]:
        """Estimate scale from shoulder-to-hip distance."""
        left_shoulder = keypoints.get('left_shoulder')
        right_shoulder = keypoints.get('right_shoulder')
        left_hip = keypoints.get('left_hip')
        right_hip = keypoints.get('right_hip')

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        # Average shoulder and hip positions
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2

        # Torso length in pixels
        torso_pixels = abs(hip_y - shoulder_y) * frame_height

        if torso_pixels < 10:  # Too small
            return None

        # Expected torso length in meters (roughly 40% of body height)
        body_height = self.athlete_heights[f"{gender}_average_m"]
        expected_torso_m = body_height * 0.40

        scale = torso_pixels / expected_torso_m

        # Confidence based on visibility of keypoints
        conf = min(
            left_shoulder.visibility,
            right_shoulder.visibility,
            left_hip.visibility,
            right_hip.visibility
        )

        return {
            'scale': scale,
            'confidence': conf * 0.8,  # Torso is moderately reliable
            'method': 'torso'
        }

    def _estimate_from_full_body(
        self,
        keypoints: Dict,
        frame_height: int,
        gender: str
    ) -> Optional[Dict]:
        """Estimate scale from head-to-foot distance."""
        nose = keypoints.get('nose')
        left_ankle = keypoints.get('left_ankle')
        right_ankle = keypoints.get('right_ankle')

        if not nose or not (left_ankle or right_ankle):
            return None

        # Use lower ankle (one on ground during standing)
        ankle_y = max(
            left_ankle.y if left_ankle else 0,
            right_ankle.y if right_ankle else 0
        )

        body_pixels = (ankle_y - nose.y) * frame_height

        if body_pixels < 50:  # Too small
            return None

        # Expected body height
        expected_height_m = self.athlete_heights[f"{gender}_average_m"]

        scale = body_pixels / expected_height_m

        # Confidence based on keypoint visibility
        conf = min(
            nose.visibility,
            (left_ankle.visibility if left_ankle else 0 +
             right_ankle.visibility if right_ankle else 0) / 2
        )

        return {
            'scale': scale,
            'confidence': conf * 0.9,  # Full body is most reliable when visible
            'method': 'full_body'
        }

    def _estimate_from_arm_span(
        self,
        keypoints: Dict,
        frame_height: int,
        gender: str
    ) -> Optional[Dict]:
        """Estimate scale from arm span (wingspan ≈ height for most people)."""
        left_wrist = keypoints.get('left_wrist')
        right_wrist = keypoints.get('right_wrist')

        if not left_wrist or not right_wrist:
            return None

        # Arm span in pixels (horizontal distance)
        # Note: This is less reliable during climbing due to arm positions
        arm_span_x = abs(right_wrist.x - left_wrist.x) * frame_height  # Use height for aspect ratio

        if arm_span_x < 30:  # Arms probably not extended
            return None

        # Arm span ≈ body height
        expected_span_m = self.athlete_heights[f"{gender}_average_m"]

        scale = arm_span_x / expected_span_m

        conf = min(left_wrist.visibility, right_wrist.visibility)

        return {
            'scale': scale,
            'confidence': conf * 0.5,  # Less reliable during climbing
            'method': 'arm_span'
        }

    def estimate_visible_wall_region(
        self,
        athlete_com_height_m: float
    ) -> Tuple[float, float]:
        """
        Estimate which part of the wall is visible in current frame.

        The camera typically follows the athlete, keeping them roughly
        centered in the frame.

        Args:
            athlete_com_height_m: Current COM height in meters

        Returns:
            Tuple of (view_bottom_m, view_top_m)
        """
        half_view = self.camera_view_height / 2

        view_bottom = max(0, athlete_com_height_m - half_view)
        view_top = min(self.wall_height, athlete_com_height_m + half_view)

        return (view_bottom, view_top)

    def get_expected_holds_in_view(
        self,
        athlete_com_height_m: float,
        lane: str = 'left'
    ) -> List[Dict]:
        """
        Get list of holds expected to be visible in current frame.

        Args:
            athlete_com_height_m: Current COM height
            lane: 'left' or 'right'

        Returns:
            List of hold dicts that should be visible
        """
        if not self.route_map:
            return []

        view_bottom, view_top = self.estimate_visible_wall_region(athlete_com_height_m)

        # Filter holds by lane and height
        lane_prefix = 'SN' if lane == 'left' else 'DX'

        visible_holds = [
            hold for hold in self.route_map.get('holds', [])
            if hold.get('panel', '').startswith(lane_prefix)
            and view_bottom <= hold.get('wall_y_m', 0) <= view_top
        ]

        return visible_holds


class RelativeMotionTracker:
    """
    Tracks athlete position using frame-to-frame relative motion.

    KEY INSIGHT for moving camera scenarios:
    - When camera follows athlete UP, athlete appears to move DOWN in frame
    - We track CUMULATIVE UPWARD distance separately from position
    - Cumulative distance is the sum of all upward movements (ignoring downward)
    - This gives us the actual climbing distance regardless of camera motion
    """

    def __init__(self, config: Dict = None):
        cfg = config or ATHLETE_CENTRIC_CONFIG

        # Smoothing parameters
        self.position_smoothing_window = 5
        self.velocity_smoothing_window = 3

        # Physical constraints
        self.max_velocity_m_s = 3.5  # Maximum climbing velocity
        self.max_acceleration_m_s2 = 15.0  # Maximum acceleration

        # Noise filtering
        self.min_movement_threshold = 0.01  # Ignore movements smaller than 1cm

    def update(
        self,
        state: LaneState,
        frame_id: int,
        timestamp: float,
        com_normalized: Tuple[float, float],
        frame_height: int,
        fps: float
    ) -> Dict:
        """
        Update position tracking with new frame data.

        For MOVING CAMERA scenarios:
        - cumulative_distance_m: Sum of all upward movements (RELIABLE)
        - current_height_m: Estimated absolute height (LESS RELIABLE)

        Args:
            state: Current lane state
            frame_id: Frame number
            timestamp: Frame timestamp
            com_normalized: COM position (x, y) in normalized coords [0, 1]
            frame_height: Frame height in pixels
            fps: Video frame rate

        Returns:
            Dict with updated position, velocity, distance
        """
        result = {
            'height_m': None,
            'velocity_m_s': None,
            'distance_delta_m': 0.0,
            'is_valid': False
        }

        if state.scale_pixels_per_meter is None:
            return result

        # Convert normalized Y to pixel Y
        com_pixel_y = com_normalized[1] * frame_height

        # Calculate frame-to-frame displacement
        if state.last_com_pixel_y is not None and state.last_frame_id is not None:
            frame_delta = frame_id - state.last_frame_id
            if frame_delta > 0:
                # Pixel displacement (note: y increases downward in image)
                pixel_delta = state.last_com_pixel_y - com_pixel_y  # Positive = moved up

                # Convert to meters
                meter_delta = pixel_delta / state.scale_pixels_per_meter

                # Calculate velocity
                time_delta = frame_delta / fps
                velocity = meter_delta / time_delta if time_delta > 0 else 0

                # Apply physical constraints
                velocity = np.clip(velocity, -self.max_velocity_m_s, self.max_velocity_m_s)

                # CRITICAL: For moving camera, only accumulate UPWARD movement
                # Downward "movement" is often just the camera moving up
                if state.phase == RacePhase.RACING:
                    if meter_delta > self.min_movement_threshold:
                        # Genuine upward movement - add to cumulative distance
                        state.cumulative_distance_m += meter_delta

                        # Update current height (this may be inaccurate due to camera)
                        state.current_height_m += meter_delta

                        # Track maximum height
                        if state.current_height_m > state.max_height_m:
                            state.max_height_m = state.current_height_m

                        result['distance_delta_m'] = meter_delta

                    elif meter_delta < -self.min_movement_threshold:
                        # Downward movement in frame - could be:
                        # 1. Camera moving up (most common)
                        # 2. Athlete actually moving down briefly
                        # 3. Noise

                        # Don't subtract from cumulative distance
                        # But do update current height estimate
                        # (though this makes it less accurate with moving camera)

                        # For now, assume most downward is camera motion
                        # Only update height if the downward movement is small
                        if abs(meter_delta) < 0.3:  # Less than 30cm
                            state.current_height_m += meter_delta
                            state.current_height_m = max(0.5, state.current_height_m)

                # Update velocity tracking
                state.current_velocity_m_s = velocity
                if velocity > 0 and velocity > state.max_velocity_m_s:
                    state.max_velocity_m_s = velocity

                # Add to history for smoothing
                state.position_history.append(state.cumulative_distance_m)  # Track cumulative, not position
                state.velocity_history.append(velocity)

                result['height_m'] = state.current_height_m
                result['velocity_m_s'] = velocity
                result['is_valid'] = True

        # Update state for next frame
        state.last_com_pixel_y = com_pixel_y
        state.last_frame_id = frame_id

        return result

    def get_smoothed_velocity(self, state: LaneState) -> float:
        """Get smoothed velocity from history."""
        if len(state.velocity_history) < 2:
            return state.current_velocity_m_s

        return float(np.mean(list(state.velocity_history)))

    def get_smoothed_height(self, state: LaneState) -> float:
        """Get smoothed height from history."""
        if len(state.position_history) < 2:
            return state.current_height_m

        return float(np.mean(list(state.position_history)))


@dataclass
class HoldAnchor:
    """Represents a tracked hold used for camera motion estimation."""
    hold_id: Optional[int]
    world_y_m: float  # Known position on wall (from route map)
    pixel_y: float  # Current pixel position in frame
    frame_id: int
    confidence: float


class HoldAnchoringSystem:
    """
    Track holds to estimate camera motion and absolute wall position.

    Key insight for moving camera scenarios:
    - When camera follows athlete UP, holds appear to move DOWN in frame
    - By tracking hold positions over time, we can estimate camera offset
    - This allows us to calculate the TRUE climbing distance

    The system works by:
    1. Detecting holds in each frame (when visible)
    2. Matching detected holds to known positions from route map
    3. Tracking how hold pixel positions change over time
    4. Using this to estimate cumulative camera motion
    5. Adding camera motion to relative athlete motion = true distance
    """

    def __init__(self, route_map: Dict = None, config: Dict = None):
        self.route_map = route_map
        cfg = config or ATHLETE_CENTRIC_CONFIG
        self.wall_height = IFSC_STANDARDS["WALL_HEIGHT_M"]

        # Hold tracking state per lane
        self.tracked_holds: Dict[str, List[HoldAnchor]] = {
            'left': [],
            'right': []
        }

        # Camera offset estimation per lane
        # This represents how much the camera has moved UP from start
        self.camera_offset_m: Dict[str, float] = {
            'left': 0.0,
            'right': 0.0
        }

        # Previous frame hold positions for motion calculation
        self.prev_hold_positions: Dict[str, Dict[int, float]] = {
            'left': {},  # hold_id -> pixel_y
            'right': {}
        }

        # Estimated scale (pixels per meter) for camera motion calculation
        self.scale_estimate: Dict[str, float] = {
            'left': None,
            'right': None
        }

        # History of camera offsets for smoothing
        self.camera_offset_history: Dict[str, deque] = {
            'left': deque(maxlen=30),
            'right': deque(maxlen=30)
        }

        # Confidence in camera motion estimation
        self.motion_confidence: Dict[str, float] = {
            'left': 0.0,
            'right': 0.0
        }

    def update_from_holds(
        self,
        lane: str,
        detected_holds: List[Any],
        frame_id: int,
        frame_height: int,
        scale_pixels_per_meter: float = None
    ) -> Dict[str, Any]:
        """
        Update camera motion estimate from detected holds.

        Args:
            lane: 'left' or 'right'
            detected_holds: List of DetectedHold objects
            frame_id: Current frame number
            frame_height: Frame height in pixels
            scale_pixels_per_meter: Scale factor (if known)

        Returns:
            Dict with camera_motion_m, confidence, anchors_used
        """
        if scale_pixels_per_meter:
            self.scale_estimate[lane] = scale_pixels_per_meter

        result = {
            'camera_motion_delta_m': 0.0,
            'total_camera_offset_m': self.camera_offset_m[lane],
            'confidence': 0.0,
            'anchors_used': 0
        }

        if not self.route_map or not detected_holds:
            return result

        # Get route holds for matching
        lane_prefix = 'SN' if lane == 'left' else 'DX'
        route_holds = {
            h['hold_num']: h for h in self.route_map.get('holds', [])
            if h.get('panel', '').startswith(lane_prefix)
        }

        if not route_holds:
            return result

        # Match detected holds to route map
        matched_holds = []
        for hold in detected_holds:
            if hold.hold_num is not None and hold.hold_num in route_holds:
                route_hold = route_holds[hold.hold_num]
                matched_holds.append(HoldAnchor(
                    hold_id=hold.hold_num,
                    world_y_m=route_hold.get('wall_y_m', 0),
                    pixel_y=hold.pixel_y,
                    frame_id=frame_id,
                    confidence=hold.confidence
                ))

        if not matched_holds:
            # No matched holds - can't update camera motion
            return result

        # Calculate camera motion from hold movement
        camera_motion_pixels = 0.0
        motion_samples = []
        prev_positions = self.prev_hold_positions[lane]

        for anchor in matched_holds:
            if anchor.hold_id in prev_positions:
                # Hold was seen in previous frame
                prev_y = prev_positions[anchor.hold_id]
                current_y = anchor.pixel_y

                # Hold moving DOWN in frame (increasing y) = camera moving UP
                pixel_delta = current_y - prev_y

                if abs(pixel_delta) < frame_height * 0.3:  # Sanity check
                    motion_samples.append(pixel_delta)

        # Update previous positions for next frame
        self.prev_hold_positions[lane] = {
            anchor.hold_id: anchor.pixel_y for anchor in matched_holds
        }

        if motion_samples:
            # Average the motion samples
            camera_motion_pixels = np.median(motion_samples)

            # Convert to meters if we have scale
            scale = self.scale_estimate[lane]
            if scale and scale > 0:
                # Positive pixel delta = hold moved down = camera moved up
                camera_motion_m = camera_motion_pixels / scale
                self.camera_offset_m[lane] += camera_motion_m

                # Add to history for smoothing
                self.camera_offset_history[lane].append(camera_motion_m)

                # Calculate confidence based on number of samples
                result['confidence'] = min(1.0, len(motion_samples) / 3.0)
                result['camera_motion_delta_m'] = camera_motion_m
                self.motion_confidence[lane] = result['confidence']

        # Update tracked holds
        self.tracked_holds[lane] = matched_holds

        result['total_camera_offset_m'] = self.camera_offset_m[lane]
        result['anchors_used'] = len(matched_holds)

        return result

    def estimate_absolute_height(
        self,
        lane: str,
        relative_height_m: float,
        detected_holds: List[Any] = None,
        frame_height: int = None
    ) -> Tuple[float, float]:
        """
        Estimate absolute height on wall using hold anchors.

        When holds are visible, we can directly calculate athlete height
        by comparing athlete position to known hold positions.

        Args:
            lane: 'left' or 'right'
            relative_height_m: Height estimated from relative motion
            detected_holds: Currently visible holds (optional)
            frame_height: Frame height in pixels (optional)

        Returns:
            Tuple of (estimated_height_m, confidence)
        """
        # Method 1: Use camera offset
        # If camera has moved up X meters, athlete is at relative_height + X
        camera_offset = self.camera_offset_m[lane]
        height_from_offset = relative_height_m + camera_offset

        confidence = self.motion_confidence[lane]

        # Method 2: Direct calculation from visible holds (if available)
        if detected_holds and frame_height and self.tracked_holds[lane]:
            direct_height = self._estimate_from_hold_reference(
                lane, detected_holds, frame_height
            )
            if direct_height is not None:
                # Blend with offset-based estimate
                height_from_offset = (height_from_offset + direct_height) / 2
                confidence = min(1.0, confidence + 0.3)

        return height_from_offset, confidence

    def _estimate_from_hold_reference(
        self,
        lane: str,
        detected_holds: List[Any],
        frame_height: int
    ) -> Optional[float]:
        """
        Estimate athlete height directly from hold positions in frame.

        If we see a hold at known height Y meters, and the athlete is
        above/below that hold in the frame, we can calculate absolute height.
        """
        if not detected_holds or not self.route_map:
            return None

        # Find a high-confidence matched hold
        lane_prefix = 'SN' if lane == 'left' else 'DX'
        route_holds = {
            h['hold_num']: h for h in self.route_map.get('holds', [])
            if h.get('panel', '').startswith(lane_prefix)
        }

        for hold in detected_holds:
            if hold.hold_num is not None and hold.hold_num in route_holds:
                route_hold = route_holds[hold.hold_num]
                hold_world_y = route_hold.get('wall_y_m', 0)

                # We know this hold is at `hold_world_y` meters on the wall
                # Its pixel position tells us where that height is in the frame
                # This gives us a reference point for camera position

                return hold_world_y  # Return known height of visible hold

        return None

    def get_corrected_distance(
        self,
        lane: str,
        relative_distance_m: float
    ) -> float:
        """
        Get the corrected total climbing distance.

        The true distance = relative motion + camera motion

        Args:
            lane: 'left' or 'right'
            relative_distance_m: Distance calculated from relative motion alone

        Returns:
            Corrected distance in meters
        """
        camera_offset = self.camera_offset_m[lane]

        # The true climbing distance is what the athlete moved relative to frame
        # PLUS how much the camera moved up (which we missed in relative tracking)
        corrected = relative_distance_m + camera_offset

        logger.debug(f"Lane {lane}: relative={relative_distance_m:.2f}m + "
                    f"camera_offset={camera_offset:.2f}m = {corrected:.2f}m")

        return corrected

    def reset(self, lane: str = None):
        """Reset tracking state."""
        lanes = [lane] if lane else ['left', 'right']
        for l in lanes:
            self.tracked_holds[l] = []
            self.camera_offset_m[l] = 0.0
            self.prev_hold_positions[l] = {}
            self.camera_offset_history[l].clear()
            self.motion_confidence[l] = 0.0


class AthleteCentricPipeline:
    """
    Main pipeline for athlete-centric race analysis.

    This pipeline:
    1. Processes both lanes simultaneously
    2. Detects race start/finish using athlete movement
    3. Tracks climbing progress using relative motion
    4. Uses hold-based anchoring to compensate for camera tracking
    5. Outputs world coordinates and velocities
    """

    def __init__(
        self,
        route_map_path: str = None,
        config: Dict = None,
        enable_hold_anchoring: bool = True
    ):
        self.config = config or ATHLETE_CENTRIC_CONFIG
        self.enable_hold_anchoring = enable_hold_anchoring

        # Load route map if provided
        self.route_map = None
        if route_map_path:
            import json
            try:
                with open(route_map_path, 'r') as f:
                    self.route_map = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load route map: {e}")

        # Initialize components
        self.segment_detector = RaceSegmentDetector(self.config)
        self.wall_estimator = WallReferenceEstimator(self.route_map, self.config)
        self.motion_tracker = RelativeMotionTracker(self.config)

        # Hold anchoring for camera motion compensation
        self.hold_anchoring = HoldAnchoringSystem(self.route_map, self.config)

        # Lane states
        self.lane_states: Dict[str, LaneState] = {
            'left': LaneState(lane='left'),
            'right': LaneState(lane='right')
        }

        # Results storage
        self.frame_results: Dict[str, List[FrameResult]] = {
            'left': [],
            'right': []
        }

    def reset(self):
        """Reset pipeline for a new video."""
        self.lane_states = {
            'left': LaneState(lane='left'),
            'right': LaneState(lane='right')
        }
        self.frame_results = {
            'left': [],
            'right': []
        }
        # Reset hold anchoring
        self.hold_anchoring.reset()

    def process_frame(
        self,
        frame_id: int,
        timestamp: float,
        left_pose: Dict,
        right_pose: Dict,
        frame_shape: Tuple[int, int, int],
        fps: float = 30.0,
        left_holds: List[Any] = None,
        right_holds: List[Any] = None
    ) -> Dict[str, FrameResult]:
        """
        Process a single frame for both lanes.

        Args:
            frame_id: Frame number
            timestamp: Frame timestamp in seconds
            left_pose: Pose result for left lane athlete
            right_pose: Pose result for right lane athlete
            frame_shape: (height, width, channels)
            fps: Video frame rate
            left_holds: Detected holds for left lane (optional, for camera compensation)
            right_holds: Detected holds for right lane (optional, for camera compensation)

        Returns:
            Dict with 'left' and 'right' FrameResult objects
        """
        frame_height = frame_shape[0]

        results = {}
        holds_dict = {'left': left_holds, 'right': right_holds}

        for lane, pose in [('left', left_pose), ('right', right_pose)]:
            state = self.lane_states[lane]
            detected_holds = holds_dict.get(lane)

            # Update hold anchoring for camera motion compensation
            if self.enable_hold_anchoring and detected_holds:
                self.hold_anchoring.update_from_holds(
                    lane=lane,
                    detected_holds=detected_holds,
                    frame_id=frame_id,
                    frame_height=frame_height,
                    scale_pixels_per_meter=state.scale_pixels_per_meter
                )

            result = self._process_lane_frame(
                state, frame_id, timestamp, pose, frame_height, fps
            )

            # Apply camera motion correction to cumulative distance
            if self.enable_hold_anchoring:
                corrected_distance = self.hold_anchoring.get_corrected_distance(
                    lane, state.cumulative_distance_m
                )
                result.cumulative_distance_m = corrected_distance

            results[lane] = result
            self.frame_results[lane].append(result)

        return results

    def _process_lane_frame(
        self,
        state: LaneState,
        frame_id: int,
        timestamp: float,
        pose: Dict,
        frame_height: int,
        fps: float
    ) -> FrameResult:
        """Process a single frame for one lane."""

        result = FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            lane=state.lane,
            has_detection=pose.get('has_detection', False),
            phase=state.phase
        )

        if not pose.get('has_detection'):
            return result

        # Get COM position
        com = pose.get('com') or pose.get('keypoints', {}).get('COM')
        if not com:
            return result

        com_x = com.get('x', com.x if hasattr(com, 'x') else 0.5)
        com_y = com.get('y', com.y if hasattr(com, 'y') else 0.5)

        result.com_normalized = (com_x, com_y)
        result.pose_confidence = pose.get('overall_confidence', 0.5)

        # Phase-specific processing
        if state.phase == RacePhase.PRE_RACE:
            self._handle_pre_race(state, frame_id, timestamp, com_y, pose, frame_height, result)

        elif state.phase == RacePhase.READY:
            self._handle_ready(state, frame_id, timestamp, com_y, result)

        elif state.phase == RacePhase.RACING:
            self._handle_racing(state, frame_id, timestamp, com_y, pose, frame_height, fps, result)

        elif state.phase in [RacePhase.FINISHED, RacePhase.FALL]:
            self._handle_post_race(state, result)

        result.phase = state.phase
        result.calibration_confidence = state.scale_confidence

        return result

    def _handle_pre_race(
        self,
        state: LaneState,
        frame_id: int,
        timestamp: float,
        com_y: float,
        pose: Dict,
        frame_height: int,
        result: FrameResult
    ):
        """Handle pre-race phase: calibration and waiting for ready position."""

        # Try to calibrate scale from athlete body
        if state.scale_pixels_per_meter is None or state.scale_confidence < 0.7:
            scale_result = self.wall_estimator.estimate_scale_from_athlete(
                pose, frame_height, 'male'  # TODO: detect gender
            )
            if scale_result and scale_result['confidence'] > state.scale_confidence:
                state.scale_pixels_per_meter = scale_result['scale_pixels_per_meter']
                state.scale_confidence = scale_result['confidence']
                state.reference_frame_id = frame_id
                logger.info(f"Lane {state.lane}: Scale calibrated at {state.scale_pixels_per_meter:.1f} px/m "
                           f"(confidence: {state.scale_confidence:.2f})")

        # Track COM positions to detect movement pattern
        if not hasattr(state, 'com_history'):
            state.com_history = []
        state.com_history.append(com_y)

        # Keep only last 30 frames for analysis
        if len(state.com_history) > 30:
            state.com_history = state.com_history[-30:]

        # Check for stillness (athlete in ready position)
        if state.baseline_com_normalized is None:
            state.baseline_com_normalized = com_y
            state.stillness_counter = 1
        else:
            movement = abs(com_y - state.baseline_com_normalized)
            if movement < self.config["stillness_threshold"]:
                state.stillness_counter += 1
                if state.stillness_counter >= self.config["stillness_min_frames"]:
                    # Athlete is still - transition to READY
                    state.phase = RacePhase.READY
                    state.baseline_com_normalized = com_y
                    logger.info(f"Lane {state.lane}: Athlete in READY position at frame {frame_id}")
            else:
                # Movement detected - check if it's significant upward movement
                # This handles videos where the race has already started
                if len(state.com_history) >= 5:
                    recent_movement = state.com_history[-5][0] if isinstance(state.com_history[-5], tuple) else state.com_history[-5]
                    current = com_y
                    upward = recent_movement - current  # Positive = moving up (y decreases)

                    # If significant sustained upward movement, skip to RACING
                    if upward > self.config["stillness_threshold"] * 5:
                        logger.info(f"Lane {state.lane}: Detected ongoing race - skipping to RACING at frame {frame_id}")
                        state.phase = RacePhase.RACING
                        state.start_frame = max(0, frame_id - 5)  # Estimate start was 5 frames ago
                        state.current_height_m = self.config["com_standing_height_min_m"]
                        result.is_start_frame = True
                        return

                # Reset baseline
                state.baseline_com_normalized = com_y
                state.stillness_counter = 1

        # Fallback: if no stillness detected after many frames, transition anyway
        if frame_id > self.config["pre_race_min_frames"] * 2:
            logger.warning(f"Lane {state.lane}: No clear ready position found, starting tracking at frame {frame_id}")
            state.phase = RacePhase.READY
            state.baseline_com_normalized = com_y

    def _handle_ready(
        self,
        state: LaneState,
        frame_id: int,
        timestamp: float,
        com_y: float,
        result: FrameResult
    ):
        """Handle ready phase: waiting for start signal."""

        # Track ready frames
        if not hasattr(state, 'ready_frame_count'):
            state.ready_frame_count = 0
        state.ready_frame_count += 1

        if state.baseline_com_normalized is None:
            state.baseline_com_normalized = com_y
            return

        # Check for upward movement (start of race)
        # Note: In image coords, lower y = higher position
        upward_movement = state.baseline_com_normalized - com_y

        # Use a lower threshold for start detection
        start_threshold = self.config["stillness_threshold"] * 2  # 0.04 = 4% of frame

        if upward_movement > start_threshold:
            # Significant upward movement - race has started!
            state.phase = RacePhase.RACING
            state.start_frame = frame_id
            state.current_height_m = self.config["com_standing_height_min_m"]  # Initial COM height

            result.is_start_frame = True
            logger.info(f"Lane {state.lane}: RACE STARTED at frame {frame_id} (t={timestamp:.2f}s), "
                       f"upward_movement={upward_movement:.4f}")

        # Fallback: if stuck in ready for too long, start tracking
        elif state.ready_frame_count > 60:  # 2 seconds at 30fps
            logger.warning(f"Lane {state.lane}: Stuck in READY, forcing transition to RACING at frame {frame_id}")
            state.phase = RacePhase.RACING
            state.start_frame = frame_id
            state.current_height_m = self.config["com_standing_height_min_m"]
            result.is_start_frame = True

    def _handle_racing(
        self,
        state: LaneState,
        frame_id: int,
        timestamp: float,
        com_y: float,
        pose: Dict,
        frame_height: int,
        fps: float,
        result: FrameResult
    ):
        """Handle racing phase: track climbing progress."""

        # Initialize tracking state if needed
        if not hasattr(state, 'negative_velocity_count'):
            state.negative_velocity_count = 0
        if not hasattr(state, 'camera_offset'):
            state.camera_offset = 0.0  # Estimated camera movement

        # Update position using relative motion
        motion_result = self.motion_tracker.update(
            state,
            frame_id,
            timestamp,
            (0.5, com_y),  # X doesn't matter for height
            frame_height,
            fps
        )

        # Compensate for camera following athlete
        # Key insight: In a moving camera, athlete stays roughly centered
        # So we track CUMULATIVE upward movement, not absolute position
        if motion_result['distance_delta_m'] > 0:
            # Upward movement is real climbing
            state.negative_velocity_count = 0
        elif motion_result['velocity_m_s'] is not None and motion_result['velocity_m_s'] < -0.5:
            # Downward movement could be:
            # 1. Camera moving up (athlete appears to go down)
            # 2. Actual downward movement
            # 3. Fall

            # If we're still early in the race and haven't climbed much, it's probably camera movement
            if state.cumulative_distance_m < 5.0:
                # Assume it's camera movement, not tracking error
                # Estimate that the camera moved up
                state.camera_offset += abs(motion_result['velocity_m_s'] / fps)
                logger.debug(f"Lane {state.lane}: Camera movement compensation applied")
            else:
                state.negative_velocity_count += 1

        result.height_m = state.current_height_m
        result.velocity_m_s = motion_result['velocity_m_s']
        result.cumulative_distance_m = state.cumulative_distance_m

        # Check for finish based on cumulative distance (more reliable than absolute height)
        if state.cumulative_distance_m >= 13.0:  # 13m+ is close to finish
            # Use a lower threshold since we might miss some small movements
            state.phase = RacePhase.FINISHED
            state.finish_frame = frame_id
            result.is_finish_frame = True
            logger.info(f"Lane {state.lane}: RACE FINISHED at frame {frame_id} "
                       f"(t={timestamp:.2f}s, distance={state.cumulative_distance_m:.2f}m)")

        # Fall detection: more robust criteria
        # Only detect fall if:
        # 1. Athlete has climbed significantly (> 3m)
        # 2. AND sustained negative velocity for multiple frames (> 10 frames = 0.3s)
        # 3. AND large cumulative downward distance
        if (state.cumulative_distance_m > 3.0 and
            state.negative_velocity_count > 10 and
            motion_result['velocity_m_s'] is not None and
            motion_result['velocity_m_s'] < self.config["fall_velocity_threshold_m_s"]):

            # Additional check: see if the athlete has actually lost height
            if state.current_height_m < state.max_height_m - 2.0:  # Dropped more than 2m from max
                state.phase = RacePhase.FALL
                state.fall_frame = frame_id
                result.is_fall_frame = True
                logger.warning(f"Lane {state.lane}: FALL detected at frame {frame_id} "
                              f"(dropped from {state.max_height_m:.1f}m to {state.current_height_m:.1f}m)")

    def _handle_post_race(self, state: LaneState, result: FrameResult):
        """Handle post-race phase."""
        result.height_m = state.current_height_m
        result.cumulative_distance_m = state.cumulative_distance_m

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of race analysis for both lanes."""
        summary = {}

        for lane in ['left', 'right']:
            state = self.lane_states[lane]
            results = self.frame_results[lane]

            # Get camera motion compensation data
            camera_offset = self.hold_anchoring.camera_offset_m.get(lane, 0.0)
            corrected_distance = self.hold_anchoring.get_corrected_distance(
                lane, state.cumulative_distance_m
            )

            summary[lane] = {
                'phase': state.phase.value,
                'start_frame': state.start_frame,
                'finish_frame': state.finish_frame,
                'fall_frame': state.fall_frame,
                # Raw distance from relative motion tracking only
                'raw_distance_m': state.cumulative_distance_m,
                # Corrected distance including camera motion compensation
                'total_distance_m': corrected_distance,
                # Camera motion offset (how much camera moved up)
                'camera_offset_m': camera_offset,
                'max_height_m': state.max_height_m,
                'max_velocity_m_s': state.max_velocity_m_s,
                'scale_pixels_per_meter': state.scale_pixels_per_meter,
                'scale_confidence': state.scale_confidence,
                'frames_processed': len(results),
                'frames_with_detection': sum(1 for r in results if r.has_detection),
                # Hold anchoring confidence
                'camera_motion_confidence': self.hold_anchoring.motion_confidence.get(lane, 0.0),
            }

            # Calculate race duration if completed
            if state.start_frame is not None and state.finish_frame is not None:
                start_result = next((r for r in results if r.frame_id == state.start_frame), None)
                finish_result = next((r for r in results if r.frame_id == state.finish_frame), None)
                if start_result and finish_result:
                    summary[lane]['race_duration_s'] = finish_result.timestamp - start_result.timestamp

        return summary

    def export_time_series(self, lane: str) -> List[Dict]:
        """Export time series data for a lane."""
        return [
            {
                'frame_id': r.frame_id,
                'timestamp': r.timestamp,
                'has_detection': r.has_detection,
                'height_m': r.height_m,
                'velocity_m_s': r.velocity_m_s,
                'cumulative_distance_m': r.cumulative_distance_m,
                'phase': r.phase.value,
                'is_start': r.is_start_frame,
                'is_finish': r.is_finish_frame,
                'is_fall': r.is_fall_frame,
            }
            for r in self.frame_results[lane]
        ]
