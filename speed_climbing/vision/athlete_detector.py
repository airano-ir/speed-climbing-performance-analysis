"""
Athlete Count Detection for Speed Climbing Videos.

Detects whether a video contains one or two athletes based on
pose detection patterns in the left and right lanes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class AthleteDetectionResult:
    """Result of athlete count detection."""
    athlete_count: int  # 1 or 2
    primary_lane: str  # 'left', 'right', or 'both'
    left_detection_rate: float  # 0.0 to 1.0
    right_detection_rate: float  # 0.0 to 1.0
    confidence: float  # Detection confidence
    recommendation: str  # User-friendly recommendation

    @property
    def is_single_athlete(self) -> bool:
        return self.athlete_count == 1

    @property
    def is_dual_athlete(self) -> bool:
        return self.athlete_count == 2


class AthleteCountDetector:
    """
    Detect the number of athletes in a speed climbing video.

    Uses pose detection rates in left/right lanes to determine
    if the video contains one or two athletes.

    Thresholds:
    - If both lanes have >50% detection rate: 2 athletes
    - If only one lane has >30% detection rate: 1 athlete
    - Otherwise: unclear, default to analyzing the better lane
    """

    def __init__(
        self,
        dual_athlete_threshold: float = 0.40,
        single_athlete_threshold: float = 0.25,
        min_difference_ratio: float = 2.0
    ):
        """
        Args:
            dual_athlete_threshold: Min detection rate for both lanes to be "dual"
            single_athlete_threshold: Min detection rate for a lane to be "valid"
            min_difference_ratio: Min ratio between lanes for "single" classification
        """
        self.dual_athlete_threshold = dual_athlete_threshold
        self.single_athlete_threshold = single_athlete_threshold
        self.min_difference_ratio = min_difference_ratio

    def detect_from_pose_data(self, pose_data: Dict) -> AthleteDetectionResult:
        """
        Detect athlete count from pose data dictionary.

        Args:
            pose_data: Dictionary with 'metadata' and 'frames' keys

        Returns:
            AthleteDetectionResult with detection details
        """
        metadata = pose_data.get('metadata', {})
        frames = pose_data.get('frames', [])

        # Try to get detection rates from metadata first
        left_rate = metadata.get('detection_rate_left', None)
        right_rate = metadata.get('detection_rate_right', None)

        # If not in metadata, calculate from frames
        if left_rate is None or right_rate is None:
            left_rate, right_rate = self._calculate_detection_rates(frames)

        return self._analyze_detection_rates(left_rate, right_rate)

    def detect_from_frames(self, frames: List[Dict]) -> AthleteDetectionResult:
        """
        Detect athlete count from frame list.

        Args:
            frames: List of frame dictionaries with left_climber/right_climber

        Returns:
            AthleteDetectionResult with detection details
        """
        left_rate, right_rate = self._calculate_detection_rates(frames)
        return self._analyze_detection_rates(left_rate, right_rate)

    def _calculate_detection_rates(
        self,
        frames: List[Dict]
    ) -> Tuple[float, float]:
        """Calculate detection rates for left and right lanes."""
        if not frames:
            return 0.0, 0.0

        left_detections = 0
        right_detections = 0
        total_frames = len(frames)

        for frame in frames:
            left_climber = frame.get('left_climber')
            right_climber = frame.get('right_climber')

            if left_climber and left_climber.get('has_detection', False):
                left_detections += 1

            if right_climber and right_climber.get('has_detection', False):
                right_detections += 1

        left_rate = left_detections / total_frames if total_frames > 0 else 0.0
        right_rate = right_detections / total_frames if total_frames > 0 else 0.0

        return left_rate, right_rate

    def _analyze_detection_rates(
        self,
        left_rate: float,
        right_rate: float
    ) -> AthleteDetectionResult:
        """Analyze detection rates to determine athlete count."""

        # Case 1: Both lanes have good detection -> 2 athletes
        if (left_rate >= self.dual_athlete_threshold and
            right_rate >= self.dual_athlete_threshold):

            # Check if they're relatively balanced
            ratio = max(left_rate, right_rate) / (min(left_rate, right_rate) + 1e-6)

            if ratio < self.min_difference_ratio:
                return AthleteDetectionResult(
                    athlete_count=2,
                    primary_lane='both',
                    left_detection_rate=left_rate,
                    right_detection_rate=right_rate,
                    confidence=min(left_rate, right_rate),
                    recommendation="Two athletes detected. Select the lane you want to analyze."
                )

        # Case 2: Only one lane has significant detection -> 1 athlete
        max_rate = max(left_rate, right_rate)
        min_rate = min(left_rate, right_rate)

        if max_rate >= self.single_athlete_threshold:
            # Determine which lane is primary
            if left_rate > right_rate:
                primary_lane = 'left'
            else:
                primary_lane = 'right'

            # Check confidence based on difference
            if min_rate < self.single_athlete_threshold:
                # Clear single athlete
                confidence = max_rate
                athlete_count = 1
                recommendation = f"Single athlete detected in {primary_lane} lane."
            else:
                # Both have detection but one is dominant
                ratio = max_rate / (min_rate + 1e-6)
                if ratio >= self.min_difference_ratio:
                    confidence = max_rate * (1 - min_rate / max_rate)
                    athlete_count = 1
                    recommendation = f"Likely single athlete in {primary_lane} lane (other lane has noise)."
                else:
                    # Unclear, treat as dual
                    confidence = min(left_rate, right_rate) * 0.7
                    athlete_count = 2
                    primary_lane = 'both'
                    recommendation = "Two athletes detected. Select the lane you want to analyze."

            return AthleteDetectionResult(
                athlete_count=athlete_count,
                primary_lane=primary_lane,
                left_detection_rate=left_rate,
                right_detection_rate=right_rate,
                confidence=confidence,
                recommendation=recommendation
            )

        # Case 3: No clear detection in either lane
        return AthleteDetectionResult(
            athlete_count=0,
            primary_lane='unknown',
            left_detection_rate=left_rate,
            right_detection_rate=right_rate,
            confidence=0.0,
            recommendation="No clear athlete detection. Check video quality or upload a different file."
        )

    def get_recommended_lane(self, pose_data: Dict) -> str:
        """
        Get the recommended lane to analyze based on detection quality.

        Args:
            pose_data: Pose data dictionary

        Returns:
            'left' or 'right' - the lane with better detection
        """
        result = self.detect_from_pose_data(pose_data)

        if result.primary_lane == 'both':
            # For dual athlete, pick the one with slightly better detection
            if result.left_detection_rate >= result.right_detection_rate:
                return 'left'
            else:
                return 'right'
        elif result.primary_lane in ['left', 'right']:
            return result.primary_lane
        else:
            # Default to left if unclear
            return 'left'


def detect_athlete_count(pose_data: Dict) -> AthleteDetectionResult:
    """
    Convenience function to detect athlete count from pose data.

    Args:
        pose_data: Dictionary with 'metadata' and 'frames'

    Returns:
        AthleteDetectionResult
    """
    detector = AthleteCountDetector()
    return detector.detect_from_pose_data(pose_data)


def get_valid_lanes(pose_data: Dict, min_detection_rate: float = 0.25) -> List[str]:
    """
    Get list of lanes with valid detection data.

    Args:
        pose_data: Pose data dictionary
        min_detection_rate: Minimum detection rate to be considered valid

    Returns:
        List of valid lane names ('left', 'right', or both)
    """
    detector = AthleteCountDetector(single_athlete_threshold=min_detection_rate)
    result = detector.detect_from_pose_data(pose_data)

    valid_lanes = []
    if result.left_detection_rate >= min_detection_rate:
        valid_lanes.append('left')
    if result.right_detection_rate >= min_detection_rate:
        valid_lanes.append('right')

    return valid_lanes


@dataclass
class LaneAssignmentResult:
    """Result of lane position analysis."""
    original_lane: str
    suggested_lane: str
    needs_reassignment: bool
    average_x_position: float  # 0.0 = left edge, 1.0 = right edge
    confidence: float
    reason: str


class LaneAssignmentAnalyzer:
    """
    Analyze keypoint positions to determine correct lane assignment.

    For single-athlete videos, the athlete might be detected in the wrong lane
    (e.g., left lane when they're actually on the right side of the frame).
    This class analyzes the actual X positions to suggest reassignment.
    """

    def __init__(
        self,
        left_threshold: float = 0.4,  # X < 0.4 = likely left lane
        right_threshold: float = 0.6,  # X > 0.6 = likely right lane
        min_samples: int = 10
    ):
        """
        Args:
            left_threshold: X position below this is considered left lane
            right_threshold: X position above this is considered right lane
            min_samples: Minimum frames needed for reliable analysis
        """
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold
        self.min_samples = min_samples

    def analyze_lane_assignment(
        self,
        pose_data: Dict,
        current_lane: str
    ) -> LaneAssignmentResult:
        """
        Analyze if the athlete is assigned to the correct lane.

        Args:
            pose_data: Pose data dictionary
            current_lane: Currently assigned lane ('left' or 'right')

        Returns:
            LaneAssignmentResult with suggestion
        """
        frames = pose_data.get('frames', [])
        if not frames:
            return LaneAssignmentResult(
                original_lane=current_lane,
                suggested_lane=current_lane,
                needs_reassignment=False,
                average_x_position=0.5,
                confidence=0.0,
                reason="No frames available"
            )

        # Collect X positions of COM (center of mass)
        x_positions = []
        climber_key = f'{current_lane}_climber'

        for frame in frames:
            climber = frame.get(climber_key)
            if not climber or not climber.get('has_detection'):
                continue

            keypoints = climber.get('keypoints', {})

            # Use hip midpoint as reference (more stable than COM)
            left_hip = keypoints.get('left_hip')
            right_hip = keypoints.get('right_hip')

            if left_hip and right_hip:
                lh_x = left_hip.get('x', 0)
                rh_x = right_hip.get('x', 0)
                lh_conf = left_hip.get('visibility', left_hip.get('confidence', 0))
                rh_conf = right_hip.get('visibility', right_hip.get('confidence', 0))

                if lh_conf > 0.3 and rh_conf > 0.3:
                    hip_x = (lh_x + rh_x) / 2
                    x_positions.append(hip_x)

        if len(x_positions) < self.min_samples:
            return LaneAssignmentResult(
                original_lane=current_lane,
                suggested_lane=current_lane,
                needs_reassignment=False,
                average_x_position=0.5,
                confidence=0.0,
                reason=f"Not enough samples ({len(x_positions)} < {self.min_samples})"
            )

        # Calculate average position
        avg_x = np.mean(x_positions)
        std_x = np.std(x_positions)

        # Determine suggested lane based on position
        if avg_x < self.left_threshold:
            suggested_lane = 'left'
            position_desc = "left side"
        elif avg_x > self.right_threshold:
            suggested_lane = 'right'
            position_desc = "right side"
        else:
            suggested_lane = current_lane
            position_desc = "center"

        # Calculate confidence based on how far from center
        distance_from_center = abs(avg_x - 0.5)
        position_confidence = min(1.0, distance_from_center * 4)  # Max at 0.25 from center

        # Lower confidence if high variance
        if std_x > 0.1:
            position_confidence *= 0.7

        needs_reassignment = suggested_lane != current_lane

        if needs_reassignment:
            reason = f"Athlete is on {position_desc} (avg X={avg_x:.2f}) but assigned to {current_lane} lane"
        else:
            reason = f"Athlete correctly assigned to {current_lane} lane (avg X={avg_x:.2f})"

        return LaneAssignmentResult(
            original_lane=current_lane,
            suggested_lane=suggested_lane,
            needs_reassignment=needs_reassignment,
            average_x_position=avg_x,
            confidence=position_confidence,
            reason=reason
        )

    def get_best_lane(self, pose_data: Dict) -> Tuple[str, float]:
        """
        Determine the best lane to use based on detection quality and position.

        Args:
            pose_data: Pose data dictionary

        Returns:
            Tuple of (best_lane, confidence)
        """
        # First check detection rates
        detector = AthleteCountDetector()
        detection_result = detector.detect_from_pose_data(pose_data)

        # If one lane has much better detection, use that
        left_rate = detection_result.left_detection_rate
        right_rate = detection_result.right_detection_rate

        if left_rate > right_rate * 2 and left_rate > 0.3:
            # Left has significantly better detection
            assignment = self.analyze_lane_assignment(pose_data, 'left')
            return assignment.suggested_lane, assignment.confidence

        if right_rate > left_rate * 2 and right_rate > 0.3:
            # Right has significantly better detection
            assignment = self.analyze_lane_assignment(pose_data, 'right')
            return assignment.suggested_lane, assignment.confidence

        # Both lanes have similar detection - check position for each
        left_assignment = self.analyze_lane_assignment(pose_data, 'left')
        right_assignment = self.analyze_lane_assignment(pose_data, 'right')

        # Pick the one that's correctly positioned
        if not left_assignment.needs_reassignment:
            return 'left', left_rate
        if not right_assignment.needs_reassignment:
            return 'right', right_rate

        # Both might need reassignment - use the one with better detection
        if left_rate >= right_rate:
            return left_assignment.suggested_lane, left_rate
        else:
            return right_assignment.suggested_lane, right_rate


def analyze_lane_assignment(pose_data: Dict, lane: str) -> LaneAssignmentResult:
    """
    Convenience function to analyze lane assignment.

    Args:
        pose_data: Pose data dictionary
        lane: Current lane assignment

    Returns:
        LaneAssignmentResult
    """
    analyzer = LaneAssignmentAnalyzer()
    return analyzer.analyze_lane_assignment(pose_data, lane)


def get_best_lane_for_analysis(pose_data: Dict) -> Tuple[str, str]:
    """
    Get the best lane to use for analysis.

    Args:
        pose_data: Pose data dictionary

    Returns:
        Tuple of (lane, reason)
    """
    analyzer = LaneAssignmentAnalyzer()
    best_lane, confidence = analyzer.get_best_lane(pose_data)

    detector = AthleteCountDetector()
    detection_result = detector.detect_from_pose_data(pose_data)

    if detection_result.athlete_count == 1:
        reason = f"Single athlete detected in {best_lane} lane"
    elif detection_result.athlete_count == 2:
        reason = f"Two athletes - analyzing {best_lane} lane"
    else:
        reason = f"Using {best_lane} lane (best available)"

    return best_lane, reason
