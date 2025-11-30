"""
Posture and body configuration analysis.

Extracts joint angles and body alignment metrics.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from .base import calculate_angle, get_keypoint_at_frame


class PostureAnalyzer:
    """
    Analyze body posture and configuration.

    Extracts:
    - avg_knee_angle: Average knee bend during climbing
    - avg_elbow_angle: Average elbow angle during reaches
    - hip_width_ratio: Body width ratio (stability indicator)
    - body_lean_angle: Average torso lean from vertical
    - reach_ratio: Hand reach relative to body height
    """

    def __init__(self, min_frames: int = 30, min_confidence: float = 0.5):
        """
        Args:
            min_frames: Minimum frames required for analysis
            min_confidence: Minimum keypoint confidence threshold
        """
        self.min_frames = min_frames
        self.min_confidence = min_confidence

    def analyze(self, frames: List[Dict[str, Any]], lane: str = 'left') -> Dict[str, float]:
        """
        Extract posture features from pose frames.

        Args:
            frames: List of frame dictionaries from pose JSON
            lane: 'left' or 'right' climber

        Returns:
            Dictionary of posture features
        """
        # Collect angles per frame
        left_knee_angles = []
        right_knee_angles = []
        left_elbow_angles = []
        right_elbow_angles = []
        hip_widths = []
        shoulder_widths = []
        body_lean_angles = []
        reach_ratios = []

        for frame in frames:
            # Knee angles
            lka = self._get_knee_angle(frame, lane, 'left')
            rka = self._get_knee_angle(frame, lane, 'right')
            if lka is not None:
                left_knee_angles.append(lka)
            if rka is not None:
                right_knee_angles.append(rka)

            # Elbow angles
            lea = self._get_elbow_angle(frame, lane, 'left')
            rea = self._get_elbow_angle(frame, lane, 'right')
            if lea is not None:
                left_elbow_angles.append(lea)
            if rea is not None:
                right_elbow_angles.append(rea)

            # Body dimensions
            hw = self._get_hip_width(frame, lane)
            sw = self._get_shoulder_width(frame, lane)
            if hw is not None:
                hip_widths.append(hw)
            if sw is not None:
                shoulder_widths.append(sw)

            # Body lean
            lean = self._get_body_lean(frame, lane)
            if lean is not None:
                body_lean_angles.append(lean)

            # Reach ratio
            reach = self._get_reach_ratio(frame, lane)
            if reach is not None:
                reach_ratios.append(reach)

        # Calculate summary features
        all_knee = left_knee_angles + right_knee_angles
        all_elbow = left_elbow_angles + right_elbow_angles

        features = {
            'avg_knee_angle': self._safe_mean(all_knee),
            'knee_angle_std': self._safe_std(all_knee),
            'avg_elbow_angle': self._safe_mean(all_elbow),
            'elbow_angle_std': self._safe_std(all_elbow),
            'hip_width_ratio': self._calculate_width_ratio(hip_widths, shoulder_widths),
            'avg_body_lean': self._safe_mean(body_lean_angles),
            'body_lean_std': self._safe_std(body_lean_angles),
            'avg_reach_ratio': self._safe_mean(reach_ratios),
            'max_reach_ratio': max(reach_ratios) if reach_ratios else 0.0,
        }

        return features

    def _get_knee_angle(self, frame: Dict, lane: str, side: str) -> Optional[float]:
        """Calculate knee angle (hip-knee-ankle)."""
        hip = get_keypoint_at_frame(frame, f'{side}_hip', lane, self.min_confidence)
        knee = get_keypoint_at_frame(frame, f'{side}_knee', lane, self.min_confidence)
        ankle = get_keypoint_at_frame(frame, f'{side}_ankle', lane, self.min_confidence)

        if hip is None or knee is None or ankle is None:
            return None

        return calculate_angle(hip, knee, ankle)

    def _get_elbow_angle(self, frame: Dict, lane: str, side: str) -> Optional[float]:
        """Calculate elbow angle (shoulder-elbow-wrist)."""
        shoulder = get_keypoint_at_frame(frame, f'{side}_shoulder', lane, self.min_confidence)
        elbow = get_keypoint_at_frame(frame, f'{side}_elbow', lane, self.min_confidence)
        wrist = get_keypoint_at_frame(frame, f'{side}_wrist', lane, self.min_confidence)

        if shoulder is None or elbow is None or wrist is None:
            return None

        return calculate_angle(shoulder, elbow, wrist)

    def _get_hip_width(self, frame: Dict, lane: str) -> Optional[float]:
        """Calculate hip width (distance between hips)."""
        left_hip = get_keypoint_at_frame(frame, 'left_hip', lane, self.min_confidence)
        right_hip = get_keypoint_at_frame(frame, 'right_hip', lane, self.min_confidence)

        if left_hip is None or right_hip is None:
            return None

        return np.sqrt((left_hip[0] - right_hip[0])**2 + (left_hip[1] - right_hip[1])**2)

    def _get_shoulder_width(self, frame: Dict, lane: str) -> Optional[float]:
        """Calculate shoulder width."""
        left = get_keypoint_at_frame(frame, 'left_shoulder', lane, self.min_confidence)
        right = get_keypoint_at_frame(frame, 'right_shoulder', lane, self.min_confidence)

        if left is None or right is None:
            return None

        return np.sqrt((left[0] - right[0])**2 + (left[1] - right[1])**2)

    def _get_body_lean(self, frame: Dict, lane: str) -> Optional[float]:
        """
        Calculate body lean angle from vertical.

        Uses line from hip midpoint to shoulder midpoint.
        Returns angle in degrees (0 = vertical, positive = leaning right).
        """
        left_hip = get_keypoint_at_frame(frame, 'left_hip', lane, self.min_confidence)
        right_hip = get_keypoint_at_frame(frame, 'right_hip', lane, self.min_confidence)
        left_shoulder = get_keypoint_at_frame(frame, 'left_shoulder', lane, self.min_confidence)
        right_shoulder = get_keypoint_at_frame(frame, 'right_shoulder', lane, self.min_confidence)

        if any(p is None for p in [left_hip, right_hip, left_shoulder, right_shoulder]):
            return None

        # Midpoints
        hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2,
                        (left_shoulder[1] + right_shoulder[1]) / 2)

        # Vector from hip to shoulder
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]

        # Note: In image coords, y increases downward
        # Vertical line would be (0, -1)
        # Angle from vertical
        angle = np.degrees(np.arctan2(dx, -dy))  # -dy because y is inverted

        return angle

    def _get_reach_ratio(self, frame: Dict, lane: str) -> Optional[float]:
        """
        Calculate reach ratio.

        Reach = distance from shoulder to wrist / distance from shoulder to hip
        Higher ratio = longer reach relative to body.
        """
        # Try both arms, use max
        left_reach = self._get_single_reach(frame, lane, 'left')
        right_reach = self._get_single_reach(frame, lane, 'right')

        reaches = [r for r in [left_reach, right_reach] if r is not None]

        if not reaches:
            return None

        return max(reaches)

    def _get_single_reach(self, frame: Dict, lane: str, side: str) -> Optional[float]:
        """Calculate reach ratio for one arm."""
        shoulder = get_keypoint_at_frame(frame, f'{side}_shoulder', lane, self.min_confidence)
        wrist = get_keypoint_at_frame(frame, f'{side}_wrist', lane, self.min_confidence)
        hip = get_keypoint_at_frame(frame, f'{side}_hip', lane, self.min_confidence)

        if shoulder is None or wrist is None or hip is None:
            return None

        # Shoulder to wrist distance
        reach = np.sqrt((wrist[0] - shoulder[0])**2 + (wrist[1] - shoulder[1])**2)

        # Shoulder to hip distance (body reference)
        body_ref = np.sqrt((hip[0] - shoulder[0])**2 + (hip[1] - shoulder[1])**2)

        if body_ref < 1e-10:
            return None

        return reach / body_ref

    def _calculate_width_ratio(self, hip_widths: List[float], shoulder_widths: List[float]) -> float:
        """Calculate average hip/shoulder width ratio."""
        if not hip_widths or not shoulder_widths:
            return 0.0

        avg_hip = np.mean(hip_widths)
        avg_shoulder = np.mean(shoulder_widths)

        if avg_shoulder < 1e-10:
            return 0.0

        return float(avg_hip / avg_shoulder)

    def _safe_mean(self, values: List[float]) -> float:
        """Calculate mean, returning 0 if empty."""
        if not values:
            return 0.0
        return float(np.mean(values))

    def _safe_std(self, values: List[float]) -> float:
        """Calculate std, returning 0 if too few values."""
        if len(values) < 2:
            return 0.0
        return float(np.std(values))
