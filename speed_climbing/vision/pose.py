"""
BlazePose Keypoint Extractor for Speed Climbing.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class Keypoint:
    """Represents a single body keypoint."""
    name: str
    x: float
    y: float
    z: float
    confidence: float
    visibility: float

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_pixel_coords(self, width: int, height: int) -> Tuple[int, int]:
        return (int(self.x * width), int(self.y * height))


@dataclass
class PoseResult:
    """Complete pose estimation result for a single frame."""
    frame_id: int
    timestamp: float
    keypoints: Dict[str, Keypoint]
    has_detection: bool
    overall_confidence: float

    def to_dict(self) -> Dict:
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'has_detection': self.has_detection,
            'overall_confidence': self.overall_confidence,
            'keypoints': {name: kp.to_dict() for name, kp in self.keypoints.items()}
        }

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        return self.keypoints.get(name)


class BlazePoseExtractor:
    """Extract 33 body keypoints using MediaPipe BlazePose."""

    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky', 'right_pinky',
        'left_index', 'right_index',
        'left_thumb', 'right_thumb',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
        smooth_landmarks: bool = True,
        static_image_mode: bool = False
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_segmentation = enable_segmentation
        self.smooth_landmarks = smooth_landmarks
        self.static_image_mode = static_image_mode

        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_pose(self, frame: np.ndarray, lane: str = None) -> Optional[PoseResult]:
        """
        Wrapper for process_frame to match the interface expected by GlobalMapVideoProcessor.
        In a real scenario, this would handle ROI cropping for the specific lane.
        For now, it just processes the full frame (or whatever is passed).
        """
        # If lane is specified, we might want to crop, but for now assume frame is already appropriate
        return self.process_frame(frame)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0
    ) -> PoseResult:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        keypoints = {}
        has_detection = False
        confidences = []

        if results.pose_landmarks:
            has_detection = True
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                name = self.LANDMARK_NAMES[idx]
                keypoint = Keypoint(
                    name=name,
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    confidence=landmark.visibility,
                    visibility=landmark.visibility
                )
                keypoints[name] = keypoint
                confidences.append(landmark.visibility)

            keypoints['COM'] = self._calculate_com(keypoints)

        overall_confidence = np.mean(confidences) if confidences else 0.0

        return PoseResult(
            frame_id=frame_id,
            timestamp=timestamp,
            keypoints=keypoints,
            has_detection=has_detection,
            overall_confidence=overall_confidence
        )

    def _calculate_com(self, keypoints: Dict[str, Keypoint]) -> Keypoint:
        left_hip = keypoints.get('left_hip')
        right_hip = keypoints.get('right_hip')

        if left_hip and right_hip:
            x = (left_hip.x + right_hip.x) / 2
            y = (left_hip.y + right_hip.y) / 2
            z = (left_hip.z + right_hip.z) / 2
            confidence = (left_hip.confidence + right_hip.confidence) / 2
            visibility = (left_hip.visibility + right_hip.visibility) / 2
        else:
            x = y = z = confidence = visibility = 0.0

        return Keypoint(
            name='COM',
            x=x, y=y, z=z,
            confidence=confidence,
            visibility=visibility
        )

    def draw_landmarks(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        show_connections: bool = True
    ) -> np.ndarray:
        if not pose_result.has_detection:
            return frame.copy()

        annotated = frame.copy()
        height, width = frame.shape[:2]

        if show_connections:
            connections = mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                start_name = self.LANDMARK_NAMES[start_idx]
                end_name = self.LANDMARK_NAMES[end_idx]
                start_kp = pose_result.keypoints.get(start_name)
                end_kp = pose_result.keypoints.get(end_name)

                if start_kp and end_kp:
                    start_pos = start_kp.to_pixel_coords(width, height)
                    end_pos = end_kp.to_pixel_coords(width, height)
                    cv2.line(annotated, start_pos, end_pos, (0, 255, 0), 2)

        for name, keypoint in pose_result.keypoints.items():
            pos = keypoint.to_pixel_coords(width, height)
            if name == 'COM':
                cv2.circle(annotated, pos, 8, (0, 0, 255), -1)
            else:
                color_intensity = int(keypoint.confidence * 255)
                cv2.circle(annotated, pos, 4, (color_intensity, color_intensity, 0), -1)

        return annotated

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        if hasattr(self, 'pose') and self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass
    
    def __del__(self):
        self.release()
