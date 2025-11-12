"""
BlazePose Keypoint Extractor for Speed Climbing
===============================================

MediaPipe BlazePose wrapper for extracting 33 body keypoints from video frames.
Optimized for speed climbing analysis with focus on critical joints.

Classes:
    BlazePoseExtractor: Main pose estimation class
    Keypoint: Data class for individual keypoint
    PoseResult: Data class for complete pose estimation result

References:
    - MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose.html
    - BlazePose paper: https://arxiv.org/abs/2006.10204
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import logging
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class Keypoint:
    """
    Represents a single body keypoint.

    Attributes:
        name: Keypoint name (e.g., 'left_knee', 'COM')
        x: X coordinate (normalized 0-1 or pixel coordinates)
        y: Y coordinate (normalized 0-1 or pixel coordinates)
        z: Z coordinate (depth, normalized)
        confidence: Detection confidence (0-1)
        visibility: Visibility score (0-1)
    """
    name: str
    x: float
    y: float
    z: float
    confidence: float
    visibility: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_pixel_coords(self, width: int, height: int) -> Tuple[int, int]:
        """
        Convert normalized coordinates to pixel coordinates.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            (x_pixel, y_pixel) tuple
        """
        return (int(self.x * width), int(self.y * height))


@dataclass
class PoseResult:
    """
    Complete pose estimation result for a single frame.

    Attributes:
        frame_id: Frame index
        timestamp: Time in seconds
        keypoints: Dictionary of keypoint_name -> Keypoint
        has_detection: Whether pose was detected
        overall_confidence: Average confidence across all keypoints
    """
    frame_id: int
    timestamp: float
    keypoints: Dict[str, Keypoint]
    has_detection: bool
    overall_confidence: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'has_detection': self.has_detection,
            'overall_confidence': self.overall_confidence,
            'keypoints': {name: kp.to_dict() for name, kp in self.keypoints.items()}
        }

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        """Get keypoint by name."""
        return self.keypoints.get(name)

    def get_critical_keypoints(self) -> Dict[str, Keypoint]:
        """
        Get critical keypoints for speed climbing analysis.

        Returns:
            Dictionary with COM, knees, elbows, wrists, ankles, shoulders
        """
        critical = [
            'COM', 'left_knee', 'right_knee',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_ankle', 'right_ankle',
            'left_shoulder', 'right_shoulder'
        ]
        return {name: self.keypoints[name] for name in critical if name in self.keypoints}


class BlazePoseExtractor:
    """
    Extract 33 body keypoints using MediaPipe BlazePose.

    Features:
        - Real-time pose detection (30+ fps)
        - 33 keypoint landmarks
        - Confidence and visibility scores
        - GPU acceleration support
        - Automatic COM calculation

    Attributes:
        model_complexity: Model complexity (0=Lite, 1=Full, 2=Heavy)
        min_detection_confidence: Minimum detection confidence threshold
        min_tracking_confidence: Minimum tracking confidence threshold
        enable_segmentation: Enable person segmentation mask
        smooth_landmarks: Enable landmark smoothing

    Example:
        >>> extractor = BlazePoseExtractor(model_complexity=1)
        >>> results = extractor.process_frame(frame, frame_id=0, timestamp=0.0)
        >>> if results.has_detection:
        ...     com = results.get_keypoint('COM')
        ...     print(f"COM: ({com.x:.3f}, {com.y:.3f})")
    """

    # MediaPipe landmark indices and names
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
        """
        Initialize BlazePose extractor.

        Args:
            model_complexity: 0 (Lite), 1 (Full), or 2 (Heavy)
            min_detection_confidence: Minimum confidence for detection (0-1)
            min_tracking_confidence: Minimum confidence for tracking (0-1)
            enable_segmentation: Enable segmentation mask (slower)
            smooth_landmarks: Apply temporal smoothing
            static_image_mode: True for images, False for video
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_segmentation = enable_segmentation
        self.smooth_landmarks = smooth_landmarks
        self.static_image_mode = static_image_mode

        # Initialize MediaPipe Pose
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info(f"BlazePose initialized (complexity={model_complexity})")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0
    ) -> PoseResult:
        """
        Process a single frame and extract keypoints.

        Args:
            frame: Input image (BGR format from OpenCV)
            frame_id: Frame index for tracking
            timestamp: Time in seconds

        Returns:
            PoseResult object with all keypoints

        Example:
            >>> frame = cv2.imread("climber.jpg")
            >>> result = extractor.process_frame(frame)
            >>> print(f"Detection: {result.has_detection}")
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.pose.process(image_rgb)

        # Extract keypoints
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
                    confidence=landmark.visibility,  # MediaPipe uses visibility as confidence
                    visibility=landmark.visibility
                )

                keypoints[name] = keypoint
                confidences.append(landmark.visibility)

            # Calculate Center of Mass (COM) - average of hip landmarks
            keypoints['COM'] = self._calculate_com(keypoints)

        # Calculate overall confidence
        overall_confidence = np.mean(confidences) if confidences else 0.0

        return PoseResult(
            frame_id=frame_id,
            timestamp=timestamp,
            keypoints=keypoints,
            has_detection=has_detection,
            overall_confidence=overall_confidence
        )

    def _calculate_com(self, keypoints: Dict[str, Keypoint]) -> Keypoint:
        """
        Calculate Center of Mass (COM) from hip keypoints.

        COM is approximated as the midpoint between left and right hips.

        Args:
            keypoints: Dictionary of extracted keypoints

        Returns:
            COM Keypoint
        """
        left_hip = keypoints.get('left_hip')
        right_hip = keypoints.get('right_hip')

        if left_hip and right_hip:
            x = (left_hip.x + right_hip.x) / 2
            y = (left_hip.y + right_hip.y) / 2
            z = (left_hip.z + right_hip.z) / 2
            confidence = (left_hip.confidence + right_hip.confidence) / 2
            visibility = (left_hip.visibility + right_hip.visibility) / 2
        else:
            # Fallback: use available hip or zeros
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
        show_connections: bool = True,
        show_labels: bool = False
    ) -> np.ndarray:
        """
        Draw pose landmarks on frame for visualization.

        Args:
            frame: Input frame (BGR)
            pose_result: PoseResult object
            show_connections: Draw skeleton connections
            show_labels: Show keypoint names

        Returns:
            Annotated frame

        Example:
            >>> annotated = extractor.draw_landmarks(frame, result)
            >>> cv2.imshow('Pose', annotated)
        """
        if not pose_result.has_detection:
            return frame.copy()

        annotated = frame.copy()
        height, width = frame.shape[:2]

        # Draw connections (skeleton)
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

        # Draw keypoints
        for name, keypoint in pose_result.keypoints.items():
            if name == 'COM':
                # Draw COM in different color
                pos = keypoint.to_pixel_coords(width, height)
                cv2.circle(annotated, pos, 8, (0, 0, 255), -1)  # Red
                if show_labels:
                    cv2.putText(annotated, "COM", (pos[0] + 10, pos[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                pos = keypoint.to_pixel_coords(width, height)
                # Color based on confidence
                color_intensity = int(keypoint.confidence * 255)
                cv2.circle(annotated, pos, 4, (color_intensity, color_intensity, 0), -1)

                if show_labels:
                    cv2.putText(annotated, name, (pos[0] + 5, pos[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return annotated

    def save_results(
        self,
        results: List[PoseResult],
        output_path: str,
        format: str = 'json'
    ) -> bool:
        """
        Save pose estimation results to file.

        Args:
            results: List of PoseResult objects
            output_path: Output file path
            format: Output format ('json' or 'csv')

        Returns:
            True if successful

        Example:
            >>> results = [extractor.process_frame(f) for f in frames]
            >>> extractor.save_results(results, "keypoints.json")
        """
        try:
            if format == 'json':
                data = [result.to_dict() for result in results]
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

            elif format == 'csv':
                import pandas as pd

                # Flatten results for CSV
                rows = []
                for result in results:
                    row = {
                        'frame_id': result.frame_id,
                        'timestamp': result.timestamp,
                        'has_detection': result.has_detection,
                        'overall_confidence': result.overall_confidence
                    }

                    # Add all keypoint coordinates
                    for name, kp in result.keypoints.items():
                        row[f'{name}_x'] = kp.x
                        row[f'{name}_y'] = kp.y
                        row[f'{name}_z'] = kp.z
                        row[f'{name}_confidence'] = kp.confidence

                    rows.append(row)

                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Results saved: {output_path} ({len(results)} frames)")
            return True

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resources."""
        self.release()

    def release(self):
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()
            logger.info("BlazePose resources released")

    def __del__(self):
        """Destructor."""
        self.release()


# ==================== Utility Functions ====================

def extract_keypoints_from_video(
    video_path: str,
    output_path: str,
    model_complexity: int = 1,
    save_format: str = 'json',
    visualize: bool = False,
    output_video_path: Optional[str] = None
) -> List[PoseResult]:
    """
    High-level function to extract keypoints from entire video.

    Args:
        video_path: Input video path
        output_path: Output file path for keypoints
        model_complexity: BlazePose model complexity (0-2)
        save_format: Output format ('json' or 'csv')
        visualize: If True, create annotated video
        output_video_path: Path for annotated video (required if visualize=True)

    Returns:
        List of PoseResult objects

    Example:
        >>> results = extract_keypoints_from_video(
        ...     "athlete.mp4",
        ...     "keypoints.json",
        ...     visualize=True,
        ...     output_video_path="annotated.mp4"
        ... )
    """
    from .video_processor import VideoProcessor

    results = []

    with VideoProcessor(video_path) as video_proc, \
         BlazePoseExtractor(model_complexity=model_complexity) as extractor:

        # Setup video writer for visualization
        writer = None
        if visualize and output_video_path:
            writer = video_proc.create_video_writer(output_video_path)

        # Process all frames
        logger.info(f"Processing video: {video_path}")
        for frame_data in video_proc.extract_frames():
            result = extractor.process_frame(
                frame_data['frame'],
                frame_data['frame_id'],
                frame_data['timestamp']
            )
            results.append(result)

            # Visualize
            if writer:
                annotated = extractor.draw_landmarks(frame_data['frame'], result)
                writer.write(annotated)

            # Progress
            if frame_data['frame_number'] % 30 == 0:
                logger.info(f"Processed {frame_data['frame_number']} frames...")

        if writer:
            writer.release()

    # Save results
    with BlazePoseExtractor() as extractor:
        extractor.save_results(results, output_path, format=save_format)

    logger.info(f"Extraction complete: {len(results)} frames")
    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python blazepose_extractor.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_json = "keypoints.json"
    output_video = "annotated_output.mp4"

    # Extract keypoints
    results = extract_keypoints_from_video(
        video_path,
        output_json,
        visualize=True,
        output_video_path=output_video
    )

    print(f"\n✓ Extracted {len(results)} frames")
    print(f"✓ Keypoints saved: {output_json}")
    print(f"✓ Annotated video: {output_video}")
