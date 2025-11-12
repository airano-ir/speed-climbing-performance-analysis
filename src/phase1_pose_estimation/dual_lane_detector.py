"""
Dual-Lane Detection for Speed Climbing Races
============================================

Detects and separates two climbers in a competitive speed climbing video.

Key Features:
- Automatic lane boundary detection (left/right split)
- Per-lane pose estimation
- Lane assignment validation
- Handles lane crossovers and occlusions

Algorithm:
1. Frame analysis to detect vertical boundary between lanes
2. Separate BlazePose extraction for each lane
3. Confidence-based lane assignment
4. Temporal consistency tracking (Kalman filter)

Author: Speed Climbing Research Team
Date: 2025-11-12
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from blazepose_extractor import BlazePoseExtractor, PoseResult
from filterpy.kalman import KalmanFilter


@dataclass
class LaneBoundary:
    """
    Represents the boundary between left and right climbing lanes.
    """
    x_center: float  # Vertical line x-coordinate (normalized 0-1)
    confidence: float  # Detection confidence
    frame_width: int
    frame_height: int

    @property
    def x_pixel(self) -> int:
        """Get boundary x-coordinate in pixels."""
        return int(self.x_center * self.frame_width)

    def is_left_lane(self, x: float, normalized: bool = True) -> bool:
        """
        Check if a point is in the left lane.

        Args:
            x: X-coordinate (normalized 0-1 or pixel)
            normalized: If True, x is normalized; if False, x is in pixels

        Returns:
            True if point is in left lane
        """
        if not normalized:
            x = x / self.frame_width
        return x < self.x_center

    def get_lane_mask(self, lane: str) -> np.ndarray:
        """
        Create a binary mask for a specific lane.

        Args:
            lane: "left" or "right"

        Returns:
            Binary mask (H x W) with 1 for the specified lane
        """
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)

        if lane == "left":
            mask[:, :self.x_pixel] = 1
        elif lane == "right":
            mask[:, self.x_pixel:] = 1
        else:
            raise ValueError(f"Invalid lane: {lane}. Must be 'left' or 'right'")

        return mask


@dataclass
class DualLaneResult:
    """
    Result of dual-lane pose detection for a single frame.
    """
    frame_id: int
    timestamp: float
    lane_boundary: LaneBoundary
    left_climber: Optional[PoseResult] = None
    right_climber: Optional[PoseResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'lane_boundary': {
                'x_center': self.lane_boundary.x_center,
                'x_pixel': self.lane_boundary.x_pixel,
                'confidence': self.lane_boundary.confidence
            },
            'left_climber': self.left_climber.to_dict() if self.left_climber else None,
            'right_climber': self.right_climber.to_dict() if self.right_climber else None
        }


class DualLaneDetector:
    """
    Detects and tracks two climbers in separate lanes of a speed climbing race.

    The detector:
    1. Identifies the vertical boundary between two lanes
    2. Runs separate pose estimation on each lane
    3. Tracks lane assignments over time using Kalman filtering
    4. Handles edge cases (occlusions, lane crossovers)

    Usage:
        detector = DualLaneDetector()
        result = detector.process_frame(frame, frame_id, timestamp)

        print(f"Left climber detected: {result.left_climber is not None}")
        print(f"Right climber detected: {result.right_climber is not None}")
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        boundary_detection_method: str = "edge",
        enable_lane_smoothing: bool = True
    ):
        """
        Initialize dual-lane detector.

        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            boundary_detection_method: Method to detect lane boundary
                - "edge": Edge detection (default)
                - "fixed": Fixed at frame center (x=0.5)
                - "motion": Motion-based detection
            enable_lane_smoothing: Use Kalman filter for boundary smoothing
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.boundary_detection_method = boundary_detection_method
        self.enable_lane_smoothing = enable_lane_smoothing

        # Initialize pose extractors (will be created on first frame)
        self.left_extractor: Optional[BlazePoseExtractor] = None
        self.right_extractor: Optional[BlazePoseExtractor] = None

        # Kalman filter for boundary tracking
        self.boundary_kf: Optional[KalmanFilter] = None
        self.boundary_initialized = False

        # Frame metadata
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None

        # Statistics
        self.total_frames = 0
        self.left_detections = 0
        self.right_detections = 0
        self.dual_detections = 0

    def __enter__(self):
        """Context manager entry."""
        self._initialize_extractors()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def _initialize_extractors(self):
        """Initialize BlazePose extractors for both lanes."""
        if self.left_extractor is None:
            self.left_extractor = BlazePoseExtractor(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.left_extractor.__enter__()

        if self.right_extractor is None:
            self.right_extractor = BlazePoseExtractor(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.right_extractor.__enter__()

    def _initialize_kalman_filter(self):
        """Initialize Kalman filter for boundary position tracking."""
        # State: [x_position, x_velocity]
        self.boundary_kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition matrix (constant velocity model)
        self.boundary_kf.F = np.array([[1., 1.],  # x = x + v*dt
                                       [0., 1.]])  # v = v

        # Measurement matrix
        self.boundary_kf.H = np.array([[1., 0.]])  # We only measure position

        # Process noise
        self.boundary_kf.Q = np.array([[0.001, 0.],
                                       [0., 0.001]])

        # Measurement noise
        self.boundary_kf.R = np.array([[0.01]])

        # Initial state (will be set on first detection)
        self.boundary_kf.x = np.array([[0.5], [0.]])  # Start at center, no velocity

        # Initial covariance
        self.boundary_kf.P = np.eye(2) * 1000

        self.boundary_initialized = True

    def detect_lane_boundary(self, frame: np.ndarray) -> LaneBoundary:
        """
        Detect the vertical boundary between left and right lanes.

        Args:
            frame: Input frame (BGR format)

        Returns:
            LaneBoundary object
        """
        h, w = frame.shape[:2]

        if self.boundary_detection_method == "fixed":
            # Simple fixed boundary at frame center
            x_center = 0.5
            confidence = 1.0

        elif self.boundary_detection_method == "edge":
            # Edge-based detection
            x_center, confidence = self._detect_boundary_by_edges(frame)

        elif self.boundary_detection_method == "motion":
            # Motion-based detection (future enhancement)
            x_center, confidence = self._detect_boundary_by_motion(frame)

        else:
            raise ValueError(f"Unknown boundary detection method: {self.boundary_detection_method}")

        # Apply Kalman smoothing if enabled
        if self.enable_lane_smoothing:
            if not self.boundary_initialized:
                self._initialize_kalman_filter()
                self.boundary_kf.x[0] = x_center

            # Predict
            self.boundary_kf.predict()

            # Update with measurement
            self.boundary_kf.update(np.array([[x_center]]))

            # Use filtered position
            x_center = self.boundary_kf.x[0].item()

        return LaneBoundary(
            x_center=x_center,
            confidence=confidence,
            frame_width=w,
            frame_height=h
        )

    def _detect_boundary_by_edges(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Detect boundary using vertical edge detection.

        The IFSC wall typically has a clear vertical line between two lanes.

        Args:
            frame: Input frame

        Returns:
            (x_center_normalized, confidence)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Vertical Sobel filter (emphasize vertical edges)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.abs(sobel_x)

        # Sum along vertical axis to find strongest vertical edge
        vertical_profile = np.sum(sobel_x, axis=0)

        # Find peak in center 60% of frame (ignore edges)
        w = frame.shape[1]
        search_start = int(w * 0.2)
        search_end = int(w * 0.8)

        search_region = vertical_profile[search_start:search_end]
        peak_idx = np.argmax(search_region)
        peak_value = search_region[peak_idx]

        # Convert to absolute position
        x_pixel = search_start + peak_idx
        x_center = x_pixel / w

        # Confidence based on peak strength
        confidence = min(1.0, peak_value / (np.mean(vertical_profile) * 3))

        return x_center, confidence

    def _detect_boundary_by_motion(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Detect boundary using motion analysis.

        (Future enhancement: analyze climber movements to infer lane boundary)

        Args:
            frame: Input frame

        Returns:
            (x_center_normalized, confidence)
        """
        # For now, fall back to fixed
        return 0.5, 0.5

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float
    ) -> DualLaneResult:
        """
        Process a single frame to detect both climbers.

        Args:
            frame: Input frame (BGR format)
            frame_id: Frame number
            timestamp: Timestamp in seconds

        Returns:
            DualLaneResult with detection results for both lanes
        """
        # Initialize extractors if needed
        if self.left_extractor is None or self.right_extractor is None:
            self._initialize_extractors()

        # Store frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]

        # Detect lane boundary
        boundary = self.detect_lane_boundary(frame)

        # Create lane masks
        left_mask = boundary.get_lane_mask("left")
        right_mask = boundary.get_lane_mask("right")

        # Apply masks to frame
        left_frame = frame.copy()
        left_frame[:, boundary.x_pixel:] = 0  # Black out right side

        right_frame = frame.copy()
        right_frame[:, :boundary.x_pixel] = 0  # Black out left side

        # Run pose estimation on each lane
        left_result = self.left_extractor.process_frame(left_frame, frame_id, timestamp)
        right_result = self.right_extractor.process_frame(right_frame, frame_id, timestamp)

        # Validate lane assignments
        left_result = self._validate_lane_assignment(left_result, boundary, "left")
        right_result = self._validate_lane_assignment(right_result, boundary, "right")

        # Update statistics
        self.total_frames += 1
        if left_result is not None:
            self.left_detections += 1
        if right_result is not None:
            self.right_detections += 1
        if left_result is not None and right_result is not None:
            self.dual_detections += 1

        return DualLaneResult(
            frame_id=frame_id,
            timestamp=timestamp,
            lane_boundary=boundary,
            left_climber=left_result,
            right_climber=right_result
        )

    def _validate_lane_assignment(
        self,
        pose_result: Optional[PoseResult],
        boundary: LaneBoundary,
        expected_lane: str
    ) -> Optional[PoseResult]:
        """
        Validate that detected pose is actually in the expected lane.

        Args:
            pose_result: Pose detection result
            boundary: Lane boundary
            expected_lane: "left" or "right"

        Returns:
            pose_result if valid, None otherwise
        """
        if pose_result is None or not pose_result.has_detection:
            return None

        # Get COM keypoint
        com = pose_result.get_keypoint('COM')
        if com is None:
            return None

        # Check if COM is in expected lane
        com_x = com.x
        is_left = boundary.is_left_lane(com_x, normalized=True)

        if expected_lane == "left" and is_left:
            return pose_result
        elif expected_lane == "right" and not is_left:
            return pose_result
        else:
            # Pose is in wrong lane - discard
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics.

        Returns:
            Dictionary with detection rates and performance metrics
        """
        if self.total_frames == 0:
            return {
                'total_frames': 0,
                'left_detection_rate': 0.0,
                'right_detection_rate': 0.0,
                'dual_detection_rate': 0.0
            }

        return {
            'total_frames': self.total_frames,
            'left_detections': self.left_detections,
            'right_detections': self.right_detections,
            'dual_detections': self.dual_detections,
            'left_detection_rate': self.left_detections / self.total_frames,
            'right_detection_rate': self.right_detections / self.total_frames,
            'dual_detection_rate': self.dual_detections / self.total_frames
        }

    def cleanup(self):
        """Clean up resources."""
        if self.left_extractor is not None:
            self.left_extractor.__exit__(None, None, None)
            self.left_extractor = None

        if self.right_extractor is not None:
            self.right_extractor.__exit__(None, None, None)
            self.right_extractor = None

    def __del__(self):
        """Destructor."""
        self.cleanup()


def visualize_dual_lane(
    frame: np.ndarray,
    result: DualLaneResult,
    show_boundary: bool = True,
    show_skeletons: bool = True
) -> np.ndarray:
    """
    Visualize dual-lane detection on a frame.

    Args:
        frame: Input frame
        result: DualLaneResult from process_frame()
        show_boundary: Draw vertical boundary line
        show_skeletons: Draw pose skeletons

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    # Draw boundary
    if show_boundary:
        x = result.lane_boundary.x_pixel
        cv2.line(annotated, (x, 0), (x, frame.shape[0]), (0, 255, 255), 2)

        # Label lanes
        cv2.putText(annotated, "LEFT", (x//2 - 30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(annotated, "RIGHT", (x + (frame.shape[1]-x)//2 - 40, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Draw skeletons
    if show_skeletons:
        if result.left_climber:
            # Draw left climber in blue
            annotated = _draw_skeleton(annotated, result.left_climber, color=(255, 0, 0))

        if result.right_climber:
            # Draw right climber in red
            annotated = _draw_skeleton(annotated, result.right_climber, color=(0, 0, 255))

    # Add detection status
    status_y = frame.shape[0] - 30
    left_status = "L: OK" if result.left_climber else "L: ---"
    right_status = "R: OK" if result.right_climber else "R: ---"

    cv2.putText(annotated, left_status, (20, status_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, right_status, (150, status_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated


def _draw_skeleton(frame: np.ndarray, pose: PoseResult, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Draw pose skeleton on frame.

    Args:
        frame: Input frame
        pose: PoseResult with keypoints
        color: RGB color tuple

    Returns:
        Annotated frame
    """
    # MediaPipe connections (simplified)
    connections = [
        (11, 12), (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),             # Right arm
        (11, 23), (12, 24),             # Torso
        (23, 24),                        # Hips
        (23, 25), (25, 27), (27, 29),  # Left leg
        (24, 26), (26, 28), (28, 30),  # Right leg
    ]

    h, w = frame.shape[:2]

    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(pose.keypoints) and end_idx < len(pose.keypoints):
            start_kp = pose.keypoints[start_idx]
            end_kp = pose.keypoints[end_idx]

            if start_kp.visibility > 0.5 and end_kp.visibility > 0.5:
                start_pt = (int(start_kp.x * w), int(start_kp.y * h))
                end_pt = (int(end_kp.x * w), int(end_kp.y * h))
                cv2.line(frame, start_pt, end_pt, color, 2)

    # Draw keypoints
    for kp in pose.keypoints:
        if kp.visibility > 0.5:
            pt = (int(kp.x * w), int(kp.y * h))
            cv2.circle(frame, pt, 4, color, -1)

    # Draw COM
    com = pose.get_keypoint('COM')
    if com:
        com_pt = (int(com.x * w), int(com.y * h))
        cv2.circle(frame, com_pt, 8, (0, 255, 0), -1)
        cv2.circle(frame, com_pt, 10, color, 2)

    return frame


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dual_lane_detector.py <video_path> [output_path]")
        print("\nExample:")
        print("  python dual_lane_detector.py race.mp4 output.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "dual_lane_output.mp4"

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video
    with DualLaneDetector() as detector:
        frame_id = 0
        results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_id / fps

            # Detect dual lanes
            result = detector.process_frame(frame, frame_id, timestamp)
            results.append(result)

            # Visualize
            annotated = visualize_dual_lane(frame, result)
            out.write(annotated)

            # Progress
            if frame_id % 30 == 0:
                print(f"  Frame {frame_id}/{total_frames} ({100*frame_id/total_frames:.1f}%)")

            frame_id += 1

        # Statistics
        stats = detector.get_statistics()
        print(f"\nDetection Statistics:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Left climber: {stats['left_detection_rate']*100:.1f}%")
        print(f"  Right climber: {stats['right_detection_rate']*100:.1f}%")
        print(f"  Both climbers: {stats['dual_detection_rate']*100:.1f}%")

        # Save results
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        print(f"\nOutput saved:")
        print(f"  Video: {output_path}")
        print(f"  JSON: {json_path}")

    cap.release()
    out.release()
