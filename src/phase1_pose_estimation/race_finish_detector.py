"""
Race Finish Detector - Detects when climber finishes (touches top button)

This module detects race finish by:
1. Detecting hand reaching top of wall (hold 20 / STOP DEVICE)
2. Detecting button press (color change, light activation)
3. Validating finish with pose estimation

Usage:
    detector = RaceFinishDetector()
    finish_result = detector.detect_from_video('video.mp4', lane='left')

    # With pose data
    finish_result = detector.detect_from_poses(poses, fps=30)

Author: Speed Climbing Analysis Project
Date: 2025-11-13
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class RaceFinishResult:
    """Result of race finish detection"""
    frame_id: int
    timestamp: float  # seconds
    confidence: float  # 0.0 to 1.0
    lane: str  # 'left', 'right', or 'unknown'
    method: str  # 'pose', 'visual', or 'combined'
    hand_position: Optional[Tuple[float, float]] = None  # (x, y) at finish
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'lane': self.lane,
            'method': self.method,
            'hand_position': self.hand_position,
            'metadata': self.metadata
        }


class TopButtonDetector:
    """
    Detects top button press using visual cues.

    The STOP DEVICE (top button) in IFSC walls typically:
    - Changes color when pressed (red → green, or lights up)
    - Located at top of wall (~15m height)
    - Width: ~300mm, centered in lane
    """

    def __init__(
        self,
        top_roi_fraction: float = 0.15,
        button_color_change_threshold: float = 30.0
    ):
        """
        Args:
            top_roi_fraction: Fraction of frame height to consider as "top"
            button_color_change_threshold: Threshold for color change detection
        """
        self.top_roi_fraction = top_roi_fraction
        self.button_color_change_threshold = button_color_change_threshold

    def detect_color_change(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        lane: str
    ) -> Tuple[bool, float]:
        """
        Detect color change in top button region.

        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            lane: 'left' or 'right'

        Returns:
            (detected, confidence)
        """
        height, width = curr_frame.shape[:2]

        # Top ROI
        top_y_start = 0
        top_y_end = int(height * self.top_roi_fraction)

        # Lane ROI
        if lane == 'left':
            lane_x_start = 0
            lane_x_end = width // 2
        elif lane == 'right':
            lane_x_start = width // 2
            lane_x_end = width
        else:
            lane_x_start = 0
            lane_x_end = width

        # Extract ROI
        prev_roi = prev_frame[top_y_start:top_y_end, lane_x_start:lane_x_end]
        curr_roi = curr_frame[top_y_start:top_y_end, lane_x_start:lane_x_end]

        # Convert to HSV for better color detection
        prev_hsv = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2HSV)
        curr_hsv = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2HSV)

        # Compute difference in Hue channel (color)
        hue_diff = cv2.absdiff(curr_hsv[:, :, 0], prev_hsv[:, :, 0])

        # Significant change?
        mean_diff = np.mean(hue_diff)

        detected = mean_diff > self.button_color_change_threshold
        confidence = min(mean_diff / (self.button_color_change_threshold * 2), 1.0)

        return detected, float(confidence)


class PoseBasedFinishDetector:
    """
    Detects finish using pose estimation data.

    Finish criteria:
    - Hand (wrist or hand keypoint) reaches top threshold (y < top_threshold)
    - Hand is in correct horizontal range (within lane)
    - Arm is extended upward
    """

    def __init__(
        self,
        top_threshold_fraction: float = 0.1,
        min_confidence: float = 0.5,
        arm_extension_threshold: float = 0.7
    ):
        """
        Args:
            top_threshold_fraction: Fraction of frame height considered "top"
            min_confidence: Minimum pose confidence
            arm_extension_threshold: Minimum arm extension ratio
        """
        self.top_threshold_fraction = top_threshold_fraction
        self.min_confidence = min_confidence
        self.arm_extension_threshold = arm_extension_threshold

    def check_hand_at_top(
        self,
        pose: Dict[str, Any],
        frame_height: int,
        lane_bounds: Optional[Tuple[float, float]] = None
    ) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
        """
        Check if hand is at top of wall.

        Args:
            pose: Pose dictionary with keypoints
            frame_height: Frame height in pixels
            lane_bounds: (x_min, x_max) for lane in normalized coords

        Returns:
            (at_top, confidence, hand_position)
        """
        # Get hand keypoints (try both wrists and hands)
        keypoint_names = ['left_wrist', 'right_wrist', 'left_hand', 'right_hand']

        best_hand = None
        best_y = float('inf')
        best_conf = 0.0

        for kp_name in keypoint_names:
            if kp_name in pose['keypoints']:
                kp = pose['keypoints'][kp_name]
                x, y, conf = kp['x'], kp['y'], kp.get('confidence', 0)

                if conf < self.min_confidence:
                    continue

                # Check if in lane (if bounds provided)
                if lane_bounds:
                    if not (lane_bounds[0] <= x <= lane_bounds[1]):
                        continue

                # Track highest (lowest y) hand
                if y < best_y:
                    best_y = y
                    best_hand = (x, y)
                    best_conf = conf

        if best_hand is None:
            return False, 0.0, None

        # Convert to pixel coordinates
        best_y_px = best_y * frame_height

        # Check if at top
        top_threshold_px = frame_height * self.top_threshold_fraction
        at_top = best_y_px < top_threshold_px

        # Confidence based on how close to top
        distance_from_top = max(best_y_px - 0, 0)  # Distance from very top
        confidence = 1.0 - min(distance_from_top / top_threshold_px, 1.0)
        confidence *= best_conf  # Weight by pose confidence

        return at_top, float(confidence), best_hand

    def detect_from_poses(
        self,
        poses: List[Dict[str, Any]],
        fps: float,
        frame_height: int,
        lane: str = 'unknown'
    ) -> Optional[RaceFinishResult]:
        """
        Detect finish from pose sequence.

        Args:
            poses: List of pose dictionaries (one per frame)
            fps: Video FPS
            frame_height: Frame height in pixels
            lane: 'left', 'right', or 'unknown'

        Returns:
            RaceFinishResult or None
        """
        # Determine lane bounds
        if lane == 'left':
            lane_bounds = (0.0, 0.5)
        elif lane == 'right':
            lane_bounds = (0.5, 1.0)
        else:
            lane_bounds = None

        # Scan poses for finish
        for frame_id, pose in enumerate(poses):
            if pose is None:
                continue

            at_top, confidence, hand_pos = self.check_hand_at_top(
                pose,
                frame_height,
                lane_bounds
            )

            if at_top and confidence > self.min_confidence:
                # Finish detected!
                timestamp = frame_id / fps

                return RaceFinishResult(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    confidence=confidence,
                    lane=lane,
                    method='pose',
                    hand_position=hand_pos,
                    metadata={'pose_confidence': confidence}
                )

        return None


class RaceFinishDetector:
    """
    Main race finish detector.

    Combines multiple detection methods:
    - Pose-based: Hand reaches top
    - Visual: Button color change
    - Combined: Fusion of both
    """

    def __init__(
        self,
        method: str = 'combined',
        pose_weight: float = 0.7,
        visual_weight: float = 0.3
    ):
        """
        Args:
            method: Detection method ('pose', 'visual', 'combined')
            pose_weight: Weight for pose in combined mode
            visual_weight: Weight for visual in combined mode
        """
        if method not in ['pose', 'visual', 'combined']:
            raise ValueError(f"Invalid method: {method}")

        self.method = method
        self.pose_weight = pose_weight
        self.visual_weight = visual_weight

        self.pose_detector = PoseBasedFinishDetector()
        self.visual_detector = TopButtonDetector()

    def detect_from_video(
        self,
        video_path: Path,
        lane: str = 'unknown',
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        poses: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[RaceFinishResult]:
        """
        Detect race finish from video.

        Args:
            video_path: Path to video file
            lane: 'left', 'right', or 'unknown'
            start_frame: Frame to start search
            end_frame: Frame to end search (None = until end)
            poses: Pre-computed poses (optional)

        Returns:
            RaceFinishResult or None
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if end_frame is None:
            end_frame = total_frames

        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Initialize result variables
        pose_result = None
        visual_result = None

        # If poses available, use pose-based detection
        if poses and self.method in ['pose', 'combined']:
            pose_result = self.pose_detector.detect_from_poses(
                poses[start_frame:end_frame],
                fps,
                height,
                lane
            )

            if self.method == 'pose':
                cap.release()
                return pose_result

        # Visual detection
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return None

        frame_id = start_frame + 1

        while frame_id < end_frame:
            ret, curr_frame = cap.read()
            if not ret:
                break

            # Check for button press
            detected, confidence = self.visual_detector.detect_color_change(
                prev_frame,
                curr_frame,
                lane
            )

            if detected and confidence > 0.5:
                # Finish detected!
                timestamp = frame_id / fps

                visual_result = RaceFinishResult(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    confidence=confidence,
                    lane=lane,
                    method='visual',
                    metadata={'visual_confidence': confidence}
                )
                break

            prev_frame = curr_frame
            frame_id += 1

        cap.release()

        # Combine results if in combined mode
        if self.method == 'combined' and poses:
            if pose_result and visual_result:
                # Both available - use weighted average
                fused_frame = int(
                    pose_result.frame_id * self.pose_weight +
                    visual_result.frame_id * self.visual_weight
                )
                fused_time = (
                    pose_result.timestamp * self.pose_weight +
                    visual_result.timestamp * self.visual_weight
                )
                fused_conf = (
                    pose_result.confidence * self.pose_weight +
                    visual_result.confidence * self.visual_weight
                )

                return RaceFinishResult(
                    frame_id=fused_frame,
                    timestamp=fused_time,
                    confidence=fused_conf,
                    lane=lane,
                    method='combined',
                    hand_position=pose_result.hand_position,
                    metadata={
                        'pose_confidence': pose_result.confidence,
                        'visual_confidence': visual_result.confidence
                    }
                )
            elif pose_result:
                return pose_result
            elif visual_result:
                return visual_result

        return visual_result if self.method == 'visual' else pose_result

    def detect_winner(
        self,
        left_result: Optional[RaceFinishResult],
        right_result: Optional[RaceFinishResult]
    ) -> Optional[str]:
        """
        Determine winner from two finish results.

        Args:
            left_result: Finish result for left climber
            right_result: Finish result for right climber

        Returns:
            'left', 'right', or None (if tie or invalid)
        """
        if left_result is None and right_result is None:
            return None

        if left_result is None:
            return 'right'

        if right_result is None:
            return 'left'

        # Compare timestamps
        time_diff = left_result.timestamp - right_result.timestamp

        if abs(time_diff) < 0.05:  # Within 50ms = tie
            return None

        return 'left' if left_result.timestamp < right_result.timestamp else 'right'


# CLI interface
if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Detect race finish in speed climbing video')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--lane', type=str, default='unknown',
                       choices=['left', 'right', 'unknown'],
                       help='Lane to detect finish for')
    parser.add_argument('--method', type=str, default='visual',
                       choices=['pose', 'visual', 'combined'],
                       help='Detection method')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Frame to start search')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='Frame to end search')
    parser.add_argument('--output', type=str, help='Output JSON file for results')

    args = parser.parse_args()

    print(f"Detecting race finish in: {args.video}")
    print(f"Lane: {args.lane}")
    print(f"Method: {args.method}")

    detector = RaceFinishDetector(method=args.method)
    result = detector.detect_from_video(
        Path(args.video),
        lane=args.lane,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )

    if result:
        print(f"\nRace finish detected!")
        print(f"  Frame: {result.frame_id}")
        print(f"  Time: {result.timestamp:.2f}s")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Lane: {result.lane}")
        print(f"  Method: {result.method}")
        if result.hand_position:
            print(f"  Hand position: ({result.hand_position[0]:.3f}, {result.hand_position[1]:.3f})")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        print("\nNo race finish detected")
        sys.exit(1)
