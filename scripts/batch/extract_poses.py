"""
Pose extraction script for specific videos.

Extracts BlazePose keypoints from videos and saves them as JSON files.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from speed_climbing.vision.pose import BlazePoseExtractor
from speed_climbing.vision.lanes import DualLaneDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def pose_to_dict(pose, roi_x_offset: int, roi_width: int,
                 frame_width: int, frame_height: int) -> Dict[str, Any]:
    """Convert pose result to dictionary with coordinate adjustment."""
    if pose is None or not pose.keypoints:
        return {
            "has_detection": False,
            "overall_confidence": 0.0,
            "keypoints": {}
        }

    adjusted_keypoints = {}
    for name, kp in pose.keypoints.items():
        # Adjust x coordinate from ROI to full frame
        original_x_pixel = kp.x * roi_width
        adjusted_x_pixel = original_x_pixel + roi_x_offset
        adjusted_x_normalized = adjusted_x_pixel / frame_width

        adjusted_keypoints[name] = {
            "name": name,
            "x": adjusted_x_normalized,
            "y": kp.y,
            "z": kp.z,
            "confidence": kp.confidence,
            "visibility": kp.visibility
        }

    return {
        "has_detection": True,
        "overall_confidence": pose.overall_confidence,
        "keypoints": adjusted_keypoints
    }


def extract_poses_from_video(video_path: Path, output_path: Path) -> Dict[str, Any]:
    """Extract poses from a video and save to JSON."""
    logger.info(f"Processing: {video_path.name}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"  {total_frames} frames, {fps} fps, {frame_width}x{frame_height}")

    # Initialize extractors
    lane_detector = DualLaneDetector()
    left_extractor = BlazePoseExtractor(
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    right_extractor = BlazePoseExtractor(
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # Process frames
    frames_data = []
    frames_with_left = 0
    frames_with_right = 0
    frames_with_both = 0

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps

        # Detect lane boundary
        boundary = lane_detector.detect_lane_boundary(frame)
        boundary_x = boundary.x_pixel

        # Calculate ROI with padding
        padding = int(frame_width * 0.10)

        # Left ROI
        left_end = min(boundary_x + padding, frame_width)
        left_roi = frame[:, 0:left_end]

        # Right ROI
        right_start = max(boundary_x - padding, 0)
        right_roi = frame[:, right_start:]

        # Extract poses
        left_pose = left_extractor.process_frame(left_roi, frame_id=frame_id, timestamp=timestamp)
        right_pose = right_extractor.process_frame(right_roi, frame_id=frame_id, timestamp=timestamp)

        # Convert to dictionaries
        left_dict = pose_to_dict(left_pose, 0, left_end, frame_width, frame_height)
        right_dict = pose_to_dict(right_pose, right_start, frame_width - right_start, frame_width, frame_height)

        # Track detection stats
        if left_dict["has_detection"]:
            frames_with_left += 1
        if right_dict["has_detection"]:
            frames_with_right += 1
        if left_dict["has_detection"] and right_dict["has_detection"]:
            frames_with_both += 1

        # Store frame data
        frames_data.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "left_climber": {
                "frame_id": frame_id,
                "timestamp": timestamp,
                **left_dict
            },
            "right_climber": {
                "frame_id": frame_id,
                "timestamp": timestamp,
                **right_dict
            }
        })

        frame_id += 1

        if frame_id % 50 == 0:
            logger.info(f"  Processed {frame_id}/{total_frames} frames")

    cap.release()

    # Create output
    output_data = {
        "metadata": {
            "video_path": video_path.name,
            "fps": fps,
            "width": frame_width,
            "height": frame_height,
            "total_frames": total_frames,
            "frames_with_left_pose": frames_with_left,
            "frames_with_right_pose": frames_with_right,
            "frames_with_both_poses": frames_with_both,
            "detection_rate_left": frames_with_left / total_frames if total_frames > 0 else 0,
            "detection_rate_right": frames_with_right / total_frames if total_frames > 0 else 0,
            "detection_rate_both": frames_with_both / total_frames if total_frames > 0 else 0
        },
        "frames": frames_data
    }

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"  Saved to: {output_path}")
    logger.info(f"  Detection rates: Left={output_data['metadata']['detection_rate_left']:.1%}, "
                f"Right={output_data['metadata']['detection_rate_right']:.1%}")

    return output_data


def main():
    """Main entry point."""
    # Videos to re-extract (replaced Villars videos)
    videos_to_process = [
        "Speed_finals_Villars_2024_race002.mp4",
        "Speed_finals_Villars_2024_race013.mp4",
        "Speed_finals_Villars_2024_race015.mp4",
        "Speed_finals_Villars_2024_race019.mp4",
        "Speed_finals_Villars_2024_race023.mp4",
    ]

    videos_dir = project_root / "data" / "race_segments" / "villars_2024"
    poses_dir = project_root / "data" / "processed" / "poses" / "villars_2024"

    logger.info(f"Re-extracting poses for {len(videos_to_process)} videos...")

    for video_name in videos_to_process:
        video_path = videos_dir / video_name
        output_name = video_name.replace(".mp4", "_poses.json")
        output_path = poses_dir / output_name

        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            continue

        try:
            extract_poses_from_video(video_path, output_path)
        except Exception as e:
            logger.error(f"Error processing {video_name}: {e}")

    logger.info("Pose extraction complete!")


if __name__ == "__main__":
    main()
