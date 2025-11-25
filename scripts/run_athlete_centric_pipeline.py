#!/usr/bin/env python3
"""
Test script for the new Athlete-Centric Pipeline.

This script demonstrates the new approach that:
1. Uses athlete pose as the primary reference (not holds)
2. Handles dual-lane racing (left and right climbers)
3. Detects actual race segments within video
4. Estimates wall position relative to camera
5. Calculates cumulative climbing distance

Usage:
    python scripts/run_athlete_centric_pipeline.py <video_path> [--output <output_dir>]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from speed_climbing.vision.pose import BlazePoseExtractor
from speed_climbing.vision.lanes import DualLaneDetector
from speed_climbing.processing.athlete_centric import (
    AthleteCentricPipeline,
    RacePhase
)
from speed_climbing.core.settings import CONFIGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_dual_lane_poses(
    frame: np.ndarray,
    boundary,
    left_extractor,
    right_extractor,
    frame_id: int,
    timestamp: float
) -> tuple:
    """
    Extract poses for both lanes using expanded ROI with padding.

    Instead of hard-splitting the frame, we:
    1. Extract each lane with 20% overlap into the other lane
    2. This gives BlazePose more context for better detection
    3. Adjust coordinates back to normalized [0,1] space relative to original frame
    """
    frame_height, frame_width = frame.shape[:2]
    boundary_x = boundary.x_pixel

    # Calculate ROI with padding (20% overlap)
    padding = int(frame_width * 0.10)  # 10% padding on each side of boundary

    # Left ROI: from 0 to boundary + padding
    left_end = min(boundary_x + padding, frame_width)
    left_roi = frame[:, 0:left_end]

    # Right ROI: from boundary - padding to end
    right_start = max(boundary_x - padding, 0)
    right_roi = frame[:, right_start:]

    # Process left lane
    left_pose = left_extractor.process_frame(left_roi, frame_id=frame_id, timestamp=timestamp)
    left_pose_dict = pose_to_dict_with_roi_adjustment(
        left_pose,
        roi_x_offset=0,
        roi_width=left_end,
        frame_width=frame_width,
        frame_height=frame_height
    )

    # Process right lane
    right_pose = right_extractor.process_frame(right_roi, frame_id=frame_id, timestamp=timestamp)
    right_pose_dict = pose_to_dict_with_roi_adjustment(
        right_pose,
        roi_x_offset=right_start,
        roi_width=frame_width - right_start,
        frame_width=frame_width,
        frame_height=frame_height
    )

    return left_pose_dict, right_pose_dict


def pose_to_dict_with_roi_adjustment(
    pose_result,
    roi_x_offset: int,
    roi_width: int,
    frame_width: int,
    frame_height: int
) -> dict:
    """
    Convert PoseResult to dict and adjust coordinates for ROI.

    Since we process a cropped ROI, we need to adjust the x coordinates
    back to the full frame coordinate system.
    """
    if pose_result is None:
        return {'has_detection': False}

    result = {
        'has_detection': pose_result.has_detection,
        'overall_confidence': pose_result.overall_confidence,
        'keypoints': {}
    }

    if pose_result.has_detection:
        for name, kp in pose_result.keypoints.items():
            # Adjust x coordinate: convert from ROI-relative to frame-relative
            # kp.x is normalized [0,1] within the ROI
            # We need to convert to normalized [0,1] within the full frame
            adjusted_x = (kp.x * roi_width + roi_x_offset) / frame_width

            # Store adjusted keypoint
            result['keypoints'][name] = type('Keypoint', (), {
                'x': adjusted_x,
                'y': kp.y,  # y doesn't need adjustment (same height)
                'z': kp.z,
                'confidence': kp.confidence,
                'visibility': kp.visibility,
                'name': kp.name
            })()

        # Add COM separately for easy access
        com = result['keypoints'].get('COM')
        if com:
            result['com'] = {
                'x': com.x,
                'y': com.y,
                'confidence': com.confidence,
                'visibility': com.visibility
            }

    return result


def process_video(
    video_path: str,
    output_dir: str = None,
    max_frames: int = None,
    visualize: bool = False
) -> dict:
    """
    Process a race video using the athlete-centric pipeline.

    Args:
        video_path: Path to video file
        output_dir: Optional output directory for results
        max_frames: Maximum frames to process (for testing)
        visualize: Whether to show live visualization

    Returns:
        Dict with processing results
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    logger.info(f"Processing video: {video_path}")

    # Initialize components
    route_map_path = CONFIGS_DIR / "ifsc_route_coordinates.json"
    pipeline = AthleteCentricPipeline(str(route_map_path))

    lane_detector = DualLaneDetector(boundary_detection_method="fixed")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {total_frames} frames, {fps} fps, {frame_width}x{frame_height}")

    if max_frames:
        total_frames = min(total_frames, max_frames)

    # Create two separate pose extractors for each lane
    # This helps with MediaPipe's internal tracking
    left_pose_extractor = BlazePoseExtractor(
        model_complexity=1,
        min_detection_confidence=0.3,  # Lower threshold for better detection
        min_tracking_confidence=0.3
    )
    right_pose_extractor = BlazePoseExtractor(
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # Process frames
    frame_id = 0
    processing_times = []

    try:
        while frame_id < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = cv2.getTickCount()
            timestamp = frame_id / fps

            # Detect lane boundary
            boundary = lane_detector.detect_lane_boundary(frame)

            # Extract poses using expanded ROI approach
            # Instead of hard split, use overlapping regions with padding
            left_pose_dict, right_pose_dict = extract_dual_lane_poses(
                frame,
                boundary,
                left_pose_extractor,
                right_pose_extractor,
                frame_id,
                timestamp
            )

            # Process frame with pipeline
            results = pipeline.process_frame(
                frame_id=frame_id,
                timestamp=timestamp,
                left_pose=left_pose_dict,
                right_pose=right_pose_dict,
                frame_shape=frame.shape,
                fps=fps
            )

            # Calculate processing time
            end_time = cv2.getTickCount()
            process_time = (end_time - start_time) / cv2.getTickFrequency()
            processing_times.append(process_time)

            # Log progress
            if frame_id % 100 == 0:
                left_state = pipeline.lane_states['left']
                right_state = pipeline.lane_states['right']
                logger.info(
                    f"Frame {frame_id}/{total_frames} | "
                    f"Left: {left_state.phase.value}, h={left_state.current_height_m:.2f}m | "
                    f"Right: {right_state.phase.value}, h={right_state.current_height_m:.2f}m"
                )

            # Optional visualization
            if visualize:
                vis_frame = visualize_frame(frame, boundary, left_pose_dict, right_pose_dict, results)
                cv2.imshow('Athlete-Centric Pipeline', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

    finally:
        cap.release()
        left_pose_extractor.release()
        right_pose_extractor.release()
        if visualize:
            cv2.destroyAllWindows()

    # Get summary
    summary = pipeline.get_summary()

    # Add processing stats
    summary['processing'] = {
        'video_path': str(video_path),
        'total_frames': frame_id,
        'fps': fps,
        'avg_process_time_ms': np.mean(processing_times) * 1000,
        'processing_date': datetime.now().isoformat()
    }

    # Export time series
    time_series = {
        'left': pipeline.export_time_series('left'),
        'right': pipeline.export_time_series('right')
    }

    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_dir / f"{video_path.stem}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to: {summary_path}")

        # Save time series
        ts_path = output_dir / f"{video_path.stem}_time_series.json"
        with open(ts_path, 'w') as f:
            json.dump(time_series, f, indent=2, default=str)
        logger.info(f"Time series saved to: {ts_path}")

    # Print summary
    print_summary(summary)

    return {
        'summary': summary,
        'time_series': time_series
    }


def visualize_frame(frame, boundary, left_pose_dict, right_pose_dict, results):
    """Create visualization of current frame."""
    vis = frame.copy()
    h, w = frame.shape[:2]

    # Draw lane boundary
    cv2.line(vis, (boundary.x_pixel, 0), (boundary.x_pixel, h),
             (255, 255, 0), 2)

    # Draw labels
    cv2.putText(vis, "LEFT", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis, "RIGHT", (boundary.x_pixel + 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw COM positions if detected
    for pose_dict, color in [(left_pose_dict, (0, 255, 0)), (right_pose_dict, (0, 0, 255))]:
        if pose_dict.get('has_detection') and pose_dict.get('com'):
            com = pose_dict['com']
            com_x = int(com['x'] * w)
            com_y = int(com['y'] * h)
            cv2.circle(vis, (com_x, com_y), 10, color, -1)
            cv2.circle(vis, (com_x, com_y), 12, (255, 255, 255), 2)

    # Draw info for each lane
    for lane, result in results.items():
        x_offset = 10 if lane == 'left' else boundary.x_pixel + 10
        y_offset = 60

        # Detection status
        pose_dict = left_pose_dict if lane == 'left' else right_pose_dict
        det_status = "✓" if pose_dict.get('has_detection') else "✗"
        cv2.putText(vis, f"Det: {det_status}", (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Phase
        phase_color = (0, 255, 0) if result.phase == RacePhase.RACING else (255, 255, 255)
        cv2.putText(vis, f"Phase: {result.phase.value}", (x_offset, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1)

        # Height
        if result.height_m is not None:
            cv2.putText(vis, f"H: {result.height_m:.2f}m", (x_offset, y_offset + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Velocity
        if result.velocity_m_s is not None:
            cv2.putText(vis, f"V: {result.velocity_m_s:.2f}m/s", (x_offset, y_offset + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Distance
        if result.cumulative_distance_m is not None:
            cv2.putText(vis, f"D: {result.cumulative_distance_m:.2f}m", (x_offset, y_offset + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def print_summary(summary: dict):
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("RACE ANALYSIS SUMMARY (Athlete-Centric Pipeline)")
    print("=" * 60)

    for lane in ['left', 'right']:
        data = summary.get(lane, {})
        print(f"\n{lane.upper()} LANE:")
        print(f"  Phase: {data.get('phase', 'unknown')}")
        print(f"  Total Distance: {data.get('total_distance_m', 0):.2f} m")
        print(f"  Max Height: {data.get('max_height_m', 0):.2f} m")
        print(f"  Max Velocity: {data.get('max_velocity_m_s', 0):.2f} m/s")

        if data.get('race_duration_s'):
            print(f"  Race Duration: {data.get('race_duration_s'):.2f} s")

        print(f"  Scale Confidence: {data.get('scale_confidence', 0):.2f}")
        print(f"  Frames with Detection: {data.get('frames_with_detection', 0)}/{data.get('frames_processed', 0)}")

    print("\n" + "=" * 60)

    # Validation
    print("\nVALIDATION:")
    for lane in ['left', 'right']:
        data = summary.get(lane, {})
        distance = data.get('total_distance_m', 0)

        # Check if distance is reasonable (should be close to 15m for completed race)
        if distance > 0:
            if 12 <= distance <= 18:
                status = "PASS (reasonable distance)"
            elif distance < 12:
                status = "WARNING (distance too short - incomplete race or calibration issue)"
            else:
                status = "WARNING (distance too long - possible calibration issue)"
        else:
            status = "FAIL (no distance recorded)"

        print(f"  {lane.upper()}: {status}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run athlete-centric pipeline on a speed climbing video"
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--max-frames", "-m", type=int, help="Maximum frames to process")
    parser.add_argument("--visualize", "-v", action="store_true", help="Show live visualization")

    args = parser.parse_args()

    try:
        results = process_video(
            video_path=args.video_path,
            output_dir=args.output,
            max_frames=args.max_frames,
            visualize=args.visualize
        )
        return 0
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
