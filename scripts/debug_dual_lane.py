#!/usr/bin/env python3
"""Debug script to check dual-lane processing"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phase1_pose_estimation.video_processor import VideoProcessor
from phase1_pose_estimation.dual_lane_detector import DualLaneDetector
from phase1_pose_estimation.blazepose_extractor import BlazePoseExtractor
from phase1_pose_estimation.hold_detector import HoldDetector

# Load video
video_path = "data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race001.mp4"
video = VideoProcessor(video_path)

# Initialize components
lane_detector = DualLaneDetector()
pose_extractor = BlazePoseExtractor()
hold_detector = HoldDetector(route_coordinates_path="configs/ifsc_route_coordinates.json")

# Test on frame 100 (middle of race)
for frame_data in video.extract_frames(100, 101):
    frame = frame_data['frame']

    # Get lane boundary
    lane_boundary = lane_detector.detect_lane_boundary(frame)
    print(f"Lane boundary: x={lane_boundary.x_pixel}px (frame width: {frame.shape[1]}px)")
    print(f"Left lane: 0-{lane_boundary.x_pixel}px")
    print(f"Right lane: {lane_boundary.x_pixel}-{frame.shape[1]}px")

    # Detect holds for each lane
    for lane in ['left', 'right']:
        holds = hold_detector.detect_holds(frame, lane=lane)
        print(f"\n{lane.upper()} lane: {len(holds)} holds detected")
        if holds:
            for i, hold in enumerate(holds[:3]):
                print(f"  {i+1}. pos=({hold.pixel_x:.0f}, {hold.pixel_y:.0f}), conf={hold.confidence:.2f}")

    # Detect pose (full frame - should detect both climbers)
    pose_result = pose_extractor.process_frame(frame, frame_id=100, timestamp=3.33)
    print(f"\nPose detection: {pose_result.has_detection}")
    if pose_result.has_detection:
        left_hip = pose_result.keypoints.get('left_hip')
        right_hip = pose_result.keypoints.get('right_hip')
        if left_hip and right_hip:
            com_x = (left_hip.x + right_hip.x) / 2 * frame.shape[1]
            com_y = (left_hip.y + right_hip.y) / 2 * frame.shape[0]
            print(f"COM position: ({com_x:.0f}, {com_y:.0f}) px")

            # Check which lane
            if com_x < lane_boundary.x_pixel:
                print(f"  → LEFT lane (x={com_x:.0f} < {lane_boundary.x_pixel})")
            else:
                print(f"  → RIGHT lane (x={com_x:.0f} >= {lane_boundary.x_pixel})")

print("\nDone!")
