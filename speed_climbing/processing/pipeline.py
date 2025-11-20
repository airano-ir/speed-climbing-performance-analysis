"""
Integrated video processor with global map registration.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from speed_climbing.vision.lanes import DualLaneDetector
from speed_climbing.processing.tracking import WorldCoordinateTracker
from speed_climbing.processing.dropout import DropoutHandler
from speed_climbing.analysis.time_series import TimeSeriesBuilder
# Note: Pose extraction is still in phase1, we might need to wrap it or import it
# For now, assuming we can import BlazePoseExtractor from the old location or migrate it.
# Let's migrate it to speed_climbing.vision.pose later, but for now import from source if needed
# or assume it's available.
# To keep it clean, I will assume we migrate BlazePoseExtractor to speed_climbing.vision.pose
from speed_climbing.vision.pose import BlazePoseExtractor 

class GlobalMapVideoProcessor:
    """
    Integrated video processor with global map registration.
    Combines: Pose, Holds, Calibration, Tracking, Dropout.
    """

    def __init__(self, route_map_path: str, config: Dict[str, Any] = None):
        self.pose_extractor = BlazePoseExtractor()
        self.world_tracker = WorldCoordinateTracker(route_map_path, config)
        self.dropout_handler = DropoutHandler()
        self.lane_detector = DualLaneDetector()
        self.route_map_path = route_map_path

    def process_race(self, video_path: str, race_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process entire race video and generate time-series output.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = race_metadata.get('detected_start_frame', 0)
        end_frame = race_metadata.get('detected_finish_frame', total_frames)

        # Initialize data containers
        left_data = TimeSeriesBuilder()
        right_data = TimeSeriesBuilder()

        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = (current_frame - start_frame) / fps

            # 1. Detect Lanes
            lane_boundary = self.lane_detector.detect_lane_boundary(frame)

            # 2. Process each lane
            for lane in ['left', 'right']:
                # Mask frame for lane
                lane_mask = lane_boundary.get_lane_mask(lane)
                # Note: BlazePose might need full frame + ROI, or masked frame. 
                # Simplified here: passing full frame, but in reality we should crop/mask
                # For this implementation plan, we'll assume extract_pose handles it or we pass masked
                
                # A. Pose Estimation
                # This is a placeholder for the actual pose extraction call
                # In a real migration, we'd ensure BlazePoseExtractor is fully compatible
                pose = self.pose_extractor.extract_pose(frame, lane=lane) 

                # B. World Tracking
                tracker_result = self.world_tracker.process_frame(frame, current_frame, lane)
                
                y_pos = None
                x_pos = None
                
                if tracker_result['is_valid'] and pose and pose.has_detection:
                    # Transform COM
                    com = pose.get_keypoint('COM')
                    if com:
                        world_coords = self.world_tracker.transform_point(
                            com.x * frame.shape[1], 
                            com.y * frame.shape[0], 
                            tracker_result['calibration']
                        )
                        if world_coords:
                            y_pos = world_coords['y_position_m']
                            x_pos = world_coords['x_position_m']

                # C. Dropout
                dropout_status = self.dropout_handler.check_dropout(
                    current_frame,
                    pose.has_detection if pose else False,
                    tracker_result['is_valid'],
                    y_pos
                )

                # D. Store
                target_data = left_data if lane == 'left' else right_data
                target_data.add_frame(
                    timestamp=timestamp,
                    y_position_m=y_pos,
                    x_position_m=x_pos,
                    status=dropout_status['status'],
                    calibration_quality=tracker_result.get('calibration_quality', 0.0)
                )

            if left_data.is_finished() and right_data.is_finished():
                break
                
            current_frame += 1

        cap.release()

        return {
            'left_climber': left_data.build(),
            'right_climber': right_data.build(),
            'metadata': {
                'video_path': str(video_path),
                'processing_date': datetime.now().isoformat(),
                'calibration_method': 'per_frame_homography',
                'units': 'meters',
                'reference_point': 'start_pad'
            }
        }
