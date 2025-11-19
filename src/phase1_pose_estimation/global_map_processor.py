#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Map Video Processor - Integrated Pipeline
=================================================

این ماژول pipeline یکپارچه برای پردازش ویدئوهای سنگنوردی سرعتی است.

Pipeline stages:
1. Video loading and frame extraction
2. Lane detection (left/right separation)
3. Pose estimation (BlazePose for both lanes)
4. World coordinate tracking (pixel→meter transformation)
5. Dropout detection (finished/DNF/out_of_frame)
6. Time-series building and JSON output

Features:
- Handles dual-lane videos (simultaneous tracking)
- Moving camera support (per-frame calibration)
- Dropout management (independent tracking per lane)
- Metric units (meters, m/s)
- Race boundaries filtering

Author: Speed Climbing Performance Analysis
Date: 2025-11-19
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import logging

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calibration.world_coordinate_tracker import WorldCoordinateTracker
from calibration.dropout_handler import DropoutHandler
from calibration.time_series_builder import TimeSeriesBuilder, TimeSeriesData, save_time_series
from phase1_pose_estimation.blazepose_extractor import BlazePoseExtractor
from phase1_pose_estimation.video_processor import VideoProcessor
from phase1_pose_estimation.dual_lane_detector import DualLaneDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalMapVideoProcessor:
    """
    Integrated pipeline برای پردازش race videos با Global Map Registration.

    Usage:
        processor = GlobalMapVideoProcessor(
            route_map_path='configs/ifsc_route_coordinates.json'
        )

        result = processor.process_race(
            video_path='data/race_segments/seoul_2024/race001.mp4',
            race_metadata={
                'race_id': 'Seoul_2024_race001',
                'competition': 'Seoul_2024',
                'detected_start_frame': 0,
                'detected_finish_frame': 180
            }
        )

        # Save result
        save_time_series(
            left_data=result['left_climber'],
            right_data=result['right_climber'],
            output_path='output.json',
            metadata=result['metadata']
        )
    """

    def __init__(
        self,
        route_map_path: str,
        recalibration_interval: int = 15,
        timeout_frames: int = 30,
        wall_height_m: float = 15.0
    ):
        """
        Initialize processor.

        Args:
            route_map_path: مسیر IFSC route map JSON
            recalibration_interval: هر چند frame calibrate کنیم
            timeout_frames: تعداد frames برای timeout در dropout detection
            wall_height_m: ارتفاع دیوار (meters)
        """
        self.route_map_path = route_map_path
        self.wall_height_m = wall_height_m

        # Initialize components
        logger.info("Initializing Global Map Video Processor...")

        # Pose estimator
        self.pose_extractor = BlazePoseExtractor(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info("  ✓ BlazePose extractor initialized")

        # Lane detector
        self.lane_detector = DualLaneDetector()
        logger.info("  ✓ Dual lane detector initialized")

        # World coordinate trackers (one per lane)
        self.trackers = {
            'left': WorldCoordinateTracker(
                route_map_path=route_map_path,
                recalibration_interval=recalibration_interval,
                wall_height_m=wall_height_m
            ),
            'right': WorldCoordinateTracker(
                route_map_path=route_map_path,
                recalibration_interval=recalibration_interval,
                wall_height_m=wall_height_m
            )
        }
        logger.info("  ✓ World coordinate trackers initialized")

        # Dropout handlers (one per lane)
        self.dropout_handlers = {
            'left': DropoutHandler(
                timeout_frames=timeout_frames,
                wall_height_m=wall_height_m
            ),
            'right': DropoutHandler(
                timeout_frames=timeout_frames,
                wall_height_m=wall_height_m
            )
        }
        logger.info("  ✓ Dropout handlers initialized")

        logger.info("✅ Processor ready!")

    def process_race(
        self,
        video_path: str,
        race_metadata: Optional[Dict] = None,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        پردازش یک race video و تولید time-series output.

        Args:
            video_path: مسیر ویدئوی race
            race_metadata: اطلاعات race (race_id, start/end frames, etc.)
            output_path: مسیر خروجی JSON (optional)

        Returns:
            Dictionary حاوی:
                - left_climber: TimeSeriesData
                - right_climber: TimeSeriesData
                - metadata: processing metadata
                - stats: processing statistics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {Path(video_path).name}")
        logger.info(f"{'='*70}")

        # Extract metadata
        race_metadata = race_metadata or {}
        race_id = race_metadata.get('race_id', Path(video_path).stem)
        start_frame = race_metadata.get('detected_start_frame', 0)
        end_frame = race_metadata.get('detected_finish_frame', None)

        # Load video
        logger.info("\nStep 1: Loading video...")
        try:
            video = VideoProcessor(video_path)
        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            raise

        total_frames = video.total_frames
        fps = video.fps

        # Adjust end frame if not specified
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames

        logger.info(f"  Video: {video.width}x{video.height} @ {fps:.1f} fps")
        logger.info(f"  Processing frames: {start_frame} - {end_frame} ({end_frame - start_frame} frames)")

        # Initialize time-series builders
        builders = {
            'left': TimeSeriesBuilder(lane='left'),
            'right': TimeSeriesBuilder(lane='right')
        }

        # Reset handlers
        for lane in ['left', 'right']:
            self.dropout_handlers[lane].reset()
            self.trackers[lane].reset()

        # Process frames
        logger.info("\nStep 2: Processing frames...")
        frames_processed = 0
        frames_with_valid_data = {'left': 0, 'right': 0}

        for frame_data in video.extract_frames(start_frame, end_frame):
            frame_id = frame_data['frame_id']
            timestamp = frame_data['timestamp'] - (start_frame / fps)  # Relative to race start
            frame = frame_data['frame']

            frames_processed += 1

            # Progress
            if frames_processed % 30 == 0:
                logger.info(f"  Processed {frames_processed}/{end_frame - start_frame} frames...")

            # Detect lane boundary
            lane_boundary = self.lane_detector.detect_boundary(frame)

            # Process each lane
            for lane in ['left', 'right']:
                # Check if already dropped out
                if builders[lane].is_finished():
                    continue

                # Extract pose for this lane
                pose_result = self.pose_extractor.extract_pose(
                    frame,
                    return_annotated=False
                )

                has_pose = pose_result is not None and pose_result.is_valid

                # Calculate COM if pose is valid
                if has_pose:
                    # Get hip position as COM (simple approximation)
                    left_hip = pose_result.keypoints.get('left_hip')
                    right_hip = pose_result.keypoints.get('right_hip')

                    if left_hip and right_hip:
                        # Average of hips
                        com_x_norm = (left_hip['x'] + right_hip['x']) / 2
                        com_y_norm = (left_hip['y'] + right_hip['y']) / 2

                        # Convert to pixels
                        com_x_px = com_x_norm * frame.shape[1]
                        com_y_px = com_y_norm * frame.shape[0]
                    else:
                        has_pose = False
                        com_x_px, com_y_px = None, None
                else:
                    com_x_px, com_y_px = None, None

                # Track world coordinates
                if has_pose and com_x_px is not None:
                    world_coords = self.trackers[lane].process_frame(
                        frame=frame,
                        frame_id=frame_id,
                        com_x_px=com_x_px,
                        com_y_px=com_y_px,
                        lane=lane
                    )
                else:
                    world_coords = None

                # Check dropout
                dropout_status = self.dropout_handlers[lane].check_dropout(
                    has_pose=has_pose,
                    has_calibration=world_coords.is_valid if world_coords else False,
                    y_position_m=world_coords.y_position_m if world_coords and world_coords.is_valid else None,
                    calibration_confidence=world_coords.calibration_quality if world_coords else 0.0
                )

                # Add to time-series
                if world_coords and world_coords.is_valid:
                    builders[lane].add_frame(
                        timestamp=timestamp,
                        y_position_m=world_coords.y_position_m,
                        x_position_m=world_coords.x_position_m,
                        status=dropout_status.status,
                        calibration_quality=world_coords.calibration_quality
                    )
                    frames_with_valid_data[lane] += 1
                else:
                    # Invalid frame - still add but with None values
                    builders[lane].add_frame(
                        timestamp=timestamp,
                        y_position_m=None,
                        x_position_m=None,
                        status='invalid',
                        calibration_quality=None
                    )

        # Build final time-series
        logger.info("\nStep 3: Building time-series...")
        left_data = builders['left'].build()
        right_data = builders['right'].build()

        logger.info(f"  Left climber:  {frames_with_valid_data['left']}/{frames_processed} valid frames")
        logger.info(f"  Right climber: {frames_with_valid_data['right']}/{frames_processed} valid frames")

        # Gather statistics
        stats = {
            'frames_processed': frames_processed,
            'frames_with_valid_data': frames_with_valid_data,
            'left_tracker_stats': self.trackers['left'].get_stats(),
            'right_tracker_stats': self.trackers['right'].get_stats(),
            'left_dropout_stats': self.dropout_handlers['left'].get_stats(),
            'right_dropout_stats': self.dropout_handlers['right'].get_stats()
        }

        # Prepare result
        result = {
            'race_id': race_id,
            'left_climber': left_data,
            'right_climber': right_data,
            'metadata': {
                'video_path': str(video_path),
                'race_id': race_id,
                'competition': race_metadata.get('competition', 'unknown'),
                'processing_method': 'global_map_registration',
                'calibration_method': 'periodic_homography',
                'units': 'meters',
                'wall_height_m': self.wall_height_m,
                'fps': fps,
                'start_frame': start_frame,
                'end_frame': end_frame
            },
            'stats': stats
        }

        # Save if output path provided
        if output_path:
            logger.info(f"\nStep 4: Saving to {output_path}...")
            save_time_series(
                left_data=left_data,
                right_data=right_data,
                output_path=output_path,
                metadata=result['metadata']
            )
            logger.info("  ✓ Saved!")

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Race ID: {race_id}")
        logger.info(f"Frames: {frames_processed}")
        logger.info(f"\nLeft climber:")
        logger.info(f"  Outcome: {left_data.race_outcome}")
        logger.info(f"  Valid frames: {frames_with_valid_data['left']}/{frames_processed} ({frames_with_valid_data['left']/frames_processed:.1%})")
        if left_data.summary:
            logger.info(f"  Time: {left_data.summary.get('total_time_s', 0):.2f}s")
            logger.info(f"  Distance: {left_data.summary.get('total_distance_m', 0):.2f}m")
            logger.info(f"  Avg velocity: {left_data.summary.get('avg_velocity_m_s', 0):.2f} m/s")

        logger.info(f"\nRight climber:")
        logger.info(f"  Outcome: {right_data.race_outcome}")
        logger.info(f"  Valid frames: {frames_with_valid_data['right']}/{frames_processed} ({frames_with_valid_data['right']/frames_processed:.1%})")
        if right_data.summary:
            logger.info(f"  Time: {right_data.summary.get('total_time_s', 0):.2f}s")
            logger.info(f"  Distance: {right_data.summary.get('total_distance_m', 0):.2f}m")
            logger.info(f"  Avg velocity: {right_data.summary.get('avg_velocity_m_s', 0):.2f} m/s")

        logger.info(f"{'='*70}\n")

        return result


# CLI interface for testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Process speed climbing race video with Global Map Registration'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to race video'
    )
    parser.add_argument(
        '--route-map',
        type=str,
        default='configs/ifsc_route_coordinates.json',
        help='Path to IFSC route map JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON path (default: auto-generated)'
    )
    parser.add_argument(
        '--race-id',
        type=str,
        default=None,
        help='Race ID (default: from filename)'
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Start frame (default: 0)'
    )
    parser.add_argument(
        '--end-frame',
        type=int,
        default=None,
        help='End frame (default: end of video)'
    )

    args = parser.parse_args()

    # Auto-generate output path if not provided
    if args.output is None:
        video_path = Path(args.video)
        args.output = f"data/processed/global_map/{video_path.stem}_global_map.json"

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    processor = GlobalMapVideoProcessor(
        route_map_path=args.route_map
    )

    # Process
    result = processor.process_race(
        video_path=args.video,
        race_metadata={
            'race_id': args.race_id or Path(args.video).stem,
            'detected_start_frame': args.start_frame,
            'detected_finish_frame': args.end_frame
        },
        output_path=args.output
    )

    print(f"\n✅ Processing complete!")
    print(f"   Output: {args.output}")
