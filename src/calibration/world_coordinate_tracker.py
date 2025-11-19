#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World Coordinate Tracker for Global Map Registration
=====================================================

این ماژول tracking موقعیت ورزشکار در مختصات جهانی (meters) است.

Features:
- Per-frame hold detection
- Periodic camera calibration (every N frames)
- Pixel→Meter transformation
- Calibration quality monitoring
- Fallback to last valid calibration

Key concept: مستقل از حرکت دوربین، موقعیت را در مختصات دیوار track می‌کنیم.

Author: Speed Climbing Performance Analysis
Date: 2025-11-19
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration.camera_calibration import PeriodicCalibrator, CalibrationResult
from phase1_pose_estimation.hold_detector import HoldDetector


@dataclass
class WorldCoordinates:
    """مختصات جهانی برای یک frame."""

    y_position_m: Optional[float]  # ارتفاع از پد شروع (meters)
    x_position_m: Optional[float]  # فاصله افقی از مرکز lane (meters)
    calibration_quality: float  # کیفیت calibration (0-1)
    calibration_rmse_m: Optional[float]  # RMSE در متر
    is_valid: bool  # آیا این coordinates معتبر است؟
    reason: Optional[str] = None  # اگر نامعتبر، دلیل چیست؟


class WorldCoordinateTracker:
    """
    Track athlete position در world coordinates (meters).

    این tracker از periodic calibration استفاده می‌کند تا overhead محاسباتی را کاهش دهد
    در حالی که همچنان با camera movement سازگار است.

    Usage:
        tracker = WorldCoordinateTracker(
            route_map_path='configs/ifsc_route_coordinates.json',
            recalibration_interval=15
        )

        for frame_id, frame in enumerate(video_frames):
            # Get COM from pose (assumed already extracted)
            com_x_px, com_y_px = get_athlete_com(frame)

            # Transform to world coordinates
            world_coords = tracker.process_frame(
                frame=frame,
                frame_id=frame_id,
                com_x_px=com_x_px,
                com_y_px=com_y_px,
                lane='left'
            )

            if world_coords.is_valid:
                print(f"Height: {world_coords.y_position_m:.2f}m")
    """

    def __init__(
        self,
        route_map_path: str,
        recalibration_interval: int = 15,  # frames
        wall_height_m: float = 15.0,
        start_pad_height_m: float = 0.0,  # برای سادگی: 0m = ground level
        min_holds_for_calibration: int = 4,
        min_calibration_confidence: float = 0.6
    ):
        """
        Initialize world coordinate tracker.

        Args:
            route_map_path: مسیر JSON حاوی IFSC route map
            recalibration_interval: هر چند frame یکبار calibrate کنیم
            wall_height_m: ارتفاع کل دیوار (meters)
            start_pad_height_m: ارتفاع پد شروع از زمین (meters)
            min_holds_for_calibration: حداقل holds مورد نیاز برای calibration
            min_calibration_confidence: حداقل confidence برای calibration معتبر
        """
        self.route_map_path = route_map_path
        self.wall_height_m = wall_height_m
        self.start_pad_height_m = start_pad_height_m
        self.min_calibration_confidence = min_calibration_confidence

        # Initialize hold detector
        self.hold_detector = HoldDetector(
            route_coordinates_path=route_map_path,
            min_area=200,
            max_area=50000,
            min_confidence=0.2  # Lower threshold برای recall بهتر
        )

        # Initialize periodic calibrator
        self.calibrator = PeriodicCalibrator(
            route_coordinates_path=route_map_path,
            recalibration_interval=recalibration_interval,
            min_holds_for_calibration=min_holds_for_calibration,
            ransac_threshold=0.05,  # 5cm
            min_inlier_ratio=0.5,
            min_confidence_for_cache=min_calibration_confidence
        )

        # Statistics
        self.total_frames_processed = 0
        self.successful_calibrations = 0
        self.failed_calibrations = 0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        com_x_px: float,
        com_y_px: float,
        lane: str = 'left'
    ) -> WorldCoordinates:
        """
        پردازش یک frame و تبدیل COM از pixels به meters.

        Args:
            frame: Video frame (BGR)
            frame_id: Frame index (0-based)
            com_x_px: Center of mass X در pixel coordinates
            com_y_px: Center of mass Y در pixel coordinates
            lane: 'left' or 'right'

        Returns:
            WorldCoordinates object
        """
        self.total_frames_processed += 1

        # Step 1: Detect holds در frame
        detected_holds = self.hold_detector.detect_holds(frame, lane=lane)

        if len(detected_holds) < 3:
            # Not enough holds - نمی‌توانیم calibrate کنیم
            return WorldCoordinates(
                y_position_m=None,
                x_position_m=None,
                calibration_quality=0.0,
                calibration_rmse_m=None,
                is_valid=False,
                reason=f'insufficient_holds ({len(detected_holds)} detected, need ≥3)'
            )

        # Step 2: Calibrate (periodic - not every frame)
        calibration = self.calibrator.calibrate_frame(
            frame=frame,
            frame_id=frame_id,
            detected_holds=detected_holds,
            lane=lane
        )

        if calibration is None:
            self.failed_calibrations += 1
            return WorldCoordinates(
                y_position_m=None,
                x_position_m=None,
                calibration_quality=0.0,
                calibration_rmse_m=None,
                is_valid=False,
                reason='calibration_failed'
            )

        # Check calibration quality
        if calibration.confidence < self.min_calibration_confidence:
            self.failed_calibrations += 1
            return WorldCoordinates(
                y_position_m=None,
                x_position_m=None,
                calibration_quality=calibration.confidence,
                calibration_rmse_m=calibration.rmse_error,
                is_valid=False,
                reason=f'low_calibration_confidence ({calibration.confidence:.2f})'
            )

        self.successful_calibrations += 1

        # Step 3: Transform COM از pixels به meters
        com_x_m, com_y_m = calibration.pixel_to_meter_func(com_x_px, com_y_px)

        # Step 4: Convert to wall-relative coordinates
        # Wall coordinate system: Y=0 at bottom, Y=15 at top
        # We want: Y=0 at start pad, Y=15 at top button
        # So: y_from_start = wall_height - wall_y_coord

        y_from_start_m = self.wall_height_m - com_y_m

        # X coordinate: distance from center of lane
        # Lane width = 1.5m (half of 3m wall width)
        x_from_center_m = com_x_m - (1.5 if lane == 'left' else 4.5)

        return WorldCoordinates(
            y_position_m=y_from_start_m,
            x_position_m=x_from_center_m,
            calibration_quality=calibration.confidence,
            calibration_rmse_m=calibration.rmse_error,
            is_valid=True,
            reason=None
        )

    def get_stats(self) -> dict:
        """آمار calibration."""
        success_rate = (self.successful_calibrations / self.total_frames_processed
                       if self.total_frames_processed > 0 else 0.0)

        # Get calibrator cache stats
        cache_stats = self.calibrator.get_cache_stats()

        return {
            'total_frames_processed': self.total_frames_processed,
            'successful_calibrations': self.successful_calibrations,
            'failed_calibrations': self.failed_calibrations,
            'success_rate': success_rate,
            'calibration_cache': cache_stats
        }

    def reset(self):
        """ریست کردن tracker برای video جدید."""
        # Note: calibrator cache خودش reset می‌شود
        self.total_frames_processed = 0
        self.successful_calibrations = 0
        self.failed_calibrations = 0


# Testing
if __name__ == '__main__':
    import cv2
    import argparse

    print("=" * 70)
    print("WorldCoordinateTracker Test")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video',
        default='data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race001.mp4',
        help='Test video path'
    )
    parser.add_argument(
        '--route-map',
        default='configs/ifsc_route_coordinates.json',
        help='Route map path'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=10,
        help='Number of frames to test'
    )
    args = parser.parse_args()

    # Initialize tracker
    tracker = WorldCoordinateTracker(
        route_map_path=args.route_map,
        recalibration_interval=5  # Every 5 frames for testing
    )

    # Load video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {args.video}")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo: {frame_width}x{frame_height} @ {fps:.1f} fps")
    print(f"Testing {args.frames} frames...\n")

    # Process frames
    valid_count = 0
    for frame_id in range(args.frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Simulate COM (center of frame for testing)
        com_x_px = frame_width / 2
        com_y_px = frame_height / 2

        # Process
        world_coords = tracker.process_frame(
            frame=frame,
            frame_id=frame_id,
            com_x_px=com_x_px,
            com_y_px=com_y_px,
            lane='left'
        )

        # Display
        if world_coords.is_valid:
            valid_count += 1
            print(f"Frame {frame_id:3d}: "
                  f"y={world_coords.y_position_m:5.2f}m, "
                  f"x={world_coords.x_position_m:5.2f}m, "
                  f"quality={world_coords.calibration_quality:.2f}, "
                  f"RMSE={world_coords.calibration_rmse_m*100:.1f}cm")
        else:
            print(f"Frame {frame_id:3d}: INVALID - {world_coords.reason}")

    cap.release()

    # Statistics
    print("\n" + "=" * 70)
    stats = tracker.get_stats()
    print("Statistics:")
    print(f"  Frames processed: {stats['total_frames_processed']}")
    print(f"  Successful: {stats['successful_calibrations']}")
    print(f"  Failed: {stats['failed_calibrations']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")

    if valid_count > 0:
        print(f"\n✅ Test passed! {valid_count}/{args.frames} frames valid")
    else:
        print(f"\n⚠️  No valid frames - check video/route-map")

    print("=" * 70)
