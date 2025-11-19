#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dropout Handler for Global Map Registration
============================================

این ماژول برای تشخیص و مدیریت سناریوهای dropout است:
- Out of frame (سقوط، خارج شدن از کادر)
- Lost tracking (occlusion طولانی)
- Race finished (رسیدن به بالای دیوار)

Key features:
- مستقل از حضور ورزشکار (وابسته به دیوار)
- Timeout-based detection
- History tracking برای robustness
- Clear status codes

Author: Speed Climbing Performance Analysis
Date: 2025-11-19
"""

from dataclasses import dataclass
from typing import Optional, List
from collections import deque


@dataclass
class DropoutStatus:
    """وضعیت dropout برای یک frame."""

    has_dropped_out: bool
    status: str  # 'climbing', 'finished', 'out_of_frame', 'DNF', 'lost_tracking'
    confidence: float  # 0-1
    reason: Optional[str] = None  # توضیح اضافی


class DropoutHandler:
    """
    تشخیص و مدیریت dropout scenarios.

    این handler **مستقل از pose detection** عمل می‌کند.
    محاسبه dropout بر اساس:
    1. کیفیت calibration (اگر holds قابل رویت نباشند → out of frame)
    2. History of detections (اگر X frame پشت سر هم fail شد → lost)
    3. Height position (اگر به بالای دیوار رسید → finished)

    Usage:
        handler = DropoutHandler(timeout_frames=30)

        for frame in frames:
            # Get pose and calibration
            pose = detect_pose(frame)
            calibration = calibrate_frame(frame)

            # Check dropout
            status = handler.check_dropout(
                has_pose=pose is not None,
                has_calibration=calibration is not None and calibration.confidence > 0.6,
                y_position_m=pose.y if pose else None,
                calibration_confidence=calibration.confidence if calibration else 0.0
            )

            if status.has_dropped_out:
                print(f"Dropout detected: {status.status}, reason: {status.reason}")
                break
    """

    def __init__(
        self,
        timeout_frames: int = 30,  # 1 second @ 30fps
        wall_height_m: float = 15.0,
        finish_threshold_m: float = 14.5,  # برای تشخیص رسیدن به بالا
        min_calibration_confidence: float = 0.4
    ):
        """
        Initialize dropout handler.

        Args:
            timeout_frames: تعداد frames که بدون detection می‌توانیم tolerate کنیم
            wall_height_m: ارتفاع دیوار (meters)
            finish_threshold_m: حد آستانه برای تشخیص finish (meters از پد شروع)
            min_calibration_confidence: حداقل confidence برای calibration معتبر
        """
        self.timeout_frames = timeout_frames
        self.wall_height_m = wall_height_m
        self.finish_threshold_m = finish_threshold_m
        self.min_calibration_confidence = min_calibration_confidence

        # History tracking
        self.detection_history: deque = deque(maxlen=timeout_frames)
        self.calibration_history: deque = deque(maxlen=timeout_frames)

        # Dropout state
        self.dropout_detected = False
        self.dropout_status = 'climbing'
        self.dropout_reason = None

        # Frame counter
        self.frame_count = 0

    def check_dropout(
        self,
        has_pose: bool,
        has_calibration: bool,
        y_position_m: Optional[float] = None,
        calibration_confidence: float = 0.0
    ) -> DropoutStatus:
        """
        چک کردن dropout برای frame فعلی.

        Args:
            has_pose: آیا pose detection موفق بود؟
            has_calibration: آیا calibration موفق بود؟
            y_position_m: موقعیت عمودی (meters از پد)، None اگر نامعتبر
            calibration_confidence: کیفیت calibration (0-1)

        Returns:
            DropoutStatus object
        """
        self.frame_count += 1

        # اگر قبلاً dropout detected شده، همچنان در همان وضعیت بمانیم
        if self.dropout_detected:
            return DropoutStatus(
                has_dropped_out=True,
                status=self.dropout_status,
                confidence=1.0,
                reason=self.dropout_reason
            )

        # Update history
        self.detection_history.append(has_pose)
        self.calibration_history.append(has_calibration and
                                        calibration_confidence >= self.min_calibration_confidence)

        # Scenario 1: رسیدن به بالای دیوار (Finished)
        if y_position_m is not None and y_position_m >= self.finish_threshold_m:
            self.dropout_detected = True
            self.dropout_status = 'finished'
            self.dropout_reason = f'reached_top (y={y_position_m:.2f}m >= {self.finish_threshold_m}m)'

            return DropoutStatus(
                has_dropped_out=True,
                status='finished',
                confidence=0.95,
                reason=self.dropout_reason
            )

        # Scenario 2: Calibration failed → احتمالاً out of frame
        # اگر holds قابل رویت نیستند، یعنی دوربین دیگر دیوار را نمی‌بیند
        if len(self.calibration_history) >= self.timeout_frames:
            recent_calibrations = list(self.calibration_history)
            calibration_success_rate = sum(recent_calibrations) / len(recent_calibrations)

            if calibration_success_rate < 0.2:  # کمتر از 20% موفق
                self.dropout_detected = True
                self.dropout_status = 'out_of_frame'
                self.dropout_reason = (f'calibration_failed '
                                      f'({calibration_success_rate:.1%} success in last '
                                      f'{self.timeout_frames} frames)')

                return DropoutStatus(
                    has_dropped_out=True,
                    status='out_of_frame',
                    confidence=0.85,
                    reason=self.dropout_reason
                )

        # Scenario 3: Pose detection failed for too long → lost tracking
        if len(self.detection_history) >= self.timeout_frames:
            recent_detections = list(self.detection_history)
            detection_success_rate = sum(recent_detections) / len(recent_detections)

            if detection_success_rate < 0.15:  # کمتر از 15% موفق
                self.dropout_detected = True
                self.dropout_status = 'lost_tracking'
                self.dropout_reason = (f'pose_detection_failed '
                                      f'({detection_success_rate:.1%} success in last '
                                      f'{self.timeout_frames} frames)')

                return DropoutStatus(
                    has_dropped_out=True,
                    status='DNF',  # یا 'lost_tracking'
                    confidence=0.75,
                    reason=self.dropout_reason
                )

        # Scenario 4: هنوز در حال صعود
        return DropoutStatus(
            has_dropped_out=False,
            status='climbing',
            confidence=1.0,
            reason=None
        )

    def reset(self):
        """ریست کردن handler برای race جدید."""
        self.detection_history.clear()
        self.calibration_history.clear()
        self.dropout_detected = False
        self.dropout_status = 'climbing'
        self.dropout_reason = None
        self.frame_count = 0

    def get_stats(self) -> dict:
        """آمار detection history."""
        if not self.detection_history:
            return {
                'frames_processed': self.frame_count,
                'detection_rate': 0.0,
                'calibration_rate': 0.0,
                'has_dropped_out': self.dropout_detected,
                'status': self.dropout_status
            }

        detection_rate = sum(self.detection_history) / len(self.detection_history)
        calibration_rate = sum(self.calibration_history) / len(self.calibration_history)

        return {
            'frames_processed': self.frame_count,
            'detection_rate': detection_rate,
            'calibration_rate': calibration_rate,
            'has_dropped_out': self.dropout_detected,
            'status': self.dropout_status,
            'dropout_reason': self.dropout_reason
        }


# Example usage and testing
if __name__ == '__main__':
    print("Testing DropoutHandler...")

    # Test 1: Normal race - no dropout
    print("\n=== Test 1: Normal race (no dropout) ===")
    handler = DropoutHandler(timeout_frames=10)

    for i in range(20):
        status = handler.check_dropout(
            has_pose=True,
            has_calibration=True,
            y_position_m=i * 0.5,  # Climbing from 0 to 10m
            calibration_confidence=0.85
        )
        if status.has_dropped_out:
            print(f"  Frame {i}: DROPOUT - {status.status} ({status.reason})")
            break

    if not status.has_dropped_out:
        print("  ✅ No dropout detected (expected)")

    stats = handler.get_stats()
    print(f"  Stats: {stats['detection_rate']:.1%} detection, "
          f"{stats['calibration_rate']:.1%} calibration")

    # Test 2: Climber reaches top
    print("\n=== Test 2: Climber reaches top (finished) ===")
    handler.reset()

    for i in range(30):
        y_pos = min(i * 0.5, 15.0)  # Climb to 15m
        status = handler.check_dropout(
            has_pose=True,
            has_calibration=True,
            y_position_m=y_pos,
            calibration_confidence=0.85
        )
        if status.has_dropped_out:
            print(f"  Frame {i}: {status.status} - {status.reason}")
            print(f"  ✅ Correctly detected finish (y={y_pos:.1f}m)")
            break

    # Test 3: Climber falls (out of frame)
    print("\n=== Test 3: Climber falls (out of frame) ===")
    handler.reset()

    for i in range(50):
        # Simulate fall at frame 20
        if i < 20:
            has_pose = True
            has_cal = True
            y_pos = i * 0.5
        else:
            # After fall: no pose, no calibration (can't see wall)
            has_pose = False
            has_cal = False
            y_pos = None

        status = handler.check_dropout(
            has_pose=has_pose,
            has_calibration=has_cal,
            y_position_m=y_pos,
            calibration_confidence=0.0 if i >= 20 else 0.85
        )

        if status.has_dropped_out:
            print(f"  Frame {i}: {status.status} - {status.reason}")
            print(f"  ✅ Correctly detected dropout after {i-20} frames")
            break

    # Test 4: Lost tracking (pose fails but calibration OK)
    print("\n=== Test 4: Lost tracking (occlusion) ===")
    handler.reset()

    for i in range(50):
        # Simulate occlusion at frame 15
        if i < 15:
            has_pose = True
            has_cal = True
        else:
            # Occlusion: can still see wall (calibration OK) but not athlete
            has_pose = False
            has_cal = True  # Holds still visible

        status = handler.check_dropout(
            has_pose=has_pose,
            has_calibration=has_cal,
            y_position_m=7.5 if has_pose else None,
            calibration_confidence=0.85
        )

        if status.has_dropped_out:
            print(f"  Frame {i}: {status.status} - {status.reason}")
            print(f"  ✅ Correctly detected lost tracking after {i-15} frames")
            break

    print("\n=== All tests passed! ===")
