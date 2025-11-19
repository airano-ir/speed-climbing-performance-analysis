#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Builder for Global Map Registration
================================================

این ماژول برای ساخت time-series data از frame-by-frame processing است.

Features:
- ساخت time-series برای position, velocity, status
- پشتیبانی از missing/invalid frames
- محاسبه summary statistics
- JSON serialization

Author: Speed Climbing Performance Analysis
Date: 2025-11-19
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import json


@dataclass
class TimeSeriesData:
    """
    Container برای time-series data یک climber.
    """
    lane: str  # 'left' or 'right'
    race_outcome: str  # 'finished', 'DNF', 'unknown'

    # Time series arrays
    timestamps: List[float] = field(default_factory=list)
    y_position_m: List[Optional[float]] = field(default_factory=list)
    x_position_m: List[Optional[float]] = field(default_factory=list)
    status: List[str] = field(default_factory=list)
    calibration_quality: List[Optional[float]] = field(default_factory=list)

    # Dropout info (if applicable)
    dropout_frame: Optional[int] = None
    dropout_time_s: Optional[float] = None
    dropout_reason: Optional[str] = None

    # Summary statistics (computed at build time)
    summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """تبدیل به dictionary برای JSON serialization."""
        return {
            'lane': self.lane,
            'race_outcome': self.race_outcome,
            'time_series': {
                'timestamps': self.timestamps,
                'y_position_m': self.y_position_m,
                'x_position_m': self.x_position_m,
                'status': self.status,
                'calibration_quality': self.calibration_quality
            },
            'dropout_info': {
                'dropout_frame': self.dropout_frame,
                'dropout_time_s': self.dropout_time_s,
                'dropout_reason': self.dropout_reason
            } if self.dropout_frame is not None else None,
            'summary': self.summary
        }


class TimeSeriesBuilder:
    """
    Builder برای ساخت time-series data.

    Usage:
        builder = TimeSeriesBuilder(lane='left')
        for frame in frames:
            builder.add_frame(timestamp, y_pos_m, status, calibration_quality)
        data = builder.build()
    """

    def __init__(self, lane: str):
        """
        Initialize builder.

        Args:
            lane: 'left' or 'right'
        """
        self.lane = lane
        self.timestamps = []
        self.y_positions = []
        self.x_positions = []
        self.statuses = []
        self.calibration_qualities = []

        # Dropout tracking
        self.dropout_frame = None
        self.dropout_time = None
        self.dropout_reason = None

        self.is_built = False

    def add_frame(
        self,
        timestamp: float,
        y_position_m: Optional[float] = None,
        x_position_m: Optional[float] = None,
        status: str = 'climbing',
        calibration_quality: Optional[float] = None
    ):
        """
        اضافه کردن یک frame به time-series.

        Args:
            timestamp: زمان frame (seconds)
            y_position_m: ارتفاع از پد شروع (meters)، None اگر نامعتبر
            x_position_m: موقعیت افقی (meters)، None اگر نامعتبر
            status: 'climbing', 'finished', 'DNF', 'out_of_frame', 'invalid'
            calibration_quality: کیفیت calibration (0-1)، None اگر نامعتبر
        """
        if self.is_built:
            raise RuntimeError("Cannot add frames after build() has been called")

        self.timestamps.append(timestamp)
        self.y_positions.append(y_position_m)
        self.x_positions.append(x_position_m)
        self.statuses.append(status)
        self.calibration_qualities.append(calibration_quality)

        # Track dropout
        if status in ['DNF', 'out_of_frame'] and self.dropout_frame is None:
            self.dropout_frame = len(self.timestamps) - 1
            self.dropout_time = timestamp
            self.dropout_reason = status

    def is_finished(self) -> bool:
        """چک کردن اینکه آیا climber تمام کرده یا dropout شده."""
        if not self.statuses:
            return False
        last_status = self.statuses[-1]
        return last_status in ['finished', 'DNF', 'out_of_frame']

    def build(self) -> TimeSeriesData:
        """
        ساخت TimeSeriesData با محاسبه summary statistics.

        Returns:
            TimeSeriesData object
        """
        if self.is_built:
            raise RuntimeError("build() has already been called")

        self.is_built = True

        # تعیین race outcome
        if not self.statuses:
            race_outcome = 'unknown'
        elif 'finished' in self.statuses:
            race_outcome = 'finished'
        elif 'DNF' in self.statuses or 'out_of_frame' in self.statuses:
            race_outcome = 'DNF'
        else:
            race_outcome = 'unknown'

        # محاسبه summary statistics
        summary = self._calculate_summary()

        return TimeSeriesData(
            lane=self.lane,
            race_outcome=race_outcome,
            timestamps=self.timestamps,
            y_position_m=self.y_positions,
            x_position_m=self.x_positions,
            status=self.statuses,
            calibration_quality=self.calibration_qualities,
            dropout_frame=self.dropout_frame,
            dropout_time_s=self.dropout_time,
            dropout_reason=self.dropout_reason,
            summary=summary
        )

    def _calculate_summary(self) -> Dict:
        """محاسبه summary statistics."""
        # فیلتر کردن valid positions
        valid_y = [y for y in self.y_positions if y is not None]
        valid_x = [x for x in self.x_positions if x is not None]
        valid_cal = [c for c in self.calibration_qualities if c is not None]

        summary = {}

        # Time statistics
        if self.timestamps:
            summary['total_time_s'] = self.timestamps[-1] - self.timestamps[0]
            summary['total_frames'] = len(self.timestamps)
            summary['valid_frames'] = len(valid_y)
            summary['completeness'] = len(valid_y) / len(self.timestamps) if self.timestamps else 0.0

        # Position statistics
        if valid_y:
            summary['final_height_m'] = valid_y[-1]
            summary['max_height_m'] = max(valid_y)
            summary['min_height_m'] = min(valid_y)
            summary['total_distance_m'] = max(valid_y) - min(valid_y)

            # Velocity estimation (simple)
            if len(valid_y) > 1 and self.timestamps:
                valid_timestamps = [self.timestamps[i] for i in range(len(self.y_positions))
                                   if self.y_positions[i] is not None]
                if len(valid_timestamps) > 1:
                    time_range = valid_timestamps[-1] - valid_timestamps[0]
                    if time_range > 0:
                        distance = valid_y[-1] - valid_y[0]
                        summary['avg_velocity_m_s'] = distance / time_range
                    else:
                        summary['avg_velocity_m_s'] = 0.0
                else:
                    summary['avg_velocity_m_s'] = 0.0
            else:
                summary['avg_velocity_m_s'] = 0.0

            # Maximum velocity (simple estimate from consecutive frames)
            if len(valid_y) > 2:
                velocities = []
                for i in range(len(self.y_positions) - 1):
                    if (self.y_positions[i] is not None and
                        self.y_positions[i+1] is not None):
                        dt = self.timestamps[i+1] - self.timestamps[i]
                        if dt > 0:
                            dy = self.y_positions[i+1] - self.y_positions[i]
                            velocities.append(abs(dy / dt))

                if velocities:
                    summary['max_velocity_m_s'] = max(velocities)
                else:
                    summary['max_velocity_m_s'] = 0.0
            else:
                summary['max_velocity_m_s'] = 0.0
        else:
            # No valid positions
            summary['final_height_m'] = 0.0
            summary['max_height_m'] = 0.0
            summary['min_height_m'] = 0.0
            summary['total_distance_m'] = 0.0
            summary['avg_velocity_m_s'] = 0.0
            summary['max_velocity_m_s'] = 0.0

        # Calibration statistics
        if valid_cal:
            summary['avg_calibration_quality'] = np.mean(valid_cal)
            summary['min_calibration_quality'] = min(valid_cal)
        else:
            summary['avg_calibration_quality'] = 0.0
            summary['min_calibration_quality'] = 0.0

        return summary


def save_time_series(
    left_data: Optional[TimeSeriesData],
    right_data: Optional[TimeSeriesData],
    output_path: str,
    metadata: Optional[Dict] = None
):
    """
    ذخیره time-series data برای هر دو lane در یک JSON file.

    Args:
        left_data: Time-series data برای left climber (optional)
        right_data: Time-series data برای right climber (optional)
        output_path: مسیر فایل خروجی JSON
        metadata: اطلاعات اضافی (video path, competition, etc.)
    """
    output = {
        'metadata': metadata or {},
        'left_climber': left_data.to_dict() if left_data else None,
        'right_climber': right_data.to_dict() if right_data else None
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def load_time_series(input_path: str) -> Dict:
    """
    بارگذاری time-series data از JSON file.

    Args:
        input_path: مسیر فایل JSON

    Returns:
        Dictionary حاوی metadata و data برای هر دو lane
    """
    with open(input_path, 'r') as f:
        return json.load(f)


# Example usage
if __name__ == '__main__':
    # Test TimeSeriesBuilder
    print("Testing TimeSeriesBuilder...")

    builder = TimeSeriesBuilder(lane='left')

    # Simulate a race (6 seconds, climbing from 0 to 15m)
    for i in range(180):  # 180 frames @ 30fps = 6 seconds
        timestamp = i / 30.0  # seconds
        y_position = (i / 180.0) * 15.0  # linear climb to 15m
        x_position = 1.5  # centered in lane
        status = 'finished' if i == 179 else 'climbing'
        calibration_quality = 0.85

        builder.add_frame(
            timestamp=timestamp,
            y_position_m=y_position,
            x_position_m=x_position,
            status=status,
            calibration_quality=calibration_quality
        )

    # Build and display
    data = builder.build()

    print(f"\nRace outcome: {data.race_outcome}")
    print(f"Total time: {data.summary['total_time_s']:.2f}s")
    print(f"Total distance: {data.summary['total_distance_m']:.2f}m")
    print(f"Avg velocity: {data.summary['avg_velocity_m_s']:.2f} m/s")
    print(f"Max velocity: {data.summary['max_velocity_m_s']:.2f} m/s")
    print(f"Calibration quality: {data.summary['avg_calibration_quality']:.2f}")

    # Save to JSON
    save_time_series(
        left_data=data,
        right_data=None,
        output_path='test_time_series.json',
        metadata={'test': True, 'description': 'Simulated race'}
    )

    print(f"\n✅ Test passed! JSON saved to test_time_series.json")
