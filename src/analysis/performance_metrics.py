#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Metrics Calculator
==============================
محاسبه متریک‌های performance از pose data

Metrics:
- Center of Mass (COM) trajectory
- Vertical velocity (m/s or pixels/s)
- Acceleration
- Movement smoothness (jerk analysis)
- Path efficiency

Author: Speed Climbing Performance Analysis Project
Date: 2025-11-14
"""

from pathlib import Path
import sys
import io
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Fix Windows console encoding for emoji/unicode support
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    # Time series data
    timestamps: np.ndarray
    com_x: np.ndarray  # Center of mass X (pixels)
    com_y: np.ndarray  # Center of mass Y (pixels - 0 at top)

    # Velocities (pixels/second)
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    velocity_magnitude: np.ndarray

    # Accelerations (pixels/second²)
    acceleration_x: np.ndarray
    acceleration_y: np.ndarray
    acceleration_magnitude: np.ndarray

    # Jerk (pixels/second³) - smoothness measure
    jerk_x: np.ndarray
    jerk_y: np.ndarray
    jerk_magnitude: np.ndarray

    # Summary statistics
    avg_vertical_velocity: float  # Average upward velocity
    max_vertical_velocity: float
    avg_acceleration: float
    max_acceleration: float
    path_length: float  # Total distance traveled
    straight_distance: float  # Direct distance start to end
    path_efficiency: float  # straight_distance / path_length
    smoothness_score: float  # Based on jerk (lower = smoother)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'summary': {
                'avg_vertical_velocity': float(self.avg_vertical_velocity),
                'max_vertical_velocity': float(self.max_vertical_velocity),
                'avg_acceleration': float(self.avg_acceleration),
                'max_acceleration': float(self.max_acceleration),
                'path_length': float(self.path_length),
                'straight_distance': float(self.straight_distance),
                'path_efficiency': float(self.path_efficiency),
                'smoothness_score': float(self.smoothness_score),
            },
            'time_series': {
                'timestamps': self.timestamps.tolist(),
                'com_x': self.com_x.tolist(),
                'com_y': self.com_y.tolist(),
                'velocity_x': self.velocity_x.tolist(),
                'velocity_y': self.velocity_y.tolist(),
                'velocity_magnitude': self.velocity_magnitude.tolist(),
                'acceleration_x': self.acceleration_x.tolist(),
                'acceleration_y': self.acceleration_y.tolist(),
                'acceleration_magnitude': self.acceleration_magnitude.tolist(),
            }
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert time series data to pandas DataFrame"""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'com_x': self.com_x,
            'com_y': self.com_y,
            'velocity_x': self.velocity_x,
            'velocity_y': self.velocity_y,
            'velocity_magnitude': self.velocity_magnitude,
            'acceleration_x': self.acceleration_x,
            'acceleration_y': self.acceleration_y,
            'acceleration_magnitude': self.acceleration_magnitude,
            'jerk_x': self.jerk_x,
            'jerk_y': self.jerk_y,
            'jerk_magnitude': self.jerk_magnitude,
        })


class PerformanceAnalyzer:
    """
    Analyzer برای محاسبه performance metrics از pose data

    Usage:
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.analyze_pose_file('race001_poses.json', lane='left')
        df = metrics.to_dataframe()
        df.to_csv('race001_metrics.csv', index=False)
    """

    # BlazePose keypoint indices
    # https://google.github.io/mediapipe/solutions/pose.html
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def __init__(self, smoothing_window: int = 5):
        """
        Initialize analyzer.

        Args:
            smoothing_window: Window size for moving average smoothing
        """
        self.smoothing_window = smoothing_window

    def calculate_com(self, keypoints_dict: Dict) -> Tuple[float, float]:
        """
        Calculate Center of Mass from pose keypoints.

        Uses weighted average of major body parts:
        - Head (nose): 8%
        - Torso (shoulders + hips): 50%
        - Legs (knees + ankles): 42%

        Args:
            keypoints_dict: Dictionary of keypoint_name -> keypoint_data

        Returns:
            (com_x, com_y) in normalized coords [0-1]
        """
        # Map keypoint names to weights
        keypoint_weights = {
            'nose': 0.08,
            'left_shoulder': 0.125,
            'right_shoulder': 0.125,
            'left_hip': 0.125,
            'right_hip': 0.125,
            'left_knee': 0.105,
            'right_knee': 0.105,
            'left_ankle': 0.105,
            'right_ankle': 0.105,
        }

        com_x = 0.0
        com_y = 0.0
        total_weight = 0.0

        for kp_name, weight in keypoint_weights.items():
            if kp_name in keypoints_dict:
                kp = keypoints_dict[kp_name]
                visibility = kp.get('visibility', 0)
                if visibility > 0.5:  # visibility threshold
                    com_x += kp['x'] * weight
                    com_y += kp['y'] * weight
                    total_weight += weight

        if total_weight > 0:
            com_x /= total_weight
            com_y /= total_weight

        return com_x, com_y

    def smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing"""
        if len(signal) < self.smoothing_window:
            return signal

        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        smoothed = np.convolve(signal, kernel, mode='same')

        # Fix edges
        half_window = self.smoothing_window // 2
        smoothed[:half_window] = signal[:half_window]
        smoothed[-half_window:] = signal[-half_window:]

        return smoothed

    def calculate_derivative(self, signal: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """Calculate numerical derivative"""
        if len(signal) < 2:
            return np.zeros_like(signal)

        derivative = np.zeros_like(signal)

        # Forward difference for first point
        derivative[0] = (signal[1] - signal[0]) / dt[0] if dt[0] > 0 else 0

        # Central difference for middle points
        for i in range(1, len(signal) - 1):
            dt_avg = (dt[i-1] + dt[i]) / 2
            if dt_avg > 0:
                derivative[i] = (signal[i+1] - signal[i-1]) / (2 * dt_avg)

        # Backward difference for last point
        derivative[-1] = (signal[-1] - signal[-2]) / dt[-1] if dt[-1] > 0 else 0

        return derivative

    def analyze_pose_file(
        self,
        pose_json_path: Path,
        lane: str = 'left',
        min_visibility: float = 0.5
    ) -> Optional[PerformanceMetrics]:
        """
        Analyze pose file and calculate performance metrics.

        Args:
            pose_json_path: Path to pose JSON file
            lane: Which climber to analyze ('left' or 'right')
            min_visibility: Minimum visibility threshold for valid poses

        Returns:
            PerformanceMetrics object or None if insufficient data
        """
        # Load pose data
        with open(pose_json_path, 'r') as f:
            data = json.load(f)

        frames = data.get('frames', [])
        if not frames:
            print(f"⚠️  No frames in {pose_json_path}")
            return None

        # Extract COM trajectory
        timestamps = []
        com_x_list = []
        com_y_list = []

        climber_key = f'{lane}_climber'

        for frame in frames:
            climber_data = frame.get(climber_key)
            if not climber_data or not climber_data.get('keypoints'):
                continue

            timestamp = frame.get('timestamp', 0)
            keypoints_dict = climber_data['keypoints']

            # Check if enough keypoints are visible
            visible_count = sum(
                1 for kp in keypoints_dict.values()
                if kp.get('visibility', 0) > min_visibility
            )
            if visible_count < 5:  # Need at least 5 visible keypoints
                continue

            com_x, com_y = self.calculate_com(keypoints_dict)

            timestamps.append(timestamp)
            com_x_list.append(com_x)
            com_y_list.append(com_y)

        if len(timestamps) < 10:
            print(f"⚠️  Insufficient valid frames ({len(timestamps)}) in {pose_json_path}")
            return None

        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        com_x = np.array(com_x_list)
        com_y = np.array(com_y_list)

        # Smooth COM trajectory
        com_x_smooth = self.smooth_signal(com_x)
        com_y_smooth = self.smooth_signal(com_y)

        # Calculate time deltas
        dt = np.diff(timestamps)
        dt = np.append(dt, dt[-1])  # Repeat last dt

        # Calculate velocities
        velocity_x = self.calculate_derivative(com_x_smooth, dt)
        velocity_y = self.calculate_derivative(com_y_smooth, dt)
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

        # Calculate accelerations
        acceleration_x = self.calculate_derivative(velocity_x, dt)
        acceleration_y = self.calculate_derivative(velocity_y, dt)
        acceleration_magnitude = np.sqrt(acceleration_x**2 + acceleration_y**2)

        # Calculate jerk (smoothness)
        jerk_x = self.calculate_derivative(acceleration_x, dt)
        jerk_y = self.calculate_derivative(acceleration_y, dt)
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2)

        # Summary statistics
        # Note: Y increases downward in image coordinates, so upward = negative velocity_y
        vertical_velocities = -velocity_y  # Make upward positive
        avg_vertical_velocity = np.mean(vertical_velocities)
        max_vertical_velocity = np.max(vertical_velocities)

        avg_acceleration = np.mean(acceleration_magnitude)
        max_acceleration = np.max(acceleration_magnitude)

        # Path metrics
        path_segments = np.sqrt(np.diff(com_x_smooth)**2 + np.diff(com_y_smooth)**2)
        path_length = np.sum(path_segments)
        straight_distance = np.sqrt(
            (com_x_smooth[-1] - com_x_smooth[0])**2 +
            (com_y_smooth[-1] - com_y_smooth[0])**2
        )
        path_efficiency = straight_distance / path_length if path_length > 0 else 0

        # Smoothness score (normalized average jerk)
        smoothness_score = np.mean(jerk_magnitude)

        return PerformanceMetrics(
            timestamps=timestamps,
            com_x=com_x_smooth,
            com_y=com_y_smooth,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            velocity_magnitude=velocity_magnitude,
            acceleration_x=acceleration_x,
            acceleration_y=acceleration_y,
            acceleration_magnitude=acceleration_magnitude,
            jerk_x=jerk_x,
            jerk_y=jerk_y,
            jerk_magnitude=jerk_magnitude,
            avg_vertical_velocity=avg_vertical_velocity,
            max_vertical_velocity=max_vertical_velocity,
            avg_acceleration=avg_acceleration,
            max_acceleration=max_acceleration,
            path_length=path_length,
            straight_distance=straight_distance,
            path_efficiency=path_efficiency,
            smoothness_score=smoothness_score,
        )


def main():
    """CLI interface for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Calculate performance metrics from pose data')
    parser.add_argument('pose_file', type=str, help='Path to pose JSON file')
    parser.add_argument('--lane', type=str, default='left', choices=['left', 'right'],
                       help='Which climber to analyze')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: same as input with _metrics.csv)')
    args = parser.parse_args()

    pose_file = Path(args.pose_file)
    if not pose_file.exists():
        print(f"❌ File not found: {pose_file}")
        return

    # Analyze
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze_pose_file(pose_file, lane=args.lane)

    if metrics is None:
        print("❌ Failed to analyze pose file")
        return

    # Print summary
    print("\n" + "="*70)
    print(f"Performance Metrics - {pose_file.name} ({args.lane} climber)")
    print("="*70)
    print(f"Average vertical velocity: {metrics.avg_vertical_velocity:.2f} px/s")
    print(f"Max vertical velocity:     {metrics.max_vertical_velocity:.2f} px/s")
    print(f"Average acceleration:      {metrics.avg_acceleration:.2f} px/s²")
    print(f"Max acceleration:          {metrics.max_acceleration:.2f} px/s²")
    print(f"Path length:               {metrics.path_length:.2f} px")
    print(f"Straight distance:         {metrics.straight_distance:.2f} px")
    print(f"Path efficiency:           {metrics.path_efficiency:.2%}")
    print(f"Smoothness score (jerk):   {metrics.smoothness_score:.2f} px/s³")
    print("="*70)

    # Save to CSV
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = pose_file.parent / f"{pose_file.stem}_metrics_{args.lane}.csv"

    df = metrics.to_dataframe()
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved metrics to: {output_path}")

    # Save summary to JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"✓ Saved summary to: {json_path}")


if __name__ == '__main__':
    main()
