#!/usr/bin/env python3
"""
Time Series Visualization
=========================
تولید نمودارهای time-series برای تحلیل performance

Plots:
- Height vs Time (trajectory)
- Velocity vs Time
- Acceleration vs Time
- Dual climber comparison

Author: Speed Climbing Performance Analysis Project
Date: 2025-11-14
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List, Tuple
import sys

# Add analysis module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))
from performance_metrics import PerformanceAnalyzer, PerformanceMetrics


class TimeSeriesVisualizer:
    """
    Visualizer برای نمودارهای time-series

    Usage:
        viz = TimeSeriesVisualizer()
        viz.plot_single_climber('race001_poses.json', 'left')
        viz.plot_dual_comparison('race001_poses.json')
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style
        """
        # Set style with fallback
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        self.analyzer = PerformanceAnalyzer()

        # Colors
        self.colors = {
            'left': '#2E86AB',   # Blue
            'right': '#A23B72',  # Purple
            'velocity': '#F18F01',  # Orange
            'acceleration': '#C73E1D',  # Red
        }

    def plot_trajectory(
        self,
        metrics: PerformanceMetrics,
        ax: plt.Axes,
        lane: str = 'left',
        title: Optional[str] = None
    ):
        """Plot COM trajectory (height vs time)"""
        # Note: Y increases downward, so invert for plotting
        height = -metrics.com_y + metrics.com_y[0]  # Relative height (upward positive)

        ax.plot(
            metrics.timestamps,
            height,
            color=self.colors.get(lane, '#333333'),
            linewidth=2,
            label=f'{lane.capitalize()} climber'
        )

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        ax.set_title(title or 'Climbing Trajectory', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_velocity(
        self,
        metrics: PerformanceMetrics,
        ax: plt.Axes,
        lane: str = 'left',
        title: Optional[str] = None
    ):
        """Plot vertical velocity vs time"""
        # Upward = negative velocity_y in image coords
        vertical_velocity = -metrics.velocity_y

        ax.plot(
            metrics.timestamps,
            vertical_velocity,
            color=self.colors['velocity'],
            linewidth=2,
            label='Vertical velocity'
        )

        # Add average line
        avg_vel = np.mean(vertical_velocity)
        ax.axhline(
            avg_vel,
            color='green',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label=f'Average: {avg_vel:.1f} px/s'
        )

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Velocity (pixels/s)', fontsize=12)
        ax.set_title(title or 'Vertical Velocity', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_acceleration(
        self,
        metrics: PerformanceMetrics,
        ax: plt.Axes,
        title: Optional[str] = None
    ):
        """Plot acceleration magnitude vs time"""
        ax.plot(
            metrics.timestamps,
            metrics.acceleration_magnitude,
            color=self.colors['acceleration'],
            linewidth=2,
            label='Acceleration magnitude'
        )

        # Add average line
        avg_acc = np.mean(metrics.acceleration_magnitude)
        ax.axhline(
            avg_acc,
            color='purple',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label=f'Average: {avg_acc:.1f} px/s²'
        )

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Acceleration (pixels/s²)', fontsize=12)
        ax.set_title(title or 'Acceleration Magnitude', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_horizontal_deviation(
        self,
        metrics: PerformanceMetrics,
        ax: plt.Axes,
        lane: str = 'left',
        title: Optional[str] = None
    ):
        """Plot horizontal deviation from starting position"""
        horizontal_deviation = metrics.com_x - metrics.com_x[0]

        ax.plot(
            metrics.timestamps,
            horizontal_deviation,
            color=self.colors.get(lane, '#333333'),
            linewidth=2,
            label='Horizontal deviation'
        )

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Horizontal deviation (pixels)', fontsize=12)
        ax.set_title(title or 'Horizontal Movement', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_single_climber(
        self,
        pose_json_path: Path,
        lane: str = 'left',
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Create comprehensive visualization for single climber.

        Args:
            pose_json_path: Path to pose JSON file
            lane: Which climber ('left' or 'right')
            save_path: Path to save figure (optional)
            show: Whether to display figure
        """
        # Analyze pose data
        metrics = self.analyzer.analyze_pose_file(pose_json_path, lane=lane)

        if metrics is None:
            print(f"❌ Failed to analyze {pose_json_path}")
            return

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # Plot each metric
        self.plot_trajectory(metrics, ax1, lane=lane)
        self.plot_velocity(metrics, ax2, lane=lane)
        self.plot_acceleration(metrics, ax3)
        self.plot_horizontal_deviation(metrics, ax4, lane=lane)

        # Overall title
        fig.suptitle(
            f'Performance Analysis - {pose_json_path.stem} ({lane.capitalize()} Climber)',
            fontsize=16,
            fontweight='bold'
        )

        # Add summary text
        summary_text = (
            f"Avg Velocity: {metrics.avg_vertical_velocity:.1f} px/s  |  "
            f"Max Velocity: {metrics.max_vertical_velocity:.1f} px/s\n"
            f"Path Efficiency: {metrics.path_efficiency:.1%}  |  "
            f"Smoothness: {metrics.smoothness_score:.1f}"
        )
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot to: {save_path}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()

    def plot_dual_comparison(
        self,
        pose_json_path: Path,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Create side-by-side comparison of both climbers.

        Args:
            pose_json_path: Path to pose JSON file
            save_path: Path to save figure
            show: Whether to display figure
        """
        # Analyze both climbers
        metrics_left = self.analyzer.analyze_pose_file(pose_json_path, lane='left')
        metrics_right = self.analyzer.analyze_pose_file(pose_json_path, lane='right')

        if metrics_left is None or metrics_right is None:
            print(f"❌ Failed to analyze both climbers in {pose_json_path}")
            return

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Trajectories comparison
        ax1 = axes[0]
        height_left = -metrics_left.com_y + metrics_left.com_y[0]
        height_right = -metrics_right.com_y + metrics_right.com_y[0]

        ax1.plot(metrics_left.timestamps, height_left,
                color=self.colors['left'], linewidth=2, label='Left climber')
        ax1.plot(metrics_right.timestamps, height_right,
                color=self.colors['right'], linewidth=2, label='Right climber')

        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Height (pixels)', fontsize=12)
        ax1.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Velocity comparison
        ax2 = axes[1]
        vel_left = -metrics_left.velocity_y
        vel_right = -metrics_right.velocity_y

        ax2.plot(metrics_left.timestamps, vel_left,
                color=self.colors['left'], linewidth=2, label='Left climber')
        ax2.plot(metrics_right.timestamps, vel_right,
                color=self.colors['right'], linewidth=2, label='Right climber')

        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Velocity (pixels/s)', fontsize=12)
        ax2.set_title('Velocity Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Summary comparison (bar chart)
        ax3 = axes[2]
        metrics_names = ['Avg Velocity', 'Max Velocity', 'Path Efficiency']
        left_values = [
            metrics_left.avg_vertical_velocity,
            metrics_left.max_vertical_velocity,
            metrics_left.path_efficiency * 100  # as percentage
        ]
        right_values = [
            metrics_right.avg_vertical_velocity,
            metrics_right.max_vertical_velocity,
            metrics_right.path_efficiency * 100
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        ax3.bar(x - width/2, left_values, width, label='Left', color=self.colors['left'])
        ax3.bar(x + width/2, right_values, width, label='Right', color=self.colors['right'])

        ax3.set_ylabel('Value', fontsize=12)
        ax3.set_title('Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics_names, rotation=15, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Overall title
        fig.suptitle(
            f'Dual Climber Comparison - {pose_json_path.stem}',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved comparison plot to: {save_path}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()


def main():
    """CLI interface for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Create time-series visualizations')
    parser.add_argument('pose_file', type=str, help='Path to pose JSON file')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'dual'],
                       help='Visualization mode (single or dual climber)')
    parser.add_argument('--lane', type=str, default='left', choices=['left', 'right'],
                       help='Which climber for single mode')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image file path (default: auto-generated)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot (only save)')
    args = parser.parse_args()

    pose_file = Path(args.pose_file)
    if not pose_file.exists():
        print(f"❌ File not found: {pose_file}")
        return

    # Auto-generate output path if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        if args.mode == 'single':
            output_path = pose_file.parent / f"{pose_file.stem}_plot_{args.lane}.png"
        else:
            output_path = pose_file.parent / f"{pose_file.stem}_comparison.png"

    # Create visualization
    viz = TimeSeriesVisualizer()

    if args.mode == 'single':
        viz.plot_single_climber(
            pose_file,
            lane=args.lane,
            save_path=output_path,
            show=not args.no_show
        )
    else:
        viz.plot_dual_comparison(
            pose_file,
            save_path=output_path,
            show=not args.no_show
        )


if __name__ == '__main__':
    main()
