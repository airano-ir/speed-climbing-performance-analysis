"""
Race Visualization Tools

Create insightful plots for race analysis.

Author: Speed Climbing Analysis Project
Date: 2025-11-15
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RacePlotter:
    """Create visualizations for race analysis."""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize plotter.

        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        self.fig_size = (12, 8)

    def plot_velocity_comparison(
        self,
        left_metrics: Dict,
        right_metrics: Dict,
        race_name: str,
        output_file: str
    ):
        """Plot velocity comparison between two climbers.

        Args:
            left_metrics: Left climber metrics
            right_metrics: Right climber metrics
            race_name: Race identifier
            output_file: Output file path
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.fig_size)

        # Bar chart - Max velocities
        lanes = ['Left', 'Right']
        max_vels = [
            left_metrics['max_vertical_velocity'],
            right_metrics['max_vertical_velocity']
        ]

        ax1.bar(lanes, max_vels, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        ax1.set_ylabel('Max Velocity (px/s)', fontsize=12)
        ax1.set_title('Max Vertical Velocity', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, v in enumerate(max_vels):
            ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=10)

        # Bar chart - Path efficiency
        efficiencies = [
            left_metrics['path_efficiency'],
            right_metrics['path_efficiency']
        ]

        ax2.bar(lanes, efficiencies, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        ax2.set_ylabel('Path Efficiency', fontsize=12)
        ax2.set_title('Path Efficiency (straight/actual)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, v in enumerate(efficiencies):
            ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

        fig.suptitle(f'Race Comparison: {race_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot: {output_path}")

    def plot_competition_summary(
        self,
        metrics_csv: str,
        output_file: str
    ):
        """Plot competition summary statistics.

        Args:
            metrics_csv: Path to aggregate metrics CSV
            output_file: Output file path
        """
        df = pd.read_csv(metrics_csv)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Velocity distribution
        ax1.hist(df['max_vertical_velocity'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.axvline(df['max_vertical_velocity'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.set_xlabel('Max Velocity (px/s)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Max Velocity Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Path efficiency distribution
        ax2.hist(df['path_efficiency'], bins=15, color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.axvline(df['path_efficiency'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.set_xlabel('Path Efficiency', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Path Efficiency Distribution', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Lane comparison
        left_df = df[df['lane'] == 'left']
        right_df = df[df['lane'] == 'right']

        lane_data = [
            left_df['max_vertical_velocity'].values,
            right_df['max_vertical_velocity'].values
        ]
        bp = ax3.boxplot(lane_data, labels=['Left', 'Right'], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax3.set_ylabel('Max Velocity (px/s)', fontsize=11)
        ax3.set_title('Lane Comparison', fontsize=13, fontweight='bold')
        ax3.grid(alpha=0.3)

        # 4. Velocity vs Efficiency scatter
        ax4.scatter(df['max_vertical_velocity'], df['path_efficiency'],
                   c=df['lane'].map({'left': 'blue', 'right': 'orange'}),
                   alpha=0.6, s=100, edgecolors='black')
        ax4.set_xlabel('Max Velocity (px/s)', fontsize=11)
        ax4.set_ylabel('Path Efficiency', fontsize=11)
        ax4.set_title('Velocity vs Efficiency', fontsize=13, fontweight='bold')
        ax4.grid(alpha=0.3)

        # Add legend for scatter
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label='Left'),
            Patch(facecolor='orange', alpha=0.6, label='Right')
        ]
        ax4.legend(handles=legend_elements)

        fig.suptitle(f'Competition Summary ({len(df)} climbers)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot: {output_path}")

    def plot_leaderboard(
        self,
        leaderboard_csv: str,
        output_file: str,
        top_n: int = 10
    ):
        """Plot leaderboard visualization.

        Args:
            leaderboard_csv: Path to leaderboard CSV
            output_file: Output file path
            top_n: Number of top performers to show
        """
        df = pd.read_csv(leaderboard_csv).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create race labels
        labels = [f"{row['race_name'].split('_')[-1]} ({row['lane'][0].upper()})"
                 for _, row in df.iterrows()]

        # Horizontal bar chart
        y_pos = np.arange(len(labels))
        velocities = df['max_vertical_velocity'].values

        colors = ['#ff7f0e' if lane == 'right' else '#1f77b4'
                 for lane in df['lane']]

        bars = ax.barh(y_pos, velocities, color=colors, alpha=0.7, edgecolor='black')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()  # Top performer at top
        ax.set_xlabel('Max Velocity (px/s)', fontsize=12)
        ax.set_title(f'Top {top_n} Performers', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, vel) in enumerate(zip(bars, velocities)):
            ax.text(vel, i, f' {vel:.1f}', va='center', fontsize=9)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.7, label='Left Lane'),
            Patch(facecolor='#ff7f0e', alpha=0.7, label='Right Lane')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot: {output_path}")


def main():
    """Example usage."""
    plotter = RacePlotter()

    # Example 1: Velocity comparison
    logger.info("Creating velocity comparison plot...")

    # Load metrics
    with open("data/processed/metrics/samples/Speed_finals_Chamonix_2024_race024_metrics_left.json") as f:
        left_metrics = json.load(f)['summary']

    with open("data/processed/metrics/samples/Speed_finals_Chamonix_2024_race024_metrics_right.json") as f:
        right_metrics = json.load(f)['summary']

    plotter.plot_velocity_comparison(
        left_metrics,
        right_metrics,
        "Chamonix 2024 - Race 024",
        "data/processed/plots/velocity_comparison_race024.png"
    )

    # Example 2: Competition summary
    logger.info("Creating competition summary plot...")
    plotter.plot_competition_summary(
        "data/processed/metrics/aggregate_metrics.csv",
        "data/processed/plots/competition_summary.png"
    )

    # Example 3: Leaderboard
    logger.info("Creating leaderboard plot...")
    plotter.plot_leaderboard(
        "data/processed/aggregates/leaderboard_top20.csv",
        "data/processed/plots/leaderboard_top10.png",
        top_n=6
    )

    logger.info("\n✓ All plots created successfully!")


if __name__ == "__main__":
    main()
