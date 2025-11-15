"""
Competition Statistics Aggregator

Analyze and aggregate performance statistics by competition, athlete, and round.

Usage:
    python scripts/aggregate_competition_stats.py
    python scripts/aggregate_competition_stats.py --metrics-csv data/processed/metrics/aggregate_metrics.csv

Author: Speed Climbing Analysis Project
Date: 2025-11-15
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompetitionAggregator:
    """Aggregate statistics by competition."""

    def __init__(self, metrics_csv: str):
        """Initialize aggregator.

        Args:
            metrics_csv: Path to aggregate metrics CSV
        """
        self.metrics_csv = Path(metrics_csv)

        if not self.metrics_csv.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {self.metrics_csv}")

        self.df = pd.read_csv(self.metrics_csv)
        logger.info(f"Loaded {len(self.df)} race metrics from {self.metrics_csv}")
        logger.info(f"  Competitions: {self.df['competition'].nunique()}")
        logger.info(f"  Races: {self.df['race_name'].nunique()}")

    def get_competition_summary(self) -> Dict:
        """Get summary statistics by competition.

        Returns:
            Dictionary with competition summaries
        """
        logger.info("\nGenerating competition summaries...")

        summaries = {}

        for competition in self.df['competition'].unique():
            comp_df = self.df[self.df['competition'] == competition]

            summary = {
                'competition': competition,
                'total_races': comp_df['race_name'].nunique(),
                'total_climbers': len(comp_df),
                'calibrated_count': int(comp_df['is_calibrated'].sum()),
                'velocity': {
                    'avg_vertical_velocity_mean': float(comp_df['avg_vertical_velocity'].mean()),
                    'avg_vertical_velocity_std': float(comp_df['avg_vertical_velocity'].std()),
                    'max_vertical_velocity_mean': float(comp_df['max_vertical_velocity'].mean()),
                    'max_vertical_velocity_max': float(comp_df['max_vertical_velocity'].max()),
                },
                'acceleration': {
                    'avg_acceleration_mean': float(comp_df['avg_acceleration'].mean()),
                    'avg_acceleration_std': float(comp_df['avg_acceleration'].std()),
                    'max_acceleration_mean': float(comp_df['max_acceleration'].mean()),
                    'max_acceleration_max': float(comp_df['max_acceleration'].max()),
                },
                'path': {
                    'path_length_mean': float(comp_df['path_length'].mean()),
                    'path_efficiency_mean': float(comp_df['path_efficiency'].mean()),
                    'path_efficiency_std': float(comp_df['path_efficiency'].std()),
                },
                'smoothness': {
                    'smoothness_score_mean': float(comp_df['smoothness_score'].mean()),
                    'smoothness_score_std': float(comp_df['smoothness_score'].std()),
                }
            }

            summaries[competition] = summary

            logger.info(f"\n  {competition}:")
            logger.info(f"    Races: {summary['total_races']}")
            logger.info(f"    Avg velocity: {summary['velocity']['avg_vertical_velocity_mean']:.1f} ± {summary['velocity']['avg_vertical_velocity_std']:.1f}")
            logger.info(f"    Path efficiency: {summary['path']['path_efficiency_mean']:.3f} ± {summary['path']['path_efficiency_std']:.3f}")

        return summaries

    def get_leaderboard(self, top_n: int = 10) -> pd.DataFrame:
        """Get top performers leaderboard.

        Args:
            top_n: Number of top performers

        Returns:
            DataFrame with top performers
        """
        logger.info(f"\nGenerating top {top_n} leaderboard...")

        # Sort by max vertical velocity (proxy for speed)
        leaderboard = self.df.nlargest(top_n, 'max_vertical_velocity')

        logger.info("\nTop performers:")
        for i, row in enumerate(leaderboard.itertuples(), 1):
            logger.info(f"  {i}. {row.race_name} ({row.lane}): {row.max_vertical_velocity:.1f} px/s")

        return leaderboard[['race_name', 'competition', 'lane', 'max_vertical_velocity',
                            'avg_vertical_velocity', 'path_efficiency']]

    def get_lane_comparison(self) -> Dict:
        """Compare left vs right lane statistics.

        Returns:
            Dictionary with lane comparison
        """
        logger.info("\nAnalyzing lane differences...")

        left_df = self.df[self.df['lane'] == 'left']
        right_df = self.df[self.df['lane'] == 'right']

        comparison = {
            'left': {
                'count': len(left_df),
                'avg_velocity': float(left_df['avg_vertical_velocity'].mean()),
                'max_velocity': float(left_df['max_vertical_velocity'].mean()),
                'path_efficiency': float(left_df['path_efficiency'].mean()),
            },
            'right': {
                'count': len(right_df),
                'avg_velocity': float(right_df['avg_vertical_velocity'].mean()),
                'max_velocity': float(right_df['max_vertical_velocity'].mean()),
                'path_efficiency': float(right_df['path_efficiency'].mean()),
            }
        }

        logger.info(f"\n  Left lane: {comparison['left']['count']} climbers")
        logger.info(f"    Avg velocity: {comparison['left']['avg_velocity']:.1f}")
        logger.info(f"  Right lane: {comparison['right']['count']} climbers")
        logger.info(f"    Avg velocity: {comparison['right']['avg_velocity']:.1f}")

        return comparison

    def export_summaries(self, output_dir: str = "data/processed/aggregates"):
        """Export all summaries to JSON files.

        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nExporting summaries to {output_dir}...")

        # Competition summaries
        comp_summaries = self.get_competition_summary()
        with open(output_dir / "competition_summaries.json", 'w') as f:
            json.dump(comp_summaries, f, indent=2)
        logger.info("  ✓ competition_summaries.json")

        # Leaderboard
        leaderboard = self.get_leaderboard(top_n=20)
        leaderboard.to_csv(output_dir / "leaderboard_top20.csv", index=False)
        logger.info("  ✓ leaderboard_top20.csv")

        # Lane comparison
        lane_comp = self.get_lane_comparison()
        with open(output_dir / "lane_comparison.json", 'w') as f:
            json.dump(lane_comp, f, indent=2)
        logger.info("  ✓ lane_comparison.json")

        # Overall statistics
        overall = {
            'total_races': int(self.df['race_name'].nunique()),
            'total_climbers': len(self.df),
            'competitions': int(self.df['competition'].nunique()),
            'calibrated_percentage': float(self.df['is_calibrated'].mean() * 100),
            'global_stats': {
                'avg_velocity': {
                    'mean': float(self.df['avg_vertical_velocity'].mean()),
                    'std': float(self.df['avg_vertical_velocity'].std()),
                    'min': float(self.df['avg_vertical_velocity'].min()),
                    'max': float(self.df['avg_vertical_velocity'].max()),
                },
                'max_velocity': {
                    'mean': float(self.df['max_vertical_velocity'].mean()),
                    'std': float(self.df['max_vertical_velocity'].std()),
                    'min': float(self.df['max_vertical_velocity'].min()),
                    'max': float(self.df['max_vertical_velocity'].max()),
                },
                'path_efficiency': {
                    'mean': float(self.df['path_efficiency'].mean()),
                    'std': float(self.df['path_efficiency'].std()),
                    'min': float(self.df['path_efficiency'].min()),
                    'max': float(self.df['path_efficiency'].max()),
                }
            }
        }

        with open(output_dir / "overall_statistics.json", 'w') as f:
            json.dump(overall, f, indent=2)
        logger.info("  ✓ overall_statistics.json")

        logger.info(f"\n✓ All summaries exported to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate competition statistics"
    )

    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="data/processed/metrics/aggregate_metrics.csv",
        help="Path to aggregate metrics CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/aggregates",
        help="Output directory for summaries"
    )

    args = parser.parse_args()

    # Initialize aggregator
    aggregator = CompetitionAggregator(args.metrics_csv)

    # Export summaries
    aggregator.export_summaries(args.output_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
