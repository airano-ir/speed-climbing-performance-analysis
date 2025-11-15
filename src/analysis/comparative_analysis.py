"""
Comparative Analysis Tools

Tools for comparing climbers, races, and identifying patterns.

Author: Speed Climbing Analysis Project
Date: 2025-11-15
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of a race comparison."""
    race_name: str
    winner: str  # 'left', 'right', or 'tie'
    left_metrics: Dict
    right_metrics: Dict
    differences: Dict
    summary: str


class RaceComparator:
    """Compare two climbers in a dual race."""

    def __init__(self, metrics_dir: str = "data/processed/metrics"):
        """Initialize comparator.

        Args:
            metrics_dir: Directory containing metrics JSON files
        """
        self.metrics_dir = Path(metrics_dir)

    def compare_dual_race(
        self,
        race_name: str,
        competition: str = "samples"
    ) -> Optional[ComparisonResult]:
        """Compare left vs right climber in a race.

        Args:
            race_name: Race identifier
            competition: Competition name

        Returns:
            ComparisonResult or None if data not available
        """
        # Load metrics for both lanes
        left_file = self.metrics_dir / competition / f"{race_name}_metrics_left.json"
        right_file = self.metrics_dir / competition / f"{race_name}_metrics_right.json"

        if not left_file.exists() or not right_file.exists():
            logger.warning(f"Metrics files not found for {race_name}")
            return None

        with open(left_file) as f:
            left_data = json.load(f)

        with open(right_file) as f:
            right_data = json.load(f)

        left_metrics = left_data['summary']
        right_metrics = right_data['summary']

        # Determine winner (higher max velocity = faster)
        left_max_vel = left_metrics['max_vertical_velocity']
        right_max_vel = right_metrics['max_vertical_velocity']

        if left_max_vel > right_max_vel * 1.05:  # 5% threshold
            winner = 'left'
        elif right_max_vel > left_max_vel * 1.05:
            winner = 'right'
        else:
            winner = 'tie'

        # Calculate differences
        differences = {
            'max_velocity_diff': right_max_vel - left_max_vel,
            'max_velocity_diff_pct': ((right_max_vel - left_max_vel) / left_max_vel * 100) if left_max_vel != 0 else 0,
            'avg_velocity_diff': right_metrics['avg_vertical_velocity'] - left_metrics['avg_vertical_velocity'],
            'path_efficiency_diff': right_metrics['path_efficiency'] - left_metrics['path_efficiency'],
            'smoothness_diff': right_metrics['smoothness_score'] - left_metrics['smoothness_score'],
        }

        # Generate summary
        summary = self._generate_summary(race_name, winner, left_metrics, right_metrics, differences)

        return ComparisonResult(
            race_name=race_name,
            winner=winner,
            left_metrics=left_metrics,
            right_metrics=right_metrics,
            differences=differences,
            summary=summary
        )

    def _generate_summary(
        self,
        race_name: str,
        winner: str,
        left_metrics: Dict,
        right_metrics: Dict,
        differences: Dict
    ) -> str:
        """Generate text summary of comparison."""
        lines = [
            f"Race: {race_name}",
            f"Winner: {winner.upper()}",
            "",
            "Max Velocity:",
            f"  Left:  {left_metrics['max_vertical_velocity']:.1f} px/s",
            f"  Right: {right_metrics['max_vertical_velocity']:.1f} px/s",
            f"  Diff:  {differences['max_velocity_diff']:.1f} px/s ({differences['max_velocity_diff_pct']:.1f}%)",
            "",
            "Path Efficiency:",
            f"  Left:  {left_metrics['path_efficiency']:.3f}",
            f"  Right: {right_metrics['path_efficiency']:.3f}",
            f"  Diff:  {differences['path_efficiency_diff']:.3f}",
        ]
        return "\n".join(lines)

    def compare_multiple_races(
        self,
        race_names: List[str],
        competition: str = "samples"
    ) -> List[ComparisonResult]:
        """Compare multiple races.

        Args:
            race_names: List of race identifiers
            competition: Competition name

        Returns:
            List of ComparisonResults
        """
        results = []

        for race_name in race_names:
            result = self.compare_dual_race(race_name, competition)
            if result:
                results.append(result)

        return results


class CompetitionAnalyzer:
    """Analyze competition-wide statistics."""

    def __init__(self, metrics_csv: str = "data/processed/metrics/aggregate_metrics.csv"):
        """Initialize analyzer.

        Args:
            metrics_csv: Path to aggregate metrics CSV
        """
        self.df = pd.read_csv(metrics_csv)
        logger.info(f"Loaded {len(self.df)} metrics from {metrics_csv}")

    def get_leaderboard(
        self,
        competition: Optional[str] = None,
        top_n: int = 10
    ) -> pd.DataFrame:
        """Get leaderboard sorted by performance.

        Args:
            competition: Filter by competition (None = all)
            top_n: Number of top performers

        Returns:
            DataFrame with leaderboard
        """
        df = self.df if competition is None else self.df[self.df['competition'] == competition]

        # Sort by max velocity (descending)
        leaderboard = df.nlargest(top_n, 'max_vertical_velocity')

        return leaderboard[['race_name', 'competition', 'lane',
                           'max_vertical_velocity', 'path_efficiency']]

    def analyze_progression(
        self,
        competition: str
    ) -> Dict:
        """Analyze performance progression through competition.

        Args:
            competition: Competition name

        Returns:
            Progression analysis dictionary
        """
        comp_df = self.df[self.df['competition'] == competition].copy()

        if len(comp_df) == 0:
            logger.warning(f"No data for competition: {competition}")
            return {}

        # Extract race numbers (assuming format: race001, race002, etc.)
        comp_df['race_num'] = comp_df['race_name'].str.extract(r'race(\d+)')[0].astype(int)

        # Sort by race number
        comp_df = comp_df.sort_values('race_num')

        progression = {
            'competition': competition,
            'total_races': int(comp_df['race_name'].nunique()),
            'velocity_trend': {
                'first_half_avg': float(comp_df.iloc[:len(comp_df)//2]['max_vertical_velocity'].mean()),
                'second_half_avg': float(comp_df.iloc[len(comp_df)//2:]['max_vertical_velocity'].mean()),
            },
            'efficiency_trend': {
                'first_half_avg': float(comp_df.iloc[:len(comp_df)//2]['path_efficiency'].mean()),
                'second_half_avg': float(comp_df.iloc[len(comp_df)//2:]['path_efficiency'].mean()),
            }
        }

        return progression


class WinnerPredictor:
    """Predict race winners based on metrics."""

    @staticmethod
    def predict_winner(
        left_metrics: Dict,
        right_metrics: Dict
    ) -> Tuple[str, float]:
        """Predict winner based on metrics.

        Args:
            left_metrics: Left climber metrics
            right_metrics: Right climber metrics

        Returns:
            Tuple of (predicted_winner, confidence)
        """
        # Simple heuristic: max velocity is strongest predictor
        left_score = left_metrics['max_vertical_velocity']
        right_score = right_metrics['max_vertical_velocity']

        # Add path efficiency bonus
        left_score += left_metrics['path_efficiency'] * 100  # Scale to comparable range
        right_score += right_metrics['path_efficiency'] * 100

        total = left_score + right_score
        if total == 0:
            return 'tie', 0.5

        left_prob = left_score / total
        right_prob = right_score / total

        if left_prob > 0.55:
            return 'left', left_prob
        elif right_prob > 0.55:
            return 'right', right_prob
        else:
            return 'tie', 0.5

    @staticmethod
    def validate_predictions(
        comparisons: List[ComparisonResult]
    ) -> Dict:
        """Validate predictions against actual results.

        Args:
            comparisons: List of comparison results

        Returns:
            Validation statistics
        """
        correct = 0
        total = 0

        for comp in comparisons:
            predicted, conf = WinnerPredictor.predict_winner(
                comp.left_metrics,
                comp.right_metrics
            )

            if predicted == comp.winner:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        return {
            'total_races': total,
            'correct_predictions': correct,
            'accuracy': accuracy
        }


def main():
    """Example usage."""
    logger.info("Comparative Analysis Tools - Examples\n")

    # Example 1: Compare a single race
    logger.info("Example 1: Compare single race")
    comparator = RaceComparator()
    result = comparator.compare_dual_race("Speed_finals_Chamonix_2024_race024", "samples")

    if result:
        print(result.summary)
        print()

    # Example 2: Leaderboard
    logger.info("\nExample 2: Competition leaderboard")
    analyzer = CompetitionAnalyzer()
    leaderboard = analyzer.get_leaderboard(top_n=5)
    print(leaderboard.to_string(index=False))

    # Example 3: Winner prediction
    logger.info("\nExample 3: Winner prediction")
    if result:
        predicted, conf = WinnerPredictor.predict_winner(
            result.left_metrics,
            result.right_metrics
        )
        print(f"Predicted winner: {predicted} (confidence: {conf:.2f})")
        print(f"Actual winner: {result.winner}")


if __name__ == "__main__":
    main()
