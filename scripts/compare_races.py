"""
Race Comparison Script

Compare climbers in dual races and generate comparison reports.

Usage:
    # Compare a single race
    python scripts/compare_races.py --race race024 --competition samples

    # Compare multiple races
    python scripts/compare_races.py --all --competition samples

    # Generate comparison report
    python scripts/compare_races.py --all --report data/processed/reports/comparisons.json

Author: Speed Climbing Analysis Project
Date: 2025-11-15
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.comparative_analysis import RaceComparator, WinnerPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_single_race(race_name: str, competition: str):
    """Compare a single race."""
    comparator = RaceComparator()

    result = comparator.compare_dual_race(race_name, competition)

    if not result:
        logger.error(f"Could not load race: {race_name}")
        return

    # Print summary
    print("\n" + "="*60)
    print(result.summary)
    print("="*60)

    # Prediction
    predicted, conf = WinnerPredictor.predict_winner(
        result.left_metrics,
        result.right_metrics
    )

    print(f"\nPrediction: {predicted.upper()} (confidence: {conf:.2%})")
    print(f"Actual:     {result.winner.upper()}")

    correct = "✓ CORRECT" if predicted == result.winner else "✗ INCORRECT"
    print(f"Result:     {correct}\n")


def compare_all_races(competition: str, output_file: str = None):
    """Compare all races in a competition."""
    comparator = RaceComparator()

    # Find all race metrics
    metrics_dir = Path("data/processed/metrics") / competition
    if not metrics_dir.exists():
        logger.error(f"No metrics found for competition: {competition}")
        return

    # Get unique race names
    race_files = list(metrics_dir.glob("*_metrics_left.json"))
    race_names = [f.stem.replace('_metrics_left', '') for f in race_files]

    logger.info(f"Found {len(race_names)} races in {competition}")

    # Compare all
    results = []
    correct = 0
    total = 0

    print("\n" + "="*60)
    print(f"COMPARING ALL RACES - {competition.upper()}")
    print("="*60 + "\n")

    for race_name in sorted(race_names):
        result = comparator.compare_dual_race(race_name, competition)

        if not result:
            continue

        # Prediction
        predicted, conf = WinnerPredictor.predict_winner(
            result.left_metrics,
            result.right_metrics
        )

        is_correct = predicted == result.winner
        if is_correct:
            correct += 1
        total += 1

        # Print concise result
        status = "✓" if is_correct else "✗"
        print(f"{status} {race_name}: Winner={result.winner.upper()}, Predicted={predicted.upper()} ({conf:.0%})")

        # Store result
        results.append({
            'race_name': race_name,
            'winner': result.winner,
            'predicted': predicted,
            'confidence': conf,
            'correct': is_correct,
            'max_velocity_left': result.left_metrics['max_vertical_velocity'],
            'max_velocity_right': result.right_metrics['max_vertical_velocity'],
            'velocity_diff_pct': result.differences['max_velocity_diff_pct']
        })

    # Summary
    accuracy = correct / total if total > 0 else 0

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total races:    {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy:       {accuracy:.1%}\n")

    # Save report if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'competition': competition,
            'total_races': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare climbers in speed climbing races"
    )

    parser.add_argument(
        "--race",
        type=str,
        help="Single race to compare (e.g., race024)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all races in competition"
    )
    parser.add_argument(
        "--competition",
        type=str,
        default="samples",
        help="Competition name (default: samples)"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Output file for comparison report (JSON)"
    )

    args = parser.parse_args()

    if not args.race and not args.all:
        parser.error("Either --race or --all must be specified")

    if args.race:
        # Add prefix if not present
        race_name = args.race
        if not race_name.startswith("Speed_finals_"):
            # Try to find full name
            metrics_dir = Path("data/processed/metrics") / args.competition
            if metrics_dir.exists():
                matches = list(metrics_dir.glob(f"*{race_name}*_metrics_left.json"))
                if matches:
                    race_name = matches[0].stem.replace('_metrics_left', '')

        compare_single_race(race_name, args.competition)
    elif args.all:
        compare_all_races(args.competition, args.report)


if __name__ == "__main__":
    main()
