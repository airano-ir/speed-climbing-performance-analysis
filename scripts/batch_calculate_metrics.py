"""
Batch Metrics Calculator - Calculate performance metrics for all processed races

This script loads pose data and calculates performance metrics for all races,
with optional calibration support.

Usage:
    # Calculate metrics for all races
    python scripts/batch_calculate_metrics.py

    # Calculate for specific competition
    python scripts/batch_calculate_metrics.py --competition chamonix_2024

    # Limit number of races
    python scripts/batch_calculate_metrics.py --max-races 20

    # Resume from previous run
    python scripts/batch_calculate_metrics.py --resume

Author: Speed Climbing Analysis Project
Date: 2025-11-15
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.performance_metrics import PerformanceAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchMetricsCalculator:
    """Calculate metrics for multiple races in batch."""

    def __init__(
        self,
        poses_dir: str = "data/processed/poses",
        calibration_dir: str = "data/processed/calibration",
        output_dir: str = "data/processed/metrics"
    ):
        """Initialize calculator.

        Args:
            poses_dir: Directory containing pose JSON files
            calibration_dir: Directory containing calibration files
            output_dir: Output directory for metrics
        """
        self.poses_dir = Path(poses_dir)
        self.calibration_dir = Path(calibration_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = PerformanceAnalyzer()

        logger.info("Batch Metrics Calculator initialized")
        logger.info(f"  Poses dir: {self.poses_dir}")
        logger.info(f"  Calibration dir: {self.calibration_dir}")
        logger.info(f"  Output dir: {self.output_dir}")

    def calculate_race_metrics(
        self,
        pose_file: Path,
        lane: str = 'left'
    ) -> Optional[Dict]:
        """Calculate metrics for a single race.

        Args:
            pose_file: Path to pose JSON file
            lane: Lane to analyze ('left' or 'right')

        Returns:
            Metrics dictionary or None if failed
        """
        logger.info(f"  Processing: {pose_file.name} (lane={lane})")

        try:
            # Find calibration file if available
            competition = pose_file.parent.name
            race_name = pose_file.stem.replace('_poses', '')
            cal_file = self.calibration_dir / competition / f"{race_name}_calibration.json"

            # Calculate metrics
            metrics = self.analyzer.analyze_pose_file(
                pose_file,
                lane=lane,
                calibration_path=cal_file if cal_file.exists() else None
            )

            if not metrics:
                logger.warning(f"    Failed to calculate metrics")
                return None

            # Create output dict
            result = {
                'race_name': race_name,
                'competition': competition,
                'lane': lane,
                'is_calibrated': metrics.is_calibrated,
                'units': metrics.units,
                'summary': {
                    'avg_vertical_velocity': float(metrics.avg_vertical_velocity),
                    'max_vertical_velocity': float(metrics.max_vertical_velocity),
                    'avg_acceleration': float(metrics.avg_acceleration),
                    'max_acceleration': float(metrics.max_acceleration),
                    'path_length': float(metrics.path_length),
                    'straight_distance': float(metrics.straight_distance),
                    'path_efficiency': float(metrics.path_efficiency),
                    'smoothness_score': float(metrics.smoothness_score)
                }
            }

            logger.info(f"    ✓ Metrics calculated (calibrated={metrics.is_calibrated})")
            return result

        except Exception as e:
            logger.error(f"    Error: {e}")
            return None

    def process_batch(
        self,
        pose_files: List[Path],
        resume: bool = False
    ) -> Dict:
        """Process multiple races.

        Args:
            pose_files: List of pose file paths
            resume: Skip already processed files

        Returns:
            Batch processing summary
        """
        results = []
        failed = []
        skipped = []

        start_time = time.time()

        for i, pose_file in enumerate(pose_files, 1):
            logger.info(f"\n[{i}/{len(pose_files)}] {pose_file.name}")

            competition = pose_file.parent.name
            race_name = pose_file.stem.replace('_poses', '')

            # Check if already processed
            metrics_file_left = self.output_dir / competition / f"{race_name}_metrics_left.json"
            metrics_file_right = self.output_dir / competition / f"{race_name}_metrics_right.json"

            if resume and metrics_file_left.exists() and metrics_file_right.exists():
                logger.info("  Skipping (already processed)")
                skipped.append(str(pose_file))
                continue

            try:
                # Process both lanes
                metrics_left = self.calculate_race_metrics(pose_file, lane='left')
                metrics_right = self.calculate_race_metrics(pose_file, lane='right')

                if metrics_left or metrics_right:
                    # Save individual metrics
                    metrics_file_left.parent.mkdir(parents=True, exist_ok=True)

                    if metrics_left:
                        with open(metrics_file_left, 'w') as f:
                            json.dump(metrics_left, f, indent=2)

                    if metrics_right:
                        with open(metrics_file_right, 'w') as f:
                            json.dump(metrics_right, f, indent=2)

                    # Add to results
                    race_result = {
                        'race_name': race_name,
                        'competition': competition,
                        'pose_file': str(pose_file),
                        'left': metrics_left,
                        'right': metrics_right
                    }
                    results.append(race_result)
                else:
                    failed.append(str(pose_file))

            except Exception as e:
                logger.error(f"  Error: {e}")
                failed.append(str(pose_file))

        elapsed = time.time() - start_time

        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(pose_files),
            'processed': len(results),
            'failed': len(failed),
            'skipped': len(skipped),
            'total_time_s': elapsed,
            'avg_time_per_race_s': elapsed / len(results) if results else 0,
            'results': results,
            'failed_files': failed,
            'skipped_files': skipped
        }

        # Save summary
        summary_file = self.output_dir / "batch_metrics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH METRICS CALCULATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total: {len(pose_files)} files")
        logger.info(f"Processed: {len(results)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Skipped: {len(skipped)}")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Avg per race: {summary['avg_time_per_race_s']:.1f}s")
        logger.info(f"\nSummary saved to: {summary_file}")

        # Create aggregate CSV
        self._create_aggregate_csv(results)

        return summary

    def _create_aggregate_csv(self, results: List[Dict]):
        """Create aggregate CSV from results.

        Args:
            results: List of race results
        """
        rows = []

        for result in results:
            race_name = result['race_name']
            competition = result['competition']

            for lane in ['left', 'right']:
                metrics = result.get(lane)
                if metrics:
                    row = {
                        'race_name': race_name,
                        'competition': competition,
                        'lane': lane,
                        'is_calibrated': metrics['is_calibrated'],
                        'units': metrics['units'],
                        **metrics['summary']
                    }
                    rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            csv_file = self.output_dir / "aggregate_metrics.csv"
            df.to_csv(csv_file, index=False)

            logger.info(f"\nAggregate CSV created: {csv_file}")
            logger.info(f"  Rows: {len(df)}")
            logger.info(f"  Competitions: {df['competition'].nunique()}")
            logger.info(f"  Calibrated: {df['is_calibrated'].sum()}/{len(df)}")


def find_pose_files(
    poses_dir: str,
    competition: Optional[str] = None,
    max_races: Optional[int] = None
) -> List[Path]:
    """Find pose JSON files.

    Args:
        poses_dir: Poses directory
        competition: Specific competition (optional)
        max_races: Maximum number to process (optional)

    Returns:
        List of pose file paths
    """
    poses_dir = Path(poses_dir)

    if not poses_dir.exists():
        logger.error(f"Directory not found: {poses_dir}")
        return []

    # Find files
    if competition:
        pattern = f"{competition}/*_poses.json"
        files = list(poses_dir.glob(pattern))
        logger.info(f"Found {len(files)} pose files in {competition}")
    else:
        files = list(poses_dir.glob("*/*_poses.json"))
        logger.info(f"Found {len(files)} pose files across all competitions")

    # Limit if requested
    if max_races and len(files) > max_races:
        files = files[:max_races]
        logger.info(f"Limited to first {max_races} files")

    return sorted(files)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate performance metrics for all processed races"
    )

    parser.add_argument(
        "--poses-dir",
        type=str,
        default="data/processed/poses",
        help="Poses directory"
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="data/processed/calibration",
        help="Calibration directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/metrics",
        help="Output directory"
    )
    parser.add_argument(
        "--competition",
        type=str,
        help="Process specific competition only"
    )
    parser.add_argument(
        "--max-races",
        type=int,
        help="Maximum number of races to process"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed files"
    )

    args = parser.parse_args()

    # Find pose files
    pose_files = find_pose_files(
        args.poses_dir,
        competition=args.competition,
        max_races=args.max_races
    )

    if not pose_files:
        logger.error("No pose files found")
        return

    # Initialize calculator
    calculator = BatchMetricsCalculator(
        poses_dir=args.poses_dir,
        calibration_dir=args.calibration_dir,
        output_dir=args.output_dir
    )

    # Process
    summary = calculator.process_batch(pose_files, resume=args.resume)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
