"""
Batch Metrics Calculation - Reliable Races Only
===============================================
Calculate performance metrics for 114 verified races with calibration.

Features:
- Uses corrected frame ranges from metadata
- Applies camera calibration for accurate measurements
- Validates metric quality

Usage:
    python scripts/batch_metrics_reliable.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.analysis.performance_metrics import PerformanceMetricsCalculator
import json
from tqdm import tqdm
from datetime import datetime


def calculate_metrics_for_reliable_races():
    """Calculate metrics for all reliable races."""

    # Load reliable races list
    reliable_races_file = Path('data/processed/reliable_races_list.json')
    with open(reliable_races_file, 'r', encoding='utf-8') as f:
        reliable_data = json.load(f)

    race_ids = reliable_data['reliable_race_ids']

    print("="*70)
    print(f"Batch Metrics Calculation - Reliable Races")
    print("="*70)
    print(f"Total races to process: {len(race_ids)}")
    print(f"Features: Calibrated measurements, frame filtering, quality validation")
    print(f"Estimated time: 3-4 hours")
    print(f"Output directory: data/processed/metrics/")
    print("="*70 + "\n")

    success_count = 0
    failed_races = []
    metrics_quality = []

    for race_id in tqdm(race_ids, desc="Calculating metrics"):
        try:
            # Find files
            pose_path = Path(f"data/processed/poses/{race_id}_pose.json")
            calib_path = Path(f"data/processed/calibration/{race_id}_calibration.json")
            metadata_path = find_metadata_path(race_id)

            if not pose_path.exists():
                raise FileNotFoundError(f"Pose file not found: {pose_path}")
            if not calib_path.exists():
                raise FileNotFoundError(f"Calibration file not found: {calib_path}")

            # Load metadata for frame range
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Calculate metrics
            calculator = PerformanceMetricsCalculator(
                pose_file=str(pose_path),
                calibration_file=str(calib_path),
                fps=metadata.get('fps', 30.0)
            )

            # Get metrics
            metrics = calculator.calculate_all_metrics(
                start_frame=metadata.get('detected_start_frame'),
                end_frame=metadata.get('detected_finish_frame')
            )

            # Add race info
            metrics['race_id'] = race_id
            metrics['competition'] = metadata.get('competition', 'unknown')
            metrics['calculation_date'] = datetime.now().isoformat()

            # Save metrics
            output_path = Path(f"data/processed/metrics/{race_id}_metrics.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            # Validate quality
            avg_velocity = metrics.get('average_velocity_ms', 0)
            total_time = metrics.get('total_time_s', 0)

            metrics_quality.append({
                'race_id': race_id,
                'avg_velocity_ms': avg_velocity,
                'total_time_s': total_time,
                'is_calibrated': metrics.get('is_calibrated', False),
                'valid_velocity': 1.0 <= avg_velocity <= 3.0,  # Typical range
                'valid_duration': 4.5 <= total_time <= 15.0     # Valid range
            })

            success_count += 1

        except Exception as e:
            tqdm.write(f"Failed for {race_id}: {e}")
            failed_races.append({'race_id': race_id, 'error': str(e)})

    # Generate report
    valid_velocity_count = sum(1 for m in metrics_quality if m['valid_velocity'])
    valid_duration_count = sum(1 for m in metrics_quality if m['valid_duration'])
    calibrated_count = sum(1 for m in metrics_quality if m['is_calibrated'])

    validity_rate = valid_velocity_count / len(metrics_quality) * 100 if metrics_quality else 0

    # Calculate statistics
    velocities = [m['avg_velocity_ms'] for m in metrics_quality if m['valid_velocity']]
    durations = [m['total_time_s'] for m in metrics_quality if m['valid_duration']]

    report = {
        'total_races': len(race_ids),
        'successful': success_count,
        'failed': len(failed_races),
        'failed_races': failed_races,
        'quality_metrics': {
            'validity_rate_percent': validity_rate,
            'valid_velocity_count': valid_velocity_count,
            'valid_duration_count': valid_duration_count,
            'calibrated_count': calibrated_count,
            'avg_velocity_ms': sum(velocities) / len(velocities) if velocities else 0,
            'avg_duration_s': sum(durations) / len(durations) if durations else 0,
            'min_velocity_ms': min(velocities) if velocities else 0,
            'max_velocity_ms': max(velocities) if velocities else 0,
            'min_duration_s': min(durations) if durations else 0,
            'max_duration_s': max(durations) if durations else 0
        },
        'metrics_quality': metrics_quality,
        'completion_date': datetime.now().isoformat()
    }

    report_path = Path('data/processed/metrics_calculation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*70)
    print("✅ Metrics calculation complete!")
    print("="*70)
    print(f"Successful: {success_count}/{len(race_ids)}")
    print(f"Failed: {len(failed_races)}")
    print(f"Calibrated races: {calibrated_count}/{len(metrics_quality)}")
    print(f"Valid velocity: {valid_velocity_count}/{len(metrics_quality)} ({validity_rate:.1f}%)")
    print(f"Valid duration: {valid_duration_count}/{len(metrics_quality)}")
    print(f"\nPerformance Statistics:")
    print(f"  Average velocity: {report['quality_metrics']['avg_velocity_ms']:.2f} m/s")
    print(f"  Average duration: {report['quality_metrics']['avg_duration_s']:.2f} s")
    print(f"  Velocity range: {report['quality_metrics']['min_velocity_ms']:.2f} - {report['quality_metrics']['max_velocity_ms']:.2f} m/s")
    print(f"  Duration range: {report['quality_metrics']['min_duration_s']:.2f} - {report['quality_metrics']['max_duration_s']:.2f} s")
    print(f"\nReport saved to: {report_path}")
    print("="*70)

    if failed_races:
        print("\n⚠️  Failed races:")
        for race in failed_races[:10]:
            print(f"   - {race['race_id']}: {race['error']}")
        if len(failed_races) > 10:
            print(f"   ... and {len(failed_races) - 10} more")


def find_metadata_path(race_id: str) -> Path:
    """Find metadata path for a race."""
    race_segments_dir = Path("data/race_segments")

    for comp_dir in race_segments_dir.iterdir():
        if not comp_dir.is_dir():
            continue

        metadata_path = comp_dir / f"{race_id}_metadata.json"
        if metadata_path.exists():
            return metadata_path

    raise FileNotFoundError(f"Metadata not found for {race_id}")


if __name__ == "__main__":
    try:
        calculate_metrics_for_reliable_races()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print("Partial results have been saved to data/processed/metrics/")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
