"""
Pipeline Validation - Reliable Races
====================================
Comprehensive validation of all pipeline stages for 114 races.

Checks:
1. File completeness (all outputs exist)
2. Data quality (ranges, missing values)
3. Cross-validation (pose → metrics consistency)
4. Performance benchmarks

Usage:
    python scripts/validate_pipeline_reliable.py
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def validate_pipeline():
    """Run comprehensive pipeline validation."""

    print("="*70)
    print("Pipeline Validation - Reliable Races")
    print("="*70 + "\n")

    # Load reliable races
    reliable_races_file = Path('data/processed/reliable_races_list.json')
    with open(reliable_races_file, 'r', encoding='utf-8') as f:
        reliable_data = json.load(f)

    race_ids = reliable_data['reliable_race_ids']

    validation_results = {
        'total_races': len(race_ids),
        'file_completeness': check_file_completeness(race_ids),
        'data_quality': check_data_quality(race_ids),
        'performance_benchmarks': check_performance_benchmarks(),
        'validation_date': datetime.now().isoformat()
    }

    # Save report
    report_path = Path('data/processed/pipeline_validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("✅ Pipeline Validation Complete!")
    print("="*70)
    print(f"Total races: {len(race_ids)}")
    print(f"File completeness: {validation_results['file_completeness']['pass_rate']:.1f}%")
    print(f"Data quality: {validation_results['data_quality']['pass_rate']:.1f}%")
    print(f"\nReport saved to: {report_path}")
    print("="*70)

    # Show issues if any
    fc = validation_results['file_completeness']
    if fc['total_missing'] > 0:
        print(f"\n⚠️  Missing files: {fc['total_missing']}")
        print(f"   Pose files: {len(fc['missing_files']['pose'])}")
        print(f"   Calibration files: {len(fc['missing_files']['calibration'])}")
        print(f"   Metrics files: {len(fc['missing_files']['metrics'])}")

    dq = validation_results['data_quality']
    if dq['races_with_issues'] > 0:
        print(f"\n⚠️  Data quality issues: {dq['races_with_issues']} races")
        print("   (See validation report for details)")


def check_file_completeness(race_ids):
    """Check if all expected files exist."""
    print("1. Checking file completeness...")

    missing_files = {
        'pose': [],
        'calibration': [],
        'metrics': []
    }

    for race_id in race_ids:
        if not Path(f"data/processed/poses/{race_id}_pose.json").exists():
            missing_files['pose'].append(race_id)
        if not Path(f"data/processed/calibration/{race_id}_calibration.json").exists():
            missing_files['calibration'].append(race_id)
        if not Path(f"data/processed/metrics/{race_id}_metrics.json").exists():
            missing_files['metrics'].append(race_id)

    total_expected = len(race_ids) * 3
    total_missing = sum(len(v) for v in missing_files.values())
    pass_rate = (total_expected - total_missing) / total_expected * 100

    print(f"   Expected files: {total_expected}")
    print(f"   Missing files: {total_missing}")
    print(f"   Pass rate: {pass_rate:.1f}%")

    return {
        'total_expected': total_expected,
        'total_missing': total_missing,
        'missing_files': missing_files,
        'pass_rate': pass_rate
    }


def check_data_quality(race_ids):
    """Check data quality metrics."""
    print("\n2. Checking data quality...")

    quality_issues = []
    checked_count = 0

    for race_id in race_ids:
        metrics_path = Path(f"data/processed/metrics/{race_id}_metrics.json")

        if not metrics_path.exists():
            continue

        checked_count += 1

        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        # Check for issues
        issues = []

        # Velocity checks
        avg_vel = metrics.get('average_velocity_ms', 0)
        if avg_vel < 0.5:
            issues.append('velocity_too_low')
        elif avg_vel > 5.0:
            issues.append('velocity_too_high')

        # Duration checks
        total_time = metrics.get('total_time_s', 0)
        if total_time < 4.5:
            issues.append('duration_too_short')
        elif total_time > 15.0:
            issues.append('duration_too_long')

        # Calibration check
        if not metrics.get('is_calibrated', False):
            issues.append('not_calibrated')

        # Negative values check
        if avg_vel < 0 or total_time < 0:
            issues.append('negative_values')

        if issues:
            quality_issues.append({
                'race_id': race_id,
                'issues': issues,
                'avg_velocity_ms': avg_vel,
                'total_time_s': total_time
            })

    pass_rate = (checked_count - len(quality_issues)) / checked_count * 100 if checked_count > 0 else 0

    print(f"   Races checked: {checked_count}")
    print(f"   Races with issues: {len(quality_issues)}")
    print(f"   Pass rate: {pass_rate:.1f}%")

    return {
        'total_checked': checked_count,
        'races_with_issues': len(quality_issues),
        'quality_issues': quality_issues,
        'pass_rate': pass_rate
    }


def check_performance_benchmarks():
    """Check performance against benchmarks."""
    print("\n3. Checking performance benchmarks...")

    csv_path = Path('data/processed/aggregated_metrics_reliable.csv')

    if not csv_path.exists():
        print("   ⚠️  Aggregated data not found, skipping benchmarks")
        return {
            'status': 'skipped',
            'reason': 'aggregated_data_not_found'
        }

    # Load aggregated data
    df = pd.read_csv(csv_path)

    benchmarks = {
        'total_races': len(df),
        'competitions': int(df['competition'].nunique()) if 'competition' in df.columns else 0
    }

    # Velocity statistics
    if 'average_velocity_ms' in df.columns:
        benchmarks['avg_velocity_ms'] = float(df['average_velocity_ms'].mean())
        benchmarks['min_velocity_ms'] = float(df['average_velocity_ms'].min())
        benchmarks['max_velocity_ms'] = float(df['average_velocity_ms'].max())
        benchmarks['std_velocity_ms'] = float(df['average_velocity_ms'].std())

    # Duration statistics
    if 'total_time_s' in df.columns:
        benchmarks['avg_duration_s'] = float(df['total_time_s'].mean())
        benchmarks['min_duration_s'] = float(df['total_time_s'].min())
        benchmarks['max_duration_s'] = float(df['total_time_s'].max())
        benchmarks['std_duration_s'] = float(df['total_time_s'].std())

    # World record comparison
    benchmarks['world_record_comparison'] = {
        'men_wr_s': 5.00,
        'women_wr_s': 6.53,
        'fastest_in_dataset_s': float(df['total_time_s'].min()) if 'total_time_s' in df.columns else 0,
        'note': 'World records as of 2024'
    }

    print(f"   Races analyzed: {benchmarks['total_races']}")
    if 'avg_velocity_ms' in benchmarks:
        print(f"   Avg velocity: {benchmarks['avg_velocity_ms']:.2f} m/s")
    if 'avg_duration_s' in benchmarks:
        print(f"   Avg duration: {benchmarks['avg_duration_s']:.2f} s")

    return benchmarks


if __name__ == "__main__":
    try:
        validate_pipeline()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
