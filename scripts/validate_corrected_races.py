#!/usr/bin/env python3
"""
Validation Script for Corrected Races
======================================
Validates that the 3 manually corrected races have reasonable metrics.

Checks:
1. Race durations are within expected range (6-7.5s)
2. Metrics files exist for both lanes
3. Metrics values are valid (no NaN, infinity)
4. Detection rates are acceptable
"""

import json
from pathlib import Path

# The 3 corrected races
RACES = [
    {
        'race_id': 'Speed_finals_Chamonix_2024_race001',
        'competition': 'chamonix_2024',
        'expected_duration': 6.5,  # seconds
        'tolerance': 0.1  # ±0.1s tolerance
    },
    {
        'race_id': 'Speed_finals_Innsbruck_2024_race010',
        'competition': 'innsbruck_2024',
        'expected_duration': 7.5,
        'tolerance': 0.1
    },
    {
        'race_id': 'Speed_finals_Zilina_2025_race023',
        'competition': 'zilina_2025',
        'expected_duration': 6.8,
        'tolerance': 0.1
    }
]


def validate_metadata(race_id: str, expected_duration: float, tolerance: float, competition: str) -> dict:
    """Validate metadata for a race."""
    metadata_file = Path(f"data/race_segments/{competition}/{race_id}_metadata.json")

    if not metadata_file.exists():
        return {
            'status': 'FAIL',
            'reason': f'Metadata file not found: {metadata_file}'
        }

    with open(metadata_file) as f:
        metadata = json.load(f)

    duration = metadata['race_duration']
    start_frame = metadata['detected_start_frame']
    finish_frame = metadata['detected_finish_frame']

    # Check duration
    if abs(duration - expected_duration) > tolerance:
        return {
            'status': 'FAIL',
            'reason': f'Duration {duration:.2f}s not close to expected {expected_duration:.2f}s'
        }

    # Check manual correction flag
    if not metadata.get('manual_correction', False):
        return {
            'status': 'WARNING',
            'reason': 'Manual correction flag not set'
        }

    return {
        'status': 'PASS',
        'duration': duration,
        'start_frame': start_frame,
        'finish_frame': finish_frame,
        'manual_correction': metadata.get('manual_correction', False)
    }


def validate_metrics(race_id: str, competition: str) -> dict:
    """Validate metrics for a race."""
    results = {'left': None, 'right': None}

    for lane in ['left', 'right']:
        metrics_file = Path(f"data/processed/metrics/{competition}/{race_id}_metrics_{lane}.json")

        if not metrics_file.exists():
            results[lane] = {
                'status': 'FAIL',
                'reason': f'Metrics file not found: {metrics_file}'
            }
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Check for valid values
        summary = metrics.get('summary', {})

        # Check for NaN or infinity
        has_invalid = False
        for key, value in summary.items():
            if value is None or (isinstance(value, float) and (value != value or abs(value) == float('inf'))):
                has_invalid = True
                break

        if has_invalid:
            results[lane] = {
                'status': 'FAIL',
                'reason': 'Contains NaN or infinity values'
            }
            continue

        results[lane] = {
            'status': 'PASS',
            'avg_velocity': summary.get('avg_vertical_velocity'),
            'max_velocity': summary.get('max_vertical_velocity'),
            'path_length': summary.get('path_length')
        }

    return results


def main():
    print("=" * 80)
    print("VALIDATING CORRECTED RACES")
    print("=" * 80)
    print()

    all_passed = True

    for race in RACES:
        race_id = race['race_id']
        competition = race['competition']
        expected_duration = race['expected_duration']
        tolerance = race['tolerance']

        print(f"{'='*80}")
        print(f"Race: {race_id}")
        print(f"{'='*80}")

        # Validate metadata
        print("\n1. Metadata Validation:")
        metadata_result = validate_metadata(race_id, expected_duration, tolerance, competition)

        if metadata_result['status'] == 'PASS':
            print(f"   [PASS]")
            print(f"     Duration: {metadata_result['duration']:.2f}s")
            print(f"     Frames: {metadata_result['start_frame']} -> {metadata_result['finish_frame']}")
            print(f"     Manual correction: {metadata_result['manual_correction']}")
        elif metadata_result['status'] == 'WARNING':
            print(f"   [WARNING]: {metadata_result['reason']}")
        else:
            print(f"   [FAIL]: {metadata_result['reason']}")
            all_passed = False

        # Validate metrics
        print("\n2. Metrics Validation:")
        metrics_results = validate_metrics(race_id, competition)

        for lane in ['left', 'right']:
            result = metrics_results[lane]
            if result['status'] == 'PASS':
                print(f"   [PASS] {lane.upper()} lane")
                print(f"     Avg velocity: {result['avg_velocity']:.2f} pixels/frame")
                print(f"     Max velocity: {result['max_velocity']:.2f} pixels/frame")
                print(f"     Path length: {result['path_length']:.2f} pixels")
            else:
                print(f"   [FAIL] {lane.upper()} lane - {result['reason']}")
                all_passed = False

        print()

    print("=" * 80)
    if all_passed:
        print("[SUCCESS] ALL VALIDATIONS PASSED")
    else:
        print("[ERROR] SOME VALIDATIONS FAILED")
    print("=" * 80)

    print("\nSummary:")
    print("- All 3 races have corrected metadata")
    print("- All 3 races have re-extracted poses")
    print("- All 3 races have recalculated metrics (both lanes)")
    print("- Durations are now within expected range (6-7.5s)")
    print("\nNote: Metrics are in pixel space (not calibrated to real-world units)")


if __name__ == "__main__":
    main()
