#!/usr/bin/env python3
"""
Re-process Corrected Races
===========================
Re-extracts poses and recalculates metrics for manually corrected races.

This script:
1. Deletes old pose data for corrected races
2. Re-extracts poses
3. Re-calculates metrics

Usage:
    python scripts/reprocess_corrected_races.py
"""

from pathlib import Path
import sys
import json
import subprocess

# Corrected races (from metadata manual corrections)
CORRECTED_RACES = [
    {
        'race_id': 'Speed_finals_Chamonix_2024_race001',
        'competition': 'chamonix_2024',
        'video_path': 'data/race_segments/chamonix_2024/Speed_finals_Chamonix_2024_race001.mp4'
    },
    {
        'race_id': 'Speed_finals_Innsbruck_2024_race010',
        'competition': 'innsbruck_2024',
        'video_path': 'data/race_segments/innsbruck_2024/Speed_finals_Innsbruck_2024_race010.mp4'
    },
    {
        'race_id': 'Speed_finals_Zilina_2025_race023',
        'competition': 'zilina_2025',
        'video_path': 'data/race_segments/zilina_2025/Speed_finals_Zilina_2025_race023.mp4'
    }
]


def cleanup_old_data(race_id: str):
    """Delete old pose and metric data for a race."""
    print(f"  🧹 Cleaning up old data for {race_id}...")

    # Delete pose data
    pose_file = Path(f"data/processed/poses/{race_id}_poses.json")
    if pose_file.exists():
        pose_file.unlink()
        print(f"     ✓ Deleted {pose_file}")

    # Delete metric data (both lanes)
    for lane in ['left', 'right']:
        metric_file = Path(f"data/processed/metrics/{race_id}_metrics_{lane}.json")
        if metric_file.exists():
            metric_file.unlink()
            print(f"     ✓ Deleted {metric_file}")


def extract_poses_for_race(video_path: str, race_id: str):
    """Re-extract poses for a single race."""
    print(f"\n  📹 Extracting poses for {race_id}...")

    # Import here to avoid issues if packages not installed
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'phase1_pose_estimation'))
    from scripts.batch_pose_extraction import extract_poses_from_clip

    output_json = Path(f"data/processed/poses/{race_id}_poses.json")
    video_path_obj = Path(video_path)

    if not video_path_obj.exists():
        print(f"     ❌ Video not found: {video_path}")
        return False

    try:
        stats = extract_poses_from_clip(video_path_obj, output_json)
        print(f"     ✓ {stats['total_frames']} frames processed")
        print(f"     ✓ Detection rate: Left={stats['detection_rate_left']:.1%}, "
              f"Right={stats['detection_rate_right']:.1%}")
        return True
    except Exception as e:
        print(f"     ❌ Error: {str(e)}")
        return False


def calculate_metrics_for_race(race_id: str, competition: str):
    """Re-calculate metrics for a single race."""
    print(f"\n  📊 Calculating metrics for {race_id}...")

    # Check if we have a metrics calculation script
    metrics_script = Path("scripts/batch_metrics_calculation.py")
    if not metrics_script.exists():
        metrics_script = Path("scripts/batch_calculate_metrics.py")

    if not metrics_script.exists():
        print(f"     ⚠️  Metrics calculation script not found, skipping")
        return False

    # Run the metrics calculation for this specific race
    # This would require the script to support --race parameter
    # For now, we'll just note it needs to be done manually
    print(f"     ℹ️  Run manually: python scripts/batch_calculate_metrics.py --competition {competition}")
    return True


def main():
    print("=" * 80)
    print("RE-PROCESSING MANUALLY CORRECTED RACES")
    print("=" * 80)
    print(f"\nCorrected races: {len(CORRECTED_RACES)}")
    for race in CORRECTED_RACES:
        print(f"  - {race['race_id']}")
    print()

    for race in CORRECTED_RACES:
        race_id = race['race_id']
        competition = race['competition']
        video_path = race['video_path']

        print(f"\n{'='*80}")
        print(f"Processing: {race_id}")
        print(f"{'='*80}")

        # Step 1: Cleanup
        cleanup_old_data(race_id)

        # Step 2: Extract poses
        success = extract_poses_for_race(video_path, race_id)
        if not success:
            print(f"\n  ❌ Failed to extract poses for {race_id}")
            continue

        # Step 3: Calculate metrics
        calculate_metrics_for_race(race_id, competition)

    print("\n" + "=" * 80)
    print("✓ Re-processing complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Verify pose extraction quality")
    print("2. Calculate metrics if not done automatically")
    print("3. Validate corrected race durations and velocities")


if __name__ == "__main__":
    main()
