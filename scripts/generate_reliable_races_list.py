"""
Generate Reliable Races List
============================
Extract list of 114 reliable races (from 188 total - 74 suspicious).

This script:
1. Loads all 188 race IDs from race_segments directory
2. Loads 74 suspicious race IDs from progress tracker
3. Computes reliable races = ALL - SUSPICIOUS
4. Saves to data/processed/reliable_races_list.json

Usage:
    python scripts/generate_reliable_races_list.py

Output:
    data/processed/reliable_races_list.json
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def get_all_race_ids():
    """Get all race IDs from race_segments directory."""
    race_segments_dir = Path("data/race_segments")
    all_races = []

    # Iterate through competition directories
    for comp_dir in sorted(race_segments_dir.iterdir()):
        if not comp_dir.is_dir():
            continue

        # Find all .mp4 files (race videos)
        for video_file in sorted(comp_dir.glob("*.mp4")):
            # Extract race_id (filename without extension)
            race_id = video_file.stem
            all_races.append(race_id)

    return all_races


def get_suspicious_race_ids():
    """Get suspicious race IDs from progress tracker."""
    progress_tracker_path = Path("data/manual_review/progress_tracker.csv")
    suspicious_races = []

    with open(progress_tracker_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            race_id = row['Race_ID'].strip()
            suspicious_races.append(race_id)

    return suspicious_races


def categorize_races_by_competition(race_ids):
    """Categorize races by competition."""
    by_competition = defaultdict(int)

    for race_id in race_ids:
        # Extract competition name from race_id
        # Format: Speed_finals_<competition>_race###
        parts = race_id.split('_')
        if len(parts) >= 4:
            # Competition name is the third part (e.g., "Chamonix")
            # and fourth part (e.g., "2024")
            comp_name = f"{parts[2].lower()}_{parts[3]}"
            by_competition[comp_name] += 1

    return dict(by_competition)


def main():
    """Generate reliable races list."""
    print("=" * 60)
    print("Generating Reliable Races List")
    print("=" * 60)

    # Get all races
    print("\n1. Loading all races from race_segments directory...")
    all_races = get_all_race_ids()
    print(f"   Found {len(all_races)} total races")

    # Get suspicious races
    print("\n2. Loading suspicious races from progress tracker...")
    suspicious_races = get_suspicious_race_ids()
    print(f"   Found {len(suspicious_races)} suspicious races")

    # Compute reliable races
    print("\n3. Computing reliable races...")
    suspicious_set = set(suspicious_races)
    reliable_races = [race for race in all_races if race not in suspicious_set]
    print(f"   Computed {len(reliable_races)} reliable races")

    # Validate count
    expected_reliable = len(all_races) - len(suspicious_races)
    assert len(reliable_races) == expected_reliable, \
        f"Mismatch: {len(reliable_races)} != {expected_reliable}"

    # Categorize by competition
    print("\n4. Categorizing by competition...")
    by_competition = categorize_races_by_competition(reliable_races)

    print("\n   Distribution:")
    for comp, count in sorted(by_competition.items()):
        print(f"      {comp}: {count} races")

    # Create output data
    output_data = {
        "total_races": len(all_races),
        "suspicious_races": len(suspicious_races),
        "reliable_races": len(reliable_races),
        "reliable_race_ids": sorted(reliable_races),
        "by_competition": by_competition,
        "generation_date": datetime.now().isoformat(),
        "notes": "Races with verified detection quality, ready for pipeline processing"
    }

    # Save to JSON
    output_path = Path("data/processed/reliable_races_list.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n5. Saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("✅ Summary")
    print("=" * 60)
    print(f"Total races:      {len(all_races)}")
    print(f"Suspicious races: {len(suspicious_races)}")
    print(f"Reliable races:   {len(reliable_races)}")
    print(f"Success rate:     {len(reliable_races) / len(all_races) * 100:.1f}%")
    print("\nReliable races list saved successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
