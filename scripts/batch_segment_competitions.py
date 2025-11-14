#!/usr/bin/env python3
"""
Batch segment multiple competitions using manual timestamps.

This script processes Seoul, Villars, Chamonix, Innsbruck 2024 and Zilina 2025 competitions
using the ManualRaceSegmenter with YAML config files.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.manual_race_segmenter import ManualRaceSegmenter


def main():
    """Process all five competitions."""

    # Configuration
    competitions = [
        {
            'name': 'Seoul 2024',
            'config': 'configs/race_timestamps/seoul_2024.yaml',
            'output_dir': 'data/race_segments/seoul_2024',
            'total_races': 31
        },
        {
            'name': 'Villars 2024',
            'config': 'configs/race_timestamps/villars_2024.yaml',
            'output_dir': 'data/race_segments/villars_2024',
            'total_races': 24
        },
        {
            'name': 'Chamonix 2024',
            'config': 'configs/race_timestamps/chamonix_2024.yaml',
            'output_dir': 'data/race_segments/chamonix_2024',
            'total_races': 32
        },
        {
            'name': 'Innsbruck 2024',
            'config': 'configs/race_timestamps/innsbruck_2024.yaml',
            'output_dir': 'data/race_segments/innsbruck_2024',
            'total_races': 32
        },
        {
            'name': 'Zilina 2025',
            'config': 'configs/race_timestamps/zilina_2025.yaml',
            'output_dir': 'data/race_segments/zilina_2025',
            'total_races': 69  # 72 original - 3 deleted (races 13, 51, 55 incomplete)
        }
    ]

    # Parameters
    buffer_before = 1.5  # seconds
    buffer_after = 1.5   # seconds
    refine_detection = True  # Enable detection refinement
    save_video = True  # Save video clips

    # Initialize segmenter
    print("="*70)
    print("BATCH RACE SEGMENTATION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Competitions: {len(competitions)}")
    print(f"Total races: {sum(c['total_races'] for c in competitions)}")
    print(f"Buffer: {buffer_before}s before, {buffer_after}s after")
    print(f"Detection refinement: {'Enabled' if refine_detection else 'Disabled'}")
    print("="*70)

    segmenter = ManualRaceSegmenter(
        buffer_before=buffer_before,
        buffer_after=buffer_after,
        refine_detection=refine_detection
    )

    # Process each competition
    all_segments = []
    failed_competitions = []

    for i, comp in enumerate(competitions, 1):
        print(f"\n{'='*70}")
        print(f"Competition {i}/{len(competitions)}: {comp['name']}")
        print(f"{'='*70}")

        try:
            segments = segmenter.segment_from_config(
                Path(comp['config']),
                Path(comp['output_dir']),
                save_video=save_video
            )

            all_segments.extend(segments)

            if len(segments) == comp['total_races']:
                print(f"[OK] {comp['name']}: {len(segments)}/{comp['total_races']} races extracted")
            else:
                print(f"[WARNING] {comp['name']}: {len(segments)}/{comp['total_races']} races extracted")

        except Exception as e:
            print(f"[ERROR] {comp['name']} failed: {e}")
            failed_competitions.append(comp['name'])

    # Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total races extracted: {len(all_segments)}")
    print(f"Expected races: {sum(c['total_races'] for c in competitions)}")

    if failed_competitions:
        print(f"\nFailed competitions: {', '.join(failed_competitions)}")
    else:
        print("\n[OK] All competitions processed successfully!")

    print("="*70)

    # Output directories
    print("\nOutput directories:")
    for comp in competitions:
        output_dir = Path(comp['output_dir'])
        if output_dir.exists():
            video_files = list(output_dir.glob('*.mp4'))
            metadata_files = list(output_dir.glob('*_metadata.json'))
            print(f"  {comp['name']}:")
            print(f"    Path: {output_dir}")
            print(f"    Video clips: {len(video_files)}")
            print(f"    Metadata files: {len(metadata_files)}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
