#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select Sample Pose Files for GitHub
====================================
انتخاب خودکار بهترین pose files برای قرار دادن در GitHub

معیارهای انتخاب:
1. Detection rate بالا (>90%)
2. تنوع در competitions
3. حجم مناسب (<5 MB per file)
4. کوتاه‌ترین زمان‌ها (fastest athletes)

Usage:
    python scripts/select_sample_poses.py \
      --input data/processed/poses \
      --output data/processed/poses/samples \
      --count 10
"""

from pathlib import Path
import json
import shutil
import argparse
from typing import List, Dict
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_processing_summary(poses_dir: Path) -> Dict:
    """Load _processing_summary.json"""
    summary_file = poses_dir / "_processing_summary.json"
    if not summary_file.exists():
        print(f"Error: {summary_file} not found!")
        return {}

    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_best_poses(summary: Dict, count: int = 10) -> List[str]:
    """
    Select best pose files based on criteria.

    Returns:
        List of video filenames (without _poses.json suffix)
    """
    videos = summary.get('videos', [])

    if not videos:
        print("No videos in summary!")
        return []

    # Sort by multiple criteria
    # 1. Detection rate (both lanes) - descending
    # 2. Total frames (shorter = faster = better) - ascending
    sorted_videos = sorted(
        videos,
        key=lambda v: (
            -v.get('detection_rate_both', 0),  # Higher detection first
            v.get('total_frames', 999999)       # Shorter videos first
        )
    )

    # Filter: detection_rate_both > 0.90
    good_videos = [v for v in sorted_videos if v.get('detection_rate_both', 0) > 0.90]

    if len(good_videos) < count:
        print(f"Warning: Only {len(good_videos)} videos with >90% detection rate")
        good_videos = sorted_videos  # Fallback to all

    # Ensure diversity across competitions
    selected = []
    competitions_seen = set()

    # First pass: pick one from each competition
    for video in good_videos:
        video_name = video.get('video', '')
        # Extract competition name (e.g., "Chamonix_2024")
        parts = video_name.split('_')
        if len(parts) >= 3:
            competition = f"{parts[2]}_{parts[3]}"  # e.g., "Chamonix_2024"

            if competition not in competitions_seen:
                selected.append(video_name)
                competitions_seen.add(competition)

                if len(selected) >= count:
                    break

    # Second pass: fill remaining with best overall
    if len(selected) < count:
        for video in good_videos:
            video_name = video.get('video', '')
            if video_name not in selected:
                selected.append(video_name)
                if len(selected) >= count:
                    break

    return selected[:count]


def copy_samples(
    input_dir: Path,
    output_dir: Path,
    selected_videos: List[str]
):
    """Copy selected pose JSONs to samples directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    total_size = 0

    for video_name in selected_videos:
        # Convert video name to pose JSON name
        pose_name = video_name.replace('.mp4', '_poses.json')

        # Find the file (check all subdirectories)
        pose_file = None
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / pose_name
                if candidate.exists():
                    pose_file = candidate
                    break

        if pose_file is None:
            # Try directly in input_dir
            pose_file = input_dir / pose_name

        if pose_file and pose_file.exists():
            # Copy to output
            dest = output_dir / pose_name

            # Skip if already in samples directory
            if pose_file.parent == output_dir:
                print(f"⏭  Skipped: {pose_name} (already in samples)")
                continue

            shutil.copy2(pose_file, dest)

            size_mb = pose_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            copied += 1

            print(f"✓ Copied: {pose_name} ({size_mb:.1f} MB)")
        else:
            print(f"✗ Not found: {pose_name}")

    print(f"\nTotal: {copied} files copied ({total_size:.1f} MB)")
    return copied, total_size


def main():
    parser = argparse.ArgumentParser(
        description="Select sample pose files for GitHub"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/processed/poses'),
        help='Input directory with pose JSONs'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/processed/poses/samples'),
        help='Output directory for samples'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=10,
        help='Number of samples to select'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SAMPLE POSE SELECTION")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Count:  {args.count}")
    print()

    # Load summary
    summary = load_processing_summary(args.input)
    if not summary:
        sys.exit(1)

    print(f"Total videos in summary: {summary.get('total_videos', 0)}")
    print(f"Avg detection rate: {summary.get('avg_detection_rate_both', 0):.1%}")
    print()

    # Select best
    selected = select_best_poses(summary, args.count)

    if not selected:
        print("No videos selected!")
        sys.exit(1)

    print(f"Selected {len(selected)} videos:")
    for i, video in enumerate(selected, 1):
        print(f"  {i}. {video}")
    print()

    # Copy
    copied, total_size = copy_samples(args.input, args.output, selected)

    if copied > 0:
        print("\n✓ Sample selection complete!")
        print(f"  Files:  {copied}")
        print(f"  Size:   {total_size:.1f} MB")
        print(f"  Output: {args.output}")
    else:
        print("\n✗ No files copied!")
        sys.exit(1)


if __name__ == '__main__':
    main()
