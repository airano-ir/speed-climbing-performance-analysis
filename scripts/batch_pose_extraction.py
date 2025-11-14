#!/usr/bin/env python3
"""
Batch Pose Extraction Script
=============================
استخراج BlazePose keypoints از race segments

Usage:
    python scripts/batch_pose_extraction.py [--max-races N] [--competition NAME]

Examples:
    # Process first 3 races from all competitions
    python scripts/batch_pose_extraction.py --max-races 3

    # Process all Seoul 2024 races
    python scripts/batch_pose_extraction.py --competition seoul_2024

    # Process all 188 races
    python scripts/batch_pose_extraction.py

Output:
    data/processed/poses/<video_name>_poses.json

Author: Speed Climbing Performance Analysis Project
Date: 2025-11-14
"""

from pathlib import Path
import sys
import json
import cv2
import argparse
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'phase1_pose_estimation'))

from dual_lane_detector import DualLaneDetector


def extract_poses_from_clip(video_path: Path, output_json: Path) -> dict:
    """
    استخراج poses از یک race clip.

    Args:
        video_path: مسیر فایل ویدیو
        output_json: مسیر خروجی JSON

    Returns:
        dict: آمار پردازش شامل تعداد فریم‌ها و میزان detection
    """
    detector = DualLaneDetector(boundary_detection_method='fixed')
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # اطلاعات ویدیو
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    frames_with_left = 0
    frames_with_right = 0
    frames_with_both = 0
    frame_count = 0

    with detector:
        pbar = tqdm(total=total_frames, desc=f"  {video_path.name}", leave=False)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps if fps > 0 else 0
            result = detector.process_frame(frame, frame_count, timestamp)
            frame_count += 1

            # آمار
            has_left = result.left_climber is not None
            has_right = result.right_climber is not None

            if has_left:
                frames_with_left += 1
            if has_right:
                frames_with_right += 1
            if has_left and has_right:
                frames_with_both += 1

            results.append({
                'frame_id': result.frame_id,
                'timestamp': result.timestamp,
                'left_climber': result.left_climber.to_dict() if result.left_climber else None,
                'right_climber': result.right_climber.to_dict() if result.right_climber else None,
            })

            pbar.update(1)

        pbar.close()

    cap.release()

    # ذخیره JSON
    output_data = {
        'metadata': {
            'video_path': str(video_path.name),
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': len(results),
            'frames_with_left_pose': frames_with_left,
            'frames_with_right_pose': frames_with_right,
            'frames_with_both_poses': frames_with_both,
            'detection_rate_left': frames_with_left / len(results) if results else 0,
            'detection_rate_right': frames_with_right / len(results) if results else 0,
            'detection_rate_both': frames_with_both / len(results) if results else 0,
        },
        'frames': results
    }

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    return output_data['metadata']


def main():
    """پردازش race clips از race_segments."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract pose data from race segments')
    parser.add_argument('--max-races', type=int, default=None,
                       help='Maximum number of races to process (default: all)')
    parser.add_argument('--competition', type=str, default=None,
                       help='Process only specific competition (e.g., seoul_2024)')
    args = parser.parse_args()

    # مسیرها
    race_segments = Path('data/race_segments')
    output_dir = Path('data/processed/poses')
    output_dir.mkdir(parents=True, exist_ok=True)

    # پیدا کردن ویدیوها از همه competitions یا یک competition خاص
    clips = []
    if args.competition:
        competition_dir = race_segments / args.competition
        if not competition_dir.exists():
            print(f"❌ Competition directory not found: {competition_dir}")
            print(f"\nAvailable competitions:")
            for comp_dir in sorted(race_segments.iterdir()):
                if comp_dir.is_dir():
                    print(f"  - {comp_dir.name}")
            return
        clips.extend(sorted(competition_dir.glob('*.mp4')))
        print(f"Processing competition: {args.competition}")
    else:
        # همه competitions
        for comp_dir in sorted(race_segments.iterdir()):
            if comp_dir.is_dir():
                clips.extend(sorted(comp_dir.glob('*.mp4')))
        print(f"Processing all competitions")

    if not clips:
        print(f"⚠️  No race segment MP4 files found in {race_segments}")
        print(f"\nExpected structure:")
        print(f"  data/race_segments/")
        print(f"    seoul_2024/")
        print(f"      Speed_finals_Seoul_2024_race001.mp4")
        print(f"      Speed_finals_Seoul_2024_race002.mp4")
        print(f"      ...")
        return

    # محدود کردن تعداد races
    if args.max_races and args.max_races > 0:
        clips = clips[:args.max_races]
        print(f"Limited to first {args.max_races} races")

    print(f"Found {len(clips)} race clip(s) to process\n")

    # پردازش
    stats_summary = []

    for clip in tqdm(clips, desc="Processing videos"):
        output_json = output_dir / f"{clip.stem}_poses.json"

        if output_json.exists():
            print(f"  ⏭️  Skipping {clip.name} (already processed)")
            continue

        try:
            print(f"\n📹 Processing: {clip.name}")
            stats = extract_poses_from_clip(clip, output_json)
            stats_summary.append({
                'video': clip.name,
                **stats
            })

            print(f"    ✓ {stats['total_frames']} frames processed")
            print(f"    ✓ Detection rate: Left={stats['detection_rate_left']:.1%}, "
                  f"Right={stats['detection_rate_right']:.1%}, "
                  f"Both={stats['detection_rate_both']:.1%}")
            print(f"    ✓ Saved to: {output_json.name}")

        except Exception as e:
            print(f"    ❌ Error processing {clip.name}: {str(e)}")
            continue

    # خلاصه نهایی
    if stats_summary:
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)

        total_frames = sum(s['total_frames'] for s in stats_summary)
        avg_detection_left = sum(s['detection_rate_left'] for s in stats_summary) / len(stats_summary)
        avg_detection_right = sum(s['detection_rate_right'] for s in stats_summary) / len(stats_summary)
        avg_detection_both = sum(s['detection_rate_both'] for s in stats_summary) / len(stats_summary)

        print(f"Videos processed: {len(stats_summary)}")
        print(f"Total frames: {total_frames:,}")
        print(f"Average detection rate:")
        print(f"  - Left lane:  {avg_detection_left:.1%}")
        print(f"  - Right lane: {avg_detection_right:.1%}")
        print(f"  - Both lanes: {avg_detection_both:.1%}")
        print(f"\nOutput directory: {output_dir.absolute()}")
        print("="*70)

        # ذخیره summary
        summary_file = output_dir / "_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'total_videos': len(stats_summary),
                'total_frames': total_frames,
                'avg_detection_rate_left': avg_detection_left,
                'avg_detection_rate_right': avg_detection_right,
                'avg_detection_rate_both': avg_detection_both,
                'videos': stats_summary
            }, f, indent=2)

        print(f"\n💾 Summary saved to: {summary_file.name}")
    else:
        print("\n⚠️  No videos were processed successfully.")


if __name__ == '__main__':
    main()
