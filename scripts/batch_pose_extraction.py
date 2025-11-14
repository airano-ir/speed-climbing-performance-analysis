#!/usr/bin/env python3
"""
Batch Pose Extraction Script
=============================
استخراج BlazePose keypoints از race segments

Usage:
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
    """پردازش تمام race clips موجود."""
    # مسیرها
    raw_videos = Path('data/raw_videos')
    output_dir = Path('data/processed/poses')
    output_dir.mkdir(parents=True, exist_ok=True)

    # پیدا کردن ویدیوها
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']
    clips = []
    for ext in video_extensions:
        clips.extend(raw_videos.glob(ext))

    clips = sorted(clips)

    if not clips:
        print(f"⚠️  No video files found in {raw_videos}")
        print(f"\nPlease add video files to: {raw_videos.absolute()}")
        print(f"Supported formats: MP4, AVI, MOV")
        print(f"\nFor help finding videos, see: HOW_TO_FIND_VIDEOS.md")
        return

    print(f"Found {len(clips)} video clip(s) to process\n")

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
