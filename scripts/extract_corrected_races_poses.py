#!/usr/bin/env python3
"""
Extract poses for the 3 manually corrected races.
"""

from pathlib import Path
import sys
import cv2
import json
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'phase1_pose_estimation'))

from dual_lane_detector import DualLaneDetector

# The 3 corrected races
RACES = [
    {
        'race_id': 'Speed_finals_Chamonix_2024_race001',
        'video': 'data/race_segments/chamonix_2024/Speed_finals_Chamonix_2024_race001.mp4'
    },
    {
        'race_id': 'Speed_finals_Innsbruck_2024_race010',
        'video': 'data/race_segments/innsbruck_2024/Speed_finals_Innsbruck_2024_race010.mp4'
    },
    {
        'race_id': 'Speed_finals_Zilina_2025_race023',
        'video': 'data/race_segments/zilina_2025/Speed_finals_Zilina_2025_race023.mp4'
    }
]


def extract_poses(video_path: Path, output_json: Path):
    """Extract poses from a race segment."""
    print(f"\n📹 Processing: {video_path.name}")

    detector = DualLaneDetector(boundary_detection_method='fixed')
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    frames_with_left = 0
    frames_with_right = 0
    frame_count = 0

    with detector:
        pbar = tqdm(total=total_frames, desc=f"  Extracting poses", leave=False)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps if fps > 0 else 0
            result = detector.process_frame(frame, frame_count, timestamp)

            has_left = result.left_climber is not None
            has_right = result.right_climber is not None

            if has_left:
                frames_with_left += 1
            if has_right:
                frames_with_right += 1

            results.append({
                'frame_id': result.frame_id,
                'timestamp': result.timestamp,
                'left_climber': result.left_climber.to_dict() if result.left_climber else None,
                'right_climber': result.right_climber.to_dict() if result.right_climber else None
            })

            frame_count += 1
            pbar.update(1)

        pbar.close()

    cap.release()

    # Save results
    output_data = {
        'video_info': {
            'filename': video_path.name,
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': total_frames / fps if fps > 0 else 0
        },
        'detection_stats': {
            'frames_with_left': frames_with_left,
            'frames_with_right': frames_with_right,
            'detection_rate_left': frames_with_left / total_frames if total_frames > 0 else 0,
            'detection_rate_right': frames_with_right / total_frames if total_frames > 0 else 0
        },
        'frames': results
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  ✓ {total_frames} frames processed")
    print(f"  ✓ Detection rate: Left={frames_with_left/total_frames:.1%}, "
          f"Right={frames_with_right/total_frames:.1%}")
    print(f"  ✓ Saved to: {output_json}")


def main():
    print("=" * 80)
    print("EXTRACTING POSES FOR CORRECTED RACES")
    print("=" * 80)

    output_dir = Path('data/processed/poses')
    output_dir.mkdir(parents=True, exist_ok=True)

    for race in RACES:
        race_id = race['race_id']
        video_path = Path(race['video'])
        output_json = output_dir / f"{race_id}_poses.json"

        if not video_path.exists():
            print(f"\n❌ Video not found: {video_path}")
            continue

        if output_json.exists():
            print(f"\n⏭️  Skipping {race_id} (already exists)")
            continue

        try:
            extract_poses(video_path, output_json)
        except Exception as e:
            print(f"\n❌ Error processing {race_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("✓ Pose extraction complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
