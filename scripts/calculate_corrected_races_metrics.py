#!/usr/bin/env python3
"""
Calculate metrics for the 3 manually corrected races.
"""

from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.performance_metrics import PerformanceMetricsCalculator

# The 3 corrected races
RACES = [
    {
        'race_id': 'Speed_finals_Chamonix_2024_race001',
        'competition': 'chamonix_2024',
        'poses': 'data/processed/poses/Speed_finals_Chamonix_2024_race001_poses.json',
        'metadata': 'data/race_segments/chamonix_2024/Speed_finals_Chamonix_2024_race001_metadata.json'
    },
    {
        'race_id': 'Speed_finals_Innsbruck_2024_race010',
        'competition': 'innsbruck_2024',
        'poses': 'data/processed/poses/Speed_finals_Innsbruck_2024_race010_poses.json',
        'metadata': 'data/race_segments/innsbruck_2024/Speed_finals_Innsbruck_2024_race010_metadata.json'
    },
    {
        'race_id': 'Speed_finals_Zilina_2025_race023',
        'competition': 'zilina_2025',
        'poses': 'data/processed/poses/Speed_finals_Zilina_2025_race023_poses.json',
        'metadata': 'data/race_segments/zilina_2025/Speed_finals_Zilina_2025_race023_metadata.json'
    }
]


def calculate_metrics(race_id: str, poses_path: Path, metadata_path: Path, output_dir: Path):
    """Calculate metrics for a single race."""
    print(f"\n📊 Calculating metrics for {race_id}...")

    # Load poses
    with open(poses_path) as f:
        poses_data = json.load(f)

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Get race boundaries from metadata
    start_frame = metadata['detected_start_frame']
    finish_frame = metadata['detected_finish_frame']
    fps = poses_data['video_info']['fps']

    # Calculate which frames in the segment correspond to the race
    # The segment starts at: detected_start_frame - buffer_before
    buffer_before = metadata.get('buffer_before', 1.5)
    buffer_before_frames = int(buffer_before * fps)
    segment_start_frame = start_frame - buffer_before_frames

    # Convert original video frame numbers to segment frame indices
    race_start_idx = start_frame - segment_start_frame
    race_finish_idx = finish_frame - segment_start_frame

    print(f"  Race boundaries in segment: frames {race_start_idx} to {race_finish_idx}")
    print(f"  Duration: {(race_finish_idx - race_start_idx) / fps:.2f}s")

    # Calculate metrics for each lane
    for lane in ['left', 'right']:
        output_file = output_dir / f"{race_id}_metrics_{lane}.json"

        try:
            # Extract pose data for this lane
            lane_poses = []
            for frame_data in poses_data['frames']:
                frame_idx = frame_data['frame_id']

                # Only include frames within race boundaries
                if race_start_idx <= frame_idx <= race_finish_idx:
                    climber_data = frame_data.get(f'{lane}_climber')
                    if climber_data:
                        lane_poses.append({
                            'frame_id': frame_idx,
                            'timestamp': frame_data['timestamp'],
                            'pose': climber_data
                        })

            if not lane_poses:
                print(f"  ⚠️  No {lane} lane data found in race boundaries")
                continue

            # Calculate basic metrics
            metrics = {
                'race_id': race_id,
                'lane': lane,
                'race_duration_seconds': (race_finish_idx - race_start_idx) / fps,
                'total_frames': race_finish_idx - race_start_idx + 1,
                'frames_with_detection': len(lane_poses),
                'detection_rate': len(lane_poses) / (race_finish_idx - race_start_idx + 1),
                'start_frame_idx': race_start_idx,
                'finish_frame_idx': race_finish_idx,
                'fps': fps
            }

            # Save metrics
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ {lane.capitalize()} lane: {len(lane_poses)} frames, {metrics['detection_rate']:.1%} detection")
            print(f"    Saved to: {output_file.name}")

        except Exception as e:
            print(f"  ❌ Error calculating {lane} lane metrics: {e}")
            continue


def main():
    print("=" * 80)
    print("CALCULATING METRICS FOR CORRECTED RACES")
    print("=" * 80)

    output_dir = Path('data/processed/metrics')
    output_dir.mkdir(parents=True, exist_ok=True)

    for race in RACES:
        race_id = race['race_id']
        poses_path = Path(race['poses'])
        metadata_path = Path(race['metadata'])

        if not poses_path.exists():
            print(f"\n❌ Poses file not found: {poses_path}")
            continue

        if not metadata_path.exists():
            print(f"\n❌ Metadata file not found: {metadata_path}")
            continue

        try:
            calculate_metrics(race_id, poses_path, metadata_path, output_dir)
        except Exception as e:
            print(f"\n❌ Error processing {race_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("✓ Metrics calculation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
