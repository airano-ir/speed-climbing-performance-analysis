"""
Helper script to calculate corrected frame numbers for problematic races.

Based on manual review timeline data from PROMPT_FOR_UI_FIX_RACE_DETECTION.md
"""

import json
from pathlib import Path
from typing import Dict, Tuple

def calculate_frame_from_timecode(timecode: str, fps: float = 30.0) -> int:
    """
    Convert timecode to frame number.
    Supports formats: MM:SS, HH:MM:SS, or SS.FF (seconds.frames)
    """
    parts = timecode.split(':')

    if len(parts) == 2:  # MM:SS
        minutes, seconds = int(parts[0]), int(parts[1])
        total_seconds = minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timecode format: {timecode}")

    return int(total_seconds * fps)


def fix_race001_chamonix() -> Dict:
    """
    Fix Race001 (Chamonix 2024)

    Current (WRONG): 1.77s (53 frames)
    Actual: ~6.5s (~195 frames)

    From manual review:
    - Segment starts at original frame: detected_start_frame - buffer_before
    - Race actually starts at ~77 seconds (01:17) into segment
    - RIGHT finishes at ~6.5s after start
    - LEFT falls during race (was incorrectly detected as finish)
    """
    metadata_path = Path("data/race_segments/chamonix_2024/Speed_finals_Chamonix_2024_race001_metadata.json")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Current values
    current_start = metadata['detected_start_frame']  # 11784
    buffer_before = metadata['buffer_before']  # 1.5s
    fps = 30.0
    buffer_before_frames = int(buffer_before * fps)  # 45 frames

    # Segment starts at this frame in original video
    segment_start_frame = current_start - buffer_before_frames  # 11739

    # According to manual review:
    # - Race starts at 01:17 (77 seconds) in segment = 2310 frames
    # - Duration is ~6.5s = 195 frames
    race_start_in_segment = 77 * fps  # 2310 frames into segment
    race_duration_frames = int(6.5 * fps)  # 195 frames

    # Calculate actual frames in original video
    actual_start_frame = segment_start_frame + race_start_in_segment
    actual_finish_frame = actual_start_frame + race_duration_frames

    # Store original values
    metadata['original_detected_start_frame'] = metadata['detected_start_frame']
    metadata['original_detected_finish_frame'] = metadata['detected_finish_frame']
    metadata['original_race_duration'] = metadata['race_duration']

    # Update with corrected values
    metadata['detected_start_frame'] = int(actual_start_frame)
    metadata['detected_finish_frame'] = int(actual_finish_frame)
    metadata['detected_start_time'] = actual_start_frame / fps
    metadata['detected_finish_time'] = actual_finish_frame / fps
    metadata['race_duration'] = race_duration_frames / fps
    metadata['manual_correction'] = True
    metadata['notes'] = (
        "MANUAL CORRECTION: Original detection failed (detected climber fall as finish). "
        "Corrected based on manual video review. Original: 1.77s (INVALID). "
        "Used RIGHT lane finish time. LEFT climber fell during race."
    )

    return metadata


def fix_race010_innsbruck() -> Dict:
    """
    Fix Race010 (Innsbruck 2024)

    Current (WRONG): 12.00s (360 frames) - includes pre-race warmup
    Actual: ~7.5s (~225 frames)

    From manual review:
    - Detected start is too early (includes warmup)
    - Actual duration should be ~7.5s
    """
    metadata_path = Path("data/race_segments/innsbruck_2024/Speed_finals_Innsbruck_2024_race010_metadata.json")

    with open(metadata_path) as f:
        metadata = json.load(f)

    fps = 30.0
    current_start = metadata['detected_start_frame']  # 49920
    current_finish = metadata['detected_finish_frame']  # 50280

    # Current duration: 12s = 360 frames (too long)
    # Actual duration: ~7.5s = 225 frames
    actual_duration_frames = int(7.5 * fps)  # 225 frames

    # The finish detection seems reasonable, start is too early
    # Move start forward to get correct duration
    actual_finish_frame = current_finish
    actual_start_frame = actual_finish_frame - actual_duration_frames

    # Store original values
    metadata['original_detected_start_frame'] = metadata['detected_start_frame']
    metadata['original_detected_finish_frame'] = metadata['detected_finish_frame']
    metadata['original_race_duration'] = metadata['race_duration']

    # Update with corrected values
    metadata['detected_start_frame'] = int(actual_start_frame)
    metadata['detected_finish_frame'] = int(actual_finish_frame)
    metadata['detected_start_time'] = actual_start_frame / fps
    metadata['detected_finish_time'] = actual_finish_frame / fps
    metadata['race_duration'] = actual_duration_frames / fps
    metadata['manual_correction'] = True
    metadata['notes'] = (
        "MANUAL CORRECTION: Original detection started too early (included warmup). "
        "Corrected based on manual video review. Original: 12.00s (included pre-race movement). "
        "End time extended +5s (early finish) | Late start (needs more buffer)"
    )

    return metadata


def fix_race023_zilina() -> Dict:
    """
    Fix Race023 (Zilina 2025)

    Current (WRONG): 19.00s (570 frames) - includes false start/replay
    Actual: ~6.6-7s (~198-210 frames)

    From manual review:
    - Detected segment includes false start or replay
    - Actual duration should be ~6.8s (middle of 6.6-7s range)
    """
    metadata_path = Path("data/race_segments/zilina_2025/Speed_finals_Zilina_2025_race023_metadata.json")

    with open(metadata_path) as f:
        metadata = json.load(f)

    fps = 30.0
    current_start = metadata['detected_start_frame']  # 84510
    current_finish = metadata['detected_finish_frame']  # 85080

    # Current duration: 19s = 570 frames (way too long)
    # Actual duration: ~6.8s = 204 frames (middle of 6.6-7s range)
    actual_duration_frames = int(6.8 * fps)  # 204 frames

    # The detection likely triggered on false start
    # Keep finish, move start forward
    actual_finish_frame = current_finish
    actual_start_frame = actual_finish_frame - actual_duration_frames

    # Store original values
    metadata['original_detected_start_frame'] = metadata['detected_start_frame']
    metadata['original_detected_finish_frame'] = metadata['detected_finish_frame']
    metadata['original_race_duration'] = metadata['race_duration']

    # Update with corrected values
    metadata['detected_start_frame'] = int(actual_start_frame)
    metadata['detected_finish_frame'] = int(actual_finish_frame)
    metadata['detected_start_time'] = actual_start_frame / fps
    metadata['detected_finish_time'] = actual_finish_frame / fps
    metadata['race_duration'] = actual_duration_frames / fps
    metadata['manual_correction'] = True
    metadata['notes'] = (
        "MANUAL CORRECTION: Original detection triggered on false start/replay. "
        "Corrected based on manual video review. Original: 19.00s (included false start). "
        "Used LEFT lane finish time. Start time -4s (video cut adjustment)"
    )

    return metadata


def main():
    """Fix all three problematic races."""
    print("=" * 80)
    print("FIXING PROBLEMATIC RACE METADATA")
    print("=" * 80)

    # Fix Race001
    print("\n1. Fixing Race001 (Chamonix 2024)...")
    race001 = fix_race001_chamonix()
    output_path = Path("data/race_segments/chamonix_2024/Speed_finals_Chamonix_2024_race001_metadata.json")
    with open(output_path, 'w') as f:
        json.dump(race001, f, indent=2)
    print(f"   ✓ Updated: {output_path}")
    print(f"   Duration: {race001['original_race_duration']:.2f}s → {race001['race_duration']:.2f}s")

    # Fix Race010
    print("\n2. Fixing Race010 (Innsbruck 2024)...")
    race010 = fix_race010_innsbruck()
    output_path = Path("data/race_segments/innsbruck_2024/Speed_finals_Innsbruck_2024_race010_metadata.json")
    with open(output_path, 'w') as f:
        json.dump(race010, f, indent=2)
    print(f"   ✓ Updated: {output_path}")
    print(f"   Duration: {race010['original_race_duration']:.2f}s → {race010['race_duration']:.2f}s")

    # Fix Race023
    print("\n3. Fixing Race023 (Zilina 2025)...")
    race023 = fix_race023_zilina()
    output_path = Path("data/race_segments/zilina_2025/Speed_finals_Zilina_2025_race023_metadata.json")
    with open(output_path, 'w') as f:
        json.dump(race023, f, indent=2)
    print(f"   ✓ Updated: {output_path}")
    print(f"   Duration: {race023['original_race_duration']:.2f}s → {race023['race_duration']:.2f}s")

    print("\n" + "=" * 80)
    print("✓ All 3 races corrected successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run validation script: python scripts/validate_race_durations.py")
    print("2. Re-extract poses for corrected races")
    print("3. Re-calculate metrics")


if __name__ == "__main__":
    main()
