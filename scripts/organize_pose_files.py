"""
Organize pose files into competition subdirectories

This script moves pose files from flat structure to organized subdirectories:
- Before: data/processed/poses/Speed_finals_Seoul_2024_race001_poses.json
- After:  data/processed/poses/seoul_2024/Speed_finals_Seoul_2024_race001_poses.json
"""

import shutil
from pathlib import Path
import re

def organize_pose_files():
    """Organize pose files into competition subdirectories."""
    poses_dir = Path("data/processed/poses")

    # Find all pose JSON files (not in subdirectories)
    pose_files = list(poses_dir.glob("Speed_finals_*_poses.json"))

    print(f"Found {len(pose_files)} pose files to organize")

    moved = 0
    for pose_file in pose_files:
        # Extract competition from filename
        # Format: Speed_finals_Chamonix_2024_race001_poses.json
        match = re.search(r'Speed_finals_([A-Za-z]+_\d{4})_race', pose_file.name)

        if match:
            competition_raw = match.group(1)  # e.g., "Chamonix_2024"
            competition = competition_raw.lower()  # e.g., "chamonix_2024"

            # Create subdirectory
            subdir = poses_dir / competition
            subdir.mkdir(exist_ok=True)

            # Move file
            dest = subdir / pose_file.name
            shutil.move(str(pose_file), str(dest))
            moved += 1

            if moved % 20 == 0:
                print(f"  Moved {moved} files...")
        else:
            print(f"  Warning: Could not parse competition from {pose_file.name}")

    print(f"\nDone! Moved {moved} files into subdirectories")

    # Show summary
    print("\nCompetition summary:")
    for comp_dir in sorted(poses_dir.glob("*_*")):
        if comp_dir.is_dir():
            count = len(list(comp_dir.glob("*_poses.json")))
            print(f"  {comp_dir.name}: {count} files")

if __name__ == "__main__":
    organize_pose_files()
