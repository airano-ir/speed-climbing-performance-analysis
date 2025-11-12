"""
Quick Start Script for Speed Climbing Analysis
==============================================

این اسکریپت به صورت خودکار:
1. ویدئوها را پیدا می‌کند
2. Pose estimation اجرا می‌کند
3. نتایج را ذخیره می‌کند

Usage:
    python quick_start.py
"""

import sys
from pathlib import Path

print("=" * 60)
print("Speed Climbing Performance Analysis - Quick Start")
print("=" * 60)

# Check dependencies
print("\n1. Checking dependencies...")
try:
    import cv2
    print(f"   ✓ OpenCV {cv2.__version__}")
except ImportError:
    print("   ✗ OpenCV not installed")
    print("   → Run: pip install opencv-python")
    sys.exit(1)

try:
    import mediapipe as mp
    print(f"   ✓ MediaPipe installed")
except ImportError:
    print("   ✗ MediaPipe not installed")
    print("   → Run: pip install mediapipe")
    sys.exit(1)

try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
except ImportError:
    print("   ✗ NumPy not installed")
    print("   → Run: pip install numpy")
    sys.exit(1)

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

print("\n2. Checking project structure...")
required_dirs = [
    'data/raw_videos',
    'data/processed',
    'src/phase1_pose_estimation',
    'src/phase2_features'
]

for dir_path in required_dirs:
    full_path = PROJECT_ROOT / dir_path
    if full_path.exists():
        print(f"   ✓ {dir_path}")
    else:
        print(f"   ✗ {dir_path} - creating...")
        full_path.mkdir(parents=True, exist_ok=True)

# Find videos
print("\n3. Searching for videos...")
video_dir = PROJECT_ROOT / 'data' / 'raw_videos'
video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi')) + list(video_dir.glob('*.mov'))

if not video_files:
    print("   ✗ No videos found!")
    print(f"\n   Please add video files to: {video_dir}")
    print("   Supported formats: .mp4, .avi, .mov")
    sys.exit(1)

print(f"   ✓ Found {len(video_files)} video(s):")
for i, video in enumerate(video_files):
    print(f"      [{i}] {video.name}")

# Ask user to select video
if len(video_files) == 1:
    selected_video = video_files[0]
    print(f"\n   → Auto-selected: {selected_video.name}")
else:
    print(f"\n   Which video to process? (0-{len(video_files)-1})")
    try:
        choice = int(input("   Enter number: "))
        if 0 <= choice < len(video_files):
            selected_video = video_files[choice]
        else:
            print("   Invalid choice, using first video")
            selected_video = video_files[0]
    except:
        print("   Invalid input, using first video")
        selected_video = video_files[0]

# Process video
print(f"\n4. Processing video: {selected_video.name}")
print("   This may take several minutes...")

try:
    from phase1_pose_estimation.blazepose_extractor import extract_keypoints_from_video

    output_json = PROJECT_ROOT / 'data' / 'processed' / f"keypoints_{selected_video.stem}.json"
    output_video = PROJECT_ROOT / 'data' / 'processed' / f"annotated_{selected_video.name}"

    print(f"   Output JSON: {output_json.name}")
    print(f"   Output video: {output_video.name}")

    results = extract_keypoints_from_video(
        video_path=str(selected_video),
        output_path=str(output_json),
        save_format='json',
        visualize=True,
        output_video_path=str(output_video)
    )

    print(f"\n✓ SUCCESS!")
    print(f"   Processed {len(results)} frames")
    print(f"   Detection rate: {sum(1 for r in results if r.has_detection) / len(results) * 100:.1f}%")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nFor detailed debugging, use the Jupyter notebook:")
    print("  jupyter notebook notebooks/01_phase1_pose_estimation.ipynb")
    sys.exit(1)

# Calculate basic metrics
print("\n5. Calculating basic metrics...")
try:
    # Extract COM trajectory
    com_data = []
    for result in results:
        if result.has_detection and 'COM' in result.keypoints:
            com = result.keypoints['COM']
            com_data.append([com.x, com.y])

    if len(com_data) > 0:
        com_trajectory = np.array(com_data)

        # Path length
        path_length = sum(np.linalg.norm(com_trajectory[i] - com_trajectory[i-1])
                         for i in range(1, len(com_trajectory)))

        # Straight distance
        straight_distance = np.linalg.norm(com_trajectory[-1] - com_trajectory[0])

        # Efficiency
        efficiency = straight_distance / path_length if path_length > 0 else 0

        print(f"   Path length: {path_length:.4f} (normalized)")
        print(f"   Straight distance: {straight_distance:.4f} (normalized)")
        print(f"   Path efficiency: {efficiency:.4f}")
        print("\n   Note: For real-world metrics (meters), camera calibration is needed")
    else:
        print("   No COM data available")

except Exception as e:
    print(f"   Warning: Could not calculate metrics: {e}")

# Next steps
print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. View annotated video:")
print(f"   → {output_video}")
print("\n2. View keypoints JSON:")
print(f"   → {output_json}")
print("\n3. Run advanced analysis:")
print("   → jupyter notebook notebooks/01_phase1_pose_estimation.ipynb")
print("\n4. Calculate path entropy:")
print("   → python -m src.phase2_features.path_entropy")
print("=" * 60)
