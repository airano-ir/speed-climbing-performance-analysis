#!/usr/bin/env python3
"""
Quick test of pose extraction on available video clips.
Tests dual-lane detector and saves pose data.
"""

import sys
import cv2
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'phase1_pose_estimation'))

from dual_lane_detector import DualLaneDetector, visualize_dual_lane

def test_video(video_path: str, max_frames: int = 300):
    """Test pose extraction on a video clip."""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(video_path).name}")
    print(f"{'='*60}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Failed to open video: {video_path}")
        return False

    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ Video info: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")

    # Initialize detector
    print(f"✓ Initializing dual-lane detector...")
    detector = DualLaneDetector(
        boundary_detection_method='fixed',  # Simple fixed boundary for testing
        enable_lane_smoothing=False  # Disable Kalman (not available)
    )

    poses_data = []
    frame_count = 0

    try:
        with detector:
            print(f"✓ Processing frames (max {max_frames})...")

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                result = detector.process_frame(frame)

                # Save pose data
                poses_data.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0,
                    'left_pose': {
                        'detected': result.left_pose.pose_detected if result.left_pose else False,
                        'confidence': result.left_pose.confidence if result.left_pose else 0.0
                    },
                    'right_pose': {
                        'detected': result.right_pose.pose_detected if result.right_pose else False,
                        'confidence': result.right_pose.confidence if result.right_pose else 0.0
                    },
                    'boundary_x': result.boundary.x_center
                })

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"  Frame {frame_count}/{min(max_frames, total_frames)}")

            print(f"✓ Processed {frame_count} frames")

            # Get statistics
            stats = detector.get_statistics()
            print(f"\n{'='*60}")
            print("Detection Statistics:")
            print(f"  Left lane detection rate:  {stats['left_detection_rate']*100:.1f}%")
            print(f"  Right lane detection rate: {stats['right_detection_rate']*100:.1f}%")
            print(f"  Both lanes detected:       {stats['both_detected_rate']*100:.1f}%")
            print(f"{'='*60}\n")

    finally:
        cap.release()

    # Save pose data
    output_file = Path('test_pose_output.json')
    with open(output_file, 'w') as f:
        json.dump({
            'video': str(video_path),
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': frame_count
            },
            'statistics': stats,
            'poses': poses_data
        }, f, indent=2)

    print(f"✓ Pose data saved to: {output_file}")
    return True

if __name__ == '__main__':
    # Test with Aleksandra Mirosław video (dual-lane speed climbing)
    video_path = "data/raw_videos/Aleksandra Mirosław now holds the top TEN fastest climbs by a woman EVER.mp4"

    success = test_video(video_path, max_frames=300)

    if success:
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Test failed!")
        sys.exit(1)
