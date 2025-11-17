"""
Generate annotated videos with pose overlay for top performers.

This script creates MP4 videos with:
- MediaPipe BlazePose skeleton overlay
- Center of Mass (COM) trajectory trail
- Performance metrics display
- Wall position indicator

Reference: notebooks/01_phase1_pose_estimation.ipynb

Output: MP4 files in data/processed/videos/annotated/

Author: Claude Code
Date: 2025-11-17
"""

import cv2
import numpy as np
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import mediapipe as mp
from tqdm import tqdm

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'aggregated_metrics_reliable.csv'
POSES_DIR = BASE_DIR / 'data' / 'processed' / 'poses'
METRICS_DIR = BASE_DIR / 'data' / 'processed' / 'metrics'
CALIBRATION_DIR = BASE_DIR / 'data' / 'processed' / 'calibration'
SEGMENTS_DIR = BASE_DIR / 'data' / 'race_segments'
OUTPUT_DIR = BASE_DIR / 'data' / 'processed' / 'videos' / 'annotated'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors (BGR format for OpenCV)
COLORS = {
    'skeleton': (0, 255, 0),      # Green
    'com': (0, 0, 255),            # Red
    'trail': (255, 128, 0),        # Blue
    'text_bg': (0, 0, 0),          # Black
    'text': (255, 255, 255),       # White
    'wall': (200, 200, 200),       # Light gray
    'velocity_low': (0, 255, 0),   # Green
    'velocity_high': (0, 0, 255)   # Red
}


def load_race_data(race_id):
    """Load pose, metrics, and calibration data for a race."""
    # Find competition directory
    comp_dirs = list(SEGMENTS_DIR.glob('*'))
    video_path = None

    for comp_dir in comp_dirs:
        candidate = comp_dir / f"{race_id}.mp4"
        if candidate.exists():
            video_path = candidate
            competition = comp_dir.name
            break

    if video_path is None:
        raise FileNotFoundError(f"Video not found for race: {race_id}")

    # Load pose data
    pose_file = POSES_DIR / f"{race_id}_pose.json"
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)

    # Load metrics
    metrics_file = METRICS_DIR / f"{race_id}_metrics.json"
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)

    # Load calibration
    calibration_file = CALIBRATION_DIR / f"{race_id}_calibration.json"
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    return video_path, pose_data, metrics_data, calibration_data


def calculate_com(landmarks, frame_width, frame_height):
    """Calculate center of mass from visible landmarks."""
    if not landmarks:
        return None, None

    x_sum, y_sum, count = 0, 0, 0
    for lm in landmarks:
        if lm.get('visibility', 0) > 0.5:
            x_sum += lm['x']
            y_sum += lm['y']
            count += 1

    if count == 0:
        return None, None

    # Convert from normalized to pixel coordinates
    com_x = int((x_sum / count) * frame_width)
    com_y = int((y_sum / count) * frame_height)

    return com_x, com_y


def calculate_velocity_color(velocity, min_vel=0.5, max_vel=2.0):
    """Calculate color based on velocity (gradient from green to red)."""
    # Normalize velocity to 0-1 range
    norm_vel = np.clip((velocity - min_vel) / (max_vel - min_vel), 0, 1)

    # Interpolate between green and red
    b = 0
    g = int(255 * (1 - norm_vel))
    r = int(255 * norm_vel)

    return (b, g, r)


def draw_metrics_overlay(frame, metrics, current_frame, total_frames, velocity=None):
    """Draw metrics overlay on frame."""
    height, width = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (width - 10, 120), COLORS['text_bg'], -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Race ID
    race_id = metrics.get('race_id', 'Unknown').replace('Speed_finals_', '')
    cv2.putText(frame, race_id, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)

    # Metrics
    y_offset = 60
    metrics_text = [
        f"Avg Velocity: {metrics.get('average_velocity_ms', 0):.3f} m/s",
        f"Max Velocity: {metrics.get('max_velocity_ms', 0):.3f} m/s",
        f"Total Time: {metrics.get('total_time_s', 0):.2f} s"
    ]

    if velocity is not None:
        metrics_text.append(f"Current Velocity: {velocity:.3f} m/s")

    for text in metrics_text:
        cv2.putText(frame, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        y_offset += 20

    # Progress bar
    progress = current_frame / total_frames
    bar_width = width - 40
    bar_height = 10
    bar_y = height - 30

    # Background
    cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + bar_height),
                  (100, 100, 100), -1)

    # Progress
    cv2.rectangle(frame, (20, bar_y),
                  (20 + int(bar_width * progress), bar_y + bar_height),
                  (0, 255, 0), -1)

    # Frame counter
    cv2.putText(frame, f"Frame: {current_frame}/{total_frames}",
                (width - 150, bar_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)


def draw_wall_indicator(frame, com_y, frame_height, vertical_displacement_m):
    """Draw wall position indicator on the side."""
    width = frame.shape[1]

    # Wall indicator dimensions
    wall_x = width - 40
    wall_top = 50
    wall_bottom = frame_height - 150
    wall_height = wall_bottom - wall_top

    # Draw wall outline
    cv2.rectangle(frame, (wall_x - 5, wall_top), (wall_x + 5, wall_bottom),
                  COLORS['wall'], 2)

    # Draw height markers
    for i in range(6):
        y = wall_top + int(wall_height * i / 5)
        height_m = 15.0 - (i * 3)
        cv2.line(frame, (wall_x - 8, y), (wall_x + 8, y), COLORS['wall'], 1)
        cv2.putText(frame, f"{height_m:.0f}m", (wall_x - 35, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS['text'], 1)

    # Draw current position
    if com_y is not None:
        # Calculate position on wall indicator (inverted because image y is top-to-bottom)
        relative_pos = com_y / frame_height
        indicator_y = wall_top + int(wall_height * relative_pos)

        cv2.circle(frame, (wall_x, indicator_y), 6, COLORS['com'], -1)
        cv2.circle(frame, (wall_x, indicator_y), 8, COLORS['text'], 2)


def generate_annotated_video(race_id, rank=None):
    """Generate annotated video for a race."""
    print(f"\nProcessing: {race_id} (Rank #{rank})")

    try:
        # Load data
        video_path, pose_data, metrics_data, calibration_data = load_race_data(race_id)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  Video: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")

        # Setup output video
        output_file = OUTPUT_DIR / f"{race_id}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_file), fourcc, fps,
                              (frame_width, frame_height))

        # Get pose frames lookup
        pose_frames = {frame['frame_number']: frame for frame in pose_data.get('frames', [])}

        # COM trail
        com_trail = []
        max_trail_length = 30  # Keep last 30 positions

        # Process frames
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc=f"  Rendering", unit="frame")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get pose data for this frame
            pose_frame = pose_frames.get(frame_idx)

            com_x, com_y = None, None
            velocity = None

            if pose_frame:
                landmarks = pose_frame.get('landmarks', [])

                # Calculate COM
                com_x, com_y = calculate_com(landmarks, frame_width, frame_height)

                if com_x is not None and com_y is not None:
                    com_trail.append((com_x, com_y))
                    if len(com_trail) > max_trail_length:
                        com_trail.pop(0)

                # Draw pose skeleton
                if landmarks:
                    # Convert landmarks to MediaPipe format for drawing
                    # Note: This is simplified - you may need to reconstruct landmark_list properly
                    for i, lm in enumerate(landmarks):
                        if lm.get('visibility', 0) > 0.5:
                            x = int(lm['x'] * frame_width)
                            y = int(lm['y'] * frame_height)
                            cv2.circle(frame, (x, y), 3, COLORS['skeleton'], -1)

                # Draw connections (simplified - connect major body parts)
                connections = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Shoulders and arms
                    (11, 23), (12, 24), (23, 24),  # Torso
                    (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
                ]

                for conn in connections:
                    if conn[0] < len(landmarks) and conn[1] < len(landmarks):
                        lm1 = landmarks[conn[0]]
                        lm2 = landmarks[conn[1]]
                        if lm1.get('visibility', 0) > 0.5 and lm2.get('visibility', 0) > 0.5:
                            x1 = int(lm1['x'] * frame_width)
                            y1 = int(lm1['y'] * frame_height)
                            x2 = int(lm2['x'] * frame_width)
                            y2 = int(lm2['y'] * frame_height)
                            cv2.line(frame, (x1, y1), (x2, y2), COLORS['skeleton'], 2)

            # Draw COM trail
            if len(com_trail) > 1:
                for i in range(1, len(com_trail)):
                    # Gradient trail (older = more transparent)
                    alpha = i / len(com_trail)
                    pt1 = com_trail[i - 1]
                    pt2 = com_trail[i]

                    # Calculate velocity color if available
                    color = COLORS['trail']
                    if metrics_data and 'average_velocity_ms' in metrics_data:
                        velocity = metrics_data['average_velocity_ms']
                        color = calculate_velocity_color(velocity)

                    cv2.line(frame, pt1, pt2, color, int(3 * alpha) + 1)

            # Draw COM current position
            if com_x is not None and com_y is not None:
                cv2.circle(frame, (com_x, com_y), 8, COLORS['com'], -1)
                cv2.circle(frame, (com_x, com_y), 10, COLORS['text'], 2)

            # Draw metrics overlay
            draw_metrics_overlay(frame, metrics_data, frame_idx, total_frames, velocity)

            # Draw wall indicator
            vertical_displacement_m = metrics_data.get('vertical_displacement_m', 15.0)
            draw_wall_indicator(frame, com_y, frame_height, vertical_displacement_m)

            # Write frame
            out.write(frame)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        out.release()

        print(f"  ✅ Saved: {output_file}")
        print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        return True

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def main():
    """Main execution."""
    print("=" * 60)
    print("ANNOTATED VIDEO GENERATOR")
    print("Speed Climbing Performance Analysis")
    print("=" * 60)
    print()

    # Load data
    print("Loading race data...")
    df = pd.read_csv(DATA_FILE)

    # Get top N performers
    N = 5
    print(f"\nSelecting top {N} performers by average velocity...")
    top_races = df.nlargest(N, 'average_velocity_ms')

    print("\nTop performers:")
    for idx, (_, row) in enumerate(top_races.iterrows(), 1):
        print(f"  {idx}. {row['race_id']}: {row['average_velocity_ms']:.3f} m/s "
              f"({row['total_time_s']:.2f}s)")

    print(f"\nGenerating annotated videos for top {N} performers...")
    print("=" * 60)

    # Generate videos
    success_count = 0
    for idx, (_, row) in enumerate(top_races.iterrows(), 1):
        if generate_annotated_video(row['race_id'], rank=idx):
            success_count += 1

    # Summary
    print()
    print("=" * 60)
    print("✅ VIDEO GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nSuccessfully generated: {success_count}/{N} videos")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generate report
    report = {
        'generation_date': datetime.now().isoformat(),
        'total_videos': success_count,
        'top_n': N,
        'races_processed': [row['race_id'] for _, row in top_races.iterrows()],
        'output_directory': str(OUTPUT_DIR)
    }

    report_file = BASE_DIR / 'data' / 'processed' / 'reports' / 'visualizations' / 'video_generation_report.json'
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved: {report_file}\n")


if __name__ == '__main__':
    main()
