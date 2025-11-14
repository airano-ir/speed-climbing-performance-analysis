#!/usr/bin/env python3
"""
Create Test Video for Batch Pose Extraction
============================================
ساخت یک ویدیو تست ساده برای آزمایش batch processing

Usage:
    python scripts/create_test_video.py
"""

import cv2
import numpy as np
from pathlib import Path


def create_synthetic_test_video(output_path: Path, duration_seconds: int = 3, fps: int = 30):
    """
    ساخت یک ویدیو تست ساده با دو climber سینتتیک.

    Args:
        output_path: مسیر خروجی
        duration_seconds: طول ویدیو به ثانیه
        fps: فریم در ثانیه
    """
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = duration_seconds * fps

    print(f"Creating test video: {output_path.name}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")

    for frame_num in range(total_frames):
        # پس‌زمینه
        frame = np.ones((height, width, 3), dtype=np.uint8) * 30

        # خط تقسیم (وسط)
        cv2.line(frame, (width // 2, 0), (width // 2, height),
                 (255, 255, 255), 2)

        # دو climber ساده (دایره‌های متحرک)
        progress = frame_num / total_frames  # 0 تا 1

        # Left climber (قرمز) - از پایین به بالا
        left_x = width // 4
        left_y = int(height * (1 - progress * 0.8))  # حرکت به بالا
        cv2.circle(frame, (left_x, left_y), 30, (0, 0, 255), -1)  # سر
        cv2.circle(frame, (left_x - 20, left_y + 40), 15, (0, 0, 200), -1)  # دست چپ
        cv2.circle(frame, (left_x + 20, left_y + 40), 15, (0, 0, 200), -1)  # دست راست
        cv2.circle(frame, (left_x, left_y + 80), 20, (0, 0, 180), -1)  # بدن
        cv2.circle(frame, (left_x - 15, left_y + 120), 12, (0, 0, 160), -1)  # پای چپ
        cv2.circle(frame, (left_x + 15, left_y + 120), 12, (0, 0, 160), -1)  # پای راست

        # Right climber (آبی) - کمی عقب‌تر
        right_x = width * 3 // 4
        right_y = int(height * (1 - progress * 0.7))  # حرکت کمی کندتر
        cv2.circle(frame, (right_x, right_y), 30, (255, 0, 0), -1)  # سر
        cv2.circle(frame, (right_x - 20, right_y + 40), 15, (200, 0, 0), -1)  # دست چپ
        cv2.circle(frame, (right_x + 20, right_y + 40), 15, (200, 0, 0), -1)  # دست راست
        cv2.circle(frame, (right_x, right_y + 80), 20, (180, 0, 0), -1)  # بدن
        cv2.circle(frame, (right_x - 15, right_y + 120), 12, (160, 0, 0), -1)  # پای چپ
        cv2.circle(frame, (right_x + 15, right_y + 120), 12, (160, 0, 0), -1)  # پای راست

        # متن
        cv2.putText(frame, "LEFT LANE", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "RIGHT LANE", (width - 250, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (width // 2 - 100, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✓ Test video created: {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """ساخت ویدیوهای تست."""
    output_dir = Path('data/raw_videos')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ساخت یک ویدیو تست کوتاه
    test_video = output_dir / 'test_dual_lane_climb.mp4'

    if test_video.exists():
        print(f"Test video already exists: {test_video}")
        print("Delete it first if you want to recreate.")
        return

    create_synthetic_test_video(test_video, duration_seconds=3, fps=30)

    print("\n✅ Done! You can now run:")
    print("    python scripts/batch_pose_extraction.py")


if __name__ == '__main__':
    main()
