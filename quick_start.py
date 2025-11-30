#!/usr/bin/env python3
"""
Quick Start Script for Speed Climbing Analysis
==============================================

This script provides a quick way to:
1. Check dependencies
2. Find videos
3. Run pose estimation
4. Generate basic feedback

Usage:
    python quick_start.py
    python quick_start.py --video path/to/video.mp4
    python quick_start.py --demo  # Create demo output without video

این اسکریپت به صورت خودکار:
1. وابستگی‌ها را بررسی می‌کند
2. ویدئوها را پیدا می‌کند
3. Pose estimation اجرا می‌کند
4. بازخورد اولیه تولید می‌کند
"""

import argparse
import json
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent


def check_dependencies():
    """Check if required packages are installed."""
    print("\n1. Checking dependencies...")

    errors = []

    try:
        import cv2
        print(f"   ✓ OpenCV {cv2.__version__}")
    except ImportError:
        errors.append("   ✗ OpenCV not installed → pip install opencv-python")

    try:
        import mediapipe as mp
        print(f"   ✓ MediaPipe {mp.__version__}")
    except ImportError:
        errors.append("   ✗ MediaPipe not installed → pip install mediapipe")

    try:
        import numpy as np
        print(f"   ✓ NumPy {np.__version__}")
    except ImportError:
        errors.append("   ✗ NumPy not installed → pip install numpy")

    try:
        import streamlit
        print(f"   ✓ Streamlit {streamlit.__version__}")
    except ImportError:
        errors.append("   ✗ Streamlit not installed → pip install streamlit")

    if errors:
        print("\n   Missing dependencies:")
        for err in errors:
            print(err)
        print("\n   Run: pip install -r requirements_core.txt")
        return False

    return True


def find_videos(video_dir: Path):
    """Find video files in directory."""
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    videos = []
    for ext in extensions:
        videos.extend(video_dir.glob(ext))
    return sorted(videos)


def create_demo_output():
    """Create demo output without processing a video."""
    print("\n   Creating demo output...")

    demo_feedback = {
        "analysis_info": {
            "version": "1.0.0",
            "language": "en",
            "source": "demo",
            "note": "This is a demo output - no video was processed"
        },
        "performance_scores": {
            "coordination": {"score": 75.0, "rating": "good"},
            "leg_technique": {"score": 72.0, "rating": "good"},
            "arm_technique": {"score": 70.0, "rating": "good"},
            "body_position": {"score": 68.0, "rating": "average"},
            "reach": {"score": 65.0, "rating": "average"}
        },
        "overall_score": 70.0,
        "overall_rating": "good",
        "recommendations": [
            {
                "priority": "high",
                "category": "body_position",
                "recommendation_en": "Keep your body closer to the wall during transitions",
                "recommendation_fa": "بدن خود را در حین انتقال نزدیک‌تر به دیوار نگه دارید"
            },
            {
                "priority": "medium",
                "category": "reach",
                "recommendation_en": "Extend your reach by improving shoulder flexibility",
                "recommendation_fa": "با بهبود انعطاف‌پذیری شانه، دسترسی خود را افزایش دهید"
            }
        ]
    }

    output_dir = PROJECT_ROOT / "data" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "demo_feedback.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(demo_feedback, f, indent=2, ensure_ascii=False)

    print(f"   ✓ Demo output saved to: {output_file}")
    return output_file


def process_video(video_path: Path, language: str = "en"):
    """Process a video and generate feedback."""
    print(f"\n   Processing: {video_path.name}")
    print("   This may take several minutes...")

    try:
        # Import speed_climbing modules
        from speed_climbing.vision.pose import BlazePoseExtractor
        from speed_climbing.analysis.feedback.feedback_generator import FeedbackGenerator

        # Extract poses
        print("   → Extracting poses...")
        extractor = BlazePoseExtractor()
        pose_data = extractor.process_video(str(video_path))

        print(f"   ✓ Processed {len(pose_data.get('frames', []))} frames")

        # Generate feedback
        print("   → Generating feedback...")
        generator = FeedbackGenerator(language=language)
        feedback = generator.generate_feedback(
            pose_data=pose_data,
            lane="left",
            include_comparison=True
        )

        # Save output
        output_dir = PROJECT_ROOT / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"feedback_{video_path.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)

        print(f"   ✓ Feedback saved to: {output_file}")
        return feedback

    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        print("   Make sure speed_climbing package is properly installed")
        return None
    except Exception as e:
        print(f"   ✗ Processing error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Speed Climbing Performance Analysis - Quick Start"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Path to video file to analyze"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create demo output without processing a video"
    )
    parser.add_argument(
        "--language", "-l",
        choices=["en", "fa"],
        default="en",
        help="Output language (en=English, fa=Persian)"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch web interface instead of CLI"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Speed Climbing Performance Analysis - Quick Start")
    print("تحلیل عملکرد صعود سرعتی - شروع سریع")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Demo mode
    if args.demo:
        print("\n2. Running in demo mode...")
        create_demo_output()
        print("\n" + "=" * 60)
        print("Demo complete! Check data/samples/demo_feedback.json")
        print("=" * 60)
        sys.exit(0)

    # Web mode
    if args.web:
        print("\n2. Launching web interface...")
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "scripts" / "analysis_app" / "app.py")
        ])
        sys.exit(0)

    # Video processing mode
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"\n   ✗ Video not found: {video_path}")
            sys.exit(1)
    else:
        # Search for videos
        print("\n2. Searching for videos...")
        video_dir = PROJECT_ROOT / "data" / "raw_videos"

        if not video_dir.exists():
            video_dir.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {video_dir}")

        videos = find_videos(video_dir)

        if not videos:
            print(f"   ✗ No videos found in: {video_dir}")
            print("\n   Options:")
            print("   1. Add videos to data/raw_videos/")
            print("   2. Use --video path/to/video.mp4")
            print("   3. Use --demo for demo output")
            print("   4. Use --web for web interface")
            sys.exit(1)

        print(f"   ✓ Found {len(videos)} video(s):")
        for i, v in enumerate(videos):
            print(f"      [{i}] {v.name}")

        if len(videos) == 1:
            video_path = videos[0]
        else:
            try:
                choice = int(input(f"\n   Select video (0-{len(videos)-1}): "))
                video_path = videos[choice]
            except (ValueError, IndexError):
                video_path = videos[0]

    print(f"\n3. Processing video: {video_path.name}")
    feedback = process_video(video_path, args.language)

    if feedback:
        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"Overall score: {feedback.get('overall_score', 'N/A')}")
        print(f"Rating: {feedback.get('overall_rating', 'N/A')}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Analysis failed. Try:")
        print("  --demo    : Create demo output")
        print("  --web     : Use web interface")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
