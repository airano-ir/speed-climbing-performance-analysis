#!/usr/bin/env python3
"""
Speed Climbing Performance Analysis - Example Script
تحلیل عملکرد صعود سرعتی - اسکریپت نمونه

This example demonstrates how to analyze a single video and get feedback.
این نمونه نحوه تحلیل یک ویدیو و دریافت بازخورد را نشان می‌دهد.

Usage / استفاده:
    python examples/analyze_single_video.py path/to/video.mp4 --language fa

Requirements / پیش‌نیازها:
    - Video file (MP4, AVI, MOV)
    - OR pose JSON file from previous extraction
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speed_climbing.analysis.feedback.feedback_generator import FeedbackGenerator


def analyze_pose_file(pose_path: str, language: str = "en", lane: str = "left"):
    """
    Analyze a pose JSON file and generate feedback.

    Args:
        pose_path: Path to pose JSON file
        language: Output language ('en' or 'fa')
        lane: Which lane to analyze ('left' or 'right')

    Returns:
        Feedback report as string
    """
    # Load pose data
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)

    # Initialize feedback generator
    generator = FeedbackGenerator(language=language)

    # Generate feedback
    report = generator.generate_feedback(
        pose_data=pose_data,
        lane=lane,
        include_comparison=True
    )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze climbing video/pose file and generate feedback"
    )
    parser.add_argument(
        "input_path",
        help="Path to video file (.mp4) or pose JSON file (.json)"
    )
    parser.add_argument(
        "--language", "-l",
        choices=["en", "fa"],
        default="en",
        help="Output language (en=English, fa=Persian/فارسی)"
    )
    parser.add_argument(
        "--lane",
        choices=["left", "right"],
        default="left",
        help="Which lane to analyze"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: print to console)"
    )

    args = parser.parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Check file type
    if input_path.suffix.lower() == '.json':
        # Direct pose file analysis
        report = analyze_pose_file(
            str(input_path),
            language=args.language,
            lane=args.lane
        )
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video file - need to extract poses first
        print("Video file detected. Extracting poses...")
        print("This may take a few minutes depending on video length.")
        print()

        # Import pose extractor
        from speed_climbing.vision.pose import BlazePoseExtractor
        from speed_climbing.analysis.features.extractor import FeatureExtractor

        # Extract poses
        extractor = BlazePoseExtractor()
        pose_data = extractor.process_video(str(input_path))

        # Generate feedback
        generator = FeedbackGenerator(language=args.language)
        report = generator.generate_feedback(
            pose_data=pose_data,
            lane=args.lane,
            include_comparison=True
        )
    else:
        print(f"Error: Unsupported file type: {input_path.suffix}")
        sys.exit(1)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding='utf-8')
        print(f"Report saved to: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
