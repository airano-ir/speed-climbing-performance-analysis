#!/usr/bin/env python3
"""
Complete video analysis pipeline with personalized feedback.

Usage:
    python scripts/analyze_video.py <pose_file.json> [--language fa|en] [--output report.txt]

Example:
    python scripts/analyze_video.py data/processed/poses/samples/race001.json --language fa
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speed_climbing.analysis.features import FeatureExtractor
from speed_climbing.analysis.feedback import FeedbackGenerator
from speed_climbing.analysis.feedback.feedback_generator import Language


def analyze_from_poses(
    pose_file: str,
    language: str = 'fa',
    output_path: str = None,
    lane: str = 'left'
) -> str:
    """
    Analyze from pose file and generate personalized feedback.

    Args:
        pose_file: Path to pose JSON file
        language: Output language ('fa' for Persian, 'en' for English)
        output_path: Optional output path
        lane: Lane to analyze ('left' or 'right')

    Returns:
        Formatted feedback report
    """
    pose_path = Path(pose_file)

    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    print(f"[1/3] Loading poses: {pose_path.name}")

    with open(pose_path, 'r') as f:
        pose_data = json.load(f)

    # Extract features
    fps = pose_data.get('fps', 30.0)
    extractor = FeatureExtractor(fps=fps)
    results = extractor.extract_from_file(str(pose_path))

    if not results:
        raise ValueError("Could not extract features")

    # Get result for specified lane
    result = None
    for r in results:
        if r.lane == lane:
            result = r
            break

    if result is None:
        result = results[0]

    print(f"[2/3] Extracted features for {result.lane} lane")

    # Generate feedback
    lang = Language.PERSIAN if language == 'fa' else Language.ENGLISH
    generator = FeedbackGenerator(language=lang)

    # Flatten features
    features = {}
    for k, v in result.frequency_features.items():
        features[f'freq_{k}'] = v
    for k, v in result.efficiency_features.items():
        features[f'eff_{k}'] = v
    for k, v in result.posture_features.items():
        features[f'post_{k}'] = v

    feedback = generator.generate(features)
    report = generator.format_report(feedback)

    print(f"[3/3] Generated feedback report")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Analyze speed climbing pose data and generate personalized feedback'
    )
    parser.add_argument('input', help='Path to pose JSON file')
    parser.add_argument('--language', '-l', choices=['fa', 'en'], default='fa',
                       help='Output language (fa=Persian, en=English)')
    parser.add_argument('--output', '-o', help='Save report to file')
    parser.add_argument('--lane', choices=['left', 'right'], default='left',
                       help='Lane to analyze')

    args = parser.parse_args()

    try:
        report = analyze_from_poses(
            args.input,
            language=args.language,
            output_path=args.output,
            lane=args.lane
        )

        print()
        print(report)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
