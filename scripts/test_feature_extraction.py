"""
Quick test for the feature extraction pipeline.

Shows detailed debug output including activity curves and race detection.
"""

import sys
import json
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from speed_climbing.analysis.features import FeatureExtractor, save_features_json, save_features_csv
from speed_climbing.analysis.features.race_detector import RaceSegmentDetector


def print_activity_curve(raw_activity: np.ndarray, smoothed: np.ndarray, start: int, end: int, fps: float = 30.0):
    """Print a text-based visualization of the activity curve."""
    n = len(smoothed)
    if n == 0:
        print("  No activity data available")
        return

    max_val = np.max(smoothed)
    if max_val < 1e-6:
        print("  No significant activity detected")
        return

    # Normalize for display
    normalized = smoothed / max_val

    # Print header
    print(f"\n  Activity Curve (frames 0-{n-1}, racing: {start}-{end}):")
    print(f"  Duration: {(end-start)/fps:.2f}s at {fps}fps")
    print()

    # Sample every few frames to fit in terminal
    step = max(1, n // 50)
    width = 40

    print("  Frame    Activity")
    print("  " + "-" * 55)

    for i in range(0, n, step):
        bar_len = int(normalized[i] * width)
        bar = "#" * bar_len

        # Mark start and end
        marker = ""
        if abs(i - start) < step:
            marker = " <-- START"
        elif abs(i - end) < step:
            marker = " <-- END"

        is_racing = start <= i <= end
        prefix = ">" if is_racing else " "

        print(f"  {prefix}{i:4d}  |{bar:<{width}}| {smoothed[i]:.4f}{marker}")

    print("  " + "-" * 55)
    print(f"  Legend: > = racing frame, # = activity level")


def main():
    # Find sample pose files
    samples_dir = project_root / 'data' / 'processed' / 'poses' / 'samples'
    chamonix_dir = project_root / 'data' / 'processed' / 'poses' / 'chamonix_2024'

    # Get a few test files
    test_files = []

    if samples_dir.exists():
        test_files.extend(list(samples_dir.glob('*_poses.json'))[:3])

    if chamonix_dir.exists() and len(test_files) < 3:
        test_files.extend(list(chamonix_dir.glob('*_poses.json'))[:3 - len(test_files)])

    if not test_files:
        print("No pose files found for testing!")
        return

    print(f"Testing feature extraction on {len(test_files)} files...")
    print("=" * 70)

    # Initialize extractor
    extractor = FeatureExtractor(fps=30.0, min_frames=30)

    all_results = []

    for pose_file in test_files:
        print(f"\n{'='*70}")
        print(f"Processing: {pose_file.name}")
        print("=" * 70)

        try:
            # Load frames for activity curve analysis
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)

            frames = pose_data.get('frames', [])
            fps = pose_data.get('metadata', {}).get('fps', 30.0)

            # Show activity curves for each lane
            detector = RaceSegmentDetector(fps=fps)

            for lane in ['left', 'right']:
                # Check if lane has data
                has_lane = any(f.get(f'{lane}_climber', {}).get('has_detection', False) for f in frames)
                if not has_lane:
                    continue

                print(f"\n--- {lane.upper()} LANE ---")

                # Get activity curves
                raw_activity, smoothed_activity = detector.get_activity_curve(frames, lane)

                # Detect segment
                segment = detector.detect(frames, lane)

                if segment:
                    print(f"  Detection Method: {segment.detection_method}")
                    print(f"  Start Frame: {segment.start_frame} ({segment.start_frame/fps:.2f}s)")
                    print(f"  End Frame: {segment.end_frame} ({segment.end_frame/fps:.2f}s)")
                    print(f"  Duration: {segment.duration_frames} frames ({segment.duration_frames/fps:.2f}s)")
                    print(f"  Confidence: {segment.confidence:.2%}")
                    print(f"  Variance Contrast: {segment.variance_contrast:.2f}")
                    print(f"  Duration Plausible: {segment.duration_plausible}")

                    # Show activity curve
                    print_activity_curve(raw_activity, smoothed_activity, segment.start_frame, segment.end_frame, fps)
                else:
                    print("  No race segment detected")

            # Extract features
            results = extractor.extract_from_file(pose_file)

            for result in results:
                print(f"\n--- FEATURES: {result.lane.upper()} LANE ---")
                print(f"  Extraction Quality: {result.extraction_quality:.2%}")
                print(f"  Total Frames: {result.total_frames}")
                print(f"  Valid Frames: {result.valid_frames}")
                print(f"  Racing Frames: {result.racing_frames} (actually analyzed)")
                print(f"  Detection Confidence: {result.detection_confidence:.2%}")
                print(f"  Race Segment Confidence: {result.race_segment_confidence:.2%}")
                print(f"  Frames filtered: {result.total_frames - result.racing_frames}")

                print(f"\n  Frequency Features:")
                for k, v in result.frequency_features.items():
                    print(f"    {k}: {v:.3f}")

                print(f"\n  Efficiency Features:")
                for k, v in result.efficiency_features.items():
                    print(f"    {k}: {v:.3f}")

                print(f"\n  Posture Features:")
                for k, v in result.posture_features.items():
                    print(f"    {k}: {v:.3f}")

                all_results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    if all_results:
        output_dir = project_root / 'data' / 'ml_dataset' / 'test_output'
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / 'test_features.json'
        csv_path = output_dir / 'test_features.csv'

        save_features_json(all_results, json_path)
        save_features_csv(all_results, csv_path)

        print(f"\n{'=' * 70}")
        print(f"SUMMARY")
        print(f"{'=' * 70}")
        print(f"Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        print(f"\nTotal results: {len(all_results)} athletes from {len(test_files)} races")

        # Feature vector info
        print(f"\nFeature vector length: {len(all_results[0].to_feature_vector())}")

        # Detection method distribution
        methods = {}
        for r in all_results:
            # We don't store detection_method in FeatureResult, but we can infer from confidence
            if r.race_segment_confidence > 0.7:
                m = "variance_primary"
            elif r.race_segment_confidence > 0.5:
                m = "variance_relaxed"
            else:
                m = "fallback"
            methods[m] = methods.get(m, 0) + 1

        print(f"\nDetection quality distribution:")
        for m, count in sorted(methods.items()):
            print(f"  {m}: {count}")


if __name__ == '__main__':
    main()
