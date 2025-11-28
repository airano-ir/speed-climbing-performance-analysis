"""
Quick test for the feature extraction pipeline.

Shows detailed debug output including activity curves, wrist height,
and comparison with known actual times.
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


# Known actual times from user feedback
ACTUAL_TIMES = {
    "Speed_finals_Seoul_2024_race023": {
        "left": {"end": 5.93},
        "right": {"end": 5.65}
    },
    "Speed_finals_Chamonix_2024_race024": {
        "left": {"end": 6.25},
        "right": {"end": 6.20}
    },
    "Speed_finals_Innsbruck_2024_race003": {
        "left": {"end": 6.96},
        "right": {"end": 6.55}
    }
}


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
    print(f"  Duration: {(end-start)/fps:.2f}s at {fps:.1f}fps")
    print()

    # Sample every few frames to fit in terminal
    step = max(1, n // 30)
    width = 30

    print("  Frame    Activity")
    print("  " + "-" * 45)

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

        print(f"  {prefix}{i:4d}  |{bar:<{width}}|{marker}")

    print("  " + "-" * 45)


def print_wrist_height_curve(wrist_y: np.ndarray, start: int, end: int, fps: float = 30.0):
    """Print a text-based visualization of wrist height (inverted Y)."""
    n = len(wrist_y)
    if n == 0:
        print("  No wrist data available")
        return

    # Invert: lower Y = higher position, so show as height
    height = 1.0 - wrist_y  # Now higher value = higher on wall

    max_height = np.max(height)
    min_height = np.min(height)

    if max_height - min_height < 0.01:
        print("  No significant height change detected")
        return

    # Normalize for display
    normalized = (height - min_height) / (max_height - min_height + 1e-6)

    # Print header
    print(f"\n  Wrist Height (higher = closer to top):")
    print(f"  Min Y: {np.min(wrist_y):.3f} (highest point)")
    print()

    # Sample every few frames
    step = max(1, n // 25)
    width = 25

    print("  Frame    Height")
    print("  " + "-" * 40)

    for i in range(0, n, step):
        bar_len = int(normalized[i] * width)
        bar = "=" * bar_len

        # Mark end (minimum Y = maximum height)
        marker = ""
        if abs(i - end) < step:
            marker = " <-- TOP"

        is_racing = start <= i <= end
        prefix = ">" if is_racing else " "

        print(f"  {prefix}{i:4d}  |{bar:<{width}}|{marker}")

    print("  " + "-" * 40)


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

    print(f"Testing HYBRID race detection on {len(test_files)} files...")
    print("=" * 70)

    # Initialize extractor
    extractor = FeatureExtractor(fps=30.0, min_frames=30)

    all_results = []
    comparison_results = []

    for pose_file in test_files:
        print(f"\n{'='*70}")
        print(f"Processing: {pose_file.name}")
        print("=" * 70)

        # Get video ID for comparison
        video_id = pose_file.stem.replace('_poses', '')

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

                # Get wrist height curves
                wrist_y_raw, wrist_y_smooth = detector.get_wrist_height_curve(frames, lane)

                # Detect segment
                segment = detector.detect(frames, lane)

                if segment:
                    print(f"  Detection Method: {segment.detection_method}")
                    print(f"  Start Method: {segment.start_method}")
                    print(f"  End Method: {segment.end_method}")
                    print(f"  Start Frame: {segment.start_frame} ({segment.start_frame/fps:.2f}s)")
                    print(f"  End Frame: {segment.end_frame} ({segment.end_frame/fps:.2f}s)")
                    print(f"  Duration: {segment.duration_frames} frames ({segment.duration_frames/fps:.2f}s)")
                    print(f"  Confidence: {segment.confidence:.2%}")

                    # Compare with actual if available
                    actual = ACTUAL_TIMES.get(video_id, {}).get(lane, {})
                    if actual.get('end'):
                        detected_end = segment.end_frame / fps
                        actual_end = actual['end']
                        error = detected_end - actual_end
                        print(f"\n  COMPARISON:")
                        print(f"    Detected End: {detected_end:.2f}s")
                        print(f"    Actual End:   {actual_end:.2f}s")
                        status = "OK" if abs(error) < 0.3 else "BAD"
                        print(f"    Error:        {error:+.2f}s [{status}]")

                        comparison_results.append({
                            'video': video_id,
                            'lane': lane,
                            'detected_end': detected_end,
                            'actual_end': actual_end,
                            'error': error,
                            'method': segment.end_method
                        })

                    # Show activity curve
                    print_activity_curve(raw_activity, smoothed_activity, segment.start_frame, segment.end_frame, fps)

                    # Show wrist height curve
                    print_wrist_height_curve(wrist_y_smooth, segment.start_frame, segment.end_frame, fps)
                else:
                    print("  No race segment detected")

            # Extract features
            results = extractor.extract_from_file(pose_file)

            for result in results:
                print(f"\n--- FEATURES: {result.lane.upper()} LANE ---")
                print(f"  Extraction Quality: {result.extraction_quality:.2%}")
                print(f"  Total Frames: {result.total_frames}")
                print(f"  Racing Frames: {result.racing_frames}")
                print(f"  Confidence: {result.race_segment_confidence:.2%}")

                all_results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    if comparison_results:
        print("\n| Video | Lane | Detected | Actual | Error | Status | Method |")
        print("|-------|------|----------|--------|-------|--------|--------|")
        total_error = 0
        for r in comparison_results:
            status = "OK" if abs(r['error']) < 0.3 else "BAD"
            print(f"| {r['video'][-8:]} | {r['lane']:5} | {r['detected_end']:.2f}s | {r['actual_end']:.2f}s | {r['error']:+.2f}s | {status:3} | {r['method']} |")
            total_error += abs(r['error'])

        avg_error = total_error / len(comparison_results)
        print(f"\nAverage absolute error: {avg_error:.2f}s")

    # Save results
    if all_results:
        output_dir = project_root / 'data' / 'ml_dataset' / 'test_output'
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / 'test_features.json'
        csv_path = output_dir / 'test_features.csv'

        save_features_json(all_results, json_path)
        save_features_csv(all_results, csv_path)

        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")


if __name__ == '__main__':
    main()
