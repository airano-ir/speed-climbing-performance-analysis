"""
Quick test for the feature extraction pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from speed_climbing.analysis.features import FeatureExtractor, save_features_json, save_features_csv


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
    print("-" * 60)

    # Initialize extractor
    extractor = FeatureExtractor(fps=30.0, min_frames=30)

    all_results = []

    for pose_file in test_files:
        print(f"\nProcessing: {pose_file.name}")

        try:
            results = extractor.extract_from_file(pose_file)

            for result in results:
                print(f"  Lane: {result.lane}")
                print(f"  Extraction Quality: {result.extraction_quality:.2%}")
                print(f"  Valid Frames: {result.valid_frames}/{result.total_frames}")
                print(f"  Detection Confidence: {result.detection_confidence:.2%}")

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
                print("-" * 40)

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

        print(f"\n{'=' * 60}")
        print(f"Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        print(f"\nTotal results: {len(all_results)} athletes from {len(test_files)} races")

        # Show feature vector example
        print(f"\nFeature vector length: {len(all_results[0].to_feature_vector())}")
        print(f"Feature names: {FeatureExtractor}")


if __name__ == '__main__':
    main()
