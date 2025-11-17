"""
Data Aggregation & ML Export - Reliable Races
==============================================
Aggregate metrics, create leaderboards, export ML-ready datasets.

Outputs:
- Leaderboards (by competition, overall)
- Comparative statistics
- ML datasets (CSV, NumPy, train/test splits)

Usage:
    python scripts/aggregate_reliable_data.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def aggregate_all_metrics():
    """Aggregate all metrics from 114 reliable races."""

    print("="*70)
    print("Data Aggregation & ML Export")
    print("="*70)

    # Load reliable races list
    reliable_races_file = Path('data/processed/reliable_races_list.json')
    with open(reliable_races_file, 'r', encoding='utf-8') as f:
        reliable_data = json.load(f)

    race_ids = reliable_data['reliable_race_ids']
    print(f"Loading metrics for {len(race_ids)} reliable races...")

    all_metrics = []
    missing_count = 0

    for race_id in race_ids:
        metrics_path = Path(f"data/processed/metrics/{race_id}_metrics.json")

        if not metrics_path.exists():
            missing_count += 1
            print(f"  ⚠️  Missing metrics for: {race_id}")
            continue

        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        all_metrics.append(metrics)

    print(f"Loaded {len(all_metrics)} metrics files ({missing_count} missing)")

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Save aggregated CSV
    csv_path = Path('data/processed/aggregated_metrics_reliable.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved aggregated CSV: {csv_path}")
    print(f"  Races: {len(df)}, Features: {len(df.columns)}")

    # Create leaderboards
    create_leaderboards(df)

    # Export ML datasets
    export_ml_datasets(df)

    print("\n" + "="*70)
    print("✅ Aggregation complete!")
    print("="*70)
    print(f"Races processed: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"Leaderboards created: {len(df['competition'].unique()) + 1}")
    print("="*70)


def create_leaderboards(df):
    """Create various leaderboards."""
    print("\n📊 Creating leaderboards...")

    # Overall leaderboard (by average velocity)
    if 'average_velocity_ms' in df.columns:
        leaderboard_overall = df.nlargest(20, 'average_velocity_ms')[
            ['race_id', 'competition', 'average_velocity_ms', 'total_time_s']
        ]
        overall_path = Path('data/processed/leaderboard_overall.csv')
        leaderboard_overall.to_csv(overall_path, index=False)
        print(f"  ✓ Overall leaderboard: {overall_path}")

    # By competition
    if 'competition' in df.columns:
        comp_count = 0
        for comp in df['competition'].unique():
            comp_df = df[df['competition'] == comp]
            if 'average_velocity_ms' in comp_df.columns:
                comp_leaderboard = comp_df.nlargest(10, 'average_velocity_ms')[
                    ['race_id', 'average_velocity_ms', 'total_time_s']
                ]
                comp_path = Path(f'data/processed/leaderboard_{comp}.csv')
                comp_leaderboard.to_csv(comp_path, index=False)
                comp_count += 1

        print(f"  ✓ Competition leaderboards: {comp_count} files")


def export_ml_datasets(df):
    """Export ML-ready datasets."""
    print("\n🤖 Exporting ML datasets...")

    # Feature selection (adjust based on available columns)
    feature_cols = [
        'average_velocity_ms',
        'max_velocity_ms',
        'acceleration_ms2',
        'smoothness_score',
        'total_time_s',
        'path_length_m',
        'avg_vertical_velocity_ms',
        'efficiency'
    ]

    # Filter to available features
    available_features = [col for col in feature_cols if col in df.columns]

    if not available_features:
        print("  ⚠️  No numeric features found for ML export")
        return

    print(f"  Features selected: {len(available_features)}")
    for feat in available_features:
        print(f"    - {feat}")

    # Filter rows with non-null values for these features
    df_clean = df[available_features].dropna()

    if len(df_clean) == 0:
        print("  ⚠️  No complete rows after removing NaN values")
        return

    X = df_clean.values
    y = df_clean['average_velocity_ms'].values if 'average_velocity_ms' in available_features else df_clean.iloc[:, 0].values

    # Train/test split (80/20)
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Save NumPy arrays
        np.save('data/processed/ml_X_train.npy', X_train)
        np.save('data/processed/ml_X_test.npy', X_test)
        np.save('data/processed/ml_y_train.npy', y_train)
        np.save('data/processed/ml_y_test.npy', y_test)

        print(f"\n  ✓ ML datasets exported:")
        print(f"    Train: {X_train.shape} (80%)")
        print(f"    Test:  {X_test.shape} (20%)")

        # Save feature names
        feature_info = {
            'features': available_features,
            'target': 'average_velocity_ms' if 'average_velocity_ms' in available_features else available_features[0],
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'num_features': len(available_features),
            'export_date': datetime.now().isoformat()
        }

        with open('data/processed/ml_feature_names.json', 'w') as f:
            json.dump(feature_info, f, indent=2)

        print(f"  ✓ Feature metadata saved")

    except ImportError:
        print("  ⚠️  scikit-learn not installed, skipping train/test split")
        print("  ℹ️  Install with: pip install scikit-learn")

        # Save full dataset as single array
        np.save('data/processed/ml_X_full.npy', X)
        np.save('data/processed/ml_y_full.npy', y)

        with open('data/processed/ml_feature_names.json', 'w') as f:
            json.dump({
                'features': available_features,
                'export_date': datetime.now().isoformat()
            }, f, indent=2)

        print(f"  ✓ Full dataset exported: {X.shape}")


if __name__ == "__main__":
    try:
        aggregate_all_metrics()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
