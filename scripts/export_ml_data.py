"""
ML-Ready Data Exporter

Export processed data in formats ready for machine learning.

Usage:
    python scripts/export_ml_data.py
    python scripts/export_ml_data.py --output-dir data/ml_ready

Author: Speed Climbing Analysis Project
Date: 2025-11-15
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLDataExporter:
    """Export data in ML-ready formats."""

    def __init__(
        self,
        metrics_csv: str,
        output_dir: str
    ):
        """Initialize exporter.

        Args:
            metrics_csv: Path to aggregate metrics CSV
            output_dir: Output directory
        """
        self.df = pd.read_csv(metrics_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loaded {len(self.df)} samples from {metrics_csv}")

    def export_features_csv(self):
        """Export feature matrix as CSV."""
        # Select numerical features
        feature_cols = [
            'avg_vertical_velocity',
            'max_vertical_velocity',
            'avg_acceleration',
            'max_acceleration',
            'path_length',
            'straight_distance',
            'path_efficiency',
            'smoothness_score'
        ]

        # Create feature matrix
        features = self.df[feature_cols].copy()

        # Add categorical features as one-hot
        features['lane_left'] = (self.df['lane'] == 'left').astype(int)
        features['lane_right'] = (self.df['lane'] == 'right').astype(int)

        # Add target variable (winner determination placeholder)
        # In full dataset, this would be actual race outcomes
        features['race_id'] = self.df['race_name']
        features['competition'] = self.df['competition']

        # Save
        output_file = self.output_dir / "features.csv"
        features.to_csv(output_file, index=False)

        logger.info(f"✓ Exported features CSV: {output_file}")
        logger.info(f"  Shape: {features.shape}")
        logger.info(f"  Columns: {list(features.columns)}")

        return features

    def export_numpy_arrays(self):
        """Export as NumPy arrays (.npz)."""
        # Feature matrix
        feature_cols = [
            'avg_vertical_velocity',
            'max_vertical_velocity',
            'avg_acceleration',
            'max_acceleration',
            'path_length',
            'straight_distance',
            'path_efficiency',
            'smoothness_score'
        ]

        X = self.df[feature_cols].values

        # Create synthetic labels (in real use, would be race outcomes)
        # For now, use max velocity quartiles as classes
        quartiles = pd.qcut(self.df['max_vertical_velocity'], q=4, labels=False)
        y = quartiles.values

        # Metadata
        metadata = {
            'race_names': self.df['race_name'].tolist(),
            'competitions': self.df['competition'].tolist(),
            'lanes': self.df['lane'].tolist(),
            'feature_names': feature_cols
        }

        # Save
        output_file = self.output_dir / "features.npz"
        np.savez(
            output_file,
            X=X,
            y=y,
            **metadata
        )

        logger.info(f"✓ Exported NumPy arrays: {output_file}")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        logger.info(f"  Classes: {len(np.unique(y))}")

        return X, y

    def create_train_test_split(self, test_size: float = 0.2):
        """Create train/test split files."""
        # Random split
        n_samples = len(self.df)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        split_idx = int(n_samples * (1 - test_size))

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        # Save split indices
        split_data = {
            'train_indices': train_indices.tolist(),
            'test_indices': test_indices.tolist(),
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'test_ratio': test_size
        }

        output_file = self.output_dir / "train_test_split.json"
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)

        logger.info(f"✓ Created train/test split: {output_file}")
        logger.info(f"  Train: {len(train_indices)} samples")
        logger.info(f"  Test:  {len(test_indices)} samples")

        # Save train and test CSVs
        train_df = self.df.iloc[train_indices]
        test_df = self.df.iloc[test_indices]

        train_df.to_csv(self.output_dir / "train.csv", index=False)
        test_df.to_csv(self.output_dir / "test.csv", index=False)

        logger.info(f"✓ Saved train.csv and test.csv")

        return train_indices, test_indices

    def export_metadata(self):
        """Export dataset metadata and documentation."""
        metadata = {
            'dataset_info': {
                'name': 'Speed Climbing Performance Dataset',
                'version': '1.0.0',
                'date': '2025-11-15',
                'total_samples': len(self.df),
                'features': 8,
                'competitions': int(self.df['competition'].nunique()),
                'races': int(self.df['race_name'].nunique())
            },
            'feature_descriptions': {
                'avg_vertical_velocity': 'Average vertical velocity (px/s)',
                'max_vertical_velocity': 'Maximum vertical velocity (px/s)',
                'avg_acceleration': 'Average acceleration (px/s²)',
                'max_acceleration': 'Maximum acceleration (px/s²)',
                'path_length': 'Total path length (px)',
                'straight_distance': 'Straight-line distance (px)',
                'path_efficiency': 'Ratio of straight to actual path',
                'smoothness_score': 'Movement smoothness (lower = smoother)'
            },
            'categorical_features': {
                'lane': 'Climbing lane (left/right)',
                'competition': 'Competition name',
                'race_name': 'Race identifier'
            },
            'statistics': {
                'max_velocity': {
                    'mean': float(self.df['max_vertical_velocity'].mean()),
                    'std': float(self.df['max_vertical_velocity'].std()),
                    'min': float(self.df['max_vertical_velocity'].min()),
                    'max': float(self.df['max_vertical_velocity'].max())
                },
                'path_efficiency': {
                    'mean': float(self.df['path_efficiency'].mean()),
                    'std': float(self.df['path_efficiency'].std()),
                    'min': float(self.df['path_efficiency'].min()),
                    'max': float(self.df['path_efficiency'].max())
                }
            }
        }

        output_file = self.output_dir / "dataset_metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Exported metadata: {output_file}")

        return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export ML-ready datasets"
    )

    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="data/processed/metrics/aggregate_metrics.csv",
        help="Path to aggregate metrics CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ml_ready",
        help="Output directory for ML datasets"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (0.0-1.0)"
    )

    args = parser.parse_args()

    # Initialize exporter
    exporter = MLDataExporter(args.metrics_csv, args.output_dir)

    logger.info("\nExporting ML-ready datasets...")
    logger.info("="*60)

    # Export different formats
    exporter.export_features_csv()
    exporter.export_numpy_arrays()
    exporter.create_train_test_split(args.test_size)
    exporter.export_metadata()

    logger.info("="*60)
    logger.info(f"\n✓ All datasets exported to: {args.output_dir}")
    logger.info("\nReady for:")
    logger.info("  - Scikit-learn (CSV/NumPy)")
    logger.info("  - PyTorch/TensorFlow (NumPy arrays)")
    logger.info("  - NARX networks (time-series data)")
    logger.info("  - Statistical analysis (CSV)")


if __name__ == "__main__":
    main()
