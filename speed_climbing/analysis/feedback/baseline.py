"""
Baseline statistics from professional athlete dataset.

Used for comparing user performance against elite climbers.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureBaseline:
    """Baseline statistics for a single feature."""
    mean: float
    std: float
    min: float
    max: float
    p25: float  # 25th percentile
    p50: float  # median
    p75: float  # 75th percentile

    def get_percentile(self, value: float) -> float:
        """
        Estimate percentile of a value using normal distribution approximation.
        Returns 0-100 percentile.
        """
        if self.std == 0:
            return 50.0
        z_score = (value - self.mean) / self.std
        # Approximate percentile from z-score
        percentile = 50 * (1 + np.tanh(z_score * 0.7))
        return np.clip(percentile, 0, 100)

    def categorize(self, value: float) -> str:
        """Categorize value as low/medium/high based on quartiles."""
        if value <= self.p25:
            return "low"
        elif value <= self.p75:
            return "medium"
        else:
            return "high"


class BaselineStatistics:
    """
    Manages baseline statistics from professional dataset.

    These statistics are used for:
    1. Fuzzy membership function calibration
    2. Percentile ranking of user performance
    3. Generating comparative feedback
    """

    # Default baselines from dataset analysis (371 samples)
    # These are extracted from feature_statistics.csv
    DEFAULT_BASELINES = {
        # Frequency features
        'freq_hand_frequency_hz': {
            'mean': 0.905, 'std': 0.479, 'min': 0.0, 'max': 3.49,
            'p25': 0.60, 'p50': 0.70, 'p75': 1.15,
            'higher_is_better': None,  # Optimal is around 0.8-1.2 Hz
            'optimal_range': (0.7, 1.2),
        },
        'freq_foot_frequency_hz': {
            'mean': 0.797, 'std': 0.433, 'min': 0.0, 'max': 2.71,
            'p25': 0.56, 'p50': 0.66, 'p75': 0.88,
            'higher_is_better': None,
            'optimal_range': (0.6, 1.0),
        },
        'freq_limb_sync_ratio': {
            'mean': 0.590, 'std': 0.204, 'min': 0.0, 'max': 0.99,
            'p25': 0.46, 'p50': 0.59, 'p75': 0.73,
            'higher_is_better': True,  # Higher sync is better
        },
        'freq_movement_regularity': {
            'mean': 0.409, 'std': 0.114, 'min': 0.0, 'max': 0.71,
            'p25': 0.34, 'p50': 0.41, 'p75': 0.48,
            'higher_is_better': True,  # More regular is better
        },
        'freq_hand_movement_amplitude': {
            'mean': 0.088, 'std': 0.042, 'min': 0.0, 'max': 0.31,
            'p25': 0.063, 'p50': 0.078, 'p75': 0.106,
            'higher_is_better': None,  # Depends on technique
        },
        'freq_foot_movement_amplitude': {
            'mean': 0.073, 'std': 0.056, 'min': 0.0, 'max': 0.62,
            'p25': 0.045, 'p50': 0.060, 'p75': 0.085,
            'higher_is_better': None,
        },

        # Efficiency features
        'eff_path_straightness': {
            'mean': 0.221, 'std': 0.150, 'min': 0.0, 'max': 0.89,
            'p25': 0.12, 'p50': 0.19, 'p75': 0.29,
            'higher_is_better': True,  # Straighter path is more efficient
        },
        'eff_lateral_movement_ratio': {
            'mean': 1.056, 'std': 3.197, 'min': 0.0, 'max': 38.8,
            'p25': 0.15, 'p50': 0.28, 'p75': 0.60,
            'higher_is_better': False,  # Less lateral movement is better
        },
        'eff_com_stability_index': {
            'mean': 0.000189, 'std': 0.000113, 'min': 0.0, 'max': 0.00088,
            'p25': 0.0001, 'p50': 0.0002, 'p75': 0.00026,
            'higher_is_better': False,  # Lower variance = more stable
        },
        'eff_movement_smoothness': {
            'mean': 0.054, 'std': 0.007, 'min': 0.0, 'max': 0.066,
            'p25': 0.052, 'p50': 0.055, 'p75': 0.057,
            'higher_is_better': True,  # Smoother is better
        },
        'eff_acceleration_variance': {
            'mean': 0.062, 'std': 0.033, 'min': 0.0, 'max': 0.16,
            'p25': 0.036, 'p50': 0.059, 'p75': 0.087,
            'higher_is_better': False,  # Lower variance = more consistent
        },

        # Posture features
        'post_avg_knee_angle': {
            'mean': 126.1, 'std': 20.4, 'min': 0.0, 'max': 174.5,
            'p25': 118.4, 'p50': 128.0, 'p75': 136.4,
            'higher_is_better': None,
            'optimal_range': (110, 140),  # Optimal knee bend
        },
        'post_avg_elbow_angle': {
            'mean': 119.6, 'std': 16.8, 'min': 37.4, 'max': 174.8,
            'p25': 109.9, 'p50': 118.1, 'p75': 128.1,
            'higher_is_better': None,
            'optimal_range': (100, 140),
        },
        'post_avg_body_lean': {
            'mean': 3.07, 'std': 20.3, 'min': -66.7, 'max': 176.4,
            'p25': -2.6, 'p50': -0.5, 'p75': 1.8,
            'higher_is_better': None,
            'optimal_range': (-5, 5),  # Close to vertical is good
        },
        'post_body_lean_std': {
            'mean': 20.5, 'std': 32.0, 'min': 0.5, 'max': 158.9,
            'p25': 3.9, 'p50': 5.0, 'p75': 18.5,
            'higher_is_better': False,  # Less variation is more stable
        },
        'post_hip_width_ratio': {
            'mean': 0.590, 'std': 0.072, 'min': 0.16, 'max': 0.89,
            'p25': 0.57, 'p50': 0.60, 'p75': 0.62,
            'higher_is_better': None,
        },
        'post_avg_reach_ratio': {
            'mean': 0.626, 'std': 0.269, 'min': 0.17, 'max': 4.79,
            'p25': 0.54, 'p50': 0.58, 'p75': 0.64,
            'higher_is_better': None,
        },
        'post_max_reach_ratio': {
            'mean': 1.99, 'std': 7.09, 'min': 0.38, 'max': 133.4,
            'p25': 1.09, 'p50': 1.18, 'p75': 1.37,
            'higher_is_better': None,
        },
    }

    def __init__(self, custom_baselines: Optional[Dict] = None):
        """
        Initialize with optional custom baselines.

        Args:
            custom_baselines: Override default baselines with custom values
        """
        self.baselines: Dict[str, FeatureBaseline] = {}
        self.metadata: Dict[str, Dict] = {}

        # Load defaults
        for name, data in self.DEFAULT_BASELINES.items():
            self.baselines[name] = FeatureBaseline(
                mean=data['mean'],
                std=data['std'],
                min=data['min'],
                max=data['max'],
                p25=data['p25'],
                p50=data['p50'],
                p75=data['p75'],
            )
            self.metadata[name] = {
                'higher_is_better': data.get('higher_is_better'),
                'optimal_range': data.get('optimal_range'),
            }

        # Apply custom overrides
        if custom_baselines:
            self._apply_custom(custom_baselines)

    def _apply_custom(self, custom: Dict):
        """Apply custom baseline overrides."""
        for name, data in custom.items():
            if name in self.baselines:
                self.baselines[name] = FeatureBaseline(**data)

    def get_baseline(self, feature_name: str) -> Optional[FeatureBaseline]:
        """Get baseline for a feature."""
        return self.baselines.get(feature_name)

    def get_percentile(self, feature_name: str, value: float) -> float:
        """Get percentile rank of a value for a feature."""
        baseline = self.get_baseline(feature_name)
        if baseline is None:
            return 50.0
        return baseline.get_percentile(value)

    def is_higher_better(self, feature_name: str) -> Optional[bool]:
        """Check if higher values are better for this feature."""
        return self.metadata.get(feature_name, {}).get('higher_is_better')

    def get_optimal_range(self, feature_name: str) -> Optional[tuple]:
        """Get optimal range for a feature if defined."""
        return self.metadata.get(feature_name, {}).get('optimal_range')

    def compare_to_baseline(self, features: Dict[str, float]) -> Dict[str, Dict]:
        """
        Compare a set of features to baselines.

        Returns dict with percentile, category, and quality assessment for each feature.
        """
        results = {}
        for name, value in features.items():
            if name not in self.baselines:
                continue

            baseline = self.baselines[name]
            percentile = baseline.get_percentile(value)
            category = baseline.categorize(value)

            # Determine quality based on higher_is_better or optimal_range
            quality = self._assess_quality(name, value, percentile)

            results[name] = {
                'value': value,
                'percentile': percentile,
                'category': category,
                'quality': quality,
                'baseline_mean': baseline.mean,
                'baseline_std': baseline.std,
            }

        return results

    def _assess_quality(self, feature_name: str, value: float, percentile: float) -> str:
        """Assess quality of a feature value."""
        higher_is_better = self.is_higher_better(feature_name)
        optimal_range = self.get_optimal_range(feature_name)

        if optimal_range:
            low, high = optimal_range
            if low <= value <= high:
                return "excellent"
            elif value < low:
                diff = (low - value) / (low if low != 0 else 1)
                return "needs_improvement" if diff > 0.3 else "good"
            else:
                diff = (value - high) / (high if high != 0 else 1)
                return "needs_improvement" if diff > 0.3 else "good"

        if higher_is_better is True:
            if percentile >= 75:
                return "excellent"
            elif percentile >= 50:
                return "good"
            elif percentile >= 25:
                return "average"
            else:
                return "needs_improvement"
        elif higher_is_better is False:
            if percentile <= 25:
                return "excellent"
            elif percentile <= 50:
                return "good"
            elif percentile <= 75:
                return "average"
            else:
                return "needs_improvement"
        else:
            # No clear better direction, medium is good
            if 25 <= percentile <= 75:
                return "good"
            else:
                return "average"

    @classmethod
    def from_csv(cls, csv_path: Path) -> 'BaselineStatistics':
        """Load baselines from feature_statistics.csv file."""
        import csv

        custom = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['feature']
                custom[name] = {
                    'mean': float(row['mean']),
                    'std': float(row['std']),
                    'min': float(row['min']),
                    'max': float(row['max']),
                    'p25': float(row['p25']),
                    'p50': float(row.get('median', row.get('p50', 0))),
                    'p75': float(row['p75']),
                }

        return cls(custom_baselines=custom)
