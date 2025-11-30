"""
Fuzzy Logic Engine for Speed Climbing Performance Analysis.

Uses fuzzy sets and rules to generate interpretable feedback.

IMPORTANT: This version only uses features that are VALID with moving cameras.
Features based on absolute COM position are excluded because cameras follow athletes.

Valid features (relative/angular):
- Joint angles (knee, elbow) - relative between body parts
- Body lean - angle from vertical
- Reach ratio - relative to body size
- Limb sync - correlation between limbs
- Movement amplitude - detrended, so camera motion is removed

Invalid features (excluded):
- path_straightness - COM appears stationary when camera follows
- com_stability_index - artifact of camera following
- vertical_progress_rate - cannot measure actual speed
- lateral_movement_ratio - artifact of camera following
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .baseline import BaselineStatistics


class FuzzyLevel(Enum):
    """Fuzzy linguistic levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class FuzzyMembership:
    """Membership degrees for all fuzzy levels."""
    very_low: float = 0.0
    low: float = 0.0
    medium: float = 0.0
    high: float = 0.0
    very_high: float = 0.0

    def dominant_level(self) -> Tuple[FuzzyLevel, float]:
        """Get the level with highest membership."""
        levels = [
            (FuzzyLevel.VERY_LOW, self.very_low),
            (FuzzyLevel.LOW, self.low),
            (FuzzyLevel.MEDIUM, self.medium),
            (FuzzyLevel.HIGH, self.high),
            (FuzzyLevel.VERY_HIGH, self.very_high),
        ]
        return max(levels, key=lambda x: x[1])

    def to_dict(self) -> Dict[str, float]:
        return {
            'very_low': self.very_low,
            'low': self.low,
            'medium': self.medium,
            'high': self.high,
            'very_high': self.very_high,
        }


@dataclass
class PerformanceCategory:
    """Aggregated performance for a category."""
    name: str
    name_fa: str  # Persian name
    score: float  # 0-100
    level: FuzzyLevel
    confidence: float
    strengths: List[str]
    weaknesses: List[str]
    features: Dict[str, FuzzyMembership]


class FuzzyFeedbackEngine:
    """
    Fuzzy logic engine for generating performance feedback.

    Uses trapezoidal membership functions calibrated on professional data.

    NOTE: Only uses camera-independent features (angles, ratios, sync).
    """

    # Category definitions - ONLY VALID FEATURES
    # Removed: efficiency category (all features are camera artifacts)
    # Modified: stability, rhythm to use only valid features
    CATEGORIES = {
        'coordination': {
            'name': 'Limb Coordination',
            'name_fa': 'هماهنگی اندام‌ها',
            'features': [
                'freq_limb_sync_ratio',        # Valid: correlation between limbs
                'freq_hand_movement_amplitude', # Valid: detrended
                'freq_foot_movement_amplitude', # Valid: detrended
            ],
            'weights': [0.5, 0.25, 0.25],
        },
        'leg_technique': {
            'name': 'Leg Technique',
            'name_fa': 'تکنیک پا',
            'features': [
                'post_avg_knee_angle',    # Valid: joint angle
                'post_knee_angle_std',    # Valid: angle variation
            ],
            'weights': [0.6, 0.4],
        },
        'arm_technique': {
            'name': 'Arm Technique',
            'name_fa': 'تکنیک دست',
            'features': [
                'post_avg_elbow_angle',   # Valid: joint angle
                'post_elbow_angle_std',   # Valid: angle variation
            ],
            'weights': [0.6, 0.4],
        },
        'body_position': {
            'name': 'Body Position',
            'name_fa': 'وضعیت بدن',
            'features': [
                'post_avg_body_lean',     # Valid: angle from vertical
                'post_body_lean_std',     # Valid: stability of lean
                'post_hip_width_ratio',   # Valid: ratio
            ],
            'weights': [0.4, 0.3, 0.3],
        },
        'reach': {
            'name': 'Reach & Extension',
            'name_fa': 'دسترسی و کشش',
            'features': [
                'post_avg_reach_ratio',   # Valid: ratio to body
                'post_max_reach_ratio',   # Valid: ratio to body
            ],
            'weights': [0.5, 0.5],
        },
    }

    def __init__(self, baseline: Optional[BaselineStatistics] = None):
        """
        Initialize fuzzy engine.

        Args:
            baseline: Baseline statistics for calibration. If None, uses defaults.
        """
        self.baseline = baseline or BaselineStatistics()

    def fuzzify(self, feature_name: str, value: float) -> FuzzyMembership:
        """
        Convert crisp value to fuzzy membership degrees.

        Uses percentile-based trapezoidal membership functions.
        """
        percentile = self.baseline.get_percentile(feature_name, value)

        # Trapezoidal membership functions based on percentiles
        membership = FuzzyMembership()

        # Very Low: 0-20 percentile
        if percentile <= 10:
            membership.very_low = 1.0
        elif percentile <= 20:
            membership.very_low = (20 - percentile) / 10

        # Low: 10-40 percentile
        if 10 <= percentile <= 20:
            membership.low = (percentile - 10) / 10
        elif 20 < percentile <= 30:
            membership.low = 1.0
        elif 30 < percentile <= 40:
            membership.low = (40 - percentile) / 10

        # Medium: 30-70 percentile
        if 30 <= percentile <= 40:
            membership.medium = (percentile - 30) / 10
        elif 40 < percentile <= 60:
            membership.medium = 1.0
        elif 60 < percentile <= 70:
            membership.medium = (70 - percentile) / 10

        # High: 60-90 percentile
        if 60 <= percentile <= 70:
            membership.high = (percentile - 60) / 10
        elif 70 < percentile <= 80:
            membership.high = 1.0
        elif 80 < percentile <= 90:
            membership.high = (90 - percentile) / 10

        # Very High: 80-100 percentile
        if 80 <= percentile <= 90:
            membership.very_high = (percentile - 80) / 10
        elif percentile > 90:
            membership.very_high = 1.0

        return membership

    def evaluate_feature(self, feature_name: str, value: float) -> Dict:
        """
        Evaluate a single feature and determine quality.

        Returns dict with fuzzy membership, quality assessment, and score.
        """
        membership = self.fuzzify(feature_name, value)
        dominant_level, confidence = membership.dominant_level()
        percentile = self.baseline.get_percentile(feature_name, value)

        # Determine if this is good or bad based on feature semantics
        higher_is_better = self.baseline.is_higher_better(feature_name)
        optimal_range = self.baseline.get_optimal_range(feature_name)

        # Calculate quality score (0-100)
        if optimal_range:
            low, high = optimal_range
            if low <= value <= high:
                score = 80 + 20 * (1 - abs(value - (low + high) / 2) / ((high - low) / 2))
            else:
                distance = min(abs(value - low), abs(value - high))
                range_size = high - low
                score = max(0, 70 - 70 * min(distance / range_size, 1))
        elif higher_is_better is True:
            score = percentile
        elif higher_is_better is False:
            score = 100 - percentile
        else:
            # Neutral - medium is best
            score = 100 - abs(percentile - 50) * 2

        # Quality label
        if score >= 80:
            quality = "excellent"
        elif score >= 60:
            quality = "good"
        elif score >= 40:
            quality = "average"
        else:
            quality = "needs_improvement"

        return {
            'value': value,
            'percentile': percentile,
            'membership': membership,
            'dominant_level': dominant_level,
            'confidence': confidence,
            'score': score,
            'quality': quality,
            'higher_is_better': higher_is_better,
        }

    def evaluate_category(self, category_name: str, features: Dict[str, float]) -> PerformanceCategory:
        """
        Evaluate a performance category using fuzzy aggregation.

        Args:
            category_name: Name of category (coordination, leg_technique, etc.)
            features: Dict of feature_name -> value

        Returns:
            PerformanceCategory with aggregated score and insights
        """
        if category_name not in self.CATEGORIES:
            raise ValueError(f"Unknown category: {category_name}")

        cat_config = self.CATEGORIES[category_name]
        cat_features = cat_config['features']
        weights = cat_config['weights']

        # Evaluate each feature
        feature_results = {}
        scores = []
        available_weights = []

        for i, feat_name in enumerate(cat_features):
            if feat_name in features:
                result = self.evaluate_feature(feat_name, features[feat_name])
                feature_results[feat_name] = result['membership']
                scores.append(result['score'])
                available_weights.append(weights[i])

        # Weighted average score
        if scores:
            total_weight = sum(available_weights)
            normalized_weights = [w / total_weight for w in available_weights]
            category_score = sum(s * w for s, w in zip(scores, normalized_weights))
        else:
            category_score = 50.0  # Default if no features available

        # Determine category level
        if category_score >= 80:
            level = FuzzyLevel.VERY_HIGH
        elif category_score >= 65:
            level = FuzzyLevel.HIGH
        elif category_score >= 45:
            level = FuzzyLevel.MEDIUM
        elif category_score >= 25:
            level = FuzzyLevel.LOW
        else:
            level = FuzzyLevel.VERY_LOW

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        for feat_name in cat_features:
            if feat_name in features:
                result = self.evaluate_feature(feat_name, features[feat_name])
                if result['quality'] == 'excellent':
                    strengths.append(feat_name)
                elif result['quality'] == 'needs_improvement':
                    weaknesses.append(feat_name)

        # Calculate confidence based on feature coverage
        coverage = len(feature_results) / len(cat_features)
        confidence = coverage * 0.8 + 0.2  # Min 20% confidence

        return PerformanceCategory(
            name=cat_config['name'],
            name_fa=cat_config['name_fa'],
            score=category_score,
            level=level,
            confidence=confidence,
            strengths=strengths,
            weaknesses=weaknesses,
            features=feature_results,
        )

    def evaluate_all(self, features: Dict[str, float]) -> Dict[str, PerformanceCategory]:
        """
        Evaluate all performance categories.

        Args:
            features: Dict of all feature values

        Returns:
            Dict of category_name -> PerformanceCategory
        """
        results = {}
        for cat_name in self.CATEGORIES:
            results[cat_name] = self.evaluate_category(cat_name, features)
        return results

    def get_overall_score(self, features: Dict[str, float]) -> Tuple[float, FuzzyLevel]:
        """
        Calculate overall performance score.

        Returns (score, level) tuple.
        """
        categories = self.evaluate_all(features)

        # Weighted average across categories
        weights = {
            'coordination': 0.25,
            'leg_technique': 0.20,
            'arm_technique': 0.20,
            'body_position': 0.20,
            'reach': 0.15,
        }

        total_score = 0
        total_weight = 0

        for cat_name, cat_result in categories.items():
            weight = weights.get(cat_name, 0.2) * cat_result.confidence
            total_score += cat_result.score * weight
            total_weight += weight

        if total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = 50.0

        # Determine level
        if overall_score >= 80:
            level = FuzzyLevel.VERY_HIGH
        elif overall_score >= 65:
            level = FuzzyLevel.HIGH
        elif overall_score >= 45:
            level = FuzzyLevel.MEDIUM
        elif overall_score >= 25:
            level = FuzzyLevel.LOW
        else:
            level = FuzzyLevel.VERY_LOW

        return overall_score, level
