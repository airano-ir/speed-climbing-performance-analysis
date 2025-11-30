"""
Feature extraction modules for speed climbing analysis.

This package provides ML-ready feature extraction from pose data,
independent of wall calibration.
"""

from .base import (
    calculate_angle,
    extract_keypoint_series,
    compute_path_length,
    normalize_series,
)
from .frequency import FrequencyAnalyzer
from .efficiency import EfficiencyAnalyzer
from .posture import PostureAnalyzer
from .race_detector import RaceSegmentDetector, RaceSegment
from .extractor import FeatureExtractor, FeatureResult, save_features_json, save_features_csv

__all__ = [
    'calculate_angle',
    'extract_keypoint_series',
    'compute_path_length',
    'normalize_series',
    'FrequencyAnalyzer',
    'EfficiencyAnalyzer',
    'PostureAnalyzer',
    'FeatureExtractor',
    'FeatureResult',
    'RaceSegmentDetector',
    'RaceSegment',
    'save_features_json',
    'save_features_csv',
]
