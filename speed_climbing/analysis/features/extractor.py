"""
Main feature extractor that combines all analyzers.

Provides a unified interface for extracting ML-ready features from pose data.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np

from .frequency import FrequencyAnalyzer
from .efficiency import EfficiencyAnalyzer
from .posture import PostureAnalyzer


@dataclass
class FeatureResult:
    """Result of feature extraction for one athlete."""
    video_id: str
    lane: str
    extraction_quality: float

    # Feature groups
    frequency_features: Dict[str, float]
    efficiency_features: Dict[str, float]
    posture_features: Dict[str, float]

    # Metadata
    total_frames: int
    valid_frames: int
    fps: float
    detection_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, float]:
        """
        Flatten all features into a single dictionary.

        Useful for ML training where we need a flat feature vector.
        """
        flat = {
            'extraction_quality': self.extraction_quality,
            'detection_confidence': self.detection_confidence,
        }

        # Add prefixed features
        for key, value in self.frequency_features.items():
            flat[f'freq_{key}'] = value

        for key, value in self.efficiency_features.items():
            flat[f'eff_{key}'] = value

        for key, value in self.posture_features.items():
            flat[f'post_{key}'] = value

        return flat

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to numpy array for ML training.

        Returns features in consistent order.
        """
        flat = self.to_flat_dict()
        # Sort keys for consistent ordering
        sorted_keys = sorted(flat.keys())
        return np.array([flat[k] for k in sorted_keys])

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get ordered list of feature names."""
        # This should match to_feature_vector() ordering
        names = ['detection_confidence', 'extraction_quality']

        freq_names = ['hand_frequency_hz', 'foot_frequency_hz', 'limb_sync_ratio',
                      'movement_regularity', 'hand_movement_amplitude', 'foot_movement_amplitude']
        names.extend([f'freq_{n}' for n in freq_names])

        eff_names = ['path_straightness', 'lateral_movement_ratio', 'vertical_progress_rate',
                     'com_stability_index', 'movement_smoothness', 'acceleration_variance']
        names.extend([f'eff_{n}' for n in eff_names])

        post_names = ['avg_knee_angle', 'knee_angle_std', 'avg_elbow_angle', 'elbow_angle_std',
                      'hip_width_ratio', 'avg_body_lean', 'body_lean_std',
                      'avg_reach_ratio', 'max_reach_ratio']
        names.extend([f'post_{n}' for n in post_names])

        return sorted(names)


class FeatureExtractor:
    """
    Main class for extracting features from pose data.

    Combines frequency, efficiency, and posture analysis.
    """

    def __init__(self, fps: float = 30.0, min_frames: int = 30):
        """
        Args:
            fps: Video frame rate (used for frequency calculations)
            min_frames: Minimum frames required for analysis
        """
        self.fps = fps
        self.min_frames = min_frames

        # Initialize analyzers
        self.frequency_analyzer = FrequencyAnalyzer(fps=fps, min_frames=min_frames)
        self.efficiency_analyzer = EfficiencyAnalyzer(fps=fps, min_frames=min_frames)
        self.posture_analyzer = PostureAnalyzer(min_frames=min_frames)

    def extract_from_file(self, pose_file: Union[str, Path]) -> List[FeatureResult]:
        """
        Extract features from a pose JSON file.

        Args:
            pose_file: Path to pose JSON file

        Returns:
            List of FeatureResult (one per detected lane)
        """
        pose_path = Path(pose_file)

        with open(pose_path, 'r', encoding='utf-8') as f:
            pose_data = json.load(f)

        return self.extract_from_data(pose_data, video_id=pose_path.stem)

    def extract_from_data(
        self,
        pose_data: Dict[str, Any],
        video_id: str = 'unknown'
    ) -> List[FeatureResult]:
        """
        Extract features from pose data dictionary.

        Args:
            pose_data: Parsed pose JSON data
            video_id: Identifier for the video

        Returns:
            List of FeatureResult (one per detected lane)
        """
        metadata = pose_data.get('metadata', {})
        frames = pose_data.get('frames', [])

        fps = metadata.get('fps', self.fps)
        total_frames = len(frames)

        results = []

        # Process each lane
        for lane in ['left', 'right']:
            detection_rate_key = f'detection_rate_{lane}'
            detection_rate = metadata.get(detection_rate_key, 0.0)

            # Skip if detection rate is too low
            if detection_rate < 0.3:
                continue

            result = self._extract_lane(
                frames=frames,
                lane=lane,
                video_id=video_id,
                fps=fps,
                total_frames=total_frames,
                detection_rate=detection_rate
            )

            if result is not None:
                results.append(result)

        return results

    def _extract_lane(
        self,
        frames: List[Dict[str, Any]],
        lane: str,
        video_id: str,
        fps: float,
        total_frames: int,
        detection_rate: float
    ) -> Optional[FeatureResult]:
        """Extract features for a single lane."""

        # Count valid frames for this lane
        valid_frames = sum(
            1 for f in frames
            if f.get(f'{lane}_climber') is not None
            and f.get(f'{lane}_climber', {}).get('has_detection', False)
        )

        if valid_frames < self.min_frames:
            return None

        # Extract features
        try:
            freq_features = self.frequency_analyzer.analyze(frames, lane)
            eff_features = self.efficiency_analyzer.analyze(frames, lane)
            post_features = self.posture_analyzer.analyze(frames, lane)
        except Exception as e:
            print(f"Warning: Feature extraction failed for {video_id}/{lane}: {e}")
            return None

        # Calculate extraction quality
        # Based on: valid frame ratio, detection rate, and feature completeness
        feature_count = (
            sum(1 for v in freq_features.values() if v != 0.0) +
            sum(1 for v in eff_features.values() if v != 0.0) +
            sum(1 for v in post_features.values() if v != 0.0)
        )
        total_features = len(freq_features) + len(eff_features) + len(post_features)

        feature_completeness = feature_count / total_features if total_features > 0 else 0
        valid_ratio = valid_frames / total_frames if total_frames > 0 else 0

        extraction_quality = (detection_rate + valid_ratio + feature_completeness) / 3

        return FeatureResult(
            video_id=video_id,
            lane=lane,
            extraction_quality=extraction_quality,
            frequency_features=freq_features,
            efficiency_features=eff_features,
            posture_features=post_features,
            total_frames=total_frames,
            valid_frames=valid_frames,
            fps=fps,
            detection_confidence=detection_rate
        )

    def extract_batch(
        self,
        pose_files: List[Union[str, Path]],
        progress_callback: Optional[callable] = None
    ) -> List[FeatureResult]:
        """
        Extract features from multiple pose files.

        Args:
            pose_files: List of paths to pose JSON files
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of all FeatureResult objects
        """
        all_results = []
        total = len(pose_files)

        for i, pose_file in enumerate(pose_files):
            try:
                results = self.extract_from_file(pose_file)
                all_results.extend(results)
            except Exception as e:
                print(f"Warning: Failed to process {pose_file}: {e}")

            if progress_callback:
                progress_callback(i + 1, total)

        return all_results


def save_features_json(
    results: List[FeatureResult],
    output_path: Union[str, Path]
) -> None:
    """Save feature results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict() for r in results]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_features_csv(
    results: List[FeatureResult],
    output_path: Union[str, Path]
) -> None:
    """Save feature results to CSV file (flat format for ML)."""
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    # Get all feature names
    feature_names = FeatureResult.get_feature_names()
    header = ['video_id', 'lane'] + feature_names

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for result in results:
            flat = result.to_flat_dict()
            row = [result.video_id, result.lane]
            row.extend([flat.get(name, 0.0) for name in feature_names])
            writer.writerow(row)
