"""
Bulk Operations Manager
========================
Batch processing for multiple races.

Features:
- Batch export (JSON, CSV, YAML, NPZ)
- Batch validation
- Batch metrics recalculation
- Progress tracking
- ML-ready data export

⭐ Phase 1.5.1 Feature (2025-11-16)
"""

from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import json
import csv
from datetime import datetime
import numpy as np


@dataclass
class BulkOperationResult:
    """Result of a bulk operation."""
    race_id: str
    success: bool
    message: str
    data: Optional[Dict] = None


class BulkOperationsManager:
    """Manages bulk operations on multiple races."""

    def __init__(self, config_manager, metadata_manager=None):
        """
        Initialize bulk operations manager.

        Args:
            config_manager: ConfigManager instance
            metadata_manager: Optional MetadataManager instance
        """
        self.config = config_manager
        self.metadata = metadata_manager

    def export_multiple_races(
        self,
        video_entries: List,
        export_format: str,
        output_path: Path,
        include_metrics: bool = False
    ) -> Tuple[bool, str]:
        """
        Export multiple races to a single file.

        Args:
            video_entries: List of VideoEntry objects
            export_format: 'json', 'csv', 'yaml', or 'npz'
            output_path: Output file path
            include_metrics: Include performance metrics if available

        Returns:
            Tuple of (success, message)
        """
        try:
            if export_format == 'json':
                return self._export_json(video_entries, output_path, include_metrics)

            elif export_format == 'csv':
                return self._export_csv(video_entries, output_path, include_metrics)

            elif export_format == 'yaml':
                return self._export_yaml(video_entries, output_path, include_metrics)

            elif export_format == 'npz':
                return self._export_npz(video_entries, output_path, include_metrics)

            else:
                return False, f"Unsupported format: {export_format}"

        except Exception as e:
            return False, f"Export failed: {str(e)}"

    def _export_json(
        self,
        video_entries: List,
        output_path: Path,
        include_metrics: bool
    ) -> Tuple[bool, str]:
        """Export to JSON format."""
        data = []

        for entry in video_entries:
            race_data = entry.to_dict()

            # Add full metadata if available
            if entry.metadata_path.exists():
                with open(entry.metadata_path, 'r', encoding='utf-8') as f:
                    full_metadata = json.load(f)
                    race_data['metadata'] = full_metadata

            # Add metrics if requested and available
            if include_metrics:
                metrics_path = self._get_metrics_path(entry)
                if metrics_path and metrics_path.exists():
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                        race_data['metrics'] = metrics

            data.append(race_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True, f"Exported {len(video_entries)} races to {output_path.name}"

    def _export_csv(
        self,
        video_entries: List,
        output_path: Path,
        include_metrics: bool
    ) -> Tuple[bool, str]:
        """Export to CSV format."""
        if not video_entries:
            return False, "No races to export"

        # Prepare headers
        headers = ['race_id', 'competition', 'duration', 'status', 'review_date', 'athletes', 'notes']

        if include_metrics:
            # Add common metric fields
            headers.extend([
                'avg_velocity', 'max_velocity', 'path_efficiency',
                'is_calibrated', 'units'
            ])

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for entry in video_entries:
                row = entry.to_dict()

                # Add metrics if requested
                if include_metrics:
                    metrics_path = self._get_metrics_path(entry)
                    if metrics_path and metrics_path.exists():
                        with open(metrics_path, 'r', encoding='utf-8') as mf:
                            metrics = json.load(mf)
                            row['avg_velocity'] = metrics.get('avg_vertical_velocity', '')
                            row['max_velocity'] = metrics.get('max_vertical_velocity', '')
                            row['path_efficiency'] = metrics.get('path_efficiency', '')
                            row['is_calibrated'] = metrics.get('is_calibrated', False)
                            row['units'] = metrics.get('units', '')

                writer.writerow(row)

        return True, f"Exported {len(video_entries)} races to {output_path.name}"

    def _export_yaml(
        self,
        video_entries: List,
        output_path: Path,
        include_metrics: bool
    ) -> Tuple[bool, str]:
        """Export to YAML format."""
        try:
            import yaml
        except ImportError:
            return False, "PyYAML not installed. Install with: pip install pyyaml"

        data = []

        for entry in video_entries:
            race_data = entry.to_dict()

            if include_metrics:
                metrics_path = self._get_metrics_path(entry)
                if metrics_path and metrics_path.exists():
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                        race_data['metrics'] = metrics

            data.append(race_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

        return True, f"Exported {len(video_entries)} races to {output_path.name}"

    def _export_npz(
        self,
        video_entries: List,
        output_path: Path,
        include_metrics: bool
    ) -> Tuple[bool, str]:
        """
        Export to NumPy NPZ format (ML-ready).

        Creates arrays for features and labels suitable for machine learning.
        """
        features = []
        labels = []
        race_ids = []
        metadata = []

        for entry in video_entries:
            metrics_path = self._get_metrics_path(entry)

            if not metrics_path or not metrics_path.exists():
                continue  # Skip races without metrics

            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)

            # Extract features (standardized)
            feature_vector = [
                metrics_data.get('avg_vertical_velocity', 0),
                metrics_data.get('max_vertical_velocity', 0),
                metrics_data.get('path_efficiency', 0),
                metrics_data.get('avg_acceleration', 0),
                metrics_data.get('max_acceleration', 0),
                metrics_data.get('avg_jerk', 0),
                entry.duration,  # Race duration as feature
            ]

            features.append(feature_vector)
            labels.append(entry.duration)  # Duration as label for prediction
            race_ids.append(entry.race_id)

            # Metadata for reference
            metadata.append({
                'race_id': entry.race_id,
                'competition': entry.competition,
                'athletes': f"{entry.left_athlete} vs {entry.right_athlete}"
            })

        if not features:
            return False, "No races with metrics found for NPZ export"

        # Convert to numpy arrays
        features_array = np.array(features, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.float32)

        # Save as NPZ (compressed)
        np.savez_compressed(
            output_path,
            features=features_array,
            labels=labels_array,
            race_ids=race_ids,
            metadata=metadata,
            feature_names=[
                'avg_velocity', 'max_velocity', 'path_efficiency',
                'avg_acceleration', 'max_acceleration', 'avg_jerk',
                'duration'
            ]
        )

        return True, f"Exported {len(features)} races to {output_path.name} (ML-ready format)"

    def _get_metrics_path(self, entry) -> Optional[Path]:
        """Get path to metrics file for a race."""
        # Metrics are typically stored in data/processed/metrics/
        metrics_dir = self.config.project_root / "data" / "processed" / "metrics"

        if not metrics_dir.exists():
            return None

        metrics_file = metrics_dir / f"{entry.race_id}_metrics.json"

        return metrics_file if metrics_file.exists() else None

    def validate_multiple_races(
        self,
        video_entries: List,
        validator
    ) -> List[BulkOperationResult]:
        """
        Validate multiple races in batch.

        Args:
            video_entries: List of VideoEntry objects
            validator: RaceValidator instance

        Returns:
            List of validation results
        """
        results = []

        for entry in video_entries:
            try:
                # Load metadata
                if not entry.metadata_path.exists():
                    results.append(BulkOperationResult(
                        race_id=entry.race_id,
                        success=False,
                        message="Metadata file not found"
                    ))
                    continue

                with open(entry.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Validate
                is_valid, issues = validator.validate_race(metadata)

                results.append(BulkOperationResult(
                    race_id=entry.race_id,
                    success=is_valid,
                    message=f"{len(issues)} issues found" if issues else "Valid",
                    data={'issues': issues}
                ))

            except Exception as e:
                results.append(BulkOperationResult(
                    race_id=entry.race_id,
                    success=False,
                    message=f"Validation error: {str(e)}"
                ))

        return results

    def apply_correction_to_multiple(
        self,
        race_ids: List[str],
        correction_func: Callable,
        **correction_kwargs
    ) -> List[BulkOperationResult]:
        """
        Apply a correction function to multiple races.

        Args:
            race_ids: List of race IDs to process
            correction_func: Function to apply to each race
            correction_kwargs: Additional arguments for correction function

        Returns:
            List of BulkOperationResult objects
        """
        results = []

        for race_id in race_ids:
            try:
                # Apply correction
                success, message = correction_func(race_id, **correction_kwargs)
                results.append(BulkOperationResult(
                    race_id=race_id,
                    success=success,
                    message=message
                ))
            except Exception as e:
                results.append(BulkOperationResult(
                    race_id=race_id,
                    success=False,
                    message=f"Error: {str(e)}"
                ))

        return results

    def generate_summary_report(
        self,
        video_entries: List,
        output_path: Path
    ) -> Tuple[bool, str]:
        """
        Generate a summary report for multiple races.

        Args:
            video_entries: List of VideoEntry objects
            output_path: Output path for report (JSON)

        Returns:
            Tuple of (success, message)
        """
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'total_races': len(video_entries),
                'summary': {
                    'by_competition': {},
                    'by_status': {},
                    'duration_stats': {}
                },
                'races': []
            }

            durations = []

            for entry in video_entries:
                # Count by competition
                comp = entry.competition
                report['summary']['by_competition'][comp] = \
                    report['summary']['by_competition'].get(comp, 0) + 1

                # Count by status
                status = entry.status
                report['summary']['by_status'][status] = \
                    report['summary']['by_status'].get(status, 0) + 1

                # Collect durations
                if entry.duration > 0:
                    durations.append(entry.duration)

                # Add race summary
                report['races'].append(entry.to_dict())

            # Duration statistics
            if durations:
                report['summary']['duration_stats'] = {
                    'min': min(durations),
                    'max': max(durations),
                    'mean': sum(durations) / len(durations),
                    'median': sorted(durations)[len(durations) // 2]
                }

            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            return True, f"Generated report for {len(video_entries)} races"

        except Exception as e:
            return False, f"Report generation failed: {str(e)}"


# Import at end to avoid circular dependency
from typing import Tuple
