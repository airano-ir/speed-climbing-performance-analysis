"""
Configuration Manager
====================
Loads and manages manual_review_config.yaml

This module provides config-driven architecture for the manual review interface,
allowing new competitions to be added without code changes.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CompetitionConfig:
    """Competition configuration data structure."""
    key: str
    name: str
    date: str
    video_format: str
    fps: float
    race_segments_path: Path
    total_races: int
    notes: str = ""


class ConfigManager:
    """Manages configuration for manual review interface."""

    def __init__(self, config_path: str = "configs/manual_review_config.yaml"):
        """
        Initialize config manager.

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """
        Load YAML config file.

        Returns:
            Parsed config dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_competitions(self) -> List[CompetitionConfig]:
        """
        Get list of all competitions.

        Returns:
            List of CompetitionConfig objects
        """
        competitions = []

        if 'competitions' not in self.config:
            return competitions

        for key, data in self.config['competitions'].items():
            # Skip comments and templates
            if key.startswith('_') or 'template' in key.lower():
                continue

            comp = CompetitionConfig(
                key=key,
                name=data['name'],
                date=data.get('date', 'Unknown'),
                video_format=data.get('video_format', 'MP4'),
                fps=data.get('fps', self.config['general']['default_fps']),
                race_segments_path=Path(data['race_segments_path']),
                total_races=data.get('total_races', 0),
                notes=data.get('notes', '')
            )
            competitions.append(comp)

        return competitions

    def get_competition(self, key: str) -> Optional[CompetitionConfig]:
        """
        Get specific competition by key.

        Args:
            key: Competition key (e.g., "chamonix_2024")

        Returns:
            CompetitionConfig or None if not found
        """
        comps = self.get_competitions()
        for comp in comps:
            if comp.key == key:
                return comp
        return None

    def get_validation_rules(self) -> Dict:
        """
        Get validation rules.

        Returns:
            Validation rules dictionary
        """
        return self.config.get('validation', {})

    def get_general_settings(self) -> Dict:
        """
        Get general settings.

        Returns:
            General settings dictionary
        """
        return self.config.get('general', {})

    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled.

        Args:
            feature: Feature name

        Returns:
            True if enabled, False otherwise
        """
        return self.config.get('features', {}).get(feature, False)

    def get_video_player_settings(self) -> Dict:
        """
        Get video player settings.

        Returns:
            Video player settings dictionary
        """
        return self.config.get('general', {}).get('video_player', {})
