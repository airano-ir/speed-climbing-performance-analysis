"""
Phase Manager
==============
Manages multi-phase workflow and plugin architecture.

Enables switching between Phase 1 (Manual Review), Phase 2 (Pose Estimation),
Phase 3 (Metrics Analysis), and Phase 4 (ML Predictions + Real-time).

Features:
- Phase switching
- Plugin activation/deactivation
- Feature availability per phase
- Future-proof extensibility

⭐ Phase 1.5.1 Feature (2025-11-16)
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import yaml
from enum import Enum


class Phase(Enum):
    """Project phases."""
    PHASE1 = "phase1"  # Race Segmentation + Manual Review
    PHASE2 = "phase2"  # Pose Estimation
    PHASE3 = "phase3"  # Performance Metrics
    PHASE4 = "phase4"  # ML Predictions + Real-time


@dataclass
class PhaseConfig:
    """Configuration for a single phase."""
    phase: Phase
    enabled: bool
    name: str
    description: str
    plugins: List[str]
    features: List[str]


class PhaseManager:
    """Manages project phases and plugins."""

    def __init__(self, config_manager):
        """
        Initialize phase manager.

        Args:
            config_manager: ConfigManager instance
        """
        self.config = config_manager
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = self.project_root / "configs" / "manual_review_config.yaml"

        # Load phase configuration
        self.phases_config = self._load_phase_config()

        # Current phase (default to Phase 1)
        self.current_phase = Phase.PHASE1

    def _load_phase_config(self) -> Dict[Phase, PhaseConfig]:
        """
        Load phase configuration from YAML.

        Returns:
            Dictionary mapping Phase to PhaseConfig
        """
        # Default configuration if file doesn't exist or doesn't have phases section
        default_config = {
            Phase.PHASE1: PhaseConfig(
                phase=Phase.PHASE1,
                enabled=True,
                name="Phase 1: Manual Review",
                description="Race segmentation and manual metadata correction",
                plugins=["video_extraction", "manual_review", "bulk_operations"],
                features=[
                    "video_library",
                    "video_player",
                    "metadata_editor",
                    "video_extraction",
                    "bulk_export"
                ]
            ),
            Phase.PHASE2: PhaseConfig(
                phase=Phase.PHASE2,
                enabled=True,
                name="Phase 2: Pose Estimation",
                description="Extract athlete poses using MediaPipe BlazePose",
                plugins=["pose_extractor", "pose_visualizer"],
                features=[
                    "pose_extraction",
                    "pose_visualization",
                    "skeleton_overlay"
                ]
            ),
            Phase.PHASE3: PhaseConfig(
                phase=Phase.PHASE3,
                enabled=True,
                name="Phase 3: Performance Metrics",
                description="Calculate biomechanical performance metrics",
                plugins=["metrics_calculator", "metrics_visualizer"],
                features=[
                    "velocity_analysis",
                    "acceleration_analysis",
                    "path_efficiency",
                    "comparative_analysis"
                ]
            ),
            Phase.PHASE4: PhaseConfig(
                phase=Phase.PHASE4,
                enabled=False,  # Future phase
                name="Phase 4: ML Predictions + Real-time",
                description="CNN-Transformer predictions and real-time streaming",
                plugins=["ml_predictor", "real_time_stream", "graphql_api"],
                features=[
                    "cnn_transformer_model",
                    "physics_informed_validation",
                    "real_time_prediction",
                    "webrtc_streaming",
                    "graphql_subscriptions"
                ]
            )
        }

        # Try to load from config file
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                if config_data and 'phases' in config_data:
                    # Override with config from file
                    phases_data = config_data['phases']

                    for phase_enum in Phase:
                        phase_key = phase_enum.value
                        if phase_key in phases_data:
                            phase_info = phases_data[phase_key]
                            default_config[phase_enum].enabled = phase_info.get('enabled', True)
                            default_config[phase_enum].plugins = phase_info.get('plugins', [])

            except Exception as e:
                print(f"Warning: Could not load phase config from {self.config_path}: {e}")
                print("Using default phase configuration.")

        return default_config

    def get_current_phase(self) -> Phase:
        """
        Get current active phase.

        Returns:
            Current Phase enum
        """
        return self.current_phase

    def set_current_phase(self, phase: Phase) -> bool:
        """
        Switch to a different phase.

        Args:
            phase: Phase to switch to

        Returns:
            True if successful, False if phase not enabled
        """
        if not self.is_phase_enabled(phase):
            return False

        self.current_phase = phase
        return True

    def is_phase_enabled(self, phase: Phase) -> bool:
        """
        Check if phase is enabled.

        Args:
            phase: Phase to check

        Returns:
            True if enabled
        """
        return self.phases_config[phase].enabled

    def get_phase_config(self, phase: Phase) -> PhaseConfig:
        """
        Get configuration for a specific phase.

        Args:
            phase: Phase to get config for

        Returns:
            PhaseConfig object
        """
        return self.phases_config[phase]

    def get_current_phase_config(self) -> PhaseConfig:
        """
        Get configuration for current phase.

        Returns:
            PhaseConfig object
        """
        return self.phases_config[self.current_phase]

    def get_available_phases(self) -> List[PhaseConfig]:
        """
        Get list of all enabled phases.

        Returns:
            List of PhaseConfig objects for enabled phases
        """
        return [
            config for config in self.phases_config.values()
            if config.enabled
        ]

    def get_phase_features(self, phase: Optional[Phase] = None) -> List[str]:
        """
        Get list of features available in a phase.

        Args:
            phase: Phase to get features for (default: current phase)

        Returns:
            List of feature names
        """
        target_phase = phase if phase else self.current_phase
        return self.phases_config[target_phase].features

    def get_phase_plugins(self, phase: Optional[Phase] = None) -> List[str]:
        """
        Get list of plugins for a phase.

        Args:
            phase: Phase to get plugins for (default: current phase)

        Returns:
            List of plugin names
        """
        target_phase = phase if phase else self.current_phase
        return self.phases_config[target_phase].plugins

    def is_feature_available(self, feature_name: str) -> bool:
        """
        Check if a feature is available in current phase.

        Args:
            feature_name: Name of feature to check

        Returns:
            True if available in current phase
        """
        return feature_name in self.get_phase_features()

    def is_plugin_active(self, plugin_name: str) -> bool:
        """
        Check if a plugin is active in current phase.

        Args:
            plugin_name: Name of plugin to check

        Returns:
            True if active in current phase
        """
        return plugin_name in self.get_phase_plugins()

    def get_phase_summary(self) -> Dict[str, Any]:
        """
        Get summary of all phases.

        Returns:
            Dictionary with phase information
        """
        summary = {
            'current_phase': self.current_phase.value,
            'current_phase_name': self.get_current_phase_config().name,
            'phases': []
        }

        for phase_enum in Phase:
            config = self.phases_config[phase_enum]
            summary['phases'].append({
                'id': phase_enum.value,
                'name': config.name,
                'description': config.description,
                'enabled': config.enabled,
                'is_current': phase_enum == self.current_phase,
                'plugins_count': len(config.plugins),
                'features_count': len(config.features)
            })

        return summary

    def update_phase_config(
        self,
        phase: Phase,
        enabled: Optional[bool] = None,
        plugins: Optional[List[str]] = None
    ) -> bool:
        """
        Update phase configuration.

        Args:
            phase: Phase to update
            enabled: Set enabled state (optional)
            plugins: Set plugins list (optional)

        Returns:
            True if successful
        """
        try:
            if enabled is not None:
                self.phases_config[phase].enabled = enabled

            if plugins is not None:
                self.phases_config[phase].plugins = plugins

            # Save to config file
            self._save_phase_config()
            return True

        except Exception as e:
            print(f"Error updating phase config: {e}")
            return False

    def _save_phase_config(self):
        """Save current phase configuration to YAML file."""
        try:
            # Load existing config
            config_data = {}
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}

            # Update phases section
            config_data['phases'] = {}
            for phase_enum, config in self.phases_config.items():
                config_data['phases'][phase_enum.value] = {
                    'enabled': config.enabled,
                    'plugins': config.plugins
                }

            # Save back to file
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)

        except Exception as e:
            print(f"Warning: Could not save phase config: {e}")


class PluginRegistry:
    """Registry for managing plugins."""

    def __init__(self):
        """Initialize plugin registry."""
        self.plugins: Dict[str, Any] = {}

    def register_plugin(self, name: str, plugin_class: Any):
        """
        Register a plugin.

        Args:
            name: Plugin name
            plugin_class: Plugin class
        """
        self.plugins[name] = plugin_class

    def get_plugin(self, name: str) -> Optional[Any]:
        """
        Get plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin class or None
        """
        return self.plugins.get(name)

    def list_plugins(self) -> List[str]:
        """
        List all registered plugins.

        Returns:
            List of plugin names
        """
        return list(self.plugins.keys())
