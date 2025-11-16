"""
Plugin Base Class
==================
Abstract base class for all plugins in the Speed Climbing Analysis system.

Enables extensibility through a plugin architecture where new features
can be added without modifying core code.

⭐ Phase 1.5.1 Feature (2025-11-16)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class PluginStatus(Enum):
    """Plugin status."""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Metadata about a plugin."""
    name: str
    version: str
    phase: str
    description: str
    author: str = "Speed Climbing Analysis Team"
    dependencies: List[str] = None
    config_required: bool = False

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class PluginBase(ABC):
    """
    Abstract base class for all plugins.

    All plugins must inherit from this class and implement the required methods.
    """

    def __init__(self):
        """Initialize plugin."""
        self.status = PluginStatus.UNINITIALIZED
        self.error_message: Optional[str] = None
        self.config: Dict[str, Any] = {}

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.

        Returns:
            PluginMetadata object with plugin information
        """
        pass

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize plugin with configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def validate_dependencies(self) -> tuple[bool, str]:
        """
        Validate that all required dependencies are available.

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    def set_config(self, config: Dict[str, Any]):
        """
        Set plugin configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set_status(self, status: PluginStatus, error_message: Optional[str] = None):
        """
        Set plugin status.

        Args:
            status: New status
            error_message: Error message if status is ERROR
        """
        self.status = status
        self.error_message = error_message

    def is_ready(self) -> bool:
        """
        Check if plugin is ready to use.

        Returns:
            True if status is READY
        """
        return self.status == PluginStatus.READY

    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin information.

        Returns:
            Dictionary with plugin info
        """
        meta = self.metadata
        return {
            'name': meta.name,
            'version': meta.version,
            'phase': meta.phase,
            'description': meta.description,
            'author': meta.author,
            'status': self.status.value,
            'error_message': self.error_message,
            'dependencies': meta.dependencies,
            'config_required': meta.config_required
        }


class UIPlugin(PluginBase):
    """
    Base class for plugins that provide UI components.

    Plugins that render UI in Streamlit should inherit from this class.
    """

    @abstractmethod
    def render_ui(self, st, context: Dict[str, Any]) -> None:
        """
        Render plugin UI in Streamlit.

        Args:
            st: Streamlit module
            context: Context dictionary with shared data
        """
        pass

    def render_sidebar(self, st, context: Dict[str, Any]) -> None:
        """
        Render plugin controls in sidebar (optional).

        Args:
            st: Streamlit module
            context: Context dictionary with shared data
        """
        pass

    def render_settings(self, st) -> Dict[str, Any]:
        """
        Render plugin settings UI (optional).

        Args:
            st: Streamlit module

        Returns:
            Dictionary with updated settings
        """
        return self.config


class DataProcessingPlugin(PluginBase):
    """
    Base class for plugins that process data.

    Plugins that perform data transformations should inherit from this class.
    """

    @abstractmethod
    def process_data(self, data: Any, **kwargs) -> tuple[bool, Any, Optional[str]]:
        """
        Process input data.

        Args:
            data: Input data to process
            kwargs: Additional processing parameters

        Returns:
            Tuple of (success, processed_data, error_message)
        """
        pass

    def validate_input(self, data: Any) -> tuple[bool, str]:
        """
        Validate input data (optional override).

        Args:
            data: Input data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, ""


class AnalysisPlugin(PluginBase):
    """
    Base class for plugins that perform analysis.

    Plugins that analyze race data should inherit from this class.
    """

    @abstractmethod
    def analyze(self, race_data: Dict[str, Any], **kwargs) -> tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Analyze race data.

        Args:
            race_data: Race data dictionary
            kwargs: Additional analysis parameters

        Returns:
            Tuple of (success, analysis_results, error_message)
        """
        pass

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate text report from analysis results (optional override).

        Args:
            analysis_results: Analysis results dictionary

        Returns:
            Formatted report string
        """
        return str(analysis_results)


class PredictionPlugin(PluginBase):
    """
    Base class for plugins that make predictions.

    ML models and prediction systems should inherit from this class.
    """

    @abstractmethod
    def predict(self, features: Any, **kwargs) -> tuple[bool, Any, Optional[str]]:
        """
        Make prediction from features.

        Args:
            features: Input features
            kwargs: Additional prediction parameters

        Returns:
            Tuple of (success, predictions, error_message)
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load prediction model from file.

        Args:
            model_path: Path to model file

        Returns:
            True if successful
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded model (optional override).

        Returns:
            Dictionary with model info
        """
        return {
            'model_loaded': False,
            'model_type': 'unknown'
        }
