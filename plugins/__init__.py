"""
Plugins Package
===============
Plugin system for Speed Climbing Performance Analysis.

Provides extensible architecture for adding features across all project phases.
"""

from .base import (
    PluginBase,
    UIPlugin,
    DataProcessingPlugin,
    AnalysisPlugin,
    PredictionPlugin,
    PluginMetadata,
    PluginStatus
)

__all__ = [
    'PluginBase',
    'UIPlugin',
    'DataProcessingPlugin',
    'AnalysisPlugin',
    'PredictionPlugin',
    'PluginMetadata',
    'PluginStatus'
]
