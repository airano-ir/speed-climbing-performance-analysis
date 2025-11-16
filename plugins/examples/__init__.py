"""
Example Plugins
===============
Demonstration plugins for Phase 4 and future features.
"""

from .ml_predictor import MLPredictorPlugin, create_plugin as create_ml_predictor

__all__ = [
    'MLPredictorPlugin',
    'create_ml_predictor'
]
