"""
Manual Review Interface
=======================
A professional Streamlit-based interface for manually reviewing and correcting
race detection errors in speed climbing videos.

Components:
- config.py: Configuration manager (YAML-driven)
- progress.py: Progress tracker (CSV-based)
- metadata_manager.py: Race metadata CRUD operations
- video_player.py: Video playback with frame navigation
- validators.py: Validation engine
- app.py: Main Streamlit application

Version: 1.0
Author: Speed Climbing Performance Analysis Team
Date: 2025-11-16
"""

__version__ = "1.0.0"
__all__ = [
    "ConfigManager",
    "ProgressTracker",
    "RaceReviewStatus",
    "MetadataManager",
    "VideoPlayer",
    "RaceValidator",
]
