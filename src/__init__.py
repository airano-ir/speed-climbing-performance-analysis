"""
Speed Climbing Performance Analysis System
===========================================

A comprehensive system for analyzing speed climbing performance using:
- MediaPipe BlazePose for pose estimation
- NARX Neural Networks for pattern learning
- Fuzzy Logic for personalized feedback

Main Components:
    - phase1_pose_estimation: Video processing and keypoint extraction
    - phase2_features: Biomechanical feature engineering
    - models: NARX neural network implementations
    - fuzzy_logic: Performance evaluation system
    - visualization: Reporting and dashboards

Author: Speed Climbing Research Team
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Speed Climbing Research Team"

# Package-level imports
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw_videos"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROJECT_ROOT / "configs"
