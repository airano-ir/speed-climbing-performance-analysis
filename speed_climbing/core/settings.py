"""
Core settings and constants for Speed Climbing Performance Analysis.
"""

from pathlib import Path

# IFSC Speed Climbing Standards
IFSC_STANDARDS = {
    "WALL_HEIGHT_M": 15.0,
    "WALL_WIDTH_M": 3.0,
    "OVERHANG_DEGREES": 5.0,
    "HOLD_SPACING_MM": 125.0,
    "PANEL_SIZE_MM": 1500.0,
    "START_PAD_HEIGHT_M": 0.2,  # Reference point above ground
    "GRID_LETTERS": 'ABCDEFGHIJKLM'
}

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"

# Default Processing Parameters
DEFAULT_PROCESSING_CONFIG = {
    "fps": 30.0,
    "calibration_interval_frames": 15,
    "min_hold_confidence": 0.3,
    "min_calibration_confidence": 0.6,
    "dropout_timeout_frames": 30
}
