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

# Athlete-Centric Pipeline Settings
ATHLETE_CENTRIC_CONFIG = {
    # COM height when athlete is standing on start pad (waist height)
    "com_standing_height_min_m": 0.8,
    "com_standing_height_max_m": 1.2,

    # Movement thresholds
    "start_movement_threshold_m": 0.15,  # Minimum upward movement to detect start
    "finish_height_threshold_m": 14.5,   # Height to consider finish
    "fall_velocity_threshold_m_s": -2.0, # Sudden downward velocity indicates fall

    # Camera view estimation (typical broadcast)
    "camera_view_height_m": 5.0,         # Approximate height visible in frame
    "camera_follows_athlete": True,       # Camera tracks athlete vertically

    # Race segment detection
    "pre_race_min_frames": 30,           # Minimum frames before race for calibration
    "post_race_frames": 30,              # Frames to include after finish
    "stillness_threshold": 0.02,         # Normalized movement threshold for "still"
    "stillness_min_frames": 15,          # Frames of stillness to detect waiting

    # Scale estimation
    "athlete_reference_heights": {
        "male_average_m": 1.77,          # Average male speed climber height
        "female_average_m": 1.63,        # Average female speed climber height
        "com_to_height_ratio": 0.55,     # COM is roughly at 55% of body height
    },

    # Dual lane settings
    "lane_boundary_default": 0.5,        # Default boundary at frame center
    "lane_names": ["left", "right"],
}
