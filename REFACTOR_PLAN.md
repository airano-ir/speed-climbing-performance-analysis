# Project Refactoring & Restructuring Plan

## 1. Current State Analysis
The current project structure reflects a phase-based evolution (`phase1_pose_estimation`, `phase2...`) rather than a domain-driven software architecture. This leads to:
- **Tight Coupling**: Logic is scattered across scripts and "phase" folders.
- **Naming Confusion**: `phase1_pose_estimation` contains logic for later phases (e.g., `global_map_processor`).
- **Difficulty in Testing**: Hard to import specific components without side effects.
- **Maintenance Overhead**: Hard to find where specific logic (like "Hold Detection") lives.

## 2. Proposed Architecture
We will transition to a standard Python package structure named `speed_climbing`.

```
root/
├── speed_climbing/             # Main Package
│   ├── __init__.py
│   ├── core/                   # Core types, config, and exceptions
│   │   ├── __init__.py
│   │   ├── settings.py         # Centralized configuration (IFSC rules, etc.)
│   │   ├── types.py            # Data classes (Race, Climber, FrameData)
│   │   └── exceptions.py
│   │
│   ├── vision/                 # Computer Vision components
│   │   ├── __init__.py
│   │   ├── pose.py             # BlazePose wrapper
│   │   ├── holds.py            # Hold detection (Red holds)
│   │   ├── lanes.py            # Lane detection/separation
│   │   └── calibration.py      # Homography & Camera calibration
│   │
│   ├── processing/             # Logic & Pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py         # GlobalMapVideoProcessor (The main orchestrator)
│   │   ├── tracking.py         # WorldCoordinateTracker
│   │   └── dropout.py          # Dropout/Fall detection
│   │
│   ├── analysis/               # Data Analysis & Metrics
│   │   ├── __init__.py
│   │   ├── metrics.py          # Velocity, distance calculations
│   │   └── time_series.py      # TimeSeriesBuilder
│   │
│   └── utils/                  # Helpers
│       ├── __init__.py
│       ├── video.py            # Video I/O
│       ├── io.py               # File operations
│       └── visualization.py    # Plotting & Video annotation
│
├── scripts/                    # Thin CLI wrappers
│   ├── process_race.py         # Single race processing
│   ├── batch_process.py        # Batch processing
│   └── test_calibration.py     # Test script for calibration
│
├── tests/                      # Unit and Integration tests
│   ├── test_vision.py
│   ├── test_pipeline.py
│   └── ...
│
└── configs/                    # YAML configurations
```

## 3. Migration Strategy (Step-by-Step)

### Step 1: Core Setup
- Create `speed_climbing/core`
- Move IFSC constants and basic configs there.

### Step 2: Vision Module
- Refactor `HoldDetector`, `DualLaneDetector`, `PeriodicCalibrator` from `src/phase1...` and `src/calibration` into `speed_climbing/vision`.
- **Improvement**: Make `HoldDetector` more robust (configurable thresholds) as per user feedback.

### Step 3: Processing Module
- Implement `WorldCoordinateTracker` and `DropoutHandler` in `speed_climbing/processing`.
- Move `GlobalMapVideoProcessor` logic here.

### Step 4: Analysis Module
- Move `performance_metrics.py` logic to `speed_climbing/analysis`.

### Step 5: Entry Points
- Create clean scripts in `scripts/` that use the new package.

## 4. Testing Plan with User Images
The user provided images (`uploaded_image_0...`, `uploaded_image_1...`) likely show the wall setup.
- **Objective**: Verify `HoldDetector` and `Calibration` on these static images before running full video pipelines.
- **Action**: Create a notebook or script `scripts/test_image_calibration.py` that:
    1. Loads the user image.
    2. Runs `HoldDetector`.
    3. Visualizes detected holds.
    4. Attempts to compute the homography matrix.
    5. Outputs a "Calibration Quality" report.

## 5. Next Steps
1. Approval of this plan.
2. Execute Step 1 & 2 (Core & Vision).
3. Run the "Image Test" to verify the foundation.
4. Proceed to Pipeline integration.
