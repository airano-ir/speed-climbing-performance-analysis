# MASTER CONTEXT: Speed Climbing Performance Analysis

## Project Overview
**Goal**: Build an ML-ready dataset from speed climbing videos for training neural networks that can analyze athlete technique and provide coaching insights.

**Current Status**: **Phase 3 - ML Feature Extraction (Active)**
Pivoted from absolute wall calibration (which had ~50% success rate) to relative feature extraction that works on ALL videos.

## Strategic Pivot (2025-11-28)

### Previous Approach (Abandoned)
- Hold-based homography calibration
- Absolute wall positioning (meters)
- ~50% success rate due to:
  - Moving cameras following athletes
  - Athletes occluding holds during climbing
  - Variable video quality across competitions

### New Approach (Current)
- **Relative feature extraction** independent of wall calibration
- Extract ML-ready features from normalized pose data
- Works on 95%+ of videos
- Focus on technique patterns, not absolute positioning

## Architecture

### `speed_climbing` Package
```
speed_climbing/
├── core/
│   └── settings.py           # IFSC standards, config
├── vision/
│   ├── holds.py              # HoldDetector (legacy, optional)
│   ├── lanes.py              # DualLaneDetector
│   ├── pose.py               # BlazePoseExtractor (33 keypoints + COM)
│   └── calibration.py        # CameraCalibrator (legacy, optional)
├── processing/
│   ├── athlete_centric.py    # Athlete-centric pipeline
│   ├── tracking.py           # WorldCoordinateTracker (legacy)
│   └── pipeline.py           # GlobalMapVideoProcessor (legacy)
└── analysis/
    ├── features/             # NEW: ML feature extraction
    │   ├── base.py           # Utility functions
    │   ├── frequency.py      # FFT-based rhythm analysis
    │   ├── efficiency.py     # Path efficiency metrics
    │   ├── posture.py        # Joint angle analysis
    │   └── extractor.py      # Main FeatureExtractor class
    ├── time_series.py        # Time series builder
    └── start_finish_detector.py
```

## Feature Extraction (New Pipeline)

### Features Extracted (23 total)

**Frequency Features (6):**
- `hand_frequency_hz`: Dominant hand movement frequency
- `foot_frequency_hz`: Dominant foot movement frequency
- `limb_sync_ratio`: Hand-foot coordination (0-1)
- `movement_regularity`: Rhythm consistency (0-1)
- `hand_movement_amplitude`: Hand movement range
- `foot_movement_amplitude`: Foot movement range

**Efficiency Features (6):**
- `path_straightness`: Direct path / actual path ratio
- `lateral_movement_ratio`: Sideways vs vertical movement
- `vertical_progress_rate`: Climbing speed (normalized)
- `com_stability_index`: COM trajectory smoothness
- `movement_smoothness`: Spectral arc length metric
- `acceleration_variance`: Movement consistency

**Posture Features (9):**
- `avg_knee_angle`: Average knee bend during climb
- `knee_angle_std`: Knee angle variation
- `avg_elbow_angle`: Average elbow angle
- `elbow_angle_std`: Elbow angle variation
- `hip_width_ratio`: Body width ratio (stability)
- `avg_body_lean`: Torso angle from vertical
- `body_lean_std`: Body lean variation
- `avg_reach_ratio`: Reach relative to body
- `max_reach_ratio`: Maximum reach achieved

### Usage

```python
from speed_climbing.analysis.features import FeatureExtractor, save_features_csv

extractor = FeatureExtractor(fps=30.0)
results = extractor.extract_from_file('pose_data.json')
save_features_csv(results, 'features.csv')
```

## Data Available

- **96 races** from Chamonix, Innsbruck, Seoul 2024
- **Pose files**: `data/processed/poses/{competition}/*.json`
- **Feature output**: `data/ml_dataset/`

## Recent Updates

### 2025-11-28
- **Major Pivot**: Abandoned absolute calibration approach
- **New Pipeline**: Created ML feature extraction system
- **Cleanup**: Removed 40 obsolete files (8799 lines)
- **Verified**: Feature extraction working on all test videos

### Previous (2025-11-20 to 2025-11-27)
- Attempted camera motion compensation
- Implemented hold-based anchoring
- Improved race start detection
- Added dual-lane pose detection

## Next Steps

1. **Batch Feature Extraction**: Run on all 96 races
2. **ML Dataset Preparation**: Create train/val/test splits
3. **Feature Validation**: Check feature distributions
4. **Model Training**: Train initial technique classifier

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_feature_extraction.py` | Test feature extraction |
| `scripts/run_athlete_centric_pipeline.py` | Full pipeline (optional) |
| `scripts/run_new_pipeline.py` | Legacy calibration pipeline |

## Contact

For questions about this project, see the original `prompt.md` for initial requirements.
