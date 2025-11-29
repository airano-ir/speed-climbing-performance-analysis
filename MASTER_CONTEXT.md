# MASTER CONTEXT: Speed Climbing Performance Analysis

## Project Overview
**Goal**: Build an AI-powered system that analyzes speed climbing videos and provides **personalized feedback** to athletes, coaches, and enthusiasts.

**Current Status**: **Phase 4 - Fuzzy Logic Feedback System (Complete)**

## Project Journey

### Phase 1: Data Collection ✅
- Downloaded videos from IFSC competitions
- Cut videos into individual race clips
- 96+ videos from 5 competitions

### Phase 2: Pose Extraction ✅
- BlazePose for 33 body keypoints
- Dual-lane detection (left/right athletes)
- JSON storage for each video

### Phase 3: Feature Extraction ✅
- 22 ML-ready features extracted
- Automatic race segment detection
- Quality reports and validation

### Phase 4: Fuzzy Feedback System ✅
- Fuzzy Logic engine for interpretable analysis
- Personalized feedback in Persian/English
- Comparison with professional athletes
- **Camera-independent features only** (angles, ratios, sync)

---

## 🗺️ Future Roadmap

### Phase 5: Web Interface 📋 (Next)
**Goal**: Create a web app where users can upload videos and get feedback

| Task | Priority | Description |
|------|----------|-------------|
| Web UI | High | Simple upload page with results display |
| Charts & Graphs | High | Visual representation of scores |
| Video Player | Medium | Show video with analysis overlay |
| Export Report | Medium | PDF/Image export of feedback |

### Phase 6: Enhanced Visualization 📋
**Goal**: Add visual feedback overlaid on video

| Task | Priority | Description |
|------|----------|-------------|
| Keypoint Overlay | High | Draw skeleton on video frames |
| Angle Indicators | Medium | Show joint angles on video |
| Score Timeline | Medium | Graph of metrics over time |
| Comparison View | Low | Side-by-side with reference athlete |

### Phase 7: Single Athlete Detection 📋
**Goal**: Fix detection for single-athlete videos

| Task | Priority | Description |
|------|----------|-------------|
| Athlete Count Detection | High | Auto-detect 1 or 2 athletes |
| Lane Selection | Medium | Let user choose which lane |
| Better Lane Assignment | Medium | Improve left/right detection |

### Phase 8: Label Collection 📋
**Goal**: Collect labels for ML training

| Task | Priority | Description |
|------|----------|-------------|
| Time Annotation Tool | High | Mark start/finish for actual time |
| Skill Level Labels | Medium | Expert annotation of skill |
| Competition Results | Medium | Import win/lose data |
| Reference Technique | Low | Mark "good" vs "bad" examples |

### Phase 9: ML Model Training 📋
**Goal**: Train supervised ML models

| Task | Priority | Description |
|------|----------|-------------|
| Time Prediction | High | Predict finish time from technique |
| Skill Classification | Medium | Classify beginner/intermediate/advanced |
| Technique Clustering | Medium | Find similar climbing styles |
| Anomaly Detection | Low | Detect unusual movements |

### Phase 10: Advanced Position Tracking 📋
**Goal**: Solve camera-motion problem for absolute positioning

| Task | Priority | Description |
|------|----------|-------------|
| Wall Detection | High | Detect climbing wall in frame |
| Hold Tracking | High | Track holds despite occlusion |
| Camera Motion Estimation | High | Estimate camera pan/tilt/zoom |
| Absolute COM Position | Medium | Real position on wall |
| Actual Climbing Speed | Medium | Meters per second |
| Distance Traveled | Low | Total path length |

---

## Architecture

### `speed_climbing` Package
```
speed_climbing/
├── core/
│   └── settings.py           # IFSC standards, config
├── vision/
│   ├── holds.py              # HoldDetector
│   ├── lanes.py              # DualLaneDetector
│   ├── pose.py               # BlazePoseExtractor
│   └── calibration.py        # CameraCalibrator
├── processing/
│   ├── athlete_centric.py    # Main pipeline
│   └── dropout.py            # Dropout handling
└── analysis/
    ├── features/             # Feature extraction
    │   ├── frequency.py      # FFT-based rhythm
    │   ├── efficiency.py     # Path efficiency (⚠️ camera-dependent)
    │   ├── posture.py        # Joint angles
    │   └── extractor.py      # Main extractor
    └── feedback/             # Fuzzy feedback
        ├── baseline.py       # Pro athlete stats
        ├── fuzzy_engine.py   # Fuzzy logic
        └── feedback_generator.py  # Report generation
```

---

## Current Feedback System

### Valid Features (Camera-Independent)
```
✅ Used in Fuzzy System:
├── Joint Angles
│   ├── post_avg_knee_angle
│   ├── post_knee_angle_std
│   ├── post_avg_elbow_angle
│   └── post_elbow_angle_std
├── Body Position
│   ├── post_avg_body_lean
│   ├── post_body_lean_std
│   └── post_hip_width_ratio
├── Reach
│   ├── post_avg_reach_ratio
│   └── post_max_reach_ratio
└── Coordination
    ├── freq_limb_sync_ratio
    ├── freq_hand_movement_amplitude
    └── freq_foot_movement_amplitude
```

### Invalid Features (Camera Artifacts)
```
❌ NOT used (camera follows athlete):
├── eff_path_straightness
├── eff_com_stability_index
├── eff_lateral_movement_ratio
├── eff_movement_smoothness
├── eff_vertical_progress_rate
└── eff_acceleration_variance
```

### Performance Categories (5)
| Category | Persian | Features |
|----------|---------|----------|
| Coordination | هماهنگی اندام‌ها | limb_sync, amplitudes |
| Leg Technique | تکنیک پا | knee angles |
| Arm Technique | تکنیک دست | elbow angles |
| Body Position | وضعیت بدن | body lean, hip ratio |
| Reach | دسترسی و کشش | reach ratios |

---

## Known Issues

### Single Athlete Videos
- System always detects 2 lanes (left/right)
- For single-athlete videos, one lane may have invalid data
- **Workaround**: Use `--lane left` or `--lane right` flag

### Camera Motion
- Cannot measure actual climbing speed
- Cannot measure absolute position on wall
- **Requires**: Wall detection + camera motion compensation (Phase 10)

---

## Usage

```bash
# Analyze a pose file
python scripts/analyze_video.py pose_file.json --language fa --lane left

# Options:
#   --language fa|en    Output language
#   --lane left|right   Which lane to analyze
#   --output file.txt   Save report to file
```

---

## Data Available

- **371 samples** from 5 competitions
- **246 high-quality** samples (extraction quality >= 0.8)
- **Pose files**: `data/processed/poses/samples/*.json`
- **ML dataset**: `data/ml_dataset/`

---

## Recent Updates

### 2025-11-29 (Latest)
- **Camera-Independent Features**: Removed 6 invalid efficiency features
- **New Categories**: 5 technique-focused categories
- **Limitation Note**: Added note about camera motion in reports
- **Tested**: Confirmed working with Ola Miroslaw video

### 2025-11-28
- **Fuzzy Feedback System**: Complete implementation
- **Baseline Statistics**: From 371 professional races
- **Bilingual Output**: Persian/English support

### Previous
- Race segment detection (variance-based)
- Feature extraction pipeline
- Project cleanup
