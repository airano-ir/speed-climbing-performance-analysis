# MASTER CONTEXT: Speed Climbing Performance Analysis

## Project Overview
**Goal**: Build an AI-powered system that analyzes speed climbing videos and provides **personalized feedback** to athletes, coaches, and enthusiasts.

**Current Status**: **Phase 4 - Fuzzy Logic Feedback System (Active)**
Implemented a Fuzzy Logic based feedback system that generates personalized coaching insights from extracted features.

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
- 22 ML-ready features
- Automatic race segment detection
- Quality reports and validation

### Phase 4: Fuzzy Feedback System ✅ (NEW - 2025-11-28)
- Fuzzy Logic engine for interpretable analysis
- Personalized feedback in Persian/English
- Comparison with professional athletes

## Architecture

### `speed_climbing` Package
```
speed_climbing/
├── core/
│   └── settings.py           # IFSC standards, config
├── vision/
│   ├── holds.py              # HoldDetector (optional)
│   ├── lanes.py              # DualLaneDetector
│   ├── pose.py               # BlazePoseExtractor (33 keypoints + COM)
│   └── calibration.py        # CameraCalibrator (optional)
├── processing/
│   ├── athlete_centric.py    # Main pipeline (relative features)
│   └── dropout.py            # Dropout handling
└── analysis/
    ├── features/             # ML feature extraction
    │   ├── base.py           # Utility functions
    │   ├── frequency.py      # FFT-based rhythm analysis
    │   ├── efficiency.py     # Path efficiency metrics
    │   ├── posture.py        # Joint angle analysis
    │   ├── race_detector.py  # Variance-based race detection
    │   └── extractor.py      # Main FeatureExtractor class
    ├── feedback/             # NEW: Fuzzy Logic Feedback System
    │   ├── baseline.py       # Professional athlete statistics
    │   ├── fuzzy_engine.py   # Fuzzy membership & evaluation
    │   └── feedback_generator.py  # Persian/English report generation
    ├── time_series.py
    └── start_finish_detector.py
```

## Fuzzy Feedback System (NEW)

### How It Works

```
Video → Pose Extraction → Feature Extraction → Fuzzy Logic → Personalized Feedback
                                                    ↑
                                            Baseline from 371
                                            professional races
```

### Performance Categories (5)

| Category | Persian | Features Used |
|----------|---------|---------------|
| Rhythm & Coordination | ریتم و هماهنگی | hand/foot frequency, sync ratio |
| Movement Efficiency | کارایی حرکت | path straightness, lateral movement |
| Balance & Stability | تعادل و ثبات | COM stability, body lean |
| Body Posture | وضعیت بدن | knee/elbow angles, hip width |
| Reach & Extension | دسترسی و کشش | reach ratio, amplitude |

### Output Format

```
==================================================
📊 گزارش تحلیل عملکرد صخره‌نوردی سرعت
==================================================

امتیاز کلی شما: 65 از ۱۰۰
سطح: متوسط

💪 نقاط قوت:
  ✓ وضعیت بهینه بدن
  ✓ خم شدن مناسب زانو برای قدرت

⚠️ فرصت‌های بهبود:
  🟡 مسیر صعود به اندازه کافی مستقیم نیست

📈 امتیاز دسته‌ها:
  ریتم و هماهنگی: ██████░░░░ 62
  کارایی حرکت: ████░░░░░░ 44
  ...

🎯 توصیه‌های تمرینی:
  1. کوتاه‌ترین مسیر را قبل از شروع تجسم کنید

📊 مقایسه با حرفه‌ای‌ها:
  شما بهتر از 65٪ ورزشکاران در دیتاست ما عمل کرده‌اید.
==================================================
```

### Usage

```python
# Analyze a pose file and get feedback
python scripts/analyze_video.py pose_file.json --language fa

# Python API
from speed_climbing.analysis.feedback import FeedbackGenerator
from speed_climbing.analysis.feedback.feedback_generator import Language

generator = FeedbackGenerator(language=Language.PERSIAN)
feedback = generator.generate(features)
print(generator.format_report(feedback))
```

## Data Available

- **371 samples** from 5 competitions (Chamonix, Innsbruck, Seoul, Villars, Zilina)
- **246 high-quality** samples (extraction quality >= 0.8)
- **Pose files**: `data/processed/poses/samples/*.json`
- **ML dataset**: `data/ml_dataset/`

## Features Extracted (22 total)

**Frequency Features (6):**
- `hand_frequency_hz`, `foot_frequency_hz`
- `limb_sync_ratio`, `movement_regularity`
- `hand_movement_amplitude`, `foot_movement_amplitude`

**Efficiency Features (6):**
- `path_straightness`, `lateral_movement_ratio`
- `vertical_progress_rate`, `com_stability_index`
- `movement_smoothness`, `acceleration_variance`

**Posture Features (10):**
- `avg_knee_angle`, `knee_angle_std`
- `avg_elbow_angle`, `elbow_angle_std`
- `hip_width_ratio`, `avg_body_lean`, `body_lean_std`
- `avg_reach_ratio`, `max_reach_ratio`

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/analyze_video.py` | **Main analysis script** - generates feedback |
| `scripts/batch/batch_feature_extraction.py` | Batch processing |
| `scripts/batch/generate_report.py` | Dataset reports |

## Next Steps

1. ~~Batch Feature Extraction~~ ✅
2. ~~ML Dataset Preparation~~ ✅
3. ~~Fuzzy Feedback System~~ ✅
4. **Web Interface**: Upload video → Get feedback
5. **Video Processing Integration**: Full pipeline from raw video
6. **More Training Data**: Expand dataset with more competitions

## Recent Updates

### 2025-11-28 (Latest)
- **Fuzzy Feedback System**: Complete implementation
  - Baseline statistics from 371 professional races
  - 5 performance categories with weighted scoring
  - Bilingual output (Persian/English)
  - Personalized strengths, weaknesses, and recommendations
- **Analysis Script**: `scripts/analyze_video.py` for easy use
- **Bug Fixes**: Removed duplicate recommendations

### Previous
- Race segment detection (variance-based)
- Feature extraction pipeline
- Project cleanup and reorganization
