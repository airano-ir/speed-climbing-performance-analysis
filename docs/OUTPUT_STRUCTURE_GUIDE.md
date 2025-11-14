# راهنمای ساختار خروجی‌ها و مسیرها
# Output Structure & Paths Guide

**نسخه**: 1.0
**تاریخ**: 2025-11-15
**زبان**: فارسی + English

---

## 📁 ساختار کلی (Overall Structure)

```
data/
├── raw_videos/                     # ویدئوهای اصلی دانلود شده
│   ├── *.mp4                      # ویدئوهای کوتاه social media (~6 MB)
│   ├── *.info.json                # اطلاعات کامل YouTube (58-548 KB) ✅
│   └── *_metadata.json            # metadata استخراج شده (< 1 KB) ✅
│
├── race_segments/                  # مسابقات استخراج شده (705 MB)
│   ├── seoul_2024/                # 31 races
│   ├── villars_2024/              # 24 races
│   ├── chamonix_2024/             # 32 races
│   ├── innsbruck_2024/            # 32 races
│   └── zilina_2025/               # 69 races
│       ├── Speed_finals_*_race###.mp4        # ویدئوی مسابقه (2-5 MB each)
│       ├── Speed_finals_*_race###_metadata.json  # اطلاعات ورزشکار
│       └── summary.json           # خلاصه competition
│
└── processed/                      # خروجی‌های پردازش
    ├── poses/                     # BlazePose keypoints (JSON)
    ├── metrics/                   # Performance metrics (CSV + JSON)
    ├── plots/                     # Visualizations (PNG)
    ├── videos/                    # Annotated videos (MP4)
    └── calibration/               # IFSC calibration data
```

---

## 🎯 خروجی‌های هر مرحله (Phase Outputs)

### Phase 1: Pose Extraction

**ورودی**: `data/race_segments/**/*.mp4` (188 files)
**خروجی**: `data/processed/poses/`

#### فرمت فایل:
```
data/processed/poses/
├── chamonix_2024/
│   ├── Speed_finals_Chamonix_2024_race001_poses.json  # ~2-6 MB
│   ├── Speed_finals_Chamonix_2024_race002_poses.json
│   └── ...
├── seoul_2024/
│   └── ...
└── _processing_summary.json       # Summary for all competitions
```

#### محتوای JSON:
```json
{
  "video_path": "data/race_segments/chamonix_2024/race001.mp4",
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "left_climber": {
        "has_detection": true,
        "overall_confidence": 0.83,
        "keypoints": {
          "nose": {"x": 0.373, "y": 0.634, "z": -0.100, "confidence": 0.998},
          "left_eye_inner": {...},
          ...  // 33 keypoints total
        }
      },
      "right_climber": {...}
    },
    ...
  ],
  "statistics": {
    "total_frames": 143,
    "detection_rate_left": 0.993,
    "detection_rate_right": 0.937
  }
}
```

**حجم کل**: ~940 MB برای 188 races (average 5 MB per race)

**برای GitHub**: فقط 5-10 sample files (~30-50 MB)

---

### Phase 2: Performance Metrics

**ورودی**: `data/processed/poses/**/*_poses.json`
**خروجی**: `data/processed/metrics/`

#### فرمت فایل:
```
data/processed/metrics/
├── chamonix_2024/
│   ├── race001_metrics_left.csv         # Time-series data
│   ├── race001_metrics_left.json        # Summary statistics
│   ├── race001_metrics_right.csv
│   ├── race001_metrics_right.json
│   └── ...
├── seoul_2024/
│   └── ...
└── aggregate_metrics.csv                 # All 188 races × 2 climbers
```

#### محتوای CSV (time-series):
```csv
timestamp,com_x,com_y,velocity_x,velocity_y,velocity_magnitude,acceleration_x,acceleration_y,acceleration_magnitude,jerk_x,jerk_y,jerk_magnitude
0.0,0.373,0.634,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
0.033,0.374,0.632,0.03,-0.06,0.067,0.9,1.8,2.01,27.0,54.0,60.1
...
```

#### محتوای JSON (summary):
```json
{
  "summary": {
    "avg_vertical_velocity": 0.523,      // meters/second (after calibration)
    "max_vertical_velocity": 2.15,
    "avg_acceleration": 0.88,
    "max_acceleration": 4.00,
    "path_length": 15.2,                 // meters
    "straight_distance": 15.0,           // meters (wall height)
    "path_efficiency": 0.987,            // 98.7% efficient
    "smoothness_score": 12.83,           // lower = smoother
    "total_time": 5.71                   // seconds
  },
  "athlete": {
    "name": "Aleksandra Mirosław",
    "country": "POL",
    "bib_color": "red"
  }
}
```

**حجم کل**: ~50 MB برای 188 races

**برای GitHub**: aggregate_metrics.csv (~2 MB) + sample files

---

### Phase 3: Visualization

**ورودی**: `data/processed/metrics/`
**خروجی**: `data/processed/plots/`

#### فرمت فایل:
```
data/processed/plots/
├── chamonix_2024/
│   ├── race001_single_left.png           # Single climber dashboard
│   ├── race001_single_right.png
│   ├── race001_dual_comparison.png       # Side-by-side comparison
│   └── ...
├── seoul_2024/
│   └── ...
└── comparisons/
    ├── top10_athletes.png                # Top 10 fastest
    ├── competition_winners.png           # All 5 competitions
    ├── velocity_distribution.png         # Statistical analysis
    └── path_efficiency_ranking.png
```

#### نوع نمودارها:

**Single Climber Dashboard** (4 subplots):
1. Trajectory (height vs time)
2. Velocity profile
3. Acceleration profile
4. Horizontal deviation

**Dual Comparison**:
1. Overlayed trajectories (left vs right)
2. Velocity comparison
3. Summary bar chart (time, efficiency, smoothness)

**حجم کل**: ~100 MB برای 188 races × 3 plots

**برای GitHub**: comparisons/ folder (~5 MB) + samples

---

### Phase 4: IFSC Calibration

**ورودی**:
- `docs/IFSC_Speed_Licence_Rules.pdf`
- `data/race_segments/**/*.mp4` (first frame)

**خروجی**: `data/processed/calibration/`

#### فرمت فایل:
```
data/processed/calibration/
├── ifsc_route_coordinates.json          # 20 holds × (panel, x, y, meters)
├── chamonix_2024/
│   ├── race001_calibration.json        # Homography matrix + params
│   ├── race001_detected_holds.json     # Which holds visible
│   └── ...
└── camera_types.json                    # static vs moving per competition
```

#### محتوای calibration.json:
```json
{
  "video_path": "data/race_segments/chamonix_2024/race001.mp4",
  "camera_type": "static",
  "homography_matrix": [[...], [...], [...]],  // 3×3
  "pixel_to_meter_scale": 0.012,               // 1 pixel = 0.012 meters
  "detected_holds": [
    {"hold_num": 1, "pixel_x": 123, "pixel_y": 456, "confidence": 0.95},
    {"hold_num": 5, "pixel_x": 234, "pixel_y": 345, "confidence": 0.89},
    ...
  ],
  "calibration_error": 0.03                    // meters (RMSE)
}
```

**حجم کل**: ~10 MB برای 188 races

**برای GitHub**: همه فایل‌ها (کوچک)

---

### Phase 5: Video Annotation (Optional)

**ورودی**:
- `data/race_segments/**/*.mp4`
- `data/processed/poses/**/*_poses.json`

**خروجی**: `data/processed/videos/`

#### فرمت فایل:
```
data/processed/videos/
├── chamonix_2024/
│   ├── race001_annotated.mp4           # با BlazePose overlay (5-10 MB)
│   └── ...
└── samples/                             # برای GitHub
    ├── sample_001_annotated.mp4        # Top 5 athletes
    ├── sample_002_annotated.mp4
    └── ...
```

**محتوای ویدئو**:
- BlazePose skeleton overlay (33 keypoints)
- COM (Center of Mass) trajectory
- Velocity/acceleration text
- Hold markers (if calibrated)
- Lane boundaries

**حجم کل**: ~1-2 GB برای 188 races (اختیاری)

**برای GitHub**: فقط samples/ (~50 MB)

---

## 📊 خلاصه حجم فایل‌ها (Storage Summary)

### Local (Google Drive) - کامل:
```
data/raw_videos/          6 MB      # Social media clips + metadata
data/race_segments/     705 MB      # 188 race MP4s
data/processed/
  ├── poses/            940 MB      # 188 JSON files
  ├── metrics/           50 MB      # CSV + JSON
  ├── plots/            100 MB      # PNG charts
  ├── videos/        ~1,500 MB      # Annotated MP4s (optional)
  └── calibration/       10 MB      # Homography data

TOTAL:              ~3,311 MB (3.2 GB)
```

### GitHub (Public) - انتخابی:
```
Code + Configs            ~5 MB     # All source code
data/raw_videos/         ~2 MB     # info.json + metadata only
data/processed/
  ├── poses/             30 MB     # 5-10 samples
  ├── metrics/           10 MB     # aggregate + samples
  ├── plots/             10 MB     # comparisons + samples
  ├── videos/samples/    50 MB     # Top 5 annotated
  └── calibration/       10 MB     # All (small files)

TOTAL:                 ~117 MB (well under GitHub limits)
```

### Gitea (Private) - کامل:
```
همه چیز از Local + Git history
```

---

## 🔄 Workflow برای کاربران آینده

### 1. Download Project

**از GitHub (برای UI claude.ai/code):**
```bash
git clone https://github.com/languageofearthcom-oss/Speed-Climbing-Performance-Analysis.git
cd Speed-Climbing-Performance-Analysis

# دسترسی به:
# - همه کد
# - 5-10 sample pose JSONs
# - aggregate metrics
# - visualizations
# - calibration data
```

**از Google Drive (برای توسعه کامل):**
```bash
# کپی کل پوشه G:\My Drive\Projects\Speed Climbing Performance Analysis
# شامل:
# - همه 188 race MP4s
# - همه pose JSONs
# - همه processed data
```

### 2. Run Analysis

**با sample data (GitHub):**
```bash
# تست سریع با samples
python src/analysis/performance_metrics.py \
  data/processed/poses/sample_001_poses.json --lane left

# Visualization
python src/visualization/time_series_plots.py \
  data/processed/poses/sample_001_poses.json --mode dual
```

**با full dataset (Local):**
```bash
# پردازش کامل 188 races
python scripts/batch_pose_extraction.py

# Aggregate metrics
python scripts/batch_metrics_calculation.py

# Generate all visualizations
python scripts/batch_visualization.py
```

### 3. Outputs Location

همه خروجی‌ها در `data/processed/` با ساختار واضح:
- **poses/**: به‌ازای هر race یک JSON
- **metrics/**: به‌ازای هر race × 2 climbers (CSV + JSON)
- **plots/**: به‌ازای هر race × 3 نمودار
- **calibration/**: به‌ازای هر race یک JSON (اگر calibration باشد)

---

## 🎓 آموزش استفاده برای مبتدیان

### مثال 1: تحلیل یک مسابقه

```python
from src.analysis.performance_metrics import PerformanceAnalyzer
from src.visualization.time_series_plots import TimeSeriesVisualizer

# Load pose data
analyzer = PerformanceAnalyzer()
metrics_left = analyzer.analyze_from_file(
    "data/processed/poses/chamonix_2024/race001_poses.json",
    lane="left"
)

# Show summary
print(f"Avg velocity: {metrics_left.avg_vertical_velocity:.2f} m/s")
print(f"Total time: {metrics_left.timestamps[-1]:.2f} s")
print(f"Path efficiency: {metrics_left.path_efficiency:.1%}")

# Visualize
viz = TimeSeriesVisualizer()
viz.plot_single_climber(
    "data/processed/poses/chamonix_2024/race001_poses.json",
    lane="left",
    output="my_analysis.png"
)
```

### مثال 2: مقایسه دو climber

```python
# Load both climbers
metrics_left = analyzer.analyze_from_file(pose_file, lane="left")
metrics_right = analyzer.analyze_from_file(pose_file, lane="right")

# Compare
if metrics_left.avg_vertical_velocity > metrics_right.avg_vertical_velocity:
    print("Left climber was faster!")
else:
    print("Right climber was faster!")

# Dual visualization
viz.plot_dual_comparison(pose_file, output="comparison.png")
```

### مثال 3: تحلیل آماری کل dataset

```python
import pandas as pd

# Load aggregate metrics
df = pd.read_csv("data/processed/metrics/aggregate_metrics.csv")

# Top 10 fastest
top10 = df.nlargest(10, 'avg_vertical_velocity')
print(top10[['athlete_name', 'country', 'avg_vertical_velocity', 'total_time']])

# Statistical summary
print(df['avg_vertical_velocity'].describe())

# Plot distribution
import matplotlib.pyplot as plt
df['avg_vertical_velocity'].hist(bins=20)
plt.xlabel('Average Vertical Velocity (m/s)')
plt.ylabel('Frequency')
plt.title('Distribution of Climbing Speeds')
plt.savefig('velocity_distribution.png')
```

---

## ⚙️ پیکربندی‌های پیشنهادی

### برای توسعه (Local):
```python
# config.py
PATHS = {
    'race_segments': 'data/race_segments',
    'poses': 'data/processed/poses',
    'metrics': 'data/processed/metrics',
    'plots': 'data/processed/plots',
    'videos': 'data/processed/videos',
    'calibration': 'data/processed/calibration',
}

# Process all 188 races
PROCESS_ALL = True
MAX_RACES = None  # No limit

# Video annotation
GENERATE_ANNOTATED_VIDEOS = True  # ~15-30 hours
```

### برای تست سریع (CI/CD):
```python
# config.py
PATHS = {
    'poses': 'data/processed/poses',  # Use samples
    ...
}

# Process only samples
PROCESS_ALL = False
MAX_RACES = 5  # Quick test

# Skip video generation
GENERATE_ANNOTATED_VIDEOS = False
```

---

## 🐛 عیب‌یابی (Troubleshooting)

### خطا: "File not found"
```bash
# بررسی که فایل وجود دارد
ls data/processed/poses/chamonix_2024/race001_poses.json

# اگر نیست، اول pose extraction را اجرا کنید
python scripts/batch_pose_extraction.py --max-races 1 --competition chamonix_2024
```

### خطا: "No module named 'src.analysis'"
```bash
# اضافه کردن src به PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows CMD
$env:PYTHONPATH += ";$(Get-Location)"    # Windows PowerShell
```

### خطا: "Memory error" در batch processing
```bash
# کاهش تعداد همزمان
python scripts/batch_pose_extraction.py --max-races 10  # Process 10 at a time

# یا تک‌تک
for i in {1..188}; do
  python scripts/batch_pose_extraction.py --max-races 1 --skip $((i-1))
done
```

---

## 📚 منابع بیشتر

- **کد**: [src/](../src/)
- **تست‌ها**: [tests/](../tests/)
- **مستندات**: [docs/](../docs/)
- **راهنمای Segmentation**: [MANUAL_SEGMENTATION_GUIDE.md](MANUAL_SEGMENTATION_GUIDE.md)
- **راهنمای Sync**: [SYNC_WORKFLOW.md](../SYNC_WORKFLOW.md)
- **Master Context**: [MASTER_CONTEXT.md](../MASTER_CONTEXT.md)

---

**آخرین به‌روزرسانی**: 2025-11-15
**نگهدارنده**: Speed Climbing Performance Analysis Team

---

**END OF OUTPUT STRUCTURE GUIDE**
