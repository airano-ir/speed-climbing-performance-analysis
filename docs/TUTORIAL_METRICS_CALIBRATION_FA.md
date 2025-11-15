# راهنمای کامل: Calibration و Frame Selection در Metrics

**تاریخ**: 2025-11-15
**مخاطب**: توسعه‌دهندگان، محققان، مدل‌های AI
**سطح**: متوسط تا پیشرفته
**زمان مطالعه**: 30-45 دقیقه

---

## 📚 فهرست مطالب

1. [مقدمه](#مقدمه)
2. [بخش 1: چرا Calibration لازم است؟](#بخش-1-چرا-calibration-لازم-است)
3. [بخش 2: چرا Frame Selection مهم است؟](#بخش-2-چرا-frame-selection-مهم-است)
4. [بخش 3: معماری سیستم](#بخش-3-معماری-سیستم)
5. [بخش 4: پیاده‌سازی کامل](#بخش-4-پیاده‌سازی-کامل)
6. [بخش 5: تست و Validation](#بخش-5-تست-و-validation)
7. [بخش 6: عیب‌یابی](#بخش-6-عیب‌یابی)
8. [بخش 7: نکات پیشرفته](#بخش-7-نکات-پیشرفته)

---

## مقدمه

### هدف این راهنما

در تست Phase 3 پروژه تحلیل سنگنوردی سرعتی، دو مشکل بحرانی کشف شد که validity همه metrics را زیر سوال برد:

1. **عدم Calibration**: metrics در pixel محاسبه می‌شوند، نه meter
2. **عدم Frame Selection**: فریم‌های قبل/بعد مسابقه هم در محاسبات هستند

این راهنما به طور کامل توضیح می‌دهد:
- **چرا** این مشکلات بحرانی هستند
- **چگونه** باید رفع شوند
- **چطور** سیستم کار می‌کند

---

## بخش 1: چرا Calibration لازم است؟

### 1.1. مفهوم Calibration

**Calibration** یعنی تبدیل مختصات **pixel** (واحد تصویر) به مختصات **meter** (واحد فیزیکی).

```
WITHOUT Calibration:
Pixel (480, 320) → ??? meters

WITH Calibration:
Pixel (480, 320) → (1.2m, 8.5m) in real world
```

---

### 1.2. چرا در این پروژه CRITICAL است؟

#### مشکل 1: دوربین متحرک (Moving Camera)

**ویدئوهای IFSC**:
- دوربین سنگنورد را دنبال می‌کند (pan + zoom)
- در شروع مسابقه: دید wide (کل دیوار) → مقیاس کوچک
- در انتهای مسابقه: zoom روی climber → مقیاس بزرگ

**مثال واقعی**:
```
Frame 1 (start - wide view):
  Climber height in image: 100 pixels
  Actual height: 1.7m
  Scale: 1 pixel = 1.7cm

Frame 143 (finish - zoomed in):
  Climber height in image: 300 pixels
  Actual height: 1.7m (unchanged!)
  Scale: 1 pixel = 0.57cm (3× smaller!)
```

**تاثیر روی velocity**:
```python
# WITHOUT calibration (WRONG):
v_pixel = (y2 - y1) / dt
# Frame 1→50: v = 10 px/s
# Frame 100→143: v = 30 px/s (3× بیشتر به دلیل zoom!)
# میانگین: 20 px/s

# PROBLEM: این سرعت چقدر سریع است؟ نامعلوم!

# WITH calibration (CORRECT):
v_meter = calibration.pixel_to_meter(y2) - calibration.pixel_to_meter(y1)) / dt
# همه frames: v ≈ 2.3 m/s (ثابت - درست!)
```

**نتیجه**: **بدون calibration، نمی‌توان velocity معتبر محاسبه کرد!**

---

#### مشکل 2: عدم قابلیت مقایسه

**مقایسه بین races**:
```
Race A (Seoul):
  Camera distance: 20 meters
  Velocity: 25 px/s

Race B (Chamonix):
  Camera distance: 15 meters (نزدیک‌تر!)
  Velocity: 35 px/s

سوال: کدام سریع‌تر است؟
جواب بدون calibration: نامعلوم! 🤷

جواب با calibration:
  Race A: 2.1 m/s
  Race B: 2.3 m/s
  → Race B سریع‌تر است ✅
```

---

#### مشکل 3: تحلیل بیومکانیکی معتبر نیست

**Biomechanics** نیاز به واحدهای فیزیکی دارد:
```
Invalid (pixels):
  "avg_velocity = 12.07 px/s"
  → چقدر سریع؟ نمی‌دانیم!
  → نمی‌توان با مقالات علمی مقایسه کرد
  → نمی‌توان به ورزشکار توصیه داد

Valid (meters):
  "avg_velocity = 2.34 m/s"
  → World record: 2.67 m/s (5.0s for 15m)
  → You are 88% of world record! ✅
  → Recommendation: increase explosive power in middle section
```

---

### 1.3. راه‌حل: IFSC Calibration

**استفاده از گیره‌های استاندارد**:

IFSC دیواره سرعتی standard دارد:
- 15 متر ارتفاع
- 3 متر عرض
- **31 گیره قرمز** با موقعیت‌های دقیق شناخته شده

```python
# IFSC Route Map
holds = {
    1: {'x': 0.75m, 'y': 0.5m},
    2: {'x': 2.25m, 'y': 1.0m},
    # ...
    31: {'x': 1.5m, 'y': 14.5m}  # top hold
}
```

**فرآیند calibration**:
```
1. Detect holds in video frame (red color detection)
2. Match detected holds to known IFSC holds
3. Compute homography matrix (pixel ↔ meter)
4. Use homography to convert any pixel coordinate to meters
```

**مثال**:
```python
from calibration.camera_calibration import CameraCalibrator

# Load IFSC route map
route_map = IFSCRouteMap.load_from_json("configs/ifsc_route_coordinates.json")

# Detect holds in frame
holds_detected = hold_detector.detect(frame)

# Calibrate
calibrator = CameraCalibrator(route_map)
result = calibrator.calibrate(frame, holds_detected)

# Convert pixel to meter
pixel_coord = (480, 320)
meter_coord = result.pixel_to_meter_func(pixel_coord[0], pixel_coord[1])
# → (1.2m, 8.5m) ✅
```

---

### 1.4. چالش: دوربین متحرک

**ساده‌ترین calibration**: یک بار در اول ویدئو
```python
# Frame 1:
calibration = calibrate_frame(frame_1)

# Use for all frames:
for frame in video:
    meters = calibration.pixel_to_meter(pixels)  # ❌ WRONG for moving camera!
```

**چرا کار نمی‌کند؟**
- Frame 1: calibration برای wide view
- Frame 143: calibration برای zoomed view
- **دو calibration کاملاً متفاوت هستند!**

**راه‌حل: PeriodicCalibrator**
```python
# Recalibrate every 30 frames (1 second)
calibrator = PeriodicCalibrator(recalibration_interval=30)

for frame_id, frame in enumerate(video):
    # Auto-recalibrates every 30 frames
    calibration = calibrator.calibrate_frame(frame, frame_id)

    # Use calibration for this frame
    meters = calibration.pixel_to_meter(pixels)  # ✅ CORRECT!
```

**مزایا**:
- Adapts to camera movement
- Caches calibration (30× faster than per-frame)
- Temporal smoothing (reduces jitter)
- Automatic fallback if calibration fails

---

## بخش 2: چرا Frame Selection مهم است؟

### 2.1. ساختار ویدئوهای مسابقه

**هر race segment شامل 3 بخش است**:

```
┌────────────────────────────────────────────────┐
│         Race Segment Video (4.77s)             │
├────────────┬─────────────┬──────────────────────┤
│ Pre-race   │   RACE      │   Post-race          │
│ 1.5s       │   1.77s     │   1.5s               │
│ (45 frames)│  (53 frames)│  (45 frames)         │
│            │             │                      │
│ Standing   │ Climbing    │ Finished             │
│ Preparing  │ (THE DATA!) │ Celebrating          │
│ v ≈ 0      │ v = 2.5 m/s │ v ≈ 0                │
└────────────┴─────────────┴──────────────────────┘
```

**Buffer چرا اضافه شده؟**
- احتیاط برای late starts (بعضی مسابقات 1-2s تأخیر دارند)
- مطمئن شدن که شروع/پایان کامل capture شود

---

### 2.2. مشکل: محاسبه روی همه فریم‌ها

**فعلاً `performance_metrics.py` چیکار می‌کند؟**

```python
# Load pose file (all 143 frames)
frames = load_pose_json(pose_file)

# Process ALL frames:
for frame in frames:  # ❌ includes pre/post!
    com = calculate_center_of_mass(frame)
    velocity = calculate_velocity(com)
    # ...

avg_velocity = mean(all_velocities)
```

**نتیجه**:
```python
velocities = [
    0, 0, 0, ...,  # Pre-race (45 frames) - standing still
    2.5, 2.4, 2.6, ...,  # Race (53 frames) - climbing fast!
    0.1, 0, 0, ...  # Post-race (45 frames) - stopped
]

avg_velocity = mean(velocities)
             = (0*45 + 2.5*53 + 0*45) / 143
             = 132.5 / 143
             = 0.93 m/s  # ❌ 2.7× too low!

correct_avg = mean(velocities[45:98])  # race frames only
            = 2.5 m/s  # ✅
```

**خطا: 2.7× underestimation!**

---

### 2.3. تاثیر روی metrics دیگر

#### Path Length
```python
# WITH pre/post frames (WRONG):
path = [
    (0, 0), (0.1, 0.2), ...,  # pre-race movement (adjusting position)
    (0, 0.5), (0, 1.5), ...,  # race (climbing upward)
    (0.2, 15), (0.3, 15), ... # post-race (lateral movement)
]
path_length = sum(distances) = 18.5m  # ❌ too long

# WITHOUT pre/post (CORRECT):
path = [
    (0, 0.5), (0, 1.5), ...,  # race only
]
path_length = sum(distances) = 15.8m  # ✅ realistic
```

#### Path Efficiency
```python
# WRONG:
efficiency = straight_distance / path_length
           = 15.0 / 18.5
           = 0.81  # seems okay?

# BUT straight_distance is also wrong!
# It's from first frame (pre-race) to last frame (post-race)
# NOT from race start to race finish

# CORRECT:
straight_distance = 15.0m  # race start to finish
path_length = 15.8m  # actual climbing path
efficiency = 15.0 / 15.8 = 0.95  # ✅ excellent!
```

---

### 2.4. راه‌حل: فیلتر فریم‌ها

**استفاده از Race Boundaries**:

Race boundaries در metadata ذخیره شده‌اند:
```json
// data/race_segments/chamonix_2024/Speed_finals_Chamonix_2024_race001_metadata.json
{
  "detected_start_frame": 11784,  // شروع واقعی مسابقه
  "detected_finish_frame": 11837,  // پایان واقعی مسابقه
  "start_frame": 11739,  // شروع segment (با buffer)
  // ...
}
```

**تبدیل به pose file frame IDs**:
```python
# Pose file frame IDs start from 0
# Metadata frame IDs are from original video

segment_start = metadata['start_frame']  # 11739
race_start = metadata['detected_start_frame']  # 11784
race_end = metadata['detected_finish_frame']  # 11837

# Convert to pose file coordinates:
pose_race_start = race_start - segment_start  # 11784 - 11739 = 45 ✅
pose_race_end = race_end - segment_start  # 11837 - 11739 = 98 ✅
```

**استفاده در metrics**:
```python
for frame in frames:
    frame_id = frame['frame_id']

    # Filter by race boundaries
    if frame_id < pose_race_start:
        continue  # skip pre-race
    if frame_id > pose_race_end:
        break  # skip post-race

    # Process only race frames
    com = calculate_center_of_mass(frame)
    # ...
```

---

## بخش 3: معماری سیستم

### 3.1. نقشه کامل: از ویدئو تا Metrics

```
┌─────────────┐
│ Race Video  │ (MP4, 4.77s, 143 frames)
│ (segment)   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Pose Extraction │ (BlazePose)
│  (Phase 2)      │
└──────┬──────────┘
       │
       ▼
┌────────────────────┐
│ Pose JSON          │ (143 frames, 33 keypoints each)
│ + Metadata JSON    │ (race boundaries, athlete info)
└──────┬─────────────┘
       │
       ├──────────────────────────────────┐
       │                                  │
       ▼                                  ▼
┌─────────────────┐              ┌──────────────────┐
│ Hold Detection  │              │ Load Race        │
│ (HSV color)     │              │ Boundaries       │
└──────┬──────────┘              └────────┬─────────┘
       │                                  │
       ▼                                  │
┌──────────────────┐                     │
│ Camera           │                     │
│ Calibration      │                     │
│ (Homography)     │                     │
└──────┬───────────┘                     │
       │                                  │
       └────────────┬─────────────────────┘
                    │
                    ▼
            ┌───────────────────┐
            │ Performance       │
            │ Metrics           │
            │ Calculator        │
            └───────┬───────────┘
                    │
                    ▼
            ┌───────────────────┐
            │ Metrics JSON      │
            │ (calibrated,      │
            │  race-only)       │
            └───────────────────┘
```

---

### 3.2. کجا Calibration اعمال می‌شود؟

**Inside PerformanceAnalyzer.analyze_pose_file()**:

```python
def analyze_pose_file(
    self,
    pose_json_path,
    lane='left',
    calibration_path=None,  # ← calibration input
    start_frame=None,       # ← race boundary
    end_frame=None          # ← race boundary
):
    # 1. Load calibration if provided
    calibration = None
    if calibration_path and calibration_path.exists():
        calibration = CameraCalibrator.load_calibration(calibration_path)

    # 2. Load pose data
    frames = load_pose_json(pose_json_path)

    # 3. Process frames (with filtering + calibration)
    com_positions = []

    for frame in frames:
        frame_id = frame['frame_id']

        # FRAME FILTERING:
        if start_frame is not None and frame_id < start_frame:
            continue  # skip pre-race
        if end_frame is not None and frame_id > end_frame:
            break  # skip post-race

        # Calculate COM in pixels
        com_x_px, com_y_px = self.calculate_com(frame['keypoints'])

        # CALIBRATION:
        if calibration:
            # Convert pixel → meter
            com_x_m, com_y_m = calibration.pixel_to_meter_func(com_x_px, com_y_px)
            com_positions.append((com_x_m, com_y_m))
        else:
            # Keep in pixels (no calibration)
            com_positions.append((com_x_px, com_y_px))

    # 4. Calculate metrics (velocity, path length, etc.)
    # Units will be m/s (if calibrated) or px/s (if not)
    metrics = self.calculate_metrics(com_positions)

    # 5. Return
    return metrics
```

**نکته کلیدی**: calibration و frame filtering **inside** performance_metrics.py اتفاق می‌افتد، نه قبل/بعد آن!

---

### 3.3. کجا Frame Boundaries load می‌شوند؟

**Inside batch_calculate_metrics.py**:

```python
def calculate_race_metrics(self, pose_file, lane='left'):
    # 1. Find calibration file
    cal_file = self.calibration_dir / competition / f"{race_name}_calibration.json"

    # 2. Load race metadata
    metadata_path = self.race_segments_dir / competition / f"{race_name}_metadata.json"

    start_frame = None
    end_frame = None

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Extract race boundaries
        start_frame_orig = metadata.get('detected_start_frame')
        end_frame_orig = metadata.get('detected_finish_frame')
        segment_start = metadata.get('start_frame')

        # Convert to pose file frame IDs
        if start_frame_orig and segment_start:
            start_frame = start_frame_orig - segment_start
        if end_frame_orig and segment_start:
            end_frame = end_frame_orig - segment_start

    # 3. Call performance_metrics with both
    metrics = self.analyzer.analyze_pose_file(
        pose_file,
        lane=lane,
        calibration_path=cal_file if cal_file.exists() else None,
        start_frame=start_frame,
        end_frame=end_frame
    )

    return metrics
```

---

## بخش 4: پیاده‌سازی کامل

### 4.1. Modification به performance_metrics.py

**قبل**:
```python
# src/analysis/performance_metrics.py (خط ~237)

def analyze_pose_file(
    self,
    pose_json_path: Path,
    lane: str = 'left',
    min_visibility: float = 0.5,
    calibration_path: Optional[Path] = None
) -> Optional[PerformanceMetrics]:
```

**بعد**:
```python
def analyze_pose_file(
    self,
    pose_json_path: Path,
    lane: str = 'left',
    min_visibility: float = 0.5,
    calibration_path: Optional[Path] = None,
    start_frame: Optional[int] = None,      # NEW
    end_frame: Optional[int] = None,        # NEW
    race_metadata: Optional[Dict] = None   # NEW (optional)
) -> Optional[PerformanceMetrics]:
    """
    Analyze pose data and calculate performance metrics.

    Args:
        pose_json_path: Path to pose JSON file
        lane: 'left' or 'right'
        min_visibility: Minimum keypoint visibility threshold
        calibration_path: Path to calibration JSON (optional)
        start_frame: First frame of race period (optional)
        end_frame: Last frame of race period (optional)
        race_metadata: Full race metadata dict (optional)

    Returns:
        PerformanceMetrics object or None if failed
    """
```

**فیلتر فریم‌ها** (خط ~289):
```python
# Track filtering statistics
frames_total = len(frames)
frames_skipped_pre = 0
frames_skipped_post = 0
frames_analyzed = 0

for frame in frames:
    frame_id = frame.get('frame_id', 0)

    # Filter by race boundaries
    if start_frame is not None and frame_id < start_frame:
        frames_skipped_pre += 1
        continue  # Skip pre-race frames

    if end_frame is not None and frame_id > end_frame:
        frames_skipped_post += 1
        continue  # Skip post-race frames (or break for efficiency)

    # Process frame (only race frames)
    climber_data = frame.get(climber_key)
    if not climber_data or not climber_data.get('keypoints'):
        continue

    frames_analyzed += 1
    # ... existing processing code
```

**اضافه metadata به خروجی**:
```python
# At the end, before returning:
metrics_dict = metrics.to_dict()

# Add race boundaries info
metrics_dict['race_boundaries'] = {
    'start_frame': start_frame,
    'end_frame': end_frame,
    'total_frames_in_file': frames_total,
    'frames_analyzed': frames_analyzed,
    'frames_skipped_pre': frames_skipped_pre,
    'frames_skipped_post': frames_skipped_post
}

return metrics_dict
```

---

### 4.2. Modification به batch_calculate_metrics.py

**Location**: `scripts/batch_calculate_metrics.py` (خط ~94)

**قبل**:
```python
metrics = self.analyzer.analyze_pose_file(
    pose_file,
    lane=lane,
    calibration_path=cal_file if cal_file.exists() else None
)
```

**بعد**:
```python
# Load race metadata
metadata_path = pose_file.parent.parent.parent / "race_segments" / competition / f"{race_name}_metadata.json"

start_frame = None
end_frame = None
race_metadata = None

if metadata_path.exists():
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            race_metadata = json.load(f)

        # Extract race boundaries (original video frame IDs)
        start_frame_orig = race_metadata.get('detected_start_frame')
        end_frame_orig = race_metadata.get('detected_finish_frame')
        segment_start = race_metadata.get('start_frame')

        # Convert to pose file frame IDs (0-indexed)
        if start_frame_orig is not None and segment_start is not None:
            start_frame = start_frame_orig - segment_start

        if end_frame_orig is not None and segment_start is not None:
            end_frame = end_frame_orig - segment_start

        logger.info(f"  Race boundaries: start={start_frame}, end={end_frame}")

    except Exception as e:
        logger.warning(f"  Failed to load metadata: {e}")
else:
    logger.warning(f"  Metadata not found: {metadata_path} - processing all frames")

# Calculate metrics with calibration + race boundaries
metrics = self.analyzer.analyze_pose_file(
    pose_file,
    lane=lane,
    calibration_path=cal_file if cal_file.exists() else None,
    start_frame=start_frame,
    end_frame=end_frame,
    race_metadata=race_metadata
)
```

---

### 4.3. ساخت batch_calibration.py

**(کد کامل در PROMPT_FOR_UI_FIX_METRICS.md موجود است)**

**خلاصه**:
```python
class BatchCalibrator:
    def __init__(self, race_segments_dir, output_dir, ifsc_map_path):
        # Load IFSC route map (31 holds)
        # Create HoldDetector
        # Setup output directory

    def calibrate_race(self, video_path, output_path):
        # Open video
        # Create PeriodicCalibrator
        # Process all frames
        # Save calibration JSON
        # Return statistics

    def run(self, competition=None, max_races=None):
        # Find all race videos
        # Calibrate each
        # Save summary
```

**استفاده**:
```bash
# Test with 5 races
python scripts/batch_calibration.py --test

# Full batch (188 races)
python scripts/batch_calibration.py
```

---

## بخش 5: تست و Validation

### 5.1. تست مرحله به مرحله

#### Test 1: Frame Selection Only (بدون calibration)

```bash
# Modify code
# Test with 1 race
python scripts/batch_calculate_metrics.py --max-races 1 --competition chamonix_2024 --force

# Check output
cat data/processed/metrics/chamonix_2024/Speed_finals_Chamonix_2024_race001_metrics_left.json
```

**چک‌لیست**:
- [x] `race_boundaries` field exists
- [x] `frames_analyzed` < `total_frames_in_file`
- [x] velocity **2-3× higher** than before
- [x] efficiency **higher** than before
- [x] No crashes or errors

---

#### Test 2: Calibration Only (بدون frame filtering)

```bash
# Temporarily disable frame filtering in code
# Run calibration
python scripts/batch_calibration.py --test

# Check calibration file
cat data/processed/calibration/chamonix_2024/Speed_finals_Chamonix_2024_race001_calibration.json
```

**چک‌لیست**:
- [x] Calibration JSON created
- [x] RMSE < 10cm (ideally < 5cm)
- [x] holds_detected >= 4 for most frames
- [x] holds_used >= 4
- [x] inlier_ratio > 0.7

---

#### Test 3: Both Combined

```bash
# Re-enable frame filtering
# Run full pipeline on 5 races
python scripts/batch_calculate_metrics.py --max-races 5 --competition chamonix_2024 --force

# Check metrics
for f in data/processed/metrics/chamonix_2024/*_left.json; do
    echo "=== $f ==="
    jq '.is_calibrated, .units, .summary.avg_vertical_velocity, .race_boundaries.frames_analyzed' "$f"
done
```

**Expected output**:
```json
true
"meters"
2.34
53
```

---

### 5.2. Validation Metrics

#### Velocity Range Check

```python
# scripts/validate_metrics.py

import json
from pathlib import Path

for metrics_file in Path("data/processed/metrics").glob("*/*_metrics_*.json"):
    with open(metrics_file) as f:
        data = json.load(f)

    v = data['summary']['avg_vertical_velocity']

    # World record: ~2.67 m/s (5.0s for 15m)
    # Elite: 2.0-2.8 m/s
    # Amateur: 1.0-1.8 m/s
    # Suspicious: < 0.5 or > 5.0 m/s

    if v < 0.5 or v > 5.0:
        print(f"❌ OUTLIER: {metrics_file.name} - velocity = {v:.2f} m/s")
    elif 2.0 <= v <= 2.8:
        print(f"✅ ELITE: {metrics_file.name} - velocity = {v:.2f} m/s")
    elif 1.0 <= v < 2.0:
        print(f"✅ GOOD: {metrics_file.name} - velocity = {v:.2f} m/s")
    else:
        print(f"⚠️ CHECK: {metrics_file.name} - velocity = {v:.2f} m/s")
```

---

#### Path Efficiency Check

```python
for metrics_file in Path("data/processed/metrics").glob("*/*_metrics_*.json"):
    with open(metrics_file) as f:
        data = json.load(f)

    eff = data['summary']['path_efficiency']

    # Realistic range: 0.6-0.95
    # Perfect straight line: 1.0 (impossible)
    # Very inefficient: < 0.5

    if eff < 0.4 or eff > 0.98:
        print(f"❌ OUTLIER: {metrics_file.name} - efficiency = {eff:.2f}")
    elif 0.8 <= eff <= 0.95:
        print(f"✅ EXCELLENT: {metrics_file.name} - efficiency = {eff:.2f}")
    elif 0.6 <= eff < 0.8:
        print(f"✅ GOOD: {metrics_file.name} - efficiency = {eff:.2f}")
    else:
        print(f"⚠️ CHECK: {metrics_file.name} - efficiency = {eff:.2f}")
```

---

### 5.3. مقایسه قبل/بعد

```python
# scripts/compare_old_new.py

import pandas as pd

old_metrics = load_metrics("data/processed/metrics_old_INVALID")
new_metrics = load_metrics("data/processed/metrics")

comparison = pd.DataFrame({
    'race': [m['race_name'] for m in new_metrics],
    'old_velocity': [m_old['summary']['avg_vertical_velocity'] for m_old in old_metrics],
    'new_velocity': [m_new['summary']['avg_vertical_velocity'] for m_new in new_metrics],
    'velocity_ratio': [new/old for new, old in zip(new_velocities, old_velocities)],
    'old_calibrated': [m_old.get('is_calibrated', False) for m_old in old_metrics],
    'new_calibrated': [m_new.get('is_calibrated', False) for m_new in new_metrics]
})

print(comparison.describe())
```

**Expected results**:
```
velocity_ratio:
  count: 188
  mean:  8.5    (8.5× improvement)
  std:   2.3
  min:   4.2
  max:   15.3

new_calibrated:
  True: 188 (100%)
  False: 0
```

---

## بخش 6: عیب‌یابی

### 6.1. مشکلات متداول Frame Selection

#### Problem: Frame IDs don't match

**علامت**:
```
WARNING: start_frame=11784 but pose file only has 143 frames!
All frames skipped, no metrics calculated
```

**علت**: frame IDs اشتباه convert شده‌اند

**راه‌حل**:
```python
# Check metadata:
print(f"detected_start_frame: {metadata['detected_start_frame']}")
print(f"start_frame (segment): {metadata['start_frame']}")

# Correct conversion:
pose_start = metadata['detected_start_frame'] - metadata['start_frame']

# NOT:
pose_start = metadata['detected_start_frame']  # ❌ WRONG!
```

---

#### Problem: frames_analyzed = 0

**علت**: boundaries خیلی محدود هستند یا اشتباه

**راه‌حل**:
```python
# Debug output:
logger.info(f"Total frames: {len(frames)}")
logger.info(f"start_frame: {start_frame}, end_frame: {end_frame}")
logger.info(f"Frame IDs in pose file: {frames[0]['frame_id']} to {frames[-1]['frame_id']}")

# If start/end are out of range, use fallback:
if start_frame is None or start_frame < 0:
    start_frame = 0
if end_frame is None or end_frame >= len(frames):
    end_frame = len(frames) - 1
```

---

### 6.2. مشکلات متداول Calibration

#### Problem: Hold detection fails (< 4 holds)

**علت**: HSV thresholds خیلی محدود

**راه‌حل**:
```python
# Lower min_confidence:
hold_detector = HoldDetector(
    min_confidence=0.15,  # was 0.2
    min_area=30           # was 50
)

# Or adjust HSV range (src/phase1_pose_estimation/hold_detector.py):
lower_red1 = np.array([0, 100, 50])    # was [0, 120, 70]
upper_red1 = np.array([10, 255, 255])  # was [10, 255, 255]
```

---

#### Problem: RMSE > 10cm

**علت**:
- Not enough holds detected
- Outlier hold matches
- Camera angle extreme

**راه‌حل**:
```python
# Use PeriodicCalibrator with outlier rejection:
calibrator = PeriodicCalibrator(
    route_map=route_map,
    hold_detector=hold_detector,
    recalibration_interval=30,
    rmse_threshold=0.10,  # Reject if RMSE > 10cm
    fallback_to_previous=True  # Use previous calibration if current fails
)
```

---

#### Problem: Video won't open

**علত**: path اشتباه یا فایل خراب

**راه‌حل**:
```python
import cv2

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    # Try absolute path:
    abs_path = video_path.resolve()
    cap = cv2.VideoCapture(str(abs_path))

    if not cap.isOpened():
        # Check if file exists:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        else:
            raise RuntimeError(f"Cannot open video (corrupt?): {video_path}")
```

---

### 6.3. Memory Issues

**Problem**: Out of memory during batch processing

**راه‌حل**: Process در chunks

```python
# Instead of:
all_videos = list(Path("data/race_segments").glob("*/*.mp4"))
for video in all_videos:  # ❌ 188 races at once
    process(video)

# Do:
competitions = ["seoul_2024", "villars_2024", ...]
for comp in competitions:
    videos = list(Path(f"data/race_segments/{comp}").glob("*.mp4"))
    for video in videos:  # ✅ ~30 races at a time
        process(video)
```

---

## بخش 7: نکات پیشرفته

### 7.1. Temporal Smoothing

**Problem**: Calibration jitter بین frameها

**راه‌حل**: PeriodicCalibrator با smoothing

```python
class PeriodicCalibrator:
    def __init__(self, ..., smoothing_window=3):
        self.recent_calibrations = []  # Last N calibrations
        self.smoothing_window = smoothing_window

    def calibrate_frame(self, frame, frame_id):
        # Get raw calibration
        raw_calib = self._calibrate_single_frame(frame)

        # Add to history
        self.recent_calibrations.append(raw_calib)
        if len(self.recent_calibrations) > self.smoothing_window:
            self.recent_calibrations.pop(0)

        # Smooth homography matrix
        smoothed_H = np.mean([c.homography for c in self.recent_calibrations], axis=0)

        # Return smoothed calibration
        return CalibrationResult(
            homography=smoothed_H,
            # ... other fields
        )
```

---

### 7.2. Hold Detection Optimization

**Wall Segmentation**: محدود کردن detection به بخش دیوار

```python
def detect_holds_with_wall_mask(frame, wall_mask):
    """
    Detect holds only in wall region (not background/audience).
    """
    # Create mask (wall = white, background = black)
    # ... (using edge detection or manual annotation)

    # Apply mask before color thresholding
    masked_frame = cv2.bitwise_and(frame, frame, mask=wall_mask)

    # Detect holds in masked frame
    holds = hold_detector.detect(masked_frame)

    return holds
```

---

### 7.3. Multi-Scale Detection

**مشکل**: holds در frames مختلف سایزهای متفاوت دارند (zoom)

**راه‌حل**: Adaptive thresholds

```python
def adaptive_hold_detection(frame, frame_id, total_frames):
    """
    Adjust detection params based on frame position (zoom level).
    """
    # Estimate zoom (linear approximation)
    zoom_factor = 1.0 + 2.0 * (frame_id / total_frames)  # 1.0 → 3.0

    # Adjust min_area based on zoom
    min_area_adaptive = base_min_area * (zoom_factor ** 2)

    # Detect with adaptive params
    holds = hold_detector.detect(frame, min_area=min_area_adaptive)

    return holds
```

---

### 7.4. Performance Benchmarks

**عملکرد مورد انتظار**:

| Operation | Time per race | Notes |
|-----------|---------------|-------|
| Hold detection | 5-10s | CPU-bound |
| Calibration (single frame) | 0.1-0.5s | Depends on # holds |
| Calibration (periodic, 143 frames) | 2-5s | Cached every 30 frames |
| Metrics calculation (calibrated) | 0.5-1s | Mostly Python overhead |
| **Total per race** | **10-20s** | Full pipeline |

**برای 188 races**: 30-60 دقیقه (با parallelization: 10-20 دقیقه)

---

## خلاصه و نتیجه‌گیری

### چیزهایی که یاد گرفتیم:

1. **Calibration ضروری است** برای:
   - تبدیل pixel → meter
   - مقایسه بین races
   - تحلیل بیومکانیکی معتبر

2. **Frame Selection ضروری است** برای:
   - حذف فریم‌های pre/post-race
   - محاسبه صحیح velocity (2-3× improvement)
   - metrics معتبر

3. **پیاده‌سازی** نیاز دارد:
   - Modification به `performance_metrics.py`
   - Modification به `batch_calculate_metrics.py`
   - ساخت `batch_calibration.py`

4. **Validation** شامل:
   - بررسی velocity range (0.5-5.0 m/s)
   - بررسی efficiency range (0.4-0.98)
   - مقایسه قبل/بعد (8-10× improvement)

5. **عیب‌یابی** نیاز به:
   - Debug frame ID conversions
   - Adjust HSV thresholds
   - Handle memory issues با chunking

---

### چک‌لیست نهایی

**قبل از production deployment**:
- [ ] همه 188 race calibrated شدند (RMSE < 10cm)
- [ ] همه metrics با frame filtering محاسبه شدند
- [ ] validation checks passed (velocity, efficiency in range)
- [ ] comparison با old metrics (8-10× improvement)
- [ ] aggregations و leaderboards updated
- [ ] documentation کامل است

---

**موفق باشید!** 🚀

این راهنما باید همه سوالات شما را پاسخ دهد. اگر سوالی مانده، به فایل‌های زیر مراجعه کنید:
- `MASTER_CONTEXT.md` - وضعیت کلی پروژه
- `PROMPT_FOR_UI_FIX_METRICS.md` - دستورالعمل دقیق برای UI
- `docs/SESSION_LOG_PHASE3_TEST.md` - گزارش کشف مشکلات

---

**تهیه شده**: 2025-11-15
**توسط**: Claude Code
**نسخه**: 1.0
