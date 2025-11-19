# طراحی سیستم Global Map Registration
# Global Map Registration System Design

**تاریخ**: 2025-11-19
**وضعیت**: طراحی معماری
**مؤلف**: Claude Code Session

---

## 📋 خلاصه اجرایی

این پروژه **بازمهندسی** سیستم موجود نیست، بلکه **یکپارچه‌سازی** (Integration) قطعات موجود است.

**خبر خوب**: تمام اجزای مورد نیاز آماده است:
- ✅ Homography-based calibration
- ✅ Hold detection
- ✅ IFSC route map
- ✅ Performance metrics calculator

**مشکل فعلی**: این اجزا در یک pipeline یکپارچه استفاده نمی‌شوند.

---

## 🔴 مشکلات کنونی

### مشکل 1: محاسبات پیکسلی با دوربین متحرک

```python
# سیستم فعلی:
velocity = 376.6 px/s  # ❌ بی‌معنی! دوربین zoom می‌کند
distance = "9 متر"     # ❌ اشتباه! فقط 15m ارتفاع دیوار است

# پس از fix:
velocity = 2.34 m/s    # ✅ واحد فیزیکی معنادار
distance = 12.5 m      # ✅ بر اساس مختصات جهانی
```

**علت**:
- `batch_calculate_metrics.py` فایل calibration می‌خواهد
- اما هیچ calibration file تولید نمی‌شود
- نتیجه: همه metrics در pixels محاسبه می‌شوند

**تأثیر**:
- دوربین Pan/Tilt حرکت می‌کند → مقیاس pixel تغییر می‌کند
- عدد 376 px/s در ثانیه 1 ≠ 376 px/s در ثانیه 5
- نمی‌توان races را مقایسه کرد
- **همه تحلیل‌ها نامعتبر است**

### مشکل 2: عدم مدیریت سقوط (Dropout)

```
سناریو: ورزشکار سقوط می‌کند
فریم 150: pose detected ✅
فریم 151: out of frame ❌ ← سقوط!
فریم 152: سیستم فعلی همچنان می‌خواهد detect کند
فریم 153: pose detection fails → error

خروجی مورد انتظار:
{
  "status": "DNF",
  "dropout_frame": 151,
  "reason": "out_of_frame",
  "valid_data_until": 2.5  // seconds
}
```

### مشکل 3: عدم استفاده از Race Boundaries

```python
# ویدئو نمونه: 4.77 ثانیه (143 فریم)
# قبل مسابقه:  1.50s (45 فریم) ← ایستاده، آماده ❌
# مسابقه:      1.77s (53 فریم) ← صعود واقعی ✅
# بعد مسابقه:  1.50s (45 فریم) ← تمام شده ❌

# فعلاً: 143 فریم (100%) پردازش می‌شود
# صحیح: 53 فریم (37%) باید پردازش شود
# خطا: 2.7× underestimation در velocity!
```

---

## 🏗️ معماری سیستم جدید

### نمای کلی (High-Level Architecture)

```
┌──────────────────────────────────────────────────────────────┐
│                    Video Input (MP4)                         │
│              1280×720, 30fps, Pan/Tilt camera                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              GlobalMapVideoProcessor                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Frame-by-Frame Processing Loop:                      │   │
│  │  1. Pose Estimation (BlazePose)                      │   │
│  │  2. Hold Detection (Red color HSV)                   │   │
│  │  3. Camera Calibration (Homography)                  │   │
│  │  4. World Coordinate Transform (pixel → meter)       │   │
│  │  5. Dropout Detection (out of frame check)           │   │
│  │  6. Lane Assignment (left/right)                     │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Output: JSON Time-Series                    │
│  {                                                           │
│    "left_climber": {                                         │
│      "timestamps": [0, 0.033, 0.066, ...],                  │
│      "y_position_m": [0.0, 0.05, 0.12, ...],  ← متر!        │
│      "status": ["climbing", "climbing", ..., "DNF"]         │
│    },                                                        │
│    "right_climber": { ... }                                  │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
```

### اجزای اصلی (Core Components)

#### 1. WorldCoordinateTracker

**مسئولیت**: تبدیل مختصات پیکسلی به مختصات جهانی (متر)

```python
class WorldCoordinateTracker:
    """
    Track athlete position in world coordinates (meters) using
    per-frame camera calibration.

    Features:
    - Per-frame hold detection
    - Per-frame homography computation
    - Pixel → meter transformation
    - Calibration quality monitoring
    """

    def __init__(self, route_map_path, ifsc_standards):
        self.hold_detector = HoldDetector(route_map_path)
        self.calibrator = PeriodicCalibrator(
            route_map_path,
            recalibration_interval=15  # Recalibrate every 15 frames (0.5s @ 30fps)
        )
        self.ifsc_standards = {
            'wall_height_m': 15.0,
            'wall_width_m': 3.0,
            'hold_spacing_mm': 125,
            'start_pad_height_m': 0.2  # مرجع: پد شروع
        }

    def process_frame(self, frame, frame_id, lane):
        """
        Process single frame and return world coordinates.

        Returns:
            {
                'y_position_m': float,  # ارتفاع از پد شروع
                'x_position_m': float,  # فاصله افقی
                'calibration_quality': float,
                'is_valid': bool
            }
        """
        # 1. Detect holds in frame
        holds = self.hold_detector.detect_holds(frame, lane=lane)

        # 2. Calibrate camera (periodic - not every frame)
        calibration = self.calibrator.calibrate_frame(
            frame, frame_id, holds, lane=lane
        )

        if calibration is None or calibration.confidence < 0.6:
            return {'is_valid': False, 'reason': 'calibration_failed'}

        # 3. Get pose (assumed already extracted)
        # pose = get_pose_from_cache(frame_id, lane)

        # 4. Convert COM from pixels to meters
        # com_x_m, com_y_m = calibration.pixel_to_meter_func(com_x_px, com_y_px)

        # 5. Transform to wall-relative coordinates
        # y_from_start = self.ifsc_standards['wall_height_m'] - com_y_m

        return {
            'y_position_m': y_from_start,
            'x_position_m': com_x_m,
            'calibration_quality': calibration.confidence,
            'calibration_rmse': calibration.rmse_error,
            'is_valid': True
        }
```

**کلیدی**: استفاده از `PeriodicCalibrator` برای کاهش هزینه محاسباتی:
- Recalibration هر 15 فریم (0.5 ثانیه در 30fps)
- Cache calibration برای فریم‌های بینابین
- سرعت: ~30× بهبود نسبت به per-frame calibration

#### 2. DropoutHandler

**مسئولیت**: تشخیص و مدیریت سناریوی سقوط

```python
class DropoutHandler:
    """
    Detect and manage athlete dropout scenarios:
    - Out of frame (fall/disqualification)
    - Lost tracking (occlusion)
    - Race finished
    """

    def __init__(self, timeout_frames=30):
        self.timeout_frames = timeout_frames  # Max frames without detection
        self.tracking_history = []

    def check_dropout(self, frame, pose_result, calibration_result, lane):
        """
        Check if athlete has dropped out.

        Returns:
            {
                'has_dropped_out': bool,
                'status': str,  # 'climbing', 'out_of_frame', 'finished', 'DNF'
                'confidence': float
            }
        """
        # Case 1: No pose detected
        if pose_result is None or not pose_result.is_valid:
            self.tracking_history.append(False)

            # Check if timeout exceeded
            if len(self.tracking_history) > self.timeout_frames:
                recent_detections = sum(self.tracking_history[-self.timeout_frames:])
                if recent_detections < 5:  # Less than 5 detections in last 30 frames
                    return {
                        'has_dropped_out': True,
                        'status': 'out_of_frame',
                        'confidence': 0.9
                    }
        else:
            self.tracking_history.append(True)

        # Case 2: Calibration failed (no holds visible → out of frame)
        if calibration_result is None or calibration_result.confidence < 0.4:
            return {
                'has_dropped_out': True,
                'status': 'out_of_frame',
                'confidence': 0.7
            }

        # Case 3: Check if reached top (y > 14.5m)
        # (این را در processing loop چک می‌کنیم)

        return {
            'has_dropped_out': False,
            'status': 'climbing',
            'confidence': 1.0
        }
```

#### 3. GlobalMapVideoProcessor

**مسئولیت**: Pipeline یکپارچه برای پردازش ویدئو

```python
class GlobalMapVideoProcessor:
    """
    Integrated video processor with global map registration.

    Combines:
    - Pose estimation
    - Hold detection
    - Camera calibration
    - World coordinate tracking
    - Dropout handling
    """

    def __init__(self, route_map_path, config):
        self.pose_extractor = BlazePoseExtractor()
        self.world_tracker = WorldCoordinateTracker(route_map_path, config)
        self.dropout_handler = DropoutHandler()
        self.lane_detector = DualLaneDetector()

    def process_race(self, video_path, race_metadata):
        """
        Process entire race video and generate time-series output.

        Args:
            video_path: Path to race video
            race_metadata: Race metadata (start_frame, end_frame, etc.)

        Returns:
            {
                'left_climber': TimeSeriesData,
                'right_climber': TimeSeriesData,
                'metadata': ProcessingMetadata
            }
        """
        # 1. Load video
        video = VideoProcessor(video_path)

        # 2. Get race boundaries
        start_frame = race_metadata.get('detected_start_frame', 0)
        end_frame = race_metadata.get('detected_finish_frame', video.total_frames)

        # 3. Initialize data containers
        left_data = TimeSeriesBuilder()
        right_data = TimeSeriesBuilder()

        # 4. Process frame-by-frame
        for frame_data in video.extract_frames(start_frame, end_frame):
            frame_id = frame_data['frame_id']
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']

            # Detect lane boundary
            lane_boundary = self.lane_detector.detect_boundary(frame)

            # Process each climber
            for lane in ['left', 'right']:
                # A. Pose estimation
                pose = self.pose_extractor.extract_pose(frame, lane=lane)

                # B. World coordinate tracking
                world_coords = self.world_tracker.process_frame(
                    frame, frame_id, lane
                )

                # C. Dropout detection
                dropout_status = self.dropout_handler.check_dropout(
                    frame, pose, world_coords.get('calibration_quality'), lane
                )

                # D. Store data
                if lane == 'left':
                    left_data.add_frame(
                        timestamp=timestamp,
                        y_position_m=world_coords.get('y_position_m'),
                        status=dropout_status['status'],
                        is_valid=world_coords.get('is_valid', False)
                    )
                else:
                    right_data.add_frame(
                        timestamp=timestamp,
                        y_position_m=world_coords.get('y_position_m'),
                        status=dropout_status['status'],
                        is_valid=world_coords.get('is_valid', False)
                    )

                # E. Early termination if both dropped out
                if (left_data.is_finished() and right_data.is_finished()):
                    break

        # 5. Build final output
        return {
            'left_climber': left_data.build(),
            'right_climber': right_data.build(),
            'metadata': {
                'video_path': str(video_path),
                'processing_date': datetime.now().isoformat(),
                'calibration_method': 'per_frame_homography',
                'units': 'meters',
                'reference_point': 'start_pad'
            }
        }
```

---

## 📊 خروجی سیستم (Output Format)

### JSON Time-Series Output

```json
{
  "race_id": "Seoul_2024_race013",
  "competition": "Seoul_2024",
  "metadata": {
    "video_path": "data/race_segments/seoul_2024/race013.mp4",
    "processing_date": "2025-11-19T10:30:00",
    "calibration_method": "periodic_homography",
    "recalibration_interval_frames": 15,
    "units": "meters",
    "reference_point": "start_pad",
    "wall_height_m": 15.0
  },

  "left_climber": {
    "lane": "left",
    "race_outcome": "finished",

    "time_series": {
      "timestamps": [0.000, 0.033, 0.066, 0.100, ..., 5.833],
      "y_position_m": [0.0, 0.05, 0.12, 0.21, ..., 14.95],
      "x_position_m": [1.2, 1.18, 1.15, 1.20, ..., 1.35],
      "status": ["climbing", "climbing", "climbing", ..., "finished"],
      "calibration_quality": [0.85, 0.87, 0.89, ..., 0.92]
    },

    "summary": {
      "total_time_s": 5.833,
      "total_distance_m": 14.95,
      "avg_velocity_m_s": 2.56,
      "max_velocity_m_s": 3.12,
      "final_height_m": 14.95
    }
  },

  "right_climber": {
    "lane": "right",
    "race_outcome": "DNF",
    "dropout_frame": 89,
    "dropout_time_s": 2.97,
    "dropout_reason": "out_of_frame",

    "time_series": {
      "timestamps": [0.000, 0.033, ..., 2.97],
      "y_position_m": [0.0, 0.04, ..., 7.85],
      "x_position_m": [4.3, 4.28, ..., 4.15],
      "status": ["climbing", "climbing", ..., "DNF"],
      "calibration_quality": [0.82, 0.84, ..., 0.35]
    },

    "summary": {
      "total_time_s": 2.97,
      "total_distance_m": 7.85,
      "avg_velocity_m_s": 2.64,
      "max_velocity_m_s": 3.05,
      "final_height_m": 7.85
    }
  },

  "processing_stats": {
    "total_frames_processed": 175,
    "avg_calibration_rmse_m": 0.0012,
    "avg_calibration_confidence": 0.86,
    "holds_detected_per_frame": 12.3,
    "processing_time_s": 45.2
  }
}
```

---

## 🚀 برنامه پیاده‌سازی (Implementation Plan)

### Phase 1: Core Components (3-4 ساعت)

**Task 1.1**: `WorldCoordinateTracker` (1.5 ساعت)
- ✅ استفاده از کدهای موجود: `HoldDetector`, `PeriodicCalibrator`
- ✅ Integration logic برای per-frame processing
- ✅ IFSC standards configuration
- ✅ تست با یک فریم نمونه

**Task 1.2**: `DropoutHandler` (1 ساعت)
- ✅ Logic تشخیص out_of_frame
- ✅ Tracking history management
- ✅ Status assignment (climbing, DNF, finished)
- ✅ تست با سناریوهای مختلف

**Task 1.3**: `TimeSeriesBuilder` (0.5 ساعت)
- ✅ Data container برای ذخیره time-series
- ✅ JSON serialization
- ✅ Summary statistics calculation

### Phase 2: Integrated Pipeline (2-3 ساعت)

**Task 2.1**: `GlobalMapVideoProcessor` (2 ساعت)
- ✅ Integration تمام components
- ✅ Frame-by-frame processing loop
- ✅ Error handling
- ✅ Progress reporting

**Task 2.2**: Output Generator (1 ساعت)
- ✅ JSON output formatting
- ✅ Metadata generation
- ✅ Summary statistics

### Phase 3: Testing & Validation (2-3 ساعت)

**Task 3.1**: تست با ویدئوهای نمونه (1.5 ساعت)
- ✅ تست با 3 race مختلف
- ✅ Validation نتایج (velocity معقول باشد: 1.5-3.5 m/s)
- ✅ چک کردن dropout detection

**Task 3.2**: Batch Processing Script (1 ساعت)
- ✅ اسکریپت برای پردازش 114 reliable races
- ✅ Progress tracking
- ✅ Error recovery

### Phase 4: Documentation & Deployment (1 ساعت)

**Task 4.1**: مستندسازی (0.5 ساعت)
- ✅ به‌روزرسانی MASTER_CONTEXT.md
- ✅ نوشتن User Guide

**Task 4.2**: Git Commit & Push (0.5 ساعت)
- ✅ Commit با commit message مناسب
- ✅ Push به branch مشخص شده

---

## 📏 معیارهای موفقیت (Success Criteria)

### کیفیت Calibration
- ✅ RMSE < 0.05m (5cm) برای هر فریم
- ✅ Confidence > 0.7 برای 80% فریم‌ها
- ✅ حداقل 8 hold detected در هر فریم

### صحت Metrics
- ✅ Velocity range: 1.5 - 3.5 m/s (معقول برای speed climbing)
- ✅ Max velocity < 4.0 m/s (فیزیکاً ممکن)
- ✅ Total climb time: 5-8 seconds (مطابق IFSC records)
- ✅ Total distance: 14-15 meters (ارتفاع دیوار)

### Dropout Detection
- ✅ 100% detection برای سقوط‌های واضح
- ✅ False positive rate < 5%
- ✅ زمان تشخیص < 1 second (30 frames @ 30fps)

### Performance
- ✅ پردازش: < 2× realtime (یک ویدئو 6 ثانیه‌ای در < 12 ثانیه)
- ✅ Memory usage: < 4GB برای ویدئو 720p
- ✅ Batch processing: 114 races در < 30 دقیقه

---

## 🔧 نکات فنی کلیدی

### 1. Periodic Calibration vs Per-Frame

**چرا هر فریم calibrate نمی‌کنیم؟**
- Hold detection: ~50ms per frame
- Homography computation: ~10ms
- برای 180 فریم (6s @ 30fps): 180 × 60ms = 10.8 seconds

**راه‌حل**: Periodic calibration
- Recalibrate هر 15 فریم (0.5s)
- Cache calibration برای فریم‌های بینابین
- زمان: 12 × 60ms = 0.72 seconds (15× بهتر!)

### 2. مرجع مختصات: پد شروع

```python
# مختصات دیوار IFSC: Y=0 در پایین دیوار
# مختصات خروجی ما: Y=0 در پد شروع (0.2m بالاتر از زمین)

wall_y_m = calibration.pixel_to_meter_func(com_x_px, com_y_px)[1]
y_from_start = WALL_HEIGHT - wall_y_m  # 15m - wall_y_m

# نتیجه: Y=0 وقتی ورزشکار روی پد است، Y=15 در بالای دیوار
```

### 3. مدیریت فریم‌های بد

```python
# Strategy: Interpolation برای missing frames
if calibration_failed or pose_failed:
    # استفاده از last valid calibration
    # یا interpolate بین valid frames
    y_position = interpolate(
        last_valid_y,
        next_valid_y,
        current_timestamp
    )
```

---

## 📚 فایل‌های کلیدی

### فایل‌های موجود (استفاده می‌شوند):
- `src/calibration/camera_calibration.py` - PeriodicCalibrator
- `src/calibration/ifsc_route_map.py` - Route map
- `src/phase1_pose_estimation/hold_detector.py` - Hold detection
- `src/phase1_pose_estimation/dual_lane_detector.py` - Lane separation
- `src/phase1_pose_estimation/video_processor.py` - Video I/O
- `src/analysis/performance_metrics.py` - Metrics calculation

### فایل‌های جدید (باید ایجاد شوند):
- `src/calibration/world_coordinate_tracker.py` ⭐ NEW
- `src/calibration/dropout_handler.py` ⭐ NEW
- `src/calibration/time_series_builder.py` ⭐ NEW
- `src/phase1_pose_estimation/global_map_processor.py` ⭐ NEW
- `scripts/batch_process_with_global_map.py` ⭐ NEW

---

## ⏱️ زمان‌بندی کلی

| Phase | Tasks | تخمین زمان |
|-------|-------|-----------|
| Phase 1 | Core Components | 3-4 ساعت |
| Phase 2 | Integration Pipeline | 2-3 ساعت |
| Phase 3 | Testing با نمونه‌ها | 2-3 ساعت |
| Phase 4 | Documentation اولیه | 1 ساعت |
| **Phase 5** | **Full Dataset Reprocessing (188 races)** | **3-4 ساعت** |
| **جمع** | | **11-15 ساعت** |

---

## 🔄 Phase 5: Full Dataset Reprocessing (پردازش مجدد کامل)

### چرا پردازش مجدد تمام 188 race؟

**وضعیت فعلی**:
- ✅ 114 races "reliable" پردازش شده (60.6%)
- ❌ 74 races "suspicious" deferred شده (39.4%)
  - 5 CRITICAL: duration نامعتبر (منفی یا نزدیک صفر)
  - 58 Zilina 2025: شکست سیستماتیک (84% از آن مسابقه!)
  - 11 دیگر: مدت زمان خیلی کوتاه یا خیلی طولانی

**با سیستم جدید Global Map Registration**:
- ✅ مختصات جهانی → تشخیص دقیق شروع/پایان
- ✅ Dropout detection → مدیریت صحیح سقوط
- ✅ Per-frame calibration → حتی با حرکت شدید دوربین

**انتظار**: بسیاری از "suspicious" races قابل پردازش می‌شوند

### استراتژی Reprocessing

#### مرحله 1: Validation Pipeline (قبل از شروع)

```python
class RaceValidator:
    """
    Validate race before processing to catch obvious issues early.
    """

    def validate_race_metadata(self, race_metadata):
        """
        Pre-flight checks برای race metadata.

        Returns:
            {
                'is_valid': bool,
                'issues': List[str],
                'severity': 'ok' | 'warning' | 'critical'
            }
        """
        issues = []

        # Check 1: Video file exists
        if not Path(race_metadata['video_path']).exists():
            issues.append('video_file_missing')
            return {'is_valid': False, 'issues': issues, 'severity': 'critical'}

        # Check 2: Start/end frames reasonable
        start = race_metadata.get('detected_start_frame', 0)
        end = race_metadata.get('detected_finish_frame', 9999)
        duration_frames = end - start

        if duration_frames <= 0:
            issues.append(f'invalid_duration_frames: {duration_frames}')
            severity = 'critical'
        elif duration_frames < 60:  # < 2s at 30fps
            issues.append(f'very_short_race: {duration_frames} frames')
            severity = 'warning'
        elif duration_frames > 900:  # > 30s at 30fps
            issues.append(f'very_long_race: {duration_frames} frames')
            severity = 'warning'
        else:
            severity = 'ok'

        # Check 3: Pose data exists (if already extracted)
        pose_path = race_metadata.get('pose_file_path')
        if pose_path and not Path(pose_path).exists():
            issues.append('pose_data_missing')
            severity = 'warning'

        is_valid = severity != 'critical'
        return {'is_valid': is_valid, 'issues': issues, 'severity': severity}
```

#### مرحله 2: Three-Pass Strategy

**Pass 1: Clean Races (114 reliable)**
- پردازش با تمام features فعال
- انتظار: 100% موفقیت
- هدف: Validate سیستم جدید

**Pass 2: Suspicious Races (74 deferred)**
- پردازش با error recovery فعال
- هر race به صورت جداگانه (resilient mode)
- انتظار: 50-80% موفقیت

**Pass 3: Failed Races (باقیمانده)**
- Manual review flag
- ذخیره diagnostics برای تحلیل دستی
- گزارش مشکلات برای بهبود آینده

#### مرحله 3: Quality Validation

```python
class ReprocessingValidator:
    """
    Validate reprocessed data quality.
    """

    def validate_output(self, race_output):
        """
        Check if output meets quality criteria.

        Returns:
            {
                'passes_validation': bool,
                'quality_score': float (0-1),
                'issues': List[str],
                'metrics': Dict
            }
        """
        issues = []
        checks = {}

        # Check 1: Velocity reasonable (1.5-4.0 m/s)
        avg_vel = race_output['left_climber']['summary']['avg_velocity_m_s']
        if not (1.5 <= avg_vel <= 4.0):
            issues.append(f'unrealistic_velocity: {avg_vel:.2f} m/s')
            checks['velocity'] = False
        else:
            checks['velocity'] = True

        # Check 2: Total distance reasonable (10-16m)
        distance = race_output['left_climber']['summary']['total_distance_m']
        if not (10.0 <= distance <= 16.0):
            issues.append(f'unrealistic_distance: {distance:.2f} m')
            checks['distance'] = False
        else:
            checks['distance'] = True

        # Check 3: Calibration quality
        avg_cal_quality = np.mean(
            race_output['left_climber']['time_series']['calibration_quality']
        )
        if avg_cal_quality < 0.6:
            issues.append(f'low_calibration_quality: {avg_cal_quality:.2f}')
            checks['calibration'] = False
        else:
            checks['calibration'] = True

        # Check 4: Data completeness
        timestamps = race_output['left_climber']['time_series']['timestamps']
        valid_frames = sum(1 for s in race_output['left_climber']['time_series']['status']
                          if s != 'invalid')
        completeness = valid_frames / len(timestamps)
        if completeness < 0.8:
            issues.append(f'low_completeness: {completeness:.1%}')
            checks['completeness'] = False
        else:
            checks['completeness'] = True

        # Quality score
        quality_score = sum(checks.values()) / len(checks)
        passes = quality_score >= 0.75

        return {
            'passes_validation': passes,
            'quality_score': quality_score,
            'issues': issues,
            'metrics': checks
        }
```

### Batch Processing Script

**فایل**: `scripts/batch_reprocess_all_races.py`

```python
"""
Batch reprocess all 188 races with Global Map Registration.

Usage:
    python scripts/batch_reprocess_all_races.py \
        --input configs/race_timestamps/ \
        --output data/processed/global_map_v2/ \
        --passes 3 \
        --resume

Features:
- Three-pass strategy (clean → suspicious → failed)
- Progress tracking with resume capability
- Quality validation per race
- Detailed error logging
- Summary report generation
"""

def main():
    # 1. Load all race metadata (188 races)
    all_races = load_all_race_metadata()

    # 2. Classify races
    clean_races = [r for r in all_races if r['status'] == 'reliable']  # 114
    suspicious_races = [r for r in all_races if r['status'] == 'suspicious']  # 74

    # 3. Initialize processors
    processor = GlobalMapVideoProcessor(...)
    validator = ReprocessingValidator()

    # 4. Pass 1: Clean races
    print("Pass 1/3: Processing 114 clean races...")
    pass1_results = process_batch(clean_races, processor, validator)
    print(f"  Success: {pass1_results['success']}/{len(clean_races)}")

    # 5. Pass 2: Suspicious races
    print("Pass 2/3: Processing 74 suspicious races...")
    pass2_results = process_batch(suspicious_races, processor, validator,
                                   resilient_mode=True)
    print(f"  Success: {pass2_results['success']}/{len(suspicious_races)}")

    # 6. Pass 3: Failed races (if any)
    failed_races = pass1_results['failed'] + pass2_results['failed']
    if failed_races:
        print(f"Pass 3/3: Attempting {len(failed_races)} failed races...")
        pass3_results = process_batch(failed_races, processor, validator,
                                       max_retries=3, verbose=True)
        print(f"  Success: {pass3_results['success']}/{len(failed_races)}")

    # 7. Generate summary report
    generate_summary_report(pass1_results, pass2_results, pass3_results)
```

### خروجی Reprocessing

**ساختار فولدر**:
```
data/processed/global_map_v2/
├── successful/                    # Races پردازش موفق
│   ├── seoul_2024/
│   │   ├── race001_global_map.json
│   │   ├── race002_global_map.json
│   │   └── ...
│   ├── villars_2024/
│   ├── chamonix_2024/
│   ├── innsbruck_2024/
│   └── zilina_2025/
│
├── failed/                        # Races ناموفق
│   ├── race_diagnostics/          # اطلاعات debug
│   │   ├── seoul_2024_race005_diagnostics.json
│   │   └── ...
│   └── failed_races_list.json
│
├── reports/
│   ├── reprocessing_summary.json   # خلاصه کلی
│   ├── quality_comparison.csv      # مقایسه با نتایج قبلی
│   └── validation_report.json      # نتایج validation
│
└── aggregated/
    ├── all_races_timeseries_v2.csv  # تمام races در یک فایل
    ├── competition_stats_v2.json     # آمار هر مسابقه
    └── ml_ready_dataset_v2.npz       # آماده ML
```

### گزارش خلاصه (Expected Output)

```json
{
  "reprocessing_summary": {
    "total_races": 188,
    "successful": 165,
    "failed": 23,
    "success_rate": "87.8%",

    "by_category": {
      "clean_races": {
        "total": 114,
        "successful": 112,
        "failed": 2,
        "success_rate": "98.2%"
      },
      "suspicious_races": {
        "total": 74,
        "successful": 53,
        "failed": 21,
        "success_rate": "71.6%"
      }
    },

    "quality_metrics": {
      "avg_velocity_m_s": 2.45,
      "avg_calibration_rmse_m": 0.0034,
      "avg_calibration_confidence": 0.82,
      "avg_data_completeness": 0.94
    },

    "improvement_over_v1": {
      "previously_unusable_now_working": 53,
      "velocity_accuracy_improvement": "215%",
      "distance_accuracy_improvement": "98%"
    },

    "failed_races_breakdown": {
      "video_corrupted": 5,
      "no_holds_visible": 8,
      "extreme_camera_movement": 6,
      "other": 4
    }
  }
}
```

### تخمین زمان Phase 5

**زمان پردازش** (با GPU):
- Clean races (114): ~1 ساعت (30 sec/race average)
- Suspicious races (74): ~1.5 ساعت (73 sec/race average - slower due to retries)
- Failed races retry: ~0.5 ساعت
- Validation & reporting: ~0.5 ساعت

**جمع**: 3-4 ساعت برای پردازش کامل

**نکته**: می‌توان با parallel processing (multiprocessing) به 1.5-2 ساعت کاهش داد

---

## ✅ مراحل نهایی (بعد از Reprocessing)

### 1. مقایسه با نتایج قبلی

```python
# Compare v1 (pixel-based) vs v2 (global map)
comparison_df = pd.DataFrame({
    'race_id': [...],
    'v1_velocity_px_s': [...],  # بی‌معنی!
    'v2_velocity_m_s': [...],    # معنادار!
    'v1_distance_px': [...],     # متغیر با zoom
    'v2_distance_m': [...],      # ثابت (واحد فیزیکی)
})
```

### 2. Dataset نهایی برای تحلیل/ML

```
data/final_dataset_v2/
├── time_series/
│   └── all_188_races_timeseries.csv  # Y(t) برای همه races
├── aggregated/
│   └── race_summaries.csv             # یک خط برای هر race
└── ml_ready/
    ├── features.npz                   # Feature matrix
    ├── train_test_split.json          # 80/20 split
    └── metadata.json                   # توضیحات کامل
```

### 3. Visualization Dashboard Update

- به‌روزرسانی Plotly dashboard با داده‌های جدید
- نمودار مقایسه v1 vs v2
- نمودار کیفیت calibration
- Interactive 2D wall view با trajectory واقعی

---

## 📊 معیارهای موفقیت نهایی

### Reprocessing Success

- ✅ **Minimum 85%** از 188 races پردازش موفق (160+ races)
- ✅ **100%** از clean races (114) پردازش موفق
- ✅ **>70%** از suspicious races (52+ از 74) پردازش موفق

### Data Quality

- ✅ Velocity range: **1.5-3.5 m/s** برای >95% races
- ✅ Total distance: **13-15.5m** برای >95% races
- ✅ Calibration RMSE: **<5cm** برای >90% frames
- ✅ Data completeness: **>85%** valid frames per race

### Scientific Validity

- ✅ Velocity مطابق IFSC world records (5.2-6.5s برای 15m)
- ✅ Distance مطابق ارتفاع دیوار (15m ±0.5m)
- ✅ نتایج reproducible (run مجدد → همان نتیجه)
- ✅ نتایج قابل مقایسه بین competitions

---

## 🎯 Deliverables نهایی

پس از تکمیل Phase 5:

1. **✅ 165+ race با داده‌های معتبر در واحد متر**
2. **✅ Pipeline یکپارچه و مستند**
3. **✅ Quality validation framework**
4. **✅ Comparison report (v1 vs v2)**
5. **✅ Dataset آماده برای paper/publication**
6. **✅ Interactive dashboard با داده‌های جدید**

---

## ⏱️ زمان‌بندی کامل (با Phase 5)

| Phase | Tasks | تخمین زمان |
|-------|-------|-----------|
| Phase 1 | Core Components | 3-4 ساعت |
| Phase 2 | Integration Pipeline | 2-3 ساعت |
| Phase 3 | Testing (3-5 نمونه) | 2-3 ساعت |
| Phase 4 | Documentation اولیه | 1 ساعت |
| **Phase 5** | **Full Reprocessing (188 races)** | **3-4 ساعت** |
| **Phase 6** | **Validation & Final Report** | **1-2 ساعت** |
| **جمع** | | **12-17 ساعت** |

---

## 📋 Checklist نهایی

### قبل از شروع پیاده‌سازی
- [x] Design document تأیید شد
- [x] Phase 5 (Full Reprocessing) به برنامه اضافه شد
- [ ] User تأیید کرد که 188 races پردازش شوند
- [ ] منابع سیستم چک شد (GPU, disk space)

### حین پیاده‌سازی
- [ ] Phase 1: Core Components ✓
- [ ] Phase 2: Integration Pipeline ✓
- [ ] Phase 3: Test با 3-5 نمونه ✓
- [ ] Phase 4: Documentation اولیه ✓

### Full Reprocessing
- [ ] Pass 1: 114 clean races ✓
- [ ] Pass 2: 74 suspicious races ✓
- [ ] Pass 3: Failed races retry ✓
- [ ] Quality validation ✓
- [ ] Summary report generated ✓

### تکمیل
- [ ] MASTER_CONTEXT.md به‌روز شد
- [ ] Git commit & push
- [ ] Pull Request ساخته شد
- [ ] Results reviewed by user

---

**پایان سند طراحی (با Phase 5)**

*این سند شامل برنامه کامل برای بازمهندسی سیستم و پردازش مجدد تمام 188 race است. تمام کدهای موجود حفظ می‌شوند و فقط یک لایه integration اضافه می‌شود.*
