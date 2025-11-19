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
| Phase 3 | Testing | 2-3 ساعت |
| Phase 4 | Documentation | 1 ساعت |
| **جمع** | | **8-11 ساعت** |

---

## ✅ مراحل بعدی

1. ✅ بررسی و تأیید این طراحی
2. ⏭️ شروع پیاده‌سازی Phase 1
3. ⏭️ تست با یک ویدئو نمونه
4. ⏭️ Batch processing روی 114 races
5. ⏭️ Validation و مستندسازی

---

**پایان سند طراحی**

*این سند خلاصه‌ای از تحلیل سیستم موجود و طراحی معماری جدید است. تمام کدهای موجود حفظ می‌شوند و فقط یک لایه integration اضافه می‌شود.*
