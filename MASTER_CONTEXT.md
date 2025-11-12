# MASTER CONTEXT - Speed Climbing Performance Analysis
# سند راهنمای کامل پروژه تحلیل سنگنوردی سرعتی

**Last Updated**: 2025-11-12
**Purpose**: این سند برای ادامه کار در صورت قطع شدن session یا شروع مجدد در conversation جدید
**Language**: Persian (Farsi) + English

---

## 📋 وضعیت کنونی پروژه (Current Project Status)

### ✅ کارهای انجام شده (Completed Tasks)

#### 1. نصب Dependencies (Dependencies Installation)
- **تاریخ**: 2025-11-12
- **فایل‌ها**:
  - `requirements_phase1_extended.txt` - Extended dependencies برای Phase 1
  - `SETUP_FFMPEG.md` - راهنمای نصب FFmpeg (اختیاری)

- **پکیج‌های نصب شده**:
  ```
  # Video/Audio Processing
  yt-dlp==2024.8.6          # YouTube downloader
  pydub==0.25.1             # Audio manipulation
  librosa==0.10.1           # Audio analysis (beep detection)
  ffmpeg-python==0.2.0      # FFmpeg wrapper
  soundfile==0.12.1         # Audio file I/O

  # Computer Vision
  opencv-python==4.8.1.78   # (در حال نصب / being installed)
  mediapipe==0.10.8         # (در حال نصب / being installed)
  scikit-image==0.22.0

  # Tracking & Filtering
  filterpy==1.4.5           # Kalman filter

  # ML & Data
  numpy, scipy, pandas      # (در حال نصب / being installed)
  scikit-learn, torch

  # Testing
  pytest, pytest-cov, pytest-mock
  ```

- **وضعیت FFmpeg**: ❌ نصب نشده (اختیاری برای audio analysis)

#### 2. YouTube Video Downloader
- **تاریخ**: 2025-11-12
- **فایل‌های ایجاد شده**:
  - `src/utils/youtube_downloader.py` - کلاس IFSCVideoDownloader
  - `scripts/download_priority_videos.py` - اسکریپت دانلود batch
  - `configs/youtube_urls.yaml` - کانفیگ URLs (نیاز به به‌روزرسانی توسط کاربر)
  - `HOW_TO_FIND_VIDEOS.md` - راهنمای کامل جستجو و دانلود

- **قابلیت‌ها**:
  - دانلود از YouTube با کیفیت‌های مختلف (720p/1080p)
  - استخراج خودکار audio به صورت WAV
  - استخراج metadata (duration, FPS, resolution)
  - تشخیص dual-race از روی title/description
  - Progress tracking

- **نکته مهم**: کاربر باید URL واقعی ویدئوهای IFSC را در `configs/youtube_urls.yaml` قرار دهد

#### 3. Dual-Lane Detection Module ✅
- **تاریخ**: 2025-11-12
- **وضعیت**: COMPLETED & TESTED
- **فایل‌های ایجاد شده**:
  - `src/phase1_pose_estimation/dual_lane_detector.py` - ماژول اصلی (823 lines)
  - `tests/test_dual_lane_detector.py` - Unit tests (17 test cases)

- **کلاس‌های اصلی**:
  - `LaneBoundary`: مرز بین دو لاین
  - `DualLaneResult`: خروجی detection برای هر frame
  - `DualLaneDetector`: کلاس اصلی detector

- **الگوریتم**:
  1. تشخیص مرز عمودی (boundary) بین دو لاین
     - روش "fixed": مرکز فریم (0.5)
     - روش "edge": edge detection با Sobel
     - روش "motion": motion-based (future)
  2. Mask کردن هر لاین
  3. BlazePose extraction جداگانه برای هر لاین
  4. Validation که هر pose در لاین صحیح است (با COM)
  5. Kalman smoothing برای boundary (اختیاری)

- **قابلیت‌ها**:
  - Context manager support (`with` statement)
  - Statistics tracking (detection rates)
  - Visualization با `visualize_dual_lane()`
  - CLI interface: `python dual_lane_detector.py video.mp4 output.mp4`

- **تست‌ها**: ✅ 17/17 unit tests PASSING (100%)
  - LaneBoundary: 6 tests
  - DualLaneDetector: 8 tests
  - Visualization: 1 test
  - Integration: 2 tests

#### 4. Git Commits
- **Commit 1** (dd66cc9): YouTube video downloader
  ```
  feat: add YouTube video downloader and configuration
  Files: 9 changed, 933 insertions(+)
  ```

- **Commit 2** (d2e7942): Dual-lane detection system
  ```
  feat: add dual-lane detection system and master context
  Files: 5 changed, 1367 insertions(+)
  ```

- **Commit 3** (c47021c): Bug fixes and test passing
  ```
  fix: resolve dual-lane detector test failures
  - Fixed COM access (use get_keypoint() not .com attribute)
  - Fixed numpy deprecation warnings
  - All 17 tests passing
  Files: 5 changed, 16 insertions(+), 8 deletions(-)
  ```

---

## 🔧 کارهای در حال انجام (In Progress)

**هیچ کار در حال انجام نیست** - آماده برای مرحله بعد!

---

## 📝 کارهای آینده (Pending Tasks)

### 1. Race Start Detection (Audio + Motion)
**Priority**: High
**Dependencies**: FFmpeg (optional), librosa

**خروجی مورد انتظار**:
- فایل: `src/phase1_pose_estimation/race_start_detector.py`
- قابلیت تشخیص:
  - Audio: تشخیص صدای بوق شروع با librosa
  - Motion: تشخیص حرکت ناگهانی climbers
  - Fusion: ترکیب دو روش برای دقت بالاتر
- خروجی: frame_id و timestamp دقیق شروع مسابقه

### 2. Race Finish Detection (Top Touch)
**Priority**: High

**خروجی مورد انتظار**:
- فایل: `src/phase1_pose_estimation/race_finish_detector.py`
- قابلیت تشخیص:
  - تشخیص دست climber رسیدن به دکمه بالایی (hold 20)
  - محاسبه timestamp دقیق finish
  - تشخیص winner (کدام climber زودتر finish کرد)

### 3. Time-Series Plots
**Priority**: Medium
**Dependencies**: matplotlib, seaborn

**خروجی مورد انتظار**:
- فایل: `src/visualization/time_series_plots.py`
- نمودارها:
  - Vertical position vs Time
  - Horizontal position vs Time
  - Velocity vs Time
  - مقایسه همزمان دو climber (dual plot)

### 4. CSV Export (Race Period Only)
**Priority**: Medium

**خروجی مورد انتظار**:
- فایل: `src/utils/csv_exporter.py`
- فیلتر کردن فقط دوره مسابقه (از start تا finish)
- Export جداگانه برای هر climber
- فیلدها: frame_id, timestamp, COM_x, COM_y, velocity, acceleration

### 5. Comparative Report
**Priority**: Medium

**خروجی مورد انتظار**:
- فایل: `src/analysis/comparative_report.py`
- گزارش شامل:
  - Winner determination
  - Time comparison
  - Path efficiency comparison
  - Movement pattern analysis
- خروجی: PDF یا HTML

### 6. Camera Calibration (IFSC Standard)
**Priority**: High

**خروجی مورد انتظار**:
- فایل: `src/calibration/ifsc_calibration.py`
- مبدل pixel → meter بر اساس:
  - ارتفاع دیوار: 15m
  - عرض هر لاین: 3m
  - 20 hold استاندارد IFSC
- Homography matrix برای perspective correction

### 7. Integration Testing
**Priority**: High

**خروجی مورد انتظار**:
- فایل: `tests/test_integration.py`
- تست end-to-end pipeline
- تست با ویدئوهای واقعی IFSC

### 8. Notebook جدید
**خروجی مورد انتظار**:
- فایل: `notebooks/02_dual_climber_race_analysis.ipynb`
- Google Colab compatible
- مثال کامل از دانلود تا گزارش نهایی

---

## 🗂️ ساختار پروژه (Project Structure)

```
speed_climbing_analysis/
├── data/
│   ├── raw_videos/           # محل ذخیره ویدئوهای دانلود شده
│   ├── processed/            # خروجی‌های پردازش
│   └── annotations/          # برچسب‌های دستی (اگر باشد)
│
├── src/
│   ├── phase1_pose_estimation/
│   │   ├── video_processor.py
│   │   ├── blazepose_extractor.py
│   │   ├── dual_lane_detector.py        # ✅ NEW
│   │   ├── race_start_detector.py       # ⏳ TODO
│   │   └── race_finish_detector.py      # ⏳ TODO
│   │
│   ├── phase2_features/
│   │   ├── path_entropy.py
│   │   ├── gait_analysis.py
│   │   └── com_tracker.py
│   │
│   ├── calibration/
│   │   └── ifsc_calibration.py          # ⏳ TODO
│   │
│   ├── utils/
│   │   ├── youtube_downloader.py        # ✅ NEW
│   │   └── csv_exporter.py              # ⏳ TODO
│   │
│   ├── visualization/
│   │   ├── overlay.py
│   │   ├── time_series_plots.py         # ⏳ TODO
│   │   └── dashboard.py
│   │
│   └── analysis/
│       └── comparative_report.py        # ⏳ TODO
│
├── configs/
│   ├── keypoints.json
│   ├── camera_calibration.json
│   └── youtube_urls.yaml                # ✅ NEW (needs user URLs)
│
├── scripts/
│   └── download_priority_videos.py      # ✅ NEW
│
├── tests/
│   ├── test_dual_lane_detector.py       # ✅ NEW (16 tests)
│   └── test_integration.py              # ⏳ TODO
│
├── notebooks/
│   ├── 01_phase1_pose_estimation.ipynb
│   └── 02_dual_climber_race_analysis.ipynb  # ⏳ TODO
│
├── requirements.txt                     # Original
├── requirements_phase1_extended.txt     # ✅ NEW
├── SETUP_FFMPEG.md                      # ✅ NEW
├── HOW_TO_FIND_VIDEOS.md               # ✅ NEW
├── MASTER_CONTEXT.md                    # ✅ THIS FILE
├── README.md
└── prompt.md                            # System architecture
```

---

## 🎯 استراتژی توسعه (Development Strategy)

### Phase 1: Core Infrastructure (Current - ~50% Complete)
1. ✅ Dependencies setup
2. ✅ Video downloader
3. ✅ Dual-lane detection
4. ⏳ Race start/finish detection
5. ⏳ Camera calibration

### Phase 2: Analysis & Export (~0% Complete)
1. ⏳ Time-series visualization
2. ⏳ CSV export (race period only)
3. ⏳ Comparative reports

### Phase 3: Integration & Testing (~0% Complete)
1. ⏳ Integration tests
2. ⏳ End-to-end pipeline
3. ⏳ Documentation

### Phase 4: Advanced Features (Future)
1. NARX neural networks
2. Fuzzy logic system
3. Dashboard

---

## 🚀 چگونه ادامه دهیم (How to Continue)

### اگر در همین session هستید:
1. منتظر بمانید تا opencv نصب شود
2. تست‌های dual-lane را اجرا کنید
3. به سراغ Race Start Detection بروید

### اگر session جدید است:
1. این فایل (`MASTER_CONTEXT.md`) را بخوانید
2. بررسی کنید آخرین commit چه بود: `git log -1`
3. بررسی کنید کدام dependencies نصب هستند: `pip list`
4. از todo list در بالا پیروی کنید
5. ادامه دهید از جایی که کار نیمه‌تمام مانده

### دستورات مفید:
```bash
# بررسی وضعیت git
git status
git log --oneline -5

# بررسی dependencies
pip list | grep -E "(opencv|mediapipe|yt-dlp|librosa)"

# اجرای تست‌ها
pytest tests/test_dual_lane_detector.py -v

# مشاهده ساختار پروژه
tree -L 2 src/
```

---

## ⚙️ تنظیمات سیستم (System Configuration)

### محیط توسعه (Development Environment)
- **OS**: Windows
- **Python**: 3.11.6
- **NPU**: Intel ARC 15GB (available for acceleration)
- **Git**: Repository initialized
- **Branch**: main

### نکات مهم (Important Notes)
1. **Windows Console Encoding**: از emojis در print statements استفاده نکنید (UnicodeEncodeError)
2. **FFmpeg**: اختیاری است - برای audio analysis نیاز است
3. **YouTube URLs**: کاربر باید خودش URLهای واقعی پیدا کند
4. **Google Colab**: به عنوان fallback option در نظر گرفته شود

---

## 📚 منابع و مستندات (Resources & Documentation)

### مستندات داخلی:
- `prompt.md` - معماری کامل سیستم (1032 lines)
- `README.md` - راهنمای کلی پروژه
- `SETUP_FFMPEG.md` - راهنمای نصب FFmpeg
- `HOW_TO_FIND_VIDEOS.md` - راهنمای پیدا کردن ویدئوها

### External Resources:
- IFSC Official: https://www.ifsc-climbing.org/
- IFSC YouTube: https://www.youtube.com/@sportclimbing
- MediaPipe BlazePose: https://arxiv.org/abs/2006.10204
- IFSC Standards: 15m height, 3m width, 5° overhang

---

## 🐛 مشکلات شناخته شده (Known Issues)

1. ~~**FFmpeg not installed**~~: ✅ نصب شده توسط کاربر
2. **YouTube video URLs**: placeholders هستند، کاربر باید URLهای واقعی IFSC وارد کند
3. ~~**Windows encoding**~~: ✅ حل شد - از ASCII استفاده می‌کنیم
4. ~~**opencv installation**~~: ✅ نصب شد و تست شد
5. ~~**mediapipe installation**~~: ✅ نصب شد و تست شد
6. **MediaPipe cleanup warning**: warning جزئی در __del__ (غیرمهم)

---

## 💡 نکات برای ادامه کار (Tips for Continuation)

### برای Claude در session بعدی:
1. این فایل را اول بخوان: `Read("MASTER_CONTEXT.md")`
2. آخرین commit را بررسی کن: `git log -1`
3. todo list را به‌روزرسانی کن با TodoWrite
4. از همان جایی که قطع شده ادامه بده
5. بعد از هر مرحله مهم، این فایل را update کن
6. بعد از هر commit، این فایل را commit کن

### برای کاربر:
1. اگر اینترنت قطع شد، نگران نباشید - این سند همه چیز را نگه داشته
2. می‌توانید با خواندن این سند به Claude دقیقاً بگویید از کجا ادامه دهد
3. برای تست سریع: `pytest tests/ -v`
4. برای commit جدید: `git add -A && git commit -m "your message"`

---

## 📊 Progress Tracker

```
Phase 1: Core Infrastructure
[████████████░░░░░░░░] 60%

├─ Dependencies Setup         [████████████████████] 100% ✅
├─ Video Downloader          [████████████████████] 100% ✅
├─ Dual-Lane Detection       [████████████████████] 100% ✅ (17/17 tests pass)
├─ Race Start Detection      [░░░░░░░░░░░░░░░░░░░░]   0% ⏸️ NEXT
├─ Race Finish Detection     [░░░░░░░░░░░░░░░░░░░░]   0% ⏸️
└─ Camera Calibration        [░░░░░░░░░░░░░░░░░░░░]   0% ⏸️

Phase 2: Analysis & Export
[░░░░░░░░░░░░░░░░░░░░] 0%

Phase 3: Integration & Testing
[░░░░░░░░░░░░░░░░░░░░] 0%
```

---

## 🔄 آخرین به‌روزرسانی (Last Update Log)

**2025-11-12 09:30 UTC**
- ✅ Created MASTER_CONTEXT.md
- ✅ Committed YouTube downloader (dd66cc9)
- ✅ Committed dual-lane detector (d2e7942)
- ✅ User installed FFmpeg + MediaPipe
- ✅ Fixed test failures (c47021c)
- ✅ All 17 tests passing (100%)
- ✅ Dual-Lane Detection module COMPLETE
- 🎯 Next: Race Start Detection (Audio + Motion)

---

## 📞 تماس با توسعه‌دهنده (Contact)

اگر سوالی دارید یا نیاز به کمک هست:
- Issues در repository بسازید
- مستندات `prompt.md` را مطالعه کنید
- این فایل را به‌روز نگه دارید

---

**END OF MASTER CONTEXT**
این سند باید بعد از هر تغییر مهم به‌روزرسانی شود.
