# MASTER CONTEXT - Speed Climbing Performance Analysis
# سند راهنمای کامل پروژه تحلیل سنگنوردی سرعتی

**Last Updated**: 2025-11-14
**Purpose**: این سند برای ادامه کار در صورت قطع شدن session یا شروع مجدد در conversation جدید
**Language**: Persian (Farsi) + English

---

## 🚀 Quick Start for New Session (شروع سریع برای Session جدید)

> **برای شروع کار در https://claude.ai/code جدید، این بخش را اول بخوانید!**

### آخرین وضعیت (2025-11-14):
✅ **Manual Race Segmentation: 100% COMPLETE**
- 188 مسابقه از 5 فاینال استخراج شد
- همه فایل‌های ویدئو در `data/race_segments/` موجود است
- همه YAML configs آماده در `configs/race_timestamps/`

### مرحله فعلی: Phase 2 - Pose Estimation & Analysis
**آماده برای شروع!** همه race clips موجود است.

### فایل‌های کلیدی برای ادامه:
1. **این فایل**: [MASTER_CONTEXT.md](MASTER_CONTEXT.md) - سند کامل پروژه
2. **راهنمای Segmentation**: [docs/MANUAL_SEGMENTATION_GUIDE.md](docs/MANUAL_SEGMENTATION_GUIDE.md)
3. **Race Segments**: `data/race_segments/` (188 MP4 files + metadata)
4. **Configs**: `configs/race_timestamps/*.yaml` (5 competitions)
5. **Parser**: [scripts/parse_timestamps_to_yaml.py](scripts/parse_timestamps_to_yaml.py)
6. **Batch Script**: [scripts/batch_segment_competitions.py](scripts/batch_segment_competitions.py)

### داده‌های موجود:
```
data/
├── raw_videos/                    # 5 ویدئوی اصلی (2-3 ساعته)
│   ├── Speed_finals_Seoul_2024.mp4
│   ├── Speed_finals_Villars_2024.mp4
│   ├── Speed_finals_Chamonix_2024.mp4
│   ├── Speed_finals_Innsbruck_2024.mp4
│   └── Speed_finals_Zilina_2025.mp4
│
└── race_segments/                 # 188 مسابقه استخراج شده (5-40s هر کدام)
    ├── seoul_2024/                # 31 races
    ├── villars_2024/              # 24 races
    ├── chamonix_2024/             # 32 races
    ├── innsbruck_2024/            # 32 races
    └── zilina_2025/               # 69 races
```

### چک کردن وضعیت:
```powershell
# اجرای اسکریپت مانیتورینگ
powershell -ExecutionPolicy Bypass -File check_progress.ps1

# باید نمایش دهد: 188/188 races (100%)
```

### مرحله بعدی پیشنهادی:
1. **Pose Extraction**: استخراج BlazePose keypoints از 188 race clip
2. **Performance Metrics**: محاسبه سرعت، timing، smoothness
3. **Visualization**: نمودارهای مقایسه‌ای
4. **IFSC Calibration**: تبدیل pixel به متر با 20 گیره استاندارد

**راهنمای دقیق**: بخش "🎯 مراحل بعدی (Next Steps)" را در پایین این سند ببینید.

### 📦 مدیریت فایل‌های ویدئو (Video Files Management) - Updated 2025-11-14

**استراتژی Multi-Environment**: پروژه در سه محیط مختلف sync می‌شود

**محیط‌های موجود**:
1. **Gitea (Primary - Private)**: https://gitea.airano.ir/dev/Speed-Climbing-Performance-Analysis
2. **GitHub (Public - Lightweight)**: https://github.com/languageofearthcom-oss/Speed-Climbing-Performance-Analysis
3. **UI claude.ai/code**: Clone از GitHub

**وضعیت فایل‌های ویدئو**:
- **Local (Google Drive)**: همه فایل‌ها (~711 MB)
  - ❌ 5 raw finals videos حذف شدند (قابل دانلود مجدد از YouTube)
  - ✅ 188 race segments (705 MB) - آماده برای commit
  - ✅ 5 sample clips (6 MB)
  - ✅ همه metadata files

- **Gitea**:
  - ✅ Committed (commit 4110df5) - 706MB data
  - ⚠️ Push pending (timeout - نیاز به manual push یا chunked push)

- **GitHub**:
  - ✅ همه کد و configs synced
  - ❌ Race segments MP4 ignored (gitignore)
  - ✅ همه metadata tracked

**برای استفاده در پروژه جدید**:
1. **گزینه 1 - کپی از Google Drive**:
   ```bash
   # همه فایل‌ها قبلاً در G:\My Drive موجود است
   # فقط path را در کد اصلاح کنید
   ```

2. **گزینه 2 - استفاده از metadata**:
   ```bash
   # اگر فقط به metadata نیاز دارید:
   # - configs/race_timestamps/*.yaml (در git)
   # - data/race_segments/*/summary.json (کوچک - می‌توان commit کرد)
   ```

3. **گزینه 3 - بازسازی از YAML**:
   ```bash
   # اگر فایل‌ها گم شدند، می‌توانید دوباره بسازید:
   python scripts/batch_segment_competitions.py
   # زمان: ~2-3 ساعت برای همه
   ```

**نکته مهم برای Git LFS** (اگر می‌خواهید ویدئوها را version control کنید):
```bash
# نصب Git LFS
git lfs install

# Track video files
git lfs track "*.mp4"
git add .gitattributes

# سپس می‌توانید commit کنید (اما توصیه نمی‌شود - حجم زیاد)
```

**توصیه**: فایل‌های ویدئو را در Google Drive نگه دارید و فقط کد + configs را در git قرار دهید.

---

## 🔄 Multi-Environment Sync Workflow (راهنمای همگام‌سازی) - New 2025-11-14

### نمای کلی:
```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Gitea     │◄───────►│  Local PC   │◄───────►│  GitHub     │
│  (Primary)  │         │  (You)      │         │  (Public)   │
│  Full Data  │         │             │         │ Lightweight │
└─────────────┘         └─────────────┘         └─────────────┘
                              ▲
                              │
                              ▼
                        ┌─────────────┐
                        │ claude.ai/  │
                        │    code     │
                        └─────────────┘
```

### فایل‌های راهنما:
- **[SYNC_WORKFLOW.md](SYNC_WORKFLOW.md)**: راهنمای جامع sync (30+ صفحه)
- **sync.bat** / **sync.sh**: اسکریپت‌های automation

### دستورات سریع:

**Windows**:
```bash
sync.bat          # Full sync (pull + push + verify)
sync.bat pull     # Pull only
sync.bat push     # Push only
```

**Linux/Mac**:
```bash
chmod +x sync.sh
./sync.sh         # Full sync
```

### Workflow روزانه:

**صبح** (شروع کار):
```bash
git pull origin main      # از Gitea
git push github main      # به GitHub
```

**عصر** (پایان کار):
```bash
git add -A
git commit -m "توضیحات"
git push origin main      # به Gitea
git push github main      # به GitHub
```

### Remotes فعلی:
```
origin: https://gitea.airano.ir/dev/Speed-Climbing-Performance-Analysis.git
github: https://github.com/languageofearthcom-oss/Speed-Climbing-Performance-Analysis.git
```

برای جزئیات بیشتر، [SYNC_WORKFLOW.md](SYNC_WORKFLOW.md) را ببینید.

---

### 💡 بهبودها و نکات مهم (Improvements & Key Notes)

**✅ موارد انجام شده**:
1. **Automated Progress Monitoring**: اسکریپت `check_progress.ps1` برای tracking real-time
2. **Late Start Handling**: buffer اتوماتیک 3s برای مسابقات با شروع دیرهنگام
3. **Timestamp Corrections**: 188 مسابقه با اصلاحات دقیق timing
4. **Race Deletion**: حذف خودکار races ناقص (Seoul race 15, Zilina races 13/51/55)
5. **Auto Renumbering**: renumber کردن اتوماتیک بعد از حذف races
6. **Comprehensive Metadata**: JSON metadata برای هر race با athlete info
7. **Session Continuity**: Quick Start section برای شروع در session جدید

**🔧 بهبودهای احتمالی آینده**:
1. **Batch Pose Extraction**: پردازش موازی 188 race با multiprocessing
2. **Resume Capability**: skip کردن فایل‌های پردازش شده (idempotent)
3. **Quality Validation**: بررسی اتوماتیک کیفیت clips (duration, resolution)
4. **Auto Timestamp Detection**: یادگیری timestamps از patterns (ML-based)
5. **Cloud Storage Integration**: sync اتوماتیک با Google Drive API
6. **Video Compression**: کاهش حجم با H.265/HEVC encoding

**⚠️ نکات مهم**:
- همیشه قبل از regenerate کردن، backup بگیرید
- check_progress.ps1 را برای مانیتورینگ استفاده کنید
- فایل‌های temp_*.mp4 را بعد از هر competition پاک کنید
- YAML configs را version control کنید (کوچک و مهم)
- ویدئوها را در Google Drive نگه دارید (نه در git)

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

#### 4. Video Dataset Downloaded ✅
- **تاریخ**: 2025-11-13
- **مکان**: `data/raw_videos/`
- **تعداد**: 11 ویدئو، حجم کل: ~4.3 GB

**ویدئوهای اصلی (Competition Finals - Dual-lane)**:
1. `Speed_finals_Zilina_2025.mp4` - 3 ساعت (1280x720@30fps)
2. `Speed_finals_Seoul_2024.mp4` - 2.1 ساعت (1280x720@30fps)
3. `Speed_finals_Villars_2024.mp4` - 2.1 ساعت (1280x720@30fps)
4. `Speed_finals_Innsbruck_2024.mp4` - 1.6 ساعت (1280x720@30fps)
5. `Speed_finals_Chamonix_2024.mp4` - 1.6 ساعت (1280x720@30fps)

**ویدئوهای کمکی**:
- 1 ویدئوی compilation (10 fastest times)
- 5 کلیپ کوتاه social media (12-24 ثانیه)

**فایل‌های همراه**:
- WAV audio files (برای race start detection)
- JSON metadata files
- YouTube info files

**چالش‌ها شناسایی شده**:
- ✅ ویدئوها بسیار طولانی (1.6-3 ساعت) - نیاز به race segmentation
- ✅ دوربین حرکت می‌کند (camera motion) - نیاز به adaptive calibration
- ✅ فرمت‌های مختلف (dual/single climber) - نیاز به auto-detection

#### 5. IFSC Standards Documentation ✅
- **تاریخ**: 2025-11-13
- **فایل**: `docs/IFSC_Speed_Licence_Rules.pdf`
- **محتوا**:
  - 20 گیره استاندارد با موقعیت‌های دقیق (grid coordinates)
  - ابعاد دیوار: 15m ارتفاع × 3m عرض × 5° overhang
  - فاصله گیره‌ها: 125mm (perfect برای calibration!)
  - نقشه کامل panel grid system

**استفاده استراتژیک**:
- کالیبراسیون خودکار با hold spacing
- Validation pose estimation با موقعیت‌های شناخته‌شده
- Hold-by-hold performance metrics
- Path optimization analysis
- حل مشکل camera motion با re-calibration

#### 6. Race Segmentation System ✅ (Priority 1)
- **تاریخ**: 2025-11-13
- **وضعیت**: COMPLETED + IMPROVED (sliding window)
- **فایل‌های ایجاد شده**:
  - `src/phase1_pose_estimation/race_start_detector.py` (490 lines)
  - `src/phase1_pose_estimation/race_finish_detector.py` (460 lines)
  - `src/utils/race_segmenter.py` (495 lines) - Updated with sliding window
  - `docs/USER_TESTING_GUIDE.md` - راهنمای جامع تست برای کاربر (Farsi+English)

**قابلیت‌های race_start_detector.py**:
- `AudioBeepDetector`: تشخیص صدای بوق شروع (800-1200 Hz) با librosa + FFT
- `MotionStartDetector`: تشخیص حرکت ناگهانی با Optical Flow
- `RaceStartDetector`: Fusion method (audio + motion)
- 3 روش: audio, motion, fusion
- CLI interface

**قابلیت‌های race_finish_detector.py**:
- `TopButtonDetector`: تشخیص تغییر رنگ دکمه بالایی
- `PoseBasedFinishDetector`: تشخیص دست رسیدن به top
- `RaceFinishDetector`: Combined detection
- `detect_winner()`: تعیین برنده
- CLI interface

**قابلیت‌های race_segmenter.py**:
- Integration start + finish detectors
- **Sliding window approach** برای تشخیص چند مسابقه (multi-race detection) ✅
- استخراج race clips از ویدئوهای طولانی (1-3 ساعت)
- تولید metadata JSON
- Validation (min/max duration) - قابل تنظیم
- Buffer zones قابل تنظیم
- min_gap_between_races برای جلوگیری از duplicates
- CLI interface با پارامترهای کامل

**خروجی**: از ویدئو 2 ساعته → 20-30 کلیپ مسابقه (هر کدام 5-15 ثانیه)

**Improvements 2025-11-13**:
- ✅ Sliding window (60s windows) برای multi-race detection
- ✅ CLI args: --min-duration, --max-duration, --min-gap
- ✅ False positive filtering با duration validation
- ✅ Tested: Successfully extracts multiple races from compilation videos

#### 7. Manual Race Segmentation System ✅ (2025-11-14)
- **وضعیت**: COMPLETED - استفاده از timestamps دستی برای دقت بالا
- **فایل‌های ایجاد شده**:
  - `scripts/parse_timestamps_to_yaml.py` (~1450 lines) - تبدیل timestamps به YAML برای 5 مسابقه
  - `src/utils/manual_race_segmenter.py` (~550 lines) - استخراج با timestamps دستی + late_start handling
  - `scripts/batch_segment_competitions.py` (~130 lines) - پردازش batch
  - `docs/MANUAL_SEGMENTATION_GUIDE.md` - راهنمای جامع کاربر (Version 3.0)
  - `configs/race_timestamps/*.yaml` - 5 فایل config (Seoul, Villars, Chamonix, Innsbruck, Zilina)

**قابلیت‌های parse_timestamps_to_yaml.py**:
- پارس timestamps از متن فارسی
- تولید YAML config با اطلاعات کامل ورزشکاران
- اصلاح خودکار end_time برای مسابقات زودتر تمام شده (+4s/+5s/+8s)
- اضافه کردن start_time برای مسابقات با شروع دیرهنگام (+20s)
- علامت‌گذاری `late_start` flag برای مسابقات با شروع تأخیری
- حذف مسابقات invalid (مثل false starts)
- خروجی: **188 مسابقه** برای 5 فاینال:
  - Seoul 2024: 31 مسابقه (race 15 حذف شد)
  - Villars 2024: 24 مسابقه (rerun 1/8 final men)
  - Chamonix 2024: 32 مسابقه (IFSC World Cup)
  - Innsbruck 2024: 32 مسابقه (European Cup)
  - Zilina 2025: 69 مسابقه (European Youth Championships - races 13, 51, 55 حذف شد)

**قابلیت‌های manual_race_segmenter.py**:
- برش frame-accurate با ffmpeg
- Buffer قابل تنظیم (default: 1.5s قبل و بعد)
- **Late start handling**: خودکار 3s buffer برای مسابقات با `late_start: true`
- Optional detection refinement (فعلاً disabled برای سرعت)
- تولید metadata کامل با اطلاعات ورزشکاران
- Winner detection (TODO)
- Progress tracking

**مزایا**:
- ⚡ سریع: ~30 ثانیه per race (با --no-refine)
- ✅ دقیق: timestamps manual = دقت 100%
- 📊 Metadata غنی: athlete info, country, round, bib colors
- 🎯 No false positives
- 🔄 Smart buffer adjustment: 3s برای late starts

**نکات مهم**:
- معمولاً قبل از شروع 3 بوق می‌زند (بوق سوم = شروع)، اما گاهی 1، 2 یا هیچ بوقی نیست
- برخی مسابقات زودتر تمام می‌شوند (parser خودکار +4-8s اضافه می‌کند)
- برخی مسابقات با تأخیر شروع می‌شوند (parser خودکار 3s buffer می‌دهد)
- Seoul Race 15 حذف شد (false start - خیلی کوتاه)
- Zilina: دیواره لغزنده، بسیاری از ورزشکاران افتادند

#### 8. Git Commits
- **Commit 1** (dd66cc9): YouTube video downloader
- **Commit 2** (d2e7942): Dual-lane detection system
- **Commit 3** (c47021c): Bug fixes and test passing
- **Commit 4** (5019acc): Enhanced YouTube downloader
- **Commit 5** (dd3370d): Documentation update
- **Commit 6** (aafa060): Documentation with video inventory (2025-11-13)
  ```
  docs: update MASTER_CONTEXT with video inventory and implementation plan
  - Add comprehensive video inventory (11 videos, 4.3 GB)
  - Document IFSC standards integration strategy
  - Add 5-priority implementation roadmap
  - Update project structure
  Files: 4 changed, 449 insertions(+), 124 deletions(-)
  ```
- **Commit 7** (238c08b): Priority 1 - Race Segmentation System (2025-11-13)
  ```
  feat: implement Priority 1 - Race Segmentation System
  - race_start_detector.py (490 lines)
  - race_finish_detector.py (460 lines)
  - race_segmenter.py (380 lines)
  Files: 3 changed, 1461 insertions(+)
  ```
- **Commit 8** (57d164f): docs: update MASTER_CONTEXT with Priority 1 completion and cleanup (2025-11-13)
- **Commit 9** (4b169da): test: add Priority 1 test results and validation (2025-11-13)
- **Commit 10** (d27fbbd): feat: add sliding window multi-race detection to race segmenter (2025-11-13)
- **Commit 11** (3efc78b): feat: implement manual race segmentation system (2025-11-14)
  ```
  feat: implement manual race segmentation system
  - parse_timestamps_to_yaml.py: Parse manual timestamps to YAML (Seoul fixed)
  - manual_race_segmenter.py: Frame-accurate extraction with ffmpeg
  - batch_segment_competitions.py: Batch processing script
  - MANUAL_SEGMENTATION_GUIDE.md: Comprehensive user guide (Farsi+English)
  - 3 YAML configs generated (Seoul: 31, Villars: 24, Chamonix: 32)
  Files: 7 changed, ~2000 insertions
  ```

---

## 🔧 کارهای در حال انجام (In Progress)

### ✅ Manual Race Segmentation - COMPLETED (2025-11-14)
- ✅ Parser و YAML configs ساخته شد (5 مسابقه، 188 race)
- ✅ Seoul 2024: COMPLETED (31 مسابقه)
- ✅ Villars 2024: COMPLETED (24 مسابقه)
- ✅ Chamonix 2024: COMPLETED (32 مسابقه)
- ✅ Innsbruck 2024: COMPLETED (32 مسابقه)
- ✅ Zilina 2025: COMPLETED (69 مسابقه - 3 races removed)
- ✅ **Total: 188/188 races extracted successfully (100%)**
- ✅ Timestamps corrections applied:
  - Seoul: +5s corrections, race 15 removed
  - Villars: +4s/+5s corrections, late_start flags
  - Chamonix: +5s/+8s corrections, late_start flags
  - Innsbruck: +5s/+8s/+20s corrections, late_start flags
  - Zilina: -4s/-6s/-10s start corrections, +10s end extensions, races 13/51/55 removed
- 📖 راهنمای کامل کاربر: [docs/MANUAL_SEGMENTATION_GUIDE.md](docs/MANUAL_SEGMENTATION_GUIDE.md) (Version 4.0)

**Output Location**: `data/race_segments/` (188 MP4 clips + 188 metadata JSONs + 5 summary JSONs)

---

## 📝 برنامه پیشنهادی پیاده‌سازی (Implementation Roadmap)

### Priority 1: Race Segmentation System ✅ COMPLETED
**هدف**: استخراج مسابقات 5-15 ثانیه‌ای از ویدئوهای 2-3 ساعته
**وضعیت**: 100% - Ready for testing

#### Task 1.1: Race Start Detection ✅
**فایل**: `src/phase1_pose_estimation/race_start_detector.py` (490 lines)
**قابلیت‌ها**:
- Audio-based: تشخیص صدای بوق شروع با librosa + FFT analysis
- Motion-based: تشخیص حرکت ناگهانی climbers با optical flow
- Fusion: ترکیب هر دو روش برای دقت بالا
- خروجی: RaceStartResult با frame_id و timestamp دقیق

**CLI Usage**:
```bash
python src/phase1_pose_estimation/race_start_detector.py video.mp4 --method fusion
```

#### Task 1.2: Race Finish Detection ✅
**فایل**: `src/phase1_pose_estimation/race_finish_detector.py` (460 lines)
**قابلیت‌ها**:
- TopButtonDetector: تشخیص تغییر رنگ دکمه بالایی (visual)
- PoseBasedFinishDetector: تشخیص دست رسیدن به top (pose-based)
- تشخیص winner (کدام climber اول finish کرد)
- Combined detection برای دقت بالا

**CLI Usage**:
```bash
python src/phase1_pose_estimation/race_finish_detector.py video.mp4 --lane left
```

#### Task 1.3: Race Segmenter ✅ + Sliding Window ✅
**فایل**: `src/utils/race_segmenter.py` (495 lines) - Updated 2025-11-13
**قابلیت‌ها**:
- Integration کامل start + finish detectors
- **Sliding window approach** (60s windows) برای multi-race detection ✅
- اسکن و استخراج تمام مسابقات از ویدئو طولانی
- ذخیره در `data/race_segments/`
- تولید metadata JSON برای هر race
- Validation: min/max duration - **fully configurable via CLI** ✅
- Buffer zones: قبل و بعد از race - configurable
- min_gap_between_races: فاصله minimum بین مسابقات (جلوگیری از duplicates)

**CLI Usage**:
```bash
# Basic usage
python src/utils/race_segmenter.py "data/raw_videos/video.mp4" \
  --output-dir "data/race_segments" \
  --max-races 5 \
  --buffer-before 1.0 \
  --buffer-after 1.0

# Advanced: با تنظیم کامل thresholds (برای compilation videos)
python src/utils/race_segmenter.py "data/raw_videos/video.mp4" \
  --output-dir "data/race_segments" \
  --max-races 10 \
  --start-method motion \
  --finish-method visual \
  --min-duration 2.0 \
  --max-duration 20.0 \
  --min-gap 20.0 \
  --metadata-only
```

**خروجی**: از Seoul_2024 (2.1 ساعت) → 20-30 کلیپ مسابقه (5-15 ثانیه هر کدام)

**نکته مهم**: حداکثر مدت مسابقه (max_race_duration) روی 15 ثانیه تنظیم شده تا مسابقات طولانی‌تر را هم پوشش دهد.

---

### Priority 2: IFSC Standards Integration 🔧
**هدف**: استفاده از گیره‌های استاندارد برای calibration و validation

#### Task 2.1: IFSC Route Map Parser
**فایل**: `src/calibration/ifsc_route_map.py`
**قابلیت‌ها**:
- پارس PDF و استخراج موقعیت 20 گیره
- تولید dictionary: `{hold_num: (panel, grid_x, grid_y, meter_x, meter_y)}`
- ذخیره در `configs/ifsc_route_coordinates.json`
- محاسبه pixel coordinates از meter coordinates

#### Task 2.2: Hold Detector
**فایل**: `src/phase1_pose_estimation/hold_detector.py`
**قابلیت‌ها**:
- Template matching برای گیره‌های قرمز IFSC
- Color-based detection (HSV thresholding)
- تطبیق با نقشه استاندارد (match detected → expected)
- خروجی: لیست detected holds در هر frame با confidence

#### Task 2.3: Camera Motion Detector
**فایل**: `src/utils/camera_motion_detector.py`
**قابلیت‌ها**:
- تشخیص خودکار: STATIC یا MOVING camera
- Optical flow analysis روی background
- Feature tracking stability metric
- اضافه flag به metadata: `"camera_type": "static"|"moving"`

---

### Priority 3: Smart Calibration System 📐
**هدف**: کالیبراسیون هوشمند برای هر دو نوع دوربین

#### Task 3.1: Static Camera Calibration
**فایل**: `src/calibration/static_camera_calibration.py`
**قابلیت‌ها**:
- Homography matrix از detected holds
- One-time calibration (first frame)
- Pixel → meter converter با دقت بالا
- Perspective correction

#### Task 3.2: Moving Camera Calibration
**فایل**: `src/calibration/moving_camera_calibration.py`
**قابلیت‌ها**:
- Adaptive per-frame calibration
- استفاده از visible holds برای re-calibration
- Normalized coordinates (0-1) fallback
- Tracking scale changes

#### Task 3.3: Unified Calibration Interface
**فایل**: `src/calibration/ifsc_calibration.py`
**قابلیت‌ها**:
- Auto-detect camera type و انتخاب strategy
- Factory pattern: `create_calibrator(camera_type)`
- Integration با hold detector
- خروجی: `CalibrationResult` با pixel↔meter converters

---

### Priority 4: Analysis & Reporting 📊
**هدف**: تحلیل performance و تولید گزارش‌های مقایسه‌ای

#### Task 4.1: Performance Metrics
**فایل**: `src/analysis/performance_metrics.py`
**متریک‌ها**:
- Hold-by-hold timing (زمان رسیدن به هر گیره 1-20)
- Velocity profile (speed vs time/height)
- Acceleration peaks
- Path efficiency (deviation از خط مستقیم)
- Movement smoothness (jerk analysis)

#### Task 4.2: Time-Series Visualization
**فایل**: `src/visualization/time_series_plots.py`
**نمودارها**:
- Vertical position vs Time
- Horizontal position vs Time
- Velocity vs Time
- Side-by-side dual climber comparison
- Animated trajectory plot

#### Task 4.3: CSV Exporter
**فایل**: `src/utils/csv_exporter.py`
**قابلیت‌ها**:
- فیلتر race period only (start→finish)
- Export per-climber
- فیلدها: frame_id, timestamp, COM_x, COM_y, velocity, acceleration, current_hold
- Support برای metric vs pixel coordinates

#### Task 4.4: Comparative Report Generator
**فایل**: `src/analysis/comparative_report.py`
**گزارش شامل**:
- Winner determination
- Time comparison (overall + per-section)
- Path efficiency comparison
- Hold-by-hold comparison
- Statistical summary
- خروجی: HTML report با charts

---

### Priority 5: Integration & Testing 🧪
**هدف**: pipeline کامل end-to-end

#### Task 5.1: Integration Pipeline
**فایل**: `src/phase1_pose_estimation/pipeline.py`
**قابلیت‌ها**:
- یکپارچه‌سازی تمام components
- Auto-workflow: video → races → poses → calibration → metrics → report
- Progress tracking
- Error handling و recovery
- CLI interface

#### Task 5.2: End-to-End Tests
**فایل**: `tests/test_integration.py`
**تست‌ها**:
- Full pipeline با sample video
- Validation با ground truth times
- Performance benchmarks

#### Task 5.3: Demo Notebook
**فایل**: `notebooks/02_dual_climber_race_analysis.ipynb`
**محتوا**:
- مثال کامل از دانلود تا گزارش
- Google Colab compatible
- Interactive visualizations
- مقایسه 2 climber واقعی

---

## 🗂️ ساختار پروژه (Project Structure)

```
speed_climbing_analysis/
├── data/
│   ├── raw_videos/                      # ✅ ویدئوهای دانلود شده (11 videos, 4.3GB)
│   │   ├── *.mp4                        # (gitignored - too large)
│   │   ├── *.wav                        # Audio files (gitignored)
│   │   ├── *_metadata.json              # (tracked - small)
│   │   └── *.info.json                  # YouTube metadata (tracked)
│   │
│   ├── race_segments/                   # ⏳ کلیپ‌های استخراج شده (5-10 sec each)
│   ├── processed/                       # خروجی‌های پردازش (gitignored)
│   ├── calibration/                     # داده‌های calibration (gitignored)
│   └── annotations/                     # برچسب‌های دستی (اگر باشد)
│
├── src/
│   ├── phase1_pose_estimation/
│   │   ├── video_processor.py
│   │   ├── blazepose_extractor.py
│   │   ├── dual_lane_detector.py        # ✅ COMPLETE (823 lines, 17 tests)
│   │   ├── race_start_detector.py       # 🎯 PRIORITY 1 (in progress)
│   │   ├── race_finish_detector.py      # 🎯 PRIORITY 1
│   │   ├── hold_detector.py             # ⏳ PRIORITY 2
│   │   └── pipeline.py                  # ⏳ PRIORITY 5
│   │
│   ├── phase2_features/
│   │   ├── path_entropy.py
│   │   ├── gait_analysis.py
│   │   └── com_tracker.py
│   │
│   ├── calibration/
│   │   ├── ifsc_route_map.py            # ⏳ PRIORITY 2
│   │   ├── static_camera_calibration.py # ⏳ PRIORITY 3
│   │   ├── moving_camera_calibration.py # ⏳ PRIORITY 3
│   │   └── ifsc_calibration.py          # ⏳ PRIORITY 3 (unified)
│   │
│   ├── utils/
│   │   ├── youtube_downloader.py        # ✅ COMPLETE
│   │   ├── race_segmenter.py            # 🎯 PRIORITY 1
│   │   ├── camera_motion_detector.py    # ⏳ PRIORITY 2
│   │   └── csv_exporter.py              # ⏳ PRIORITY 4
│   │
│   ├── visualization/
│   │   ├── overlay.py
│   │   ├── time_series_plots.py         # ⏳ PRIORITY 4
│   │   └── dashboard.py
│   │
│   └── analysis/
│       ├── performance_metrics.py       # ⏳ PRIORITY 4
│       └── comparative_report.py        # ⏳ PRIORITY 4
│
├── configs/
│   ├── keypoints.json
│   ├── camera_calibration.json
│   ├── youtube_urls.yaml                # ✅ (user filled)
│   └── ifsc_route_coordinates.json      # ⏳ PRIORITY 2 (from PDF)
│
├── scripts/
│   └── download_priority_videos.py      # ✅ COMPLETE
│
├── tests/
│   ├── test_dual_lane_detector.py       # ✅ COMPLETE (17 tests passing)
│   ├── test_race_detector.py            # ⏳ PRIORITY 1
│   ├── test_calibration.py              # ⏳ PRIORITY 3
│   └── test_integration.py              # ⏳ PRIORITY 5
│
├── notebooks/
│   ├── 01_phase1_pose_estimation.ipynb
│   └── 02_dual_climber_race_analysis.ipynb  # ⏳ PRIORITY 5
│
├── docs/                                # ✅ NEW
│   ├── IFSC_Speed_Licence_Rules.pdf     # ✅ (moved from Desktop)
│   └── implementation_notes.md          # Technical decisions
│
├── requirements.txt                     # Original
├── requirements_phase1_extended.txt     # ✅ Extended deps
├── SETUP_FFMPEG.md                      # ✅ FFmpeg guide
├── HOW_TO_FIND_VIDEOS.md               # ✅ Video search guide
├── MASTER_CONTEXT.md                    # ✅ THIS FILE
├── .gitignore                           # ✅ Updated for large files
├── README.md
└── prompt.md                            # System architecture (1032 lines)
```

---

## 🎯 استراتژی توسعه (Development Strategy)

### Phase 1: Core Infrastructure (~65% Complete)
1. ✅ Dependencies setup (100%)
2. ✅ Video downloader (100%)
3. ✅ Video dataset collected (11 videos)
4. ✅ IFSC standards documented (PDF)
5. ✅ Dual-lane detection (100% - 17/17 tests)
6. 🎯 Race start/finish detection (0% - CURRENT)
7. ⏳ IFSC route map parser (0%)
8. ⏳ Hold detector (0%)
9. ⏳ Camera motion detector (0%)
10. ⏳ Calibration system (0%)

### Phase 2: Analysis & Export (~0% Complete)
1. ⏳ Performance metrics calculator
2. ⏳ Time-series visualization
3. ⏳ CSV export (race period only)
4. ⏳ Comparative reports

### Phase 3: Integration & Testing (~0% Complete)
1. ⏳ Integration pipeline
2. ⏳ Integration tests
3. ⏳ End-to-end testing
4. ⏳ Demo notebook

### Phase 4: Advanced Features (Future)
1. NARX neural networks
2. Fuzzy logic system
3. Interactive dashboard
4. Real-time processing

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
[██████████████░░░░░░] 70%

├─ Dependencies Setup            [████████████████████] 100% ✅
├─ Video Downloader             [████████████████████] 100% ✅
├─ Video Dataset                [████████████████████] 100% ✅ (11 videos, 4.3GB)
├─ IFSC Standards Doc           [████████████████████] 100% ✅ (PDF)
├─ Dual-Lane Detection          [████████████████████] 100% ✅ (17/17 tests)
├─ Race Start Detection         [████████████████████] 100% ✅ (490 lines)
├─ Race Finish Detection        [████████████████████] 100% ✅ (460 lines)
├─ Race Segmenter               [████████████████████] 100% ✅ (380 lines)
├─ IFSC Route Map Parser        [░░░░░░░░░░░░░░░░░░░░]   0% 🎯 NEXT
├─ Hold Detector                [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
├─ Camera Motion Detector       [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
└─ Calibration System           [░░░░░░░░░░░░░░░░░░░░]   0% ⏳

Phase 2: Analysis & Export
[░░░░░░░░░░░░░░░░░░░░] 0%

├─ Performance Metrics          [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
├─ Time-Series Plots            [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
├─ CSV Exporter                 [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
└─ Comparative Report           [░░░░░░░░░░░░░░░░░░░░]   0% ⏳

Phase 3: Integration & Testing
[░░░░░░░░░░░░░░░░░░░░] 0%

├─ Integration Pipeline         [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
├─ Integration Tests            [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
└─ Demo Notebook                [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
```

---

## 🔄 آخرین به‌روزرسانی (Last Update Log)

**2025-11-14 Multi-Environment Setup + Data Cleanup**
- ✅ Merged UI improvements از claude.ai/code:
  - filterpy اختیاری شد (backward compatible)
  - SETUP_WEB_ENVIRONMENT.md ایجاد شد
  - requirements_core.txt برای محیط وب
  - test_pose_extraction.py
- ✅ پاکسازی کامل (16GB → 711MB):
  - حذف 5 ویدئوی فاینال بزرگ (قابل دانلود مجدد)
  - حذف فایل‌های موقت YouTube و WAV
  - نگه‌داری 188 race segments + metadata
- ✅ Multi-Environment Sync Setup:
  - SYNC_WORKFLOW.md (راهنمای جامع 30+ صفحه)
  - sync.bat / sync.sh (automation scripts)
  - Gitea ↔ GitHub ↔ UI workflow
- ✅ Data committed locally (commit 4110df5 - 706MB):
  - 188 race segment MP4s
  - 188 metadata JSONs
  - 21 raw video metadata
- ⚠️ Push to Gitea pending (timeout - نیاز به manual)
- ✅ GitHub synced (code + docs + scripts)
- 🎯 Ready for UI claude.ai/code to continue Phase 2
- 📝 START_UI_DEVELOPMENT.md ایجاد شد

**2025-11-13 Late Update - Priority 1 COMPLETED**
- ✅ Implemented complete Race Segmentation System (1330+ lines)
- ✅ race_start_detector.py: Audio + Motion + Fusion detection
- ✅ race_finish_detector.py: Visual + Pose-based detection + Winner determination
- ✅ race_segmenter.py: Full integration with CLI interface
- ✅ Committed Priority 1 (238c08b)
- ✅ Updated MASTER_CONTEXT with Priority 1 completion
- ✅ Updated progress tracker (Phase 1: 70%)
- ✅ Configured max_race_duration=15s for longer races
- 🎯 Ready for testing with real videos
- 🎯 Next: Test and validate, then Priority 2

**2025-11-13 Early Update - Documentation & Planning**
- ✅ Downloaded 11 competition videos (4.3 GB total)
- ✅ Collected IFSC Speed Licence Rules PDF
- ✅ Analyzed video challenges (long videos, camera motion, formats)
- ✅ Analyzed IFSC standards (20 holds, 125mm spacing, grid system)
- ✅ Updated MASTER_CONTEXT with comprehensive implementation roadmap
- ✅ Documented all 5 priorities
- ✅ Updated project structure (added docs/, data/race_segments/, etc.)
- ✅ Committed documentation updates (aafa060)

**2025-11-12 Initial Development**
- ✅ Created MASTER_CONTEXT.md
- ✅ Committed YouTube downloader (dd66cc9)
- ✅ Committed dual-lane detector (d2e7942)
- ✅ User installed FFmpeg + MediaPipe
- ✅ Fixed test failures (c47021c)
- ✅ All 17 tests passing (100%)
- ✅ Dual-Lane Detection module COMPLETE

---

## 🎯 مراحل بعدی (Next Steps)

### گام فوری: اتمام پردازش مسابقات (2025-11-14)
1. ✅ Seoul 2024: COMPLETED (31 مسابقه)
2. ✅ Seoul, Villars, Chamonix, Innsbruck: COMPLETED (119 races)
3. 🔄 **در حال اجرا**: Zilina 2025 (~40 minutes)
   - `python src/utils/manual_race_segmenter.py` برای Zilina در حال اجرا
   - 69 مسابقه در حال پردازش
4. ✅ Verify extracted segments:
   ```bash
   ls -l data/race_segments/*/
   # Expected: 188 .mp4 files + 188 metadata JSONs + 5 summary JSONs
   ```

### Phase 2: Pose Estimation & Analysis (بعد از اتمام segmentation)

#### گام 1: Pose Estimation روی Race Segments (اولویت بالا)
**هدف**: استخراج BlazePose keypoints از 188 کلیپ مسابقه

**تسک‌ها**:
1. **Pipeline ساده برای batch processing**:
   - ورودی: directory از race clips (188 فایل .mp4)
   - خروجی: JSON/NPZ files با pose keypoints
   - قابلیت resumable (skip already processed)

2. **استفاده از کد موجود**:
   - `dual_lane_detector.py` آماده است و تست شده
   - نیاز به script ساده برای batch processing:
     ```python
     # scripts/batch_pose_extraction.py
     for clip in race_segments:
         detector = DualLaneDetector(method='fixed')
         result = detector.process_frame(frame)
         save_keypoints(result, output_path)
     ```

3. **خروجی موردنیاز**:
   - Format: JSON یا NPZ (numpy compressed)
   - محتوا: frame_id, left_keypoints, right_keypoints, confidence
   - ذخیره در: `data/processed/poses/`

#### گام 2: IFSC Calibration Integration (اولویت متوسط)
**هدف**: استفاده از 20 گیره استاندارد برای calibration

**تسک‌ها**:
1. **Parse IFSC PDF** → grid coordinates
2. **Hold Detection** با color/template matching
3. **Pixel-to-Meter Conversion** با homography
4. **مزیت**: حل مشکل camera motion با re-calibration per frame

#### گام 3: Performance Metrics (اولویت بالا)
**هدف**: محاسبه متریک‌های performance از pose data

**متریک‌های کلیدی**:
- **Vertical velocity**: سرعت صعود (m/s)
- **Hold-by-hold timing**: زمان رسیدن به هر گیره (اگر calibration باشد)
- **Movement smoothness**: jerk analysis
- **Path efficiency**: انحراف از خط مستقیم

**تسک‌ها**:
1. Load pose keypoints
2. Calculate metrics frame-by-frame
3. Export to CSV for each race
4. Aggregate statistics (mean, std, percentiles)

#### گام 4: Visualization & Comparison (اولویت بالا)
**هدف**: مقایسه بصری دو climber

**نمودارها**:
- Time-series plots (height vs time)
- Velocity profiles
- Side-by-side video + overlay
- Comparative dashboard

### Priority Roadmap بعد از Phase 2:

1. **Short-term** (1-2 weeks):
   - ✅ Race segmentation (DONE - 188 races)
   - 🔄 Pose extraction (batch processing)
   - 📊 Basic metrics (velocity, timing)
   - 📈 Simple visualizations

2. **Medium-term** (1 month):
   - 📐 IFSC calibration (20 holds)
   - 🎯 Hold-by-hold analysis
   - 📊 Advanced metrics (jerk, path efficiency)
   - 📈 Interactive dashboard

3. **Long-term** (2-3 months):
   - 🧠 NARX neural networks (performance prediction)
   - 🤖 Fuzzy logic systems (technique evaluation)
   - 🌐 Real-time processing
   - 📱 Mobile/web deployment

### فایل‌های کلیدی برای Phase 2:
```
scripts/
  batch_pose_extraction.py        # NEW - batch processing for 188 clips

src/analysis/
  performance_metrics.py           # NEW - calculate metrics from poses

src/visualization/
  time_series_plots.py             # NEW - velocity, height plots
  comparative_overlay.py           # NEW - side-by-side comparison

notebooks/
  02_pose_analysis.ipynb           # NEW - exploratory analysis
```

---

## 📞 تماس با توسعه‌دهنده (Contact)

اگر سوالی دارید یا نیاز به کمک هست:
- Issues در repository بسازید
- مستندات `prompt.md` را مطالعه کنید
- این فایل را به‌روز نگه دارید

---

**END OF MASTER CONTEXT**
این سند باید بعد از هر تغییر مهم به‌روزرسانی شود.
