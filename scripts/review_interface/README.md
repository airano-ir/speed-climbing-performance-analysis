# Manual Review Interface
# رابط بررسی دستی

**Version**: 1.0.0
**Date**: 2025-11-16
**Language**: Bilingual (English / فارسی)

---

## 📖 Quick Start / شروع سریع

### English

**Install dependencies:**
```bash
pip install streamlit opencv-python pyyaml numpy
```

**Run the interface:**
```bash
cd scripts/review_interface
streamlit run app.py
```

**Test components first:**
```bash
python scripts/review_interface/test_components.py
```

### فارسی

**نصب وابستگی‌ها:**
```bash
pip install streamlit opencv-python pyyaml numpy
```

**اجرای رابط:**
```bash
cd scripts/review_interface
streamlit run app.py
```

**ابتدا تست کامپوننت‌ها:**
```bash
python scripts/review_interface/test_components.py
```

---

## 🏗️ Architecture / معماری

### Component Structure / ساختار کامپوننت‌ها

```
scripts/review_interface/
├── __init__.py                 # Package initialization
├── app.py                      # Main Streamlit application (اپلیکیشن اصلی)
├── config.py                   # Configuration manager (مدیر پیکربندی)
├── progress.py                 # Progress tracker (ردیاب پیشرفت)
├── metadata_manager.py         # Metadata CRUD operations (عملیات متادیتا)
├── video_player.py             # Video playback component (پخش‌کننده ویدئو)
├── validators.py               # Validation engine (موتور اعتبارسنجی)
├── test_components.py          # Component tests (تست کامپوننت‌ها)
└── README.md                   # This file (این فایل)
```

### Component Descriptions / توضیحات کامپوننت‌ها

#### 1. `config.py` - Configuration Manager
**English**: Loads and manages `manual_review_config.yaml`. Provides config-driven architecture for adding new competitions without code changes.

**فارسی**: بارگذاری و مدیریت فایل `manual_review_config.yaml`. معماری مبتنی بر پیکربندی برای افزودن مسابقات جدید بدون تغییر کد.

**Key Features:**
- ✅ Competition management
- ✅ Validation rules
- ✅ General settings
- ✅ Feature flags

#### 2. `progress.py` - Progress Tracker
**English**: Manages CSV file tracking review progress for 74 suspicious races.

**فارسی**: مدیریت فایل CSV برای ردیابی پیشرفت بررسی 74 مسابقه مشکوک.

**Key Features:**
- ✅ Load/update race status
- ✅ Filter by status (pending/completed/skipped)
- ✅ Calculate statistics
- ✅ Thread-safe operations

#### 3. `metadata_manager.py` - Metadata Manager
**English**: CRUD operations for race metadata JSON files with automatic backups.

**فارسی**: عملیات CRUD برای فایل‌های JSON متادیتا با پشتیبان‌گیری خودکار.

**Key Features:**
- ✅ Load/save metadata
- ✅ Update race boundaries
- ✅ Automatic backup creation (`.bak`)
- ✅ Correction audit trail

#### 4. `video_player.py` - Video Player
**English**: Streamlit component for frame-by-frame video navigation.

**فارسی**: کامپوننت Streamlit برای ناوبری فریم به فریم در ویدئو.

**Key Features:**
- ✅ Frame slider
- ✅ Navigation buttons (±1, ±5, ±30 frames)
- ✅ Time/frame display
- ✅ Bilingual UI support

#### 5. `validators.py` - Validators
**English**: Validation engine for race corrections with multiple checks.

**فارسی**: موتور اعتبارسنجی برای اصلاحات مسابقه با چندین بررسی.

**Key Features:**
- ✅ Duration validation (4.5s - 15s)
- ✅ Frame order validation
- ✅ Frame bounds checking
- ✅ Critical error detection

#### 6. `app.py` - Main Application
**English**: Bilingual Streamlit interface for manual race review.

**فارسی**: رابط دوزبانه Streamlit برای بررسی دستی مسابقات.

**Key Features:**
- ✅ Language switching (EN/FA)
- ✅ Progress statistics
- ✅ Race filtering
- ✅ Video playback
- ✅ Frame marking
- ✅ Validation feedback
- ✅ Metadata editing

---

## 🎯 Use Cases / موارد استفاده

### 1. Review Suspicious Races / بررسی مسابقات مشکوک
**Purpose**: Correct race detection errors for 74 suspicious races (39.4% of dataset).

**مقصد**: اصلاح خطاهای تشخیص مسابقه برای 74 مسابقه مشکوک (39.4% از داده‌ها).

### 2. Add New Competition / افزودن مسابقه جدید
**Steps**:
1. Edit `configs/manual_review_config.yaml`
2. Add new competition entry
3. Place race videos in `data/race_segments/[competition]/`
4. Run interface

**مراحل**:
1. ویرایش فایل `configs/manual_review_config.yaml`
2. افزودن مسابقه جدید
3. قرار دادن ویدئوها در `data/race_segments/[competition]/`
4. اجرای رابط

### 3. Review Previous Videos / بررسی ویدئوهای قبلی
**Purpose**: Re-review and update corrections for any race.

**مقصد**: بررسی مجدد و به‌روزرسانی اصلاحات برای هر مسابقه.

**Features**:
- ✅ Load previously corrected races
- ✅ View correction history
- ✅ Update corrections
- ✅ Backup previous versions

---

## 📊 Data Flow / جریان داده

```
1. User selects race
   (انتخاب مسابقه توسط کاربر)
   ↓
2. Load metadata & video
   (بارگذاری متادیتا و ویدئو)
   ↓
3. Navigate video to find correct frames
   (ناوبری در ویدئو برای یافتن فریم‌های صحیح)
   ↓
4. Mark start/finish frames
   (علامت‌گذاری فریم‌های شروع/پایان)
   ↓
5. Validate corrections
   (اعتبارسنجی اصلاحات)
   ↓
6. Save with backup
   (ذخیره با پشتیبان‌گیری)
   ↓
7. Update progress tracker
   (به‌روزرسانی ردیاب پیشرفت)
```

---

## 🔧 Configuration / پیکربندی

### Config File / فایل پیکربندی

**Location**: `configs/manual_review_config.yaml`

**Sections**:
- `general`: Default settings (FPS, buffers, etc.)
- `validation`: Validation rules (duration, confidence)
- `competitions`: Competition definitions
- `features`: Feature flags for future enhancements

### Example Competition Entry / نمونه مسابقه

```yaml
competitions:
  paris_2026:
    name: "Speed Finals Paris 2026"
    date: "2026-08-01"
    video_format: "MP4"
    fps: 60.0
    race_segments_path: "data/race_segments/paris_2026"
    total_races: 32
    notes: "Paris 2026 Olympics"
```

---

## 🧪 Testing / تست

### Run All Tests / اجرای همه تست‌ها

```bash
python scripts/review_interface/test_components.py
```

### Test Individual Components / تست کامپوننت‌های جداگانه

```python
from scripts.review_interface.config import ConfigManager
from scripts.review_interface.progress import ProgressTracker
from scripts.review_interface.metadata_manager import MetadataManager
from scripts.review_interface.validators import RaceValidator

# Test config
config = ConfigManager()
print(config.get_competitions())

# Test progress
tracker = ProgressTracker()
print(tracker.get_statistics())

# Test metadata
mgr = MetadataManager()
metadata = mgr.load_metadata('chamonix_2024', 'Speed_finals_Chamonix_2024_race001')
print(metadata)

# Test validator
validator = RaceValidator()
results = validator.validate_all(100, 300, 30.0)
print(results)
```

---

## 📚 Documentation / مستندات

### English Documentation
- **Build Guide**: `docs/PROMPT_FOR_UI_BUILD_REVIEW_INTERFACE.md`
- **Project Context**: `MASTER_CONTEXT.md`
- **Segmentation Guide**: `docs/MANUAL_SEGMENTATION_GUIDE.md`

### Persian Documentation / مستندات فارسی
- **User Guide**: `docs/MANUAL_REVIEW_INTERFACE_GUIDE_FA.md`
- **README**: This file (این فایل)

---

## 🐛 Troubleshooting / عیب‌یابی

### Interface won't start / رابط شروع نمی‌شود

```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit
```

### Video not loading / ویدئو بارگذاری نمی‌شود

```bash
# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Reinstall if needed
pip install --upgrade opencv-python
```

### Config not found / پیکربندی یافت نشد

```bash
# Verify config file exists
ls configs/manual_review_config.yaml

# Check path in code
# Should be: "configs/manual_review_config.yaml"
```

### Progress tracker errors / خطاهای ردیاب پیشرفت

```bash
# Verify CSV file exists
ls data/manual_review/progress_tracker.csv

# Check CSV format (should have headers)
head -n 5 data/manual_review/progress_tracker.csv
```

---

## 🚀 Future Enhancements / بهبودهای آینده

### Planned Features / ویژگی‌های برنامه‌ریزی شده

1. **ML-based Suggestions**: Auto-suggest corrections based on similar races
   (پیشنهادات مبتنی بر یادگیری ماشین)

2. **Batch Review Mode**: Review multiple races sequentially
   (حالت بررسی دسته‌ای)

3. **Collaborative Review**: Multi-user support with review assignments
   (بررسی مشارکتی)

4. **Audio Analysis**: Start detection using starting beep
   (آنالیز صوتی)

5. **Pose Overlay**: Visualize detected keypoints on video
   (نمایش نقاط کلیدی روی ویدئو)

6. **Export Report**: Generate correction summary report
   (گزارش خلاصه اصلاحات)

### Plugin System / سیستم افزونه

The interface is designed for extensibility:
- Custom validators per competition
- Export pipeline customization
- Feature flags for gradual rollout

رابط برای گسترش‌پذیری طراحی شده است:
- اعتبارسنجی سفارشی برای هر مسابقه
- سفارشی‌سازی خط لوله خروجی
- پرچم‌های ویژگی برای عرضه تدریجی

---

## 📝 License / مجوز

Part of the Speed Climbing Performance Analysis project.

بخشی از پروژه تحلیل عملکرد سنگنوردی سرعتی.

---

## 👥 Contributing / مشارکت

### Adding New Components / افزودن کامپوننت‌های جدید

1. Create module in `scripts/review_interface/`
2. Add to `__init__.py`
3. Write tests in `test_components.py`
4. Update this README
5. Add Persian documentation

---

## 📞 Support / پشتیبانی

For questions or issues:
- See comprehensive guides in `docs/`
- Check `MASTER_CONTEXT.md` for project overview
- Review test output for diagnostics

برای سوالات یا مشکلات:
- راهنماهای جامع در `docs/` را ببینید
- `MASTER_CONTEXT.md` را برای دید کلی پروژه بررسی کنید
- خروجی تست را برای تشخیص مشکلات بررسی کنید

---

**Happy reviewing! / بررسی موفق!** 🎉
