# Web Environment Setup Guide
# راهنمای راه‌اندازی محیط وب

**تاریخ**: 2025-11-14
**محیط**: Claude.ai Code (Web Environment)
**Python**: 3.11.14
**OS**: Linux

---

## ✅ وضعیت نصب (Installation Status)

### Core Dependencies (نصب شده)

```bash
✓ opencv-python     4.12.0   # Computer vision
✓ mediapipe         0.10.14  # BlazePose pose estimation
✓ numpy             2.2.6    # Numerical computing
✓ pandas            2.3.3    # Data analysis
✓ scipy             1.16.3   # Scientific computing
✓ matplotlib        3.10.7   # Visualization
✓ seaborn           0.13.2   # Statistical plots
✓ pillow            12.0.0   # Image processing
✓ pyyaml            6.0.1    # YAML config files
✓ tqdm              4.67.1   # Progress bars
✓ pytest            9.0.1    # Testing framework
```

### Optional Dependencies (فعلاً نصب نشده)

```bash
⚠ filterpy          -        # Kalman filtering (اختیاری - کد اصلاح شد)
⚠ librosa           -        # Audio analysis (برای race detection - Phase 1)
⚠ torch             -        # Deep learning (برای Phase 3+)
⚠ plotly            -        # Interactive plots (برای Phase 4+)
```

---

## 🎯 تست‌های انجام شده (Tests Performed)

### 1. Unit Tests (17/17 Passed ✅)

```bash
python3 -m pytest tests/test_dual_lane_detector.py -v
```

**نتیجه**:
- ✅ 17/17 tests PASSED
- ⚠ 4 warnings (MediaPipe cleanup - غیرمهم)
- ✅ همه قابلیت‌های dual-lane detector کار می‌کند

### 2. Import Tests (✅)

```bash
python3 -c "import cv2; import mediapipe; import numpy; print('OK')"
```

**نتیجه**: ✅ همه کتابخانه‌های اصلی load می‌شوند

---

## 🔧 بهبودهای انجام شده (Improvements Made)

### 1. filterpy Optional ✅

**مشکل**: `filterpy` در محیط وب install نمی‌شود (setuptools incompatibility)

**راه‌حل**: کد `dual_lane_detector.py` اصلاح شد:

```python
# Before
from filterpy.kalman import KalmanFilter

# After
try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    print("Warning: filterpy not available. Kalman smoothing disabled.")
```

**مزایا**:
- ✅ کد بدون `filterpy` کار می‌کند
- ✅ اگر `filterpy` نصب شود، از آن استفاده می‌کند
- ✅ Unit tests همچنان pass می‌شوند (17/17)
- ✅ Graceful degradation: از simple smoothing استفاده می‌کند

---

## 📦 فایل‌های موجود در Repo

### کد و Configs
✅ همه فایل‌های Python و configs موجود است:
- `src/` - کد اصلی
- `configs/` - YAML configs (5 فایل)
- `tests/` - Unit tests
- `scripts/` - Batch processing scripts

### Metadata
✅ همه metadata files موجود است:
- `data/race_segments/*/metadata.json` - اطلاعات 188 مسابقه
- `configs/race_timestamps/*.yaml` - Timestamps دستی

### ویدئوها ❌ (فقط 5 کلیپ کوتاه)
- ✅ 5 کلیپ social media (12-24s) در `data/raw_videos/`
- ❌ 5 ویدئوی فاینال بزرگ (2-3 ساعته) - در Google Drive
- ❌ 188 race clips - باید regenerate شوند

**نکته**: کلیپ‌های کوتاه موجود **AV1 codec** دارند که OpenCV مشکل دارد.
برای تست با ویدئوهای واقعی، از فایل‌های بزرگ‌تر (MP4/H.264) استفاده کنید.

---

## 🚀 نصب در محیط جدید (Fresh Installation)

### گام 1: Clone Repository

```bash
git clone <repo-url>
cd Speed-Climbing-Performance-Analysis
```

### گام 2: نصب Core Dependencies

```bash
# روش 1: نصب فقط dependencies اصلی (سریع)
pip3 install opencv-python mediapipe numpy pandas matplotlib pyyaml tqdm pillow pytest scipy seaborn --break-system-packages

# روش 2: از فایل requirements
pip3 install -r requirements_core.txt --break-system-packages
```

### گام 3: تست محیط

```bash
# Import test
python3 -c "import cv2; import mediapipe as mp; import numpy as np; print('✓ All OK')"

# Unit tests
python3 -m pytest tests/test_dual_lane_detector.py -v
```

**انتظار**: 17/17 tests pass

---

## 📝 تفاوت‌های محیط وب vs VS Code

| ویژگی | VS Code (قبل) | Web (حالا) | وضعیت |
|-------|--------------|-----------|--------|
| **OS** | Windows | Linux | ✅ کد portable است |
| **Python** | 3.11.6 | 3.11.14 | ✅ compatible |
| **Dependencies** | همه نصب | Core نصب | ✅ کافی است |
| **filterpy** | نصب شده | نصب نشده | ✅ اصلاح شد (optional) |
| **ویدئوهای بزرگ** | Local (Google Drive) | ❌ | 📥 نیاز به آپلود |
| **188 race clips** | موجود | ❌ (فقط metadata) | 🔄 قابل regenerate |
| **Unit tests** | ✅ 17/17 | ✅ 17/17 | ✅ |

---

## 🎬 کار با ویدئوها

### گزینه 1: استفاده از Metadata (توصیه می‌شود)
برای توسعه کد و testing:
```python
# کد می‌تواند با metadata کار کند بدون نیاز به ویدئوهای بزرگ
import json
with open('data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race001_metadata.json') as f:
    metadata = json.load(f)
```

### گزینه 2: آپلود ویدئوهای بزرگ
اگر به pose extraction واقعی نیاز دارید:
1. آپلود 5 ویدئوی فاینال از Google Drive → `data/raw_videos/`
2. اجرای `batch_segment_competitions.py` برای regenerate کردن 188 clip

### گزینه 3: تست با کلیپ‌های ساده
ساخت test videos:
```python
import cv2
import numpy as np

# Create simple test video
out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
for i in range(300):  # 10 seconds @ 30fps
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw test pattern
    cv2.rectangle(frame, (0, 0), (640, 720), (255, 0, 0), -1)  # Left blue
    cv2.rectangle(frame, (640, 0), (1280, 720), (0, 255, 0), -1)  # Right green
    out.write(frame)
out.release()
```

---

## 📊 مراحل بعدی (Next Steps)

### مرحله فعلی: Batch Pose Extraction Pipeline

**هدف**: طراحی pipeline برای پردازش 188 race clip (یا test data)

**تسک‌ها**:
1. ✅ Setup environment (DONE)
2. 🔄 ایجاد `scripts/batch_pose_extraction.py`
3. 📝 طراحی output format (JSON/NPZ)
4. 🧪 تست با mock data

### مراحل آینده:
1. **Performance Metrics** (Phase 2)
2. **Visualization Tools** (Phase 2)
3. **IFSC Calibration** (Phase 3)
4. **Advanced Analytics** (Phase 4)

---

## 🐛 مشکلات شناخته شده (Known Issues)

### 1. AV1 Codec
**مشکل**: کلیپ‌های کوتاه موجود (5 فایل) AV1 codec دارند
**علامت**: `[av1 @ ...] Missing Sequence Header`
**راه‌حل موقت**: از ویدئوهای H.264/MP4 استفاده کنید

### 2. filterpy Not Available
**مشکل**: filterpy در محیط وب install نمی‌شود
**راه‌حل**: ✅ کد اصلاح شد - حالا optional است
**تأثیر**: Kalman smoothing disable می‌شود (تأثیر جزئی)

### 3. MediaPipe Cleanup Warnings
**مشکل**: warning در `__del__` method
**تأثیر**: فقط warning - عملکرد تحت تأثیر نیست
**راه‌حل**: می‌توان ignore کرد

---

## 💡 نکات مهم

1. **Virtual Environment**: در محیط وب از `--break-system-packages` استفاده می‌شود
2. **GPU**: MediaPipe روی CPU کار می‌کند (NPU/GPU اختیاری)
3. **Memory**: برای 188 race clip، batch size را کم نگه دارید
4. **Storage**: ویدئوها را در `.gitignore` نگه دارید (حجم بالا)

---

## 📞 کمک و Support

اگر مشکلی پیش آمد:
1. چک کنید که همه core dependencies نصب شده: `pip3 list`
2. Unit tests را اجرا کنید: `pytest tests/ -v`
3. لاگ‌ها را بررسی کنید
4. Issues را در repo بررسی کنید

---

**آخرین به‌روزرسانی**: 2025-11-14
**نویسنده**: Speed Climbing Analysis Team
**وضعیت**: ✅ Environment Ready
