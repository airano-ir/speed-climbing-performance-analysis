# 🚀 START: Development در UI claude.ai/code

**تاریخ**: 2025-11-14
**Repository**: https://github.com/languageofearthcom-oss/Speed-Climbing-Performance-Analysis
**محیط**: Linux + Python 3.11.14

---

## ✅ مرحله فعلی پروژه

**Phase 1 کامل شد**:
- ✅ 188 race segment استخراج شده (metadata موجود)
- ✅ Dual-lane detection آماده (17/17 tests pass)
- ✅ filterpy optional (کد در هر محیطی کار می‌کند)

**Phase 2 شروع می‌شود**: Pose Estimation & Analysis

---

## 🎯 Quick Start (3 دقیقه)

### 1. Clone Repository

```bash
git clone https://github.com/languageofearthcom-oss/Speed-Climbing-Performance-Analysis.git
cd Speed-Climbing-Performance-Analysis
```

### 2. Setup Dependencies

```bash
pip install --break-system-packages -r requirements_core.txt
```

**نکته**: filterpy اختیاری است - کد بدون آن هم کار می‌کند.

### 3. تست

```bash
# تست imports
python -c "import cv2; import mediapipe; import numpy; print('✓ OK')"

# تست unit tests
pytest tests/test_dual_lane_detector.py -v
# باید: 17/17 tests PASSED
```

---

## 📋 Phase 2: Batch Pose Extraction

**هدف**: استخراج BlazePose keypoints از race segments

### Task 1: ساخت Batch Script

**فایل**: `scripts/batch_pose_extraction.py`

**الگو**:
```python
from pathlib import Path
import sys
sys.path.insert(0, 'src/phase1_pose_estimation')
from dual_lane_detector import DualLaneDetector
import cv2
import json
from tqdm import tqdm

def extract_poses_from_clip(video_path, output_json):
    """استخراج poses از یک race clip."""
    detector = DualLaneDetector(method='fixed')
    cap = cv2.VideoCapture(str(video_path))

    results = []
    with detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.process_frame(frame)
            results.append({
                'frame_id': result.frame_id,
                'left_pose': result.left_pose.to_dict() if result.left_pose else None,
                'right_pose': result.right_pose.to_dict() if result.right_pose else None,
            })

    cap.release()

    # ذخیره JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    return len(results)

def main():
    # پردازش 5 sample clip (تست)
    raw_videos = Path('data/raw_videos')
    output_dir = Path('data/processed/poses')
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = list(raw_videos.glob('*.mp4'))
    print(f"Found {len(clips)} sample clips")

    for clip in tqdm(clips):
        output_json = output_dir / f"{clip.stem}_poses.json"
        if output_json.exists():
            print(f"  Skipping {clip.name} (already processed)")
            continue

        print(f"  Processing: {clip.name}")
        num_frames = extract_poses_from_clip(clip, output_json)
        print(f"    ✓ {num_frames} frames processed")

if __name__ == '__main__':
    main()
```

**اجرا**:
```bash
python scripts/batch_pose_extraction.py
```

---

### Task 2: Performance Metrics

**فایل**: `src/analysis/performance_metrics.py`

**متریک‌های کلیدی**:
- Vertical velocity (m/s)
- Movement smoothness (jerk)
- Path efficiency

---

### Task 3: Visualization

**فایل**: `src/visualization/time_series_plots.py`

**نمودارها**:
- Height vs Time
- Velocity profile
- Comparison charts

---

## 🔄 Workflow: توسعه و Push

### 1. ایجاد Branch

```bash
git checkout -b feature/phase2-pose-extraction
```

### 2. توسعه

```python
# کد بنویس، تست کن
pytest tests/ -v
```

### 3. Commit

```bash
git add -A
git commit -m "feat(pose): implement batch pose extraction

- Add scripts/batch_pose_extraction.py
- Process 5 sample clips successfully
- Output: JSON with pose keypoints
- Tested with sample data

Ready for full dataset processing."
```

### 4. Push

```bash
git push origin feature/phase2-pose-extraction
```

---

## 💡 نکات مهم

### محدودیت‌های UI claude.ai/code:

❌ **Race segments (188 clips) موجود نیستند** (705 MB در Gitea)
✅ **5 sample clips موجود است** (6 MB)

**راه‌حل**:
- ابتدا با 5 sample clips تست کنید
- Pipeline را بنویسید و آماده کنید
- بعداً در محیط اصلی (Gitea) روی 188 race اجرا کنید

---

## 📦 ساختار خروجی

```
data/processed/poses/
├── Aleksandra_Miroslaw_poses.json
├── Current_mens_Olympic_record_poses.json
├── Meet_Ola_Miroslaw_poses.json
├── Sarina_Ghafari_poses.json
└── SPEED_CLIMBING_IS_ELITE_poses.json
```

هر JSON:
```json
[
  {
    "frame_id": 0,
    "left_pose": {
      "keypoints": [...],
      "confidence": 0.95
    },
    "right_pose": {
      "keypoints": [...],
      "confidence": 0.92
    }
  },
  ...
]
```

---

## 🎯 Checklist Phase 2

- [ ] ساخت `scripts/batch_pose_extraction.py`
- [ ] تست با 5 sample clips
- [ ] ساخت `src/analysis/performance_metrics.py`
- [ ] ساخت `src/visualization/time_series_plots.py`
- [ ] Commit و Push
- [ ] Merge در محیط اصلی (Gitea)

---

**🚀 آماده برای شروع! Good luck!**

برای سوالات: ببینید [MASTER_CONTEXT.md](MASTER_CONTEXT.md)
