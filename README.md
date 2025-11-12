# Speed Climbing Performance Analysis System 🧗‍♀️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

سیستم هوشمند تحلیل خودکار ویدئوی سنگنوردی سرعتی با استفاده از Computer Vision و Machine Learning.

## 🎯 ویژگی‌های کلیدی

- **Pose Estimation**: استخراج 33 keypoint بدن با MediaPipe BlazePose (بدون مارکر فیزیکی)
- **Biomechanics Analysis**: محاسبه COM trajectory، path entropy، step length، movement frequency
- **NARX Neural Networks**: یادگیری الگوهای بهینه حرکت
- **Fuzzy Logic System**: بازخورد شخصی‌سازی شده و توصیه‌های بهبود
- **Real-time Processing**: پردازش 30+ fps
- **Gender-specific Analysis**: تحلیل اختصاصی برای مردان و زنان

## 📋 فهرست مطالب

- [نصب](#نصب)
- [شروع سریع](#شروع-سریع)
- [معماری سیستم](#معماری-سیستم)
- [استفاده](#استفاده)
- [Google Colab](#google-colab)
- [مستندات](#مستندات)
- [مشارکت](#مشارکت)

## 🚀 نصب

### پیش‌نیازها

- Python 3.8 یا بالاتر
- GPU (اختیاری، برای سرعت بیشتر)
- حداقل 8GB RAM

### نصب Dependencies

```bash
# Clone repository
git clone https://github.com/yourusername/speed-climbing-analysis.git
cd speed-climbing-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### بررسی نصب

```bash
# Test OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Test MediaPipe
python -c "import mediapipe as mp; print('MediaPipe: OK')"

# Test GPU (PyTorch)
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

## ⚡ شروع سریع

### 1. استخراج Keypoints از ویدئو

```python
from src.phase1_pose_estimation.blazepose_extractor import extract_keypoints_from_video

# Process video
results = extract_keypoints_from_video(
    video_path="athlete_001.mp4",
    output_path="keypoints.json",
    visualize=True,
    output_video_path="annotated_output.mp4"
)

print(f"Processed {len(results)} frames")
```

### 2. محاسبه Path Entropy

```python
from src.phase2_features.path_entropy import calculate_path_entropy
import numpy as np

# Load COM trajectory (from previous step)
com_trajectory = np.array([[x1, y1], [x2, y2], ...])  # in meters

# Calculate entropy
result = calculate_path_entropy(com_trajectory)

print(f"Entropy: {result['entropy']:.3f}")
print(f"Efficiency: {result['path_efficiency']:.3f}")
print(f"Recommendation: {'Excellent' if result['entropy'] < 0.12 else 'Needs improvement'}")
```

### 3. استفاده از Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/01_phase1_pose_estimation.ipynb
```

## 🏗️ معماری سیستم

```
┌─────────────────────────────────────────────────────────┐
│                    VIDEO INPUT (60-240 fps)              │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   Phase 1: Pose       │  MediaPipe BlazePose
         │   Estimation          │  33 keypoints + COM
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Phase 2: Feature    │  Path Entropy
         │   Extraction          │  Gait Analysis
         │                       │  Kinematics
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Phase 3: NARX       │  Time-series
         │   Neural Network      │  Prediction
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Phase 4: Fuzzy      │  Performance
         │   Logic System        │  Evaluation
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Phase 5:            │  Dashboard
         │   Visualization       │  Reports
         └───────────────────────┘
```

## 📂 ساختار پروژه

```
speed_climbing_analysis/
├── data/
│   ├── raw_videos/                 # ویدئوهای خام
│   ├── processed/                  # خروجی‌ها
│   └── annotations/                # برچسب‌های دستی
├── src/
│   ├── phase1_pose_estimation/     # فاز 1
│   │   ├── video_processor.py      # OpenCV wrapper
│   │   ├── blazepose_extractor.py  # MediaPipe wrapper
│   │   └── calibration.py          # Camera calibration
│   ├── phase2_features/            # فاز 2
│   │   ├── path_entropy.py         # محاسبه entropy
│   │   ├── gait_analysis.py        # تحلیل گام
│   │   └── com_tracker.py          # COM tracking
│   ├── models/                     # فاز 3
│   │   ├── narx_network.py         # PyTorch NARX
│   │   └── train.py                # Training loop
│   ├── fuzzy_logic/                # فاز 4
│   │   ├── rules.py                # Fuzzy rules
│   │   └── feedback_generator.py  # Feedback system
│   └── visualization/              # فاز 5
│       ├── overlay.py              # Video overlay
│       └── dashboard.py            # Dashboard
├── configs/
│   ├── keypoints.json              # Keypoint definitions
│   └── camera_calibration.json     # Calibration data
├── notebooks/
│   └── 01_phase1_pose_estimation.ipynb
├── tests/
│   └── ...
├── requirements.txt
├── README.md
└── prompt.md                       # معماری کامل
```

## 🎓 استفاده

### Command Line Interface

```bash
# Extract keypoints
python -m src.phase1_pose_estimation.blazepose_extractor video.mp4

# Calculate path entropy (after keypoint extraction)
python -m src.phase2_features.path_entropy keypoints.json
```

### Python API

```python
# Video Processing
from src.phase1_pose_estimation import VideoProcessor, BlazePoseExtractor

with VideoProcessor("video.mp4", target_fps=30) as video:
    with BlazePoseExtractor(model_complexity=1) as extractor:
        for frame_data in video.extract_frames():
            result = extractor.process_frame(
                frame_data['frame'],
                frame_data['frame_id'],
                frame_data['timestamp']
            )
            # Process result...
```

## 🌐 Google Colab

برای استفاده بدون نصب local:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/speed-climbing-analysis/blob/main/notebooks/01_phase1_pose_estimation.ipynb)

### گام‌های Colab:

1. باز کردن notebook
2. آپلود ویدئو به Google Drive
3. اجرای سلول‌ها به ترتیب
4. دانلود نتایج

## 📊 مثال نتایج

### Input: ویدئوی سنگنوردی سرعتی (15m، 6.5 ثانیه)

**خروجی:**

```json
{
  "total_time": 6.53,
  "path_entropy": 0.14,
  "avg_step_length": 0.89,
  "vertical_efficiency": 0.87,
  "technique_rating": "good",
  "recommendations": [
    "کاهش حرکات جانبی در بخش میانی",
    "افزایش استفاده از dynamic movements"
  ]
}
```

### Visualization

- ✅ Skeleton overlay روی ویدئو
- ✅ COM trajectory plot
- ✅ Velocity profile
- ✅ Joint angle time-series
- ✅ Entropy heatmap

## 📖 مستندات

### کامل:

- [معماری سیستم](prompt.md) - راهنمای کامل فنی
- [API Reference](docs/api.md) - مستندات API
- [Tutorial](docs/tutorial.md) - آموزش گام به گام

### مفاهیم کلیدی:

- **Path Entropy (H)**: معیار انحراف از مسیر مستقیم
  - Optimal: H < 0.12
  - Acceptable: 0.12-0.18
  - Poor: > 0.18

- **COM Trajectory**: مسیر مرکز جرم (Center of Mass)

- **Step Length**: طول گام
  - Women optimal: 0.75-0.95m
  - Men optimal: 0.85-1.05m

- **Movement Frequency**: فرکانس حرکت دست/پا (Hz)

## 🔬 تحقیقات و منابع

- IFSC Speed Climbing Standards (2024)
- "Gender-specific biomechanics in speed climbing" (2023)
- MediaPipe BlazePose: [Paper](https://arxiv.org/abs/2006.10204)
- NARX Networks: [Tutorial](https://www.mathworks.com/help/deeplearning/ug/design-time-series-narx-feedback-neural-networks.html)

## 🤝 مشارکت

برای مشارکت:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open Pull Request

## 📄 License

این پروژه تحت لایسنس MIT منتشر شده است. جزئیات در [LICENSE](LICENSE).

## 🙏 تشکر

- Google MediaPipe Team
- IFSC Research Committee
- OpenCV Contributors

## 📧 تماس

- **نویسندگان**: Speed Climbing Research Team
- **ایمیل**: research@speedclimbing.ai
- **وبسایت**: https://speedclimbing.ai

---

**Made with ❤️ for the climbing community**
