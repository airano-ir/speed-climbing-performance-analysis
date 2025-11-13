# راهنمای تست و استفاده - سیستم تحلیل سنگنوردی سرعتی

**تاریخ**: 2025-11-13
**نسخه**: 1.0
**زبان**: فارسی + English

---

## 📋 فهرست مطالب

1. [تست Race Start Detector](#1-تست-race-start-detector)
2. [تست Race Finish Detector](#2-تست-race-finish-detector)
3. [تست Race Segmenter (استخراج کامل مسابقات)](#3-تست-race-segmenter)
4. [نکات مهم و توصیه‌ها](#4-نکات-مهم-و-توصیهها)
5. [عیب‌یابی و حل مشکلات](#5-عیبیابی-و-حل-مشکلات)

---

## 1. تست Race Start Detector

### 1.1 تست با Motion Detection (سریع)

برای ویدئوهایی که صدا ندارند یا می‌خواهید سریع تست کنید:

```bash
cd "g:\My Drive\Projects\Speed Climbing Performance Analysis"

python src/phase1_pose_estimation/race_start_detector.py \
  "data/raw_videos/VIDEO_NAME.mp4" \
  --method motion \
  --output results/start_detection.json
```

**مثال عملی**:
```bash
python src/phase1_pose_estimation/race_start_detector.py \
  "data/raw_videos/Meet Ola Miroslaw, the fastest female speed climber in the world.mp4" \
  --method motion \
  --output test_start.json
```

**خروجی انتظاری**:
```
Race start detected!
  Frame: 16
  Time: 0.53s
  Confidence: 1.00
  Method: motion
```

---

### 1.2 تست با Audio Detection

برای ویدئوهایی که فایل صوتی WAV دارند:

```bash
python src/phase1_pose_estimation/race_start_detector.py \
  "data/raw_videos/Speed_finals_Seoul_2024.mp4" \
  --method audio \
  --output results/start_audio.json
```

**نکته**: فایل WAV باید با همان نام ویدئو وجود داشته باشد:
- Video: `Speed_finals_Seoul_2024.mp4`
- Audio: `Speed_finals_Seoul_2024.wav` ✅

---

### 1.3 تست با Fusion (Audio + Motion)

**بهترین دقت** - توصیه می‌شود:

```bash
python src/phase1_pose_estimation/race_start_detector.py \
  "data/raw_videos/Speed_finals_Seoul_2024.mp4" \
  --method fusion \
  --output results/start_fusion.json
```

**⚠️ هشدار**: برای ویدئوهای طولانی (2+ ساعت)، audio analysis ممکن است **5-10 دقیقه** طول بکشد.

---

## 2. تست Race Finish Detector

### 2.1 تست Visual Detection (سریع)

```bash
python src/phase1_pose_estimation/race_finish_detector.py \
  "data/raw_videos/VIDEO_NAME.mp4" \
  --method visual \
  --lane left \
  --start-frame 0 \
  --end-frame 300 \
  --output results/finish_visual.json
```

**پارامترها**:
- `--lane`: left | right | unknown
- `--start-frame`: از کجا شروع کنیم (0 = ابتدای ویدئو)
- `--end-frame`: تا کجا جستجو کنیم (None = تا انتها)

---

### 2.2 تست Pose-based Detection

نیاز به داده‌های pose estimation دارد (فعلاً غیرفعال).

---

## 3. تست Race Segmenter

### 3.1 تست ساده (یک مسابقه)

برای تست سریع و اطمینان از کارکرد:

```bash
cd "g:\My Drive\Projects\Speed Climbing Performance Analysis"

python src/utils/race_segmenter.py \
  "data/raw_videos/10_Fastest_Speed_climbing_times_at_Paris2024.mp4" \
  --output-dir "data/race_segments" \
  --max-races 1 \
  --start-method motion \
  --finish-method visual \
  --min-duration 2.0 \
  --max-duration 10.0
```

**خروجی**:
- ویدئوی کلیپ: `data/race_segments/10_Fastest_*_race001.mp4`
- Metadata: `data/race_segments/10_Fastest_*_race001_metadata.json`
- Summary: `data/race_segments/10_Fastest_*_summary.json`

---

### 3.2 تست جامع (چندین مسابقه)

⚠️ **این دستور زمان‌بر است (30-60 دقیقه برای ویدئوی 2 ساعته)**

```bash
python src/utils/race_segmenter.py \
  "data/raw_videos/Speed_finals_Seoul_2024.mp4" \
  --output-dir "data/race_segments/seoul_2024" \
  --max-races 10 \
  --start-method motion \
  --finish-method visual \
  --min-duration 3.0 \
  --max-duration 15.0 \
  --buffer-before 1.0 \
  --buffer-after 1.5
```

**پارامترها**:
- `--max-races`: حداکثر تعداد مسابقات برای extract (None = همه)
- `--min-duration`: حداقل مدت مسابقه (ثانیه) - برای رد کردن false positives
- `--max-duration`: حداکثر مدت مسابقه (ثانیه)
- `--buffer-before`: بافر قبل از شروع (ثانیه)
- `--buffer-after`: بافر بعد از پایان (ثانیه)

---

### 3.3 تست بدون ذخیره ویدئو (فقط metadata)

برای بررسی سریع detection بدون ذخیره کلیپ‌ها:

```bash
python src/utils/race_segmenter.py \
  "data/raw_videos/VIDEO_NAME.mp4" \
  --output-dir "data/race_segments" \
  --max-races 5 \
  --metadata-only
```

---

## 4. نکات مهم و توصیه‌ها

### 4.1 انتخاب روش Detection

| روش | سرعت | دقت | استفاده |
|-----|------|-----|---------|
| **motion** | ⚡ سریع | 🟡 متوسط | تست سریع، ویدئوهای بدون صدا |
| **audio** | 🐌 کند | 🟢 بالا | ویدئوهای با صدای بوق واضح |
| **fusion** | 🐌 کند | 🟢🟢 خیلی بالا | تولید نهایی، دقت بالا |

### 4.2 تنظیم Duration Thresholds

**مسابقات سرعت معمولی**:
- حداقل: `3.0` ثانیه (رکوردها معمولاً 4.5-6 ثانیه)
- حداکثر: `15.0` ثانیه (بیشتر از این احتمالاً اشتباه است)

**ویدئوهای کامپایل (مثل "10 Fastest")**:
- حداقل: `2.0` ثانیه (ممکن است کلیپ‌های کوتاه‌تر داشته باشند)
- حداکثر: `10.0` ثانیه

### 4.3 مدیریت فایل‌های بزرگ

**ویدئوهای فاینال** معمولاً بسیار بزرگ هستند:
- Seoul 2024: **852 MB** (126 دقیقه)
- Zilina 2025: **1.1 GB** (180 دقیقه)

**توصیه**:
1. ابتدا با `--metadata-only` تست کنید
2. اگر نتایج خوب بود، بدون `--metadata-only` اجرا کنید
3. کلیپ‌های extract شده **gitignored** هستند (نگران حجم نباشید)

### 4.4 بررسی نتایج

بعد از اجرای segmenter، حتماً این فایل‌ها را بررسی کنید:

```bash
# Summary file
cat "data/race_segments/VIDEO_NAME_summary.json"

# Metadata for each race
cat "data/race_segments/VIDEO_NAME_race001_metadata.json"
```

**چک‌لیست بررسی**:
- ✅ مدت مسابقه منطقی است؟ (3-15 ثانیه)
- ✅ Confidence scores معقول هستند؟ (> 0.5)
- ✅ تعداد مسابقات extract شده درست است؟

---

## 5. عیب‌یابی و حل مشکلات

### مشکل 1: "No race start detected"

**علت**: ویدئو شامل intro یا محتوای غیرمسابقه است

**راه‌حل**:
1. از `--start-method motion` به جای `audio` استفاده کنید
2. ویدئو را از نقطه‌ای که مسابقه شروع می‌شود، trim کنید
3. با کلیپ‌های کوتاه‌تر تست کنید

---

### مشکل 2: "Duration < minimum, Skipping"

**علت**: Detection اشتباه یا کلیپ خیلی کوتاه

**راه‌حل**:
```bash
# کاهش حداقل مدت
--min-duration 2.0  # به جای 3.0
```

---

### مشکل 3: Extract کرد اما مسابقه واقعی نیست

**علت**: False positive - حرکت یا صدای دیگر detect شده

**راه‌حل**:
```bash
# افزایش حداقل مدت
--min-duration 4.0  # به جای 3.0

# استفاده از fusion method
--start-method fusion
```

---

### مشکل 4: Audio analysis خیلی کند است

**علت**: FFT analysis روی فایل 2 ساعته زمان می‌برد

**راه‌حل**:
```bash
# استفاده از motion-only
--start-method motion

# یا trim کردن ویدئو قبل از processing
```

---

### مشکل 5: فقط اولین مسابقه extract می‌شود

**علت**: فعلاً سیستم با sliding window approach کار نمی‌کند

**وضعیت**: 🔄 در حال بهبود - به‌زودی رفع می‌شود

**Workaround فعلی**: برای هر مسابقه، ویدئو را manual trim کنید

---

## 6. مثال‌های عملی Step-by-Step

### مثال 1: تست کامل یک کلیپ کوتاه

```bash
cd "g:\My Drive\Projects\Speed Climbing Performance Analysis"

# Step 1: Test start detector
python src/phase1_pose_estimation/race_start_detector.py \
  "data/raw_videos/Meet Ola Miroslaw, the fastest female speed climber in the world.mp4" \
  --method motion \
  --output test1_start.json

# Step 2: Extract race
python src/utils/race_segmenter.py \
  "data/raw_videos/Meet Ola Miroslaw, the fastest female speed climber in the world.mp4" \
  --output-dir "data/race_segments/test1" \
  --max-races 1 \
  --start-method motion \
  --min-duration 2.0

# Step 3: Review results
cat "data/race_segments/test1/*_summary.json"
```

---

### مثال 2: Extract کردن 5 مسابقه اول از فاینال

```bash
# این دستور ممکن است 30-45 دقیقه طول بکشد
python src/utils/race_segmenter.py \
  "data/raw_videos/Speed_finals_Seoul_2024.mp4" \
  --output-dir "data/race_segments/seoul_test" \
  --max-races 5 \
  --start-method motion \
  --finish-method visual \
  --min-duration 3.0 \
  --max-duration 15.0 \
  --buffer-before 1.0 \
  --buffer-after 1.5

# بعد از اتمام، بررسی نتایج:
ls -lh "data/race_segments/seoul_test/"
cat "data/race_segments/seoul_test/Speed_finals_Seoul_2024_summary.json"
```

---

## 7. ساختار خروجی‌ها

### 7.1 فایل Summary

```json
{
  "source_video": "path/to/video.mp4",
  "total_races": 3,
  "races": [
    {
      "race_id": "video_race001",
      "duration": 5.2,
      "start_timestamp": 10.5,
      "finish_timestamp": 15.7,
      "start_confidence": 1.0,
      "finish_confidence": 0.85,
      ...
    }
  ],
  "processing_date": "2025-11-13T..."
}
```

### 7.2 فایل Metadata (هر مسابقه)

```json
{
  "race_id": "video_race001",
  "source_video": "path/to/video.mp4",
  "start_frame": 315,
  "finish_frame": 471,
  "start_timestamp": 10.5,
  "finish_timestamp": 15.7,
  "duration": 5.2,
  "start_confidence": 1.0,
  "finish_confidence": 0.85,
  "lane": "dual",
  "output_path": "data/race_segments/video_race001.mp4",
  "metadata": {
    "start_method": "motion",
    "finish_method": "visual",
    "buffer_before": 1.0,
    "buffer_after": 1.5,
    "extraction_date": "2025-11-13T..."
  }
}
```

---

## 8. چک‌لیست قبل از شروع

قبل از شروع تست، مطمئن شوید:

- [ ] Python 3.11 نصب است
- [ ] Dependencies نصب شده‌اند: `pip install -r requirements_phase1_extended.txt`
- [ ] ویدئوها در `data/raw_videos/` هستند
- [ ] پوشه `data/race_segments/` ایجاد شده (خودکار ایجاد می‌شود)
- [ ] حداقل 1 GB فضای خالی برای کلیپ‌های extract شده
- [ ] برای ویدئوهای طولانی، زمان کافی دارید (1-2 ساعت)

---

## 9. نتایج را با من به اشتراک بگذارید

بعد از اجرای تست‌ها، این فایل‌ها را برای بررسی در اختیارم قرار دهید:

```bash
# Summary files
data/race_segments/*_summary.json

# Metadata files (sample)
data/race_segments/*_race001_metadata.json

# Screenshots/screen recordings (optional)
```

یا خروجی console را کپی کنید:
```bash
python src/utils/race_segmenter.py ... 2>&1 | tee output.log
# بعد فایل output.log را برای من بفرستید
```

---

## 10. سوالات متداول (FAQ)

**Q: چند وقت طول می‌کشد؟**
A: بستگی به طول ویدئو و روش detection دارد:
- کلیپ 1 دقیقه: ~10 ثانیه
- ویدئو 10 دقیقه: ~1-2 دقیقه
- ویدئو 2 ساعت: ~30-60 دقیقه

**Q: می‌توانم process را متوقف کنم؟**
A: بله، Ctrl+C بزنید. فایل‌های تا آن لحظه ذخیره می‌شوند.

**Q: فضای کافی ندارم**
A: از `--metadata-only` استفاده کنید تا فقط JSON files ذخیره شوند.

**Q: چطور بفهمم detection درست است؟**
A: Confidence scores را بررسی کنید. بالای 0.7 معمولاً خوب است.

---

**تماس و پشتیبانی**:
در صورت مشکل یا سوال، خروجی console و فایل‌های JSON را برای من ارسال کنید.

**موفق باشید! 🎯**
