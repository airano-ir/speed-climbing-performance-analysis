# راهنمای تست Phase 3 - تحلیل پیشرفته (Advanced Analytics)
# Phase 3 Testing & Validation Guide

**تاریخ**: 2025-11-15
**نسخه**: 1.0
**مخاطب**: کاربر Local PC (Windows)
**پیش‌نیاز**: Phase 3 کامل شده توسط UI claude.ai/code

---

## 📋 فهرست مطالب

1. [خلاصه اجرایی](#خلاصه-اجرایی)
2. [پیش‌نیازها](#پیش-نیازها)
3. [تست اولیه (Quick Test)](#تست-اولیه)
4. [تست کامل (Full Pipeline)](#تست-کامل)
5. [بررسی کیفیت خروجی](#بررسی-کیفیت-خروجی)
6. [عیب‌یابی](#عیب-یابی)
7. [مراحل بعدی](#مراحل-بعدی)

---

## 🎯 خلاصه اجرایی

Phase 3 (Advanced Analytics) توسط UI claude.ai/code با موفقیت پیاده‌سازی شده است:

### ✅ کارهای انجام شده:
- **7 اسکریپت جدید**: batch metrics, aggregation, comparison, visualization, dashboard, ML export
- **2 ماژول تحلیلی**: comparative analysis, race plots
- **2,093 خط کد**: با کیفیت production-ready
- **تست موفق**: 100% روی 3 race نمونه
- **مستندات کامل**: 587 خط documentation

### 🎯 هدف این راهنما:
کمک به شما برای:
1. **تست کردن** کدهای Phase 3
2. **اجرای کامل** pipeline روی 188 race
3. **بررسی کیفیت** خروجی‌ها
4. **شناسایی مشکلات** احتمالی
5. **آماده‌سازی** برای مراحل بعدی

---

## ⚙️ پیش‌نیازها

### 1. بررسی محیط Python
```bash
# بررسی نسخه Python
python --version
# انتظار: Python 3.11.x

# بررسی dependencies
pip list | findstr "numpy pandas matplotlib scikit-learn"
```

**چک‌لیست**:
- [x] Python 3.11+ نصب شده
- [x] همه dependencies از requirements.txt نصب هستند
- [x] Virtual environment فعال است

### 2. بررسی داده‌های موجود
```bash
# بررسی تعداد pose files
dir /s /b data\processed\poses\*.json | find /c ".json"
# انتظار: حداقل 10 (samples) یا 188+ (full dataset)

# بررسی race segments
dir /s /b data\race_segments\*.mp4 | find /c ".mp4"
# انتظار: 188 فایل MP4
```

**چک‌لیست**:
- [x] حداقل 10 sample pose files موجود است
- [x] 188 race segment MP4 موجود است (اختیاری برای تست)
- [x] Metadata JSONs برای هر race موجود است

### 3. بررسی فایل‌های Phase 3
```bash
# بررسی scripts جدید
dir scripts\batch_*.py
dir scripts\compare_races.py
dir scripts\generate_html_dashboard.py
dir scripts\export_ml_data.py

# بررسی modules جدید
dir src\analysis\comparative_analysis.py
dir src\visualization\race_plots.py
```

**چک‌لیست**:
- [x] همه 5 اسکریپت Phase 3 موجود است
- [x] هر دو ماژول جدید موجود است
- [x] Documentation files (PHASE_3_*.md) موجود است

---

## 🧪 تست اولیه (Quick Test)

### مرحله 1: تست با یک Race
زمان: ~10 ثانیه

```bash
# تست محاسبه metrics برای یک race
python scripts/batch_calculate_metrics.py --max-races 1 --competition samples

# بررسی خروجی
dir data\processed\metrics
type data\processed\metrics\batch_metrics_summary.json
```

**انتظار**:
- فولدر `data\processed\metrics\samples\` ایجاد شده
- فایل‌های `*_metrics_left.json` و `*_metrics_right.json` موجود است
- Summary JSON شامل 1 race است

**در صورت موفقیت**: ✅ ادامه دهید
**در صورت خطا**: ⚠️ به بخش [عیب‌یابی](#عیب-یابی) بروید

---

### مرحله 2: تست مقایسه Races
زمان: ~5 ثانیه

```bash
# پیدا کردن یک race برای تست
dir data\race_segments\chamonix_2024\*.mp4 /b
# یک نام race را یادداشت کنید (مثلاً race001)

# اجرای comparison
python scripts/compare_races.py --race race001 --competition chamonix_2024
```

**انتظار**:
- خروجی console شامل مقایسه left vs right
- پیش‌بینی برنده (left یا right)
- آمار velocity و path efficiency

**در صورت موفقیت**: ✅ ادامه دهید
**در صورت خطا**: ⚠️ بررسی کنید که pose files موجود باشد

---

### مرحله 3: تست Visualization
زمان: ~5 ثانیه

```bash
# تولید نمودار برای یک race
python -c "from src.visualization.race_plots import RacePlotter; plotter = RacePlotter(); plotter.plot_velocity_comparison('data/processed/metrics/samples', output_path='test_viz.png')"

# باز کردن نمودار
start test_viz.png
```

**انتظار**:
- فایل PNG ایجاد شده
- نمودار واضح و خوانا است
- هر دو lane (left/right) نمایش داده می‌شود

**در صورت موفقیت**: ✅ تست اولیه موفق! به تست کامل بروید
**در صورت خطا**: ⚠️ بررسی کنید matplotlib نصب باشد

---

## 🚀 تست کامل (Full Pipeline)

### گزینه A: استفاده از اسکریپت خودکار (پیشنهادی)
زمان: ~5 دقیقه

```bash
# اجرای اسکریپت batch
run_full_pipeline.bat

# اسکریپت به صورت خودکار 6 مرحله را اجرا می‌کند:
# 1. Calculate metrics (188 races)
# 2. Aggregate statistics
# 3. Compare all races
# 4. Generate plots
# 5. Create dashboard
# 6. Export ML data
```

**مزایا**:
- ✅ خودکار و بدون نیاز به دخالت
- ✅ Error handling داخلی
- ✅ Progress tracking
- ✅ خودکار dashboard را باز می‌کند

**معایب**:
- ⚠️ اگر یک مرحله fail شود، کل pipeline متوقف می‌شود

**در صورت خطا**: لاگ‌ها را بررسی کنید و به گزینه B بروید

---

### گزینه B: اجرای دستی مرحله به مرحله
زمان: ~5 دقیقه (با بررسی بین مراحل)

#### 1️⃣ محاسبه Metrics
```bash
echo "مرحله 1: محاسبه metrics برای 188 race..."
python scripts/batch_calculate_metrics.py --resume

# بررسی تعداد فایل‌های تولید شده
dir /s /b data\processed\metrics\*.json | find /c ".json"
# انتظار: ~380 فایل (188 races × 2 lanes + summaries)
```

**زمان تخمینی**: 20-30 ثانیه
**خروجی**: `data\processed\metrics\[competition]\[race]_metrics_[lane].json`

**چک‌پوینت**:
- [ ] همه 188 race پردازش شدند
- [ ] هر race دارای 2 فایل است (left + right)
- [ ] Summary JSON معتبر است

---

#### 2️⃣ Aggregation آمار
```bash
echo "مرحله 2: تولید آمار competition..."
python scripts/aggregate_competition_stats.py

# بررسی خروجی
dir data\processed\aggregates
type data\processed\aggregates\overall_statistics.json
```

**زمان تخمینی**: 5 ثانیه
**خروجی**:
- `aggregate_metrics.csv` - metrics تجمیعی
- `competition_summaries.json` - آمار هر مسابقه
- `leaderboard_top20.csv` - برترین‌ها
- `overall_statistics.json` - آمار کلی

**چک‌پوینت**:
- [ ] 4 فایل اصلی تولید شدند
- [ ] CSV ها قابل باز شدن هستند
- [ ] آمار منطقی است (velocity > 0, efficiency در 0-1)

---

#### 3️⃣ مقایسه Races
```bash
echo "مرحله 3: مقایسه همه races..."
python scripts/compare_races.py --all --competition all

# این مرحله ممکن است طولانی باشد
# خروجی در console نمایش داده می‌شود
```

**زمان تخمینی**: 20-40 ثانیه
**خروجی**: console output با نتایج مقایسه

**چک‌پوینت**:
- [ ] همه races مقایسه شدند
- [ ] پیش‌بینی برنده برای هر race موجود است
- [ ] هیچ خطای critical نیست

---

#### 4️⃣ تولید نمودارها
```bash
echo "مرحله 4: تولید visualizations..."
python src/visualization/race_plots.py

# بررسی plots
dir data\processed\plots
dir data\processed\plots\*.png
```

**زمان تخمینی**: 15-20 ثانیه
**خروجی**: PNG files در `data\processed\plots\`

**انواع نمودارها**:
- `velocity_comparison_*.png` - مقایسه سرعت
- `competition_summary.png` - خلاصه مسابقه
- `leaderboard_top10.png` - برترین‌ها

**چک‌پوینت**:
- [ ] حداقل 3 نوع plot تولید شد
- [ ] نمودارها واضح و خوانا هستند
- [ ] کیفیت تصویر مناسب است (150 DPI)

---

#### 5️⃣ ساخت Dashboard
```bash
echo "مرحله 5: ساخت interactive dashboard..."
python scripts/generate_html_dashboard.py

# باز کردن dashboard
start data\processed\dashboard\index.html
```

**زمان تخمینی**: 5 ثانیه
**خروجی**: `data\processed\dashboard\index.html`

**چک‌پوینت در مرورگر**:
- [ ] Dashboard باز می‌شود بدون خطا
- [ ] همه آمارها نمایش داده می‌شوند
- [ ] نمودارها embed شده‌اند
- [ ] responsive است (تغییر اندازه پنجره)
- [ ] طراحی professional و جذاب است

---

#### 6️⃣ خروجی ML
```bash
echo "مرحله 6: export ML-ready data..."
python scripts/export_ml_data.py --test-size 0.2 --random-seed 42

# بررسی فایل‌ها
dir data\processed\ml_ready
type data\processed\ml_ready\dataset_metadata.json
```

**زمان تخمینی**: 10 ثانیه
**خروجی**:
- `features.csv` - feature matrix
- `features.npz` - NumPy format
- `train.csv` / `test.csv` - 80/20 split
- `dataset_metadata.json` - documentation

**چک‌پوینت**:
- [ ] 6 فایل اصلی تولید شدند
- [ ] features.csv حدود 376 ردیف دارد (188×2)
- [ ] train/test split صحیح است (80/20)
- [ ] metadata شامل توضیحات features است

---

## 📊 بررسی کیفیت خروجی

### 1. آمار کلی
```bash
# باز کردن overall statistics
type data\processed\aggregates\overall_statistics.json | more

# موارد کلیدی برای بررسی:
# - total_climbers: باید ~376 باشد (188 races × 2 lanes)
# - avg_max_velocity: باید مثبت و معقول باشد
# - avg_path_efficiency: باید بین 0 تا 1 باشد
# - competition_count: باید 5 باشد
```

**سوالات بررسی**:
- ✅ آیا تعداد climbers منطقی است؟
- ✅ آیا میانگین velocity مثبت است؟
- ✅ آیا path efficiency در بازه 0-1 است؟
- ✅ آیا همه 5 competition در نتایج هستند؟

---

### 2. Leaderboard
```bash
# مشاهده برترین‌ها
type data\processed\aggregates\leaderboard_top20.csv

# خواندن به صورت table-formatted (در PowerShell):
powershell -Command "Import-Csv data\processed\aggregates\leaderboard_top20.csv | Format-Table"
```

**سوالات بررسی**:
- ✅ آیا برترین climbers با انتظارات شما مطابقت دارند؟
- ✅ آیا velocities منطقی هستند؟
- ✅ آیا هر competition نماینده‌ای در top 20 دارد؟

---

### 3. کیفیت نمودارها
```bash
# باز کردن نمودارهای کلیدی
start data\processed\plots\competition_summary.png
start data\processed\plots\leaderboard_top10.png
start data\processed\plots\velocity_comparison_race001.png
```

**معیارهای کیفی**:
- ✅ وضوح: آیا متن‌ها خوانا هستند؟
- ✅ رنگ: آیا رنگ‌بندی مناسب است؟
- ✅ Labels: آیا محورها و عنوان‌ها صحیح هستند؟
- ✅ Legend: آیا راهنما واضح است؟
- ✅ Layout: آیا چیدمان مناسب است؟

---

### 4. Winner Prediction Accuracy
```bash
# بررسی چند race نمونه
python scripts/compare_races.py --race race001 --competition seoul_2024
python scripts/compare_races.py --race race050 --competition villars_2024
python scripts/compare_races.py --race race100 --competition zilina_2025

# مقایسه پیش‌بینی با نتیجه واقعی (از metadata)
```

**محاسبه دقت**:
1. چند race را تست کنید (حداقل 20)
2. نتیجه واقعی را از metadata بررسی کنید
3. accuracy = (تعداد درست) / (تعداد کل) × 100
4. هدف: accuracy > 66.7%

---

### 5. ML Data Validation
```bash
# بررسی feature matrix
powershell -Command "Import-Csv data\processed\ml_ready\features.csv | Measure-Object"

# بررسی NumPy file
python -c "import numpy as np; data = np.load('data/processed/ml_ready/features.npz'); print('Features shape:', data['X'].shape); print('Labels shape:', data['y'].shape)"
```

**معیارهای صحت**:
- ✅ تعداد samples: ~376 (188 races × 2 lanes)
- ✅ تعداد features: 8
- ✅ Missing values: نباید وجود داشته باشد
- ✅ Feature ranges: منطقی و معتبر

---

## 🔧 عیب‌یابی (Troubleshooting)

### مشکل 1: خطای "Pose file not found"

**علامت**:
```
FileNotFoundError: Pose file not found: data/processed/poses/[competition]/[race]_poses.json
```

**راه‌حل**:
```bash
# بررسی موجودی pose files
dir /s /b data\processed\poses\*.json

# اگر کم هستند، دوباره pose extraction اجرا کنید
python scripts/batch_pose_extraction.py --competition [competition_name]
```

---

### مشکل 2: Metrics در pixel هستند نه meter

**علامت**:
- velocities بسیار بزرگ (مثلاً 500+ به جای 2-3 m/s)

**راه‌حل**:
```bash
# اجرای batch calibration
python scripts/run_batch_calibration_tests.py --max-races 188

# سپس دوباره metrics را محاسبه کنید
python scripts/batch_calculate_metrics.py --resume --force
```

---

### مشکل 3: خطای Memory (Out of Memory)

**علامت**:
```
MemoryError: Unable to allocate array
```

**راه‌حل**:
```bash
# پردازش در chunks کوچکتر
python scripts/batch_calculate_metrics.py --max-races 50 --competition seoul_2024
python scripts/batch_calculate_metrics.py --max-races 50 --competition villars_2024
python scripts/batch_calculate_metrics.py --max-races 50 --competition chamonix_2024
python scripts/batch_calculate_metrics.py --max-races 50 --competition innsbruck_2024
python scripts/batch_calculate_metrics.py --max-races 75 --competition zilina_2025

# سپس aggregation
python scripts/aggregate_competition_stats.py
```

---

### مشکل 4: نمودارها ایجاد نمی‌شوند

**علامت**:
- خطا در matplotlib
- فایل PNG خالی است

**راه‌حل**:
```bash
# بررسی matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
# باید 'Agg' باشد

# اگر مشکل ادامه داشت:
pip install --upgrade matplotlib
```

---

### مشکل 5: Dashboard باز نمی‌شود یا خالی است

**علامت**:
- صفحه سفید در مرورگر
- خطای 404

**راه‌حل**:
```bash
# بررسی وجود فایل
dir data\processed\dashboard\index.html

# اگر وجود ندارد، دوباره generate کنید
python scripts/generate_html_dashboard.py

# اگر باز هم مشکل داشت، با مرورگر دیگری تست کنید
```

---

## 🎯 مراحل بعدی

### کوتاه‌مدت (این هفته)

#### 1. Commit نتایج
```bash
# اضافه کردن aggregate outputs
git add data/processed/aggregates/*.csv
git add data/processed/aggregates/*.json
git add data/processed/plots/*.png
git add data/processed/dashboard/index.html
git add data/processed/ml_ready/*.csv
git add data/processed/ml_ready/*.npz
git add data/processed/ml_ready/*.json

# Commit
git commit -m "data: add Phase 3 analytics results for 188 races

- Aggregate metrics and competition statistics
- Leaderboard and comparative analysis
- Visualizations (plots and dashboard)
- ML-ready datasets (CSV + NumPy formats)

Generated from full pipeline run (188 races, 5 competitions)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push
git push origin main
git push github main
```

---

#### 2. ایجاد خلاصه‌ای برای مستندات
```bash
# ایجاد یک فایل RESULTS_SUMMARY.md
# شامل:
# - Overall statistics
# - Top 10 performers
# - Competition comparison
# - Key insights
```

---

### میان‌مدت (هفته آینده)

#### گزینه A: بهبود Phase 3
درخواست از UI claude.ai/code:
- بهبود winner prediction با ML model
- Integration کامل calibration
- Interactive visualizations (Plotly)
- Performance optimization (multiprocessing)

#### گزینه B: شروع Phase 4 (Machine Learning)
درخواست از UI claude.ai/code:
- NARX neural networks
- Climber classification
- Performance prediction
- Anomaly detection

#### گزینه C: آماده‌سازی Publication
درخواست از UI claude.ai/code:
- High-resolution figures (300 DPI)
- Statistical analysis report
- LaTeX tables
- Methodology description

---

## 📝 چک‌لیست نهایی

### قبل از اتمام تست:
- [ ] همه scripts اجرا شدند بدون خطای critical
- [ ] 188 race پردازش شدند
- [ ] همه خروجی‌های کلیدی تولید شدند
- [ ] کیفیت نمودارها قابل قبول است
- [ ] Dashboard کار می‌کند و زیباست
- [ ] ML datasets معتبر هستند
- [ ] نتایج commit شدند

### قبل از درخواست از UI:
- [ ] مشکلات شناسایی شده مستند شدند
- [ ] بهبودهای موردنیاز لیست شدند
- [ ] اولویت‌بندی انجام شد
- [ ] Prompt واضح و مشخص آماده شد

---

## 💡 نکات مهم

1. **صبر داشته باشید**: پردازش 188 race ممکن است 5-10 دقیقه طول بکشد

2. **Resume capability**: اگر process قطع شد، با flag `--resume` دوباره اجرا کنید

3. **حجم داده**: خروجی‌های کامل حدود 50-100 MB هستند

4. **Calibration**: برای نتایج دقیق‌تر (meter-based)، ابتدا calibration اجرا کنید

5. **Backup**: قبل از regenerate، backup بگیرید

6. **Documentation**: همیشه MASTER_CONTEXT.md را به‌روز نگه دارید

---

## 📞 کمک و پشتیبانی

اگر با مشکلی روبرو شدید:

1. **لاگ‌ها را بررسی کنید**: اکثر scripts خطاهای واضح می‌دهند

2. **مستندات را بخوانید**:
   - `docs/PHASE_3_COMPLETION_REPORT.md` - جزئیات پیاده‌سازی
   - `docs/PHASE_3_PLAN.md` - برنامه اولیه
   - `MASTER_CONTEXT.md` - نمای کلی پروژه

3. **دوباره تست کنید**: با یک race ساده شروع کنید

4. **پرسیدن از Claude**: در session جدید، context کامل ارائه دهید

---

**موفق باشید!** 🎉

این راهنما برای کمک به شما در تست و validation کامل Phase 3 طراحی شده است. اگر سوالی دارید، به MASTER_CONTEXT.md مراجعه کنید یا یک session جدید با Claude شروع کنید.

---

**تهیه شده توسط**: Claude Code
**تاریخ**: 2025-11-15
**نسخه**: 1.0
