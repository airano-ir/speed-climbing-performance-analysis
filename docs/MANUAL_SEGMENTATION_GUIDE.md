# راهنمای استخراج دستی مسابقات - Manual Race Segmentation Guide

**تاریخ**: 2025-11-14
**نسخه**: 2.0 (اصلاح شده)
**زبان**: فارسی + English

---

## 📊 وضعیت فعلی (Current Status)

### ✅ کامل شده:
- **Parser Script**: اصلاح timestamps و حذف race 15
- **YAML Configs**: تولید شده برای 3 مسابقه
- **Seoul 2024**: در حال پردازش مجدد با timestamps اصلاح شده (31 مسابقه)

### ⏳ باقی‌مانده برای شما:
- **Villars 2024**: 24 مسابقه (~15 دقیقه)
- **Chamonix 2024**: 32 مسابقه (~20 دقیقه)
- **Total**: 56 مسابقه + بررسی Seoul (31 مسابقه)

---

## 🔧 اصلاحات انجام شده

### Seoul 2024:
1. **مسابقات زودتر تمام شده** → +5 ثانیه به end_time:
   - Races 1-7 (همه 1/8 final Women)
   - Races 10, 13, 16, 17, 18, 20
   - Races 25, 26 (Semi finals Women)
   - Races 29-32 (Small finals + Finals)

2. **Race 15 حذف شد**: False start خیلی کوتاه (Michael Holm vs Sam Watson)

3. **Total**: 31 مسابقه (قبلاً 32 بود)

### نکته مهم:
**در همه ویدئوها قبل از شروع 3 بوق می‌زند و مسابقه از بوق سوم شروع می‌شود.**

---

## 🚀 دستورات اجرا

### گام 1: بررسی Seoul 2024 (پس از اتمام)

منتظر بمانید تا Seoul تمام شود، سپس:

```bash
# بررسی تعداد
cd "g:\My Drive\Projects\Speed Climbing Performance Analysis"
ls -1 data/race_segments/seoul_2024/*.mp4 | wc -l
# باید 31 نمایش دهد
```

**PowerShell:**
```powershell
(Get-ChildItem "data/race_segments/seoul_2024/*.mp4").Count
# باید 31 نمایش دهد
```

**بررسی یک نمونه:**
```bash
# بررسی metadata مسابقه اول
cat "data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race001_metadata.json"

# بررسی summary
cat "data/race_segments/seoul_2024/Speed_finals_Seoul_2024_summary.json"
```

---

### گام 2: استخراج Villars 2024 (24 مسابقه)

```bash
cd "g:\My Drive\Projects\Speed Climbing Performance Analysis"

python src/utils/manual_race_segmenter.py ^
  "configs/race_timestamps/villars_2024.yaml" ^
  --output-dir "data/race_segments/villars_2024" ^
  --buffer-before 1.5 ^
  --buffer-after 1.5 ^
  --no-refine
```

**زمان تخمینی**: 12-15 دقیقه
**خروجی انتظاری**: 24 کلیپ MP4 + 24 metadata JSON + 1 summary JSON

**نکته Villars**:
- دور 1/8 نهایی مردان مشکل فنی داشت و دوباره اجرا شد (Rerun)
- Auto belay malfunction در lane چپ

---

### گام 3: استخراج Chamonix 2024 (32 مسابقه)

```bash
cd "g:\My Drive\Projects\Speed Climbing Performance Analysis"

python src/utils/manual_race_segmenter.py ^
  "configs/race_timestamps/chamonix_2024.yaml" ^
  --output-dir "data/race_segments/chamonix_2024" ^
  --buffer-before 1.5 ^
  --buffer-after 1.5 ^
  --no-refine
```

**زمان تخمینی**: 18-20 دقیقه
**خروجی انتظاری**: 32 کلیپ MP4 + 32 metadata JSON + 1 summary JSON

---

## 📁 ساختار نهایی

بعد از اتمام همه:

```
data/race_segments/
├── seoul_2024/                     ✅ (31 مسابقه - اصلاح شده)
│   ├── Speed_finals_Seoul_2024_race001.mp4
│   ├── Speed_finals_Seoul_2024_race001_metadata.json
│   ├── ...
│   ├── Speed_finals_Seoul_2024_race031.mp4
│   ├── Speed_finals_Seoul_2024_race031_metadata.json
│   └── Speed_finals_Seoul_2024_summary.json
│
├── villars_2024/                   ⏳ (24 مسابقه)
│   ├── Speed_finals_Villars_2024_race001.mp4
│   ├── ...
│   └── Speed_finals_Villars_2024_summary.json
│
└── chamonix_2024/                  ⏳ (32 مسابقه)
    ├── Speed_finals_Chamonix_2024_race001.mp4
    ├── ...
    └── Speed_finals_Chamonix_2024_summary.json
```

**Total**: 87 مسابقه (31 + 24 + 32)

---

## 🔍 بررسی کیفیت

### Checklist بعد از هر مسابقه:

**Seoul:**
- [ ] تعداد فایل‌های MP4: 31
- [ ] تعداد metadata files: 31
- [ ] فایل summary وجود دارد
- [ ] Race 15 وجود ندارد (حذف شده)
- [ ] یک نمونه ویدئو را باز کنید و بررسی کنید

**Villars:**
- [ ] تعداد فایل‌های MP4: 24
- [ ] تعداد metadata files: 24
- [ ] فایل summary وجود دارد
- [ ] یک نمونه ویدئو را باز کنید

**Chamonix:**
- [ ] تعداد فایل‌های MP4: 32
- [ ] تعداد metadata files: 32
- [ ] فایل summary وجود دارد
- [ ] یک نمونه ویدئو را باز کنید

---

## 📊 دستورات بررسی سریع

### تعداد کل مسابقات:

**Bash:**
```bash
echo "Seoul: $(ls data/race_segments/seoul_2024/*.mp4 2>/dev/null | wc -l) / 31"
echo "Villars: $(ls data/race_segments/villars_2024/*.mp4 2>/dev/null | wc -l) / 24"
echo "Chamonix: $(ls data/race_segments/chamonix_2024/*.mp4 2>/dev/null | wc -l) / 32"
echo "Total: $(find data/race_segments -name '*.mp4' 2>/dev/null | wc -l) / 87"
```

**PowerShell:**
```powershell
Write-Host "Seoul:" (Get-ChildItem "data\race_segments\seoul_2024\*.mp4" -ErrorAction SilentlyContinue).Count "/ 31"
Write-Host "Villars:" (Get-ChildItem "data\race_segments\villars_2024\*.mp4" -ErrorAction SilentlyContinue).Count "/ 24"
Write-Host "Chamonix:" (Get-ChildItem "data\race_segments\chamonix_2024\*.mp4" -ErrorAction SilentlyContinue).Count "/ 32"
$total = (Get-ChildItem -Recurse "data\race_segments\*.mp4" -ErrorAction SilentlyContinue).Count
Write-Host "Total:" $total "/ 87"
```

### حجم کل:

**Bash:**
```bash
du -sh data/race_segments/
```

**PowerShell:**
```powershell
$size = (Get-ChildItem -Recurse data\race_segments | Measure-Object -Property Length -Sum).Sum
Write-Host "Total size:" ([math]::Round($size/1GB, 2)) "GB"
```

**انتظار**: حدود 2-3 GB

---

## ⚙️ پارامترها

### توضیح:

| پارامتر | مقدار | چرا؟ |
|---------|-------|------|
| `--buffer-before 1.5` | 1.5 ثانیه | برای دیدن 3 بوق قبل از شروع |
| `--buffer-after 1.5` | 1.5 ثانیه | برای دیدن واکنش بعد از پایان |
| `--no-refine` | بله | timestamps شما دقیق است، نیازی به detection نیست (سریع‌تر) |

---

## 🛠️ عیب‌یابی

### مشکل 1: Seoul بیشتر از 31 مسابقه دارد

**بررسی:**
```bash
cat configs/race_timestamps/seoul_2024.yaml | grep "race_id:" | wc -l
# باید 31 باشد
```

**راه‌حل**: دوباره parser را اجرا کنید:
```bash
python scripts/parse_timestamps_to_yaml.py
```

---

### مشکل 2: "ffmpeg not found"

**راه‌حل**:
```bash
# بررسی ffmpeg
ffmpeg -version

# اگر نصب نیست: دانلود از https://ffmpeg.org/
```

---

### مشکل 3: مسابقه‌ای خیلی کوتاه است (< 3 ثانیه)

**علت**: ممکن است end_time نیاز به اصلاح داشته باشد

**راه‌حل**: به من اطلاع دهید:
- شماره مسابقه
- مدت فعلی
- مسابقه Seoul, Villars یا Chamonix؟

---

### مشکل 4: مسابقه‌ای طولانی‌تر از انتظار است (> 15 ثانیه)

**علت**: ممکن است سقوط، لغزش یا replay داشته باشد

**بررسی metadata**:
```bash
cat "data/race_segments/.../race_metadata.json" | grep duration
```

اگر منطقی است، مشکلی نیست. اگر خیلی طولانی است (> 20s)، به من اطلاع دهید.

---

## 📞 گزارش نهایی به من

بعد از اتمام همه، این اطلاعات را ارسال کنید:

```bash
echo "=== FINAL REPORT ==="
echo "Seoul: $(ls data/race_segments/seoul_2024/*.mp4 | wc -l) / 31"
echo "Villars: $(ls data/race_segments/villars_2024/*.mp4 | wc -l) / 24"
echo "Chamonix: $(ls data/race_segments/chamonix_2024/*.mp4 | wc -l) / 32"
echo "Total: $(find data/race_segments -name '*.mp4' | wc -l) / 87"
echo ""
echo "=== Sample Metadata (Seoul Race 1) ==="
cat "data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race001_metadata.json"
```

یا ساده‌تر: فقط بگویید "همه تمام شد - 87 مسابقه آماده!"

---

## 🎯 مرحله بعدی (بعد از اتمام)

1. ✅ سازماندهی single-race videos (5 ویدئوی تک مسابقه)
2. ✅ آپدیت MASTER_CONTEXT
3. ✅ Git commit
4. 🚀 شروع Phase 2: Pose Estimation & Analysis

---

## 💡 نکات مهم

1. **سرعت**: با `--no-refine` هر مسابقه ~30 ثانیه طول می‌کشد
2. **دقت**: timestamps شما دقیق است، نیازی به detection نیست
3. **Buffer**: 1.5s قبل و بعد کافی است (3 بوق + واکنش)
4. **Race 15**: حذف شده است (false start)
5. **Total**: 87 مسابقه (نه 88)

---

**موفق باشید! 🎯**

اگر مشکلی پیش آمد یا سوالی داشتید، به من اطلاع دهید.
