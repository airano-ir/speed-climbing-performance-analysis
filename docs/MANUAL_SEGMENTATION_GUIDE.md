# راهنمای استخراج دستی مسابقات - Manual Race Segmentation Guide

**تاریخ**: 2025-11-14
**نسخه**: 4.0 (با Zilina 2025 و Innsbruck corrections)
**زبان**: فارسی + English

---

## 📊 وضعیت فعلی (Current Status)

### ✅ کامل شده:
- **Parser Script**: اصلاح همه timestamps برای 5 مسابقه
- **YAML Configs**: تولید شده برای 5 مسابقه (Seoul, Villars, Chamonix, Innsbruck, Zilina)
- **Seoul 2024**: ✅ اتمام (31 مسابقه)
- **Villars 2024**: ✅ اتمام (24 مسابقه)
- **Chamonix 2024**: ✅ اتمام (32 مسابقه)
- **Innsbruck 2024**: ✅ اتمام (32 مسابقه)
- **Late Start Handling**: پشتیبانی از مسابقات با شروع دیرهنگام (3s buffer)
- **Manual Race Segmenter**: اصلاح شده برای late_start flag
- **Batch Processing Script**: پشتیبانی از 5 مسابقه

### ⏳ در حال پردازش:
- **Zilina 2025**: 69 مسابقه (~40 دقیقه) - European Youth Championships

### 📈 پیشرفت کلی:
- **تمام شده**: 119/188 مسابقه (63.3%)
- **باقی‌مانده**: 69 مسابقه Zilina

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

### Villars 2024:
1. **مسابقات زودتر تمام شده**:
   - Races 1, 7, 8, 12 → +5s به end_time
   - Race 2 → +4s به end_time (خاص)

2. **شروع دیرهنگام** (3s buffer):
   - Races 2, 13, 15, 23 → `late_start: true`

3. **Total**: 24 مسابقه (1/8 final Men rerun به دلیل مشکل auto belay)

### Chamonix 2024:
1. **مسابقات زودتر تمام شده**:
   - Races 1, 2, 4, 5, 6, 7, 11, 14, 15, 18, 19, 20, 21, 26, 29, 32 → +5s
   - Race 30 → +8s (خاص)

2. **شروع دیرهنگام** (3s buffer):
   - Races 20, 26 → `late_start: true`

3. **Total**: 32 مسابقه

### Innsbruck 2024:
1. **اصلاحات زمانی**:
   - **Race 2** (خاص): +20s به start_time + +8s به end_time
   - **Races 3, 10, 11, 18, 23, 30**: +5s به end_time

2. **شروع دیرهنگام** (3s buffer) - خیلی زیاد:
   - Races 2, 4, 6, 8, 9, 10, 12, 14, 15, 16, 17, 20, 21, 23, 24, 25, 27, 32
   - Total: 18 از 32 مسابقه دارای late_start

3. **نکات**:
   - نام‌های ناقص ورزشکاران (فقط نام یا نام خانوادگی)
   - مکان: Innsbruck, Austria (European Cup)
   - گزارشگر: Matthew Fall (عضو تیم سرعت بریتانیا)

4. **Total**: 32 مسابقه

### Zilina 2025 (جدید!):
1. **مسابقه بزرگ** - European Youth Championships:
   - **U17 Women**: 8 races (1/8 final) + 4 (QF) + 2 (SF) + 1 (Bronze) + 1 (Gold) = 16 races
   - **U17 Men**: 7 races (1/8 final) + 4 (QF) + 2 (SF) + 1 (Bronze) + 1 (Gold) = 15 races
   - **U19 Women**: 8 races (1/8 final) + 4 (QF) + 2 (SF) + 1 (Bronze) + 1 (Gold) = 16 races
   - **U19 Men**: 7 races (1/8 final) + 4 (QF) + 2 (SF) + 1 (Bronze) + 1 (Gold) = 15 races
   - **U21 Women**: 4 races (QF) + 1 (SF) = 5 races
   - **U21 Men**: 2 races (QF) + 2 (SF) + 1 (Bronze - rerun) = 5 races
   - **Total**: 69 مسابقه (72 اصلی - 3 حذف شد: races 13, 51, 55)

2. **اصلاحات زمانی** (بعد از بررسی ویدئوها):
   - **42 مسابقه**: start_time -4s
   - **Race 62**: start_time -6s
   - **Races 56, 58**: start_time -10s
   - **Races 15, 16, 19, 20, 38, 48**: end_time +10s
   - **Races 13, 51, 55**: حذف شدند (incomplete)

3. **نکات مهم**:
   - دیوار خیلی لیز بود و سقوط‌های زیادی رخ داد
   - Race 6 (U17 Men): rerun به دلیل سقوط هر دو ورزشکار
   - Race 68 (U21 Men Bronze): rerun به دلیل سقوط هر دو ورزشکار
   - Races 13, 38: false starts
   - نام‌های ناقص ورزشکاران (فقط نام یا نام خانوادگی)

3. **قهرمانان برجسته**:
   - Leo (France U17): قهرمان جهان و اروپا
   - Aidan (Germany U19): قهرمان جهان و اروپا

4. **مکان**: Zilina, Slovakia (European Youth Championships)

### نکته مهم:
**در همه ویدئوها معمولاً قبل از شروع 3 بوق می‌زند و مسابقه از بوق سوم شروع می‌شود، اما گاهی 1، 2 یا هیچ بوقی نیست و بلافاصله شروع می‌شود.**

---

## 🚀 دستورات اجرا

### گام 0: بررسی وضعیت فعلی (استفاده از اسکریپت check_progress.ps1)

**راه سریع**: اسکریپت `check_progress.ps1` را اجرا کنید:

```powershell
cd "g:\My Drive\Projects\Speed Climbing Performance Analysis"
powershell -ExecutionPolicy Bypass -File check_progress.ps1
```

این اسکریپت نشان می‌دهد:
- ✅ تعداد فایل‌های ساخته شده برای هر مسابقه
- 🕐 زمان ساخت آخرین فایل (برای تشخیص فایل‌های قدیمی)
- 📊 درصد پیشرفت کلی
- 🔄 وضعیت پروسس‌های Python در حال اجرا
- ⚠️ فایل‌های قدیمی که شاید نیاز به regenerate داشته باشند

**خروجی نمونه:**
```
chamonix_2024: 32/32 races - COMPLETE (newest: 6.8 hours ago)
innsbruck_2024: 32/32 races - COMPLETE (newest: 5.7 hours ago)
seoul_2024: 31/31 races - COMPLETE (newest: 1 minutes ago)
villars_2024: 24/24 races - COMPLETE (newest: 6.8 hours ago)
zilina_2025: 15/69 races - IN PROGRESS (15/69) (newest: 2 minutes ago)

Total Progress: 134/188 races
Completion: 70.2%
```

### گام 1: پاک کردن فایل‌های موقت (اگر لازم است)

اگر فایل‌های `temp_*.mp4` در Seoul دیدید:

```powershell
cd "g:\My Drive\Projects\Speed Climbing Performance Analysis\data\race_segments\seoul_2024"
Remove-Item temp_*.mp4
```

### گام 2: بررسی Seoul 2024 (پس از اتمام)

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

### گام 3: استخراج Chamonix 2024 (32 مسابقه) ✅ COMPLETE

**Status**: مسابقه Chamonix کامل شده است.

**بررسی**:
```powershell
(Get-ChildItem "data\race_segments\chamonix_2024\*.mp4").Count
# باید 32 باشد
```

---

### گام 4: استخراج Innsbruck 2024 (32 مسابقه) ✅ COMPLETE

**Status**: مسابقه Innsbruck کامل شده است.

**بررسی**:
```powershell
(Get-ChildItem "data\race_segments\innsbruck_2024\*.mp4").Count
# باید 32 باشد
```

**نکته Innsbruck**:
- 18 از 32 مسابقه دارای late_start بودند
- Race 2 اصلاحات خاص داشت: +20s start, +8s end

---

### گام 5: منتظر اتمام Zilina 2025 (72 مسابقه) ⏳ IN PROGRESS

**Status**: Zilina در حال پردازش است (اتوماتیک via batch script).

**مانیتور کردن**:
```powershell
# استفاده از اسکریپت check_progress.ps1
powershell -ExecutionPolicy Bypass -File check_progress.ps1

# یا مستقیم
(Get-ChildItem "data\race_segments\zilina_2025\*.mp4" -ErrorAction SilentlyContinue).Count
# انتظار: 0-72 (بسته به پیشرفت)
```

**زمان تخمینی**: ~35-40 دقیقه (69 مسابقه)
**خروجی انتظاری**: 69 کلیپ MP4 + 69 metadata JSON + 1 summary JSON

**نکات Zilina**:
- بزرگترین مسابقه (69 races - 3 races حذف شد)
- European Youth Championships
- 3 رده سنی: U17, U19, U21
- دیوار لیز - سقوط‌های زیاد
- Race 6 و 68: reruns
- Races 13, 51, 55: حذف شدند (incomplete)

---

## 📁 ساختار نهایی

بعد از اتمام همه:

```
data/race_segments/
├── seoul_2024/                     ✅ (31 مسابقه - COMPLETE)
│   ├── Speed_finals_Seoul_2024_race001.mp4
│   ├── Speed_finals_Seoul_2024_race001_metadata.json
│   ├── ...
│   ├── Speed_finals_Seoul_2024_race031.mp4
│   ├── Speed_finals_Seoul_2024_race031_metadata.json
│   └── Speed_finals_Seoul_2024_summary.json
│
├── villars_2024/                   ✅ (24 مسابقه - COMPLETE)
│   ├── Speed_finals_Villars_2024_race001.mp4
│   ├── ...
│   └── Speed_finals_Villars_2024_summary.json
│
├── chamonix_2024/                  ✅ (32 مسابقه - COMPLETE)
│   ├── Speed_finals_Chamonix_2024_race001.mp4
│   ├── ...
│   └── Speed_finals_Chamonix_2024_summary.json
│
├── innsbruck_2024/                 ✅ (32 مسابقه - COMPLETE)
│   ├── Speed_finals_Innsbruck_2024_race001.mp4
│   ├── ...
│   └── Speed_finals_Innsbruck_2024_summary.json
│
└── zilina_2025/                    ⏳ (69 مسابقه - IN PROGRESS)
    ├── Speed_finals_Zilina_2025_race001.mp4
    ├── Speed_finals_Zilina_2025_race001_metadata.json
    ├── ...
    ├── Speed_finals_Zilina_2025_race069.mp4
    ├── Speed_finals_Zilina_2025_race069_metadata.json
    └── Speed_finals_Zilina_2025_summary.json
```

**Total**: 188 مسابقه (31 + 24 + 32 + 32 + 69)
**Status**: 119/188 COMPLETE (63.3%)

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

**Innsbruck:**
- [ ] تعداد فایل‌های MP4: 32
- [ ] تعداد metadata files: 32
- [ ] فایل summary وجود دارد
- [ ] یک نمونه ویدئو را باز کنید

**Zilina:**
- [ ] تعداد فایل‌های MP4: 69
- [ ] تعداد metadata files: 69
- [ ] فایل summary وجود دارد
- [ ] یک نمونه ویدئو را باز کنید
- [ ] Races 13, 51, 55 وجود ندارند (حذف شده)

---

## 📊 دستورات بررسی سریع

### ⭐ راه بهتر: استفاده از check_progress.ps1

```powershell
powershell -ExecutionPolicy Bypass -File check_progress.ps1
```

این اسکریپت همه چیز را نشان می‌دهد (تعداد، وضعیت، زمان، فایل‌های قدیمی).

### تعداد کل مسابقات (روش دستی):

**PowerShell (پیشنهادی):**
```powershell
Write-Host "Seoul:" (Get-ChildItem "data\race_segments\seoul_2024\*.mp4" -ErrorAction SilentlyContinue).Count "/ 31"
Write-Host "Villars:" (Get-ChildItem "data\race_segments\villars_2024\*.mp4" -ErrorAction SilentlyContinue).Count "/ 24"
Write-Host "Chamonix:" (Get-ChildItem "data\race_segments\chamonix_2024\*.mp4" -ErrorAction SilentlyContinue).Count "/ 32"
Write-Host "Innsbruck:" (Get-ChildItem "data\race_segments\innsbruck_2024\*.mp4" -ErrorAction SilentlyContinue).Count "/ 32"
Write-Host "Zilina:" (Get-ChildItem "data\race_segments\zilina_2025\*.mp4" -ErrorAction SilentlyContinue).Count "/ 69"
$total = (Get-ChildItem -Recurse "data\race_segments\*.mp4" -ErrorAction SilentlyContinue).Count
Write-Host "Total:" $total "/ 188"
```

**Bash:**
```bash
echo "Seoul: $(ls data/race_segments/seoul_2024/*.mp4 2>/dev/null | wc -l) / 31"
echo "Villars: $(ls data/race_segments/villars_2024/*.mp4 2>/dev/null | wc -l) / 24"
echo "Chamonix: $(ls data/race_segments/chamonix_2024/*.mp4 2>/dev/null | wc -l) / 32"
echo "Innsbruck: $(ls data/race_segments/innsbruck_2024/*.mp4 2>/dev/null | wc -l) / 32"
echo "Zilina: $(ls data/race_segments/zilina_2025/*.mp4 2>/dev/null | wc -l) / 69"
echo "Total: $(find data/race_segments -name '*.mp4' 2>/dev/null | wc -l) / 188"
```

### حجم کل:

**PowerShell:**
```powershell
$size = (Get-ChildItem -Recurse data\race_segments | Measure-Object -Property Length -Sum).Sum
Write-Host "Total size:" ([math]::Round($size/1GB, 2)) "GB"
```

**Bash:**
```bash
du -sh data/race_segments/
```

**انتظار**: حدود 5-6 GB (188 مسابقه)

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

بعد از اتمام Zilina (یا هر زمان که می‌خواهید):

**راه ساده:** فقط `check_progress.ps1` را اجرا کنید:
```powershell
powershell -ExecutionPolicy Bypass -File check_progress.ps1
```

**یا این دستور:**
```powershell
echo "=== FINAL REPORT ==="
Write-Host "Seoul:" (Get-ChildItem "data\race_segments\seoul_2024\*.mp4").Count "/ 31"
Write-Host "Villars:" (Get-ChildItem "data\race_segments\villars_2024\*.mp4").Count "/ 24"
Write-Host "Chamonix:" (Get-ChildItem "data\race_segments\chamonix_2024\*.mp4").Count "/ 32"
Write-Host "Innsbruck:" (Get-ChildItem "data\race_segments\innsbruck_2024\*.mp4").Count "/ 32"
Write-Host "Zilina:" (Get-ChildItem "data\race_segments\zilina_2025\*.mp4").Count "/ 69"
$total = (Get-ChildItem -Recurse "data\race_segments\*.mp4").Count
Write-Host "Total:" $total "/ 188"
```

**پیام ساده:** "همه تمام شد - 188 مسابقه آماده!" 🎉

---

## 🎯 مرحله بعدی (بعد از اتمام)

1. ✅ سازماندهی single-race videos (5 ویدئوی تک مسابقه)
2. ✅ آپدیت MASTER_CONTEXT
3. ✅ Git commit
4. 🚀 شروع Phase 2: Pose Estimation & Analysis

---

## 💡 نکات مهم

1. **check_progress.ps1**: همیشه برای چک کردن وضعیت از این اسکریپت استفاده کنید
2. **سرعت**: با `--no-refine` هر مسابقه ~30-40 ثانیه طول می‌کشد
3. **دقت**: timestamps دقیق است، نیازی به detection نیست
4. **Buffer**: 1.5s قبل و بعد کافی است (3 بوق + واکنش)
5. **Late Start**: سیستم خودکار 3s buffer برای مسابقات با `late_start: true` می‌دهد
6. **Seoul Race 15**: حذف شده است (false start)
7. **Total**: 188 مسابقه (31 + 24 + 32 + 32 + 69)
8. **Status**: 4 مسابقه کامل (Seoul, Villars, Chamonix, Innsbruck) + Zilina در حال پردازش
9. **اصلاحات**: همه timestamps اصلاح شده‌اند
10. **Innsbruck**: Race 2 اصلاحات خاص (+20s start, +8s end) + 18 مسابقه late_start
11. **Zilina**: بزرگترین مسابقه (69 races) + European Youth Championships + 3 races حذف شد
12. **Temp Files**: فایل‌های `temp_*.mp4` را پاک کنید (با `Remove-Item`)

---

**موفق باشید! 🎯**

**مراحل پیشنهادی:**
1. هر 10-15 دقیقه یکبار `check_progress.ps1` را اجرا کنید
2. وقتی Zilina تمام شد، فایل‌های temp را پاک کنید
3. یک بار دیگر `check_progress.ps1` را اجرا کنید و مطمئن شوید 188/188 است
4. به من اطلاع دهید!

اگر مشکلی پیش آمد یا سوالی داشتید، به من اطلاع دهید.
