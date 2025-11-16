# Quick Start Guide - Manual Review Interface
# راهنمای شروع سریع - رابط بررسی دستی

**Created**: 2025-11-16
**Version**: 1.0
**Languages**: English + فارسی

---

## 🚀 Quick Start (English)

### 1. Install Dependencies

```bash
pip install streamlit opencv-python pyyaml numpy
```

### 2. Test Components

```bash
python scripts/review_interface/test_components.py
```

You should see:
```
✓ ALL TESTS PASSED!
You can now run the interface
```

### 3. Run the Interface

```bash
cd scripts/review_interface
streamlit run app.py
```

The interface will open in your browser at `http://localhost:8501`

### 4. Select Language

In the sidebar, choose your language:
- 🇬🇧 English
- 🇮🇷 فارسی

### 5. Start Reviewing

1. **Filter races**: Use sidebar filters (Competition, Priority, Status)
2. **Select race**: Choose from dropdown
3. **Watch video**: Use frame navigation (±1, ±5, ±30 frames)
4. **Mark frames**: Click "Mark as START" and "Mark as FINISH" buttons
5. **Add reason**: Enter correction reason (required)
6. **Save**: Click "Save Correction"

---

## 🚀 شروع سریع (فارسی)

### 1. نصب وابستگی‌ها

```bash
pip install streamlit opencv-python pyyaml numpy
```

### 2. تست کامپوننت‌ها

```bash
python scripts/review_interface/test_components.py
```

باید این پیام را ببینید:
```
✓ ALL TESTS PASSED! / همه تست‌ها موفق!
می‌توانید رابط را اجرا کنید
```

### 3. اجرای رابط

```bash
cd scripts/review_interface
streamlit run app.py
```

رابط در مرورگر شما باز می‌شود: `http://localhost:8501`

### 4. انتخاب زبان

در نوار کناری، زبان خود را انتخاب کنید:
- 🇬🇧 English (انگلیسی)
- 🇮🇷 فارسی

### 5. شروع بررسی

1. **فیلتر مسابقات**: از فیلترهای نوار کناری استفاده کنید (مسابقه، اولویت، وضعیت)
2. **انتخاب مسابقه**: از منوی کشویی انتخاب کنید
3. **مشاهده ویدئو**: از ناوبری فریم استفاده کنید (±1، ±5، ±30 فریم)
4. **علامت‌گذاری فریم‌ها**: روی دکمه‌های "علامت‌گذاری به عنوان شروع" و "علامت‌گذاری به عنوان پایان" کلیک کنید
5. **افزودن دلیل**: دلیل اصلاح را وارد کنید (اجباری)
6. **ذخیره**: روی "ذخیره اصلاح" کلیک کنید

---

## 📊 Understanding the Interface / درک رابط کاربری

### Progress Statistics / آمار پیشرفت

**Location**: Sidebar top / محل: بالای نوار کناری

Shows:
- Total races to review / مجموع مسابقات برای بررسی
- Completed count / تعداد تکمیل شده
- Pending count / تعداد در انتظار
- Critical count / تعداد بحرانی
- Progress percentage / درصد پیشرفت

### Race Priorities / اولویت‌های مسابقه

**Priority 1 - Critical** (بحرانی):
- Negative duration / مدت زمان منفی
- Near-zero duration / مدت زمان نزدیک به صفر
- Must fix immediately / باید فوراً اصلاح شود

**Priority 2 - High** (بالا):
- Zilina 2025 systematic failure / شکست سیستماتیک Zilina 2025
- Duration issues / مشکلات مدت زمان
- Review soon / به زودی بررسی شود

**Priority 3 - Medium** (متوسط):
- Too short or too long / خیلی کوتاه یا بلند
- Review when possible / در صورت امکان بررسی شود

**Priority 4 - Low** (پایین):
- Minor issues / مشکلات جزئی
- Review if time permits / در صورت وجود زمان بررسی شود

### Video Player Controls / کنترل‌های پخش‌کننده ویدئو

**Navigation Buttons** / دکمه‌های ناوبری:
- ⏮️ -30: Go back 30 frames / 30 فریم به عقب
- ⬅️ -5: Go back 5 frames / 5 فریم به عقب
- ◀️ -1: Go back 1 frame / 1 فریم به عقب
- ▶️ +1: Go forward 1 frame / 1 فریم به جلو
- ➡️ +5: Go forward 5 frames / 5 فریم به جلو
- ⏭️ +30: Go forward 30 frames / 30 فریم به جلو

**Slider** / اسلایدر:
- Drag to quickly navigate / برای ناوبری سریع بکشید
- Click on bar to jump / روی نوار کلیک کنید تا جهش کند

### Validation Indicators / نشانگرهای اعتبارسنجی

**✅ Green** (سبز):
- Valid duration (4.5s - 15s) / مدت زمان معتبر
- All checks passed / همه بررسی‌ها موفق

**⚠️ Yellow** (زرد):
- Warning: Below/above thresholds / هشدار: زیر/بالای آستانه
- Review carefully / با دقت بررسی کنید

**❌ Red** (قرمز):
- Critical error: Must fix / خطای بحرانی: باید اصلاح شود
- Invalid data / داده نامعتبر

---

## 🎯 Common Scenarios / سناریوهای رایج

### Scenario 1: Athlete Fall Detected as Finish
### سناریو 1: سقوط ورزشکار به عنوان پایان

**Problem**: System detected fall as race finish
**مشکل**: سیستم سقوط را به عنوان پایان مسابقه تشخیص داد

**Solution**:
1. Find actual finish frame (hand reaches top button)
2. Mark new finish frame
3. Reason: "Climber fall detected as finish"

**راه‌حل**:
1. یافتن فریم پایان واقعی (دست به دکمه بالا می‌رسد)
2. علامت‌گذاری فریم پایان جدید
3. دلیل: "سقوط ورزشکار به عنوان پایان تشخیص داده شد"

### Scenario 2: Pre-race Warmup Included
### سناریو 2: گرم‌کردن قبل از مسابقه شامل شده

**Problem**: Warmup movements detected as race start
**مشکل**: حرکات گرم‌کردن به عنوان شروع مسابقه تشخیص داده شد

**Solution**:
1. Skip warmup footage
2. Find actual start (starting beep or first movement)
3. Mark new start frame
4. Reason: "Pre-race warmup included"

**راه‌حل**:
1. رد کردن فیلم گرم‌کردن
2. یافتن شروع واقعی (بوق شروع یا اولین حرکت)
3. علامت‌گذاری فریم شروع جدید
4. دلیل: "گرم کردن قبل از مسابقه شامل شد"

### Scenario 3: Negative Duration
### سناریو 3: مدت زمان منفی

**Problem**: Finish frame < Start frame (impossible!)
**مشکل**: فریم پایان < فریم شروع (غیرممکن!)

**Solution**:
1. Carefully review entire video
2. Find correct start and finish
3. Ensure finish > start
4. Reason: "Detection error - frame order was incorrect"

**راه‌حل**:
1. بررسی دقیق کل ویدئو
2. یافتن شروع و پایان صحیح
3. اطمینان از پایان > شروع
4. دلیل: "خطای تشخیص - ترتیب فریم‌ها اشتباه بود"

---

## 💡 Tips & Best Practices / نکات و بهترین شیوه‌ها

### English

**Finding Start Frame**:
- ✅ Listen for starting beep (if audio available)
- ✅ Look for first upward movement
- ✅ Watch for climber's feet leaving ground
- ❌ Don't include warmup movements
- ❌ Don't include pre-race preparation

**Finding Finish Frame**:
- ✅ Look for hand touching top button
- ✅ Watch for finish light activation
- ✅ Identify exact moment of button contact
- ❌ Don't use celebration moment
- ❌ Don't use when climber stops climbing

**Efficient Workflow**:
1. Start with Critical priority races
2. Group similar issues together
3. Use ±30 for rough navigation, ±1 for precise
4. Add detailed notes for complex cases
5. Take breaks every 10-15 races

### فارسی

**یافتن فریم شروع**:
- ✅ گوش دادن به بوق شروع (در صورت وجود صدا)
- ✅ جستجوی اولین حرکت به سمت بالا
- ✅ مشاهده لحظه‌ای که پای ورزشکار از زمین بلند می‌شود
- ❌ شامل نکردن حرکات گرم‌کردن
- ❌ شامل نکردن آماده‌سازی قبل از مسابقه

**یافتن فریم پایان**:
- ✅ جستجوی لحظه لمس دکمه بالا توسط دست
- ✅ مشاهده روشن شدن چراغ پایان
- ✅ شناسایی لحظه دقیق تماس با دکمه
- ❌ استفاده نکردن از لحظه جشن
- ❌ استفاده نکردن از زمانی که ورزشکار صعود را متوقف می‌کند

**گردش کار کارآمد**:
1. شروع با مسابقات اولویت بحرانی
2. گروه‌بندی مسائل مشابه
3. استفاده از ±30 برای ناوبری تقریبی، ±1 برای دقت
4. افزودن یادداشت‌های تفصیلی برای موارد پیچیده
5. استراحت هر 10-15 مسابقه

---

## 🔍 Validation Reference / مرجع اعتبارسنجی

### Duration Ranges / محدوده‌های مدت زمان

**World Records** (رکوردهای جهانی):
- Men / مردان: 5.00s (Reza Alipour, 2023)
- Women / زنان: 6.53s (Aleksandra Miroslaw, 2024)

**Acceptable Ranges** (محدوده‌های قابل قبول):
- Minimum / حداقل: 4.5s (below world record / زیر رکورد جهانی)
- Maximum / حداکثر: 15.0s (slow climbers + falls / ورزشکاران آهسته + سقوط)

**Critical Errors** (خطاهای بحرانی):
- Negative / منفی: < 0s (impossible! / غیرممکن!)
- Too short / خیلی کوتاه: < 3s (detection error / خطای تشخیص)
- Too long / خیلی بلند: > 20s (includes non-race / شامل غیرمسابقه)

---

## 📁 File Locations / محل فایل‌ها

### Configuration / پیکربندی
```
configs/manual_review_config.yaml
```

### Progress Tracker / ردیاب پیشرفت
```
data/manual_review/progress_tracker.csv
```

### Race Videos / ویدئوهای مسابقه
```
data/race_segments/
├── chamonix_2024/
├── innsbruck_2024/
├── villars_2024/
└── zilina_2025/
```

### Metadata Files / فایل‌های متادیتا
```
data/race_segments/[competition]/[race_id]_metadata.json
```

### Backups / پشتیبان‌ها
```
data/race_segments/[competition]/[race_id]_metadata.json.bak
```

---

## 🆘 Getting Help / دریافت کمک

### Documentation / مستندات

**English**:
- Build guide: `docs/PROMPT_FOR_UI_BUILD_REVIEW_INTERFACE.md`
- Project context: `MASTER_CONTEXT.md`
- Component README: `scripts/review_interface/README.md`

**فارسی**:
- راهنمای کاربری: `docs/MANUAL_REVIEW_INTERFACE_GUIDE_FA.md`
- این فایل: `docs/QUICK_START_REVIEW_INTERFACE.md`

### Testing / تست

```bash
# Test all components
python scripts/review_interface/test_components.py

# If tests fail, check error messages
# اگر تست‌ها شکست خوردند، پیام‌های خطا را بررسی کنید
```

### Troubleshooting / عیب‌یابی

**Interface won't start** / رابط شروع نمی‌شود:
```bash
pip install --upgrade streamlit
```

**Video won't load** / ویدئو بارگذاری نمی‌شود:
```bash
pip install --upgrade opencv-python
```

**Config errors** / خطاهای پیکربندی:
```bash
# Check file exists
ls configs/manual_review_config.yaml
```

---

## 🎓 Training Videos / ویدئوهای آموزشی

### Recommended Review Order / ترتیب بررسی پیشنهادی

1. **First 3 races**: Already corrected (Race001, Race010, Race023)
   - Review these to see examples
   - اول 3 مسابقه: از قبل اصلاح شده
   - برای دیدن نمونه‌ها، این‌ها را بررسی کنید

2. **Next: Critical Priority** (4 races)
   - Negative or near-zero duration
   - بعدی: اولویت بحرانی (4 مسابقه)
   - مدت زمان منفی یا نزدیک به صفر

3. **Then: High Priority** (37 races)
   - Zilina 2025 systematic issues
   - سپس: اولویت بالا (37 مسابقه)
   - مسائل سیستماتیک Zilina 2025

4. **Finally: Medium Priority** (33 races)
   - Various duration issues
   - در نهایت: اولویت متوسط (33 مسابقه)
   - مسائل مختلف مدت زمان

---

## ✅ Success Checklist / چک‌لیست موفقیت

Before starting:
- [ ] Dependencies installed / وابستگی‌ها نصب شد
- [ ] Tests passed / تست‌ها موفق شد
- [ ] Interface runs / رابط اجرا می‌شود
- [ ] Language selected / زبان انتخاب شد

During review:
- [ ] Understand the issue / درک مشکل
- [ ] Find correct start frame / یافتن فریم شروع صحیح
- [ ] Find correct finish frame / یافتن فریم پایان صحیح
- [ ] Validate duration / اعتبارسنجی مدت زمان
- [ ] Add correction reason / افزودن دلیل اصلاح
- [ ] Save with backup / ذخیره با پشتیبان

After each race:
- [ ] Check progress updated / بررسی به‌روزرسانی پیشرفت
- [ ] Backup created / پشتیبان ایجاد شد
- [ ] Validation passed / اعتبارسنجی موفق شد

---

## 🎉 You're Ready! / آماده‌اید!

You now have everything you need to start reviewing races!

اکنون همه چیزهایی که برای شروع بررسی مسابقات نیاز دارید را دارید!

**Good luck with your reviews!**
**موفق باشید در بررسی‌های خود!**

---

**Version**: 1.0
**Last Updated**: 2025-11-16
**Languages**: English + فارسی
