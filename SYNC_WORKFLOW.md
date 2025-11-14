# 🔄 راهنمای کامل Sync بین Gitea ↔ GitHub ↔ UI

**تاریخ**: 2025-11-14
**نسخه**: 1.0

---

## 📊 نمای کلی (Overview)

این پروژه در **سه محیط مختلف** توسعه می‌یابد:

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Gitea (Main)  │◄───────►│  Local Machine  │◄───────►│     GitHub      │
│   (Private)     │         │   (You)         │         │   (Public)      │
│   FULL Dataset  │         │                 │         │  Lightweight    │
│   706 MB        │         │                 │         │  ~50 MB         │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                      ▲
                                      │
                                      ▼
                            ┌─────────────────┐
                            │  claude.ai/code │
                            │     (UI)        │
                            │   Clone GitHub  │
                            └─────────────────┘
```

---

## 🎯 استراتژی Repository

### **Gitea** (Primary - Full Dataset):
- ✅ همه کد
- ✅ همه configs
- ✅ **188 race segments MP4** (705 MB)
- ✅ همه metadata
- 🔒 Private (فقط شما دسترسی دارید)

### **GitHub** (Public - Lightweight):
- ✅ همه کد
- ✅ همه configs
- ❌ **بدون race segments MP4** (gitignore شده)
- ✅ همه metadata
- 🌍 Public (برای اشتراک و UI)

### **UI claude.ai/code** (Development):
- Clone از GitHub
- بدون ویدئوهای بزرگ
- برای توسعه سریع

---

## 🔧 Setup اولیه (یکبار انجام شود)

### 1. بررسی Remotes موجود

```bash
git remote -v
```

**خروجی مورد انتظار**:
```
origin  https://gitea.airano.ir/dev/Speed-Climbing-Performance-Analysis.git (fetch)
origin  https://gitea.airano.ir/dev/Speed-Climbing-Performance-Analysis.git (push)
github  https://github.com/languageofearthcom-oss/Speed-Climbing-Performance-Analysis.git (fetch)
github  https://github.com/languageofearthcom-oss/Speed-Climbing-Performance-Analysis.git (push)
```

### 2. اگر GitHub remote نیست، اضافه کنید:

```bash
git remote add github https://github.com/languageofearthcom-oss/Speed-Climbing-Performance-Analysis.git
```

---

## 📥 دریافت تغییرات (Pull)

### از Gitea (Primary):

```bash
# Fetch
git fetch origin

# Merge به branch فعلی
git merge origin/main

# یا Pull (fetch + merge)
git pull origin main
```

### از GitHub:

```bash
# Fetch
git fetch github

# Merge
git merge github/main

# یا Pull
git pull github main
```

### از UI claude.ai/code:

```bash
# فرض: UI یک branch جدید ساخته (مثلاً feature/phase2-pose-extraction)
git fetch github feature/phase2-pose-extraction

# بررسی تغییرات
git log github/feature/phase2-pose-extraction --oneline -10

# Merge
git checkout main
git merge github/feature/phase2-pose-extraction
```

---

## 📤 ارسال تغییرات (Push)

### به Gitea (Primary):

```bash
# Push main branch
git push origin main

# Push یک branch خاص
git push origin feature/my-feature
```

⚠️ **نکته مهم**: برای فایل‌های بزرگ (>500MB)، ممکن است timeout بخورید.

**راه‌حل**:
```bash
# افزایش buffer size
git config http.postBuffer 524288000  # 500MB

# افزایش timeout
git config http.timeout 600  # 10 minutes

# سپس push
git push origin main
```

اگر باز هم timeout خورد:
```bash
# Push فقط کد (بدون data/)
git push origin main --no-verify

# یا از طریق dashboard Gitea فایل‌های بزرگ را manual upload کنید
```

### به GitHub (Public):

```bash
# Push main branch
git push github main

# Force push (فقط اگر نیاز بود)
git push github main --force-with-lease
```

---

## 🔄 Workflow روزانه

### 🌅 صبح (شروع کار):

```bash
# 1. Pull از Gitea (primary source)
git pull origin main

# 2. اگر تغییری بود، push به GitHub
git push github main
```

### 🌆 عصر (پایان کار):

```bash
# 1. Commit تغییرات
git add -A
git commit -m "feat: توضیحات تغییرات"

# 2. Push به Gitea
git push origin main

# 3. Push به GitHub
git push github main
```

---

## 🔀 Workflow با UI claude.ai/code

### UI یک feature جدید ساخته:

**1. در UI** (claude.ai/code):
```bash
# توسعه
git checkout -b feature/phase2-pose-extraction
# ... کد بنویسید ...
git add -A
git commit -m "feat: batch pose extraction"
git push origin feature/phase2-pose-extraction
```

**2. در محیط شما** (Local):
```bash
# Pull feature از GitHub
git fetch github feature/phase2-pose-extraction
git checkout -b feature/phase2-pose-extraction github/feature/phase2-pose-extraction

# بررسی و تست
pytest tests/ -v

# اگر OK بود، merge به main
git checkout main
git merge feature/phase2-pose-extraction

# Push به Gitea و GitHub
git push origin main
git push github main

# حذف branch موقت
git branch -d feature/phase2-pose-extraction
```

---

## ⚠️ مشکلات رایج و راه‌حل

### 1. **Conflict بین Gitea و GitHub**

```bash
# Fetch از هر دو
git fetch origin
git fetch github

# بررسی تفاوت‌ها
git log origin/main..github/main --oneline
git log github/main..origin/main --oneline

# تصمیم‌گیری:
# - اگر Gitea جلوتر است:
git push github main --force-with-lease

# - اگر GitHub جلوتر است:
git pull github main
git push origin main
```

### 2. **Gitea timeout برای فایل‌های بزرگ**

```bash
# گزینه 1: افزایش timeout
git config http.postBuffer 1048576000  # 1GB
git config http.timeout 1200  # 20 minutes
git push origin main

# گزینه 2: Push فایل‌های کوچک ابتدا
git add src/ configs/ *.md
git commit -m "code and docs"
git push origin main

# سپس data/
git add data/
git commit -m "add data"
git push origin main
```

### 3. **race_segments در GitHub نمی‌خواهیم**

```bash
# بررسی .gitignore
cat .gitignore | grep race_segments

# باید این خط‌ها uncomment باشند برای GitHub:
# data/race_segments/*.mp4
# data/race_segments/*.avi

# برای Gitea commented باشند:
# # data/race_segments/*.mp4    # Commented for Gitea
```

**حل**:
```bash
# برای GitHub: un-ignore کردن MP4ها
sed -i 's/# data\/race_segments\/\*.mp4/data\/race_segments\/\*.mp4/' .gitignore
git add .gitignore
git commit -m "chore: ignore race MP4 for GitHub"
git push github main --force
```

### 4. **UI نیاز به race segments دارد**

**راه‌حل 1**: Clone از Gitea (اگر دسترسی دارد)
```bash
# در UI
git clone https://gitea.airano.ir/dev/Speed-Climbing-Performance-Analysis.git
```

**راه‌حل 2**: Download manual از Gitea
- رفتن به Gitea dashboard
- Download کردن `data/race_segments/` به صورت ZIP
- آپلود به UI

**راه‌حل 3**: کار با sample clips فقط
- UI روی 5 sample clip کار می‌کند
- Pipeline می‌سازد
- بعداً در محیط اصلی روی 188 race اجرا می‌کنید

---

## 🔍 دستورات مفید

### بررسی وضعیت Sync:

```bash
# آخرین commit هر remote
git log origin/main --oneline -1
git log github/main --oneline -1

# مقایسه
git diff origin/main github/main --stat

# لیست commits که در یکی هست ولی در دیگری نیست
git log origin/main..github/main --oneline  # در GitHub ولی نه در Gitea
git log github/main..origin/main --oneline  # در Gitea ولی نه در GitHub
```

### Sync کامل:

```bash
# Pull از Gitea
git pull origin main

# Push به GitHub
git push github main

# بررسی یکسان بودن
git log origin/main --oneline -1
git log github/main --oneline -1
# باید commit hash یکسان باشد
```

---

## 📋 Checklist قبل از هر Push

- [ ] `git status` - بررسی فایل‌های commit نشده
- [ ] `git diff` - بررسی تغییرات
- [ ] `pytest tests/ -v` - اجرای تست‌ها (اگر لازم)
- [ ] `git log --oneline -3` - بررسی آخرین commits
- [ ] `git push origin main` - Push به Gitea
- [ ] `git push github main` - Push به GitHub
- [ ] بررسی sync: `git log origin/main` == `git log github/main`

---

## 🎓 مثال عملی: یک روز کاری کامل

### شروع روز:

```bash
# ورود به پروژه
cd "G:\My Drive\Projects\Speed Climbing Performance Analysis"

# Pull جدیدترین تغییرات از Gitea
git pull origin main

# Sync با GitHub
git push github main

# شروع کار
```

### توسعه:

```bash
# ایجاد branch جدید
git checkout -b feature/improve-detection

# کد بنویسید...
# فایل‌ها ویرایش کنید...

# تست
pytest tests/ -v

# Commit
git add -A
git commit -m "feat: improve dual-lane detection accuracy

- بهبود boundary detection algorithm
- افزودن confidence threshold
- تست‌های جدید"
```

### پایان روز:

```bash
# Merge به main
git checkout main
git merge feature/improve-detection

# Push به Gitea
git push origin main

# Push به GitHub
git push github main

# بررسی sync
git log origin/main --oneline -1
git log github/main --oneline -1

# حذف branch موقت
git branch -d feature/improve-detection
```

---

## 🚀 استفاده از اسکریپت‌های کمکی

ما دو اسکریپت کمکی ساخته‌ایم:

### `sync.bat` (Windows):
```bash
# Sync دوطرفه کامل
sync.bat

# Pull فقط
sync.bat pull

# Push فقط
sync.bat push
```

### `sync.sh` (Linux/Mac):
```bash
# Sync دوطرفه کامل
chmod +x sync.sh
./sync.sh

# Pull فقط
./sync.sh pull

# Push فقط
./sync.sh push
```

---

## 📞 اگر مشکلی پیش آمد

1. **بررسی remotes**: `git remote -v`
2. **بررسی branch**: `git branch -a`
3. **بررسی log**: `git log --oneline -10`
4. **بررسی diff**: `git diff origin/main github/main`
5. **Reset (آخرین راه‌حل)**:
   ```bash
   git fetch origin
   git reset --hard origin/main
   git push github main --force
   ```

---

**🎉 با این راهنما، شما می‌توانید بدون مشکل بین Gitea، GitHub و UI جابه‌جا شوید!**
