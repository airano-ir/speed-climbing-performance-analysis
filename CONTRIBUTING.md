# Contributing to Speed Climbing Performance Analysis
# راهنمای مشارکت در پروژه تحلیل عملکرد صعود سرعتی

[English](#english) | [فارسی](#فارسی)

---

## English

Thank you for your interest in contributing to Speed Climbing Performance Analysis! This document provides guidelines for contributing to the project.

### How to Contribute

#### Reporting Bugs

1. **Check existing issues** - Search the issue tracker to avoid duplicates
2. **Create a detailed report** including:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment (OS, Python version, etc.)
   - Screenshots or error logs if applicable

#### Suggesting Features

1. **Check existing feature requests** in issues
2. **Open a new issue** with:
   - Clear description of the feature
   - Use case and benefits
   - Possible implementation approach (optional)

#### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following our code style
4. **Write/update tests** if applicable
5. **Commit with clear messages**: Use conventional commits
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `refactor:` for code refactoring
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Create a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots if UI changes

### Code Style

- **Python**: Follow PEP 8
- **Type hints**: Use type annotations where possible
- **Documentation**: Add docstrings to functions and classes
- **Comments**: Write clear comments for complex logic

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/speed-climbing-performance-analysis.git
cd speed-climbing-performance-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
```

### Pull Request Process

1. Update documentation if needed
2. Ensure all tests pass
3. Request review from maintainers
4. Address feedback promptly
5. Once approved, maintainers will merge

### Questions?

- Open an issue for questions
- Email: gitea@airano.ir
- Website: https://airano.ir

---

## فارسی

از علاقه شما به مشارکت در پروژه تحلیل عملکرد صعود سرعتی متشکریم! این سند راهنمای مشارکت در پروژه را ارائه می‌دهد.

### نحوه مشارکت

#### گزارش باگ

1. **بررسی مشکلات موجود** - جستجو در issue tracker برای جلوگیری از تکرار
2. **ایجاد گزارش دقیق** شامل:
   - توضیح واضح باگ
   - مراحل بازتولید
   - رفتار مورد انتظار در مقابل رفتار واقعی
   - محیط (سیستم عامل، نسخه Python و غیره)
   - اسکرین‌شات یا لاگ خطا در صورت وجود

#### پیشنهاد ویژگی

1. **بررسی درخواست‌های ویژگی موجود** در issues
2. **باز کردن issue جدید** با:
   - توضیح واضح ویژگی
   - مورد استفاده و مزایا
   - رویکرد پیاده‌سازی احتمالی (اختیاری)

#### مشارکت کد

1. **Fork کردن مخزن**
2. **ایجاد branch ویژگی**: `git checkout -b feature/your-feature-name`
3. **اعمال تغییرات** طبق سبک کد ما
4. **نوشتن/به‌روزرسانی تست‌ها** در صورت نیاز
5. **Commit با پیام‌های واضح**: استفاده از conventional commits
   - `feat:` برای ویژگی‌های جدید
   - `fix:` برای رفع باگ
   - `docs:` برای مستندات
   - `refactor:` برای بازسازی کد
6. **Push به fork خود**: `git push origin feature/your-feature-name`
7. **ایجاد Pull Request** با:
   - توضیح واضح تغییرات
   - ارجاع به issues مرتبط
   - اسکرین‌شات در صورت تغییرات UI

### سبک کد

- **Python**: پیروی از PEP 8
- **Type hints**: استفاده از type annotation در صورت امکان
- **مستندات**: اضافه کردن docstring به توابع و کلاس‌ها
- **کامنت**: نوشتن کامنت‌های واضح برای منطق پیچیده

### راه‌اندازی توسعه

```bash
# Clone کردن fork
git clone https://github.com/YOUR-USERNAME/speed-climbing-performance-analysis.git
cd speed-climbing-performance-analysis

# ایجاد محیط مجازی
python -m venv venv
source venv/bin/activate  # Linux/Mac
# یا: venv\Scripts\activate  # Windows

# نصب وابستگی‌ها
pip install -r requirements.txt

# اجرای تست‌ها
pytest
```

### فرآیند Pull Request

1. به‌روزرسانی مستندات در صورت نیاز
2. اطمینان از pass شدن تمام تست‌ها
3. درخواست بررسی از نگهدارندگان
4. پاسخ سریع به بازخوردها
5. پس از تأیید، نگهدارندگان merge می‌کنند

### سوالات؟

- باز کردن issue برای سوالات
- ایمیل: gitea@airano.ir
- وبسایت: https://airano.ir

---

Made with ❤️ by Airano | ساخته شده با ❤️ توسط آیرانو
