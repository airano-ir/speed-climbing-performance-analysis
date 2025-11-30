# Security Policy
# سیاست امنیتی

[English](#english) | [فارسی](#فارسی)

---

## English

### Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

#### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email us at: **gitea@airano.ir**
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

#### What to Expect

- **Response time**: We will acknowledge your report within 48 hours
- **Updates**: We will keep you informed of the progress
- **Credit**: We will credit you in the fix (unless you prefer anonymity)
- **Disclosure**: We aim to fix and disclose within 90 days

### Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

### Security Best Practices

When using this software:

1. **Keep dependencies updated** - Run `pip install --upgrade -r requirements.txt` regularly
2. **Use environment variables** - Never hardcode sensitive data
3. **Secure your deployment** - Use HTTPS, proper authentication
4. **Monitor logs** - Watch for unusual activity

### Known Security Considerations

- **Video data**: Videos may contain personal information (faces, locations)
- **Model outputs**: AI analysis results should not be shared without consent
- **API access**: If exposing as an API, implement proper authentication

---

## فارسی

### گزارش آسیب‌پذیری

ما امنیت را جدی می‌گیریم. اگر آسیب‌پذیری امنیتی کشف کردید، لطفاً آن را به صورت مسئولانه گزارش دهید.

#### نحوه گزارش

1. برای آسیب‌پذیری‌های امنیتی issue عمومی در GitHub **ایجاد نکنید**
2. به ما ایمیل بزنید: **gitea@airano.ir**
3. اطلاعات زیر را شامل کنید:
   - توضیح آسیب‌پذیری
   - مراحل بازتولید
   - تأثیر احتمالی
   - پیشنهاد رفع (در صورت وجود)

#### انتظارات

- **زمان پاسخ**: گزارش شما را ظرف 48 ساعت تأیید می‌کنیم
- **به‌روزرسانی**: شما را از پیشرفت مطلع نگه می‌داریم
- **اعتبار**: در رفع مشکل به شما اعتبار می‌دهیم (مگر اینکه ناشناس بمانید)
- **افشا**: هدف ما رفع و افشا در 90 روز است

### نسخه‌های پشتیبانی شده

| نسخه    | پشتیبانی           |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

### بهترین شیوه‌های امنیتی

هنگام استفاده از این نرم‌افزار:

1. **به‌روز نگه دارید** - مرتباً `pip install --upgrade -r requirements.txt` اجرا کنید
2. **از متغیرهای محیطی استفاده کنید** - هرگز داده‌های حساس را hardcode نکنید
3. **deployment خود را امن کنید** - از HTTPS و احراز هویت مناسب استفاده کنید
4. **لاگ‌ها را نظارت کنید** - فعالیت غیرعادی را زیر نظر داشته باشید

### ملاحظات امنیتی شناخته شده

- **داده‌های ویدیو**: ویدیوها ممکن است شامل اطلاعات شخصی باشند (چهره‌ها، مکان‌ها)
- **خروجی مدل**: نتایج تحلیل AI نباید بدون رضایت به اشتراک گذاشته شوند
- **دسترسی API**: اگر به عنوان API expose می‌کنید، احراز هویت مناسب پیاده‌سازی کنید

---

Made with ❤️ by Airano | ساخته شده با ❤️ توسط آیرانو
