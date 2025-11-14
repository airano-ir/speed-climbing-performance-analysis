# Sample Pose Files for GitHub

این پوشه شامل نمونه‌هایی از pose JSON files است که برای demo و تست در GitHub قرار داده می‌شوند.

## انتخاب Samples

بعد از اتمام batch pose extraction، این فایل‌ها را **manually** کپی کنید:

### معیار انتخاب:
1. **Top 5 fastest athletes** (سریع‌ترین زمان‌ها)
2. **Different competitions** (تنوع مسابقات)
3. **Different camera angles** (تنوع زاویه دوربین)
4. **Good detection rate** (>95% detection)
5. **Reasonable file size** (<5 MB)

### دستور انتخاب خودکار:

```bash
# بعد از اتمام batch processing، این اسکریپت را اجرا کنید:
python scripts/select_sample_poses.py \
  --input data/processed/poses \
  --output data/processed/poses/samples \
  --count 10 \
  --criteria fastest,diverse
```

این اسکریپت:
1. همه pose files را می‌خواند
2. Top 10 را بر اساس velocity انتخاب می‌کند
3. تنوع در competitions را تضمین می‌کند
4. فایل‌ها را کپی می‌کند (نه move)

### فایل‌های پیشنهادی (بعد از analysis):

```
samples/
├── sample_001_Miroslaw_Aleksandra_POL_6.84s.json    # World record
├── sample_002_Kałucka_Aleksandra_POL_6.96s.json
├── sample_003_Deng_Lijuan_CHN_7.02s.json
├── sample_004_Cao_Longrui_CHN_5.00s.json           # Men's record
├── sample_005_Khaibullin_Veddrakh_KAZ_5.12s.json
├── ...
└── sample_010_*.json
```

### حجم تخمینی:
- 10 files × 5 MB average = **~50 MB**
- مناسب برای GitHub (limit: 100 MB per file, 1 GB total)

---

**نکته**: این فایل‌ها فقط برای **demo و test** هستند. برای analysis کامل، از Google Drive استفاده کنید.
