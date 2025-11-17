# Reliable Data Pipeline - Complete Guide
# راهنمای کامل پایپلاین داده‌های قابل اطمینان

**Version**: 1.0
**Date**: 2025-11-17
**Status**: ✅ Implementation Complete

---

## 📋 Overview / خلاصه

This pipeline processes **114 reliable races** (from 188 total) through a complete 7-phase workflow:
- Phase 1: Race List Generation ✅
- Phase 2: Pose Extraction (3-4 hours)
- Phase 3: Calibration (2-3 hours)
- Phase 4: Metrics Calculation (3-4 hours)
- Phase 5: Data Aggregation
- Phase 6: Validation
- Phase 7: Dashboard

**Total Time**: 12-16 hours
**Output**: 342+ files (114 races × 3 file types + aggregations)

---

## 🎯 Quick Start / شروع سریع

### Option 1: Run Everything (Recommended)

```bash
# Run complete pipeline (all 7 phases)
python scripts/run_reliable_pipeline.py

# Resume from checkpoint if interrupted
python scripts/run_reliable_pipeline.py --resume

# Dry run (see what will be executed)
python scripts/run_reliable_pipeline.py --dry-run
```

### Option 2: Run Individual Phases

```bash
# Phase 1: Generate race list (5 minutes)
python scripts/generate_reliable_races_list.py

# Phase 2: Extract poses (3-4 hours)
python scripts/batch_pose_extraction_reliable.py

# Phase 3: Calibrate cameras (2-3 hours)
python scripts/batch_calibration_reliable.py

# Phase 4: Calculate metrics (3-4 hours)
python scripts/batch_metrics_reliable.py

# Phase 5: Aggregate data (10 minutes)
python scripts/aggregate_reliable_data.py

# Phase 6: Validate pipeline (5 minutes)
python scripts/validate_pipeline_reliable.py

# Phase 7: Generate dashboard (5 minutes)
python scripts/generate_dashboard_reliable.py
```

### Option 3: Start from Specific Phase

```bash
# Start from Phase 3 (if Phases 1-2 already complete)
python scripts/run_reliable_pipeline.py --phase 3
```

---

## 📦 Prerequisites / پیش‌نیازها

### Required Data
- ✅ 188 race segments in `data/race_segments/`
- ✅ Race metadata files (`*_metadata.json`)
- ✅ Progress tracker: `data/manual_review/progress_tracker.csv`

### Required Python Packages

```bash
# Core dependencies
pip install opencv-python mediapipe numpy pandas

# Optional (for ML export)
pip install scikit-learn

# Optional (for enhanced visualizations)
pip install plotly
```

**Check dependencies:**
```bash
python -c "import cv2, mediapipe, numpy, pandas; print('✅ All dependencies installed')"
```

---

## 🔍 Phase Details / جزئیات هر فاز

### Phase 1: Generate Reliable Races List ✅

**Purpose**: Identify 114 reliable races (exclude 74 suspicious)

**Input**:
- `data/race_segments/*/` (all race videos)
- `data/manual_review/progress_tracker.csv` (suspicious races)

**Output**:
- `data/processed/reliable_races_list.json`

**Validation**:
```json
{
  "total_races": 188,
  "suspicious_races": 74,
  "reliable_races": 114,
  "by_competition": {
    "chamonix_2024": 29,
    "innsbruck_2024": 29,
    "seoul_2024": 29,
    "villars_2024": 19,
    "zilina_2025": 8
  }
}
```

**Time**: ~5 minutes

---

### Phase 2: Batch Pose Extraction

**Purpose**: Extract BlazePose keypoints for all 114 races

**Method**: MediaPipe Pose (model_complexity=2, highest quality)

**Output** (per race):
```json
{
  "race_id": "Speed_finals_Chamonix_2024_race001",
  "competition": "chamonix_2024",
  "frames": [
    {
      "frame_number": 45,
      "timestamp": 1.5,
      "landmarks": [/* 33 keypoints */]
    }
  ],
  "total_frames": 180,
  "extracted_frames": 172,
  "success_rate": 95.6
}
```

**Expected Files**: 114 pose files
**Path**: `data/processed/poses/<race_id>_pose.json`
**Report**: `data/processed/pose_extraction_report.json`

**Quality Metrics**:
- Success rate > 95% per race
- Missing frames < 5%
- 33 landmarks per frame

**Time**: 3-4 hours (depends on CPU/GPU)

**Monitoring**:
```bash
# Check progress
ls data/processed/poses/*.json | wc -l

# Expected: 114 files at completion
```

---

### Phase 3: Batch Calibration

**Purpose**: Generate camera calibration for accurate measurements

**Method**: IFSC 20-hold standard (15m wall)

**Output** (per race):
```json
{
  "race_id": "Speed_finals_Chamonix_2024_race001",
  "calibration_method": "periodic_holds",
  "wall_height_m": 15.0,
  "num_holds": 20,
  "pixels_per_meter": 123.45,
  "rmse_cm": 0.8,
  "transformation_matrix": [/* 3x3 matrix */]
}
```

**Expected Files**: 114 calibration files
**Path**: `data/processed/calibration/<race_id>_calibration.json`
**Report**: `data/processed/calibration_report.json`

**Quality Target**:
- RMSE < 1cm for 95%+ races
- Average RMSE: ~0.5cm

**Time**: 2-3 hours

**Monitoring**:
```bash
# Check average RMSE
python -c "
import json
with open('data/processed/calibration_report.json') as f:
    report = json.load(f)
    print(f'Average RMSE: {report[\"quality_metrics\"][\"average_rmse_cm\"]:.2f} cm')
"
```

---

### Phase 4: Batch Metrics Calculation

**Purpose**: Calculate performance metrics with calibration

**Features**:
- Frame filtering (race boundaries only)
- Calibrated measurements (meters, not pixels)
- Quality validation

**Output** (per race):
```json
{
  "race_id": "Speed_finals_Chamonix_2024_race001",
  "is_calibrated": true,
  "units": "meters",
  "average_velocity_ms": 2.34,
  "max_velocity_ms": 3.12,
  "total_time_s": 6.42,
  "path_length_m": 15.02,
  "acceleration_ms2": 0.78,
  "smoothness_score": 0.92,
  "efficiency": 0.998
}
```

**Expected Files**: 114 metrics files
**Path**: `data/processed/metrics/<race_id>_metrics.json`
**Report**: `data/processed/metrics_calculation_report.json`

**Validation Ranges**:
- Velocity: 1.0 - 3.0 m/s (typical range)
- Duration: 4.5 - 15.0 s (valid range)
- All races calibrated: 100%

**Time**: 3-4 hours

---

### Phase 5: Data Aggregation & ML Export

**Purpose**: Aggregate metrics and export ML-ready datasets

**Outputs**:

1. **Aggregated CSV**: `data/processed/aggregated_metrics_reliable.csv`
   - All metrics for 114 races
   - 20+ features per race

2. **Leaderboards**:
   - `leaderboard_overall.csv` (top 20 fastest)
   - `leaderboard_<competition>.csv` (5 files, top 10 per competition)

3. **ML Datasets**:
   - `ml_X_train.npy` (80% of data, features)
   - `ml_X_test.npy` (20% of data, features)
   - `ml_y_train.npy` (target: average velocity)
   - `ml_y_test.npy`
   - `ml_feature_names.json` (feature metadata)

**Feature List**:
```
- average_velocity_ms
- max_velocity_ms
- acceleration_ms2
- smoothness_score
- total_time_s
- path_length_m
- avg_vertical_velocity_ms
- efficiency
```

**Time**: ~10 minutes

---

### Phase 6: Pipeline Validation

**Purpose**: Comprehensive quality assurance

**Checks**:

1. **File Completeness**
   - Expected: 342+ files (114 × 3 + reports)
   - Missing files report

2. **Data Quality**
   - Velocity range validation
   - Duration range validation
   - Calibration verification
   - Negative values check

3. **Performance Benchmarks**
   - Average velocity vs expected
   - Fastest time vs world records
   - Distribution statistics

**Output**: `data/processed/pipeline_validation_report.json`

**Example Report**:
```json
{
  "file_completeness": {
    "pass_rate": 100.0,
    "total_missing": 0
  },
  "data_quality": {
    "pass_rate": 98.2,
    "races_with_issues": 2
  },
  "performance_benchmarks": {
    "avg_velocity_ms": 2.15,
    "avg_duration_s": 7.2,
    "fastest_in_dataset_s": 5.4,
    "world_record_comparison": {
      "men_wr_s": 5.00,
      "women_wr_s": 6.53
    }
  }
}
```

**Time**: ~5 minutes

---

### Phase 7: Interactive Dashboard

**Purpose**: Visual analytics and exploration

**Features**:
- Overview statistics (cards)
- Top 10 leaderboard (table)
- Velocity distribution (histogram)
- Competition comparison (box plots)
- Quality metrics summary

**Output**: `data/processed/dashboard_reliable_races.html`

**Usage**:
```bash
# Open in browser
xdg-open data/processed/dashboard_reliable_races.html  # Linux
open data/processed/dashboard_reliable_races.html      # macOS
start data/processed/dashboard_reliable_races.html     # Windows
```

**Time**: ~5 minutes

---

## 📊 Expected Outputs Summary

### Total Files Generated: 342+

```
data/processed/
├── reliable_races_list.json                    # Phase 1
├── poses/
│   └── <race_id>_pose.json                    # 114 files (Phase 2)
├── calibration/
│   └── <race_id>_calibration.json             # 114 files (Phase 3)
├── metrics/
│   └── <race_id>_metrics.json                 # 114 files (Phase 4)
├── aggregated_metrics_reliable.csv             # Phase 5
├── leaderboard_overall.csv                     # Phase 5
├── leaderboard_<competition>.csv               # 5 files (Phase 5)
├── ml_X_train.npy                              # Phase 5
├── ml_X_test.npy                               # Phase 5
├── ml_y_train.npy                              # Phase 5
├── ml_y_test.npy                               # Phase 5
├── ml_feature_names.json                       # Phase 5
├── pipeline_validation_report.json             # Phase 6
├── dashboard_reliable_races.html               # Phase 7
└── [reports]
    ├── pose_extraction_report.json
    ├── calibration_report.json
    ├── metrics_calculation_report.json
    └── pipeline_execution_log.json
```

---

## 🔧 Troubleshooting / عیب‌یابی

### Issue: Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'mediapipe'`

**Solution**:
```bash
pip install mediapipe opencv-python numpy pandas
```

---

### Issue: Out of Memory (Pose Extraction)

**Error**: `MemoryError` or system slowdown

**Solutions**:
1. **Process in batches**: Manually run subsets
2. **Close other applications**: Free up RAM
3. **Lower model complexity**: Edit script, change `model_complexity=2` to `1`

---

### Issue: Calibration Failure

**Error**: `Calibration failed for race X`

**Possible Causes**:
- Video corruption
- Insufficient holds visible
- Camera too shaky

**Solution**: Check failed races in report, may need manual review

---

### Issue: Interrupted Execution

**Error**: User pressed Ctrl+C or system reboot

**Solution**:
```bash
# Resume from last completed phase
python scripts/run_reliable_pipeline.py --resume
```

---

### Issue: Invalid Metrics

**Error**: Velocity or duration out of range

**Check**:
```bash
# View validation report
cat data/processed/pipeline_validation_report.json | grep "races_with_issues"
```

**Action**: Review specific races in quality_issues list

---

## 📈 Performance Benchmarks

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB free

**Recommended**:
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA (CUDA support for MediaPipe)
- Storage: 50 GB free (SSD preferred)

### Expected Processing Times

| Phase | Time (CPU) | Time (GPU) | Bottleneck |
|-------|------------|------------|------------|
| 1. Race List | 5 min | 5 min | I/O |
| 2. Pose Extraction | 3-4 hours | 1-2 hours | CPU/GPU |
| 3. Calibration | 2-3 hours | 2-3 hours | Algorithm |
| 4. Metrics | 3-4 hours | 2-3 hours | CPU |
| 5. Aggregation | 10 min | 10 min | I/O |
| 6. Validation | 5 min | 5 min | I/O |
| 7. Dashboard | 5 min | 5 min | I/O |
| **Total** | **12-16 hours** | **8-12 hours** | - |

---

## 🎓 Next Steps

### After Pipeline Completion

1. **Explore Dashboard**
   ```bash
   open data/processed/dashboard_reliable_races.html
   ```

2. **Analyze Leaderboards**
   ```bash
   head data/processed/leaderboard_overall.csv
   ```

3. **Train ML Models**
   ```python
   import numpy as np
   X_train = np.load('data/processed/ml_X_train.npy')
   y_train = np.load('data/processed/ml_y_train.npy')
   # Your ML code here
   ```

4. **Manual Review 74 Suspicious Races**
   - Use Phase 1.5 interface: `streamlit run src/manual_review/app.py`
   - Correct detection errors
   - Re-run pipeline with full 188 races

---

## 📚 Related Documentation

- [MASTER_CONTEXT.md](../MASTER_CONTEXT.md) - Full project context
- [PROMPT_FOR_UI_RELIABLE_DATA_PIPELINE.md](PROMPT_FOR_UI_RELIABLE_DATA_PIPELINE.md) - Original specification
- [PHASE_1.5.1_IMPLEMENTATION_PLAN.md](PHASE_1.5.1_IMPLEMENTATION_PLAN.md) - Manual review interface

---

## 🤝 Contributing

This pipeline is part of the Speed Climbing Performance Analysis project. For issues or improvements:

1. Check existing documentation
2. Review validation reports for data quality issues
3. Submit detailed bug reports with logs

---

## 📝 Changelog

### Version 1.0 (2025-11-17)
- ✅ Initial implementation of 7-phase pipeline
- ✅ All scripts created and tested
- ✅ Master runner with checkpoint support
- ✅ Comprehensive documentation

---

**Ready to Process!** 🚀

Run the pipeline and generate high-quality metrics for 114 reliable races.

```bash
python scripts/run_reliable_pipeline.py
```
