# Phase 1.5.1 Features - Quick Start Guide
# راهنمای سریع ویژگی‌های Phase 1.5.1

**Date**: 2025-11-16
**Version**: 1.0

---

## 📚 What's New in Phase 1.5.1

Phase 1.5.1 transforms the Manual Review Interface into a **complete video project management system** with 5 major enhancements:

1. **Video Library Manager** - View and manage all 188+ races
2. **Video Extraction** - Add new races manually with FFmpeg
3. **Enhanced Video Player** - Better frame navigation with synchronization
4. **Bulk Operations** - Batch export, validation, and reporting
5. **Multi-Phase Support** - Plugin architecture for Phase 1-4

---

## 🚀 Getting Started

### 1. Launch the Application

```bash
cd /path/to/Speed-Climbing-Performance-Analysis
streamlit run scripts/review_interface/app.py
```

The interface will open in your browser at `http://localhost:8501`

### 2. Navigation

Use the **page selector** at the top to switch between pages:

- **🏁 Race Review** - Review and correct suspicious races
- **📚 Video Library** - Browse all race videos
- **➕ Add Video** - Extract new races from source videos
- **⚡ Bulk Operations** - Batch processing operations

---

## 📚 Feature 1: Video Library Manager

**Purpose**: Centralized view of all race videos across all competitions

### How to Use:

1. Navigate to **📚 Video Library** page
2. Use filters to narrow down races:
   - **Competition**: Select specific competition or "All"
   - **Status**: Filter by reviewed, suspicious, pending, or failed
   - **Search**: Search race ID, athlete names, or notes

3. View statistics:
   - Total races, reviewed, suspicious, pending

4. **Quick Actions**:
   - Select a race and click **👁️ View in Player** to open in Race Review page
   - Click **📥 Export** to download filtered videos

### Export Options:

- **JSON**: Full metadata for each race
- **CSV**: Spreadsheet format for Excel/Google Sheets
- **YAML**: Human-readable configuration format

### Example Use Cases:

```
✅ Find all suspicious races in Zilina 2025 competition
✅ Export all reviewed races to CSV for analysis
✅ Search for specific athlete's races
✅ Get overview of dataset status
```

---

## ➕ Feature 2: Video Extraction

**Purpose**: Add new race videos with manual timestamp entry

### Prerequisites:

- **FFmpeg** must be installed (check is automatic)
- Source video file accessible
- Competition configured in `configs/manual_review_config.yaml`

### Step-by-Step Process:

#### Step 1: Competition Selection
- Select target competition from dropdown
- Competition settings (FPS, paths) are loaded automatically

#### Step 2: Source Video
- Enter full path to source video file
- Video info displayed automatically (duration, FPS, resolution)
- Suggested race ID shown (auto-incremented)

#### Step 3: Race Information
- **Race ID**: Unique identifier (use suggested or customize)
- **Round Name**: Optional (e.g., "1/8 final - Men")

#### Step 4: Manual Timestamps
- **Start Time**: Enter in MM:SS or HH:MM:SS format (e.g., `01:30` or `00:01:30`)
- **End Time**: Enter finish time
- **Buffer Before**: Seconds to include before start (default: 3.0s)
- **Buffer After**: Seconds to include after finish (default: 1.5s)

**Timestamp Validation**:
- ✅ Valid format check
- ✅ End must be after start
- ✅ Real-time duration calculation

#### Step 5: Athlete Information
- **Left Lane**: Name, Country, Bib Color
- **Right Lane**: Name, Country, Bib Color

#### Step 6: Extract
- Click **🎬 Extract Race Segment**
- FFmpeg processes video (progress shown)
- Success: Video + metadata saved to competition directory

### Example:

```
Competition: Seoul 2024
Source Video: /path/to/Seoul_2024_full_broadcast.mp4
Race ID: Speed_finals_Seoul_2024_race032
Start Time: 05:23
End Time: 05:29
Left Athlete: John Doe (USA, red bib)
Right Athlete: Jane Smith (FRA, blue bib)

→ Output: data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race032.mp4
→ Metadata: data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race032_metadata.json
```

---

## 🎯 Feature 3: Enhanced Video Player

**Improvements**: Full synchronization between all controls

### New Features:

#### 1. **Synchronized Controls**
- **Slider**: Drag to navigate
- **Frame Number Input**: Type exact frame number
- **Navigation Buttons**: -30, -1, +1, +30, Start, End
- **All controls update each other instantly**

#### 2. **Progress Bar**
- Visual indication of video position
- Shows percentage (e.g., "45.3% through video")

#### 3. **Bookmark System**
- Click **➕ Add Current Frame** to bookmark
- Save important frames for reference
- Jump to bookmarked frames quickly
- Visual indicator on bookmarked frames

#### 4. **Custom Jump**
- Enter any frame offset (positive or negative)
- Click **🎯** to jump by that amount

### Keyboard Tips:

While not yet fully automated, you can use standard Streamlit interactions:
- **Tab**: Move between inputs
- **Enter**: Confirm number input
- **Arrow keys**: Fine-tune slider

---

## ⚡ Feature 4: Bulk Operations

**Purpose**: Batch processing for multiple races

### Operation Types:

#### 1. **Export**

Export selected races to various formats:

**JSON** (Full data):
- Complete metadata for each race
- Includes all corrections and notes
- Best for: Data archiving, backup

**CSV** (Spreadsheet):
- Tabular format with key fields
- Opens in Excel, Google Sheets
- Best for: Quick analysis, sharing with non-technical users

**YAML** (Config-friendly):
- Human-readable configuration format
- Preserves structure and comments
- Best for: Configuration management

**NPZ** (ML-ready):
- NumPy compressed arrays
- Features + labels for machine learning
- Standardized feature vectors
- Best for: Training ML models (Phase 4)

**Options**:
- ☑️ Include Performance Metrics (if available)

**Example NPZ structure**:
```python
import numpy as np
data = np.load('export.npz', allow_pickle=True)

features = data['features']  # Shape: (N, 7) - velocity, acceleration, etc.
labels = data['labels']      # Shape: (N,) - race durations
race_ids = data['race_ids']  # List of race IDs
feature_names = data['feature_names']  # Feature descriptions
```

#### 2. **Validate**

Batch validation of selected races:

- Checks duration validity
- Checks frame boundaries
- Detects common issues
- **Output**: Summary with ✅ valid / ❌ invalid counts
- **Details**: Expandable list with specific issues per race

**Use Cases**:
- Verify corrections before export
- Find remaining problematic races
- Quality assurance

#### 3. **Summary Report**

Generate statistical summary:

**Includes**:
- Total races count
- Breakdown by competition
- Breakdown by status (reviewed, suspicious, etc.)
- Duration statistics (min, max, mean, median)

**Output**: JSON report with visualizable data

---

## 🔌 Feature 5: Multi-Phase Support

**Purpose**: Extensible plugin architecture for Phase 1-4

### Plugin System:

The interface now supports **plugins** that can be activated per phase:

#### Phase 1 (Active):
- `video_extraction` - Manual video extraction ✅
- `manual_review` - Race correction interface ✅
- `bulk_operations` - Batch processing ✅

#### Phase 2 (Available):
- `pose_extractor` - Extract poses from videos
- `pose_visualizer` - Visualize skeleton overlays

#### Phase 3 (Available):
- `metrics_calculator` - Calculate performance metrics
- `metrics_visualizer` - Comparative analysis charts

#### Phase 4 (Future):
- `ml_predictor` - CNN-Transformer predictions (demo available)
- `real_time_stream` - WebRTC streaming
- `graphql_api` - Real-time data API

### Configuration:

Phases are configured in `configs/manual_review_config.yaml`:

```yaml
phases:
  phase1:
    enabled: true
    plugins:
      - video_extraction
      - manual_review
      - bulk_operations

  phase2:
    enabled: true
    plugins:
      - pose_extractor
      - pose_visualizer
```

### Plugin Development:

To create a custom plugin:

1. Inherit from `PluginBase` (see `plugins/base.py`)
2. Implement required methods:
   - `metadata` - Plugin information
   - `initialize()` - Setup
   - `validate_dependencies()` - Check requirements

3. For UI plugins, inherit from `UIPlugin`:
   - `render_ui()` - Main UI
   - `render_sidebar()` - Sidebar controls (optional)
   - `render_settings()` - Settings page (optional)

**Example**: See `plugins/examples/ml_predictor.py` for Phase 4 ML demo

---

## 💡 Common Workflows

### Workflow 1: Add New Competition

```bash
# 1. Update config
edit configs/manual_review_config.yaml
# Add new competition entry

# 2. Prepare source video
# Place full broadcast video in data/original_videos/

# 3. Extract races
- Open "Add Video" page
- Select competition
- Enter timestamps for each race
- Extract one by one

# 4. Review in Video Library
- Check all races extracted
- Verify metadata
```

### Workflow 2: Prepare ML Training Data

```bash
# 1. Filter races in Video Library
- Competition: All
- Status: Reviewed (only corrected races)

# 2. Bulk Operations → Export
- Format: NPZ (ML-ready)
- ✅ Include Performance Metrics
- Execute

# 3. Download file
→ bulk_export_all_20251116_143022.npz

# 4. Use in ML training (Phase 4)
import numpy as np
data = np.load('export.npz', allow_pickle=True)
X = data['features']
y = data['labels']
```

### Workflow 3: Dataset Quality Check

```bash
# 1. Video Library → Filter
- Status: All
- Export to CSV

# 2. Bulk Operations → Validate
- Select all races
- Execute validation
- Check results

# 3. Bulk Operations → Summary Report
- Generate statistics
- Download JSON report
- Review duration_stats
```

---

## 🐛 Troubleshooting

### FFmpeg Not Found

**Error**: "FFmpeg not found. Please install ffmpeg first."

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
# Add to PATH
```

### Video Path Not Found

**Error**: "Source video not found: /path/to/video.mp4"

**Solution**:
- Verify file exists
- Use absolute path (not relative)
- Check file permissions
- On Windows: Use forward slashes or double backslashes

### Timestamp Validation Fails

**Error**: "Invalid format. Use MM:SS or HH:MM:SS"

**Solution**:
- Correct formats: `01:30`, `00:01:30`, `1:30`
- Incorrect formats: `1.5`, `90`, `01:30:00.5`
- Seconds must be 0-59
- Minutes must be 0-59

### No Metrics in Export

**Warning**: "No races with metrics found for NPZ export"

**Solution**:
- Metrics are only available after Phase 3 processing
- Run `scripts/phase3_metrics/batch_calculate_metrics.py` first
- Check `data/processed/metrics/` directory

---

## 📊 Performance Benchmarks

Tested with 188 races on Ubuntu 22.04:

| Operation | Time | Notes |
|-----------|------|-------|
| Library Load | <3s | All 188 videos |
| Video Extraction | 10-30s | Depends on segment length |
| Bulk Export (JSON) | <5s | 100 races |
| Bulk Export (NPZ) | <10s | 100 races with metrics |
| Bulk Validation | <15s | 100 races |
| Summary Report | <3s | 188 races |

---

## 🎓 Next Steps

### For Data Quality:
1. Use **Video Library** to identify remaining suspicious races
2. Use **Race Review** to manually correct each one
3. Use **Bulk Validation** to verify corrections

### For ML Preparation:
1. Ensure all corrections complete
2. Run Phase 3 metrics calculation (if not done)
3. Export to NPZ with metrics included
4. Proceed to Phase 4 model training

### For New Data:
1. Use **Add Video** to extract races from new competitions
2. Review extracted races for quality
3. Add to dataset

---

## 📚 Additional Resources

- **Full Implementation Plan**: [docs/PHASE_1.5.1_IMPLEMENTATION_PLAN.md](PHASE_1.5.1_IMPLEMENTATION_PLAN.md)
- **Phase 4 Research**: [docs/PHASE4_RESEARCH_REPORT.md](PHASE4_RESEARCH_REPORT.md)
- **User Guide** (Phase 1.5): [docs/manual_review_interface_user_guide.md](manual_review_interface_user_guide.md)
- **Developer Guide**: [docs/manual_review_interface_developer_guide.md](manual_review_interface_developer_guide.md)

---

## 🤝 Support

For issues, questions, or feature requests:
1. Check this guide first
2. Review error messages carefully
3. Consult implementation plan for technical details
4. Check plugin examples for customization

---

**Happy Reviewing! 🏔️**
**موفق باشید! 🧗**
