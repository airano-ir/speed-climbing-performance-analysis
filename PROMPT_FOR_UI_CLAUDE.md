# Prompt for UI claude.ai/code - Speed Climbing Performance Analysis

**Session Goal**: Complete Phase 2-5 implementation with comprehensive testing and documentation

---

## 🎯 CONTEXT (Current Project State)

### What's DONE ✅:

**Phase 1: Race Segmentation & Pose Extraction - 100%**
- 188 race segments extracted from 5 competitions (705 MB MP4s)
- Manual timestamps with late_start handling
- BlazePose dual-lane detector implemented (17/17 tests passing)
- batch_pose_extraction.py working (tested with 8 races)

**Phase 2 Core: Metrics & Visualization - 100%**
- performance_metrics.py (407 lines) - COM, velocity, acceleration, jerk, path efficiency
- time_series_plots.py (403 lines) - single climber + dual comparison
- Tested successfully with 2 races (96-99% detection rate)

**Current Issue**:
- Destructor warning in blazepose_extractor.py (non-critical but needs fix)
- Coordinates are normalized (0-1) - need pixel/meter conversion
- batch_pose_extraction.py is currently RUNNING on user's Windows machine (~3 hours for 188 races)
- Pose JSONs will be committed to GitHub AFTER batch processing completes

### What's NEXT ⏳:

You will implement:
1. IFSC Calibration (pixel → meter)
2. Video annotation pipeline
3. Advanced features (Phase 3-5)
4. Comprehensive testing
5. Documentation updates

---

## 📋 YOUR TASKS (Priority Order)

### PRIORITY HIGH (Must Complete - ~6-8 hours)

#### Task 1: Fix Destructor Error ✅ (30 min)
**File**: `src/phase1_pose_estimation/blazepose_extractor.py`
**Issue**: Line 446-451, AttributeError in `__del__`
**Fix**:
```python
def release(self):
    if hasattr(self, 'pose') and self.pose is not None:
        try:
            self.pose.close()
        except:
            pass  # Already closed
        logger.info("BlazePose resources released")
```
**Test**: Run batch_pose_extraction.py with --max-races 1
**Commit**: "fix(blazepose): handle MediaPipe cleanup in destructor properly"

---

#### Task 2: IFSC Calibration System ✅ (3-4 hours)

**Goal**: Convert pixel coordinates to meters using IFSC 20 standardized holds

**2.1. Parse IFSC PDF** (1 hour)
**File**: `src/calibration/ifsc_route_map.py`
**Input**: `docs/IFSC_Speed_Licence_Rules.pdf`
**Output**: `configs/ifsc_route_coordinates.json`

```python
# Expected output format:
{
  "holds": [
    {
      "hold_num": 1,
      "panel": "A1",
      "grid_x": 2,
      "grid_y": 1,
      "meter_x": 0.25,  # 2 × 125mm
      "meter_y": 0.125,
      "description": "Starting hold"
    },
    ...  # 20 holds total
  ],
  "wall": {
    "height_m": 15.0,
    "width_m": 3.0,
    "overhang_degrees": 5.0,
    "hold_spacing_mm": 125
  }
}
```

**Implementation hints**:
- Use PyPDF2 or pdfplumber to extract tables
- IFSC PDF has route map with panel grid (1500mm × 1500mm panels)
- Hold spacing: 125mm (perfect for calibration!)
- Coordinates relative to bottom-left corner

**Test**: Load JSON and verify 20 holds exist
**Commit**: "feat(calibration): parse IFSC route map from PDF"

---

**2.2. Hold Detector** (1 hour)
**File**: `src/phase1_pose_estimation/hold_detector.py`

```python
class HoldDetector:
    """Detect red IFSC holds in video frames."""

    def __init__(self, route_coordinates_path: str):
        self.holds = self._load_route_coordinates(route_coordinates_path)
        # HSV range for red holds
        self.hsv_lower = np.array([0, 100, 100])
        self.hsv_upper = np.array([10, 255, 255])

    def detect_holds(self, frame: np.ndarray) -> List[DetectedHold]:
        """
        Detect holds using color-based HSV thresholding.

        Returns:
            List of (hold_num, pixel_x, pixel_y, confidence)
        """
        # 1. Convert to HSV
        # 2. Threshold for red
        # 3. Find contours
        # 4. Match to expected positions (nearest neighbor)
        # 5. Return detected holds with confidence
        pass
```

**Test**: Process first frame of a race, detect at least 5 holds
**Commit**: "feat(calibration): implement hold detection with HSV"

---

**2.3. Static Camera Calibration** (1 hour)
**File**: `src/calibration/static_camera_calibration.py`

```python
class StaticCameraCalibrator:
    """Calibrate static camera using detected holds."""

    def calibrate(
        self,
        frame: np.ndarray,
        detected_holds: List[DetectedHold],
        route_coordinates: dict
    ) -> CalibrationResult:
        """
        Compute homography matrix from detected holds.

        Returns:
            CalibrationResult with:
            - homography_matrix (3×3)
            - pixel_to_meter_scale
            - rmse_error
            - pixel_to_meter() converter function
            - meter_to_pixel() converter function
        """
        # 1. Match detected holds to route coordinates
        # 2. Get pixel points (x_px, y_px)
        # 3. Get meter points (x_m, y_m)
        # 4. Compute homography with cv2.findHomography()
        # 5. Return CalibrationResult
        pass
```

**Test**: Calibrate first frame, convert nose keypoint (0.5, 0.5) → meters
**Commit**: "feat(calibration): implement static camera calibration"

---

**2.4. Update Performance Metrics** (30 min)
**File**: `src/analysis/performance_metrics.py`

Add calibration support:
```python
class PerformanceAnalyzer:
    def analyze_from_file(
        self,
        pose_json_path: Path,
        lane: str = 'left',
        calibration_path: Optional[Path] = None  # NEW
    ) -> PerformanceMetrics:
        # If calibration provided:
        #   - Load calibration
        #   - Convert all keypoints to meters
        #   - Calculate velocity in m/s (not px/s)
        #   - Calculate height in meters (0-15m)
        pass
```

**Test**: Re-run metrics with calibration, verify units are m/s
**Commit**: "feat(metrics): add IFSC calibration support for meter units"

---

#### Task 3: Video Annotation Pipeline ✅ (2-3 hours)

**Goal**: Generate annotated videos with BlazePose overlay

**3.1. Video Annotator** (2 hours)
**File**: `src/visualization/video_annotator.py`

```python
class VideoAnnotator:
    """Annotate videos with BlazePose keypoints."""

    def __init__(self, show_com: bool = True, show_velocity: bool = True):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def annotate_race(
        self,
        video_path: Path,
        pose_json_path: Path,
        output_path: Path,
        calibration_path: Optional[Path] = None
    ):
        """
        Create annotated video with:
        - BlazePose 33 keypoints skeleton
        - COM trajectory trail (last 30 frames)
        - Velocity/acceleration text overlay
        - Lane boundaries
        - Hold markers (if calibration provided)

        Args:
            video_path: Input race MP4
            pose_json_path: Pre-computed poses
            output_path: Output annotated MP4
            calibration_path: Optional calibration for hold overlay
        """
        # 1. Load video with cv2.VideoCapture
        # 2. Load poses from JSON
        # 3. Load calibration (if provided)
        # 4. For each frame:
        #    - Draw BlazePose skeleton (mp_drawing.draw_landmarks)
        #    - Draw COM as red circle
        #    - Draw COM trajectory trail
        #    - Draw text: velocity, height, time
        #    - Draw holds (if calibrated)
        # 5. Write output with cv2.VideoWriter
        pass
```

**Test**: Annotate 1 sample race, verify output looks good
**Commit**: "feat(viz): implement video annotation with pose overlay"

---

**3.2. Batch Annotation Script** (30 min)
**File**: `scripts/batch_annotate_videos.py`

```bash
python scripts/batch_annotate_videos.py \
  --input-videos data/race_segments \
  --input-poses data/processed/poses \
  --output data/processed/videos \
  --max-videos 5 \
  --calibration data/processed/calibration
```

**Test**: Annotate 5 sample races
**Commit**: "feat(scripts): add batch video annotation"

---

#### Task 4: Metrics Aggregation ✅ (1 hour)

**File**: `scripts/batch_metrics_calculation.py`

**Goal**: Process all 188 pose JSONs → aggregate CSV

```python
# For each pose JSON:
#   1. Calculate metrics for left climber
#   2. Calculate metrics for right climber
#   3. Append to aggregate_metrics.csv

# Output columns:
# race_id, competition, athlete_left, country_left, time_left, velocity_left, efficiency_left,
# athlete_right, country_right, time_right, velocity_right, efficiency_right, winner
```

**Output**: `data/processed/metrics/aggregate_metrics.csv` (~2 MB, 376 rows)

**Test**: Run on all poses, verify 188 races × 2 climbers = 376 rows
**Commit**: "feat(scripts): aggregate metrics for all 188 races"

---

#### Task 5: Comparative Visualizations ✅ (1 hour)

**File**: `scripts/generate_comparison_plots.py`

Generate:
1. **Top 10 Fastest Athletes** (bar chart)
2. **Velocity Distribution** (histogram)
3. **Path Efficiency Ranking** (scatter plot)
4. **Competition Winners** (grouped bar chart)

Save to: `data/processed/plots/comparisons/`

**Test**: Verify 4 PNG files generated
**Commit**: "feat(viz): generate comparative analysis plots"

---

#### Task 6: Update Notebooks ✅ (1 hour)

**6.1. Update 01_phase1_pose_estimation.ipynb**
- Use new dual_lane_detector.py (not inline code)
- Add calibration step
- Add metrics calculation
- Add video annotation

**6.2. Create 02_full_pipeline_demo.ipynb**
- Load sample pose JSON
- Calculate metrics
- Generate visualizations
- Compare two climbers
- Export results

**Test**: Run both notebooks in Google Colab
**Commit**: "docs(notebooks): update with latest pipeline"

---

### PRIORITY MEDIUM (If Time Permits - ~6-8 hours)

#### Task 7: Phase 3 - NARX Neural Network (3-4 hours)

**Goal**: Time-series prediction of climbing trajectory

**File**: `src/phase3_narx/narx_predictor.py`

```python
class NARXModel(nn.Module):
    """
    Nonlinear AutoRegressive with eXogenous inputs (NARX)

    Input:
        - Past poses (t-10 to t-1): 10 timesteps × 66 features (33 keypoints × 2D)
        - Current height: 1 feature
    Output:
        - Next pose (t+1): 66 features

    Architecture:
        Input(670) → LSTM(128) → LSTM(64) → Dense(66)
    """
    def __init__(self, input_size=670, hidden_size=128, output_size=66):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, 64, batch_first=True)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        # x shape: (batch, sequence=10, features=67)
        # LSTM → Dense
        pass
```

**Training**:
- Dataset: 188 races × frames
- Train/Val/Test: 70/15/15 split
- Loss: MSE for keypoint positions
- Metric: Average position error (pixels or meters)

**Test**: Train on 50 epochs, evaluate on test set
**Commit**: "feat(phase3): implement NARX neural network for trajectory prediction"

---

#### Task 8: Phase 4 - Fuzzy Logic System (2-3 hours)

**Goal**: Technique evaluation using fuzzy rules

**File**: `src/phase4_fuzzy/fuzzy_evaluator.py`

```python
import skfuzzy as fuzz

class TechniqueEvaluator:
    """
    Fuzzy logic system for climbing technique evaluation.

    Inputs (fuzzified):
        - velocity: slow/medium/fast
        - smoothness: jerky/smooth/very_smooth
        - path_efficiency: inefficient/efficient/optimal

    Rules:
        IF velocity is fast AND smoothness is very_smooth AND efficiency is optimal
        THEN technique is excellent

        IF velocity is medium AND smoothness is smooth
        THEN technique is good

        ... (10-15 rules total)

    Output (defuzzified):
        - technique_score: 0-100
    """
    def __init__(self):
        # Define membership functions
        self.velocity = fuzz.trimf(...)
        self.smoothness = fuzz.trimf(...)
        # ...

    def evaluate(self, metrics: PerformanceMetrics) -> float:
        # Fuzzify inputs
        # Apply rules
        # Defuzzify output
        return technique_score
```

**Test**: Evaluate top 10 athletes, verify score 80-100 for fast climbers
**Commit**: "feat(phase4): implement fuzzy logic technique evaluation"

---

#### Task 9: Phase 5 - Interactive Dashboard (2-3 hours)

**Goal**: Streamlit/Dash web interface

**File**: `src/phase5_dashboard/app.py`

```python
import streamlit as st

st.title("Speed Climbing Performance Analysis")

# Sidebar: Select competition & race
competition = st.sidebar.selectbox("Competition", competitions)
race = st.sidebar.selectbox("Race", races)

# Main: Display results
st.subheader(f"Race {race} - {competition}")

col1, col2 = st.columns(2)
with col1:
    st.image(f"plots/{race}_single_left.png")
with col2:
    st.image(f"plots/{race}_single_right.png")

st.subheader("Comparison")
st.image(f"plots/{race}_dual.png")

# Metrics table
st.dataframe(metrics_df)

# Video player (if available)
if annotated_video_exists:
    st.video(f"videos/{race}_annotated.mp4")
```

**Test**: Run `streamlit run src/phase5_dashboard/app.py`
**Commit**: "feat(phase5): create interactive Streamlit dashboard"

---

## ⚙️ IMPORTANT INSTRUCTIONS

### 🧪 Testing Protocol

**After EACH task**:
1. Write unit tests in `tests/test_<module>.py`
2. Run: `pytest tests/test_<module>.py -v`
3. Fix any failures
4. Test with real data (1-2 sample races)
5. Verify output files exist and are valid

### 📝 Commit Protocol

**After EACH completed task**:
1. Stage changes: `git add <files>`
2. Commit with descriptive message:
   ```bash
   git commit -m "feat(<scope>): <description>

   - Bullet point 1
   - Bullet point 2
   - Tested with X races, Y% success rate

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```
3. **Update MASTER_CONTEXT.md** with:
   - Mark task as complete ✅
   - Add commit hash
   - Update progress tracker
   - Add any important notes
4. Commit MASTER_CONTEXT:
   ```bash
   git add MASTER_CONTEXT.md
   git commit -m "docs: update MASTER_CONTEXT with Task X completion"
   ```

### 🛑 Error Handling

**IF you encounter a blocking error**:
1. Document the error clearly
2. Add TODO comment in code
3. Commit current progress
4. Update MASTER_CONTEXT with status: "⚠️ BLOCKED: <reason>"
5. Move to next task
6. **DO NOT spend more than 30 min debugging** - mark as blocked and continue

### 📊 Progress Reporting

**After every 2-3 tasks**, create a progress report commit:
```bash
# Create PROGRESS_REPORT_<DATE>.md
# Include:
# - Tasks completed ✅
# - Tasks blocked ⚠️
# - Tests passing/failing
# - Output files generated
# - Next steps

git add PROGRESS_REPORT_*.md
git commit -m "docs: progress report for tasks X-Y"
```

---

## 📂 Expected Outputs

By the end of your session, these should exist:

### Code (new files):
```
src/calibration/
  ├── ifsc_route_map.py
  ├── static_camera_calibration.py
  └── __init__.py

src/visualization/
  └── video_annotator.py

src/phase3_narx/
  └── narx_predictor.py           # If time permits

src/phase4_fuzzy/
  └── fuzzy_evaluator.py          # If time permits

src/phase5_dashboard/
  └── app.py                      # If time permits

scripts/
  ├── batch_annotate_videos.py
  ├── batch_metrics_calculation.py
  ├── generate_comparison_plots.py
  └── select_sample_poses.py

tests/
  ├── test_calibration.py
  ├── test_video_annotator.py
  ├── test_narx.py                # If time permits
  └── test_fuzzy.py               # If time permits
```

### Data (generated):
```
configs/
  └── ifsc_route_coordinates.json

data/processed/
  ├── calibration/              # 188 calibration JSONs (~10 MB)
  ├── metrics/
  │   └── aggregate_metrics.csv  # 376 rows (~2 MB)
  ├── plots/comparisons/
  │   ├── top10_athletes.png
  │   ├── velocity_distribution.png
  │   ├── path_efficiency_ranking.png
  │   └── competition_winners.png
  └── videos/samples/           # 5-10 annotated MP4s (~50-100 MB)
```

### Docs (updated):
```
docs/
  ├── CALIBRATION_GUIDE.md       # NEW
  ├── PHASE3_NARX_DESIGN.md      # NEW (if implemented)
  └── PHASE4_FUZZY_RULES.md      # NEW (if implemented)

notebooks/
  ├── 01_phase1_pose_estimation.ipynb  # UPDATED
  └── 02_full_pipeline_demo.ipynb      # NEW

MASTER_CONTEXT.md               # UPDATED
PROGRESS_REPORT_2025-11-15.md   # NEW
```

---

## 🎯 Success Criteria

### Minimum (Must Have):
- ✅ Destructor error fixed
- ✅ IFSC calibration working (parse PDF → detect holds → calibrate)
- ✅ Metrics in meters (not pixels)
- ✅ Video annotation working (5+ samples)
- ✅ Aggregate metrics CSV generated
- ✅ Comparative plots generated
- ✅ All tests passing
- ✅ MASTER_CONTEXT updated

### Ideal (Nice to Have):
- ✅ NARX model implemented and trained
- ✅ Fuzzy logic evaluator working
- ✅ Streamlit dashboard functional
- ✅ Comprehensive documentation

---

## 📚 Resources Available

### Documentation:
- [MASTER_CONTEXT.md](MASTER_CONTEXT.md) - Project overview
- [OUTPUT_STRUCTURE_GUIDE.md](docs/OUTPUT_STRUCTURE_GUIDE.md) - File structure guide
- [MANUAL_SEGMENTATION_GUIDE.md](docs/MANUAL_SEGMENTATION_GUIDE.md) - Race segmentation
- [IFSC_Speed_Licence_Rules.pdf](docs/IFSC_Speed_Licence_Rules.pdf) - IFSC standards

### Existing Code:
- `src/phase1_pose_estimation/` - All Phase 1 modules
- `src/analysis/performance_metrics.py` - Metrics calculator
- `src/visualization/time_series_plots.py` - Plotting
- `tests/` - Existing test suite

### Data:
- `data/race_segments/` - 188 race MP4s (user has locally)
- `data/processed/poses/` - Will have pose JSONs after batch processing completes
- Sample poses will be in `data/processed/poses/samples/` for testing

---

## 🚀 START HERE

1. **Read MASTER_CONTEXT.md** to understand project
2. **Read OUTPUT_STRUCTURE_GUIDE.md** to understand file structure
3. **Start with Task 1** (destructor fix)
4. **Test, commit, update MASTER_CONTEXT**
5. **Continue through tasks in order**
6. **If blocked, document and move on**
7. **Commit progress every 1-2 hours**

Good luck! Remember:
- Test frequently
- Commit often
- Update docs
- If stuck, move on

---

**END OF PROMPT**
