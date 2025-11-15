# Prompt Templates برای UI Session بعدی
# UI Next Session Prompt Templates

**تاریخ**: 2025-11-15
**مخاطب**: UI claude.ai/code
**پیش‌نیاز**: Phase 3 کامل شده و تست شده روی Local PC

---

## 📋 فهرست گزینه‌ها

این سند شامل 3 prompt template برای سناریوهای مختلف ادامه کار است:

1. **گزینه A**: بهبود و Refinement Phase 3
2. **گزینه B**: شروع Phase 4 (Machine Learning)
3. **گزینه C**: آماده‌سازی Publication و Documentation

هر گزینه شامل:
- شرح وضعیت فعلی
- اهداف مشخص
- Tasks قابل اجرا
- معیارهای موفقیت

---

## 🔧 گزینه A: بهبود Phase 3 (Refinement & Optimization)

### کی از این گزینه استفاده کنیم؟
- نتایج Phase 3 خوب است اما نیاز به بهبود دارد
- Winner prediction accuracy پایین است (<70%)
- می‌خواهید calibration را کامل integrate کنید
- می‌خواهید visualizations تعاملی داشته باشید
- نیاز به performance optimization دارید

---

### Prompt برای UI:

```markdown
# بهبود و Optimization Phase 3

با سلام! Phase 3 را با موفقیت تکمیل کردید و من آن را روی full dataset (188 races) تست کردم.

## وضعیت فعلی (Current Status)

✅ **موفقیت‌ها**:
- همه 188 race پردازش شدند (100% success rate)
- Aggregate metrics تولید شدند
- Dashboard کار می‌کند
- ML datasets آماده هستند

⚠️ **نقاط قابل بهبود**:
- Winner prediction accuracy: [X]% (هدف: >85%)
- Metrics فعلاً در pixel هستند (نیاز به calibration integration)
- Visualizations static هستند (نیاز به interactivity)
- Processing time: ~30s برای 188 races (هدف: <10s)

## درخواست بهبودها (Improvement Requests)

### Task A.1: Winner Prediction Enhancement
**هدف**: بهبود accuracy از [X]% به >85%

**Subtasks**:
1. **ML Model Training**:
   - استفاده از Random Forest یا XGBoost
   - Training روی ml_ready/train.csv
   - Validation روی ml_ready/test.csv
   - Feature importance analysis

2. **Feature Engineering**:
   - افزودن features جدید:
     - Acceleration patterns (avg, max, variance)
     - Split times (first 5s, middle, last 5s)
     - Movement consistency (jerk statistics)
     - Lane advantage (left vs right bias)

3. **Hyperparameter Tuning**:
   - Grid search یا Random search
   - Cross-validation (k=5)
   - Model comparison (RF vs XGB vs SVM)

4. **Validation**:
   - Test accuracy on full 188 races
   - Confusion matrix
   - Per-competition accuracy breakdown

**Deliverables**:
- `src/ml/winner_predictor_ml.py` - ML-based predictor
- `scripts/train_winner_model.py` - Training script
- `models/winner_predictor.pkl` - Trained model
- `docs/WINNER_PREDICTION_REPORT.md` - Performance report

---

### Task A.2: Calibration Integration
**هدف**: تبدیل همه metrics به meter-based units

**Subtasks**:
1. **Batch Calibration Pipeline**:
   - Integration با batch_calculate_metrics.py
   - خودکار load کردن calibration files
   - Fallback به pixel اگر calibration موجود نبود

2. **Unit Conversion**:
   - velocity: pixel/s → m/s
   - acceleration: pixel/s² → m/s²
   - positions: (x_px, y_px) → (x_m, y_m)

3. **Validation**:
   - مقایسه با IFSC standards (15m wall height)
   - Sanity checks (velocity 2-3 m/s, not 200-300 px/s)

**Deliverables**:
- Updated `batch_calculate_metrics.py` با calibration support
- Updated aggregate files با meter units
- Documentation در metrics fields

---

### Task A.3: Interactive Visualizations
**هدف**: Plotly interactive plots به جای static PNG

**Subtasks**:
1. **Plotly Integration**:
   - تبدیل race_plots.py به Plotly
   - Interactive features:
     - Zoom/pan
     - Hover tooltips (race info, exact values)
     - Click to highlight climber
     - Toggle lanes on/off

2. **Dashboard Enhancement**:
   - Embed Plotly plots در HTML dashboard
   - Add filters: competition, date range, velocity range
   - Add search: find specific climber/race
   - Responsive design improvements

3. **Export Options**:
   - HTML standalone (با embedded data)
   - JSON export برای web integration
   - SVG export برای publication

**Deliverables**:
- `src/visualization/interactive_plots.py` - Plotly version
- Updated dashboard با interactive plots
- `docs/INTERACTIVE_DASHBOARD_GUIDE.md` - Usage guide

---

### Task A.4: Performance Optimization
**هدف**: کاهش processing time از ~30s به <10s

**Subtasks**:
1. **Multiprocessing**:
   - Parallel processing با `multiprocessing.Pool`
   - Process N races simultaneously (N = CPU cores)
   - Progress bar با `tqdm`

2. **Caching**:
   - Cache pose data برای استفاده مجدد
   - Cache calibration data
   - Smart invalidation

3. **Vectorization**:
   - NumPy vectorization برای metric calculations
   - Avoid Python loops where possible

4. **Profiling**:
   - Identify bottlenecks با `cProfile`
   - Optimize hotspots
   - Benchmark before/after

**Deliverables**:
- Optimized versions of batch scripts
- Performance benchmark report
- Documentation در code comments

---

## معیارهای موفقیت (Success Criteria)

- [ ] Winner prediction accuracy >85%
- [ ] همه metrics در meter units
- [ ] Interactive dashboard کامل کار می‌کند
- [ ] Processing time <10s برای 188 races
- [ ] همه tests می‌گذرند
- [ ] Documentation کامل است

## اولویت‌بندی

**Priority 1 (Critical)**: Task A.2 - Calibration Integration
**Priority 2 (High)**: Task A.1 - Winner Prediction Enhancement
**Priority 3 (Medium)**: Task A.3 - Interactive Visualizations
**Priority 4 (Low)**: Task A.4 - Performance Optimization

## زمان تخمینی

- Task A.1: 6-8 hours
- Task A.2: 3-4 hours
- Task A.3: 4-6 hours
- Task A.4: 2-3 hours
- **Total**: 15-21 hours

لطفاً با Priority 1 شروع کنید و به ترتیب پیش بروید. بعد از هر task، test و commit کنید.

آیا می‌توانید این بهبودها را انجام دهید؟
```

---

## 🚀 گزینه B: شروع Phase 4 (Machine Learning)

### کی از این گزینه استفاده کنیم؟
- Phase 3 کامل و satisfactory است
- می‌خواهید به سراغ ML models پیشرفته بروید
- نیاز به performance prediction و classification دارید
- می‌خواهید NARX neural networks پیاده‌سازی کنید

---

### Prompt برای UI:

```markdown
# Phase 4: Machine Learning Models

با سلام! Phase 3 را با موفقیت تکمیل کردید و من کل dataset (188 races) را پردازش کردم.

## آماده برای Phase 4 🚀

همه زیرساخت‌ها آماده است:
- ✅ 188 race با pose data
- ✅ Calibrated metrics (pixel → meter)
- ✅ ML-ready datasets (376 samples, 8 features)
- ✅ Train/test splits (80/20)

حالا زمان ML models پیشرفته است!

---

## Phase 4 Tasks

### Task 4.1: NARX Neural Network (Performance Prediction)
**هدف**: پیش‌بینی finish time از time-series data

**Background**:
NARX (Nonlinear AutoRegressive with eXogenous inputs) یک RNN است که برای time-series prediction عالی است.

**Subtasks**:
1. **Data Preparation**:
   - استخراج time-series sequences از pose data
   - Window size: 30 frames (1 second at 30 FPS)
   - Features per timestep: COM_y, velocity_y, acceleration
   - Target: remaining time to finish

2. **Model Architecture**:
   - Input: sequence of (COM_y, vel, acc) × 30 timesteps
   - LSTM layers: 2-3 layers با 64-128 units
   - Dense layers: 2 layers با dropout
   - Output: predicted finish time (scalar)

3. **Training**:
   - Loss: Mean Squared Error (MSE)
   - Optimizer: Adam
   - Validation: time-based split (early races → later races)
   - Epochs: 50-100 با early stopping

4. **Evaluation**:
   - MAE (Mean Absolute Error) در seconds
   - R² score
   - Predictions vs Actual scatter plot

**Deliverables**:
- `src/ml/narx_model.py` - Model definition
- `scripts/train_narx.py` - Training script
- `models/narx_predictor.h5` - Trained model (Keras)
- `docs/NARX_PERFORMANCE_REPORT.md` - Results

**Expected Accuracy**: MAE <0.5s (excellent), <1.0s (good)

---

### Task 4.2: Climber Classification (Technique Clustering)
**هدف**: طبقه‌بندی climbers بر اساس technique

**Approach**: Unsupervised learning (clustering) + supervised classification

**Subtasks**:
1. **Feature Engineering**:
   - **Speed features**: avg/max/std velocity
   - **Smoothness features**: jerk statistics
   - **Efficiency features**: path efficiency, deviation
   - **Rhythm features**: FFT of velocity (dominant frequencies)

2. **Clustering (Unsupervised)**:
   - K-Means با K=3-5 clusters
   - Cluster interpretation:
     - Cluster 1: "Explosive" (high velocity, low smoothness)
     - Cluster 2: "Steady" (medium velocity, high smoothness)
     - Cluster 3: "Technical" (high efficiency, moderate velocity)

3. **Classification (Supervised)**:
   - Train Random Forest classifier
   - Input: 12+ features
   - Output: technique class (0, 1, 2, ...)
   - Cross-validation accuracy

4. **Visualization**:
   - 2D PCA projection of clusters
   - Cluster characteristics (radar charts)
   - Per-cluster statistics

**Deliverables**:
- `src/ml/climber_classifier.py` - Clustering + classification
- `scripts/cluster_climbers.py` - Clustering script
- `data/processed/ml_models/climber_clusters.pkl` - Cluster assignments
- `docs/CLIMBER_TAXONOMY_REPORT.md` - Cluster analysis

---

### Task 4.3: Early Performance Prediction
**هدف**: پیش‌بینی finish time از اولین 5 ثانیه (150 frames)

**Application**: Real-time prediction در مسابقات زنده

**Subtasks**:
1. **Dataset Creation**:
   - Extract first 5s of each race (150 frames)
   - Features: velocity profile, acceleration peaks, smoothness
   - Target: actual finish time

2. **Model Training**:
   - Compare models:
     - Random Forest Regressor
     - Gradient Boosting (XGBoost)
     - Neural Network (MLP)
   - 5-fold cross-validation

3. **Feature Importance**:
   - Which early features predict performance best?
   - SHAP values for interpretability

4. **Real-time Simulation**:
   - Simulate live prediction at t=1s, 2s, 3s, 4s, 5s
   - Track prediction improvement over time

**Deliverables**:
- `src/ml/early_predictor.py` - Early prediction model
- `scripts/train_early_predictor.py` - Training
- `notebooks/early_prediction_analysis.ipynb` - Analysis
- `docs/EARLY_PREDICTION_REPORT.md` - Results

**Expected Accuracy**: R² >0.7 (good), >0.85 (excellent)

---

### Task 4.4: Anomaly Detection (Injury/Slip Detection)
**هدف**: تشخیص unusual patterns (افتادن، لغزش، injury)

**Approach**: One-class classification (outlier detection)

**Subtasks**:
1. **Normal Pattern Learning**:
   - Train on "normal" races (finished successfully)
   - Features: velocity consistency, smoothness, acceleration variance
   - Model: Isolation Forest یا One-Class SVM

2. **Anomaly Scoring**:
   - Score each race (0-1, 0=normal, 1=anomaly)
   - Threshold tuning (e.g., top 5% as anomalies)

3. **Anomaly Analysis**:
   - Manual review of flagged races
   - Common anomaly patterns:
     - Sudden velocity drop (slip)
     - High jerk spikes (injury)
     - Abnormal trajectory (wall issue)

4. **Visualization**:
   - Anomaly heatmap across competitions
   - Time-series of anomalous races
   - Feature contributions to anomaly score

**Deliverables**:
- `src/ml/anomaly_detector.py` - Anomaly detection
- `scripts/detect_anomalies.py` - Batch detection
- `data/processed/anomalies/flagged_races.json` - Results
- `docs/ANOMALY_DETECTION_REPORT.md` - Analysis

---

## معیارهای موفقیت (Success Criteria)

- [ ] NARX MAE <1.0s
- [ ] Climber classification با 3-5 distinct clusters
- [ ] Early prediction R² >0.7
- [ ] Anomaly detection flags <10% of races
- [ ] همه models documented و reproducible
- [ ] Jupyter notebooks برای exploratory analysis

## اولویت‌بندی

**Priority 1**: Task 4.3 - Early Prediction (most practical)
**Priority 2**: Task 4.2 - Climber Classification (insightful)
**Priority 3**: Task 4.1 - NARX (advanced)
**Priority 4**: Task 4.4 - Anomaly Detection (safety)

## زمان تخمینی

- Task 4.1: 10-12 hours (complex)
- Task 4.2: 6-8 hours
- Task 4.3: 5-7 hours
- Task 4.4: 4-6 hours
- **Total**: 25-33 hours

## ملاحظات مهم

1. **Dependencies**: ممکن است نیاز به نصب باشد:
   ```
   tensorflow>=2.13.0  # برای NARX
   keras>=2.13.0
   xgboost>=1.7.0
   shap>=0.42.0  # برای interpretability
   ```

2. **GPU**: NARX training با GPU بسیار سریعتر است (اما CPU هم کار می‌کند)

3. **Reproducibility**: همه models با random seed=42 برای reproducibility

4. **Documentation**: هر model نیاز به توضیحات architecture و hyperparameters دارد

آیا آماده شروع Phase 4 هستید؟ لطفاً با Priority 1 شروع کنید.
```

---

## 📄 گزینه C: آماده‌سازی Publication

### کی از این گزینه استفاده کنیم؟
- می‌خواهید paper/conference presentation تهیه کنید
- نیاز به figures با کیفیت publication دارید
- می‌خواهید statistical analysis جامع داشته باشید
- نیاز به LaTeX tables و formatted outputs دارید

---

### Prompt برای UI:

```markdown
# آماده‌سازی Publication Materials

با سلام! قصد دارم از این پروژه یک paper/presentation علمی تهیه کنم.

## هدف (Goal)

ایجاد محتوای publication-ready شامل:
- High-resolution figures (300 DPI)
- Statistical analysis reports
- LaTeX-formatted tables
- Methodology documentation
- Reproducibility package

---

## Tasks

### Task C.1: Publication-Quality Figures
**هدف**: تولید figures برای journal paper

**Subtasks**:
1. **Figure 1: System Architecture**
   - Flowchart diagram: Video → Segmentation → Pose → Calibration → Analysis
   - Software: Graphviz یا manual creation
   - Format: PDF (vector) + PNG (300 DPI)
   - Size: 7 inches wide (single column) or 14 inches (double column)

2. **Figure 2: Pose Extraction Pipeline**
   - Sample video frame با overlay:
     - Detected holds (red circles)
     - Calibration grid
     - Pose keypoints (skeleton)
     - Dual-lane masks
   - Before/after comparison
   - High resolution (300 DPI PNG)

3. **Figure 3: Performance Comparison**
   - Multi-panel figure:
     - (a) Velocity time-series (best vs average vs slowest)
     - (b) Trajectory comparison (spatial plot)
     - (c) Acceleration profiles
     - (d) Path efficiency distribution
   - Publication fonts (10-12pt), clear labels
   - Color-blind friendly palette

4. **Figure 4: Competition Analysis**
   - Box plots of velocity distributions per competition
   - Statistical significance markers (p-values)
   - N values annotated
   - Error bars (std or sem)

5. **Figure 5: ML Model Performance**
   - (a) Confusion matrix (winner prediction)
   - (b) Feature importance (bar chart)
   - (c) Learning curves (training/validation)
   - (d) Predictions vs Actual (scatter + R²)

**Deliverables**:
- 5 high-res figures (PDF + PNG 300 DPI)
- Source code برای reproducibility
- `docs/FIGURES_CAPTIONS.md` - Figure captions

---

### Task C.2: Statistical Analysis Report
**هدف**: comprehensive statistical analysis برای Methods/Results sections

**Subtasks**:
1. **Descriptive Statistics**:
   - Per-competition summary:
     - N races, N climbers
     - Velocity: mean ± std, median, [min, max], quartiles
     - Path efficiency: mean ± std
     - Finish time: mean ± std (if calibrated)
   - Overall dataset summary
   - Export to CSV + LaTeX table

2. **Inferential Statistics**:
   - **Comparison tests**:
     - ANOVA: velocity differences across competitions
     - Post-hoc tests: Tukey HSD
     - Effect sizes: Cohen's d or η²
   - **Correlation analysis**:
     - Features vs finish time (Pearson r, p-values)
     - Correlation matrix heatmap
   - **Lane bias analysis**:
     - Paired t-test: left vs right lanes
     - Win rate per lane
     - Statistical significance

3. **Regression Analysis**:
   - Linear regression: finish time ~ features
   - Coefficients, R², p-values
   - Residual plots
   - Model assumptions checks (normality, homoscedasticity)

4. **Report Generation**:
   - PDF report با همه tests
   - Interpretation و conclusions
   - LaTeX source برای copy-paste به paper

**Deliverables**:
- `reports/statistical_analysis_report.pdf` - Main report
- `reports/statistical_analysis_report.tex` - LaTeX source
- `data/statistics/` - همه test results (CSV, JSON)

---

### Task C.3: LaTeX Tables
**هدف**: تولید tables برای paper

**Subtasks**:
1. **Table 1: Dataset Summary**
   - Columns: Competition, Date, N Races, N Climbers, Duration Range
   - Formatted با booktabs package
   - Caption: "Summary of competition dataset..."

2. **Table 2: Performance Statistics**
   - Rows: Competitions
   - Columns: Mean Velocity (m/s), Std, Mean Efficiency, Std, Best Time (s)
   - با statistical significance markers (*)

3. **Table 3: ML Model Comparison**
   - Rows: Models (Heuristic, Random Forest, XGBoost, ...)
   - Columns: Accuracy, Precision, Recall, F1-Score, Training Time
   - Best values highlighted

4. **Table 4: System Accuracy Validation**
   - Rows: Pipeline stages (Pose, Calibration, Metrics, ...)
   - Columns: Metric, Value, Threshold, Status (Pass/Fail)
   - e.g., "Pose Detection Rate: 97.2%, >95%, Pass"

**Deliverables**:
- `tables/` directory با .tex files
- Preview PDFs برای each table
- `tables/README.md` - Usage instructions

---

### Task C.4: Methodology Documentation
**هدف**: شرح دقیق methodology برای Methods section

**Subtasks**:
1. **Algorithm Pseudo-code**:
   - Race segmentation algorithm
   - Dual-lane detection
   - Calibration procedure
   - Metric calculation
   - Format: LaTeX algorithm2e package

2. **Parameter Justification**:
   - لیست همه hyperparameters
   - Justification برای هر کدام:
     - Literature reference
     - Empirical testing results
     - Sensitivity analysis

3. **Validation Methodology**:
   - Test set selection strategy
   - Cross-validation scheme
   - Evaluation metrics rationale

4. **Reproducibility Package**:
   - `REPRODUCIBILITY.md` - Step-by-step guide
   - Environment specification (requirements.txt + versions)
   - Sample data subset
   - Expected outputs
   - Runtime estimates

**Deliverables**:
- `docs/METHODOLOGY_DETAILED.md` - Comprehensive methods
- `docs/algorithm_pseudocode.tex` - Pseudo-code
- `REPRODUCIBILITY.md` - Reproducibility guide
- `environment.yml` - Conda environment (optional)

---

## معیارهای موفقیت

- [ ] همه figures در 300 DPI و publication-ready
- [ ] Statistical tests انجام شده و documented
- [ ] LaTeX tables با صحیح compile می‌شوند
- [ ] Methodology کامل و واضح است
- [ ] Reproducibility package تست شده

## Timeline

- Task C.1: 4-6 hours
- Task C.2: 6-8 hours
- Task C.3: 2-3 hours
- Task C.4: 4-5 hours
- **Total**: 16-22 hours

## ملاحظات

1. **Journal Target**: اگر journal خاصی در نظر دارید (e.g., Sports Engineering, PLOS ONE)، formatting guidelines را مشخص کنید

2. **Color vs Grayscale**: برخی journals grayscale print دارند - نمودارها باید در grayscale هم readable باشند

3. **Figure Size Limits**: معمولاً max 10 MB per figure - optimize کنید اگر بزرگتر شد

4. **LaTeX Packages**: booktabs, siunitx, algorithm2e - مطمئن شوید compatible هستند

آیا می‌توانید این materials را آماده کنید؟ لطفاً با Task C.1 شروع کنید.
```

---

## 📊 جدول مقایسه گزینه‌ها

| معیار | گزینه A (Refinement) | گزینه B (ML) | گزینه C (Publication) |
|-------|----------------------|---------------|------------------------|
| **زمان** | 15-21 hours | 25-33 hours | 16-22 hours |
| **پیچیدگی** | Medium | High | Medium |
| **نیاز به ML expertise** | Medium | High | Low |
| **خروجی اصلی** | Improved system | ML models | Paper materials |
| **Impact** | Better accuracy | New capabilities | Scientific output |
| **Dependencies جدید** | Plotly | TensorFlow, XGBoost | Graphviz, LaTeX |

---

## 💡 توصیه انتخاب

### اگر هدف شما:
- **بهبود accuracy و usability**: گزینه A
- **تحقیق و innovation**: گزینه B
- **انتشار paper**: گزینه C

### اگر زمان محدود دارید:
- گزینه A (سریع‌ترین ROI)

### اگر می‌خواهید comprehensive باشد:
- ترکیب: A → C → B

---

## 🔄 استفاده از Prompts

### چگونه از این prompts استفاده کنیم:

1. **انتخاب گزینه**: بر اساس نیاز و هدف
2. **سفارشی‌سازی**:
   - جایگزین کردن [X] با مقادیر واقعی
   - اضافه/حذف tasks بر اساس نیاز
   - تنظیم priorities
3. **کپی کامل prompt** در UI claude.ai/code
4. **پیگیری**: بررسی progress و feedback

### نکات:
- ✅ همیشه context کامل بدهید (Phase 3 complete, 188 races tested)
- ✅ مشخص کنید کدام parts اولویت دارند
- ✅ اگر timeline خاصی دارید، mention کنید
- ✅ اگر محدودیت‌هایی دارید (e.g., no GPU), بگویید

---

**موفق باشید!** 🎉

این templates برای کمک به شما در communication موثر با UI claude.ai/code طراحی شده‌اند.

---

**تهیه شده توسط**: Claude Code (Local PC)
**تاریخ**: 2025-11-15
