# PROMPT: Complete Pipeline with Reliable Data (114 Races)

**Version**: 2.0
**Date**: 2025-11-17
**Target**: claude.ai/code (UI Claude)
**Priority**: CRITICAL
**Estimated Time**: 12-16 hours (across multiple sessions)

---

## 🎯 Mission

Build a **complete, production-ready data pipeline** using the **114 reliable races** (188 total - 74 suspicious) to:
1. Generate high-quality pose data
2. Perform accurate calibration
3. Calculate validated performance metrics
4. Create comprehensive visualizations and dashboards
5. Export ML-ready datasets

**Why Now?**: We have 114 races with verified detection quality. Let's process these first, validate the entire pipeline, then add the remaining 74 after manual review.

---

## 📊 Current Status & Context

### Data Inventory (2025-11-17)

**Total Races**: 188 across 5 competitions
- ✅ **114 Reliable Races** (60.6% of dataset) - Can use NOW
  - No detection issues
  - Verified duration ranges (4.5-15s)
  - Metadata complete
- ⏸️ **74 Suspicious Races** (39.4% of dataset) - Need manual review
  - 5 CRITICAL: Negative/near-zero duration
  - 58 Zilina 2025: Systematic failure (84% of that competition!)
  - 11 Others: Too short or too long

**Current Pipeline Status**:
```
Phase 1: Race Segmentation     ✅ 100% (188 videos extracted)
Phase 2: Pose Estimation       ⚠️  1% (2 races processed)
Phase 2.5: Calibration         ⚠️  0% (0 races calibrated)
Phase 3: Metrics Calculation   ⚠️  0.5% (1 race processed)
Phase 3.5: Aggregation         ❌ 0% (no aggregated data)
Phase 3.6: Visualization       ❌ 0% (no dashboards)
```

**Why Processing Stopped**:
- Discovered 74 problematic races during Phase 3 testing
- Built Manual Review Interface (Phase 1.5) to fix them
- **Decision**: Process 114 reliable races now, add 74 later

---

## 🎯 Objectives

### Primary Goals

1. **Complete Pose Extraction** for 114 reliable races
   - Extract BlazePose keypoints (33 landmarks × ~200 frames each)
   - Save to `data/processed/poses/<race_id>_pose.json`
   - Validate extraction quality (missing frames < 5%)

2. **Complete Calibration** for 114 reliable races
   - Generate camera calibration using 20-hold IFSC standard
   - Save to `data/processed/calibration/<race_id>_calibration.json`
   - Validate RMSE < 1cm (target based on Phase 3 testing)

3. **Calculate Performance Metrics** for 114 reliable races
   - Use corrected frame ranges from metadata
   - Apply calibration for accurate measurements
   - Save to `data/processed/metrics/<race_id>_metrics.json`
   - Validate velocity 2-3× higher than before

4. **Aggregate and Analyze**
   - Generate leaderboards, comparisons, statistics
   - Create ML-ready datasets (CSV, NumPy arrays)
   - Export train/test splits (80/20)

5. **Build Interactive Dashboard**
   - HTML dashboard with visualizations
   - Comparative analysis tools
   - Quality assurance reports

6. **Validate End-to-End**
   - Verify all 114 races processed successfully
   - Check data quality metrics
   - Generate validation report

---

## 📋 Task Breakdown

### Phase 1: Identify Reliable Races (1 hour)

**Goal**: Create definitive list of 114 reliable races

**Steps**:
1. Load progress tracker: `data/manual_review/progress_tracker.csv`
2. Extract 74 suspicious race IDs
3. Get all 188 race IDs from `data/race_segments/*/`
4. Compute reliable races = ALL - SUSPICIOUS
5. Save to `data/processed/reliable_races_list.json`

**Output File Format** (`reliable_races_list.json`):
```json
{
  "total_races": 188,
  "suspicious_races": 74,
  "reliable_races": 114,
  "reliable_race_ids": [
    "Speed_finals_Chamonix_2024_race002",
    "Speed_finals_Chamonix_2024_race003",
    ...
  ],
  "by_competition": {
    "chamonix_2024": 28,
    "innsbruck_2024": 30,
    "seoul_2024": 25,
    "villars_2024": 20,
    "zilina_2025": 11
  },
  "generation_date": "2025-11-17",
  "notes": "Races with verified detection quality, ready for pipeline processing"
}
```

**Validation**:
- Verify sum = 114
- Check all competitions represented
- Ensure no duplicates

**Deliverable**:
- `scripts/generate_reliable_races_list.py` (NEW)
- `data/processed/reliable_races_list.json` (NEW)

---

### Phase 2: Batch Pose Extraction (3-4 hours)

**Goal**: Extract BlazePose keypoints for all 114 reliable races

**Implementation**:

Create `scripts/batch_pose_extraction_reliable.py`:

```python
"""
Batch Pose Extraction - Reliable Races Only
============================================
Extract BlazePose keypoints for 114 verified races.

Usage:
    python scripts/batch_pose_extraction_reliable.py

Output:
    data/processed/poses/<race_id>_pose.json (114 files)

Estimated Time: 3-4 hours (depends on hardware)
"""

import json
import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class BatchPoseExtractor:
    """Extract poses for reliable races in batch."""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Highest quality
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_race_poses(self, video_path: Path, metadata_path: Path) -> dict:
        """
        Extract poses for a single race.

        Args:
            video_path: Path to race video
            metadata_path: Path to race metadata JSON

        Returns:
            dict with frames, poses, quality metrics
        """
        # Load metadata for frame range
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        start_frame = metadata.get('detected_start_frame', 0)
        finish_frame = metadata.get('detected_finish_frame', 10000)

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_data = []
        missing_frames = 0

        for frame_idx in range(start_frame, finish_frame + 1):
            ret, frame = cap.read()
            if not ret:
                missing_frames += 1
                continue

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with BlazePose
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract landmarks
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })

                frames_data.append({
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / metadata.get('fps', 30.0),
                    'landmarks': landmarks
                })
            else:
                missing_frames += 1

        cap.release()

        # Quality metrics
        total_frames = finish_frame - start_frame + 1
        success_rate = (total_frames - missing_frames) / total_frames * 100

        return {
            'race_id': metadata['race_id'],
            'competition': video_path.parent.name,
            'frames': frames_data,
            'total_frames': total_frames,
            'extracted_frames': len(frames_data),
            'missing_frames': missing_frames,
            'success_rate': success_rate,
            'extraction_date': datetime.now().isoformat()
        }

    def process_all_reliable_races(self, reliable_races_file: Path):
        """Process all reliable races."""
        # Load reliable races list
        with open(reliable_races_file, 'r', encoding='utf-8') as f:
            reliable_data = json.load(f)

        race_ids = reliable_data['reliable_race_ids']

        print(f"Processing {len(race_ids)} reliable races...")

        success_count = 0
        failed_races = []

        for race_id in tqdm(race_ids, desc="Extracting poses"):
            try:
                # Find video and metadata
                video_path, metadata_path = self._find_race_files(race_id)

                # Extract poses
                pose_data = self.extract_race_poses(video_path, metadata_path)

                # Save
                output_path = Path(f"data/processed/poses/{race_id}_pose.json")
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(pose_data, f, indent=2)

                success_count += 1

            except Exception as e:
                print(f"Failed for {race_id}: {e}")
                failed_races.append({'race_id': race_id, 'error': str(e)})

        # Generate report
        report = {
            'total_races': len(race_ids),
            'successful': success_count,
            'failed': len(failed_races),
            'failed_races': failed_races,
            'completion_date': datetime.now().isoformat()
        }

        with open('data/processed/pose_extraction_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Pose extraction complete!")
        print(f"   Successful: {success_count}/{len(race_ids)}")
        print(f"   Failed: {len(failed_races)}")

    def _find_race_files(self, race_id: str):
        """Find video and metadata for a race."""
        # Search in all competition folders
        race_segments_dir = Path("data/race_segments")

        for comp_dir in race_segments_dir.iterdir():
            if not comp_dir.is_dir():
                continue

            video_path = comp_dir / f"{race_id}.mp4"
            metadata_path = comp_dir / f"{race_id}_metadata.json"

            if video_path.exists() and metadata_path.exists():
                return video_path, metadata_path

        raise FileNotFoundError(f"Files not found for {race_id}")


if __name__ == "__main__":
    extractor = BatchPoseExtractor()
    extractor.process_all_reliable_races(
        Path("data/processed/reliable_races_list.json")
    )
```

**Validation Criteria**:
- All 114 races processed
- Success rate > 95% per race
- Missing frames < 5% per race
- Output JSON files valid

**Deliverables**:
- `scripts/batch_pose_extraction_reliable.py` (NEW)
- `data/processed/poses/<race_id>_pose.json` (114 files)
- `data/processed/pose_extraction_report.json` (NEW)

---

### Phase 3: Batch Calibration (2-3 hours)

**Goal**: Generate camera calibration for all 114 reliable races

**Use Existing Code**: `src/calibration/camera_calibration.py` (already tested!)

**Implementation**:

Create `scripts/batch_calibration_reliable.py`:

```python
"""
Batch Calibration - Reliable Races Only
========================================
Generate camera calibration for 114 verified races using IFSC 20-hold standard.

Reference: Previous batch calibration achieved RMSE < 1cm for all races.

Usage:
    python scripts/batch_calibration_reliable.py

Output:
    data/processed/calibration/<race_id>_calibration.json (114 files)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.calibration.camera_calibration import PeriodicCalibrator
import json
from tqdm import tqdm
from datetime import datetime

def calibrate_reliable_races():
    """Calibrate all reliable races."""

    # Load reliable races list
    with open('data/processed/reliable_races_list.json', 'r', encoding='utf-8') as f:
        reliable_data = json.load(f)

    race_ids = reliable_data['reliable_race_ids']

    print(f"Calibrating {len(race_ids)} reliable races...")

    success_count = 0
    failed_races = []
    calibration_quality = []

    for race_id in tqdm(race_ids, desc="Calibrating"):
        try:
            # Find video file
            video_path = find_video_path(race_id)

            # Calibrate using PeriodicCalibrator
            calibrator = PeriodicCalibrator(
                video_path=str(video_path),
                wall_height_m=15.0,  # IFSC standard
                num_holds=20,        # IFSC standard
                output_dir="data/processed/calibration"
            )

            # Run calibration
            calibration_data = calibrator.calibrate()

            # Extract quality metrics
            rmse = calibration_data.get('rmse_cm', 999)
            calibration_quality.append({
                'race_id': race_id,
                'rmse_cm': rmse,
                'pass': rmse < 1.0  # Target: < 1cm
            })

            success_count += 1

        except Exception as e:
            print(f"Failed for {race_id}: {e}")
            failed_races.append({'race_id': race_id, 'error': str(e)})

    # Calculate statistics
    rmse_values = [c['rmse_cm'] for c in calibration_quality]
    avg_rmse = sum(rmse_values) / len(rmse_values) if rmse_values else 0
    pass_rate = sum(1 for c in calibration_quality if c['pass']) / len(calibration_quality) * 100

    # Generate report
    report = {
        'total_races': len(race_ids),
        'successful': success_count,
        'failed': len(failed_races),
        'failed_races': failed_races,
        'quality_metrics': {
            'average_rmse_cm': avg_rmse,
            'pass_rate_percent': pass_rate,
            'target_rmse_cm': 1.0
        },
        'calibration_quality': calibration_quality,
        'completion_date': datetime.now().isoformat()
    }

    with open('data/processed/calibration_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Calibration complete!")
    print(f"   Successful: {success_count}/{len(race_ids)}")
    print(f"   Average RMSE: {avg_rmse:.2f} cm")
    print(f"   Pass rate (< 1cm): {pass_rate:.1f}%")


def find_video_path(race_id: str) -> Path:
    """Find video path for a race."""
    race_segments_dir = Path("data/race_segments")

    for comp_dir in race_segments_dir.iterdir():
        if not comp_dir.is_dir():
            continue

        video_path = comp_dir / f"{race_id}.mp4"
        if video_path.exists():
            return video_path

    raise FileNotFoundError(f"Video not found for {race_id}")


if __name__ == "__main__":
    calibrate_reliable_races()
```

**Validation Criteria**:
- All 114 races calibrated
- Average RMSE < 1cm
- Pass rate (RMSE < 1cm) > 95%

**Deliverables**:
- `scripts/batch_calibration_reliable.py` (NEW)
- `data/processed/calibration/<race_id>_calibration.json` (114 files)
- `data/processed/calibration_report.json` (NEW)

---

### Phase 4: Batch Metrics Calculation (3-4 hours)

**Goal**: Calculate performance metrics for all 114 reliable races with calibration

**Use Existing Code**: `scripts/batch_calculate_metrics.py` (already exists!)

**Modifications Needed**:
1. Filter to only process 114 reliable races
2. Ensure frame filtering is applied (already done in Phase 3)
3. Validate metric quality

**Implementation**:

Create `scripts/batch_metrics_reliable.py`:

```python
"""
Batch Metrics Calculation - Reliable Races Only
===============================================
Calculate performance metrics for 114 verified races with calibration.

Features:
- Uses corrected frame ranges from metadata
- Applies camera calibration for accurate measurements
- Validates metric quality

Usage:
    python scripts/batch_metrics_reliable.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.analysis.performance_metrics import PerformanceMetricsCalculator
import json
from tqdm import tqdm
from datetime import datetime

def calculate_metrics_for_reliable_races():
    """Calculate metrics for all reliable races."""

    # Load reliable races list
    with open('data/processed/reliable_races_list.json', 'r', encoding='utf-8') as f:
        reliable_data = json.load(f)

    race_ids = reliable_data['reliable_race_ids']

    print(f"Calculating metrics for {len(race_ids)} reliable races...")

    success_count = 0
    failed_races = []
    metrics_quality = []

    for race_id in tqdm(race_ids, desc="Calculating metrics"):
        try:
            # Find files
            pose_path = Path(f"data/processed/poses/{race_id}_pose.json")
            calib_path = Path(f"data/processed/calibration/{race_id}_calibration.json")
            metadata_path = find_metadata_path(race_id)

            if not pose_path.exists():
                raise FileNotFoundError(f"Pose file not found: {pose_path}")
            if not calib_path.exists():
                raise FileNotFoundError(f"Calibration file not found: {calib_path}")

            # Load metadata for frame range
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Calculate metrics
            calculator = PerformanceMetricsCalculator(
                pose_file=str(pose_path),
                calibration_file=str(calib_path),
                fps=metadata.get('fps', 30.0)
            )

            # Get metrics
            metrics = calculator.calculate_all_metrics(
                start_frame=metadata.get('detected_start_frame'),
                end_frame=metadata.get('detected_finish_frame')
            )

            # Add race info
            metrics['race_id'] = race_id
            metrics['competition'] = metadata.get('competition', 'unknown')
            metrics['calculation_date'] = datetime.now().isoformat()

            # Save metrics
            output_path = Path(f"data/processed/metrics/{race_id}_metrics.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            # Validate quality
            avg_velocity = metrics.get('average_velocity_ms', 0)
            metrics_quality.append({
                'race_id': race_id,
                'avg_velocity_ms': avg_velocity,
                'valid': 1.0 <= avg_velocity <= 3.0  # Typical range
            })

            success_count += 1

        except Exception as e:
            print(f"Failed for {race_id}: {e}")
            failed_races.append({'race_id': race_id, 'error': str(e)})

    # Generate report
    valid_metrics = sum(1 for m in metrics_quality if m['valid'])
    validity_rate = valid_metrics / len(metrics_quality) * 100 if metrics_quality else 0

    report = {
        'total_races': len(race_ids),
        'successful': success_count,
        'failed': len(failed_races),
        'failed_races': failed_races,
        'quality_metrics': {
            'validity_rate_percent': validity_rate,
            'valid_count': valid_metrics
        },
        'metrics_quality': metrics_quality,
        'completion_date': datetime.now().isoformat()
    }

    with open('data/processed/metrics_calculation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Metrics calculation complete!")
    print(f"   Successful: {success_count}/{len(race_ids)}")
    print(f"   Validity rate: {validity_rate:.1f}%")


def find_metadata_path(race_id: str) -> Path:
    """Find metadata path for a race."""
    race_segments_dir = Path("data/race_segments")

    for comp_dir in race_segments_dir.iterdir():
        if not comp_dir.is_dir():
            continue

        metadata_path = comp_dir / f"{race_id}_metadata.json"
        if metadata_path.exists():
            return metadata_path

    raise FileNotFoundError(f"Metadata not found for {race_id}")


if __name__ == "__main__":
    calculate_metrics_for_reliable_races()
```

**Validation Criteria**:
- All 114 races processed
- Velocity ranges realistic (1-3 m/s)
- No negative or zero values
- All metrics complete

**Deliverables**:
- `scripts/batch_metrics_reliable.py` (NEW)
- `data/processed/metrics/<race_id>_metrics.json` (114 files)
- `data/processed/metrics_calculation_report.json` (NEW)

---

### Phase 5: Data Aggregation & ML Export (2-3 hours)

**Goal**: Aggregate metrics, create leaderboards, export ML datasets

**Use Existing Code**: `scripts/aggregate_metrics.py` (may exist from Phase 3)

**Implementation**:

Create `scripts/aggregate_reliable_data.py`:

```python
"""
Data Aggregation & ML Export - Reliable Races
==============================================
Aggregate metrics, create leaderboards, export ML-ready datasets.

Outputs:
- Leaderboards (by competition, overall)
- Comparative statistics
- ML datasets (CSV, NumPy, train/test splits)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def aggregate_all_metrics():
    """Aggregate all metrics from 114 reliable races."""

    # Load reliable races list
    with open('data/processed/reliable_races_list.json', 'r', encoding='utf-8') as f:
        reliable_data = json.load(f)

    race_ids = reliable_data['reliable_race_ids']

    all_metrics = []

    for race_id in race_ids:
        metrics_path = Path(f"data/processed/metrics/{race_id}_metrics.json")

        if not metrics_path.exists():
            continue

        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        all_metrics.append(metrics)

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Save aggregated CSV
    df.to_csv('data/processed/aggregated_metrics_reliable.csv', index=False)

    # Create leaderboards
    create_leaderboards(df)

    # Export ML datasets
    export_ml_datasets(df)

    print(f"✅ Aggregation complete!")
    print(f"   Races processed: {len(df)}")
    print(f"   Features: {len(df.columns)}")


def create_leaderboards(df):
    """Create various leaderboards."""

    # Overall leaderboard (by average velocity)
    leaderboard_overall = df.nlargest(20, 'average_velocity_ms')[
        ['race_id', 'competition', 'average_velocity_ms', 'total_time_s']
    ]
    leaderboard_overall.to_csv('data/processed/leaderboard_overall.csv', index=False)

    # By competition
    for comp in df['competition'].unique():
        comp_df = df[df['competition'] == comp]
        comp_leaderboard = comp_df.nlargest(10, 'average_velocity_ms')[
            ['race_id', 'average_velocity_ms', 'total_time_s']
        ]
        comp_leaderboard.to_csv(f'data/processed/leaderboard_{comp}.csv', index=False)

    print(f"   Leaderboards created: {len(df['competition'].unique()) + 1}")


def export_ml_datasets(df):
    """Export ML-ready datasets."""

    # Feature selection
    feature_cols = [
        'average_velocity_ms',
        'max_velocity_ms',
        'acceleration_ms2',
        'smoothness_score',
        'total_time_s'
    ]

    # Filter to available features
    available_features = [col for col in feature_cols if col in df.columns]

    X = df[available_features].values
    y = df['average_velocity_ms'].values  # Regression target

    # Train/test split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save NumPy arrays
    np.save('data/processed/ml_X_train.npy', X_train)
    np.save('data/processed/ml_X_test.npy', X_test)
    np.save('data/processed/ml_y_train.npy', y_train)
    np.save('data/processed/ml_y_test.npy', y_test)

    # Save feature names
    with open('data/processed/ml_feature_names.json', 'w') as f:
        json.dump(available_features, f, indent=2)

    print(f"   ML datasets exported:")
    print(f"      Train: {X_train.shape}")
    print(f"      Test: {X_test.shape}")


if __name__ == "__main__":
    aggregate_all_metrics()
```

**Deliverables**:
- `scripts/aggregate_reliable_data.py` (NEW)
- `data/processed/aggregated_metrics_reliable.csv`
- `data/processed/leaderboard_overall.csv`
- `data/processed/leaderboard_<comp>.csv` (5 files)
- `data/processed/ml_*.npy` (4 files)
- `data/processed/ml_feature_names.json`

---

### Phase 6: Quality Assurance & Validation (1-2 hours)

**Goal**: Comprehensive validation of entire pipeline

**Create** `scripts/validate_pipeline_reliable.py`:

```python
"""
Pipeline Validation - Reliable Races
====================================
Comprehensive validation of all pipeline stages for 114 races.

Checks:
1. File completeness (all outputs exist)
2. Data quality (ranges, missing values)
3. Cross-validation (pose → metrics consistency)
4. Performance benchmarks
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def validate_pipeline():
    """Run comprehensive pipeline validation."""

    # Load reliable races
    with open('data/processed/reliable_races_list.json', 'r', encoding='utf-8') as f:
        reliable_data = json.load(f)

    race_ids = reliable_data['reliable_race_ids']

    validation_results = {
        'total_races': len(race_ids),
        'file_completeness': check_file_completeness(race_ids),
        'data_quality': check_data_quality(race_ids),
        'performance_benchmarks': check_performance_benchmarks(),
        'validation_date': datetime.now().isoformat()
    }

    # Save report
    with open('data/processed/pipeline_validation_report.json', 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2)

    # Print summary
    print(f"✅ Pipeline Validation Complete!")
    print(f"   Total races: {len(race_ids)}")
    print(f"   File completeness: {validation_results['file_completeness']['pass_rate']:.1f}%")
    print(f"   Data quality: {validation_results['data_quality']['pass_rate']:.1f}%")


def check_file_completeness(race_ids):
    """Check if all expected files exist."""

    missing_files = {
        'pose': [],
        'calibration': [],
        'metrics': []
    }

    for race_id in race_ids:
        if not Path(f"data/processed/poses/{race_id}_pose.json").exists():
            missing_files['pose'].append(race_id)
        if not Path(f"data/processed/calibration/{race_id}_calibration.json").exists():
            missing_files['calibration'].append(race_id)
        if not Path(f"data/processed/metrics/{race_id}_metrics.json").exists():
            missing_files['metrics'].append(race_id)

    total_expected = len(race_ids) * 3
    total_missing = sum(len(v) for v in missing_files.values())
    pass_rate = (total_expected - total_missing) / total_expected * 100

    return {
        'total_expected': total_expected,
        'total_missing': total_missing,
        'missing_files': missing_files,
        'pass_rate': pass_rate
    }


def check_data_quality(race_ids):
    """Check data quality metrics."""

    quality_issues = []

    for race_id in race_ids:
        metrics_path = Path(f"data/processed/metrics/{race_id}_metrics.json")

        if not metrics_path.exists():
            continue

        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        # Check for issues
        issues = []

        if metrics.get('average_velocity_ms', 0) < 0.5:
            issues.append('velocity_too_low')
        if metrics.get('average_velocity_ms', 0) > 5.0:
            issues.append('velocity_too_high')
        if metrics.get('total_time_s', 0) < 4.5:
            issues.append('duration_too_short')
        if metrics.get('total_time_s', 0) > 15.0:
            issues.append('duration_too_long')

        if issues:
            quality_issues.append({
                'race_id': race_id,
                'issues': issues
            })

    pass_rate = (len(race_ids) - len(quality_issues)) / len(race_ids) * 100

    return {
        'total_checked': len(race_ids),
        'races_with_issues': len(quality_issues),
        'quality_issues': quality_issues,
        'pass_rate': pass_rate
    }


def check_performance_benchmarks():
    """Check performance against benchmarks."""

    # Load aggregated data
    df = pd.read_csv('data/processed/aggregated_metrics_reliable.csv')

    return {
        'total_races': len(df),
        'avg_velocity_ms': df['average_velocity_ms'].mean(),
        'avg_duration_s': df['total_time_s'].mean(),
        'world_record_comparison': {
            'men_wr': 5.00,
            'women_wr': 6.53,
            'fastest_in_dataset': df['total_time_s'].min()
        }
    }


if __name__ == "__main__":
    validate_pipeline()
```

**Deliverables**:
- `scripts/validate_pipeline_reliable.py` (NEW)
- `data/processed/pipeline_validation_report.json`

---

### Phase 7: Interactive Dashboard (2-3 hours)

**Goal**: Create HTML dashboard for visualization and analysis

**Use Existing Code**: May leverage Phase 3 visualization tools

**Create** `scripts/generate_dashboard_reliable.py`:

```python
"""
Interactive Dashboard Generator - Reliable Races
================================================
Generate comprehensive HTML dashboard with visualizations.

Features:
- Overall statistics
- Leaderboards
- Comparative charts
- Quality metrics
- Interactive plots (Plotly)
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

def generate_dashboard():
    """Generate interactive HTML dashboard."""

    # Load data
    df = pd.read_csv('data/processed/aggregated_metrics_reliable.csv')

    # Create HTML components
    html_parts = []

    # Header
    html_parts.append(create_header(df))

    # Statistics cards
    html_parts.append(create_stats_cards(df))

    # Leaderboard
    html_parts.append(create_leaderboard_section(df))

    # Velocity distribution
    html_parts.append(create_velocity_distribution(df))

    # Competition comparison
    html_parts.append(create_competition_comparison(df))

    # Quality metrics
    html_parts.append(create_quality_section())

    # Combine HTML
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speed Climbing Analytics - 114 Reliable Races</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {''.join(html_parts)}
        </div>
    </body>
    </html>
    """

    # Save dashboard
    output_path = Path('data/processed/dashboard_reliable_races.html')
    output_path.write_text(full_html, encoding='utf-8')

    print(f"✅ Dashboard generated: {output_path}")


def create_header(df):
    """Create dashboard header."""
    return f"""
    <h1>🏔️ Speed Climbing Performance Analytics</h1>
    <h3>Reliable Races Dataset ({len(df)} races)</h3>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <hr>
    """


def create_stats_cards(df):
    """Create statistics cards."""
    return f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Total Races</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{df['average_velocity_ms'].mean():.2f} m/s</div>
            <div class="stat-label">Avg Velocity</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{df['total_time_s'].mean():.2f}s</div>
            <div class="stat-label">Avg Duration</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{df['total_time_s'].min():.2f}s</div>
            <div class="stat-label">Fastest Time</div>
        </div>
    </div>
    """


def create_leaderboard_section(df):
    """Create leaderboard table."""
    top_10 = df.nlargest(10, 'average_velocity_ms')[
        ['race_id', 'competition', 'average_velocity_ms', 'total_time_s']
    ]

    table_html = top_10.to_html(index=False, classes='leaderboard-table')

    return f"""
    <h2>🏆 Top 10 Fastest Races</h2>
    {table_html}
    """


def create_velocity_distribution(df):
    """Create velocity distribution chart."""
    fig = px.histogram(
        df,
        x='average_velocity_ms',
        nbins=20,
        title='Velocity Distribution',
        labels={'average_velocity_ms': 'Average Velocity (m/s)'}
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_competition_comparison(df):
    """Create competition comparison chart."""
    fig = px.box(
        df,
        x='competition',
        y='average_velocity_ms',
        title='Velocity by Competition',
        labels={'average_velocity_ms': 'Average Velocity (m/s)', 'competition': 'Competition'}
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_quality_section():
    """Create quality metrics section."""
    # Load validation report
    with open('data/processed/pipeline_validation_report.json', 'r') as f:
        validation = json.load(f)

    return f"""
    <h2>📊 Data Quality Metrics</h2>
    <ul>
        <li>File Completeness: {validation['file_completeness']['pass_rate']:.1f}%</li>
        <li>Data Quality: {validation['data_quality']['pass_rate']:.1f}%</li>
        <li>Total Races Validated: {validation['total_races']}</li>
    </ul>
    """


if __name__ == "__main__":
    generate_dashboard()
```

**Deliverables**:
- `scripts/generate_dashboard_reliable.py` (NEW)
- `data/processed/dashboard_reliable_races.html`

---

## 📦 Final Deliverables Summary

### Scripts (NEW - 7 files)
1. `scripts/generate_reliable_races_list.py`
2. `scripts/batch_pose_extraction_reliable.py`
3. `scripts/batch_calibration_reliable.py`
4. `scripts/batch_metrics_reliable.py`
5. `scripts/aggregate_reliable_data.py`
6. `scripts/validate_pipeline_reliable.py`
7. `scripts/generate_dashboard_reliable.py`

### Data Outputs (116+ files)
1. `data/processed/reliable_races_list.json` (1 file)
2. `data/processed/poses/<race_id>_pose.json` (114 files)
3. `data/processed/calibration/<race_id>_calibration.json` (114 files)
4. `data/processed/metrics/<race_id>_metrics.json` (114 files)
5. `data/processed/aggregated_metrics_reliable.csv` (1 file)
6. `data/processed/leaderboard_*.csv` (6 files)
7. `data/processed/ml_*.npy` (4 files)
8. `data/processed/ml_feature_names.json` (1 file)

### Reports (6 files)
1. `data/processed/pose_extraction_report.json`
2. `data/processed/calibration_report.json`
3. `data/processed/metrics_calculation_report.json`
4. `data/processed/pipeline_validation_report.json`

### Dashboard (1 file)
- `data/processed/dashboard_reliable_races.html`

### Documentation
- Update `MASTER_CONTEXT.md` with pipeline status
- Create `docs/RELIABLE_DATA_PIPELINE_REPORT.md`

---

## ✅ Success Criteria

### Completion Criteria
- [ ] All 114 races have pose files
- [ ] All 114 races have calibration files
- [ ] All 114 races have metrics files
- [ ] Calibration RMSE < 1cm for 95%+ races
- [ ] Metrics velocity in 1-3 m/s range for 95%+ races
- [ ] Dashboard generated and viewable
- [ ] ML datasets exported with train/test splits

### Quality Criteria
- [ ] File completeness > 99%
- [ ] Data quality > 95%
- [ ] No negative or zero metrics
- [ ] Leaderboards generated for all competitions
- [ ] Validation report shows all checks passed

---

## 🚀 Execution Plan

### Session 1 (4-5 hours)
1. Generate reliable races list (30 min)
2. Batch pose extraction (3-4 hours)
   - Process 30-40 races per hour
   - Monitor success rate
3. Review pose extraction report

### Session 2 (3-4 hours)
1. Batch calibration (2-3 hours)
2. Review calibration quality
3. Start metrics calculation (if time)

### Session 3 (4-5 hours)
1. Complete metrics calculation (3-4 hours)
2. Data aggregation (1 hour)
3. Export ML datasets

### Session 4 (3-4 hours)
1. Pipeline validation (1-2 hours)
2. Dashboard generation (2 hours)
3. Documentation update
4. Final review

**Total Time**: 14-18 hours across 4 sessions

---

## 📊 Expected Outcomes

### Data Pipeline
- **Complete, validated pipeline** for 114 races
- **High-quality metrics** with calibration
- **ML-ready datasets** with train/test splits
- **Interactive dashboard** for analysis

### Business Value
- **Production-ready system** for future competitions
- **Baseline performance metrics** for 114 races
- **Quality-assured dataset** for ML model training
- **Extensible architecture** for adding 74 suspicious races later

### Next Steps After Completion
1. **Manual review** of 74 suspicious races using Phase 1.5 interface
2. **Add reviewed races** to pipeline (incremental processing)
3. **Train ML models** using 114-race dataset
4. **Deploy dashboard** for stakeholders

---

## 🎯 Why This Approach

### Strategic Benefits
1. **Don't wait for perfect data** - Process 114 reliable races now
2. **Validate entire pipeline** before adding problematic races
3. **Baseline metrics** for comparison after manual reviews
4. **ML development** can start immediately with clean data
5. **Incremental improvement** - Add 74 races as they're reviewed

### Technical Benefits
1. **Proven components** - Reuse Phase 3 tested code
2. **Quality assurance** - Multiple validation layers
3. **Reproducible** - Batch scripts for future competitions
4. **Scalable** - Easy to add more races incrementally

---

## 📝 Notes for Implementation

### Important Considerations
1. **Hardware**: Pose extraction is CPU/GPU intensive (3-4 hours for 114 races)
2. **Disk Space**: ~10GB for pose files, calibration, metrics
3. **Dependencies**: Ensure MediaPipe, OpenCV, scikit-learn installed
4. **Testing**: Run with 5 races first to validate pipeline

### Error Handling
- Each script should have try/except blocks
- Generate detailed error reports
- Continue processing even if individual races fail
- Save progress incrementally (don't lose work on crash)

### Optimization Opportunities
- Parallel processing for pose extraction (if multi-core)
- Batch calibration in groups
- Cache calibration results

---

**Ready to Build!** 🚀

Start with Session 1, validate each stage, and proceed systematically. The 114-race dataset will provide a solid foundation for all future work.
