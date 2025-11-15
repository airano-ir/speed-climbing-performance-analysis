# Prompt for UI claude.ai/code - Fix Critical Metrics Issues

**Date**: 2025-11-15
**Priority**: CRITICAL - Data Quality Fix
**Estimated Time**: 6-10 hours
**Deliverables**: 3 code modifications + 188 calibration files + validated metrics

---

## 🎯 Mission

You are tasked with fixing TWO CRITICAL issues discovered during Phase 3 testing that invalidate all current performance metrics for 188 speed climbing races.

**Current status**:
- ✅ Phase 3 implementation: COMPLETE
- 🧪 Testing: 5 sample races analyzed
- 🔴 **TWO CRITICAL BUGS FOUND** (affecting data validity)
- ⏸️ Full 188-race processing: ON HOLD until fixed

**Your goal**: Fix both issues, re-process all metrics, and validate results.

---

## 📋 Context & Background

### What was tested:
- Phase 3 Advanced Analytics pipeline
- Metrics calculation on 5 races from Chamonix 2024
- Aggregation, comparison, and leaderboard generation

### What was discovered:

**Issue 1: Calibration NOT being used** 🎯
- System implemented: ✅ (Phase 2.5 - `CameraCalibrator`, `PeriodicCalibrator`)
- System tested: ✅ (RMSE=0.1±0.2cm on 20 videos)
- **System deployed: ❌ (NO calibration files exist for 188 races)**
- **Result**: All metrics in PIXELS, not METERS
- **Impact**: Meaningless values (can't compare across races, invalid biomechanics)

**Issue 2: Pre/post-race frames NOT filtered** 📽️
- Race detection implemented: ✅ (`RaceStartDetector`, `RaceFinishDetector`)
- Race boundaries saved: ✅ (in metadata JSON files)
- **Boundaries used in metrics: ❌ (performance_metrics.py processes ALL frames)**
- **Result**: 143 frames processed instead of 53 race frames (2.7× error)
- **Impact**: Velocity underestimated by 2-3×, efficiency artificially low

### Combined impact:
```
Error from Issue 1: metrics in pixels × camera zoom changes
Error from Issue 2: includes non-race frames × 2-3
Combined error: 5-10× distortion!

Example:
Actual velocity: ~2.5 m/s
Current output: ~0.4 arbitrary pixel units
```

**Conclusion**: All 188 races need complete reprocessing after fixes.

---

## 🔧 Task 1: Fix Frame Selection (Priority 1 - 2-3h)

### Problem
`performance_metrics.py` processes ALL frames from pose JSON files, including:
- Pre-race frames (45): athletes standing, preparing
- Race frames (53): actual climbing ✅ ONLY THESE SHOULD BE ANALYZED
- Post-race frames (45): finished, celebrating

### Files to modify:
1. **`src/analysis/performance_metrics.py`** (primary fix)
2. **`scripts/batch_calculate_metrics.py`** (load race boundaries)

---

### Step 1.1: Modify `performance_metrics.py`

**Location**: `src/analysis/performance_metrics.py`

**Current signature** (line ~237):
```python
def analyze_pose_file(
    self,
    pose_json_path: Path,
    lane: str = 'left',
    min_visibility: float = 0.5,
    calibration_path: Optional[Path] = None
) -> Optional[PerformanceMetrics]:
```

**Add new parameters**:
```python
def analyze_pose_file(
    self,
    pose_json_path: Path,
    lane: str = 'left',
    min_visibility: float = 0.5,
    calibration_path: Optional[Path] = None,
    start_frame: Optional[int] = None,      # NEW: race start frame
    end_frame: Optional[int] = None,        # NEW: race end frame
    race_metadata: Optional[Dict] = None   # NEW: full metadata (optional)
) -> Optional[PerformanceMetrics]:
```

**Modify frame processing loop** (around line 289-322):
```python
# OLD CODE:
for frame in frames:
    climber_data = frame.get(climber_key)
    if not climber_data or not climber_data.get('keypoints'):
        continue
    # ... process frame

# NEW CODE:
for frame in frames:
    frame_id = frame.get('frame_id', 0)

    # Filter by race boundaries if provided
    if start_frame is not None and frame_id < start_frame:
        continue  # Skip pre-race frames
    if end_frame is not None and frame_id > end_frame:
        break  # Stop after race ends (optimization)

    climber_data = frame.get(climber_key)
    if not climber_data or not climber_data.get('keypoints'):
        continue
    # ... process frame (only race frames now)
```

**Add metadata to output** (so we know what was filtered):
```python
# In PerformanceMetrics dataclass or output JSON:
{
    "race_boundaries": {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "total_frames_in_file": len(frames),
        "frames_analyzed": count_of_frames_actually_processed,
        "frames_skipped_pre": count_skipped_before_start,
        "frames_skipped_post": count_skipped_after_end
    },
    # ... existing metrics
}
```

---

### Step 1.2: Modify `batch_calculate_metrics.py`

**Location**: `scripts/batch_calculate_metrics.py`

**Find the metrics calculation call** (around line 94-104):
```python
# Current code:
metrics = self.analyzer.analyze_pose_file(
    pose_file,
    lane=lane,
    calibration_path=cal_file if cal_file.exists() else None
)
```

**Load race metadata and extract boundaries**:
```python
# Load race metadata to get race boundaries
race_metadata_path = pose_file.parent.parent.parent / "race_segments" / competition / f"{race_name}_metadata.json"

start_frame = None
end_frame = None
race_metadata = None

if race_metadata_path.exists():
    with open(race_metadata_path, 'r') as f:
        race_metadata = json.load(f)

    # Extract race boundaries (in original video frame IDs)
    start_frame_orig = race_metadata.get('detected_start_frame', None)
    end_frame_orig = race_metadata.get('detected_finish_frame', None)

    # Convert to pose file frame IDs (which start from 0)
    # Pose file starts at `start_time` of the race segment
    # Frame 0 in pose file = frame `start_frame_orig` in original video
    # So we need to offset:
    video_start_frame = race_metadata.get('start_frame', 0)  # Where segment starts

    if start_frame_orig is not None:
        start_frame = start_frame_orig - video_start_frame
    if end_frame_orig is not None:
        end_frame = end_frame_orig - video_start_frame

    logger.info(f"  Race boundaries: start={start_frame}, end={end_frame} (pose file frame IDs)")
else:
    logger.warning(f"  Metadata not found: {race_metadata_path} - processing all frames")

# Call with race boundaries
metrics = self.analyzer.analyze_pose_file(
    pose_file,
    lane=lane,
    calibration_path=cal_file if cal_file.exists() else None,
    start_frame=start_frame,        # NEW
    end_frame=end_frame,            # NEW
    race_metadata=race_metadata     # NEW
)
```

---

### Step 1.3: Test with sample races

**Test script**:
```bash
# Re-run metrics on 5 sample races
python scripts/batch_calculate_metrics.py --max-races 5 --competition chamonix_2024 --force

# Check output
cat data/processed/metrics/chamonix_2024/Speed_finals_Chamonix_2024_race001_metrics_left.json
```

**Validation checks**:
1. ✅ `race_boundaries` field exists in output
2. ✅ `frames_analyzed` < `total_frames_in_file` (e.g., 53 vs 143)
3. ✅ `avg_vertical_velocity` is 2-3× HIGHER than before
4. ✅ `path_efficiency` is HIGHER (closer to realistic 0.6-0.9 range)

**Expected results**:
```json
// BEFORE (incorrect - all frames):
{
  "avg_vertical_velocity": 12.07,  // Too low
  "path_efficiency": 0.13          // Unrealistically low
}

// AFTER (correct - race frames only):
{
  "avg_vertical_velocity": 32.6,   // 2.7× higher ✅
  "path_efficiency": 0.35,         // More realistic ✅
  "race_boundaries": {
    "start_frame": 45,
    "end_frame": 98,
    "total_frames_in_file": 143,
    "frames_analyzed": 53
  }
}
```

---

## 🎯 Task 2: Batch Calibration (Priority 2 - 3-5h)

### Problem
No calibration files exist for 188 races, despite calibration system being implemented and tested.

### Solution
Create and run a batch calibration script using existing components.

---

### Step 2.1: Create `scripts/batch_calibration.py`

**New file**: `scripts/batch_calibration.py`

**Template** (adapt from existing scripts):
```python
"""
Batch Camera Calibration Script

Generates calibration files for all race segments using:
- HoldDetector (HSV-based red hold detection)
- PeriodicCalibrator (efficient per-30-frames calibration)
- IFSC route map (31 standard holds)

Usage:
    python scripts/batch_calibration.py
    python scripts/batch_calibration.py --competition chamonix_2024
    python scripts/batch_calibration.py --max-races 10 --test
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import List, Dict, Optional
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calibration.camera_calibration import PeriodicCalibrator
from phase1_pose_estimation.hold_detector import HoldDetector
from calibration.ifsc_route_map import IFSCRouteMap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchCalibrator:
    """Batch calibration for race segments."""

    def __init__(
        self,
        race_segments_dir: str = "data/race_segments",
        output_dir: str = "data/processed/calibration",
        ifsc_map_path: str = "configs/ifsc_route_coordinates.json"
    ):
        self.race_segments_dir = Path(race_segments_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load IFSC route map
        self.route_map = IFSCRouteMap.load_from_json(ifsc_map_path)
        logger.info(f"Loaded IFSC route map with {len(self.route_map.holds)} holds")

        # Hold detector
        self.hold_detector = HoldDetector(
            min_confidence=0.2,  # Lower threshold for better detection
            min_area=50
        )

    def calibrate_race(
        self,
        video_path: Path,
        output_path: Path
    ) -> Optional[Dict]:
        """Calibrate a single race video."""

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create periodic calibrator (every 30 frames = 1 second)
        calibrator = PeriodicCalibrator(
            route_map=self.route_map,
            hold_detector=self.hold_detector,
            recalibration_interval=30
        )

        # Process frames
        frame_count = 0
        calibrations = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calibrate (uses caching internally)
            result = calibrator.calibrate_frame(frame, frame_count)

            if result is not None:
                calibrations.append({
                    'frame_id': frame_count,
                    'rmse': result.rmse,
                    'holds_detected': result.holds_detected,
                    'holds_used': result.holds_used,
                    'inlier_ratio': result.inlier_ratio,
                    'confidence': result.confidence
                })

            frame_count += 1

            # Progress
            if frame_count % 30 == 0:
                logger.info(f"  Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

        cap.release()

        # Get final calibration result
        final_calibration = calibrator.get_current_calibration()

        if final_calibration is None:
            logger.error(f"  Calibration failed for {video_path.name}")
            return None

        # Save calibration
        final_calibration.save_to_json(output_path)

        # Statistics
        avg_rmse = sum(c['rmse'] for c in calibrations) / len(calibrations)
        avg_holds = sum(c['holds_used'] for c in calibrations) / len(calibrations)

        logger.info(f"  ✓ Calibration saved: {output_path.name}")
        logger.info(f"    RMSE: {avg_rmse:.2f}cm")
        logger.info(f"    Holds used (avg): {avg_holds:.1f}")

        return {
            'video_path': str(video_path),
            'output_path': str(output_path),
            'avg_rmse': avg_rmse,
            'avg_holds_used': avg_holds,
            'calibrations_performed': len(calibrations),
            'success': True
        }

    def run(
        self,
        competition: Optional[str] = None,
        max_races: Optional[int] = None
    ):
        """Run batch calibration."""

        # Find race videos
        if competition:
            pattern = f"{competition}/*.mp4"
        else:
            pattern = "*/*.mp4"

        race_videos = sorted(self.race_segments_dir.glob(pattern))

        if max_races:
            race_videos = race_videos[:max_races]

        logger.info(f"Found {len(race_videos)} race videos to calibrate")

        # Process each
        results = []
        start_time = time.time()

        for idx, video_path in enumerate(race_videos, 1):
            logger.info(f"\n[{idx}/{len(race_videos)}] {video_path.name}")

            # Output path
            competition_name = video_path.parent.name
            race_name = video_path.stem
            output_dir = self.output_dir / competition_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{race_name}_calibration.json"

            # Skip if exists
            if output_path.exists():
                logger.info(f"  Skipping (already calibrated): {output_path.name}")
                continue

            # Calibrate
            result = self.calibrate_race(video_path, output_path)
            if result:
                results.append(result)

        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("BATCH CALIBRATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total races: {len(race_videos)}")
        logger.info(f"Calibrated: {len(results)}")
        logger.info(f"Time: {elapsed/60:.1f} minutes")
        logger.info(f"Avg per race: {elapsed/len(results):.1f}s")

        # Save summary
        summary_path = self.output_dir / "batch_calibration_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': str(time.time()),
                'total': len(race_videos),
                'calibrated': len(results),
                'results': results
            }, f, indent=2)

        logger.info(f"\nSummary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch calibration for race segments")
    parser.add_argument('--competition', type=str, help="Specific competition (e.g., chamonix_2024)")
    parser.add_argument('--max-races', type=int, help="Limit number of races")
    parser.add_argument('--test', action='store_true', help="Test mode (max 5 races)")

    args = parser.parse_args()

    if args.test:
        args.max_races = 5

    calibrator = BatchCalibrator()
    calibrator.run(
        competition=args.competition,
        max_races=args.max_races
    )


if __name__ == "__main__":
    main()
```

---

### Step 2.2: Test calibration script

**Test with 5 races**:
```bash
python scripts/batch_calibration.py --test
```

**Check outputs**:
```bash
ls data/processed/calibration/chamonix_2024/
# Should see: Speed_finals_Chamonix_2024_race001_calibration.json, etc.

cat data/processed/calibration/chamonix_2024/Speed_finals_Chamonix_2024_race001_calibration.json
```

**Validate**:
1. ✅ JSON files created
2. ✅ RMSE < 10cm for 80%+ of races
3. ✅ holds_used >= 4 for most frames
4. ✅ No crashes or errors

---

### Step 2.3: Run full batch calibration

**Process all 188 races**:
```bash
# This will take 3-4 hours (1 minute per race × 188)
python scripts/batch_calibration.py
```

**Monitor progress**:
```bash
# In another terminal:
watch -n 10 'ls data/processed/calibration/*/*.json | wc -l'
# Should increase from 0 to 188
```

---

## 🔄 Task 3: Re-process Metrics (Priority 3 - 1-2h)

### Step 3.1: Delete old metrics

**Backup first**:
```bash
mv data/processed/metrics data/processed/metrics_old_INVALID
mkdir data/processed/metrics
```

### Step 3.2: Re-run batch metrics

**With calibration + frame filtering** (automatic now):
```bash
# This will take ~1 hour (batch_calculate_metrics.py already optimized)
python scripts/batch_calculate_metrics.py --force

# --force: overwrite existing files
# Automatically uses calibration files if they exist
# Automatically filters frames if metadata exists
```

### Step 3.3: Verify new metrics

**Check sample output**:
```bash
cat data/processed/metrics/chamonix_2024/Speed_finals_Chamonix_2024_race001_metrics_left.json
```

**Expected**:
```json
{
  "is_calibrated": true,           // ✅ Changed from false
  "units": "meters",                // ✅ Changed from pixels
  "avg_vertical_velocity": 2.34,   // ✅ In m/s (realistic!)
  "path_efficiency": 0.67,          // ✅ Realistic (was 0.13)
  "race_boundaries": {              // ✅ NEW field
    "start_frame": 45,
    "end_frame": 98,
    "frames_analyzed": 53
  }
}
```

---

## ✅ Task 4: Validation & Comparison (1h)

### Step 4.1: Compare old vs new metrics

**Create comparison script**:
```python
# scripts/compare_old_new_metrics.py
import json
import pandas as pd
from pathlib import Path

old_dir = Path("data/processed/metrics_old_INVALID")
new_dir = Path("data/processed/metrics")

comparisons = []

for new_file in new_dir.glob("*/*_metrics_left.json"):
    old_file = old_dir / new_file.relative_to(new_dir)

    if not old_file.exists():
        continue

    with open(old_file) as f:
        old = json.load(f)
    with open(new_file) as f:
        new = json.load(f)

    comparisons.append({
        'race': new_file.stem,
        'old_velocity': old['summary']['avg_vertical_velocity'],
        'new_velocity': new['summary']['avg_vertical_velocity'],
        'velocity_ratio': new['summary']['avg_vertical_velocity'] / old['summary']['avg_vertical_velocity'],
        'old_efficiency': old['summary']['path_efficiency'],
        'new_efficiency': new['summary']['path_efficiency'],
        'old_units': old.get('units', 'pixels'),
        'new_units': new.get('units', 'pixels'),
        'old_calibrated': old.get('is_calibrated', False),
        'new_calibrated': new.get('is_calibrated', False)
    })

df = pd.DataFrame(comparisons)
print(df.describe())
df.to_csv("data/processed/metrics_comparison.csv", index=False)
```

**Run**:
```bash
python scripts/compare_old_new_metrics.py
```

**Expected results**:
```
velocity_ratio:
  mean: 8.5    (new metrics are 8.5× higher on average)
  min:  4.2    (at least 4× improvement)
  max:  15.3   (some races had extreme underestimation)

new_calibrated:
  True: 188    (100% calibrated now)
  False: 0

new_units:
  meters: 188  (100% in meters now)
  pixels: 0
```

---

### Step 4.2: Sanity checks

**Realistic velocity range**:
```python
# World record: ~2.67 m/s (5.0s for 15m wall)
# Elite climbers: 2.0-2.8 m/s
# Amateur: 1.0-1.8 m/s

# Check all velocities are in realistic range
for metrics_file in Path("data/processed/metrics").glob("*/*_metrics_*.json"):
    with open(metrics_file) as f:
        data = json.load(f)

    v = data['summary']['avg_vertical_velocity']

    if v < 0.5 or v > 5.0:
        print(f"OUTLIER: {metrics_file.name} - velocity = {v:.2f} m/s")
```

**Realistic path efficiency**:
```python
# Typical path efficiency: 0.6-0.9 (60-90%)
# Below 0.5: very inefficient or error
# Above 0.95: suspiciously perfect

for metrics_file in Path("data/processed/metrics").glob("*/*_metrics_*.json"):
    with open(metrics_file) as f:
        data = json.load(f)

    eff = data['summary']['path_efficiency']

    if eff < 0.4 or eff > 0.98:
        print(f"OUTLIER: {metrics_file.name} - efficiency = {eff:.2f}")
```

---

## 📦 Task 5: Update Downstream Outputs (1h)

### Step 5.1: Re-run aggregations

**Delete old**:
```bash
rm -rf data/processed/aggregates/*
```

**Regenerate**:
```bash
python scripts/aggregate_competition_stats.py
```

**Verify**: Leaderboard now shows realistic velocities (m/s, not px/s)

---

### Step 5.2: Re-run comparisons

```bash
# Re-compare all races (now with valid metrics)
python scripts/compare_races.py --all --competition all
```

---

### Step 5.3: Update visualizations (if time permits)

```bash
python src/visualization/race_plots.py
python scripts/generate_html_dashboard.py
```

---

## 🎯 Success Criteria

### Definition of Done:

**Priority 1 - Frame Selection**:
- [x] `performance_metrics.py` accepts `start_frame`/`end_frame`
- [x] `batch_calculate_metrics.py` loads race boundaries from metadata
- [x] Test with 5 races: velocity 2-3× higher
- [x] Output includes `race_boundaries` field

**Priority 2 - Calibration**:
- [x] `scripts/batch_calibration.py` created and tested
- [x] 188 calibration JSON files generated
- [x] 90%+ of races have RMSE < 10cm
- [x] All calibrations use PeriodicCalibrator (efficient)

**Priority 3 - Reprocessing**:
- [x] All 188 metrics re-calculated
- [x] 100% have `is_calibrated: true`
- [x] 100% have `units: "meters"`
- [x] Velocities in realistic range (0.5-5.0 m/s)
- [x] Efficiencies in realistic range (0.4-0.98)

**Priority 4 - Validation**:
- [x] Comparison CSV generated (old vs new)
- [x] Mean improvement ratio: 5-10×
- [x] No critical outliers
- [x] Aggregations updated
- [x] Leaderboard shows m/s not px/s

---

## 📝 Deliverables

**Code changes**:
1. `src/analysis/performance_metrics.py` - modified (frame filtering)
2. `scripts/batch_calculate_metrics.py` - modified (load boundaries)
3. `scripts/batch_calibration.py` - NEW file (188 calibrations)
4. `scripts/compare_old_new_metrics.py` - NEW file (validation)

**Data outputs**:
- `data/processed/calibration/{competition}/{race}_calibration.json` (188 files)
- `data/processed/metrics/{competition}/{race}_metrics_{lane}.json` (376 files, regenerated)
- `data/processed/aggregates/*` (regenerated)
- `data/processed/metrics_comparison.csv` (validation report)

**Commits**:
```bash
# Commit 1: Frame selection fix
git add src/analysis/performance_metrics.py scripts/batch_calculate_metrics.py
git commit -m "fix: add frame filtering to metrics calculation

- performance_metrics.py now accepts start_frame/end_frame
- batch_calculate_metrics.py loads race boundaries from metadata
- Filters out pre/post-race frames
- Includes race_boundaries in output JSON

Impact: 2-3× improvement in velocity accuracy"

# Commit 2: Batch calibration
git add scripts/batch_calibration.py data/processed/calibration/
git commit -m "feat: batch calibration for all 188 races

- New batch_calibration.py script
- Uses PeriodicCalibrator for efficiency
- Generated 188 calibration JSON files
- Average RMSE: X.XX cm (validate after run)

Impact: Enables meter-based metrics"

# Commit 3: Regenerated metrics
git add data/processed/metrics/ data/processed/aggregates/
git commit -m "data: regenerate all metrics with calibration + frame filtering

- 188 races reprocessed
- All metrics now calibrated (meters, not pixels)
- Frame boundaries applied (race period only)
- 5-10× improvement in accuracy

Breaking change: Previous metrics were invalid"
```

---

## ⏰ Estimated Timeline

| Task | Time | Cumulative |
|------|------|------------|
| 1. Frame Selection Fix | 2-3h | 3h |
| 2. Batch Calibration Script | 1h | 4h |
| 3. Run Calibration (188 races) | 3h | 7h |
| 4. Re-run Metrics | 1h | 8h |
| 5. Validation & Comparison | 1h | 9h |
| 6. Update Aggregations | 1h | 10h |

**Total: 6-10 hours** (can be parallelized where possible)

---

## 🆘 If You Get Stuck

### Common issues:

**Frame ID mismatches**:
- Pose file frame IDs start from 0
- Metadata frame IDs are from original video
- Need to offset: `pose_frame = metadata_frame - segment_start_frame`

**Calibration failures**:
- Not enough holds detected (< 4) → lower `min_confidence` to 0.15
- RMSE too high → check if camera is moving (use PeriodicCalibrator)
- Video won't open → check path, try absolute paths

**Memory issues**:
- Process in batches (e.g., 50 races at a time)
- Use `--max-races` argument

### Where to find help:
- `docs/SESSION_LOG_PHASE3_TEST.md` - detailed analysis of issues
- `docs/TUTORIAL_METRICS_CALIBRATION_FA.md` - comprehensive guide
- `MASTER_CONTEXT.md` - project context and history

---

## 🎯 Final Notes

**This is CRITICAL work**:
- Without these fixes, all metrics are scientifically invalid
- Cannot publish, cannot use for athlete coaching
- Cannot proceed with Phase 4 (ML) on bad data

**You are the blocker resolver**:
- Phase 3 is stuck until you fix this
- 188 races are waiting
- The entire project's data quality depends on you

**Take your time, be thorough**:
- Test extensively before full batch run
- Validate at each step
- Document any issues you encounter
- Ask questions if anything is unclear

**Good luck!** 🚀

---

**END OF PROMPT**
