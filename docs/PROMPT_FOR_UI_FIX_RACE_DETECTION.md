# Race Detection Issues - Fix Instructions for UI Claude

**Date**: 2025-11-16
**Priority**: URGENT - Blocking full pipeline
**Estimated Time**: 4-6 hours
**Complexity**: Medium-High

---

## 🚨 CRITICAL ISSUES DISCOVERED

During manual video review, **3 races with critical race detection errors** were discovered:

| Race | Competition | Detected Duration | Actual Duration | Error Factor | Impact |
|------|-------------|------------------|-----------------|--------------|---------|
| **Race001** | Chamonix 2024 | 1.77s (53 frames) | ~6.5s (~195 frames) | **3.7× underestimated** | 70% of race data lost! |
| **Race010** | Innsbruck 2024 | 12.00s (360 frames) | ~7.5s (~225 frames) | **1.6× overestimated** | Includes pre-race warmup |
| **Race023** | Zilina 2025 | 19.00s (570 frames) | ~6.6-7s (~198-210 frames) | **2.7× overestimated** | Includes false start/replay |

### Root Causes (Comprehensive Investigation Done)

1. **Motion-based Start Detection** - Too Sensitive:
   - Optical flow (Farneback) threshold too low (5.0)
   - Triggers on: camera movement, climber warmup, crowd motion
   - No multi-frame confirmation
   - File: `src/phase1_pose_estimation/race_start_detector.py:544`

2. **Visual Finish Detection** - Highly Unreliable:
   - Color change detection in top 15% of frame
   - Finish confidence always low (~0.40) for ALL races
   - False triggers: climber fall, camera angle change, lighting change
   - File: `src/phase1_pose_estimation/race_finish_detector.py:513`

3. **No Validation Layer**:
   - No duration sanity check (world record: 5.0s men, 6.53s women)
   - Single-frame noise can trigger detection
   - No diagnostic/debug information stored

### Why This Is Critical

- **Data Integrity**: 3/188 races (1.6%) are corrupted, possibly more undetected
- **ML Training**: Invalid data → biased models
- **Comparative Analysis**: These 3 races cannot be compared with others
- **Scientific Validity**: Entire detection pipeline questionable

---

## 📋 YOUR TASKS

### Task 1: Manual Correction of 3 Problematic Races (2 hours)

You need to manually review videos and correct the metadata for these 3 races.

#### Race001 (Chamonix 2024) - CRITICAL

**Current (WRONG)**:
```json
{
  "detected_start_frame": 11784,
  "detected_finish_frame": 11837,
  "race_duration": 53 frames (1.77s)
}
```

**Actual Timeline (from user's manual review)**:
- Video starts at 00:00
- Race starts: **00:01:17** (frame ~2310 if 30fps)
- Climber LEFT falls during race (system detected this as finish!)
- Climber RIGHT finishes: **00:07:47** (frame ~14610 if 30fps)
- Climber LEFT finishes later: **00:08:86** (frame ~16980 if 30fps)

**What to do**:
1. Open video: `data/race_segments/chamonix_2024/Speed_finals_Chamonix_2024_race001.mp4`
2. Verify actual start/finish times using video player
3. Calculate correct frame numbers (fps = 30)
4. Update `data/race_segments/chamonix_2024/Speed_finals_Chamonix_2024_race001_metadata.json`:
   ```json
   {
     "detected_start_frame": <correct_start>,
     "detected_finish_frame": <correct_finish_right_lane>,  // Use RIGHT lane finish
     "notes": "MANUAL CORRECTION: Original detection failed (climber fall). Updated based on manual review.",
     "manual_correction": true,
     "original_detected_duration": "1.77s (INVALID)"
   }
   ```

**Note**: Use RIGHT lane finish time since LEFT climber fell and finished later.

#### Race010 (Innsbruck 2024)

**Current (WRONG)**:
```json
{
  "detected_start_frame": 49920,
  "detected_finish_frame": 50280,
  "race_duration": 360 frames (12.00s)
}
```

**Actual Timeline**:
- Race ACTUALLY starts: **00:04:46** (frame ~8280 if 30fps)
- Climber LEFT finishes: **00:12:38** (frame ~22140 if 30fps)
- Climber RIGHT finishes: **00:12:43** (frame ~22290 if 30fps)
- Actual race duration: ~7.5s

**What to do**:
1. Open video: `data/race_segments/innsbruck_2024/Speed_finals_Innsbruck_2024_race010.mp4`
2. Find exact start time (when climbers actually leave ground, not warmup)
3. Find finish time (when hand hits button)
4. Update metadata with correct values
5. Add note explaining the fix

#### Race023 (Zilina 2025)

**Current (WRONG)**:
```json
{
  "detected_start_frame": 84510,
  "detected_finish_frame": 85080,
  "race_duration": 570 frames (19.00s)
}
```

**Actual Timeline**:
- Race ACTUALLY starts: **00:00:13** (frame ~390 if 30fps)
- Climber LEFT finishes: **00:06:73** (frame ~6030 if 30fps)
- Climber RIGHT reaches button but doesn't hit: **00:07:20** (frame ~6480 if 30fps)
- Actual race duration: ~6.6-7s

**What to do**:
1. Open video: `data/race_segments/zilina_2025/Speed_finals_Zilina_2025_race023.mp4`
2. Identify actual race start (not false start or replay)
3. Use LEFT climber finish time (RIGHT didn't complete)
4. Update metadata with correct values

---

### Task 2: Add Duration Validation (1 hour)

Create a validation script to flag suspicious races based on duration.

**Create**: `scripts/validate_race_durations.py`

```python
"""
Race Duration Validator
========================
Flags races with suspicious durations for manual review.

World Records (2024):
- Men: 5.00s
- Women: 6.53s

Reasonable range: 4.5s - 15s
(4.5s: slightly below WR, 15s: slower climbers + falls allowed)
"""

import json
from pathlib import Path
from typing import List, Dict
import sys

def validate_race_duration(metadata_path: Path) -> Dict:
    """Validate single race duration."""
    with open(metadata_path) as f:
        metadata = json.load(f)

    start = metadata['detected_start_frame']
    finish = metadata['detected_finish_frame']
    fps = 30.0  # Assume 30fps
    duration_seconds = (finish - start) / fps

    # Validation thresholds
    MIN_DURATION = 4.5  # Slightly below world record
    MAX_DURATION = 15.0  # Slower climbers + falls

    status = "OK"
    warnings = []

    if duration_seconds < MIN_DURATION:
        status = "SUSPICIOUS - TOO SHORT"
        warnings.append(f"Duration {duration_seconds:.2f}s < {MIN_DURATION}s (below world record!)")

    if duration_seconds > MAX_DURATION:
        status = "SUSPICIOUS - TOO LONG"
        warnings.append(f"Duration {duration_seconds:.2f}s > {MAX_DURATION}s (unusually slow)")

    return {
        'race_name': metadata_path.stem.replace('_metadata', ''),
        'duration_seconds': duration_seconds,
        'duration_frames': finish - start,
        'status': status,
        'warnings': warnings,
        'confidence_start': metadata.get('confidence_start', 'N/A'),
        'confidence_finish': metadata.get('confidence_finish', 'N/A')
    }

def validate_all_races() -> None:
    """Validate all 188 races."""
    metadata_dir = Path("data/race_segments")
    results = []

    for comp_dir in metadata_dir.iterdir():
        if not comp_dir.is_dir():
            continue

        for metadata_file in comp_dir.glob("*_metadata.json"):
            result = validate_race_duration(metadata_file)
            results.append(result)

    # Print summary
    total = len(results)
    ok = sum(1 for r in results if r['status'] == 'OK')
    suspicious = total - ok

    print(f"=" * 80)
    print(f"RACE DURATION VALIDATION SUMMARY")
    print(f"=" * 80)
    print(f"Total races: {total}")
    print(f"OK: {ok} ({ok/total*100:.1f}%)")
    print(f"Suspicious: {suspicious} ({suspicious/total*100:.1f}%)")
    print()

    # Print suspicious races
    if suspicious > 0:
        print(f"SUSPICIOUS RACES (need manual review):")
        print(f"-" * 80)
        for result in results:
            if result['status'] != 'OK':
                print(f"\n{result['race_name']}")
                print(f"  Duration: {result['duration_seconds']:.2f}s ({result['duration_frames']} frames)")
                print(f"  Status: {result['status']}")
                print(f"  Confidence: start={result['confidence_start']}, finish={result['confidence_finish']}")
                for warning in result['warnings']:
                    print(f"  ! {warning}")

    # Save report
    report_path = Path("data/processed/race_duration_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump({
            'total': total,
            'ok_count': ok,
            'suspicious_count': suspicious,
            'results': results
        }, f, indent=2)

    print(f"\n" + "=" * 80)
    print(f"Report saved to: {report_path}")
    print(f"=" * 80)

if __name__ == "__main__":
    validate_all_races()
```

**Run**:
```bash
python scripts/validate_race_durations.py
```

**Expected Output**:
- 3 suspicious races: Race001, Race010, Race023 (already known)
- Possibly more suspicious races → need manual review

---

### Task 3: Implement Enhanced Detection (Optional - 2-3 hours)

Improve detection system to prevent future errors.

#### 3.1: Add Multi-frame Confirmation

**File**: `src/phase1_pose_estimation/race_start_detector.py`

**Current**:
```python
# Line ~250
if motion_magnitude > self.motion_threshold:
    return frame_idx  # Single frame trigger!
```

**Improved**:
```python
# Add to __init__:
self.min_consecutive_frames = 5  # Require 5 consecutive frames

# Modify detection logic:
consecutive_count = 0
for frame_idx in range(start_frame, end_frame):
    motion_magnitude = self._calculate_motion(frame)

    if motion_magnitude > self.motion_threshold:
        consecutive_count += 1
        if consecutive_count >= self.min_consecutive_frames:
            # Confirmed detection
            return frame_idx - self.min_consecutive_frames + 1
    else:
        consecutive_count = 0  # Reset on drop
```

#### 3.2: Add Pose-based Finish Validation

**File**: `src/phase1_pose_estimation/race_finish_detector.py`

**Add new method**:
```python
def _validate_finish_with_pose(
    self,
    frame: np.ndarray,
    pose_data: Dict
) -> bool:
    """Validate finish using hand position (top 10% of frame)."""
    if not pose_data or 'left_wrist' not in pose_data:
        return False

    # Get hand position
    hand_y = pose_data['left_wrist']['y']  # or right_wrist
    frame_height = frame.shape[0]

    # Check if hand is in top 10%
    is_at_top = (hand_y / frame_height) < 0.1

    # Check if hand velocity is near zero (contact with button)
    hand_velocity = self._estimate_velocity(pose_data)
    is_stationary = hand_velocity < 1.0  # pixels/frame

    return is_at_top and is_stationary
```

**Integrate with visual detection**:
```python
# Combine both signals
visual_finish = self._detect_color_change(frame)
pose_finish = self._validate_finish_with_pose(frame, pose_data)

# Require BOTH to be true
if visual_finish and pose_finish:
    return True
```

#### 3.3: Add Duration Validation to Pipeline

**File**: `src/utils/race_segmenter.py`

**Add validation after detection**:
```python
def _validate_detection(
    self,
    start_frame: int,
    finish_frame: int,
    fps: float = 30.0
) -> bool:
    """Validate detected race duration."""
    duration_seconds = (finish_frame - start_frame) / fps

    MIN_DURATION = 4.5
    MAX_DURATION = 15.0

    if not (MIN_DURATION <= duration_seconds <= MAX_DURATION):
        logger.warning(
            f"Suspicious duration: {duration_seconds:.2f}s "
            f"(expected: {MIN_DURATION}-{MAX_DURATION}s)"
        )
        return False

    return True
```

---

### Task 4: Testing & Validation (1 hour)

#### 4.1: Re-generate Metrics for Corrected Races

After fixing metadata for Race001, Race010, Race023:

```bash
# Re-extract poses (if boundaries changed significantly)
python scripts/batch_extract_poses.py --competition chamonix_2024 --races race001
python scripts/batch_extract_poses.py --competition innsbruck_2024 --races race010
python scripts/batch_extract_poses.py --competition zilina_2025 --races race023

# Re-calculate metrics
python scripts/batch_calculate_metrics.py --competition chamonix_2024 --races race001
python scripts/batch_calculate_metrics.py --competition innsbruck_2024 --races race010
python scripts/batch_calculate_metrics.py --competition zilina_2025 --races race023
```

#### 4.2: Validation Checks

For each corrected race, verify:

1. **Duration is now reasonable**:
   ```python
   # Race001: should be ~6.5s (was 1.77s)
   # Race010: should be ~7.5s (was 12.00s)
   # Race023: should be ~6.6-7s (was 19.00s)
   ```

2. **Metrics are now valid**:
   ```python
   # Velocity should be in reasonable range (2-4 m/s)
   # Path length should be close to 15m (IFSC wall height)
   # No negative velocities or impossible accelerations
   ```

3. **Compare with similar races**:
   ```python
   # Find races with similar finish times
   # Metrics should be comparable
   ```

---

## 📊 SUCCESS CRITERIA

### Must-Have (Before Proceeding)

1. ✅ All 3 problematic races have corrected metadata
2. ✅ Duration validation script created and run
3. ✅ Validation report generated showing all suspicious races
4. ✅ Corrected races re-processed with new metrics
5. ✅ Metrics validated (durations ~6-7s, velocities 2-4 m/s)

### Nice-to-Have (Optional Improvements)

1. 🔄 Multi-frame confirmation implemented
2. 🔄 Pose-based finish validation added
3. 🔄 Duration validation integrated into pipeline
4. 🔄 Diagnostic dashboard for reviewing detection quality

---

## 🎯 DELIVERABLES

1. **Corrected Metadata Files** (3 files):
   - `data/race_segments/chamonix_2024/Speed_finals_Chamonix_2024_race001_metadata.json`
   - `data/race_segments/innsbruck_2024/Speed_finals_Innsbruck_2024_race010_metadata.json`
   - `data/race_segments/zilina_2025/Speed_finals_Zilina_2025_race023_metadata.json`

2. **Validation Script**:
   - `scripts/validate_race_durations.py`
   - `data/processed/race_duration_validation_report.json`

3. **Re-processed Metrics** (6 files):
   - `data/processed/metrics/chamonix_2024/Speed_finals_Chamonix_2024_race001_metrics_left.json`
   - `data/processed/metrics/chamonix_2024/Speed_finals_Chamonix_2024_race001_metrics_right.json`
   - (same for race010 and race023)

4. **Documentation Update**:
   - `MASTER_CONTEXT.md` already updated with findings
   - Add notes to README about known limitations

---

## ⚠️ IMPORTANT NOTES

1. **Don't Re-segment Videos**: Only update metadata JSON files. The race segment MP4 files are correct (they include buffer before/after).

2. **Frame Numbers vs Video Time**:
   - `detected_start_frame` / `detected_finish_frame` refer to frames in the ORIGINAL video
   - NOT frames in the race segment
   - Use buffer_before/buffer_after to calculate position in segment

3. **Manual Review Required**:
   - You MUST watch the videos to verify correct start/finish times
   - Don't just calculate from user's time estimates
   - Use video player with frame-by-frame seek

4. **World Records Reference**:
   - Men: 5.00s (Reza Alipour, 2023)
   - Women: 6.53s (Aleksandra Miroslaw, 2024)
   - Any race < 4.5s is suspicious (faster than WR!)
   - Any race > 15s is suspicious (very slow or includes non-race footage)

---

## 🔗 RELATED FILES

- Investigation report: `docs/RACE_DETECTION_INVESTIGATION_REPORT.md` (comprehensive analysis)
- Race detection code: `src/phase1_pose_estimation/race_start_detector.py`
- Race detection code: `src/phase1_pose_estimation/race_finish_detector.py`
- Manual segmenter: `src/utils/manual_race_segmenter.py`
- MASTER_CONTEXT.md: Updated with this issue

---

## 📞 QUESTIONS?

If anything is unclear, check:
1. MASTER_CONTEXT.md (comprehensive project overview)
2. Investigation report (detailed analysis of root causes)
3. Academic papers on sports timing detection (references in investigation report)

**Good luck! This is critical for data integrity.** 🚀
