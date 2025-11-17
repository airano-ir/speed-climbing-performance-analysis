# Task: Fix Velocity Calculation in Reliable Pipeline

## Context

The reliable data pipeline (Phases 1-7) has been completed for 114 races. All phases executed successfully, but there is a **critical issue** with velocity calculation in Phase 4.

**Problem:** Average velocity is ~1000x too low (0.001 m/s instead of expected 2-2.5 m/s for elite climbers)

**Root Cause:** Pose landmarks from MediaPipe are in **normalized coordinates** (0-1 range), but the metrics calculation multiplies them directly by pixel-to-meter scale without first converting to pixel coordinates using frame dimensions.

## Current Incorrect Calculation

In `scripts/batch_metrics_reliable.py` lines 187-191:

```python
# WRONG: Using normalized coordinates directly
vertical_displacement_px = com_array[0, 1] - com_array[-1, 1]  # This is in 0-1 range!
vertical_displacement_m = abs(vertical_displacement_px * pixel_to_meter)
```

For example:
- Start COM y: 0.8 (normalized)
- End COM y: 0.2 (normalized)
- Displacement: 0.6 (normalized, dimensionless)
- Current calculation: 0.6 × 0.025 = 0.015 m (WRONG!)

## Required Fix

### Step 1: Add Frame Dimensions to Pose Extraction

Modify `scripts/batch_pose_extraction_reliable.py` around line 90-120 (where pose data is saved):

```python
# After extracting poses, add frame dimensions to output
pose_output = {
    'race_id': race_id,
    'competition': metadata.get('competition', 'unknown'),
    'frames': frames_data,
    'total_frames': finish_frame - start_frame + 1,
    'extracted_frames': len(frames_data),
    'fps': fps,
    'frame_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),  # ADD THIS
    'frame_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), # ADD THIS
    'extraction_date': datetime.now().isoformat()
}
```

### Step 2: Update Metrics Calculation

Modify `scripts/batch_metrics_reliable.py` lines 150-191:

```python
def calculate_simple_metrics(pose_data, calibration, metadata):
    """Calculate basic performance metrics from pose data."""

    frames = pose_data.get('frames', [])
    if len(frames) < 2:
        return None

    fps = pose_data.get('fps', 30.0)
    pixel_to_meter = calibration.get('pixel_to_meter_scale', 0.025)

    # GET FRAME DIMENSIONS
    frame_width = pose_data.get('frame_width', 1920)  # ADD THIS
    frame_height = pose_data.get('frame_height', 1080) # ADD THIS

    # Extract COM positions (simple average of visible landmarks)
    com_positions = []
    timestamps = []

    for i, frame in enumerate(frames):
        landmarks = frame.get('landmarks', [])
        if len(landmarks) < 10:
            continue

        # Calculate COM as average of visible landmarks
        x_sum, y_sum, count = 0, 0, 0
        for lm in landmarks:
            if lm.get('visibility', 0) > 0.5:
                x_sum += lm['x']
                y_sum += lm['y']
                count += 1

        if count > 0:
            com_x = x_sum / count
            com_y = y_sum / count
            com_positions.append((com_x, com_y))
            timestamps.append(i / fps)

    if len(com_positions) < 2:
        return None

    com_array = np.array(com_positions)

    # FIXED: Convert normalized coords to pixels BEFORE applying calibration
    # Vertical movement (upward is negative y in image coords)
    vertical_displacement_normalized = com_array[0, 1] - com_array[-1, 1]  # Start y - End y
    vertical_displacement_px = vertical_displacement_normalized * frame_height  # CONVERT TO PIXELS
    vertical_displacement_m = abs(vertical_displacement_px * pixel_to_meter)

    # Total time
    total_time_s = timestamps[-1] - timestamps[0]

    # Average velocity
    if total_time_s > 0:
        avg_velocity_ms = vertical_displacement_m / total_time_s
    else:
        avg_velocity_ms = 0

    # FIXED: Path length calculation
    path_length_px = 0
    for i in range(1, len(com_array)):
        dx = (com_array[i, 0] - com_array[i-1, 0]) * frame_width  # CONVERT TO PIXELS
        dy = (com_array[i, 1] - com_array[i-1, 1]) * frame_height # CONVERT TO PIXELS
        path_length_px += np.sqrt(dx**2 + dy**2)

    path_length_m = path_length_px * pixel_to_meter

    # FIXED: Efficiency calculation
    straight_distance_px = np.sqrt(
        ((com_array[-1, 0] - com_array[0, 0]) * frame_width)**2 +
        ((com_array[-1, 1] - com_array[0, 1]) * frame_height)**2
    )
    straight_distance_m = straight_distance_px * pixel_to_meter

    efficiency = straight_distance_m / path_length_m if path_length_m > 0 else 0

    # FIXED: Smoothness calculation
    velocities = []
    for i in range(1, len(com_array)):
        dt = timestamps[i] - timestamps[i-1]
        if dt > 0:
            dy = abs((com_array[i-1, 1] - com_array[i, 1]) * frame_height) * pixel_to_meter
            v = dy / dt
            velocities.append(v)

    smoothness_score = 1.0 / (1.0 + np.std(velocities)) if velocities else 0

    # Rest of metrics calculation remains the same...
    return metrics
```

### Step 3: Re-run Phase 2 and Phase 4

After making the changes:

```bash
# Re-run Phase 2 to add frame dimensions
python scripts/batch_pose_extraction_reliable.py

# Re-run Phase 4 to recalculate metrics with correct values
python scripts/batch_metrics_reliable.py

# Re-run Phase 5 to update aggregated data
python scripts/aggregate_reliable_data.py

# Re-run Phase 6 to validate
python scripts/validate_pipeline_reliable.py
```

## Expected Results After Fix

With correct calculation (example):
- Start COM y: 0.8 (normalized) × 1080 px = 864 px
- End COM y: 0.2 (normalized) × 1080 px = 216 px
- Displacement: 648 px × 0.025 m/px = **16.2 m** ✓
- Time: ~7 seconds
- Velocity: 16.2 m / 7 s = **2.31 m/s** ✓ (correct for elite climbers!)

**Validation:**
- Average velocity should be 0.5-3.0 m/s (elite climbers: 2.0-2.5 m/s)
- Currently: 0.001 m/s (WRONG)
- After fix: ~2.3 m/s (CORRECT)

## Files to Modify

1. `scripts/batch_pose_extraction_reliable.py` - Add frame dimensions to output
2. `scripts/batch_metrics_reliable.py` - Fix all calculations using normalized coordinates

## Verification

After fix, check `data/processed/metrics_calculation_report.json`:
- `avg_velocity_ms` should be around 2.0-2.5 (not 0.001)
- `valid_velocity_count` should be close to 114 (not 0)

## Commit Message Template

```
fix: Correct velocity calculation using frame dimensions

Problem: Velocity values were ~1000x too low (0.001 m/s vs 2.3 m/s)

Root cause: MediaPipe landmarks are in normalized coords (0-1), but
calculations multiplied them directly by pixel-to-meter scale without
first converting to pixel coordinates.

Solution:
- Added frame_width and frame_height to pose extraction output
- Updated all metrics calculations to convert normalized coords to
  pixels before applying calibration scale

Results:
- Average velocity: 2.3 m/s (was 0.001 m/s) ✓
- Valid velocity count: 113/114 (was 0/114) ✓
- All metrics now physically accurate

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Notes

- This is a **data quality fix**, not a new feature
- Pipeline structure remains the same
- Only Phases 2, 4, 5, 6 need to be re-run
- Estimated time: 3-5 minutes total
- Phase 3 (calibration) does NOT need to be re-run

## Priority

**HIGH** - This must be fixed before ML modeling, as all velocity-based features are currently incorrect.
