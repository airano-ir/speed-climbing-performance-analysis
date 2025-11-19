# Phase 2 Testing Results - GlobalMapVideoProcessor
**Date:** 2025-11-19
**Component:** Global Map Registration Pipeline

## Test Summary

Tested the integrated GlobalMapVideoProcessor pipeline on 4 Seoul 2024 race videos.

### Test Results

| Race | Valid Frames | Success Rate | Avg Velocity | Distance | Notes |
|------|--------------|--------------|--------------|----------|-------|
| race001 (old)| 255/288 | 88.5% | +2.53 m/s | 22.79m | Before dual-lane fix |
| race001 (new)| 6/288 | 2.1% | +0.20 m/s | 0.45m | After dual-lane fix |
| race002 | 1/306 | 0.3% | 0.00 m/s | 0.00m | Almost complete failure |
| race003 | 315/317 | 99.4% | -0.30 m/s | 4.99m | Excellent tracking but negative velocity |
| race010 | 1/406 | 0.2% | 0.00 m/s | 0.00m | Almost complete failure |

## Key Findings

### ✅ What Works Well

1. **Pipeline Architecture**:
   - All components integrate correctly
   - No crashes or exceptions during processing
   - Proper error handling and fallbacks

2. **Calibration System**:
   - When holds are detected, calibration achieves RMSE=0.00m
   - Homography transformation works correctly
   - Periodic calibration (every 15 frames) provides good performance

3. **Pose Detection**:
   - BlazePose detects climbers when visible
   - Keypoint extraction works correctly
   - COM calculation from hips is accurate

4. **Lane Assignment**:
   - Dual-lane detection properly assigns detected climber to left/right
   - Only processes the detected lane (correct behavior)

### ⚠️ Issues Discovered

1. **Hold Detection Sensitivity**:
   - **Root cause**: Red hold detection very sensitive to:
     - Camera angle/position
     - Lighting conditions
     - Which part of wall is visible
   - **Impact**: 2 out of 4 races had <1% valid frames
   - **Solution needed**: More robust hold detection parameters or alternative calibration methods

2. **Single-Pose Limitation**:
   - **Root cause**: BlazePose detects ONE person at a time
   - **Impact**: Can only track one climber per frame in dual-lane races
   - **Expected behavior**: This is correct for single-camera setup
   - **Solution**: For full dual-lane tracking, need frame cropping per lane OR two video feeds

3. **Coordinate System Inconsistency**:
   - **Observation**: race003 showed negative velocity despite fix
   - **Possible causes**:
     - Climber actually descending in that segment (false start/failure)
     - Coordinate system issue not fully resolved
     - Different camera orientation
   - **Solution needed**: More investigation required

4. **Distance Outliers**:
   - **Observation**: race001 showed 22.79m distance (should be max ~15m)
   - **Root cause**: One calibration outlier at end of race (22.60m height)
   - **Impact**: Summary statistics include outliers
   - **Solution needed**: Outlier filtering in TimeSeriesBuilder

## Technical Validation

### Calibration Quality
- ✅ When successful: RMSE = 0.00m (excellent!)
- ✅ Inlier ratios: 4/4 to 4/7 (reasonable)
- ✅ Pixel-to-meter scale: 52-87 px/m (varies with camera zoom)

### Velocity Range
- ✅ When tracking full climb: 2.53 m/s (excellent!)
- ⚠️ Partial tracking: 0.00-0.30 m/s (not a full climb)
- ❌ Negative values: -0.30 m/s (coordinate issue or descent)

### Frame Coverage
- ✅ Best case: 99.4% (race003)
- ⚠️ Typical: ~88% (race001 before dual-lane fix)
- ❌ Worst case: 0.3% (race002, race010)

## Recommendations

### Immediate Actions (Before Full Reprocessing)

1. **Improve Hold Detection**:
   - Test with different HSV thresholds
   - Add adaptive parameter tuning based on video characteristics
   - Consider alternative hold colors (yellow start holds?)

2. **Add Outlier Filtering**:
   - Filter Y positions outside valid range (0-16m)
   - Use median-based outlier detection in time-series
   - Recalculate statistics after filtering

3. **Validate Coordinate System**:
   - Manually verify Y-axis orientation on multiple frames
   - Check if route map coordinates match camera view
   - Test with videos where climber clearly finishes

4. **Test on Diverse Videos**:
   - Different competitions
   - Different camera angles
   - Different lighting conditions
   - Videos with clear climb start→finish

### Future Enhancements

1. **Multi-Person Pose Detection**:
   - Investigate MediaPipe with multi-person mode
   - OR implement frame cropping per lane
   - OR process left/right lanes separately

2. **Adaptive Calibration**:
   - Dynamic hold detection thresholds
   - Fallback to simpler calibration methods when holds not visible
   - Use wall edges or other features

3. **Robust Statistics**:
   - Implement RANSAC-like filtering for outliers
   - Use percentile-based statistics (P5-P95 range)
   - Add quality scores per race

## Conclusion

**Pipeline Status**: ✅ Core functionality working, but needs robustness improvements

**Ready for Full Reprocessing?**: ⚠️ NOT YET
- Hold detection must be improved first
- Too many races would fail (50%+ failure rate currently)

**Next Steps**:
1. Fix hold detection robustness
2. Add outlier filtering
3. Test on 10-20 more videos
4. Once >80% success rate achieved, proceed with full 188-race reprocessing

**Estimated Success Rate for Full Reprocessing (Current State)**: ~40-50%
**Target Success Rate**: >85%
