# Manual Review Interface - User Guide

**Version**: 1.0
**Last Updated**: 2025-11-16 (To be updated after UI build)
**Status**: 📝 PLACEHOLDER - Will be completed by UI Claude

---

## Quick Start

**To launch the interface**:
```bash
cd "g:\My Drive\Projects\Speed Climbing Performance Analysis"
streamlit run scripts/review_interface/app.py
```

**Browser will open automatically** at `http://localhost:8501`

---

## Overview

The Manual Review Interface is a Streamlit-based tool for manually correcting race detection errors in speed climbing videos.

**Purpose**:
- Review and correct 74 suspicious races with detection errors
- Permanent tool for future competitions
- Extensible for ML suggestions and batch processing

---

## Interface Sections

### 1. Sidebar - Statistics & Filters

**Progress Statistics**:
- Total races: 74
- Completed: X / 74
- Pending: Y / 74
- Critical races: 5

**Filters**:
- Competition (All, Chamonix 2024, Innsbruck 2024, Zilina 2025, etc.)
- Priority (All, Critical, High, Medium, Low)
- Status (Pending Only, All, Completed, Skipped)

### 2. Main Area - Race Review

**Race Selection**:
- Dropdown list of filtered races
- Shows: Priority, Race ID, Duration, Issue description

**Race Information**:
- Race ID, Competition, Priority, Status
- Detection issue description
- Current detected duration and confidence scores
- Correction status (if already corrected)

**Video Player**:
- Frame-by-frame navigation
- Slider for quick seek
- Buttons: -30, -1, +1, +30 frames, Reset
- Displays current frame, time, frame number

**Correction Interface**:
- Mark current frame as START or FINISH
- Input fields for start/finish frames
- Real-time validation (duration check)
- Correction reason (required)
- Reviewer notes (optional)
- Save / Skip / Reset buttons

---

## Workflow

### Step 1: Filter Races

1. In sidebar, select filters:
   - Start with "Critical (1)" priority
   - Status: "Pending Only"

### Step 2: Select Race

1. Choose race from dropdown
2. Review issue description and detected duration

### Step 3: Watch Video

1. Use video player to find actual race start and finish
2. Navigation tips:
   - Use slider for rough position
   - Use ±30 buttons to get close
   - Use ±1 buttons for frame-perfect marking

**How to identify race start**:
- Frame where climber's feet leave the ground
- NOT warmup movements
- NOT false starts

**How to identify race finish**:
- Frame where climber's hand hits the finish button
- Look for hand in top 10% of frame
- Hand should stop moving (contact with button)

### Step 4: Mark Boundaries

1. Navigate to race start frame
2. Click "Mark Current as START"
3. Navigate to race finish frame
4. Click "Mark Current as FINISH"

**Validation**:
- Duration must be 4.5s - 15s
- Interface shows validation status

### Step 5: Document Correction

1. Enter correction reason (required):
   - "Climber fall detected as finish"
   - "Pre-race warmup included"
   - "False start or replay included"
   - "Camera motion triggered start detection"

2. Add reviewer notes (optional):
   - Any additional context
   - Observations about the race
   - Suggestions for future improvements

### Step 6: Save or Skip

**Save Correction**:
- Click "Save Correction"
- Metadata will be updated with backup
- Progress tracker will be marked as "Completed"
- Move to next race

**Skip Race**:
- If uncertain or video quality issues
- Click "Skip Race"
- Can return to it later

**Reset**:
- Clear marked start/finish frames
- Start over

---

## Tips & Best Practices

### Video Navigation

- **Rough positioning**: Use slider to jump to approximate time
- **Fine-tuning**: Use ±30 buttons to get within a few frames
- **Precision**: Use ±1 buttons for frame-perfect selection
- **Verification**: After marking, navigate away and back to verify

### Common Issues

**1. Climber fall detected as finish** (Race001 example):
- Find frame where RIGHT lane finishes (if left fell)
- Or vice versa
- Use the lane that completed successfully

**2. Pre-race warmup included**:
- Look for moment feet leave ground
- Ignore earlier movements, camera adjustments
- Start = actual climb begins

**3. False start or replay**:
- Find the SECOND start (after false start reset)
- Duration should be reasonable (4.5-15s)

**4. Multiple climbs in video**:
- Choose the primary competition climb
- Ignore practice runs or replays

### Validation Rules

**Duration Range**:
- Minimum: 4.5s (slightly below world record)
- Maximum: 15s (slower climbers + falls)

**World Records** (for reference):
- Men: 5.00s
- Women: 6.53s

**Critical Warnings**:
- < 3s: Impossible! Check again
- > 20s: Includes non-race footage

---

## Keyboard Shortcuts

*(To be implemented in future versions)*

- Arrow keys: Frame navigation
- Space: Play/pause
- Enter: Confirm correction
- Esc: Cancel/reset

---

## Troubleshooting

### Video not loading
- Check file path in config
- Verify video file exists in `data/race_segments/`
- Check file permissions

### Cannot save correction
- Ensure correction reason is filled
- Check duration is in valid range (4.5-15s)
- Verify start < finish

### Progress not updating
- Check CSV file permissions
- Restart Streamlit app
- Clear browser cache

---

## Advanced Features

*(Future enhancements - not yet implemented)*

- **ML Suggestions**: AI-powered start/finish detection
- **Batch Review**: Review multiple races sequentially
- **Comparison View**: Compare with similar races
- **Pose Overlay**: Show detected keypoints on video
- **Audio Analysis**: Starting beep detection
- **Collaborative Review**: Multi-user support

---

## Support

For issues or questions:
1. Check [MASTER_CONTEXT.md](../MASTER_CONTEXT.md) for project overview
2. Review [developer guide](manual_review_interface_developer_guide.md) for technical details
3. See validation report: `data/processed/race_duration_validation_report.json`

---

**Note**: This guide will be expanded after UI implementation is complete.
