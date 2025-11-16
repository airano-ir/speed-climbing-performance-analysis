# Manual Review Templates

This directory contains templates for extending the manual review interface to handle new competitions and track corrections.

## Files

### 1. `new_competition_template.yaml`
**Purpose**: Template for adding new competitions to the review interface.

**Usage**:
1. Copy this file
2. Replace all `<PLACEHOLDER>` values with actual competition data
3. Add the modified entry to `configs/manual_review_config.yaml` under `competitions:`
4. Run validation to identify suspicious races

**Time to add new competition**: ~5 minutes

**Example**:
```yaml
paris_2026:
  name: "Speed Finals Paris 2026"
  date: "2026-07-15"
  video_format: "mp4"
  fps: 60.0
  race_segments_path: "data/race_segments/paris_2026"
  total_races: 32
```

---

### 2. `corrected_metadata_template.json`
**Purpose**: Template showing the format for manually corrected race metadata.

**Usage**:
1. Use this as reference when correcting race metadata files
2. Always preserve original values in `correction_metadata` section
3. Set `manual_correction: true` flag
4. Document reason for correction

**Key Fields**:
- `detected_start_frame` / `detected_finish_frame`: Corrected values
- `manual_correction`: Must be `true`
- `correction_metadata.original_*`: Preserve original wrong values
- `correction_metadata.correction_reason`: Why was it wrong?

**Real Example**: See `data/race_segments/chamonix_2024/Speed_finals_Chamonix_2024_race001_metadata.json`

---

### 3. `review_history_template.json`
**Purpose**: Template for tracking all manual reviews and corrections.

**Usage**:
- The interface automatically populates `data/manual_review/review_history.json`
- This template shows the expected format
- Used for audit trail and generating reports

**Tracked Information**:
- Original vs corrected values
- Issue type (climber_fall, false_start, etc.)
- Validation results
- Reviewer notes

---

## Adding a New Competition (Step-by-Step)

1. **Prepare video files**:
   ```
   data/race_segments/new_competition_YYYY/
   ├── Speed_finals_NewCompetition_YYYY_race001.mp4
   ├── Speed_finals_NewCompetition_YYYY_race002.mp4
   └── ...
   ```

2. **Copy and modify template**:
   - Copy `new_competition_template.yaml`
   - Update all values
   - Add to `configs/manual_review_config.yaml`

3. **Run validation**:
   ```bash
   python scripts/validate_race_durations.py
   ```

4. **Review suspicious races**:
   - Open manual review interface
   - Interface automatically loads new competition from config
   - Review and correct as needed

---

## Future Extensions

These templates are designed to support:
- ✅ Multiple competitions
- ✅ Different video formats (MP4, AVI, MOV)
- ✅ Different frame rates (30fps, 60fps)
- 🔄 Custom validation rules per competition
- 🔄 Multi-language support
- 🔄 Custom data fields

**Plugin System**: Future versions will support custom validators and processors defined in config.

---

## Related Documentation

- **User Guide**: `docs/manual_review_interface_user_guide.md` (to be created)
- **Developer Guide**: `docs/manual_review_interface_developer_guide.md` (to be created)
- **Config Reference**: `configs/manual_review_config.yaml` (comprehensive config file)

---

**Last Updated**: 2025-11-16
**Version**: 1.0
