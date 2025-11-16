# Phase 1.5.1 - Complete Implementation Plan
# برنامه جامع پیاده‌سازی فاز 1.5.1

**Date**: 2025-11-16
**Version**: 2.0 (Enhanced)
**Role**: Project Manager + Developer + QA + User

---

## Executive Summary (خلاصه اجرایی)

این سند roadmap کامل برای تکمیل Phase 1.5.1 و آماده‌سازی برای فازهای بعدی است.

**Status**:
- ✅ Enhanced Video Player: COMPLETE (با sync کامل)
- ✅ Video Library Manager: COMPLETE
- 🔄 Video Extraction: IN PROGRESS
- ⏳ Bulk Operations: PENDING
- ⏳ Multi-Phase Support: PENDING

---

## Part 1: Remaining Features (3 features)

### Feature 3: Video Extraction Integration ⚙️

**Purpose**: افزودن ویدیوهای جدید با extraction خودکار

**User Stories**:
```
به عنوان reviewer:
- می‌خواهم ویدیوی جدید اضافه کنم
- می‌خواهم timestamps را manual وارد کنم
- می‌خواهم segment استخراج شود
- می‌خواهم metadata به صورت خودکار ساخته شود
```

**Technical Specs**:
```python
# video_extraction.py
class VideoExtractor:
    - extract_manual_segment() - با ffmpeg
    - validate_timestamps() - چک کردن format
    - create_metadata() - ساخت JSON
    - integrate_with_library() - افزودن به Video Library
```

**UI Components**:
```
Add Video Page:
├─ Source Video Selection (file picker or path)
├─ Manual Timestamp Entry
│  ├─ Start Time (MM:SS or HH:MM:SS)
│  ├─ End Time (MM:SS or HH:MM:SS)
│  └─ Buffers (before/after)
├─ Athlete Information
│  ├─ Left: Name, Country, Bib Color
│  └─ Right: Name, Country, Bib Color
├─ Race Details
│  ├─ Race ID (auto-suggest format)
│  ├─ Competition (dropdown)
│  └─ Round (text input)
└─ Extract Button → Show Progress → Success/Error
```

**Implementation Steps**:
1. Create `video_extraction.py` with VideoExtractor class
2. Add ffmpeg wrapper for segment extraction
3. Create UI in `app.py` - "Add Video" page
4. Test with sample video
5. Integrate with Video Library

**Time Estimate**: 2 hours

---

### Feature 4: Bulk Operations Manager 📦

**Purpose**: عملیات دسته‌ای روی چند race

**User Stories**:
```
به عنوان researcher:
- می‌خواهم همه races یک competition را export کنم
- می‌خواهم batch validation کنم
- می‌خواهم چندین race را همزمان process کنم
```

**Operations**:
1. **Batch Export**:
   - Export selected races to JSON/CSV/YAML
   - Include metadata + metrics
   - ML-ready format (NPZ with features)

2. **Batch Validation**:
   - Run validators on multiple races
   - Generate validation report
   - Flag suspicious races

3. **Batch Re-calculation**:
   - Recalculate metrics for selected races
   - Use updated calibration
   - Save results

**Technical Specs**:
```python
# bulk_operations.py
class BulkOperationsManager:
    - select_races() - با filters
    - export_multiple() - به formats مختلف
    - validate_multiple() - batch validation
    - recalculate_metrics() - با progress bar
```

**UI Components**:
```
Bulk Operations Page:
├─ Race Selection
│  ├─ By Competition
│  ├─ By Status
│  ├─ By Date Range
│  └─ Manual Selection (multiselect)
├─ Operation Type
│  ├─ Export (JSON/CSV/YAML/NPZ)
│  ├─ Validate
│  └─ Recalculate Metrics
├─ Options (per operation)
└─ Execute → Progress Bar → Download/Results
```

**Implementation Steps**:
1. Create `bulk_operations.py`
2. Implement export functions (multi-format)
3. Implement validation wrapper
4. Create UI with progress tracking
5. Test with 10+ races

**Time Estimate**: 2 hours

---

### Feature 5: Multi-Phase Support 🔌

**Purpose**: استفاده در تمام فازهای پروژه + plugin architecture

**Goals**:
- Support Phase 1, 2, 3, 4 workflows
- Extensible plugin system
- Future-proof design

**Technical Specs**:
```python
# phase_manager.py
class PhaseManager:
    - get_current_phase() - از config
    - get_phase_features() - list of enabled features
    - enable_plugin() - activate feature
    - disable_plugin() - deactivate feature
```

**Plugin Architecture**:
```python
# plugins/base.py
class PluginBase:
    name: str
    version: str
    phase: str

    def initialize() -> bool
    def render_ui(st) -> None
    def process_data(data) -> Result

# plugins/ml_predictor.py (Phase 4 example)
class MLPredictorPlugin(PluginBase):
    name = "CNN-Transformer Predictor"
    phase = "phase4"

    def render_ui(st):
        st.subheader("🧠 ML Predictions")
        # Show predictions
```

**Config Extension**:
```yaml
# manual_review_config.yaml
phases:
  phase1:
    enabled: true
    plugins:
      - video_extraction
      - manual_review

  phase2:
    enabled: true
    plugins:
      - pose_visualization

  phase3:
    enabled: true
    plugins:
      - metrics_analysis

  phase4:
    enabled: false  # Future
    plugins:
      - ml_predictor
      - real_time_stream
```

**UI Integration**:
```
Sidebar:
├─ Phase Selector (dropdown)
├─ Active Plugins (badges)
└─ Plugin Settings (expander)

Main Area:
├─ Phase-specific pages
└─ Plugin-rendered content
```

**Implementation Steps**:
1. Create `phase_manager.py`
2. Create `plugins/base.py`
3. Extend config with phases
4. Update `app.py` with phase switching
5. Create example plugin

**Time Estimate**: 1.5 hours

---

## Part 2: UX Enhancements (بهبودهای پیشنهادی)

### Enhancement 1: Keyboard Shortcuts ⌨️

**Purpose**: سرعت بخشیدن به workflow

```python
# keyboard_shortcuts.py
SHORTCUTS = {
    'Arrow Left': 'Previous frame (-1)',
    'Arrow Right': 'Next frame (+1)',
    'Arrow Up': 'Jump +30 frames',
    'Arrow Down': 'Jump -30 frames',
    'Home': 'Jump to start (frame 0)',
    'End': 'Jump to end (last frame)',
    'Space': 'Toggle play/pause (future)',
    'B': 'Add bookmark',
    'J': 'Jump to bookmark',
    'S': 'Save corrections',
    'R': 'Reset to original',
    'Z': 'Undo',
    'Y': 'Redo'
}
```

**Implementation**: Use `streamlit-keyup` or JavaScript injection

---

### Enhancement 2: Auto-Save ⚡

**Purpose**: جلوگیری از از دست رفتن corrections

```python
# auto_save.py
class AutoSaveManager:
    save_interval: int = 30  # seconds

    def auto_save_corrections():
        # هر 30 ثانیه ذخیره خودکار
        pass

    def recover_unsaved():
        # بازیابی در صورت crash
        pass
```

---

### Enhancement 3: Undo/Redo System 🔄

**Purpose**: امکان برگشت از تغییرات

```python
# undo_manager.py
class UndoManager:
    history: List[State]
    current_index: int

    def record_change(state):
        pass

    def undo() -> State:
        pass

    def redo() -> State:
        pass
```

---

## Part 3: Integration Plan (اتصال همه‌چیز)

### app.py Structure (Updated)

```python
# scripts/review_interface/app.py

import streamlit as st
from config import ConfigManager
from video_library import VideoLibrary
from video_extraction import VideoExtractor
from bulk_operations import BulkOperationsManager
from phase_manager import PhaseManager
from keyboard_shortcuts import KeyboardShortcuts

def main():
    st.set_page_config(page_title="Speed Climbing Analysis", layout="wide")

    # Initialize managers
    config = ConfigManager()
    phase_manager = PhaseManager(config)
    library = VideoLibrary(config)

    # Sidebar
    with st.sidebar:
        render_sidebar(config, phase_manager)

    # Main area - phase-specific pages
    page = st.session_state.get('page', 'home')

    if page == 'home':
        render_home_page()
    elif page == 'review':
        render_review_page(config)
    elif page == 'library':
        render_library_page(library)
    elif page == 'add_video':
        render_add_video_page(config)
    elif page == 'bulk_ops':
        render_bulk_operations_page(config, library)
    elif page == 'settings':
        render_settings_page(config, phase_manager)

def render_sidebar(config, phase_manager):
    # Phase selector
    current_phase = phase_manager.get_current_phase()
    st.selectbox("Phase", phases, key='current_phase')

    # Navigation
    st.radio("Navigation", [
        "🏠 Home",
        "🔍 Review Races",
        "📚 Video Library",
        "➕ Add Video",
        "⚡ Bulk Operations",
        "⚙️ Settings"
    ], key='page')

    # Statistics
    stats = library.get_statistics(library.get_all_videos())
    st.metric("Total Videos", stats['total'])
    st.metric("Reviewed", stats['by_status'].get('reviewed', 0))
    st.metric("Suspicious", stats['by_status'].get('suspicious', 0))
```

---

## Part 4: Testing Plan (برنامه تست)

### Unit Tests

```python
# tests/test_video_extraction.py
def test_extract_manual_segment():
    # Test ffmpeg extraction
    pass

def test_validate_timestamps():
    # Test timestamp parsing
    pass

# tests/test_bulk_operations.py
def test_export_json():
    # Test JSON export
    pass

def test_export_csv():
    # Test CSV export
    pass

# tests/test_phase_manager.py
def test_phase_switching():
    # Test phase activation
    pass

def test_plugin_loading():
    # Test plugin system
    pass
```

### Integration Tests

```bash
# Test full workflow
1. Add new video → Check library updated
2. Bulk export → Check files created
3. Phase switch → Check UI changes
4. Keyboard shortcuts → Check navigation works
```

### User Acceptance Tests

```
Scenario 1: Add New Competition
- User adds Paris_2026 competition
- User uploads source video
- User extracts 5 races manually
- All races appear in library ✓

Scenario 2: Batch Export for ML
- User selects all "reviewed" races (74 races)
- User exports to NPZ format
- File contains features + labels ✓

Scenario 3: Multi-Phase Workflow
- User switches to Phase 2
- UI shows pose-related tools
- User switches to Phase 3
- UI shows metrics tools ✓
```

---

## Part 5: Documentation Updates

### Files to Update:

1. **User Guide** (`docs/manual_review_interface_user_guide.md`):
   - Add Video Extraction tutorial
   - Add Bulk Operations guide
   - Add Keyboard Shortcuts reference

2. **Developer Guide** (`docs/manual_review_interface_developer_guide.md`):
   - Plugin development tutorial
   - API reference for all classes
   - Phase system architecture

3. **MASTER_CONTEXT.md**:
   - Update Phase 1.5.1 status
   - Add new features list
   - Update roadmap

4. **PROMPT_FOR_UI_ENHANCED_FEATURES.md**:
   - Mark completed features
   - Add enhancement notes

---

## Part 6: Deployment Checklist

### Before Release:

- [ ] All features implemented
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] User testing completed
- [ ] Documentation updated
- [ ] Performance benchmarks met (<3s library load)
- [ ] No regressions in existing features
- [ ] Code reviewed
- [ ] Git commits organized
- [ ] Branch merged to main

### Performance Targets:

```
✓ Library load time: <3 seconds (188 videos)
✓ Video extraction: <30 seconds per race
✓ Bulk export (100 races): <60 seconds
✓ Phase switching: <1 second
✓ UI responsiveness: <100ms
```

---

## Part 7: Timeline

```
Day 1 (Today):
├─ Video Extraction: 2h
├─ Bulk Operations: 2h
├─ Multi-Phase: 1.5h
└─ Integration: 1h
Total: 6.5h

Day 2:
├─ UX Enhancements: 2h
├─ Testing: 3h
├─ Documentation: 2h
└─ Bug Fixes: 1h
Total: 8h

Day 3:
├─ Final Testing: 2h
├─ User Acceptance: 2h
├─ Deployment Prep: 1h
└─ Release: 1h
Total: 6h
```

**Total Estimate**: 20.5 hours (3 days)

---

## Part 8: Risk Management

### Risks & Mitigation:

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| FFmpeg not available | Low | High | Check ffmpeg on startup, show install guide |
| Large video files crash | Medium | High | Add file size limits, streaming extraction |
| Plugin system too complex | Low | Medium | Start simple, iterate based on needs |
| Performance issues | Medium | Medium | Lazy loading, caching, pagination |
| User confusion | High | Low | Tooltips, tutorials, contextual help |

---

## Part 9: Success Metrics

### KPIs:

1. **Functionality**:
   - ✅ All 8 features working (5 core + 3 enhancements)
   - ✅ Zero critical bugs
   - ✅ 100% feature test coverage

2. **Performance**:
   - ✅ <3s load time
   - ✅ <100ms UI response
   - ✅ Handles 188+ videos smoothly

3. **User Experience**:
   - ✅ <5 min onboarding time
   - ✅ Intuitive UI (no manual needed for basic tasks)
   - ✅ Bilingual support working

4. **Code Quality**:
   - ✅ All functions documented
   - ✅ Type hints throughout
   - ✅ PEP 8 compliant

---

## Part 10: Next Steps (After Phase 1.5.1)

### Immediate (Week 1):
- Manual review of 74 suspicious races
- Dataset cleanup
- Prepare for Phase 4

### Short-term (Month 1):
- Design GraphQL API
- Prototype CNN-Transformer model
- Setup development environment for Phase 4

### Long-term (Months 2-6):
- Phase 4 implementation (18 weeks)
- Real-time streaming
- Web/Mobile deployment

---

**Plan Ready! Let's Execute! 🚀**

---

## Appendix A: File Structure

```
scripts/review_interface/
├── app.py                      # Main application (UPDATED)
├── config.py                   # Config manager (existing)
├── video_player.py             # Video player (ENHANCED ✓)
├── video_library.py            # Library manager (NEW ✓)
├── video_extraction.py         # Extraction (NEW - pending)
├── bulk_operations.py          # Bulk ops (NEW - pending)
├── phase_manager.py            # Phase system (NEW - pending)
├── keyboard_shortcuts.py       # Shortcuts (NEW - pending)
├── auto_save.py                # Auto-save (NEW - pending)
├── undo_manager.py             # Undo/Redo (NEW - pending)
├── metadata_manager.py         # Metadata CRUD (existing)
├── progress.py                 # Progress tracker (existing)
└── validators.py               # Validators (existing)

plugins/
├── base.py                     # Plugin base class
└── examples/
    └── ml_predictor.py         # Phase 4 example plugin

tests/
├── test_video_extraction.py
├── test_bulk_operations.py
├── test_phase_manager.py
└── test_integration.py

docs/
├── PHASE_1.5.1_IMPLEMENTATION_PLAN.md  # This file
├── manual_review_interface_user_guide.md
├── manual_review_interface_developer_guide.md
└── KEYBOARD_SHORTCUTS_REFERENCE.md
```

---

**END OF IMPLEMENTATION PLAN**

Ready to execute! 💪
