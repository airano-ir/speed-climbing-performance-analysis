# Manual Review Interface - Build Instructions for UI Claude

**Date**: 2025-11-16
**Priority**: HIGH - Blocking 71 race reviews
**Estimated Time**: 4-6 hours
**Complexity**: Medium-High
**Type**: INFRASTRUCTURE (Permanent Tool, Not One-Time Fix)

---

## 🎯 MISSION: Build Permanent, Extensible Manual Review Tool

### Executive Summary

You need to build a **professional-grade Streamlit interface** for manually reviewing and correcting race detection errors. This is **NOT** a quick one-time fix script — it's **permanent project infrastructure** that must:

1. ✅ **Handle current need**: Review 74 suspicious races (39.4% of dataset)
2. ✅ **Support future competitions**: Add new competition in < 5 minutes via config
3. ✅ **Be extensible**: Plugin system, modular design, feature flags for future enhancements
4. ✅ **Production quality**: Error handling, validation, autosave, progress tracking

**User's explicit requirement**: "این نکته را درنظر داشته باشید که بشه Interface را برای سایر ویدئو‌ها و یا بعدا برای اضافه کردن ویدئو در صورت نیاز پروژه و یا بهبود آن هم استفاده کرد"

Translation: "Consider that the interface should be usable for other videos or later for adding videos to the project if needed or for improvements"

---

## 📊 Background Context

### The Discovery

During Phase 3 testing, we discovered **3 critical race detection errors**. We fixed those 3 races and created validation scripts. The validation revealed:

**🚨 74 suspicious races out of 188 total (39.4%!):**
- **5 CRITICAL**: Negative or near-zero duration (impossible!)
  - Example: `Villars_2024_race015` → -2.57s
  - Example: `Villars_2024_race023` → 0.03s
- **58 HIGH**: Zilina 2025 systematic failure (84% of that competition!)
  - All races have duration > 15s (likely different video format)
- **11 MEDIUM**: Other races too short or too long

### Why Infrastructure Approach?

**Time Calculation**:
- Manual fix per race: ~15-20 minutes
- 74 races × 15 min = **18.5 hours of tedious work**
- Building interface: 4-6 hours
- Time saved: **12+ hours** + reusable for future competitions

**Strategic Value**:
- Permanent asset for project
- Future competitions can be added in minutes
- Extensible for ML suggestions, batch processing, etc.
- Professional-quality data validation

---

## 🏗️ Architecture & Design

### Config-Driven System

**All configuration in YAML** ([configs/manual_review_config.yaml](../configs/manual_review_config.yaml)):
```yaml
competitions:
  paris_2026:  # Add new competition easily!
    name: "Speed Finals Paris 2026"
    fps: 60.0
    race_segments_path: "data/race_segments/paris_2026"
    total_races: 32
```

**Benefits**:
- No code changes to add new competition
- Competition-specific settings (FPS, buffers, validation rules)
- Feature flags for enabling/disabling features

### Modular Component Structure

```
scripts/review_interface/
├── __init__.py
├── app.py                # Streamlit main app
├── config.py             # Config loader & manager
├── video_player.py       # Video playback component
├── metadata_manager.py   # CRUD for race metadata
├── validators.py         # Validation engine
├── progress.py           # Progress tracker (CSV)
└── export.py             # Export pipeline
```

**Why Modular?**
- Each component testable independently
- Easy to extend (add new validators, exporters, etc.)
- Reusable components (video player can be used elsewhere)
- Clear separation of concerns

### Plugin System (Future-Ready)

```yaml
# In config file
custom_validators:
  - name: "negative_duration_check"
    enabled: true
    severity: "critical"

  - name: "velocity_sanity_check"
    enabled: false  # Disabled for now, can enable later
    severity: "medium"
    parameters:
      max_velocity: 5.0  # m/s
```

---

## 📋 TASKS BREAKDOWN

### Phase 1: Core Infrastructure (2-3 hours)

#### Task 1.1: Project Setup (15 min)
```bash
# Create directory structure
mkdir -p scripts/review_interface
cd scripts/review_interface

# Create __init__.py
touch __init__.py
```

#### Task 1.2: Config Manager (30 min)

**File**: `scripts/review_interface/config.py`

```python
"""
Configuration Manager
====================
Loads and manages manual_review_config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class CompetitionConfig:
    """Competition configuration"""
    key: str
    name: str
    date: str
    video_format: str
    fps: float
    race_segments_path: Path
    total_races: int
    notes: str = ""

class ConfigManager:
    """Manages configuration for manual review interface."""

    def __init__(self, config_path: str = "configs/manual_review_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load YAML config file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_competitions(self) -> List[CompetitionConfig]:
        """Get list of all competitions."""
        competitions = []
        for key, data in self.config['competitions'].items():
            if key.startswith('_') or 'template' in key.lower():
                continue  # Skip comments and templates

            comp = CompetitionConfig(
                key=key,
                name=data['name'],
                date=data['date'],
                video_format=data['video_format'],
                fps=data['fps'],
                race_segments_path=Path(data['race_segments_path']),
                total_races=data['total_races'],
                notes=data.get('notes', '')
            )
            competitions.append(comp)

        return competitions

    def get_competition(self, key: str) -> Optional[CompetitionConfig]:
        """Get specific competition by key."""
        comps = self.get_competitions()
        for comp in comps:
            if comp.key == key:
                return comp
        return None

    def get_validation_rules(self) -> Dict:
        """Get validation rules."""
        return self.config['validation']

    def get_general_settings(self) -> Dict:
        """Get general settings."""
        return self.config['general']

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.config['features'].get(feature, False)
```

**Requirements**:
- Load YAML config
- Parse competitions list (skip templates/comments)
- Provide easy access to settings, validation rules, features
- Type-safe dataclasses for configuration

#### Task 1.3: Progress Tracker (30 min)

**File**: `scripts/review_interface/progress.py`

```python
"""
Progress Tracker
================
Manages CSV file tracking review progress for 74 suspicious races.
"""

import csv
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class RaceReviewStatus:
    """Status of a single race review."""
    priority: int
    race_id: str
    competition: str
    detected_duration_s: float
    duration_frames: int
    confidence_start: float
    confidence_finish: float
    status: str  # "SUSPICIOUS - TOO SHORT" etc.
    issue_description: str
    review_status: str  # "Pending" / "Completed" / "Skipped"
    corrected_duration_s: str  # Empty or float value
    corrected_start_frame: str  # Empty or int value
    corrected_finish_frame: str  # Empty or int value
    reviewer_notes: str
    review_date: str

class ProgressTracker:
    """Manages progress tracking CSV file."""

    def __init__(self, csv_path: str = "data/manual_review/progress_tracker.csv"):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Progress tracker not found: {self.csv_path}")

    def load_all_races(self) -> List[RaceReviewStatus]:
        """Load all races from CSV."""
        races = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                race = RaceReviewStatus(
                    priority=int(row['Priority']),
                    race_id=row['Race_ID'],
                    competition=row['Competition'],
                    detected_duration_s=float(row['Detected_Duration_s']),
                    duration_frames=int(row['Duration_Frames']),
                    confidence_start=float(row['Confidence_Start']),
                    confidence_finish=float(row['Confidence_Finish']),
                    status=row['Status'],
                    issue_description=row['Issue_Description'],
                    review_status=row['Review_Status'],
                    corrected_duration_s=row['Corrected_Duration_s'],
                    corrected_start_frame=row['Corrected_Start_Frame'],
                    corrected_finish_frame=row['Corrected_Finish_Frame'],
                    reviewer_notes=row['Reviewer_Notes'],
                    review_date=row['Review_Date']
                )
                races.append(race)
        return races

    def get_pending_races(self) -> List[RaceReviewStatus]:
        """Get only pending races."""
        all_races = self.load_all_races()
        return [r for r in all_races if r.review_status == 'Pending']

    def update_race(self, race: RaceReviewStatus) -> None:
        """Update a single race in CSV."""
        all_races = self.load_all_races()

        # Find and update
        for i, r in enumerate(all_races):
            if r.race_id == race.race_id:
                all_races[i] = race
                break

        # Write back to CSV
        self._write_all_races(all_races)

    def _write_all_races(self, races: List[RaceReviewStatus]) -> None:
        """Write all races to CSV."""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'Priority', 'Race_ID', 'Competition',
                'Detected_Duration_s', 'Duration_Frames',
                'Confidence_Start', 'Confidence_Finish',
                'Status', 'Issue_Description',
                'Review_Status', 'Corrected_Duration_s',
                'Corrected_Start_Frame', 'Corrected_Finish_Frame',
                'Reviewer_Notes', 'Review_Date'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for race in races:
                writer.writerow({
                    'Priority': race.priority,
                    'Race_ID': race.race_id,
                    'Competition': race.competition,
                    'Detected_Duration_s': race.detected_duration_s,
                    'Duration_Frames': race.duration_frames,
                    'Confidence_Start': race.confidence_start,
                    'Confidence_Finish': race.confidence_finish,
                    'Status': race.status,
                    'Issue_Description': race.issue_description,
                    'Review_Status': race.review_status,
                    'Corrected_Duration_s': race.corrected_duration_s,
                    'Corrected_Start_Frame': race.corrected_start_frame,
                    'Corrected_Finish_Frame': race.corrected_finish_frame,
                    'Reviewer_Notes': race.reviewer_notes,
                    'Review_Date': race.review_date
                })

    def get_statistics(self) -> Dict:
        """Get review statistics."""
        all_races = self.load_all_races()
        return {
            'total': len(all_races),
            'completed': sum(1 for r in all_races if r.review_status == 'Completed'),
            'pending': sum(1 for r in all_races if r.review_status == 'Pending'),
            'skipped': sum(1 for r in all_races if r.review_status == 'Skipped'),
            'critical': sum(1 for r in all_races if r.priority == 1),
            'high': sum(1 for r in all_races if r.priority == 2),
            'medium': sum(1 for r in all_races if r.priority == 3)
        }
```

**Requirements**:
- Load CSV with 74 suspicious races
- Filter by status (pending/completed/skipped)
- Update individual races
- Calculate statistics
- Thread-safe file operations

#### Task 1.4: Metadata Manager (45 min)

**File**: `scripts/review_interface/metadata_manager.py`

```python
"""
Metadata Manager
================
CRUD operations for race metadata JSON files.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import shutil

class MetadataManager:
    """Manages race metadata JSON files."""

    def __init__(self, race_segments_dir: str = "data/race_segments"):
        self.race_segments_dir = Path(race_segments_dir)

    def load_metadata(self, competition: str, race_id: str) -> Dict:
        """Load metadata for a race."""
        metadata_path = self._get_metadata_path(competition, race_id)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_metadata(
        self,
        competition: str,
        race_id: str,
        metadata: Dict,
        backup: bool = True
    ) -> None:
        """Save updated metadata (with optional backup)."""
        metadata_path = self._get_metadata_path(competition, race_id)

        # Create backup
        if backup and metadata_path.exists():
            backup_path = metadata_path.with_suffix('.json.bak')
            shutil.copy(metadata_path, backup_path)

        # Write updated metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def update_race_boundaries(
        self,
        competition: str,
        race_id: str,
        new_start_frame: int,
        new_finish_frame: int,
        fps: float,
        correction_reason: str,
        reviewer_notes: str = ""
    ) -> Dict:
        """Update race boundaries with manual correction."""
        metadata = self.load_metadata(competition, race_id)

        # Preserve original values
        if not metadata.get('manual_correction', False):
            # First time correction
            correction_metadata = {
                'original_detected_start_frame': metadata['detected_start_frame'],
                'original_detected_finish_frame': metadata['detected_finish_frame'],
                'original_detected_duration': f"{metadata['race_duration']}s (INVALID)",
                'correction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'correction_reason': correction_reason,
                'reviewer_notes': reviewer_notes
            }
            metadata['correction_metadata'] = correction_metadata
        else:
            # Update existing correction
            metadata['correction_metadata']['correction_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata['correction_metadata']['reviewer_notes'] = reviewer_notes

        # Update detected values
        metadata['detected_start_frame'] = new_start_frame
        metadata['detected_finish_frame'] = new_finish_frame

        # Recalculate duration and times
        duration_frames = new_finish_frame - new_start_frame
        duration_seconds = duration_frames / fps

        metadata['race_duration'] = duration_seconds
        metadata['detected_start_time'] = new_start_frame / fps
        metadata['detected_finish_time'] = new_finish_frame / fps

        # Set manual correction flag
        metadata['manual_correction'] = True
        metadata['confidence_start'] = 1.0  # Manual review → perfect confidence
        metadata['confidence_finish'] = 1.0

        # Save
        self.save_metadata(competition, race_id, metadata, backup=True)

        return metadata

    def _get_metadata_path(self, competition: str, race_id: str) -> Path:
        """Get path to metadata JSON file."""
        return self.race_segments_dir / competition / f"{race_id}_metadata.json"

    def get_video_path(self, competition: str, race_id: str) -> Path:
        """Get path to race video file."""
        return self.race_segments_dir / competition / f"{race_id}.mp4"
```

**Requirements**:
- Load/save race metadata JSON files
- Update race boundaries with correction tracking
- Preserve original values (audit trail)
- Create backups before modifications
- Calculate durations automatically

---

### Phase 2: Video Player Component (1 hour)

#### Task 2.1: Video Player Module (1 hour)

**File**: `scripts/review_interface/video_player.py`

```python
"""
Video Player Component
======================
Streamlit component for video playback with frame-by-frame navigation.
"""

import streamlit as st
import cv2
from pathlib import Path
import tempfile
from typing import Tuple, Optional

class VideoPlayer:
    """Video player with frame navigation for Streamlit."""

    def __init__(self, video_path: Path, fps: float):
        self.video_path = video_path
        self.fps = fps
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.total_frames / fps

    def render(self, key_prefix: str = "video") -> Tuple[int, float]:
        """
        Render video player with controls.

        Returns:
            (current_frame, current_time_seconds)
        """
        st.subheader("Video Player")

        # Display video info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frames", self.total_frames)
        with col2:
            st.metric("FPS", f"{self.fps:.1f}")
        with col3:
            st.metric("Duration", f"{self.duration_seconds:.2f}s")

        # Frame slider
        current_frame = st.slider(
            "Frame",
            min_value=0,
            max_value=self.total_frames - 1,
            value=st.session_state.get(f'{key_prefix}_current_frame', 0),
            step=1,
            key=f"{key_prefix}_frame_slider"
        )

        # Time display
        current_time = current_frame / self.fps
        st.text(f"Time: {current_time:.3f}s (Frame {current_frame})")

        # Navigation buttons
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("⏮️ -30", key=f"{key_prefix}_minus30"):
                current_frame = max(0, current_frame - 30)
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        with col2:
            if st.button("⬅️ -1", key=f"{key_prefix}_minus1"):
                current_frame = max(0, current_frame - 1)
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        with col3:
            if st.button("➡️ +1", key=f"{key_prefix}_plus1"):
                current_frame = min(self.total_frames - 1, current_frame + 1)
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        with col4:
            if st.button("⏭️ +30", key=f"{key_prefix}_plus30"):
                current_frame = min(self.total_frames - 1, current_frame + 30)
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        with col5:
            if st.button("⏹️ Reset", key=f"{key_prefix}_reset"):
                current_frame = 0
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        # Display current frame
        frame_image = self._get_frame(current_frame)
        if frame_image is not None:
            st.image(frame_image, channels="BGR", use_container_width=True)
        else:
            st.error(f"Failed to load frame {current_frame}")

        # Store current frame in session state
        st.session_state[f'{key_prefix}_current_frame'] = current_frame

        return current_frame, current_time

    def _get_frame(self, frame_number: int) -> Optional:
        """Get specific frame from video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def mark_frame(self, frame_number: int, label: str, color: Tuple[int, int, int] = (0, 255, 0)):
        """Mark a specific frame with label (for start/finish marking)."""
        # This can be enhanced to draw on the frame
        pass

    def close(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
```

**Requirements**:
- Load video with OpenCV
- Frame-by-frame navigation (slider + buttons)
- Display current frame as image
- Show time and frame number
- Navigation: ±1 frame, ±30 frames, reset
- Store state in Streamlit session_state

---

### Phase 3: Streamlit Main App (1.5-2 hours)

#### Task 3.1: Main Application (2 hours)

**File**: `scripts/review_interface/app.py`

```python
"""
Manual Review Interface - Main Streamlit App
============================================
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.review_interface.config import ConfigManager
from scripts.review_interface.progress import ProgressTracker, RaceReviewStatus
from scripts.review_interface.metadata_manager import MetadataManager
from scripts.review_interface.video_player import VideoPlayer

# Page config
st.set_page_config(
    page_title="Race Detection Review Interface",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize managers
@st.cache_resource
def get_managers():
    config_mgr = ConfigManager()
    progress_tracker = ProgressTracker()
    metadata_mgr = MetadataManager()
    return config_mgr, progress_tracker, metadata_mgr

config_mgr, progress_tracker, metadata_mgr = get_managers()

# Title
st.title("🏔️ Speed Climbing - Manual Race Review Interface")
st.markdown("**Fix race detection errors for 74 suspicious races**")

# Sidebar - Statistics & Navigation
with st.sidebar:
    st.header("📊 Progress Statistics")

    stats = progress_tracker.get_statistics()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Races", stats['total'])
        st.metric("Completed", stats['completed'], delta=f"{stats['completed']/stats['total']*100:.1f}%")
    with col2:
        st.metric("Pending", stats['pending'])
        st.metric("Critical", stats['critical'], delta="Priority 1", delta_color="inverse")

    st.markdown("---")

    # Competition filter
    st.header("🎯 Filter Races")

    competitions = config_mgr.get_competitions()
    competition_names = ["All"] + [c.name for c in competitions]
    selected_competition = st.selectbox("Competition", competition_names)

    # Priority filter
    selected_priority = st.selectbox(
        "Priority",
        ["All", "Critical (1)", "High (2)", "Medium (3)", "Low (4)"]
    )

    # Status filter
    selected_status = st.selectbox(
        "Status",
        ["Pending Only", "All", "Completed", "Skipped"]
    )

# Main content - Race list and review
st.header("🏁 Race Review")

# Load races based on filters
all_races = progress_tracker.load_all_races()

# Apply filters
filtered_races = all_races

if selected_competition != "All":
    comp_key = next((c.key for c in competitions if c.name == selected_competition), None)
    filtered_races = [r for r in filtered_races if r.competition == comp_key]

if selected_priority != "All":
    priority_num = int(selected_priority.split('(')[1].strip(')'))
    filtered_races = [r for r in filtered_races if r.priority == priority_num]

if selected_status == "Pending Only":
    filtered_races = [r for r in filtered_races if r.review_status == 'Pending']
elif selected_status != "All":
    filtered_races = [r for r in filtered_races if r.review_status == selected_status]

# Sort by priority
filtered_races.sort(key=lambda r: (r.priority, r.race_id))

st.info(f"📋 Showing {len(filtered_races)} races (filtered from {len(all_races)} total)")

# Race selection
if filtered_races:
    race_options = [f"[P{r.priority}] {r.race_id} ({r.detected_duration_s:.2f}s → {r.issue_description})" for r in filtered_races]
    selected_race_idx = st.selectbox("Select Race to Review", range(len(race_options)), format_func=lambda i: race_options[i])

    selected_race = filtered_races[selected_race_idx]

    st.markdown("---")

    # Display race information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📁 Race Info")
        st.text(f"Race ID: {selected_race.race_id}")
        st.text(f"Competition: {selected_race.competition}")
        st.text(f"Priority: {selected_race.priority}")
        st.text(f"Status: {selected_race.review_status}")

    with col2:
        st.subheader("⚠️ Detection Issue")
        st.error(f"**{selected_race.issue_description}**")
        st.text(f"Detected Duration: {selected_race.detected_duration_s:.2f}s")
        st.text(f"Frames: {selected_race.duration_frames}")
        st.text(f"Start Confidence: {selected_race.confidence_start:.2f}")
        st.text(f"Finish Confidence: {selected_race.confidence_finish:.2f}")

    with col3:
        st.subheader("✅ Correction Status")
        if selected_race.corrected_duration_s:
            st.success(f"Corrected: {selected_race.corrected_duration_s}s")
            st.text(f"Start Frame: {selected_race.corrected_start_frame}")
            st.text(f"Finish Frame: {selected_race.corrected_finish_frame}")
        else:
            st.warning("Not yet corrected")

    st.markdown("---")

    # Load video and metadata
    competition_config = config_mgr.get_competition(selected_race.competition)

    if competition_config:
        video_path = metadata_mgr.get_video_path(selected_race.competition, selected_race.race_id)

        if video_path.exists():
            # Load current metadata
            metadata = metadata_mgr.load_metadata(selected_race.competition, selected_race.race_id)

            # Video player
            player = VideoPlayer(video_path, competition_config.fps)
            current_frame, current_time = player.render(key_prefix=f"race_{selected_race.race_id}")

            st.markdown("---")

            # Correction interface
            st.subheader("✏️ Correct Race Boundaries")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🟢 Start Frame**")
                st.text(f"Current detected: {metadata['detected_start_frame']}")

                if st.button("📍 Mark Current as START", key="mark_start"):
                    st.session_state['new_start_frame'] = current_frame
                    st.success(f"Start marked at frame {current_frame} ({current_time:.2f}s)")

                new_start_frame = st.number_input(
                    "New Start Frame",
                    min_value=0,
                    max_value=player.total_frames - 1,
                    value=st.session_state.get('new_start_frame', metadata['detected_start_frame']),
                    key="start_frame_input"
                )

            with col2:
                st.markdown("**🔴 Finish Frame**")
                st.text(f"Current detected: {metadata['detected_finish_frame']}")

                if st.button("📍 Mark Current as FINISH", key="mark_finish"):
                    st.session_state['new_finish_frame'] = current_frame
                    st.success(f"Finish marked at frame {current_frame} ({current_time:.2f}s)")

                new_finish_frame = st.number_input(
                    "New Finish Frame",
                    min_value=0,
                    max_value=player.total_frames - 1,
                    value=st.session_state.get('new_finish_frame', metadata['detected_finish_frame']),
                    key="finish_frame_input"
                )

            # Calculate new duration
            new_duration_frames = new_finish_frame - new_start_frame
            new_duration_seconds = new_duration_frames / competition_config.fps

            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("New Duration (s)", f"{new_duration_seconds:.2f}")
            with col2:
                st.metric("New Duration (frames)", new_duration_frames)
            with col3:
                # Validation
                validation_rules = config_mgr.get_validation_rules()
                min_dur = validation_rules['duration']['min']
                max_dur = validation_rules['duration']['max']

                if new_duration_seconds < min_dur:
                    st.error(f"⚠️ Below minimum ({min_dur}s)")
                elif new_duration_seconds > max_dur:
                    st.error(f"⚠️ Above maximum ({max_dur}s)")
                else:
                    st.success("✅ Valid duration")

            # Correction reason
            st.markdown("---")
            st.subheader("📝 Correction Details")

            correction_reason = st.text_input(
                "Correction Reason",
                value="",
                placeholder="e.g., Climber fall detected as finish, Pre-race warmup included, False start"
            )

            reviewer_notes = st.text_area(
                "Reviewer Notes",
                value="",
                placeholder="Additional notes about this correction..."
            )

            # Save buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("💾 Save Correction", type="primary", key="save_correction"):
                    if not correction_reason:
                        st.error("Please provide a correction reason!")
                    else:
                        # Update metadata
                        updated_metadata = metadata_mgr.update_race_boundaries(
                            competition=selected_race.competition,
                            race_id=selected_race.race_id,
                            new_start_frame=new_start_frame,
                            new_finish_frame=new_finish_frame,
                            fps=competition_config.fps,
                            correction_reason=correction_reason,
                            reviewer_notes=reviewer_notes
                        )

                        # Update progress tracker
                        selected_race.review_status = 'Completed'
                        selected_race.corrected_duration_s = str(new_duration_seconds)
                        selected_race.corrected_start_frame = str(new_start_frame)
                        selected_race.corrected_finish_frame = str(new_finish_frame)
                        selected_race.reviewer_notes = reviewer_notes
                        selected_race.review_date = updated_metadata['correction_metadata']['correction_date']

                        progress_tracker.update_race(selected_race)

                        st.success("✅ Correction saved successfully!")
                        st.balloons()

                        # Clear session state
                        if 'new_start_frame' in st.session_state:
                            del st.session_state['new_start_frame']
                        if 'new_finish_frame' in st.session_state:
                            del st.session_state['new_finish_frame']

                        st.rerun()

            with col2:
                if st.button("⏭️ Skip Race", key="skip_race"):
                    selected_race.review_status = 'Skipped'
                    progress_tracker.update_race(selected_race)
                    st.info("Race skipped")
                    st.rerun()

            with col3:
                if st.button("🔄 Reset", key="reset_form"):
                    if 'new_start_frame' in st.session_state:
                        del st.session_state['new_start_frame']
                    if 'new_finish_frame' in st.session_state:
                        del st.session_state['new_finish_frame']
                    st.rerun()

            # Cleanup
            player.close()

        else:
            st.error(f"Video file not found: {video_path}")
    else:
        st.error(f"Competition config not found: {selected_race.competition}")

else:
    st.info("No races match the selected filters. Adjust filters in the sidebar.")
```

**Requirements**:
- Streamlit interface with sidebar (stats + filters)
- Race list (sortable, filterable)
- Video player integration
- Frame marking (start/finish)
- Metadata editing with validation
- Save corrections to metadata + progress tracker
- Progress statistics
- Session state management

---

### Phase 4: Validation & Testing (1 hour)

#### Task 4.1: Validators Module (30 min)

**File**: `scripts/review_interface/validators.py`

```python
"""
Validators
==========
Validation functions for race corrections.
"""

from typing import Dict, List, Tuple

class RaceValidator:
    """Validates race corrections."""

    def __init__(self, min_duration: float = 4.5, max_duration: float = 15.0):
        self.min_duration = min_duration
        self.max_duration = max_duration

    def validate_duration(self, duration_seconds: float) -> Tuple[bool, str]:
        """
        Validate race duration.

        Returns:
            (is_valid, message)
        """
        if duration_seconds < 0:
            return False, f"Negative duration: {duration_seconds:.2f}s (INVALID!)"

        if duration_seconds < self.min_duration:
            return False, f"Too short: {duration_seconds:.2f}s < {self.min_duration}s (below world record)"

        if duration_seconds > self.max_duration:
            return False, f"Too long: {duration_seconds:.2f}s > {self.max_duration}s (unusually slow)"

        return True, f"Valid: {duration_seconds:.2f}s"

    def validate_frame_order(self, start_frame: int, finish_frame: int) -> Tuple[bool, str]:
        """Validate that finish_frame > start_frame."""
        if finish_frame <= start_frame:
            return False, f"Finish frame ({finish_frame}) must be after start frame ({start_frame})"

        return True, "Frame order valid"

    def validate_all(
        self,
        start_frame: int,
        finish_frame: int,
        fps: float
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Run all validations.

        Returns:
            Dict of {validator_name: (is_valid, message)}
        """
        results = {}

        # Frame order
        results['frame_order'] = self.validate_frame_order(start_frame, finish_frame)

        # Duration
        duration = (finish_frame - start_frame) / fps
        results['duration'] = self.validate_duration(duration)

        return results
```

#### Task 4.2: Testing (30 min)

**Create test script**: `scripts/review_interface/test_app.py`

```python
"""
Test Script for Review Interface
=================================
Run this before using the app.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.review_interface.config import ConfigManager
from scripts.review_interface.progress import ProgressTracker
from scripts.review_interface.metadata_manager import MetadataManager

def test_config_manager():
    """Test config manager."""
    print("Testing ConfigManager...")
    config = ConfigManager()

    comps = config.get_competitions()
    print(f"  ✓ Loaded {len(comps)} competitions")

    for comp in comps:
        print(f"    - {comp.name} ({comp.key}): {comp.total_races} races")

    rules = config.get_validation_rules()
    print(f"  ✓ Duration range: {rules['duration']['min']}-{rules['duration']['max']}s")

    print()

def test_progress_tracker():
    """Test progress tracker."""
    print("Testing ProgressTracker...")
    tracker = ProgressTracker()

    all_races = tracker.load_all_races()
    print(f"  ✓ Loaded {len(all_races)} suspicious races")

    pending = tracker.get_pending_races()
    print(f"  ✓ Pending races: {len(pending)}")

    stats = tracker.get_statistics()
    print(f"  ✓ Statistics: {stats}")

    print()

def test_metadata_manager():
    """Test metadata manager."""
    print("Testing MetadataManager...")
    mgr = MetadataManager()

    # Test with Race001 (already corrected)
    try:
        metadata = mgr.load_metadata('chamonix_2024', 'Speed_finals_Chamonix_2024_race001')
        print(f"  ✓ Loaded metadata for Race001")
        print(f"    Duration: {metadata['race_duration']}s")
        print(f"    Manual correction: {metadata.get('manual_correction', False)}")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")

    print()

def main():
    """Run all tests."""
    print("=" * 60)
    print("MANUAL REVIEW INTERFACE - COMPONENT TESTS")
    print("=" * 60)
    print()

    test_config_manager()
    test_progress_tracker()
    test_metadata_manager()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print()
    print("To run the app:")
    print("  cd scripts/review_interface")
    print("  streamlit run app.py")

if __name__ == "__main__":
    main()
```

**Run tests**:
```bash
python scripts/review_interface/test_app.py
```

---

## 🎯 Success Criteria

### Must-Have (Before Delivery)

1. ✅ **All components implemented**:
   - Config manager
   - Progress tracker
   - Metadata manager
   - Video player
   - Main Streamlit app
   - Validators

2. ✅ **Core functionality working**:
   - Load and display 74 suspicious races
   - Video playback with frame navigation
   - Mark start/finish frames
   - Save corrections to metadata
   - Update progress tracker
   - Real-time validation

3. ✅ **Tested with real data**:
   - Test with 1 already-corrected race (Race001)
   - Test with 1 pending critical race
   - Verify metadata backup creation
   - Verify progress tracker updates

4. ✅ **Documentation**:
   - Inline code comments
   - Module docstrings
   - README for usage instructions

### Nice-to-Have (Optional Enhancements)

1. 🔄 **Keyboard shortcuts**: Arrow keys for frame navigation
2. 🔄 **Batch actions**: Mark multiple races as skipped
3. 🔄 **Export report**: CSV summary of all corrections
4. 🔄 **Undo function**: Revert last correction
5. 🔄 **Visual timeline**: Show start/finish markers on video timeline

---

## 📦 Deliverables

### Code Files (7 files minimum)

1. `scripts/review_interface/__init__.py`
2. `scripts/review_interface/config.py` (~150 lines)
3. `scripts/review_interface/progress.py` (~150 lines)
4. `scripts/review_interface/metadata_manager.py` (~120 lines)
5. `scripts/review_interface/video_player.py` (~120 lines)
6. `scripts/review_interface/validators.py` (~60 lines)
7. `scripts/review_interface/app.py` (~400 lines)
8. `scripts/review_interface/test_app.py` (~80 lines)

**Total**: ~1,080 lines of production-quality Python code

### Documentation (2 files)

1. `docs/manual_review_interface_user_guide.md` (brief usage guide)
2. `scripts/review_interface/README.md` (developer guide)

---

## ⚠️ IMPORTANT NOTES

### 1. Don't Reinvent the Wheel

**Existing Infrastructure** (already created):
- ✅ `configs/manual_review_config.yaml` (300+ lines, comprehensive)
- ✅ `data/manual_review/progress_tracker.csv` (74 races populated)
- ✅ `data/manual_review/templates/*` (templates for future)
- ✅ `scripts/validate_corrected_races.py` (validation script)

**Use these!** Don't create new config formats or tracking systems.

### 2. Metadata Structure

**Existing metadata format** (already defined in 3 corrected races):
```json
{
  "race_id": "Speed_finals_Chamonix_2024_race001",
  "detected_start_frame": 14049,
  "detected_finish_frame": 14244,
  "race_duration": 6.5,
  "manual_correction": true,
  "correction_metadata": {
    "original_detected_start_frame": 11784,
    "original_detected_finish_frame": 11837,
    "original_detected_duration": "1.77s (INVALID)",
    "correction_date": "2025-11-16",
    "correction_reason": "Climber fall detected as finish",
    "reviewer_notes": "Used RIGHT lane finish time..."
  }
}
```

**Follow this exact format** when saving corrections!

### 3. Real-World Validation Reference

**World Records** (for sanity checking):
- Men: 5.00s (Reza Alipour, 2023)
- Women: 6.53s (Aleksandra Miroslaw, 2024)

**Reasonable Range**:
- Minimum: 4.5s (slightly below WR)
- Maximum: 15s (slower climbers + falls)

**Flag for review**:
- < 3s: CRITICAL (impossible!)
- 3-4.5s: Very suspicious (faster than WR)
- 15-20s: Suspicious (unusually slow)
- > 20s: CRITICAL (includes non-race footage)

### 4. Future Extensibility

**Design for**:
- Adding new competitions (via config only, no code changes)
- Adding new validators (plugin system in config)
- ML-based suggestions (feature flag disabled for now)
- Collaborative review (multi-user support later)
- Batch operations (review multiple races sequentially)

**Architecture principles**:
- Config-driven (YAML)
- Modular components (each file = one responsibility)
- Plugin system (validators, exporters extensible)
- State management (Streamlit session_state for UI state)
- Error handling (try/except with user-friendly messages)

---

## 🚀 Getting Started (Step-by-Step)

### Phase 1: Setup (15 min)
```bash
# 1. Create directory structure
mkdir -p scripts/review_interface
cd scripts/review_interface
touch __init__.py

# 2. Verify existing infrastructure
ls configs/manual_review_config.yaml
ls data/manual_review/progress_tracker.csv
ls data/manual_review/templates/
```

### Phase 2: Implement Components (2-3 hours)
1. Start with `config.py` (simplest, no dependencies)
2. Then `progress.py` (uses config)
3. Then `metadata_manager.py` (file I/O)
4. Then `validators.py` (pure logic)
5. Then `video_player.py` (Streamlit + OpenCV)

### Phase 3: Build Main App (1.5-2 hours)
1. Implement `app.py` step-by-step:
   - Sidebar (stats, filters)
   - Race list display
   - Video player integration
   - Correction interface
   - Save functionality

### Phase 4: Test & Document (1 hour)
1. Create `test_app.py`
2. Test all components
3. Test with real races:
   - Race001 (already corrected - verify load)
   - Villars_2024_race015 (critical - test correction)
4. Write README and user guide

### Phase 5: Demo & Deliver (30 min)
1. Run app: `streamlit run scripts/review_interface/app.py`
2. Demo workflow:
   - Filter to "Critical" races
   - Select Villars_2024_race015
   - Navigate video to find start/finish
   - Mark frames
   - Save correction
   - Verify metadata updated
   - Verify progress tracker updated
3. Take screenshots for documentation
4. Commit all code

---

## 📞 Questions?

If anything is unclear:
1. **Config format**: See [configs/manual_review_config.yaml](../configs/manual_review_config.yaml)
2. **Metadata format**: See corrected Race001 metadata
3. **Progress tracker format**: See [data/manual_review/progress_tracker.csv](../data/manual_review/progress_tracker.csv)
4. **Validation report**: See [data/processed/race_duration_validation_report.json](../data/processed/race_duration_validation_report.json)
5. **Architectural vision**: See [MASTER_CONTEXT.md](../MASTER_CONTEXT.md) → Phase 1.5 section

---

**Good luck! This interface will save 15-20 hours of manual work and become permanent project infrastructure.** 🚀

**Remember**: Build for the future, not just for the current 74 races. This tool should make it trivial to add Paris 2026 Olympics races when they become available!
