# PROMPT: Enhanced Manual Review Interface - Video Management & Advanced Features

**Version**: 1.5.1
**Date**: 2025-11-16
**Target**: claude.ai/code (UI Claude)
**Priority**: HIGH
**Estimated Time**: 6-8 hours

---

## Mission

Enhance the existing Manual Review Interface with **video management** and **advanced features** to transform it from a correction tool into a **complete video project management system** that can be used across all project phases.

---

## Context & Background

### What We Have (Phase 1.5 - COMPLETE)

The basic Manual Review Interface is **operational and tested**:
- ✅ Frame-by-frame video player
- ✅ Metadata correction interface
- ✅ Progress tracker (74 suspicious races)
- ✅ Validation engine
- ✅ Bilingual support (English + فارسی)
- ✅ Config-driven architecture

**User Feedback**: "این عالیه" (This is great!)

### What We Need Now (Phase 1.5.1 - Enhanced Features)

**User's Explicit Requirements**:
> "قابلیت مشاهده همه موارد مورد استفاده استفاده و افزودن و حذف و جایگزینی ویدئو (در صورت افزودن سیستم استخراج و ... هم گزینه داشته باشد) و فیلد نوشتن برای انتخاب فریم و سایر موارد مورد نیاز را در این مورد درنظر بگیرید و در تمام مراحل می‌تواند باشد."

**Translation**:
"Capability to view all used items and add/remove/replace videos (if adding, the extraction system should also be an option) and text field for frame selection and other necessary features, and can be used in all phases."

**Key Requirements**:
1. **Video Library Management**: View all videos in project
2. **Add Videos**: With automatic extraction workflow
3. **Remove/Replace Videos**: Clean management of video assets
4. **Text Input for Frames**: Direct frame number entry (not just slider)
5. **Bulk Operations**: Batch corrections, batch exports
6. **Multi-Phase Usability**: Use across Phase 1, 2, 3, etc.

---

## Architecture Overview

### Enhanced Components

```
scripts/review_interface/
├── app.py                    # Main Streamlit app (MODIFY)
├── config.py                 # Config manager (EXTEND)
├── video_player.py           # Video player (ENHANCE - add text input)
├── metadata_manager.py       # Metadata CRUD (existing)
├── progress.py               # Progress tracker (existing)
├── validators.py             # Validation engine (existing)
├── video_library.py          # ⭐ NEW: Video library manager
├── video_extraction.py       # ⭐ NEW: Integration with Phase 1 extraction
├── bulk_operations.py        # ⭐ NEW: Batch processing
└── export_manager.py         # ⭐ NEW: Multi-format export
```

### New UI Sections

```
┌─────────────────────────────────────────────────┐
│ Sidebar                                         │
│  ├─ 📊 Progress Statistics                      │
│  ├─ 🔍 Filters (Competition, Priority, Status)  │
│  └─ 📚 Video Library (⭐ NEW)                    │
│      ├─ View All Videos                         │
│      ├─ Add New Video                           │
│      ├─ Remove/Replace Video                    │
│      └─ Bulk Operations                         │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Main Area                                       │
│  ├─ Race Review (existing)                      │
│  │   ├─ Video Player (enhanced with text input) │
│  │   └─ Correction Interface                    │
│  │                                               │
│  ├─ 📚 Video Library View (⭐ NEW)               │
│  │   ├─ Table of all videos                     │
│  │   ├─ Filtering & search                      │
│  │   └─ Quick actions                           │
│  │                                               │
│  └─ ➕ Add New Video (⭐ NEW)                     │
│      ├─ Upload or select source video           │
│      ├─ Manual segmentation OR auto-extraction  │
│      └─ Metadata entry                          │
└─────────────────────────────────────────────────┘
```

---

## Feature Specifications

### Feature 1: Video Library Manager

**Purpose**: Central view of all videos in project across all competitions.

#### 1.1 Library View Component

**Location**: New tab or sidebar section

**Display**: Paginated table showing:
- Competition name
- Race ID
- Duration
- Status (✅ Reviewed, ⚠️ Suspicious, ❌ Failed, ⏳ Pending)
- Video path
- Metadata status
- Quick actions (View, Edit, Remove)

**Implementation** (`video_library.py`):

```python
"""
Video Library Manager
=====================
Provides centralized view of all videos across all competitions.
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

@dataclass
class VideoEntry:
    """Single video entry in library."""
    race_id: str
    competition: str
    video_path: Path
    metadata_path: Path
    duration: float
    status: str  # 'reviewed', 'suspicious', 'failed', 'pending'
    review_date: Optional[str] = None
    notes: str = ""

class VideoLibrary:
    """Manages video library across all competitions."""

    def __init__(self, config_manager):
        self.config = config_manager
        self.competitions = config_manager.get_competitions()

    def get_all_videos(self) -> List[VideoEntry]:
        """
        Get all videos across all competitions.

        Returns:
            List of VideoEntry objects sorted by competition and race_id
        """
        videos = []
        for comp in self.competitions:
            race_segments_path = Path(comp.race_segments_path)

            # Find all video files in competition directory
            if not race_segments_path.exists():
                continue

            for video_file in race_segments_path.glob("*.mp4"):
                race_id = video_file.stem
                metadata_file = video_file.with_suffix('.json').name.replace('.json', '_metadata.json')
                metadata_path = race_segments_path / metadata_file

                # Load metadata to get duration and status
                duration = 0.0
                status = 'pending'
                notes = ""

                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        duration = metadata.get('race_duration', 0.0)

                        # Determine status
                        if metadata.get('manual_correction', False):
                            status = 'reviewed'
                        elif duration < 0 or duration < 4.5 or duration > 15.0:
                            status = 'suspicious'
                        else:
                            status = 'pending'

                        notes = metadata.get('notes', '')

                entry = VideoEntry(
                    race_id=race_id,
                    competition=comp.key,
                    video_path=video_file,
                    metadata_path=metadata_path,
                    duration=duration,
                    status=status,
                    notes=notes
                )
                videos.append(entry)

        # Sort by competition and race_id
        videos.sort(key=lambda x: (x.competition, x.race_id))
        return videos

    def filter_videos(
        self,
        videos: List[VideoEntry],
        competition: Optional[str] = None,
        status: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> List[VideoEntry]:
        """
        Filter videos by criteria.

        Args:
            videos: List of VideoEntry objects
            competition: Filter by competition key
            status: Filter by status
            search_query: Search in race_id or notes

        Returns:
            Filtered list of VideoEntry objects
        """
        filtered = videos

        if competition and competition != "all":
            filtered = [v for v in filtered if v.competition.lower() == competition.lower()]

        if status and status != "all":
            filtered = [v for v in filtered if v.status == status]

        if search_query:
            query_lower = search_query.lower()
            filtered = [
                v for v in filtered
                if query_lower in v.race_id.lower() or query_lower in v.notes.lower()
            ]

        return filtered

    def get_statistics(self, videos: List[VideoEntry]) -> Dict:
        """
        Get statistics about video library.

        Returns:
            Dictionary with counts by status and competition
        """
        stats = {
            'total': len(videos),
            'by_status': {},
            'by_competition': {}
        }

        for video in videos:
            # Count by status
            stats['by_status'][video.status] = stats['by_status'].get(video.status, 0) + 1

            # Count by competition
            stats['by_competition'][video.competition] = stats['by_competition'].get(video.competition, 0) + 1

        return stats

    def remove_video(self, video_entry: VideoEntry, remove_files: bool = False) -> bool:
        """
        Remove video from library.

        Args:
            video_entry: VideoEntry to remove
            remove_files: If True, delete actual files. If False, just mark as removed.

        Returns:
            True if successful, False otherwise
        """
        try:
            if remove_files:
                # Delete video file
                if video_entry.video_path.exists():
                    video_entry.video_path.unlink()

                # Delete metadata file
                if video_entry.metadata_path.exists():
                    video_entry.metadata_path.unlink()

                # Delete backup files
                backup_pattern = f"{video_entry.metadata_path.stem}_backup_*.json"
                backup_dir = video_entry.metadata_path.parent
                for backup_file in backup_dir.glob(backup_pattern):
                    backup_file.unlink()
            else:
                # Mark as removed in metadata
                if video_entry.metadata_path.exists():
                    with open(video_entry.metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    metadata['removed'] = True
                    metadata['removal_date'] = str(datetime.now())

                    with open(video_entry.metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)

            return True
        except Exception as e:
            print(f"Error removing video: {e}")
            return False
```

#### 1.2 Streamlit UI Integration

**In `app.py`**, add new page/tab:

```python
def render_video_library_page():
    """Render video library management page."""
    st.title("📚 Video Library")

    # Initialize library manager
    library = VideoLibrary(config_manager)
    all_videos = library.get_all_videos()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_competition = st.selectbox(
            "Competition",
            ["all"] + [comp.key for comp in config_manager.get_competitions()]
        )
    with col2:
        filter_status = st.selectbox(
            "Status",
            ["all", "reviewed", "suspicious", "failed", "pending"]
        )
    with col3:
        search_query = st.text_input("Search", placeholder="Race ID or notes...")

    # Filter videos
    filtered_videos = library.filter_videos(
        all_videos,
        competition=filter_competition,
        status=filter_status,
        search_query=search_query
    )

    # Statistics
    stats = library.get_statistics(filtered_videos)
    st.metric("Total Videos", stats['total'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Reviewed", stats['by_status'].get('reviewed', 0))
    with col2:
        st.metric("Suspicious", stats['by_status'].get('suspicious', 0))
    with col3:
        st.metric("Failed", stats['by_status'].get('failed', 0))
    with col4:
        st.metric("Pending", stats['by_status'].get('pending', 0))

    # Video table
    st.subheader(f"Videos ({len(filtered_videos)})")

    # Convert to dataframe for display
    import pandas as pd
    if filtered_videos:
        df_data = []
        for video in filtered_videos:
            df_data.append({
                'Race ID': video.race_id,
                'Competition': video.competition,
                'Duration (s)': f"{video.duration:.2f}",
                'Status': video.status,
                'Notes': video.notes[:50] + '...' if len(video.notes) > 50 else video.notes
            })

        df = pd.DataFrame(df_data)

        # Display with interactive table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # Quick actions
        st.subheader("Quick Actions")
        selected_race = st.selectbox(
            "Select race for action:",
            [v.race_id for v in filtered_videos]
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("View in Player"):
                # Switch to review page with this race
                st.session_state['selected_race'] = selected_race
                st.rerun()
        with col2:
            if st.button("Remove (soft)"):
                video_to_remove = next(v for v in filtered_videos if v.race_id == selected_race)
                if library.remove_video(video_to_remove, remove_files=False):
                    st.success(f"Marked {selected_race} as removed")
                    st.rerun()
        with col3:
            if st.button("Delete (hard)", type="primary"):
                st.warning("⚠️ This will permanently delete video and metadata files!")
                if st.checkbox(f"I confirm deletion of {selected_race}"):
                    video_to_remove = next(v for v in filtered_videos if v.race_id == selected_race)
                    if library.remove_video(video_to_remove, remove_files=True):
                        st.success(f"Deleted {selected_race}")
                        st.rerun()
    else:
        st.info("No videos found matching filters.")
```

---

### Feature 2: Add New Videos with Extraction

**Purpose**: Allow users to add new competition videos with automatic extraction workflow.

#### 2.1 Video Extraction Integration

**Implementation** (`video_extraction.py`):

```python
"""
Video Extraction Integration
=============================
Integrates with Phase 1 race segmentation scripts to extract new races.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess
import json
from datetime import datetime

class VideoExtractor:
    """Integrates with Phase 1 extraction scripts."""

    def __init__(self, config_manager):
        self.config = config_manager
        self.project_root = Path(__file__).parent.parent.parent

    def extract_manual_segment(
        self,
        source_video: Path,
        competition_key: str,
        race_id: str,
        start_time: str,
        end_time: str,
        left_athlete: Dict,
        right_athlete: Dict,
        buffer_before: float = 3.0,
        buffer_after: float = 1.5
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Extract single race segment using manual timestamps.

        Args:
            source_video: Path to source video file
            competition_key: Competition identifier
            race_id: Unique race ID
            start_time: Manual start time (format: "MM:SS" or "HH:MM:SS")
            end_time: Manual end time
            left_athlete: Dict with name, country, bib_color
            right_athlete: Dict with name, country, bib_color
            buffer_before: Seconds before start
            buffer_after: Seconds after finish

        Returns:
            Tuple of (success, message, output_path)
        """
        # Get competition config
        comp = self.config.get_competition(competition_key)
        if not comp:
            return False, f"Competition {competition_key} not found in config", None

        # Prepare metadata
        metadata = {
            "race_id": race_id,
            "source_video": str(source_video),
            "round": "Manual Segmentation",
            "manual_start_time": start_time,
            "manual_end_time": end_time,
            "left_athlete": left_athlete,
            "right_athlete": right_athlete,
            "buffer_before": buffer_before,
            "buffer_after": buffer_after,
            "extraction_date": datetime.now().isoformat(),
            "notes": "Manually added via Review Interface"
        }

        # Output paths
        output_dir = Path(comp.race_segments_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_video = output_dir / f"{race_id}.mp4"
        output_metadata = output_dir / f"{race_id}_metadata.json"

        # Convert time strings to seconds
        def time_to_seconds(time_str: str) -> float:
            parts = list(map(int, time_str.split(':')))
            if len(parts) == 2:  # MM:SS
                return parts[0] * 60 + parts[1]
            elif len(parts) == 3:  # HH:MM:SS
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            else:
                raise ValueError(f"Invalid time format: {time_str}")

        try:
            start_seconds = time_to_seconds(start_time)
            end_seconds = time_to_seconds(end_time)

            # Add buffers
            extract_start = max(0, start_seconds - buffer_before)
            extract_end = end_seconds + buffer_after
            duration = extract_end - extract_start

            # Use ffmpeg to extract segment
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(source_video),
                '-ss', str(extract_start),
                '-t', str(duration),
                '-c', 'copy',  # Copy codec (fast)
                '-y',  # Overwrite
                str(output_video)
            ]

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                return False, f"FFmpeg error: {result.stderr}", None

            # Save metadata
            with open(output_metadata, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            return True, f"Successfully extracted {race_id}", output_video

        except Exception as e:
            return False, f"Extraction error: {str(e)}", None

    def extract_using_detection(
        self,
        source_video: Path,
        competition_key: str,
        race_prefix: str
    ) -> Tuple[bool, str, List[Path]]:
        """
        Extract races using automated detection (Phase 1 scripts).

        Args:
            source_video: Path to source video file
            competition_key: Competition identifier
            race_prefix: Prefix for race IDs (e.g., "Speed_finals_Paris_2026")

        Returns:
            Tuple of (success, message, list of extracted video paths)
        """
        # This would integrate with existing Phase 1 detection scripts
        # For now, return placeholder
        return False, "Automated detection not yet implemented - use manual segmentation", []
```

#### 2.2 Add Video UI Component

**In `app.py`**, add new page:

```python
def render_add_video_page():
    """Render add new video page."""
    st.title("➕ Add New Video")

    st.markdown("""
    Add a new race video to the project. You can either:
    1. **Manual Segmentation**: Specify exact timestamps
    2. **Automated Detection** (Future): Use Phase 1 detection scripts
    """)

    # Select competition
    competition_key = st.selectbox(
        "Competition",
        [comp.key for comp in config_manager.get_competitions()]
    )

    comp = config_manager.get_competition(competition_key)

    # Source video selection
    st.subheader("1. Source Video")
    source_video_path = st.text_input(
        "Source Video Path",
        placeholder="g:\\My Drive\\Projects\\...\\source_video.mp4"
    )

    if source_video_path and Path(source_video_path).exists():
        st.success(f"✅ Video found: {Path(source_video_path).name}")
    elif source_video_path:
        st.error("❌ Video file not found")

    # Method selection
    st.subheader("2. Extraction Method")
    method = st.radio(
        "Method",
        ["Manual Segmentation", "Automated Detection (Future)"],
        help="Manual segmentation requires start/end timestamps. Automated uses Phase 1 detection."
    )

    if method == "Manual Segmentation":
        render_manual_segmentation_form(competition_key, source_video_path)
    else:
        st.info("Automated detection will be implemented in future updates.")

def render_manual_segmentation_form(competition_key: str, source_video_path: str):
    """Render manual segmentation form."""
    st.subheader("3. Race Details")

    col1, col2 = st.columns(2)
    with col1:
        race_id = st.text_input(
            "Race ID",
            placeholder=f"Speed_finals_{competition_key}_raceXXX"
        )
        start_time = st.text_input(
            "Start Time",
            placeholder="MM:SS or HH:MM:SS"
        )
    with col2:
        round_name = st.text_input(
            "Round Name",
            placeholder="1/8 final - Men"
        )
        end_time = st.text_input(
            "End Time",
            placeholder="MM:SS or HH:MM:SS"
        )

    st.subheader("4. Athletes")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Left Athlete**")
        left_name = st.text_input("Name", key="left_name")
        left_country = st.text_input("Country", key="left_country")
        left_bib = st.text_input("Bib Color", key="left_bib", placeholder="blue/white/red")

    with col2:
        st.markdown("**Right Athlete**")
        right_name = st.text_input("Name", key="right_name")
        right_country = st.text_input("Country", key="right_country")
        right_bib = st.text_input("Bib Color", key="right_bib", placeholder="black/yellow")

    st.subheader("5. Buffers")
    col1, col2 = st.columns(2)
    with col1:
        buffer_before = st.number_input("Buffer Before (s)", value=3.0, min_value=0.0, max_value=10.0, step=0.5)
    with col2:
        buffer_after = st.number_input("Buffer After (s)", value=1.5, min_value=0.0, max_value=10.0, step=0.5)

    # Extract button
    if st.button("Extract Race Segment", type="primary"):
        # Validate inputs
        if not all([race_id, start_time, end_time, left_name, right_name]):
            st.error("Please fill all required fields")
            return

        if not source_video_path or not Path(source_video_path).exists():
            st.error("Invalid source video path")
            return

        # Perform extraction
        extractor = VideoExtractor(config_manager)

        with st.spinner("Extracting race segment..."):
            success, message, output_path = extractor.extract_manual_segment(
                source_video=Path(source_video_path),
                competition_key=competition_key,
                race_id=race_id,
                start_time=start_time,
                end_time=end_time,
                left_athlete={
                    "name": left_name,
                    "country": left_country,
                    "bib_color": left_bib
                },
                right_athlete={
                    "name": right_name,
                    "country": right_country,
                    "bib_color": right_bib
                },
                buffer_before=buffer_before,
                buffer_after=buffer_after
            )

        if success:
            st.success(f"✅ {message}")
            st.info(f"Output: {output_path}")

            # Add to progress tracker
            # (code to add new row to CSV)
        else:
            st.error(f"❌ {message}")
```

---

### Feature 3: Enhanced Video Player with Text Input

**Purpose**: Allow direct frame number entry, not just slider navigation.

#### 3.1 Enhanced VideoPlayer Component

**Modify `video_player.py`**:

```python
def render(self, key_prefix: str = "video") -> Tuple[int, float]:
    """
    Render video player with frame-by-frame navigation.

    **ENHANCED**: Now includes text input for direct frame entry.

    Args:
        key_prefix: Prefix for session state keys

    Returns:
        Tuple of (current_frame, current_time)
    """
    # Initialize session state
    if f'{key_prefix}_current_frame' not in st.session_state:
        st.session_state[f'{key_prefix}_current_frame'] = 0

    current_frame = st.session_state[f'{key_prefix}_current_frame']

    # === ENHANCED: Frame selection methods ===
    st.markdown("**Frame Navigation**")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Slider (existing)
        slider_frame = st.slider(
            "Frame Slider",
            min_value=0,
            max_value=self.total_frames - 1,
            value=current_frame,
            step=1,
            key=f"{key_prefix}_slider",
            label_visibility="collapsed"
        )

    with col2:
        # ⭐ NEW: Direct text input
        text_input_frame = st.number_input(
            "Frame #",
            min_value=0,
            max_value=self.total_frames - 1,
            value=current_frame,
            step=1,
            key=f"{key_prefix}_text_input",
            help="Enter frame number directly"
        )

    # Sync frame from either input method
    if text_input_frame != current_frame:
        current_frame = text_input_frame
    elif slider_frame != current_frame:
        current_frame = slider_frame

    st.session_state[f'{key_prefix}_current_frame'] = current_frame

    # Navigation buttons (existing)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("⏮️ -30", key=f"{key_prefix}_minus30"):
            current_frame = max(0, current_frame - 30)
            st.session_state[f'{key_prefix}_current_frame'] = current_frame
            st.rerun()
    with col2:
        if st.button("◀️ -1", key=f"{key_prefix}_minus1"):
            current_frame = max(0, current_frame - 1)
            st.session_state[f'{key_prefix}_current_frame'] = current_frame
            st.rerun()
    with col3:
        if st.button("▶️ +1", key=f"{key_prefix}_plus1"):
            current_frame = min(self.total_frames - 1, current_frame + 1)
            st.session_state[f'{key_prefix}_current_frame'] = current_frame
            st.rerun()
    with col4:
        if st.button("⏭️ +30", key=f"{key_prefix}_plus30"):
            current_frame = min(self.total_frames - 1, current_frame + 30)
            st.session_state[f'{key_prefix}_current_frame'] = current_frame
            st.rerun()
    with col5:
        if st.button("🔄 Reset", key=f"{key_prefix}_reset"):
            current_frame = 0
            st.session_state[f'{key_prefix}_current_frame'] = current_frame
            st.rerun()

    # Display frame info
    current_time = current_frame / self.fps
    st.caption(f"Frame: {current_frame} / {self.total_frames - 1} | Time: {current_time:.2f}s")

    # Display video frame
    frame = self._get_frame(current_frame)
    if frame is not None:
        st.image(frame, channels="BGR", use_column_width=True)

    return current_frame, current_time
```

---

### Feature 4: Bulk Operations

**Purpose**: Enable batch processing of multiple races (corrections, exports, etc.).

#### 4.1 Bulk Operations Manager

**Implementation** (`bulk_operations.py`):

```python
"""
Bulk Operations Manager
=======================
Batch processing for multiple races.
"""

from pathlib import Path
from typing import List, Dict, Callable
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

@dataclass
class BulkOperationResult:
    """Result of a bulk operation."""
    race_id: str
    success: bool
    message: str
    data: Dict = None

class BulkOperationsManager:
    """Manages bulk operations on multiple races."""

    def __init__(self, config_manager, metadata_manager):
        self.config = config_manager
        self.metadata = metadata_manager

    def apply_correction_to_multiple(
        self,
        race_ids: List[str],
        correction_func: Callable,
        **correction_kwargs
    ) -> List[BulkOperationResult]:
        """
        Apply a correction function to multiple races.

        Args:
            race_ids: List of race IDs to process
            correction_func: Function to apply to each race
            correction_kwargs: Additional arguments for correction function

        Returns:
            List of BulkOperationResult objects
        """
        results = []

        for race_id in race_ids:
            try:
                # Apply correction
                success, message = correction_func(race_id, **correction_kwargs)
                results.append(BulkOperationResult(
                    race_id=race_id,
                    success=success,
                    message=message
                ))
            except Exception as e:
                results.append(BulkOperationResult(
                    race_id=race_id,
                    success=False,
                    message=f"Error: {str(e)}"
                ))

        return results

    def export_multiple_races(
        self,
        race_ids: List[str],
        export_format: str,
        output_path: Path
    ) -> BulkOperationResult:
        """
        Export multiple races to a single file.

        Args:
            race_ids: List of race IDs to export
            export_format: 'json', 'csv', 'yaml'
            output_path: Output file path

        Returns:
            BulkOperationResult with export status
        """
        try:
            data = []

            for race_id in race_ids:
                # Load metadata for each race
                # (implementation depends on metadata structure)
                pass

            # Export based on format
            if export_format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            elif export_format == 'csv':
                import csv
                # CSV export logic
                pass
            elif export_format == 'yaml':
                import yaml
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f)

            return BulkOperationResult(
                race_id="bulk_export",
                success=True,
                message=f"Exported {len(race_ids)} races to {output_path}"
            )
        except Exception as e:
            return BulkOperationResult(
                race_id="bulk_export",
                success=False,
                message=f"Export failed: {str(e)}"
            )

    def validate_multiple_races(
        self,
        race_ids: List[str],
        validator
    ) -> List[BulkOperationResult]:
        """
        Validate multiple races in batch.

        Args:
            race_ids: List of race IDs to validate
            validator: RaceValidator instance

        Returns:
            List of validation results
        """
        results = []

        for race_id in race_ids:
            # Load race metadata
            # Validate
            # Append result
            pass

        return results
```

#### 4.2 Bulk Operations UI

**In `app.py`**, add bulk operations section:

```python
def render_bulk_operations_page():
    """Render bulk operations page."""
    st.title("⚡ Bulk Operations")

    st.markdown("""
    Perform batch operations on multiple races:
    - Batch validation
    - Batch export (JSON, CSV, YAML)
    - Batch corrections
    - Batch metrics recalculation
    """)

    # Select races
    st.subheader("1. Select Races")

    # Get all races from library
    library = VideoLibrary(config_manager)
    all_videos = library.get_all_videos()

    selection_method = st.radio(
        "Selection Method",
        ["Select by Competition", "Select by Status", "Manual Selection"]
    )

    selected_races = []

    if selection_method == "Select by Competition":
        comp_key = st.selectbox(
            "Competition",
            [comp.key for comp in config_manager.get_competitions()]
        )
        selected_races = [v.race_id for v in all_videos if v.competition == comp_key]
        st.info(f"Selected {len(selected_races)} races from {comp_key}")

    elif selection_method == "Select by Status":
        status = st.selectbox("Status", ["reviewed", "suspicious", "failed", "pending"])
        selected_races = [v.race_id for v in all_videos if v.status == status]
        st.info(f"Selected {len(selected_races)} races with status '{status}'")

    else:  # Manual Selection
        selected_races = st.multiselect(
            "Select Races",
            [v.race_id for v in all_videos]
        )

    # Select operation
    st.subheader("2. Select Operation")
    operation = st.selectbox(
        "Operation",
        [
            "Validate All",
            "Export Metadata",
            "Recalculate Metrics",
            "Apply Correction"
        ]
    )

    # Execute
    if st.button("Execute Bulk Operation", type="primary"):
        if not selected_races:
            st.error("No races selected")
            return

        bulk_mgr = BulkOperationsManager(config_manager, metadata_manager)

        with st.spinner(f"Processing {len(selected_races)} races..."):
            if operation == "Export Metadata":
                export_format = st.selectbox("Format", ["json", "csv", "yaml"])
                output_path = Path(f"bulk_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}")

                result = bulk_mgr.export_multiple_races(
                    selected_races,
                    export_format,
                    output_path
                )

                if result.success:
                    st.success(result.message)
                    st.download_button(
                        "Download Export",
                        data=open(output_path, 'rb').read(),
                        file_name=output_path.name
                    )
                else:
                    st.error(result.message)
```

---

### Feature 5: Multi-Phase Usability

**Purpose**: Make interface usable across all project phases (Phase 1, 2, 3, etc.).

#### 5.1 Phase-Aware Configuration

**Extend `config.py`**:

```python
class ConfigManager:
    """Extended with phase awareness."""

    def get_phase_settings(self, phase: str) -> Dict:
        """
        Get settings specific to a project phase.

        Args:
            phase: "phase1", "phase2", "phase3", etc.

        Returns:
            Phase-specific settings dictionary
        """
        return self.config.get('phases', {}).get(phase, {})

    def is_phase_enabled(self, phase: str) -> bool:
        """Check if a phase is enabled."""
        return self.config.get('phases', {}).get(phase, {}).get('enabled', False)
```

**In `manual_review_config.yaml`**, add:

```yaml
# Multi-Phase Settings
phases:
  phase1:
    enabled: true
    name: "Race Segmentation"
    features:
      - video_extraction
      - manual_segmentation
      - race_detection

  phase2:
    enabled: true
    name: "Pose Estimation"
    features:
      - pose_extraction
      - keypoint_visualization

  phase3:
    enabled: true
    name: "Performance Metrics"
    features:
      - metrics_calculation
      - calibration_integration
      - performance_visualization

  phase4:
    enabled: false
    name: "ML Predictions (Future)"
    features:
      - model_training
      - predictions
```

#### 5.2 Phase-Specific UI Elements

**In `app.py`**, add phase selector:

```python
def main():
    """Main application entry point."""

    # Sidebar: Phase selector
    st.sidebar.title("🎯 Project Phase")

    available_phases = []
    for phase_key in ['phase1', 'phase2', 'phase3', 'phase4']:
        if config_manager.is_phase_enabled(phase_key):
            settings = config_manager.get_phase_settings(phase_key)
            available_phases.append((phase_key, settings['name']))

    selected_phase = st.sidebar.selectbox(
        "Current Phase",
        available_phases,
        format_func=lambda x: x[1]
    )

    phase_key = selected_phase[0]
    phase_settings = config_manager.get_phase_settings(phase_key)

    st.sidebar.info(f"Active: {phase_settings['name']}")

    # Show phase-specific features
    if phase_key == "phase1":
        render_phase1_features()
    elif phase_key == "phase2":
        render_phase2_features()
    elif phase_key == "phase3":
        render_phase3_features()

def render_phase1_features():
    """Render Phase 1 specific features."""
    st.sidebar.markdown("**Phase 1 Features:**")
    st.sidebar.markdown("- Video Extraction")
    st.sidebar.markdown("- Manual Segmentation")
    st.sidebar.markdown("- Race Detection Review")

def render_phase2_features():
    """Render Phase 2 specific features."""
    st.sidebar.markdown("**Phase 2 Features:**")
    st.sidebar.markdown("- Pose Extraction")
    st.sidebar.markdown("- Keypoint Visualization")

def render_phase3_features():
    """Render Phase 3 specific features."""
    st.sidebar.markdown("**Phase 3 Features:**")
    st.sidebar.markdown("- Metrics Calculation")
    st.sidebar.markdown("- Calibration Integration")
    st.sidebar.markdown("- Performance Viz")
```

---

## Task Breakdown

### Task 1: Video Library Manager (2-3 hours)
- [ ] Create `video_library.py` with VideoLibrary class
- [ ] Implement `get_all_videos()` method
- [ ] Implement `filter_videos()` method
- [ ] Implement `get_statistics()` method
- [ ] Implement `remove_video()` method
- [ ] Add video library page to `app.py`
- [ ] Test with existing 188 race videos

### Task 2: Video Extraction Integration (2-3 hours)
- [ ] Create `video_extraction.py` with VideoExtractor class
- [ ] Implement `extract_manual_segment()` method
- [ ] Add ffmpeg integration
- [ ] Create add video page in `app.py`
- [ ] Add manual segmentation form
- [ ] Test with sample video extraction

### Task 3: Enhanced Video Player (1 hour)
- [ ] Modify `video_player.py::render()` method
- [ ] Add text input for direct frame entry
- [ ] Sync slider and text input
- [ ] Test frame navigation

### Task 4: Bulk Operations (1-2 hours)
- [ ] Create `bulk_operations.py` with BulkOperationsManager class
- [ ] Implement `export_multiple_races()` method
- [ ] Implement `validate_multiple_races()` method
- [ ] Add bulk operations page to `app.py`
- [ ] Test batch export

### Task 5: Multi-Phase Usability (1 hour)
- [ ] Extend `config.py` with phase methods
- [ ] Add phases section to YAML config
- [ ] Add phase selector to `app.py`
- [ ] Add phase-specific UI elements
- [ ] Test phase switching

### Task 6: Documentation & Testing (1 hour)
- [ ] Update user guide with new features
- [ ] Update developer guide
- [ ] Add tests for new components
- [ ] Test all features end-to-end

---

## Success Criteria

### Functional Requirements
- [ ] Video library displays all 188 existing race videos
- [ ] Can filter by competition, status, search query
- [ ] Can add new video with manual timestamps
- [ ] Video extraction creates valid MP4 + metadata JSON
- [ ] Can remove videos (soft delete or hard delete)
- [ ] Text input for frame numbers works correctly
- [ ] Can export metadata for multiple races in JSON/CSV/YAML
- [ ] Can switch between Phase 1/2/3 views
- [ ] All existing features still work (no regressions)

### Non-Functional Requirements
- [ ] Video library loads in <3 seconds for 188 videos
- [ ] Bulk operations show progress indicator
- [ ] Error handling for invalid inputs
- [ ] Persian + English bilingual support maintained
- [ ] Responsive UI (works on different screen sizes)

---

## Technical Constraints

- **ffmpeg Required**: For video extraction, ensure ffmpeg is installed and accessible
- **File Paths**: Handle Windows paths with spaces correctly
- **Concurrent Access**: Consider file locking for CSV/JSON updates
- **Memory**: Loading 188 videos' metadata should not exceed 100MB RAM
- **Compatibility**: Maintain backward compatibility with existing config files

---

## Deliverables

### Code Files
1. `scripts/review_interface/video_library.py` (NEW)
2. `scripts/review_interface/video_extraction.py` (NEW)
3. `scripts/review_interface/bulk_operations.py` (NEW)
4. `scripts/review_interface/video_player.py` (MODIFIED - add text input)
5. `scripts/review_interface/config.py` (MODIFIED - add phase methods)
6. `scripts/review_interface/app.py` (MODIFIED - add new pages)

### Configuration
7. `configs/manual_review_config.yaml` (MODIFIED - add phases section)

### Documentation
8. `docs/manual_review_interface_user_guide.md` (UPDATED)
9. `docs/manual_review_interface_developer_guide.md` (UPDATED)

### Tests
10. `scripts/review_interface/test_enhanced_features.py` (NEW)

---

## Example Usage Scenarios

### Scenario 1: Add New Competition (Paris 2026)

**User wants to add videos from Paris 2026 Olympics.**

1. User adds `paris_2026` to config YAML (2 minutes)
2. User goes to "➕ Add New Video" page
3. Selects competition: "paris_2026"
4. Provides source video path
5. Chooses "Manual Segmentation"
6. Enters race details (race ID, timestamps, athletes)
7. Clicks "Extract Race Segment"
8. Video is extracted and added to library
9. Appears in video library view
10. Can immediately review and correct if needed

**Time**: ~5 minutes per race

### Scenario 2: Bulk Export All Reviewed Races

**User wants to export all reviewed races for analysis.**

1. User goes to "⚡ Bulk Operations" page
2. Selects "Select by Status"
3. Chooses status: "reviewed"
4. Selects operation: "Export Metadata"
5. Chooses format: "JSON"
6. Clicks "Execute Bulk Operation"
7. Downloads exported file with all reviewed races

**Time**: ~1 minute

### Scenario 3: Quick Frame Jump

**User reviewing race needs to jump to frame 450.**

1. User in video player
2. Types "450" in frame number text input
3. Frame immediately jumps to 450
4. User marks as start/finish
5. Much faster than sliding from frame 0 to 450

**Time**: <5 seconds

---

## Integration with Existing Code

### Files to Keep Unchanged
- `metadata_manager.py` (already working)
- `progress.py` (already working)
- `validators.py` (already working)

### Files to Modify
- `app.py` - Add new pages and navigation
- `config.py` - Add phase methods
- `video_player.py` - Add text input
- `manual_review_config.yaml` - Add phases section

### Dependencies to Add
None beyond existing requirements (streamlit, opencv-python, pyyaml, pandas)

---

## Testing Checklist

### Video Library
- [ ] Displays all 188 existing videos
- [ ] Filtering by competition works
- [ ] Filtering by status works
- [ ] Search works (race ID, notes)
- [ ] Statistics show correct counts
- [ ] Soft delete marks as removed
- [ ] Hard delete removes files

### Video Extraction
- [ ] Manual segmentation creates valid video
- [ ] Metadata JSON created correctly
- [ ] Timestamps parsed correctly (MM:SS and HH:MM:SS)
- [ ] Buffers applied correctly
- [ ] Handles invalid inputs gracefully

### Enhanced Player
- [ ] Text input syncs with slider
- [ ] Can type frame number directly
- [ ] Frame navigation buttons still work
- [ ] Frame info displays correctly

### Bulk Operations
- [ ] Export JSON works
- [ ] Export CSV works
- [ ] Export YAML works
- [ ] Progress indicator shows

### Multi-Phase
- [ ] Can switch between phases
- [ ] Phase-specific features show/hide
- [ ] Config loads phase settings

---

## Notes for Implementation

### Priority Order
1. **Text input for frames** (quickest win, user requested)
2. **Video library view** (foundation for other features)
3. **Video extraction** (add new videos)
4. **Bulk operations** (efficiency gain)
5. **Multi-phase usability** (future-proofing)

### Persian Translation Needed
Add these to TRANSLATIONS dict in `app.py`:

```python
TRANSLATIONS_FA.update({
    'video_library': 'کتابخانه ویدئو',
    'add_video': 'افزودن ویدئو',
    'bulk_operations': 'عملیات دسته‌ای',
    'manual_segmentation': 'تقسیم‌بندی دستی',
    'extract_segment': 'استخراج بخش',
    'frame_number': 'شماره فریم',
    'competition': 'مسابقه',
    'status': 'وضعیت',
    'total_videos': 'مجموع ویدئوها',
    'remove_video': 'حذف ویدئو',
    'export_metadata': 'خروجی متادیتا'
})
```

---

## Questions to Clarify (If Any)

1. **Video Extraction**: Should we integrate with existing Phase 1 detection scripts, or just manual segmentation for now?
   - **Recommendation**: Start with manual only, add automated later

2. **Bulk Operations**: Which operations are highest priority?
   - **Recommendation**: Export and validation first

3. **File Deletion**: Should hard delete require additional confirmation?
   - **Recommendation**: Yes, add checkbox confirmation

---

## Final Checklist Before Delivery

- [ ] All new Python files have docstrings
- [ ] All methods have type hints
- [ ] Persian translations added for new UI elements
- [ ] User guide updated with screenshots/examples
- [ ] Developer guide updated with API reference
- [ ] All tests pass
- [ ] No regression in existing features
- [ ] Git commit with descriptive message
- [ ] Code follows existing style (PEP 8)

---

**Ready to Build!** 🚀

This prompt provides complete specifications for all enhanced features. Proceed with implementation following the task breakdown order.
