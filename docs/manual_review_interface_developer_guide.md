# Manual Review Interface - Developer Guide

**Version**: 1.0
**Last Updated**: 2025-11-16 (To be updated after UI build)
**Status**: 📝 PLACEHOLDER - Will be completed by UI Claude

---

## Architecture Overview

### Design Principles

1. **Config-Driven**: All settings in YAML, no hardcoded values
2. **Modular**: Each component has single responsibility
3. **Extensible**: Plugin system for validators, exporters
4. **Production-Ready**: Error handling, validation, autosave
5. **Future-Proof**: Feature flags for planned enhancements

### Component Structure

```
scripts/review_interface/
├── __init__.py               # Package initialization
├── app.py                    # Streamlit main application
├── config.py                 # Configuration manager
├── video_player.py           # Video playback component
├── metadata_manager.py       # Race metadata CRUD
├── progress.py               # CSV progress tracker
├── validators.py             # Validation engine
├── export.py                 # Export pipeline (future)
└── README.md                 # Developer documentation
```

---

## Component Details

### 1. ConfigManager (`config.py`)

**Purpose**: Load and manage `configs/manual_review_config.yaml`

**Key Classes**:
- `CompetitionConfig`: Dataclass for competition settings
- `ConfigManager`: Singleton config loader

**Methods**:
```python
get_competitions() -> List[CompetitionConfig]
get_competition(key: str) -> CompetitionConfig
get_validation_rules() -> Dict
get_general_settings() -> Dict
is_feature_enabled(feature: str) -> bool
```

**Usage**:
```python
config = ConfigManager()
comps = config.get_competitions()
rules = config.get_validation_rules()
```

### 2. ProgressTracker (`progress.py`)

**Purpose**: Manage CSV file tracking 74 suspicious races

**Key Classes**:
- `RaceReviewStatus`: Dataclass for race status
- `ProgressTracker`: CSV CRUD operations

**Methods**:
```python
load_all_races() -> List[RaceReviewStatus]
get_pending_races() -> List[RaceReviewStatus]
update_race(race: RaceReviewStatus) -> None
get_statistics() -> Dict
```

**CSV Format**:
```csv
Priority,Race_ID,Competition,Detected_Duration_s,Duration_Frames,
Confidence_Start,Confidence_Finish,Status,Issue_Description,
Review_Status,Corrected_Duration_s,Corrected_Start_Frame,
Corrected_Finish_Frame,Reviewer_Notes,Review_Date
```

### 3. MetadataManager (`metadata_manager.py`)

**Purpose**: CRUD operations for race metadata JSON files

**Key Methods**:
```python
load_metadata(competition: str, race_id: str) -> Dict
save_metadata(competition: str, race_id: str, metadata: Dict, backup: bool)
update_race_boundaries(
    competition: str,
    race_id: str,
    new_start_frame: int,
    new_finish_frame: int,
    fps: float,
    correction_reason: str,
    reviewer_notes: str
) -> Dict
get_video_path(competition: str, race_id: str) -> Path
```

**Metadata Structure**:
```json
{
  "race_id": "Speed_finals_Competition_YYYY_raceNNN",
  "detected_start_frame": 14049,
  "detected_finish_frame": 14244,
  "race_duration": 6.5,
  "manual_correction": true,
  "correction_metadata": {
    "original_detected_start_frame": 11784,
    "original_detected_finish_frame": 11837,
    "original_detected_duration": "1.77s (INVALID)",
    "correction_date": "2025-11-16 14:30:00",
    "correction_reason": "Climber fall detected as finish",
    "reviewer_notes": "..."
  }
}
```

### 4. VideoPlayer (`video_player.py`)

**Purpose**: Streamlit component for video playback

**Key Class**:
- `VideoPlayer`: OpenCV-based video player

**Methods**:
```python
render(key_prefix: str) -> Tuple[int, float]  # Returns (frame, time)
_get_frame(frame_number: int) -> np.ndarray
mark_frame(frame_number: int, label: str, color: Tuple)
close()
```

**Session State**:
- `{key_prefix}_current_frame`: Current frame number
- Persists across Streamlit reruns

### 5. RaceValidator (`validators.py`)

**Purpose**: Validation logic for corrections

**Methods**:
```python
validate_duration(duration_seconds: float) -> Tuple[bool, str]
validate_frame_order(start_frame: int, finish_frame: int) -> Tuple[bool, str]
validate_all(start_frame: int, finish_frame: int, fps: float) -> Dict
```

**Validation Rules** (from config):
- Duration: 4.5s - 15s
- Frame order: finish > start
- No negative durations

---

## Adding New Features

### 1. Adding a New Competition

**Step 1**: Add to config YAML
```yaml
# configs/manual_review_config.yaml
competitions:
  paris_2026:
    name: "Speed Finals Paris 2026"
    date: "2026-07-15"
    video_format: "mp4"
    fps: 60.0  # Note: 60fps!
    race_segments_path: "data/race_segments/paris_2026"
    total_races: 32
    notes: "Olympics - higher quality videos"
```

**Step 2**: That's it! No code changes needed.

**Time**: ~2 minutes

### 2. Adding a Custom Validator

**Step 1**: Add to config YAML
```yaml
custom_validators:
  - name: "my_custom_validator"
    enabled: true
    severity: "medium"
    parameters:
      threshold: 10.0
```

**Step 2**: Implement in `validators.py`
```python
class RaceValidator:
    def validate_custom(self, param: float) -> Tuple[bool, str]:
        """Custom validation logic."""
        if param > self.config['threshold']:
            return False, f"Failed: {param}"
        return True, "Passed"
```

**Step 3**: Call in validation pipeline
```python
results['custom'] = validator.validate_custom(value)
```

### 3. Adding an Export Format

**Step 1**: Create `export.py`
```python
class ExportManager:
    def export_to_yaml(self, races: List) -> str:
        """Export corrections as YAML."""
        # Implementation
```

**Step 2**: Add to app interface
```python
if st.button("Export as YAML"):
    exporter = ExportManager()
    yaml_output = exporter.export_to_yaml(corrected_races)
    st.download_button("Download", yaml_output, "corrections.yaml")
```

---

## Plugin System (Future)

### Validator Plugins

**Interface**:
```python
class ValidatorPlugin(ABC):
    @abstractmethod
    def validate(self, race_data: Dict) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def get_severity(self) -> str:
        pass
```

**Example**:
```python
class VelocitySanityValidator(ValidatorPlugin):
    def validate(self, race_data: Dict) -> Tuple[bool, str]:
        # Check if velocity is physically possible
        if race_data['max_velocity'] > 5.0:  # m/s
            return False, "Velocity exceeds human capability"
        return True, "Velocity valid"

    def get_severity(self) -> str:
        return "medium"
```

**Registration** (in config):
```yaml
custom_validators:
  - name: "velocity_sanity_check"
    class: "VelocitySanityValidator"
    enabled: true
    parameters:
      max_velocity: 5.0
```

---

## Testing

### Unit Tests

**Create**: `tests/test_review_interface.py`

```python
import pytest
from scripts.review_interface.config import ConfigManager
from scripts.review_interface.progress import ProgressTracker
from scripts.review_interface.validators import RaceValidator

def test_config_loading():
    config = ConfigManager()
    comps = config.get_competitions()
    assert len(comps) > 0

def test_progress_tracker():
    tracker = ProgressTracker()
    races = tracker.load_all_races()
    assert len(races) == 74  # Should have 74 suspicious races

def test_validator():
    validator = RaceValidator(min_duration=4.5, max_duration=15.0)

    # Valid duration
    is_valid, msg = validator.validate_duration(6.5)
    assert is_valid

    # Invalid duration (too short)
    is_valid, msg = validator.validate_duration(2.0)
    assert not is_valid

    # Invalid duration (negative)
    is_valid, msg = validator.validate_duration(-1.0)
    assert not is_valid
```

**Run**:
```bash
pytest tests/test_review_interface.py -v
```

### Integration Tests

**Test workflow**:
1. Load race from progress tracker
2. Load video and metadata
3. Mark start/finish frames
4. Validate correction
5. Save metadata
6. Update progress tracker
7. Verify all files updated correctly

---

## Performance Considerations

### Video Loading

**Issue**: Loading large videos can be slow

**Solution**:
```python
@st.cache_resource
def load_video(video_path: str):
    return VideoPlayer(video_path, fps)
```

**Cache invalidation**:
- Clear cache when video file changes
- Use `hash_funcs` for custom cache keys

### CSV Updates

**Issue**: Writing CSV after each race update can cause concurrency issues

**Solution**:
- Use file locking (fcntl on Unix, msvcrt on Windows)
- Implement atomic writes (write to temp, then rename)
- Add retry logic with exponential backoff

```python
import fcntl  # Unix
import msvcrt  # Windows

def atomic_csv_write(path: Path, data: List):
    temp_path = path.with_suffix('.tmp')

    # Write to temp file
    with open(temp_path, 'w') as f:
        # Lock file
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Write data
            writer = csv.writer(f)
            writer.writerows(data)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # Atomic rename
    temp_path.replace(path)
```

---

## Deployment

### Local Development

```bash
# Install dependencies
pip install streamlit opencv-python pyyaml

# Run app
streamlit run scripts/review_interface/app.py
```

### Production Deployment (Future)

**Options**:
1. **Streamlit Cloud**: Free hosting for public apps
2. **Docker**: Containerized deployment
3. **AWS/Azure**: Cloud deployment with auth

**Docker Example**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "scripts/review_interface/app.py"]
```

---

## Troubleshooting

### Common Issues

**1. Config not found**
```
FileNotFoundError: configs/manual_review_config.yaml
```
**Fix**: Ensure running from project root directory

**2. Video codec issues**
```
cv2.VideoCapture returns None
```
**Fix**: Install OpenCV with ffmpeg support
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

**3. Streamlit session state issues**
```
KeyError: 'video_current_frame'
```
**Fix**: Initialize session state before use
```python
if 'video_current_frame' not in st.session_state:
    st.session_state['video_current_frame'] = 0
```

---

## Future Enhancements

### Planned Features

1. **ML-Based Suggestions** (Phase 2):
   - Analyze similar races for patterns
   - Suggest likely start/finish frames
   - Confidence scoring

2. **Batch Review Mode** (Phase 2):
   - Queue of pending races
   - Auto-advance to next race
   - Keyboard shortcuts

3. **Collaborative Review** (Phase 3):
   - Multi-user support
   - Review history per user
   - Conflict resolution

4. **Pose Overlay** (Phase 2):
   - Show BlazePose keypoints on video
   - Visualize hand reaching top
   - Detect finish by hand position

5. **Audio Analysis** (Phase 3):
   - Detect starting beep (800-1200 Hz)
   - Auto-suggest start frame

### Feature Flags (in config)

```yaml
features:
  ml_suggestions: false          # Disabled for now
  batch_review_mode: false        # Future
  collaborative_review: false     # Future
  pose_overlay: false             # Future
  audio_analysis: false           # Future
```

**Enable when ready**:
```yaml
features:
  ml_suggestions: true  # Enable ML suggestions
```

No code changes needed!

---

## API Reference

*(To be completed after implementation)*

### ConfigManager API

```python
class ConfigManager:
    def __init__(self, config_path: str = "configs/manual_review_config.yaml")
    def get_competitions(self) -> List[CompetitionConfig]
    def get_competition(self, key: str) -> Optional[CompetitionConfig]
    def get_validation_rules(self) -> Dict
    def get_general_settings(self) -> Dict
    def is_feature_enabled(self, feature: str) -> bool
```

### ProgressTracker API

```python
class ProgressTracker:
    def __init__(self, csv_path: str = "data/manual_review/progress_tracker.csv")
    def load_all_races(self) -> List[RaceReviewStatus]
    def get_pending_races(self) -> List[RaceReviewStatus]
    def update_race(self, race: RaceReviewStatus) -> None
    def get_statistics(self) -> Dict
```

*(More to be added after implementation)*

---

## Contributing

### Code Style

- Follow PEP 8
- Use type hints
- Docstrings for all public methods
- Max line length: 100 characters

### Git Workflow

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes
3. Test thoroughly
4. Commit with descriptive message
5. Push and create pull request

### Documentation

- Update this guide when adding features
- Update user guide for UI changes
- Add inline comments for complex logic
- Document breaking changes in CHANGELOG

---

**Note**: This guide will be expanded after UI implementation is complete.
