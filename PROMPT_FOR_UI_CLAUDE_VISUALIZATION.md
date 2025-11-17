# Task: Implement Interactive Dashboard & UI/UX for Speed Climbing Visualizations

## Context
You are working on a Speed Climbing Performance Analysis project. The Reliable Data Pipeline (Phases 2-7) has been completed successfully in Claude Code environment:
- **114 reliable races processed** (60.6% of 188 total races)
- **26,737 frames extracted** with 97% quality
- **Performance metrics calculated**: 33/114 have valid velocities (0.5-3.0 m/s range)
- **Average velocity**: 0.733 m/s
- **Top performer**: Innsbruck_2024_race019 (1.883 m/s, 16.8m in 8.9s)
- **Data ready** at: `data/processed/aggregated_metrics_reliable.csv`

## Background: What Claude Code Environment Already Did
The following components are **already completed** in the Claude Code environment:
1. ✅ **Directory Structure**: Created in `data/processed/`:
   - `plots/statistical/` - For static PNG/PDF plots
   - `plots/interactive/` - For your HTML dashboards
   - `videos/annotated/` - For annotated videos
   - `reports/visualizations/` - For summary reports

2. ✅ **Statistical Visualizations Script**: `scripts/generate_visualizations_reliable.py`
   - Publication-quality plots using Matplotlib + Seaborn
   - Velocity distributions, competition comparisons, correlations
   - Already generates 10+ static plots

3. ✅ **Video Generation Script**: `scripts/generate_annotated_videos_reliable.py`
   - Generates annotated videos with pose overlay for top performers
   - Based on `notebooks/01_phase1_pose_estimation.ipynb` implementation

## Your Task: Interactive Dashboard & UI/UX
You will create the **interactive web-based visualization layer** that complements the static analysis scripts.

### Primary Deliverable
Create `scripts/create_dashboard_reliable.py` that generates an interactive Plotly dashboard.

### Core Features Required

#### 1. Multi-Page Dashboard Structure
Create a standalone HTML file with multiple pages/tabs:
- **Overview Page**: Summary statistics and top performers
- **Performance Analysis**: Detailed metrics exploration
- **Competition Comparison**: Side-by-side competition analysis
- **Athlete Tracker**: Individual race analysis with position visualization
- **Data Explorer**: Interactive data table with filters

#### 2. **Athlete Position Visualization (PRIORITY FEATURE)**
This is the most important unique feature you must implement:

**2D Wall Position Visualization**:
- Display a 2D representation of the 15-meter climbing wall
- Show athlete's climb trajectory (path of Center of Mass)
- Visualize progression from start (bottom) to finish (top)
- Color-code the path by velocity (green=slow, yellow=medium, red=fast)

**Technical Specifications**:
```python
# Wall dimensions
wall_height_m = 15.0
wall_width_m = 3.0  # Standard IFSC wall width

# Coordinate system
# - Y-axis: 0 (bottom/start) to 15m (top/finish)
# - X-axis: -1.5m to +1.5m (centered)

# Data needed from metrics file
- COM positions over time (x, y coordinates)
- Velocity at each position
- Start and finish positions
- Hold positions (if available from calibration data)
```

**Visualization Elements**:
1. **Wall Outline**: Gray rectangle representing the climbing wall
2. **Grid Lines**: Horizontal lines every 3 meters (3m, 6m, 9m, 12m, 15m marks)
3. **Climb Path**: Line plot showing COM trajectory with gradient coloring
4. **Start/Finish Markers**:
   - Green circle at start position
   - Red circle at finish position
5. **Current Position Indicator** (if interactive playback):
   - Animated marker showing progression
6. **Velocity Legend**: Color scale showing velocity ranges
7. **Hold Positions** (optional, if calibration data available):
   - Small circles at detected hold positions

**Example Implementation Pattern**:
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_wall_position_plot(race_data, metrics):
    # Extract COM trajectory from pose data
    com_x = race_data['com_positions_x']  # Normalized x coordinates
    com_y = race_data['com_positions_y']  # Normalized y coordinates
    velocities = race_data['velocities']  # Velocity at each frame

    # Convert to meters (use frame dimensions and calibration)
    frame_height = race_data['frame_height']
    frame_width = race_data['frame_width']
    pixel_to_meter = metrics['pixel_to_meter_scale']

    # Create wall background
    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=-1.5, x1=1.5, y0=0, y1=15,
        line=dict(color="gray", width=2),
        fillcolor="lightgray", opacity=0.1
    )

    # Add grid lines
    for h in [3, 6, 9, 12, 15]:
        fig.add_hline(y=h, line_dash="dot", line_color="gray", opacity=0.5)

    # Add climb trajectory with velocity coloring
    fig.add_trace(go.Scatter(
        x=com_x_meters,
        y=com_y_meters,
        mode='lines+markers',
        marker=dict(
            size=5,
            color=velocities,
            colorscale='RdYlGn_r',  # Red (fast) to Green (slow)
            showscale=True,
            colorbar=dict(title="Velocity (m/s)")
        ),
        line=dict(width=2),
        name='Climb Path'
    ))

    # Add start marker
    fig.add_trace(go.Scatter(
        x=[com_x_meters[0]],
        y=[com_y_meters[0]],
        mode='markers',
        marker=dict(size=15, color='green', symbol='circle'),
        name='Start'
    ))

    # Add finish marker
    fig.add_trace(go.Scatter(
        x=[com_x_meters[-1]],
        y=[com_y_meters[-1]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Finish'
    ))

    fig.update_layout(
        title=f"Wall Position - {race_data['race_id']}",
        xaxis_title="Horizontal Position (m)",
        yaxis_title="Vertical Height (m)",
        height=800,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return fig
```

#### 3. Competition Comparison Tools
- **Box plots**: Velocity distribution by competition
- **Scatter plots**: Performance metrics correlation
- **Bar charts**: Average metrics by competition
- **Interactive filters**:
  - Select/deselect competitions
  - Filter by velocity range
  - Filter by race duration

#### 4. Performance Metrics Explorer
- **Correlation Heatmap**: All metrics correlation (interactive)
- **Scatter Matrix**: Multi-dimensional analysis
- **Time Series**: Velocity profile over race duration
- **Distribution Plots**: Histograms and KDE plots for each metric

### UI/UX Requirements

#### Design Principles
1. **Responsive Design**: Must work on desktop (1920x1080) and tablet (1024x768)
2. **Professional Appearance**:
   - Use consistent color scheme
   - Clean, modern layout
   - Professional typography
3. **Accessibility**:
   - Colorblind-friendly palettes (use ColorBrewer schemes)
   - High contrast text
   - Clear labels and legends
4. **Performance**:
   - Fast loading (<3 seconds for initial dashboard)
   - Smooth interactions (<100ms response)
   - Lazy loading for heavy components

#### Color Scheme
Use this professional, colorblind-friendly palette:
```python
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'success': '#06A77D',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'background': '#F6F8FA',   # Light gray
    'text': '#24292E',         # Dark gray
    'border': '#D1D5DB'        # Medium gray
}
```

#### Interactive Controls
Must include:
1. **Dropdowns**:
   - Competition selection
   - Metric selection for axes
   - Race selection for detailed view
2. **Sliders**:
   - Velocity range filter
   - Time range filter
   - Frame playback (for position animation)
3. **Checkboxes**:
   - Show/hide competitions
   - Toggle grid lines
   - Show/hide hold positions
4. **Hover Tooltips**:
   - Display full race info on hover
   - Show exact values on plot hover
   - Format: "Race: {race_id}<br>Velocity: {velocity:.3f} m/s<br>Time: {time:.2f} s"

#### Export Capabilities
Provide export buttons for:
1. **Plot Export**: PNG, PDF, SVG formats
2. **Data Export**: CSV, JSON, Excel formats
3. **Report Export**: Full HTML report with all visualizations

### Data Access

#### Input Data
```python
# Main dataset
import pandas as pd
df = pd.read_csv('data/processed/aggregated_metrics_reliable.csv')

# Columns available:
# - race_id, competition, total_time_s, average_velocity_ms, max_velocity_ms
# - vertical_displacement_m, path_length_m, straight_distance_m
# - path_efficiency, smoothness_score, frames_analyzed
# - is_calibrated, calibration_type, calculation_date

# For detailed position data (needed for wall visualization):
import json
pose_file = f'data/processed/poses/{race_id}_pose.json'
metrics_file = f'data/processed/metrics/{race_id}_metrics.json'

# Pose file structure:
# {
#   "race_id": "Speed_finals_Chamonix_2024_race001",
#   "frames": [
#     {
#       "frame_number": 45,
#       "timestamp": 1.5,
#       "landmarks": [{"x": 0.344, "y": 0.506, "z": 0.088, "visibility": 0.999}, ...]
#     }
#   ],
#   "fps": 30.0,
#   "frame_width": 1280,
#   "frame_height": 720
# }
```

#### Coordinate Conversion
MediaPipe landmarks are in normalized coordinates (0-1). Convert to meters:
```python
def normalize_to_meters(landmark_y, frame_height, pixel_to_meter):
    """Convert normalized landmark to meters (vertical)."""
    pixel_y = landmark_y * frame_height
    meters_from_top = pixel_y * pixel_to_meter
    # Invert Y (image coords have origin at top-left)
    meters_from_bottom = 15.0 - meters_from_top
    return meters_from_bottom

# Get pixel_to_meter from calibration file
cal_file = f'data/processed/calibration/{race_id}_calibration.json'
pixel_to_meter = calibration['pixel_to_meter_scale']  # 0.025 for simplified
```

### Implementation Guidelines

#### Script Structure
```python
"""
Generate interactive Plotly dashboard for 114 reliable races.
Output: data/processed/plots/interactive/dashboard_reliable_races.html
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

def load_data():
    """Load aggregated metrics and race details."""
    pass

def create_overview_page():
    """Summary statistics and top performers."""
    pass

def create_performance_page():
    """Detailed metrics exploration."""
    pass

def create_comparison_page():
    """Competition comparison tools."""
    pass

def create_athlete_tracker_page():
    """Wall position visualization + race playback."""
    pass

def create_data_explorer_page():
    """Interactive data table."""
    pass

def create_dashboard():
    """Combine all pages into single HTML dashboard."""
    # Use Plotly Dash or pure Plotly with tabs
    pass

if __name__ == '__main__':
    dashboard = create_dashboard()
    dashboard.write_html('data/processed/plots/interactive/dashboard_reliable_races.html')
    print("✅ Dashboard created successfully!")
```

#### Testing & Validation
After creating the dashboard:
1. Open in browser (Chrome/Firefox)
2. Test all interactive controls
3. Verify athlete position visualization works
4. Check export functionality
5. Test on different screen sizes
6. Validate data accuracy (spot-check 3-5 races)

### Expected Output
A single file: `data/processed/plots/interactive/dashboard_reliable_races.html`
- Size: 2-5 MB (including embedded Plotly.js)
- Fully standalone (no external dependencies)
- Works offline
- Professional appearance
- All features functional

### Success Criteria
✅ Dashboard loads in <3 seconds
✅ All 114 races displayed correctly
✅ Athlete position visualization shows 2D wall view
✅ Interactive filters work smoothly
✅ Export functions work (PNG, CSV)
✅ Responsive on desktop and tablet
✅ Colorblind-friendly color scheme used
✅ No errors in browser console

### Reference Materials
- **Plotly Dash Documentation**: https://dash.plotly.com/
- **Plotly Python**: https://plotly.com/python/
- **Color Schemes**: ColorBrewer (https://colorbrewer2.org/)
- **MediaPipe BlazePose**: 33 landmarks (https://google.github.io/mediapipe/solutions/pose.html)

### Notes
- Focus on **clarity and usability** for non-technical users
- Add **error handling** for missing data (some races may have incomplete metrics)
- Include **progress indicators** for slow operations
- **Document all visualization choices** in code comments
- Keep code **modular and maintainable** for future enhancements

### Questions to Consider
1. Should the wall position visualization support animation (frame-by-frame playback)?
2. Should we include a "race replay" feature with synchronized video?
3. Do you want athlete comparison view (2+ climbers on same wall)?
4. Should we add ML predictions/insights (e.g., predicted finish time)?

If you have questions or need clarification on any requirements, please ask before implementation!

---

**Ready to start? Create the dashboard and make the data come alive! 🚀**
