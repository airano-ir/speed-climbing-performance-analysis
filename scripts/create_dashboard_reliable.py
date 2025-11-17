#!/usr/bin/env python3
"""
Generate Interactive Plotly Dashboard for Speed Climbing Performance Analysis
==============================================================================

This script creates a comprehensive interactive dashboard with:
- Overview page: Summary statistics and top performers
- Performance Analysis: Detailed metrics exploration
- Competition Comparison: Side-by-side competition analysis
- Athlete Tracker: Wall position visualization (2D climbing path)
- Data Explorer: Interactive data table with filters

Output: data/processed/plots/interactive/dashboard_reliable_races.html

Author: Claude AI
Date: 2025-11-17
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Professional color scheme (colorblind-friendly)
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

# Wall dimensions (IFSC standard)
WALL_HEIGHT_M = 15.0
WALL_WIDTH_M = 3.0

# Simplified calibration scale
PIXEL_TO_METER = 0.025  # pixels to meters conversion


def load_data() -> pd.DataFrame:
    """Load aggregated metrics for all reliable races."""
    logger.info("Loading aggregated metrics...")
    df = pd.read_csv('data/processed/aggregated_metrics_reliable.csv')
    logger.info(f"Loaded {len(df)} races")
    return df


def load_pose_data(race_id: str) -> Optional[Dict]:
    """Load pose data for a specific race."""
    # Check in samples directory
    pose_file = Path(f'data/processed/poses/samples/{race_id}_poses.json')

    if not pose_file.exists():
        logger.warning(f"Pose file not found: {pose_file}")
        return None

    try:
        with open(pose_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded pose data for {race_id}: {len(data.get('frames', []))} frames")
        return data
    except Exception as e:
        logger.error(f"Error loading pose data for {race_id}: {e}")
        return None


def calculate_center_of_mass(keypoints: Dict) -> Tuple[float, float]:
    """
    Calculate center of mass from keypoints.
    Uses torso and hip keypoints as main reference points.

    Args:
        keypoints: Dictionary of keypoint names to coordinates

    Returns:
        Tuple of (x, y) coordinates (normalized 0-1)
    """
    # Key body points for COM calculation
    key_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']

    x_coords = []
    y_coords = []

    for point_name in key_points:
        if point_name in keypoints and keypoints[point_name]['confidence'] > 0.5:
            x_coords.append(keypoints[point_name]['x'])
            y_coords.append(keypoints[point_name]['y'])

    # If we don't have enough points, use all available keypoints with high confidence
    if len(x_coords) < 2:
        for kp in keypoints.values():
            if kp['confidence'] > 0.7:
                x_coords.append(kp['x'])
                y_coords.append(kp['y'])

    if not x_coords:
        return 0.5, 0.5  # Default to center if no valid points

    return np.mean(x_coords), np.mean(y_coords)


def normalize_to_meters(landmark_y: float, frame_height: int) -> float:
    """
    Convert normalized landmark Y coordinate to meters from bottom.

    Args:
        landmark_y: Normalized Y coordinate (0-1, where 0 is top)
        frame_height: Frame height in pixels

    Returns:
        Height in meters from bottom of wall
    """
    pixel_y = landmark_y * frame_height
    meters_from_top = pixel_y * PIXEL_TO_METER
    # Invert Y (image coords have origin at top-left, we want bottom origin)
    meters_from_bottom = WALL_HEIGHT_M - meters_from_top
    return meters_from_bottom


def normalize_x_to_meters(landmark_x: float, frame_width: int) -> float:
    """
    Convert normalized landmark X coordinate to meters from center.

    Args:
        landmark_x: Normalized X coordinate (0-1)
        frame_width: Frame width in pixels

    Returns:
        Horizontal position in meters from center (-1.5 to +1.5)
    """
    pixel_x = landmark_x * frame_width
    meters_from_left = pixel_x * PIXEL_TO_METER
    # Center the coordinate (assume wall is centered in frame)
    wall_width_pixels = WALL_WIDTH_M / PIXEL_TO_METER
    frame_center = frame_width / 2
    wall_center_offset = (frame_center * PIXEL_TO_METER) - (WALL_WIDTH_M / 2)
    meters_from_center = meters_from_left - wall_center_offset - (WALL_WIDTH_M / 2)

    # Clamp to wall boundaries
    return np.clip(meters_from_center, -WALL_WIDTH_M/2, WALL_WIDTH_M/2)


def extract_climb_trajectory(pose_data: Dict, climber: str = 'left') -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Extract climb trajectory (COM path) from pose data.

    Args:
        pose_data: Full pose data dictionary
        climber: 'left' or 'right' climber

    Returns:
        Tuple of (x_coords, y_coords, velocities, timestamps)
    """
    metadata = pose_data.get('metadata', {})
    frames = pose_data.get('frames', [])

    frame_height = metadata.get('height', 720)
    frame_width = metadata.get('width', 1280)
    fps = metadata.get('fps', 30.0)

    x_meters = []
    y_meters = []
    timestamps = []

    for frame in frames:
        climber_data = frame.get(f'{climber}_climber')

        if not climber_data or not climber_data.get('has_detection'):
            continue

        keypoints = climber_data.get('keypoints', {})
        if not keypoints:
            continue

        # Calculate center of mass
        com_x, com_y = calculate_center_of_mass(keypoints)

        # Convert to meters
        x_m = normalize_x_to_meters(com_x, frame_width)
        y_m = normalize_to_meters(com_y, frame_height)

        x_meters.append(x_m)
        y_meters.append(y_m)
        timestamps.append(frame.get('timestamp', 0.0))

    # Calculate velocities
    velocities = []
    for i in range(len(y_meters)):
        if i == 0:
            velocities.append(0.0)
        else:
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dy = y_meters[i] - y_meters[i-1]
                dx = x_meters[i] - x_meters[i-1]
                distance = np.sqrt(dx**2 + dy**2)
                velocity = distance / dt
                velocities.append(velocity)
            else:
                velocities.append(0.0)

    return x_meters, y_meters, velocities, timestamps


def create_wall_position_plot(race_id: str, race_metrics: pd.Series) -> go.Figure:
    """
    Create 2D wall position visualization showing athlete's climb trajectory.

    Args:
        race_id: Race identifier
        race_metrics: Series with race metrics

    Returns:
        Plotly figure
    """
    logger.info(f"Creating wall position plot for {race_id}")

    # Load pose data
    pose_data = load_pose_data(race_id)

    fig = go.Figure()

    if pose_data is None:
        # Create empty plot with message
        fig.add_annotation(
            text=f"Pose data not available for {race_id}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['text'])
        )
    else:
        # Extract trajectory
        x_meters, y_meters, velocities, timestamps = extract_climb_trajectory(pose_data, climber='left')

        if len(x_meters) == 0:
            fig.add_annotation(
                text=f"No valid trajectory data for {race_id}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=COLORS['text'])
            )
        else:
            # Add wall background
            fig.add_shape(
                type="rect",
                x0=-WALL_WIDTH_M/2, x1=WALL_WIDTH_M/2,
                y0=0, y1=WALL_HEIGHT_M,
                line=dict(color=COLORS['border'], width=2),
                fillcolor=COLORS['background'],
                opacity=0.2,
                layer='below'
            )

            # Add grid lines every 3 meters
            for h in [3, 6, 9, 12, 15]:
                fig.add_hline(
                    y=h,
                    line_dash="dot",
                    line_color=COLORS['border'],
                    opacity=0.5,
                    annotation_text=f"{h}m",
                    annotation_position="right"
                )

            # Add climb trajectory with velocity coloring
            fig.add_trace(go.Scatter(
                x=x_meters,
                y=y_meters,
                mode='lines+markers',
                marker=dict(
                    size=4,
                    color=velocities,
                    colorscale='RdYlGn_r',  # Red (fast) to Green (slow)
                    showscale=True,
                    colorbar=dict(
                        title="Velocity<br>(m/s)",
                        x=1.15
                    ),
                    cmin=0,
                    cmax=min(3.0, max(velocities) if velocities else 3.0)
                ),
                line=dict(width=2, color=COLORS['primary']),
                name='Climb Path',
                hovertemplate='<b>Position</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Velocity: %{marker.color:.2f} m/s<extra></extra>'
            ))

            # Add start marker
            fig.add_trace(go.Scatter(
                x=[x_meters[0]],
                y=[y_meters[0]],
                mode='markers',
                marker=dict(size=15, color=COLORS['success'], symbol='circle', line=dict(width=2, color='white')),
                name='Start',
                hovertemplate='<b>Start Position</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<extra></extra>'
            ))

            # Add finish marker
            fig.add_trace(go.Scatter(
                x=[x_meters[-1]],
                y=[y_meters[-1]],
                mode='markers',
                marker=dict(size=15, color=COLORS['danger'], symbol='star', line=dict(width=2, color='white')),
                name='Finish',
                hovertemplate='<b>Finish Position</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<extra></extra>'
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Wall Position Visualization - {race_id}<br><sub>Time: {race_metrics.get('total_time_s', 0):.2f}s | Avg Velocity: {race_metrics.get('average_velocity_ms', 0):.2f} m/s</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Horizontal Position (m)",
        yaxis_title="Vertical Height (m)",
        height=800,
        width=600,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            range=[0, WALL_HEIGHT_M + 1]
        ),
        xaxis=dict(
            range=[-WALL_WIDTH_M/2 - 0.5, WALL_WIDTH_M/2 + 0.5]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color=COLORS['text']),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    return fig


def create_overview_page(df: pd.DataFrame) -> go.Figure:
    """Create overview page with summary statistics."""
    logger.info("Creating overview page...")

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Velocity Distribution',
            'Top 10 Performers',
            'Race Duration Distribution',
            'Competition Statistics'
        ),
        specs=[
            [{'type': 'histogram'}, {'type': 'bar'}],
            [{'type': 'box'}, {'type': 'table'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # 1. Velocity distribution
    fig.add_trace(
        go.Histogram(
            x=df['average_velocity_ms'],
            nbinsx=30,
            marker_color=COLORS['primary'],
            name='Velocity Distribution',
            hovertemplate='Velocity: %{x:.2f} m/s<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. Top 10 performers
    top_10 = df.nlargest(10, 'average_velocity_ms')
    fig.add_trace(
        go.Bar(
            x=top_10['average_velocity_ms'],
            y=top_10['race_id'].str[-10:],  # Show last 10 chars for readability
            orientation='h',
            marker_color=COLORS['success'],
            name='Top Performers',
            hovertemplate='<b>%{y}</b><br>Velocity: %{x:.2f} m/s<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Race duration distribution by competition
    competitions = df['competition'].unique()
    for comp in competitions[:5]:  # Show top 5 competitions
        comp_data = df[df['competition'] == comp]
        fig.add_trace(
            go.Box(
                y=comp_data['total_time_s'],
                name=comp,
                hovertemplate='<b>%{fullData.name}</b><br>Time: %{y:.2f}s<extra></extra>'
            ),
            row=2, col=1
        )

    # 4. Competition statistics table
    comp_stats = df.groupby('competition').agg({
        'race_id': 'count',
        'average_velocity_ms': 'mean',
        'total_time_s': 'mean'
    }).reset_index()
    comp_stats.columns = ['Competition', 'Races', 'Avg Velocity (m/s)', 'Avg Time (s)']
    comp_stats = comp_stats.sort_values('Avg Velocity (m/s)', ascending=False).head(10)

    fig.add_trace(
        go.Table(
            header=dict(
                values=list(comp_stats.columns),
                fill_color=COLORS['primary'],
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[comp_stats[col] for col in comp_stats.columns],
                fill_color=COLORS['background'],
                align='left',
                format=[None, None, '.3f', '.2f']
            )
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="Average Velocity (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Average Velocity (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Race ID", row=1, col=2)
    fig.update_yaxes(title_text="Time (s)", row=2, col=1)

    fig.update_layout(
        height=900,
        showlegend=False,
        title_text="<b>Speed Climbing Performance - Overview Dashboard</b>",
        title_x=0.5,
        title_font_size=20,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def create_performance_analysis_page(df: pd.DataFrame) -> go.Figure:
    """Create performance analysis page with detailed metrics."""
    logger.info("Creating performance analysis page...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Velocity vs Time Correlation',
            'Path Efficiency Distribution',
            'Smoothness Score Distribution',
            'Vertical Displacement vs Path Length'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'histogram'}],
            [{'type': 'histogram'}, {'type': 'scatter'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # 1. Velocity vs Time scatter
    fig.add_trace(
        go.Scatter(
            x=df['total_time_s'],
            y=df['average_velocity_ms'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['average_velocity_ms'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Velocity<br>(m/s)", x=1.15, y=0.75, len=0.4)
            ),
            text=df['race_id'],
            hovertemplate='<b>%{text}</b><br>Time: %{x:.2f}s<br>Velocity: %{y:.2f} m/s<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. Path efficiency histogram
    fig.add_trace(
        go.Histogram(
            x=df['path_efficiency'],
            nbinsx=25,
            marker_color=COLORS['secondary'],
            hovertemplate='Efficiency: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Smoothness score histogram
    fig.add_trace(
        go.Histogram(
            x=df['smoothness_score'],
            nbinsx=25,
            marker_color=COLORS['warning'],
            hovertemplate='Smoothness: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Vertical displacement vs Path length
    fig.add_trace(
        go.Scatter(
            x=df['vertical_displacement_m'],
            y=df['path_length_m'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['path_efficiency'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Path<br>Efficiency", x=1.15, y=0.25, len=0.4)
            ),
            text=df['race_id'],
            hovertemplate='<b>%{text}</b><br>Vertical: %{x:.2f}m<br>Path: %{y:.2f}m<extra></extra>'
        ),
        row=2, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="Total Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Average Velocity (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Path Efficiency", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Smoothness Score", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Vertical Displacement (m)", row=2, col=2)
    fig.update_yaxes(title_text="Path Length (m)", row=2, col=2)

    fig.update_layout(
        height=900,
        showlegend=False,
        title_text="<b>Performance Metrics Analysis</b>",
        title_x=0.5,
        title_font_size=20,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def create_competition_comparison_page(df: pd.DataFrame) -> go.Figure:
    """Create competition comparison page."""
    logger.info("Creating competition comparison page...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Average Velocity by Competition',
            'Race Count by Competition',
            'Performance Metrics Heatmap',
            'Time Distribution by Competition'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'heatmap'}, {'type': 'box'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # Group by competition
    comp_stats = df.groupby('competition').agg({
        'average_velocity_ms': 'mean',
        'race_id': 'count',
        'total_time_s': 'mean',
        'path_efficiency': 'mean',
        'smoothness_score': 'mean'
    }).reset_index()
    comp_stats = comp_stats.sort_values('average_velocity_ms', ascending=False)

    # 1. Average velocity by competition
    fig.add_trace(
        go.Bar(
            x=comp_stats['competition'],
            y=comp_stats['average_velocity_ms'],
            marker_color=COLORS['primary'],
            hovertemplate='<b>%{x}</b><br>Avg Velocity: %{y:.2f} m/s<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. Race count by competition
    fig.add_trace(
        go.Bar(
            x=comp_stats['competition'],
            y=comp_stats['race_id'],
            marker_color=COLORS['success'],
            hovertemplate='<b>%{x}</b><br>Races: %{y}<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Performance metrics heatmap
    heatmap_data = comp_stats[['competition', 'average_velocity_ms', 'path_efficiency', 'smoothness_score', 'total_time_s']].head(15)
    heatmap_data_normalized = heatmap_data.copy()
    for col in ['average_velocity_ms', 'path_efficiency', 'smoothness_score']:
        heatmap_data_normalized[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())
    heatmap_data_normalized['total_time_s'] = 1 - (heatmap_data['total_time_s'] - heatmap_data['total_time_s'].min()) / (heatmap_data['total_time_s'].max() - heatmap_data['total_time_s'].min())

    fig.add_trace(
        go.Heatmap(
            x=['Velocity', 'Efficiency', 'Smoothness', 'Time'],
            y=heatmap_data['competition'],
            z=heatmap_data_normalized[['average_velocity_ms', 'path_efficiency', 'smoothness_score', 'total_time_s']].values,
            colorscale='RdYlGn',
            hovertemplate='Competition: %{y}<br>Metric: %{x}<br>Normalized Score: %{z:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Time distribution by competition
    for comp in comp_stats['competition'].head(8):
        comp_data = df[df['competition'] == comp]
        fig.add_trace(
            go.Box(
                y=comp_data['total_time_s'],
                name=comp,
                hovertemplate='<b>%{fullData.name}</b><br>Time: %{y:.2f}s<extra></extra>'
            ),
            row=2, col=2
        )

    # Update axes
    fig.update_xaxes(title_text="Competition", row=1, col=1, tickangle=45)
    fig.update_yaxes(title_text="Avg Velocity (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Competition", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="Number of Races", row=1, col=2)
    fig.update_yaxes(title_text="Competition", row=2, col=1)
    fig.update_yaxes(title_text="Time (s)", row=2, col=2)

    fig.update_layout(
        height=1000,
        showlegend=False,
        title_text="<b>Competition Comparison</b>",
        title_x=0.5,
        title_font_size=20,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def create_athlete_tracker_page(df: pd.DataFrame) -> List[go.Figure]:
    """
    Create athlete tracker page with wall position visualizations.
    Returns a list of figures for races with available pose data.
    """
    logger.info("Creating athlete tracker page...")

    # Get races that have pose data
    available_pose_files = list(Path('data/processed/poses/samples').glob('*_poses.json'))
    available_race_ids = [f.stem.replace('_poses', '') for f in available_pose_files]

    logger.info(f"Found {len(available_race_ids)} races with pose data")

    # Filter to races that are in our dataset and have pose data
    available_races = df[df['race_id'].isin(available_race_ids)]

    if len(available_races) == 0:
        logger.warning("No races with both metrics and pose data found")
        return []

    # Sort by velocity and select top performers with pose data
    selected_races = available_races.nlargest(6, 'average_velocity_ms')

    figures = []
    for _, race_row in selected_races.iterrows():
        race_id = race_row['race_id']
        fig = create_wall_position_plot(race_id, race_row)
        figures.append(fig)

    return figures


def create_dashboard(df: pd.DataFrame) -> str:
    """
    Create complete multi-page dashboard.
    Returns path to saved HTML file.
    """
    logger.info("Creating complete dashboard...")

    # Create output directory
    output_dir = Path('data/processed/plots/interactive')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all pages
    overview_fig = create_overview_page(df)
    performance_fig = create_performance_analysis_page(df)
    comparison_fig = create_competition_comparison_page(df)
    tracker_figs = create_athlete_tracker_page(df)

    # Combine into single HTML with tabs
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Speed Climbing Performance Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background-color: {COLORS['background']};
                color: {COLORS['text']};
            }}

            .header {{
                background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
                color: white;
                padding: 2rem;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            .header h1 {{
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
            }}

            .header p {{
                font-size: 1.1rem;
                opacity: 0.9;
            }}

            .tabs {{
                display: flex;
                justify-content: center;
                background-color: white;
                border-bottom: 2px solid {COLORS['border']};
                position: sticky;
                top: 0;
                z-index: 100;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}

            .tab {{
                padding: 1rem 2rem;
                cursor: pointer;
                border: none;
                background: none;
                font-size: 1rem;
                font-weight: 500;
                color: {COLORS['text']};
                transition: all 0.3s ease;
                border-bottom: 3px solid transparent;
            }}

            .tab:hover {{
                background-color: {COLORS['background']};
            }}

            .tab.active {{
                color: {COLORS['primary']};
                border-bottom-color: {COLORS['primary']};
            }}

            .content {{
                padding: 2rem;
                max-width: 1600px;
                margin: 0 auto;
            }}

            .page {{
                display: none;
            }}

            .page.active {{
                display: block;
            }}

            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }}

            .stat-card {{
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-left: 4px solid {COLORS['primary']};
            }}

            .stat-card h3 {{
                color: {COLORS['text']};
                font-size: 0.9rem;
                text-transform: uppercase;
                margin-bottom: 0.5rem;
                opacity: 0.7;
            }}

            .stat-card .value {{
                font-size: 2rem;
                font-weight: bold;
                color: {COLORS['primary']};
            }}

            .plot-container {{
                background: white;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }}

            .grid-2 {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 2rem;
            }}

            @media (max-width: 768px) {{
                .tabs {{
                    overflow-x: auto;
                }}

                .tab {{
                    padding: 1rem;
                    font-size: 0.9rem;
                }}

                .header h1 {{
                    font-size: 1.8rem;
                }}

                .grid-2 {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧗 Speed Climbing Performance Dashboard</h1>
            <p>Interactive Analysis of {len(df)} Reliable Races</p>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showPage('overview')">📊 Overview</button>
            <button class="tab" onclick="showPage('performance')">📈 Performance</button>
            <button class="tab" onclick="showPage('comparison')">🏆 Competition</button>
            <button class="tab" onclick="showPage('tracker')">🎯 Athlete Tracker</button>
            <button class="tab" onclick="showPage('data')">📋 Data Explorer</button>
        </div>

        <div class="content">
            <!-- Overview Page -->
            <div id="overview" class="page active">
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Total Races</h3>
                        <div class="value">{len(df)}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Average Velocity</h3>
                        <div class="value">{df['average_velocity_ms'].mean():.2f} m/s</div>
                    </div>
                    <div class="stat-card">
                        <h3>Top Velocity</h3>
                        <div class="value">{df['average_velocity_ms'].max():.2f} m/s</div>
                    </div>
                    <div class="stat-card">
                        <h3>Competitions</h3>
                        <div class="value">{df['competition'].nunique()}</div>
                    </div>
                </div>
                <div class="plot-container" id="overview-plot"></div>
            </div>

            <!-- Performance Page -->
            <div id="performance" class="page">
                <div class="plot-container" id="performance-plot"></div>
            </div>

            <!-- Comparison Page -->
            <div id="comparison" class="page">
                <div class="plot-container" id="comparison-plot"></div>
            </div>

            <!-- Athlete Tracker Page -->
            <div id="tracker" class="page">
                <h2 style="margin-bottom: 1.5rem;">Wall Position Visualizations - Top Performers</h2>
                <div class="grid-2">
    """

    # Add wall position plots
    for i, fig in enumerate(tracker_figs):
        html_content += f'<div class="plot-container" id="tracker-plot-{i}"></div>\n'

    html_content += """
                </div>
            </div>

            <!-- Data Explorer Page -->
            <div id="data" class="page">
                <div class="plot-container" id="data-table"></div>
            </div>
        </div>

        <script>
            function showPage(pageId) {
                // Hide all pages
                document.querySelectorAll('.page').forEach(page => {
                    page.classList.remove('active');
                });

                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });

                // Show selected page
                document.getElementById(pageId).classList.add('active');

                // Add active class to clicked tab
                event.target.classList.add('active');
            }

            // Load plots
            const overviewData = """ + overview_fig.to_json() + """;
            Plotly.newPlot('overview-plot', overviewData.data, overviewData.layout);

            const performanceData = """ + performance_fig.to_json() + """;
            Plotly.newPlot('performance-plot', performanceData.data, performanceData.layout);

            const comparisonData = """ + comparison_fig.to_json() + """;
            Plotly.newPlot('comparison-plot', comparisonData.data, comparisonData.layout);
    """

    # Add tracker plots
    for i, fig in enumerate(tracker_figs):
        html_content += f"""
            const trackerData{i} = {fig.to_json()};
            Plotly.newPlot('tracker-plot-{i}', trackerData{i}.data, trackerData{i}.layout);
        """

    # Create data table
    table_data = df[['race_id', 'competition', 'total_time_s', 'average_velocity_ms',
                     'max_velocity_ms', 'vertical_displacement_m', 'path_efficiency',
                     'smoothness_score']].copy()
    table_data.columns = ['Race ID', 'Competition', 'Time (s)', 'Avg Velocity (m/s)',
                          'Max Velocity (m/s)', 'Vertical Disp (m)', 'Path Efficiency',
                          'Smoothness']

    table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(table_data.columns),
            fill_color=COLORS['primary'],
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[table_data[col] for col in table_data.columns],
            fill_color=[COLORS['background']],
            align='left',
            height=30,
            font=dict(size=11)
        )
    )])
    table_fig.update_layout(height=800, margin=dict(l=0, r=0, t=0, b=0))

    html_content += f"""
            const tableData = {table_fig.to_json()};
            Plotly.newPlot('data-table', tableData.data, tableData.layout);
        </script>
    </body>
    </html>
    """

    # Save HTML file
    output_file = output_dir / 'dashboard_reliable_races.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Dashboard saved to: {output_file}")
    return str(output_file)


def main():
    """Main execution function."""
    try:
        logger.info("=" * 80)
        logger.info("Speed Climbing Performance Dashboard Generator")
        logger.info("=" * 80)

        # Load data
        df = load_data()

        # Create dashboard
        output_file = create_dashboard(df)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ Dashboard created successfully!")
        logger.info(f"📊 Output file: {output_file}")
        logger.info(f"📈 Total races: {len(df)}")
        logger.info(f"🏆 Average velocity: {df['average_velocity_ms'].mean():.3f} m/s")
        logger.info(f"⚡ Top velocity: {df['average_velocity_ms'].max():.3f} m/s")
        logger.info("=" * 80)

        print("\n✅ Dashboard generation complete!")
        print(f"📂 Open in browser: {output_file}")

    except Exception as e:
        logger.error(f"❌ Error creating dashboard: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
