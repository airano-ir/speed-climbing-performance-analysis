"""
Interactive inspection tool for 33 valid races (0.5-3.0 m/s).

This script generates a comprehensive HTML report for manual review of each valid race
before proceeding to ML modeling.

Output: data/processed/reports/valid_races_inspection_report.html

Author: Claude Code
Date: 2025-11-17
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import base64

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'aggregated_metrics_reliable.csv'
VALID_RACES_FILE = BASE_DIR / 'data' / 'processed' / 'valid_races_quality_filtered.json'
POSES_DIR = BASE_DIR / 'data' / 'processed' / 'poses'
METRICS_DIR = BASE_DIR / 'data' / 'processed' / 'metrics'
CALIBRATION_DIR = BASE_DIR / 'data' / 'processed' / 'calibration'
SEGMENTS_DIR = BASE_DIR / 'data' / 'race_segments'
OUTPUT_FILE = BASE_DIR / 'data' / 'processed' / 'reports' / 'visualizations' / 'valid_races_inspection_report.html'
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_valid_races():
    """Load valid races list."""
    with open(VALID_RACES_FILE, 'r') as f:
        data = json.load(f)
    return data['race_ids']


def load_race_details(race_id):
    """Load all details for a race."""
    details = {
        'race_id': race_id,
        'has_pose': False,
        'has_metrics': False,
        'has_calibration': False,
        'has_video': False
    }

    # Competition
    for comp_dir in SEGMENTS_DIR.glob('*'):
        if (comp_dir / f"{race_id}.mp4").exists():
            details['competition'] = comp_dir.name
            details['has_video'] = True
            break

    # Pose data
    pose_file = POSES_DIR / f"{race_id}_pose.json"
    if pose_file.exists():
        details['has_pose'] = True
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
            details['pose'] = {
                'total_frames': pose_data.get('total_frames', 0),
                'extracted_frames': pose_data.get('extracted_frames', 0),
                'success_rate': pose_data.get('success_rate', 0),
                'fps': pose_data.get('fps', 30),
                'frame_width': pose_data.get('frame_width', 0),
                'frame_height': pose_data.get('frame_height', 0)
            }

    # Metrics
    metrics_file = METRICS_DIR / f"{race_id}_metrics.json"
    if metrics_file.exists():
        details['has_metrics'] = True
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
            details['metrics'] = {
                'average_velocity_ms': metrics_data.get('average_velocity_ms', 0),
                'max_velocity_ms': metrics_data.get('max_velocity_ms', 0),
                'total_time_s': metrics_data.get('total_time_s', 0),
                'vertical_displacement_m': metrics_data.get('vertical_displacement_m', 0),
                'path_length_m': metrics_data.get('path_length_m', 0),
                'smoothness_score': metrics_data.get('smoothness_score', 0),
                'path_efficiency': metrics_data.get('path_efficiency', 0),
                'frames_analyzed': metrics_data.get('frames_analyzed', 0)
            }

    # Calibration
    cal_file = CALIBRATION_DIR / f"{race_id}_calibration.json"
    if cal_file.exists():
        details['has_calibration'] = True
        with open(cal_file, 'r') as f:
            cal_data = json.load(f)
            details['calibration'] = {
                'type': cal_data.get('calibration_type', 'unknown'),
                'pixel_to_meter': cal_data.get('pixel_to_meter_scale', 0),
                'wall_height_m': cal_data.get('wall_height_m', 15.0)
            }

    return details


def check_quality_issues(details):
    """Check for potential quality issues."""
    issues = []

    # Check pose quality
    if details.get('has_pose'):
        success_rate = details['pose'].get('success_rate', 0)
        if success_rate < 90:
            issues.append(f"Low pose detection rate: {success_rate:.1f}%")

        if details['pose'].get('frame_width', 0) == 0:
            issues.append("Missing frame dimensions")

    # Check metrics
    if details.get('has_metrics'):
        metrics = details['metrics']

        # Velocity check
        if metrics.get('average_velocity_ms', 0) > 2.5:
            issues.append(f"Unusually high velocity: {metrics['average_velocity_ms']:.3f} m/s")

        # Time check
        if metrics.get('total_time_s', 0) < 4:
            issues.append(f"Very short race: {metrics['total_time_s']:.2f}s")
        elif metrics.get('total_time_s', 0) > 15:
            issues.append(f"Very long race: {metrics['total_time_s']:.2f}s")

        # Displacement check
        displacement = metrics.get('vertical_displacement_m', 0)
        if displacement < 10:
            issues.append(f"Low vertical displacement: {displacement:.2f}m (partial climb?)")
        elif displacement > 20:
            issues.append(f"High vertical displacement: {displacement:.2f}m (calibration issue?)")

        # Path efficiency
        efficiency = metrics.get('path_efficiency', 0)
        if efficiency < 0.001:
            issues.append(f"Very low path efficiency: {efficiency:.4f}")

    # Check completeness
    if not details.get('has_pose'):
        issues.append("Missing pose data")
    if not details.get('has_metrics'):
        issues.append("Missing metrics data")
    if not details.get('has_calibration'):
        issues.append("Missing calibration data")
    if not details.get('has_video'):
        issues.append("Missing video file")

    return issues


def generate_html_report(valid_races):
    """Generate HTML inspection report."""

    # Load aggregated metrics for comparison
    df = pd.read_csv(DATA_FILE)
    valid_df = df[df['race_id'].isin(valid_races)].copy()

    # Collect details for all races
    all_details = []
    for race_id in valid_races:
        details = load_race_details(race_id)
        issues = check_quality_issues(details)
        details['issues'] = issues
        details['issue_count'] = len(issues)
        all_details.append(details)

    # Sort by issue count (problematic first)
    all_details.sort(key=lambda x: (-x['issue_count'], -x.get('metrics', {}).get('average_velocity_ms', 0)))

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Valid Races Inspection Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .summary {{
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}

        .stat-card h3 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 5px;
        }}

        .stat-card p {{
            color: #6c757d;
            font-size: 0.9em;
        }}

        .filters {{
            padding: 20px 30px;
            background: #fff;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            gap: 15px;
            align-items: center;
        }}

        .filter-btn {{
            padding: 8px 16px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
        }}

        .filter-btn:hover {{
            background: #667eea;
            color: white;
        }}

        .filter-btn.active {{
            background: #667eea;
            color: white;
        }}

        .races-list {{
            padding: 30px;
        }}

        .race-card {{
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
            transition: all 0.3s;
        }}

        .race-card:hover {{
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}

        .race-header {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}

        .race-title {{
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
        }}

        .race-badges {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}

        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}

        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}

        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}

        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}

        .race-details {{
            padding: 20px;
            display: none;
        }}

        .race-details.show {{
            display: block;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}

        .metric-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }}

        .metric-label {{
            font-size: 0.85em;
            color: #6c757d;
            margin-bottom: 5px;
        }}

        .metric-value {{
            font-size: 1.3em;
            font-weight: 600;
            color: #2c3e50;
        }}

        .issues-section {{
            margin-top: 20px;
            padding: 15px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 5px;
        }}

        .issues-section h4 {{
            color: #856404;
            margin-bottom: 10px;
        }}

        .issues-section ul {{
            list-style-position: inside;
            color: #856404;
        }}

        .no-issues {{
            background: #d4edda;
            border-left-color: #28a745;
        }}

        .no-issues h4 {{
            color: #155724;
        }}

        footer {{
            padding: 20px;
            text-align: center;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Valid Races Inspection Report</h1>
            <p>33 Races Ready for ML Modeling (0.5-3.0 m/s)</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>{len(valid_races)}</h3>
                    <p>Total Valid Races</p>
                </div>
                <div class="stat-card">
                    <h3>{valid_df['average_velocity_ms'].mean():.3f}</h3>
                    <p>Avg Velocity (m/s)</p>
                </div>
                <div class="stat-card">
                    <h3>{valid_df['total_time_s'].mean():.2f}</h3>
                    <p>Avg Duration (s)</p>
                </div>
                <div class="stat-card">
                    <h3>{sum(1 for d in all_details if d['issue_count'] > 0)}</h3>
                    <p>Races with Issues</p>
                </div>
            </div>
        </div>

        <div class="filters">
            <span style="font-weight: 600;">Filter:</span>
            <button class="filter-btn active" onclick="filterRaces('all')">All Races</button>
            <button class="filter-btn" onclick="filterRaces('issues')">With Issues Only</button>
            <button class="filter-btn" onclick="filterRaces('clean')">Clean Only</button>
            <button class="filter-btn" onclick="filterRaces('fast')">Fast (>1.0 m/s)</button>
        </div>

        <div class="races-list" id="races-list">
"""

    # Add race cards
    for idx, details in enumerate(all_details, 1):
        race_id = details['race_id']
        race_short = race_id.replace('Speed_finals_', '')
        metrics = details.get('metrics', {})
        pose = details.get('pose', {})
        issues = details.get('issues', [])

        # Determine status class
        if details['issue_count'] == 0:
            status_class = 'clean'
        elif details['issue_count'] <= 2:
            status_class = 'warning'
        else:
            status_class = 'issues'

        # Velocity class
        velocity = metrics.get('average_velocity_ms', 0)
        velocity_class = 'fast' if velocity > 1.0 else 'normal'

        html += f"""
            <div class="race-card" data-status="{status_class}" data-velocity="{velocity_class}">
                <div class="race-header" onclick="toggleDetails('{race_id}')">
                    <div>
                        <span class="race-title">#{idx}. {race_short}</span>
                    </div>
                    <div class="race-badges">
"""

        # Badges
        if details['issue_count'] == 0:
            html += f'<span class="badge badge-success">Clean</span>'
        else:
            html += f'<span class="badge badge-warning">{details["issue_count"]} Issue{"s" if details["issue_count"] > 1 else ""}</span>'

        if velocity > 1.5:
            html += f'<span class="badge badge-danger">Very Fast</span>'
        elif velocity > 1.0:
            html += f'<span class="badge badge-info">Fast</span>'

        html += f'<span class="badge badge-info">{metrics.get("average_velocity_ms", 0):.3f} m/s</span>'
        html += f'<span class="badge badge-info">{metrics.get("total_time_s", 0):.2f}s</span>'

        html += f"""
                    </div>
                </div>
                <div class="race-details" id="details-{race_id}">
                    <h3>Performance Metrics</h3>
                    <div class="metrics-grid">
                        <div class="metric-box">
                            <div class="metric-label">Average Velocity</div>
                            <div class="metric-value">{metrics.get('average_velocity_ms', 0):.3f} m/s</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Max Velocity</div>
                            <div class="metric-value">{metrics.get('max_velocity_ms', 0):.3f} m/s</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Total Time</div>
                            <div class="metric-value">{metrics.get('total_time_s', 0):.2f} s</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Vertical Displacement</div>
                            <div class="metric-value">{metrics.get('vertical_displacement_m', 0):.2f} m</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Path Length</div>
                            <div class="metric-value">{metrics.get('path_length_m', 0):.2f} m</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Smoothness Score</div>
                            <div class="metric-value">{metrics.get('smoothness_score', 0):.4f}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Path Efficiency</div>
                            <div class="metric-value">{metrics.get('path_efficiency', 0):.4f}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Frames Analyzed</div>
                            <div class="metric-value">{metrics.get('frames_analyzed', 0)}</div>
                        </div>
                    </div>

                    <h3 style="margin-top: 20px;">Data Quality</h3>
                    <div class="metrics-grid">
                        <div class="metric-box">
                            <div class="metric-label">Pose Detection Rate</div>
                            <div class="metric-value">{pose.get('success_rate', 0):.1f}%</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Frame Resolution</div>
                            <div class="metric-value">{pose.get('frame_width', 0)}x{pose.get('frame_height', 0)}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">FPS</div>
                            <div class="metric-value">{pose.get('fps', 0):.1f}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Total Frames</div>
                            <div class="metric-value">{pose.get('total_frames', 0)}</div>
                        </div>
                    </div>
"""

        # Issues section
        if issues:
            html += f"""
                    <div class="issues-section">
                        <h4>Issues Detected ({len(issues)})</h4>
                        <ul>
"""
            for issue in issues:
                html += f'                            <li>{issue}</li>\n'
            html += """
                        </ul>
                    </div>
"""
        else:
            html += """
                    <div class="issues-section no-issues">
                        <h4>No Issues Detected - Ready for ML</h4>
                    </div>
"""

        html += """
                </div>
            </div>
"""

    html += """
        </div>

        <footer>
            <p>Speed Climbing Performance Analysis - Valid Races Inspection Report</p>
            <p>Review each race carefully before proceeding to ML Modeling</p>
        </footer>
    </div>

    <script>
        function toggleDetails(raceId) {
            const details = document.getElementById('details-' + raceId);
            details.classList.toggle('show');
        }

        function filterRaces(filter) {
            const cards = document.querySelectorAll('.race-card');
            const buttons = document.querySelectorAll('.filter-btn');

            // Update button states
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            // Filter cards
            cards.forEach(card => {
                const status = card.dataset.status;
                const velocity = card.dataset.velocity;

                let show = false;

                if (filter === 'all') {
                    show = true;
                } else if (filter === 'issues') {
                    show = (status === 'issues' || status === 'warning');
                } else if (filter === 'clean') {
                    show = (status === 'clean');
                } else if (filter === 'fast') {
                    show = (velocity === 'fast');
                }

                card.style.display = show ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>
"""

    return html


def main():
    """Main execution."""
    print("=" * 70)
    print("VALID RACES INSPECTION TOOL")
    print("Speed Climbing Performance Analysis")
    print("=" * 70)
    print()

    # Load valid races
    print("Loading valid races list...")
    valid_races = load_valid_races()
    print(f"  Found {len(valid_races)} valid races")
    print()

    # Generate report
    print("Generating inspection report...")
    html = generate_html_report(valid_races)

    # Save report
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  Report saved: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")
    print()

    # Summary
    print("=" * 70)
    print("INSPECTION REPORT READY!")
    print("=" * 70)
    print()
    print(f"Open the report in your browser:")
    print(f"  {OUTPUT_FILE}")
    print()
    print("Review each race carefully before proceeding to ML Modeling.")
    print()


if __name__ == '__main__':
    main()
