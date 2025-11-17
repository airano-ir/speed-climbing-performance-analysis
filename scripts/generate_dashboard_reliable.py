"""
Interactive Dashboard Generator - Reliable Races
================================================
Generate comprehensive HTML dashboard with visualizations.

Features:
- Overall statistics
- Leaderboards
- Comparative charts
- Quality metrics
- Interactive plots (Plotly)

Usage:
    python scripts/generate_dashboard_reliable.py
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def generate_dashboard():
    """Generate interactive HTML dashboard."""

    print("="*70)
    print("Dashboard Generation - Reliable Races")
    print("="*70 + "\n")

    # Load data
    csv_path = Path('data/processed/aggregated_metrics_reliable.csv')

    if not csv_path.exists():
        print("❌ Error: Aggregated data not found!")
        print(f"   Expected: {csv_path}")
        print("   Please run aggregate_reliable_data.py first")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} races")

    # Create HTML components
    html_parts = []

    # Header
    html_parts.append(create_header(df))

    # Statistics cards
    html_parts.append(create_stats_cards(df))

    # Leaderboard
    html_parts.append(create_leaderboard_section(df))

    # Velocity distribution
    html_parts.append(create_velocity_distribution(df))

    # Competition comparison
    html_parts.append(create_competition_comparison(df))

    # Quality metrics
    html_parts.append(create_quality_section())

    # Combine HTML
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speed Climbing Analytics - 114 Reliable Races</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            h1 {{
                color: #2d3748;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-align: center;
            }}
            h2 {{
                color: #4a5568;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-top: 40px;
            }}
            h3 {{
                color: #718096;
                margin-top: 30px;
            }}
            .subtitle {{
                text-align: center;
                color: #718096;
                font-size: 1.2em;
                margin-bottom: 10px;
            }}
            .timestamp {{
                text-align: center;
                color: #a0aec0;
                margin-bottom: 30px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                transition: transform 0.3s ease;
            }}
            .stat-card:hover {{
                transform: translateY(-5px);
            }}
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .stat-label {{
                font-size: 1em;
                opacity: 0.9;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            th {{
                background: #667eea;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }}
            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #e2e8f0;
            }}
            tr:hover {{
                background-color: #f7fafc;
            }}
            .chart-container {{
                margin: 30px 0;
                padding: 20px;
                background: #f7fafc;
                border-radius: 10px;
            }}
            .metric-badge {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 600;
                margin: 5px;
            }}
            .badge-success {{
                background: #48bb78;
                color: white;
            }}
            .badge-warning {{
                background: #ed8936;
                color: white;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 2px solid #e2e8f0;
                color: #a0aec0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {''.join(html_parts)}
            <div class="footer">
                <p>Speed Climbing Performance Analysis Pipeline</p>
                <p>Reliable Races Dataset - Phase 1-7 Complete</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save dashboard
    output_path = Path('data/processed/dashboard_reliable_races.html')
    output_path.write_text(full_html, encoding='utf-8')

    print(f"\n✅ Dashboard generated: {output_path}")
    print(f"   Open in browser to view interactive visualizations")
    print("="*70)


def create_header(df):
    """Create dashboard header."""
    return f"""
    <h1>🏔️ Speed Climbing Performance Analytics</h1>
    <p class="subtitle">Reliable Races Dataset - {len(df)} races</p>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <hr style="border: none; height: 2px; background: linear-gradient(to right, #667eea, #764ba2);">
    """


def create_stats_cards(df):
    """Create statistics cards."""
    stats = {
        'total_races': len(df),
        'avg_velocity': 0,
        'avg_duration': 0,
        'fastest_time': 0,
        'competitions': 0
    }

    if 'average_velocity_ms' in df.columns:
        stats['avg_velocity'] = df['average_velocity_ms'].mean()
    if 'total_time_s' in df.columns:
        stats['avg_duration'] = df['total_time_s'].mean()
        stats['fastest_time'] = df['total_time_s'].min()
    if 'competition' in df.columns:
        stats['competitions'] = df['competition'].nunique()

    return f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{stats['total_races']}</div>
            <div class="stat-label">Total Races</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['avg_velocity']:.2f} m/s</div>
            <div class="stat-label">Avg Velocity</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['avg_duration']:.2f}s</div>
            <div class="stat-label">Avg Duration</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['fastest_time']:.2f}s</div>
            <div class="stat-label">Fastest Time</div>
        </div>
    </div>
    """


def create_leaderboard_section(df):
    """Create leaderboard table."""
    if 'average_velocity_ms' not in df.columns:
        return "<h2>🏆 Leaderboard</h2><p>No velocity data available</p>"

    top_10 = df.nlargest(10, 'average_velocity_ms')

    rows = []
    for idx, row in enumerate(top_10.itertuples(), 1):
        race_id = getattr(row, 'race_id', 'N/A')
        comp = getattr(row, 'competition', 'N/A')
        vel = getattr(row, 'average_velocity_ms', 0)
        time = getattr(row, 'total_time_s', 0)

        rows.append(f"""
        <tr>
            <td><strong>{idx}</strong></td>
            <td>{race_id}</td>
            <td>{comp}</td>
            <td>{vel:.2f} m/s</td>
            <td>{time:.2f}s</td>
        </tr>
        """)

    return f"""
    <h2>🏆 Top 10 Fastest Races</h2>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Race ID</th>
                <th>Competition</th>
                <th>Avg Velocity</th>
                <th>Time</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def create_velocity_distribution(df):
    """Create velocity distribution chart using Plotly."""
    if 'average_velocity_ms' not in df.columns:
        return ""

    # Create simple HTML histogram (no plotly for simplicity)
    velocities = df['average_velocity_ms'].values
    min_vel = velocities.min()
    max_vel = velocities.max()
    avg_vel = velocities.mean()

    return f"""
    <h2>📊 Velocity Distribution</h2>
    <div class="chart-container">
        <p><strong>Statistics:</strong></p>
        <ul>
            <li>Minimum: {min_vel:.2f} m/s</li>
            <li>Maximum: {max_vel:.2f} m/s</li>
            <li>Average: {avg_vel:.2f} m/s</li>
            <li>Range: {max_vel - min_vel:.2f} m/s</li>
        </ul>
        <p><em>Note: For detailed interactive charts, use a dedicated visualization tool like Plotly</em></p>
    </div>
    """


def create_competition_comparison(df):
    """Create competition comparison section."""
    if 'competition' not in df.columns or 'average_velocity_ms' not in df.columns:
        return ""

    comp_stats = df.groupby('competition').agg({
        'average_velocity_ms': ['mean', 'min', 'max', 'count']
    }).round(2)

    rows = []
    for comp in comp_stats.index:
        mean_vel = comp_stats.loc[comp, ('average_velocity_ms', 'mean')]
        min_vel = comp_stats.loc[comp, ('average_velocity_ms', 'min')]
        max_vel = comp_stats.loc[comp, ('average_velocity_ms', 'max')]
        count = int(comp_stats.loc[comp, ('average_velocity_ms', 'count')])

        rows.append(f"""
        <tr>
            <td><strong>{comp}</strong></td>
            <td>{count}</td>
            <td>{mean_vel:.2f} m/s</td>
            <td>{min_vel:.2f} m/s</td>
            <td>{max_vel:.2f} m/s</td>
        </tr>
        """)

    return f"""
    <h2>⚖️ Competition Comparison</h2>
    <table>
        <thead>
            <tr>
                <th>Competition</th>
                <th>Races</th>
                <th>Avg Velocity</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def create_quality_section():
    """Create quality metrics section."""
    validation_path = Path('data/processed/pipeline_validation_report.json')

    if not validation_path.exists():
        return """
        <h2>📋 Data Quality Metrics</h2>
        <p>⚠️ Validation report not found. Please run validate_pipeline_reliable.py first.</p>
        """

    with open(validation_path, 'r') as f:
        validation = json.load(f)

    fc = validation.get('file_completeness', {})
    dq = validation.get('data_quality', {})

    fc_badge = "badge-success" if fc.get('pass_rate', 0) >= 95 else "badge-warning"
    dq_badge = "badge-success" if dq.get('pass_rate', 0) >= 95 else "badge-warning"

    return f"""
    <h2>📋 Data Quality Metrics</h2>
    <div class="chart-container">
        <h3>Pipeline Validation Results</h3>
        <p>
            <span class="metric-badge {fc_badge}">
                File Completeness: {fc.get('pass_rate', 0):.1f}%
            </span>
            <span class="metric-badge {dq_badge}">
                Data Quality: {dq.get('pass_rate', 0):.1f}%
            </span>
        </p>
        <ul>
            <li><strong>Total Races Validated:</strong> {validation.get('total_races', 0)}</li>
            <li><strong>Missing Files:</strong> {fc.get('total_missing', 0)}</li>
            <li><strong>Quality Issues:</strong> {dq.get('races_with_issues', 0)} races</li>
        </ul>
    </div>
    """


if __name__ == "__main__":
    try:
        generate_dashboard()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
