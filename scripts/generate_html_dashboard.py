"""
HTML Dashboard Generator

Generate interactive HTML dashboard for race analysis.

Usage:
    python scripts/generate_html_dashboard.py
    python scripts/generate_html_dashboard.py --output data/processed/dashboard.html

Author: Speed Climbing Analysis Project
Date: 2025-11-15
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_dashboard_html(
    metrics_csv: str,
    aggregates_dir: str,
    plots_dir: str,
    output_file: str
):
    """Generate interactive HTML dashboard.

    Args:
        metrics_csv: Path to aggregate metrics CSV
        aggregates_dir: Directory with aggregate JSON files
        plots_dir: Directory with plot images
        output_file: Output HTML file path
    """
    logger.info("Generating HTML dashboard...")

    # Load data
    df = pd.read_csv(metrics_csv)

    with open(Path(aggregates_dir) / "overall_statistics.json") as f:
        overall = json.load(f)

    with open(Path(aggregates_dir) / "competition_summaries.json") as f:
        competitions = json.load(f)

    # Create HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speed Climbing Analysis Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}

        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}

        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}

        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .section {{
            padding: 40px;
        }}

        .section-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}

        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}

        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}

        th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }}

        .badge-left {{
            background: #1f77b4;
            color: white;
        }}

        .badge-right {{
            background: #ff7f0e;
            color: white;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧗 Speed Climbing Analysis Dashboard</h1>
            <p class="subtitle">Performance Analytics & Insights</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Races</div>
                <div class="stat-value">{overall['total_races']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Climbers</div>
                <div class="stat-value">{overall['total_climbers']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Max Velocity</div>
                <div class="stat-value">{overall['global_stats']['max_velocity']['mean']:.1f}</div>
                <div class="stat-label">px/s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Path Efficiency</div>
                <div class="stat-value">{overall['global_stats']['path_efficiency']['mean']:.3f}</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">📊 Competition Summary</h2>
            <div class="plot-container">
                <img src="../plots/competition_summary.png" alt="Competition Summary">
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">🏆 Top Performers</h2>
            <div class="plot-container">
                <img src="../plots/leaderboard_top10.png" alt="Leaderboard">
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Race</th>
                        <th>Lane</th>
                        <th>Max Velocity (px/s)</th>
                        <th>Path Efficiency</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Add top 10 rows
    top_df = df.nlargest(10, 'max_vertical_velocity')
    for i, row in enumerate(top_df.itertuples(), 1):
        lane_badge = f'<span class="badge badge-{row.lane}">{row.lane.upper()}</span>'
        html += f"""
                    <tr>
                        <td><strong>{i}</strong></td>
                        <td>{row.race_name.split('_')[-1]}</td>
                        <td>{lane_badge}</td>
                        <td>{row.max_vertical_velocity:.1f}</td>
                        <td>{row.path_efficiency:.3f}</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">📈 Performance Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Max Velocity</strong></td>
                        <td>{0:.1f} px/s</td>
                        <td>{1:.1f} px/s</td>
                        <td>{2:.1f} px/s</td>
                        <td>{3:.1f} px/s</td>
                    </tr>
                    <tr>
                        <td><strong>Path Efficiency</strong></td>
                        <td>{4:.3f}</td>
                        <td>{5:.3f}</td>
                        <td>{6:.3f}</td>
                        <td>{7:.3f}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <footer>
            <p>Generated with Speed Climbing Analysis Pipeline</p>
            <p>Claude Code - Advanced Analytics Dashboard</p>
        </footer>
    </div>
</body>
</html>
""".format(
        overall['global_stats']['max_velocity']['mean'],
        overall['global_stats']['max_velocity']['std'],
        overall['global_stats']['max_velocity']['min'],
        overall['global_stats']['max_velocity']['max'],
        overall['global_stats']['path_efficiency']['mean'],
        overall['global_stats']['path_efficiency']['std'],
        overall['global_stats']['path_efficiency']['min'],
        overall['global_stats']['path_efficiency']['max']
    )

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"✓ Dashboard saved to: {output_path}")
    logger.info(f"  Open in browser: file://{output_path.absolute()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate HTML dashboard"
    )

    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="data/processed/metrics/aggregate_metrics.csv",
        help="Path to metrics CSV"
    )
    parser.add_argument(
        "--aggregates-dir",
        type=str,
        default="data/processed/aggregates",
        help="Directory with aggregate JSON files"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="data/processed/plots",
        help="Directory with plot images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/dashboard/index.html",
        help="Output HTML file"
    )

    args = parser.parse_args()

    generate_dashboard_html(
        args.metrics_csv,
        args.aggregates_dir,
        args.plots_dir,
        args.output
    )


if __name__ == "__main__":
    main()
