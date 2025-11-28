"""
Generate comprehensive HTML report for ML dataset.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_data(ml_dir: Path):
    """Load all data files."""
    data = {
        'features': pd.read_csv(ml_dir / 'all_features.csv'),
        'quality': pd.read_csv(ml_dir / 'quality_report.csv'),
        'statistics': pd.read_csv(ml_dir / 'feature_statistics.csv'),
        'outliers': pd.read_csv(ml_dir / 'outliers_report.csv'),
    }
    return data


def generate_html_report(data: dict, output_path: Path, plots_dir: Path):
    """Generate comprehensive HTML report."""
    features_df = data['features']
    quality_df = data['quality']
    stats_df = data['statistics']
    outliers_df = data['outliers']

    # Calculate summary statistics
    total_samples = len(features_df)
    total_features = len(stats_df)

    # Quality summary
    ok_count = (quality_df['status'] == 'OK').sum()
    warning_count = (quality_df['status'] == 'WARNING').sum()
    error_count = (quality_df['status'] == 'ERROR').sum()

    # Competition summary
    quality_df['competition'] = quality_df['video_id'].apply(
        lambda x: '_'.join(x.replace('_poses', '').split('_')[2:4]) if len(x.split('_')) >= 4 else 'unknown'
    )
    comp_summary = quality_df.groupby('competition').agg({
        'status': lambda x: (x == 'OK').sum(),
        'video_id': 'count',
        'extraction_quality': 'mean',
        'race_segment_confidence': 'mean'
    }).rename(columns={'status': 'ok_count', 'video_id': 'total'})
    comp_summary['warning_count'] = quality_df.groupby('competition')['status'].apply(lambda x: (x == 'WARNING').sum())

    # Feature categories
    freq_features = [f for f in stats_df['feature'] if f.startswith('freq_')]
    eff_features = [f for f in stats_df['feature'] if f.startswith('eff_')]
    post_features = [f for f in stats_df['feature'] if f.startswith('post_')]

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speed Climbing ML Dataset Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; font-size: 2em; }}
        h2 {{ color: #34495e; margin: 30px 0 15px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h3 {{ color: #2980b9; margin: 20px 0 10px; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-box.warning {{ background: linear-gradient(135deg, #f39c12, #d68910); }}
        .stat-box.success {{ background: linear-gradient(135deg, #27ae60, #1e8449); }}
        .stat-box.error {{ background: linear-gradient(135deg, #e74c3c, #c0392b); }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; font-weight: 500; }}
        tr:hover {{ background: #f8f9fa; }}
        .progress-bar {{
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 10px;
        }}
        .progress-ok {{ background: #27ae60; }}
        .progress-warning {{ background: #f39c12; }}
        .progress-error {{ background: #e74c3c; }}
        .img-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .img-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .feature-table td:nth-child(2),
        .feature-table td:nth-child(3),
        .feature-table td:nth-child(4),
        .feature-table td:nth-child(5) {{
            text-align: right;
            font-family: monospace;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        .badge-ok {{ background: #27ae60; color: white; }}
        .badge-warning {{ background: #f39c12; color: white; }}
        .badge-error {{ background: #e74c3c; color: white; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Speed Climbing ML Dataset Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Overview</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{total_samples}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{total_features}</div>
                <div class="stat-label">ML Features</div>
            </div>
            <div class="stat-box success">
                <div class="stat-value">{ok_count}</div>
                <div class="stat-label">OK Extractions</div>
            </div>
            <div class="stat-box warning">
                <div class="stat-value">{warning_count}</div>
                <div class="stat-label">Warnings</div>
            </div>
        </div>

        <div class="card">
            <h3>Dataset Quality</h3>
            <div class="progress-bar" style="margin: 15px 0;">
                <div class="progress-fill progress-ok" style="width: {ok_count/len(quality_df)*100:.1f}%; float: left;"></div>
                <div class="progress-fill progress-warning" style="width: {warning_count/len(quality_df)*100:.1f}%; float: left;"></div>
                <div class="progress-fill progress-error" style="width: {error_count/len(quality_df)*100:.1f}%; float: left;"></div>
            </div>
            <p>
                <span class="badge badge-ok">OK: {ok_count} ({ok_count/len(quality_df)*100:.1f}%)</span>
                <span class="badge badge-warning">Warning: {warning_count} ({warning_count/len(quality_df)*100:.1f}%)</span>
                <span class="badge badge-error">Error: {error_count} ({error_count/len(quality_df)*100:.1f}%)</span>
            </p>
        </div>

        <h2>By Competition</h2>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Competition</th>
                        <th>Total</th>
                        <th>OK</th>
                        <th>Warning</th>
                        <th>Avg Quality</th>
                        <th>Avg Confidence</th>
                    </tr>
                </thead>
                <tbody>
"""

    for comp, row in comp_summary.iterrows():
        html += f"""                    <tr>
                        <td><strong>{comp}</strong></td>
                        <td>{int(row['total'])}</td>
                        <td><span class="badge badge-ok">{int(row['ok_count'])}</span></td>
                        <td><span class="badge badge-warning">{int(row['warning_count'])}</span></td>
                        <td>{row['extraction_quality']:.1%}</td>
                        <td>{row['race_segment_confidence']:.1%}</td>
                    </tr>
"""

    html += """                </tbody>
            </table>
        </div>

        <h2>Feature Statistics</h2>
        <div class="card">
            <p>Total: {total_features} features ({n_freq} frequency, {n_eff} efficiency, {n_post} posture)</p>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>CV</th>
                    </tr>
                </thead>
                <tbody>
""".format(total_features=total_features, n_freq=len(freq_features),
           n_eff=len(eff_features), n_post=len(post_features))

    for _, row in stats_df.iterrows():
        cv_class = 'badge-warning' if abs(row['cv']) > 0.5 else ''
        html += f"""                    <tr>
                        <td>{row['feature']}</td>
                        <td>{row['mean']:.4f}</td>
                        <td>{row['std']:.4f}</td>
                        <td>{row['min']:.4f}</td>
                        <td>{row['max']:.4f}</td>
                        <td><span class="badge {cv_class}">{row['cv']:.2f}</span></td>
                    </tr>
"""

    html += """                </tbody>
            </table>
        </div>

        <h2>Visualizations</h2>
        <div class="two-col">
            <div class="card">
                <h3>Quality Distribution</h3>
                <div class="img-container">
                    <img src="plots/quality_distribution.png" alt="Quality Distribution">
                </div>
            </div>
            <div class="card">
                <h3>Competition Comparison</h3>
                <div class="img-container">
                    <img src="plots/competition_comparison.png" alt="Competition Comparison">
                </div>
            </div>
        </div>

        <div class="card">
            <h3>Feature Distributions</h3>
            <div class="img-container">
                <img src="plots/feature_distributions.png" alt="Feature Distributions">
            </div>
        </div>

        <div class="card">
            <h3>Correlation Matrix</h3>
            <div class="img-container">
                <img src="plots/correlation_matrix.png" alt="Correlation Matrix">
            </div>
        </div>

        <h2>Problematic Videos</h2>
        <div class="card">
            <p>Total: """ + str(warning_count) + """ videos need review</p>
            <table>
                <thead>
                    <tr>
                        <th>Video ID</th>
                        <th>Lane</th>
                        <th>Competition</th>
                        <th>Quality</th>
                        <th>Confidence</th>
                        <th>Frames</th>
                        <th>Issues</th>
                    </tr>
                </thead>
                <tbody>
"""

    problematic = quality_df[quality_df['status'] != 'OK'].sort_values('extraction_quality')
    for _, row in problematic.iterrows():
        html += f"""                    <tr>
                        <td>{row['video_id'].replace('_poses', '')}</td>
                        <td>{row['lane']}</td>
                        <td>{row['competition']}</td>
                        <td>{row['extraction_quality']:.1%}</td>
                        <td>{row['race_segment_confidence']:.1%}</td>
                        <td>{row['racing_frames']}</td>
                        <td><span class="badge badge-warning">{row['issues']}</span></td>
                    </tr>
"""

    html += """                </tbody>
            </table>
        </div>

        <h2>Outlier Analysis</h2>
        <div class="card">
            <p>Total outliers detected (|z| > 3): """ + str(len(outliers_df)) + """</p>
            <h3>Outliers by Feature</h3>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
"""

    if len(outliers_df) > 0:
        outlier_counts = outliers_df['feature'].value_counts().head(10)
        for feat, count in outlier_counts.items():
            html += f"""                    <tr>
                        <td>{feat}</td>
                        <td>{count}</td>
                    </tr>
"""

    html += """                </tbody>
            </table>
        </div>

        <footer style="text-align: center; padding: 20px; color: #7f8c8d; margin-top: 40px;">
            <p>Speed Climbing Performance Analysis - ML Dataset Report</p>
            <p>Phase 3: Feature Extraction Complete</p>
        </footer>
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report saved to: {output_path}")


def main():
    """Main entry point."""
    ml_dir = project_root / 'data' / 'ml_dataset'
    plots_dir = ml_dir / 'plots'
    output_path = ml_dir / 'dataset_report.html'

    print("Loading data...")
    data = load_data(ml_dir)

    print("Generating HTML report...")
    generate_html_report(data, output_path, plots_dir)

    print("Done!")


if __name__ == '__main__':
    main()
