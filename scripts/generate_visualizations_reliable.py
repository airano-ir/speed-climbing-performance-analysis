"""
Generate statistical visualizations for 114 reliable races.

This script creates publication-quality static plots using Matplotlib and Seaborn
for the Speed Climbing Performance Analysis project.

Output: PNG/PDF files in data/processed/plots/statistical/

Author: Claude Code
Date: 2025-11-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.2)

# Professional color palette (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D'
}

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'aggregated_metrics_reliable.csv'
OUTPUT_DIR = BASE_DIR / 'data' / 'processed' / 'plots' / 'statistical'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load aggregated metrics."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df)} races")
    return df


def plot_velocity_distribution(df):
    """Plot 1: Velocity distribution with KDE."""
    print("Creating velocity distribution plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Average velocity
    ax1.hist(df['average_velocity_ms'], bins=20, alpha=0.7,
             color=COLORS['primary'], edgecolor='black')
    ax1.axvline(df['average_velocity_ms'].mean(), color=COLORS['danger'],
                linestyle='--', linewidth=2, label=f"Mean: {df['average_velocity_ms'].mean():.3f} m/s")
    ax1.axvline(df['average_velocity_ms'].median(), color=COLORS['success'],
                linestyle='--', linewidth=2, label=f"Median: {df['average_velocity_ms'].median():.3f} m/s")
    ax1.set_xlabel('Average Velocity (m/s)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Average Velocity Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Max velocity
    ax2.hist(df['max_velocity_ms'], bins=20, alpha=0.7,
             color=COLORS['secondary'], edgecolor='black')
    ax2.axvline(df['max_velocity_ms'].mean(), color=COLORS['danger'],
                linestyle='--', linewidth=2, label=f"Mean: {df['max_velocity_ms'].mean():.3f} m/s")
    ax2.axvline(df['max_velocity_ms'].median(), color=COLORS['success'],
                linestyle='--', linewidth=2, label=f"Median: {df['max_velocity_ms'].median():.3f} m/s")
    ax2.set_xlabel('Max Velocity (m/s)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Max Velocity Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'velocity_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'velocity_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'velocity_distribution.png'}")


def plot_competition_comparison(df):
    """Plot 2: Competition comparison box plots."""
    print("Creating competition comparison plot...")

    # Filter out 'unknown' competition
    df_comp = df[df['competition'] != 'unknown'].copy()

    if len(df_comp) == 0:
        print("  WARNING: All competitions are 'unknown', skipping competition comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Average velocity by competition
    sns.boxplot(data=df_comp, x='competition', y='average_velocity_ms',
                ax=axes[0, 0], palette='colorblind')
    axes[0, 0].set_title('Average Velocity by Competition')
    axes[0, 0].set_xlabel('Competition')
    axes[0, 0].set_ylabel('Average Velocity (m/s)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # Total time by competition
    sns.boxplot(data=df_comp, x='competition', y='total_time_s',
                ax=axes[0, 1], palette='colorblind')
    axes[0, 1].set_title('Total Time by Competition')
    axes[0, 1].set_xlabel('Competition')
    axes[0, 1].set_ylabel('Total Time (s)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Smoothness score by competition
    sns.boxplot(data=df_comp, x='competition', y='smoothness_score',
                ax=axes[1, 0], palette='colorblind')
    axes[1, 0].set_title('Smoothness Score by Competition')
    axes[1, 0].set_xlabel('Competition')
    axes[1, 0].set_ylabel('Smoothness Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Path efficiency by competition
    sns.boxplot(data=df_comp, x='competition', y='path_efficiency',
                ax=axes[1, 1], palette='colorblind')
    axes[1, 1].set_title('Path Efficiency by Competition')
    axes[1, 1].set_xlabel('Competition')
    axes[1, 1].set_ylabel('Path Efficiency')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'competition_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'competition_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'competition_comparison.png'}")


def plot_correlation_heatmap(df):
    """Plot 3: Correlation heatmap of all metrics."""
    print("Creating correlation heatmap...")

    # Select numeric columns
    numeric_cols = ['average_velocity_ms', 'max_velocity_ms', 'total_time_s',
                    'vertical_displacement_m', 'path_length_m', 'straight_distance_m',
                    'path_efficiency', 'smoothness_score', 'frames_analyzed']

    # Filter valid numeric data
    df_numeric = df[numeric_cols].select_dtypes(include=[np.number])

    # Calculate correlation
    corr = df_numeric.corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Performance Metrics Correlation Matrix', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'correlation_heatmap.png'}")


def plot_efficiency_scatter(df):
    """Plot 4: Performance vs efficiency scatter plots."""
    print("Creating efficiency scatter plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Velocity vs Path Efficiency
    scatter1 = ax1.scatter(df['path_efficiency'], df['average_velocity_ms'],
                          c=df['total_time_s'], cmap='viridis', alpha=0.6, s=50,
                          edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Path Efficiency')
    ax1.set_ylabel('Average Velocity (m/s)')
    ax1.set_title('Velocity vs Path Efficiency')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Total Time (s)')

    # Velocity vs Smoothness
    scatter2 = ax2.scatter(df['smoothness_score'], df['average_velocity_ms'],
                          c=df['total_time_s'], cmap='viridis', alpha=0.6, s=50,
                          edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Smoothness Score')
    ax2.set_ylabel('Average Velocity (m/s)')
    ax2.set_title('Velocity vs Smoothness')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Total Time (s)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'efficiency_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'efficiency_scatter.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'efficiency_scatter.png'}")


def plot_outlier_detection(df):
    """Plot 5: Outlier detection box plots."""
    print("Creating outlier detection plot...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics = [
        ('average_velocity_ms', 'Average Velocity (m/s)'),
        ('max_velocity_ms', 'Max Velocity (m/s)'),
        ('total_time_s', 'Total Time (s)'),
        ('vertical_displacement_m', 'Vertical Displacement (m)'),
        ('path_length_m', 'Path Length (m)'),
        ('smoothness_score', 'Smoothness Score')
    ]

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        # Box plot
        bp = ax.boxplot(df[col], vert=True, patch_artist=True,
                        boxprops=dict(facecolor=COLORS['primary'], alpha=0.6),
                        medianprops=dict(color=COLORS['danger'], linewidth=2),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'))

        # Identify outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        ax.set_title(f'{title}\n({len(outliers)} outliers)', fontsize=10)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text
        stats_text = f"Mean: {df[col].mean():.2f}\nMedian: {df[col].median():.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'outlier_detection.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'outlier_detection.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'outlier_detection.png'}")


def plot_top_performers(df):
    """Plot 6: Top 10 performers bar chart."""
    print("Creating top performers plot...")

    # Sort by average velocity and get top 10
    top10 = df.nlargest(10, 'average_velocity_ms').copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar chart
    bars = ax.barh(range(len(top10)), top10['average_velocity_ms'],
                   color=COLORS['success'], edgecolor='black', linewidth=1)

    # Customize
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels([f"{row['race_id'].replace('Speed_finals_', '')}\n({row['total_time_s']:.1f}s)"
                        for _, row in top10.iterrows()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Average Velocity (m/s)', fontsize=12)
    ax.set_title('Top 10 Performers (by Average Velocity)', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (_, row) in enumerate(top10.iterrows()):
        ax.text(row['average_velocity_ms'] + 0.02, i, f"{row['average_velocity_ms']:.3f} m/s",
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_performers.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'top_performers.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'top_performers.png'}")


def plot_time_distribution(df):
    """Plot 7: Race time distribution."""
    print("Creating time distribution plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with KDE
    ax1.hist(df['total_time_s'], bins=25, alpha=0.7, color=COLORS['warning'],
             edgecolor='black', density=True)

    # KDE overlay
    from scipy import stats
    kde = stats.gaussian_kde(df['total_time_s'])
    x_range = np.linspace(df['total_time_s'].min(), df['total_time_s'].max(), 100)
    ax1.plot(x_range, kde(x_range), color=COLORS['danger'], linewidth=2, label='KDE')

    ax1.axvline(df['total_time_s'].mean(), color='black', linestyle='--',
                linewidth=2, label=f"Mean: {df['total_time_s'].mean():.2f}s")
    ax1.set_xlabel('Total Time (s)')
    ax1.set_ylabel('Density')
    ax1.set_title('Race Time Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_times = np.sort(df['total_time_s'])
    cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times) * 100
    ax2.plot(sorted_times, cumulative, color=COLORS['primary'], linewidth=2)
    ax2.set_xlabel('Total Time (s)')
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_title('Cumulative Time Distribution')
    ax2.grid(True, alpha=0.3)

    # Add percentile markers
    for p in [25, 50, 75]:
        val = df['total_time_s'].quantile(p / 100)
        ax2.axvline(val, color=COLORS['danger'], linestyle='--', alpha=0.5)
        ax2.text(val, 5, f'P{p}\n{val:.1f}s', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'time_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'time_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'time_distribution.png'}")


def plot_vertical_displacement(df):
    """Plot 8: Vertical displacement analysis."""
    print("Creating vertical displacement plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Displacement distribution
    ax1.hist(df['vertical_displacement_m'], bins=20, alpha=0.7,
             color=COLORS['secondary'], edgecolor='black')
    ax1.axvline(15.0, color=COLORS['danger'], linestyle='--', linewidth=2,
                label='IFSC Standard (15m)')
    ax1.axvline(df['vertical_displacement_m'].median(), color=COLORS['success'],
                linestyle='--', linewidth=2, label=f"Median: {df['vertical_displacement_m'].median():.2f}m")
    ax1.set_xlabel('Vertical Displacement (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Vertical Displacement Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Displacement vs Velocity
    scatter = ax2.scatter(df['vertical_displacement_m'], df['average_velocity_ms'],
                         c=df['path_efficiency'], cmap='RdYlGn', alpha=0.6, s=50,
                         edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Vertical Displacement (m)')
    ax2.set_ylabel('Average Velocity (m/s)')
    ax2.set_title('Displacement vs Velocity')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Path Efficiency')

    # Add reference line for IFSC standard
    ax2.axvline(15.0, color=COLORS['danger'], linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'vertical_displacement.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'vertical_displacement.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'vertical_displacement.png'}")


def plot_frames_analysis(df):
    """Plot 9: Frames analyzed distribution."""
    print("Creating frames analysis plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Frames distribution
    ax1.hist(df['frames_analyzed'], bins=25, alpha=0.7, color=COLORS['primary'],
             edgecolor='black')
    ax1.set_xlabel('Frames Analyzed')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Frames Analyzed Distribution')
    ax1.grid(True, alpha=0.3)

    # Frames vs Time
    ax2.scatter(df['total_time_s'], df['frames_analyzed'], alpha=0.6, s=50,
                color=COLORS['secondary'], edgecolors='black', linewidth=0.5)

    # Add regression line
    z = np.polyfit(df['total_time_s'], df['frames_analyzed'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['total_time_s'].min(), df['total_time_s'].max(), 100)
    ax2.plot(x_line, p(x_line), color=COLORS['danger'], linewidth=2, linestyle='--',
             label=f'Fit: {z[0]:.1f} frames/s')

    ax2.set_xlabel('Total Time (s)')
    ax2.set_ylabel('Frames Analyzed')
    ax2.set_title('Frames vs Race Duration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'frames_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'frames_analysis.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'frames_analysis.png'}")


def plot_summary_statistics(df):
    """Plot 10: Summary statistics table."""
    print("Creating summary statistics table...")

    # Calculate statistics
    metrics = ['average_velocity_ms', 'max_velocity_ms', 'total_time_s',
               'vertical_displacement_m', 'path_efficiency', 'smoothness_score']

    stats_data = []
    for metric in metrics:
        stats_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Mean': f"{df[metric].mean():.3f}",
            'Median': f"{df[metric].median():.3f}",
            'Std': f"{df[metric].std():.3f}",
            'Min': f"{df[metric].min():.3f}",
            'Max': f"{df[metric].max():.3f}"
        })

    stats_df = pd.DataFrame(stats_data)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                     cellLoc='center', loc='center', colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_df) + 1):
        for j in range(len(stats_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')

    ax.set_title('Summary Statistics - 114 Reliable Races', fontsize=14, pad=20, weight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'summary_statistics.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved to {OUTPUT_DIR / 'summary_statistics.png'}")


def generate_report():
    """Generate a summary report."""
    print("\nGenerating visualization report...")

    report = {
        'generation_date': datetime.now().isoformat(),
        'total_races': 114,
        'visualizations_created': [
            'velocity_distribution.png',
            'competition_comparison.png',
            'correlation_heatmap.png',
            'efficiency_scatter.png',
            'outlier_detection.png',
            'top_performers.png',
            'time_distribution.png',
            'vertical_displacement.png',
            'frames_analysis.png',
            'summary_statistics.png'
        ],
        'output_directory': str(OUTPUT_DIR),
        'formats': ['PNG (300 DPI)', 'PDF (vector)']
    }

    report_file = OUTPUT_DIR.parent.parent / 'reports' / 'visualizations' / 'visualization_report.json'
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  Report saved to {report_file}")
    return report


def main():
    """Main execution."""
    print("=" * 60)
    print("STATISTICAL VISUALIZATIONS GENERATOR")
    print("Speed Climbing Performance Analysis")
    print("=" * 60)
    print()

    # Load data
    df = load_data()
    print(f"\nData summary:")
    print(f"  Total races: {len(df)}")
    print(f"  Competitions: {df['competition'].nunique()}")
    print(f"  Avg velocity: {df['average_velocity_ms'].mean():.3f} m/s")
    print(f"  Top performer: {df.loc[df['average_velocity_ms'].idxmax(), 'race_id']}")
    print()

    # Generate all plots
    print("Generating visualizations...\n")
    plot_velocity_distribution(df)
    plot_competition_comparison(df)
    plot_correlation_heatmap(df)
    plot_efficiency_scatter(df)
    plot_outlier_detection(df)
    plot_top_performers(df)
    plot_time_distribution(df)
    plot_vertical_displacement(df)
    plot_frames_analysis(df)
    plot_summary_statistics(df)

    # Generate report
    report = generate_report()

    print()
    print("=" * 60)
    print("OK - ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Total files created: {len(report['visualizations_created']) * 2} (PNG + PDF)")
    print("\nVisualization types:")
    for viz in report['visualizations_created']:
        print(f"  - {viz}")
    print()


if __name__ == '__main__':
    main()
