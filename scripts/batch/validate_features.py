"""
Feature Validation and Statistics Generation.

Analyzes the extracted features and generates:
- Feature statistics (mean, std, min, max, percentiles)
- Correlation matrix
- Feature distribution plots
- Outlier detection report
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_features(features_path: Path) -> pd.DataFrame:
    """Load features from CSV file."""
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} records from {features_path.name}")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (excluding metadata)."""
    metadata_cols = [
        'video_id', 'lane', 'extraction_quality', 'total_frames',
        'valid_frames', 'racing_frames', 'fps', 'detection_confidence',
        'race_segment_confidence'
    ]
    # Feature columns have prefixes: freq_, eff_, post_
    feature_prefixes = ('freq_', 'eff_', 'post_')
    return [col for col in df.columns if col.startswith(feature_prefixes)]


def calculate_statistics(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Calculate comprehensive statistics for each feature."""
    stats_list = []

    for col in feature_cols:
        values = df[col].dropna()

        if len(values) == 0:
            continue

        stats = {
            'feature': col,
            'count': len(values),
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'p25': values.quantile(0.25),
            'median': values.quantile(0.50),
            'p75': values.quantile(0.75),
            'max': values.max(),
            'range': values.max() - values.min(),
            'cv': values.std() / values.mean() if values.mean() != 0 else np.nan,  # Coefficient of variation
            'skewness': values.skew(),
            'kurtosis': values.kurtosis(),
            'nan_count': df[col].isna().sum(),
        }
        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def detect_outliers(df: pd.DataFrame, feature_cols: List[str], threshold: float = 3.0) -> pd.DataFrame:
    """Detect outliers using z-score method."""
    outliers_list = []

    for col in feature_cols:
        values = df[col].dropna()
        if len(values) < 3:
            continue

        mean = values.mean()
        std = values.std()

        if std == 0:
            continue

        z_scores = (df[col] - mean) / std
        outlier_mask = abs(z_scores) > threshold

        for idx in df[outlier_mask].index:
            outliers_list.append({
                'video_id': df.loc[idx, 'video_id'],
                'lane': df.loc[idx, 'lane'],
                'feature': col,
                'value': df.loc[idx, col],
                'z_score': z_scores[idx],
                'mean': mean,
                'std': std
            })

    return pd.DataFrame(outliers_list)


def calculate_correlations(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Calculate correlation matrix for features."""
    return df[feature_cols].corr()


def analyze_by_competition(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Analyze feature distributions by competition."""
    # Extract competition from video_id
    df['competition'] = df['video_id'].apply(
        lambda x: '_'.join(x.split('_')[2:4]) if len(x.split('_')) >= 4 else 'unknown'
    )

    # Group by competition and calculate means
    comp_stats = df.groupby('competition')[feature_cols].mean()

    return comp_stats


def print_summary(stats_df: pd.DataFrame, outliers_df: pd.DataFrame, df: pd.DataFrame):
    """Print summary to console."""
    print("\n" + "="*60)
    print("FEATURE VALIDATION SUMMARY")
    print("="*60)

    print(f"\nTotal samples: {len(df)}")
    print(f"Total features: {len(stats_df)}")

    # Feature categories based on prefixes
    freq_features = [f for f in stats_df['feature'] if f.startswith('freq_')]
    eff_features = [f for f in stats_df['feature'] if f.startswith('eff_')]
    post_features = [f for f in stats_df['feature'] if f.startswith('post_')]

    print(f"\nFeature categories:")
    print(f"  Frequency: {len(freq_features)}")
    print(f"  Efficiency: {len(eff_features)}")
    print(f"  Posture: {len(post_features)}")

    # Outliers summary
    print(f"\nOutliers detected (|z| > 3): {len(outliers_df)}")
    if len(outliers_df) > 0:
        outlier_features = outliers_df['feature'].value_counts().head(5)
        print("  Top features with outliers:")
        for feat, count in outlier_features.items():
            print(f"    {feat}: {count}")

    # High variance features
    high_cv = stats_df[stats_df['cv'] > 0.5].sort_values('cv', ascending=False)
    print(f"\nHigh variance features (CV > 0.5): {len(high_cv)}")
    for _, row in high_cv.head(5).iterrows():
        print(f"  {row['feature']}: CV={row['cv']:.2f}")

    # NaN summary
    nan_features = stats_df[stats_df['nan_count'] > 0]
    if len(nan_features) > 0:
        print(f"\nFeatures with NaN values: {len(nan_features)}")
        for _, row in nan_features.iterrows():
            print(f"  {row['feature']}: {row['nan_count']} NaN values")
    else:
        print("\nNo NaN values in features!")


def generate_plots(df: pd.DataFrame, feature_cols: List[str], output_dir: Path):
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("\nWarning: matplotlib/seaborn not available. Skipping plot generation.")
        return

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # 1. Feature distributions (histograms)
    print("\nGenerating distribution plots...")
    n_features = len(feature_cols)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        values = df[col].dropna()
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(col.replace('_', ' ').title(), fontsize=8)
        ax.tick_params(labelsize=6)

    # Hide empty subplots
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_distributions.png', dpi=150)
    plt.close()
    print(f"  Saved: {plots_dir / 'feature_distributions.png'}")

    # 2. Correlation matrix
    print("Generating correlation matrix...")
    corr_matrix = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(plots_dir / 'correlation_matrix.png', dpi=150)
    plt.close()
    print(f"  Saved: {plots_dir / 'correlation_matrix.png'}")

    # 3. Quality distribution
    print("Generating quality plots...")

    # Check if extraction_quality column exists
    if 'extraction_quality' in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.hist(df['extraction_quality'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0.7, color='r', linestyle='--', label='Threshold (0.7)')
        ax.set_xlabel('Extraction Quality')
        ax.set_ylabel('Count')
        ax.set_title('Extraction Quality Distribution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'quality_distribution.png', dpi=150)
        plt.close()
        print(f"  Saved: {plots_dir / 'quality_distribution.png'}")
    else:
        print("  Skipping quality plots - extraction_quality column not found")

    # 4. Competition comparison (box plots for key features)
    print("Generating competition comparison...")
    df['competition'] = df['video_id'].apply(
        lambda x: '_'.join(x.split('_')[2:4]) if len(x.split('_')) >= 4 else 'unknown'
    )

    # Use prefixed column names
    key_features = ['freq_hand_frequency_hz', 'eff_path_straightness', 'post_avg_knee_angle', 'freq_limb_sync_ratio']
    available_features = [f for f in key_features if f in df.columns]

    if available_features:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, feat in enumerate(available_features):
            ax = axes[i]
            df.boxplot(column=feat, by='competition', ax=ax)
            ax.set_title(feat.replace('_', ' ').title())
            ax.set_xlabel('')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.suptitle('Key Features by Competition', fontsize=14)
        plt.tight_layout()
        plt.savefig(plots_dir / 'competition_comparison.png', dpi=150)
        plt.close()
        print(f"  Saved: {plots_dir / 'competition_comparison.png'}")


def main():
    """Main entry point."""
    # Paths
    ml_dataset_dir = project_root / 'data' / 'ml_dataset'
    features_path = ml_dataset_dir / 'all_features.csv'

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        print("Run batch_feature_extraction.py first.")
        return

    # Load features
    df = load_features(features_path)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Found {len(feature_cols)} feature columns")

    # Calculate statistics
    print("\nCalculating feature statistics...")
    stats_df = calculate_statistics(df, feature_cols)

    # Detect outliers
    print("Detecting outliers...")
    outliers_df = detect_outliers(df, feature_cols)

    # Calculate correlations
    print("Calculating correlations...")
    corr_df = calculate_correlations(df, feature_cols)

    # Analyze by competition
    print("Analyzing by competition...")
    comp_stats = analyze_by_competition(df, feature_cols)

    # Save outputs
    print("\nSaving outputs...")
    stats_df.to_csv(ml_dataset_dir / 'feature_statistics.csv', index=False)
    print(f"  Saved: {ml_dataset_dir / 'feature_statistics.csv'}")

    outliers_df.to_csv(ml_dataset_dir / 'outliers_report.csv', index=False)
    print(f"  Saved: {ml_dataset_dir / 'outliers_report.csv'}")

    corr_df.to_csv(ml_dataset_dir / 'correlation_matrix.csv')
    print(f"  Saved: {ml_dataset_dir / 'correlation_matrix.csv'}")

    comp_stats.to_csv(ml_dataset_dir / 'competition_comparison.csv')
    print(f"  Saved: {ml_dataset_dir / 'competition_comparison.csv'}")

    # Generate plots
    generate_plots(df, feature_cols, ml_dataset_dir)

    # Print summary
    print_summary(stats_df, outliers_df, df)

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
