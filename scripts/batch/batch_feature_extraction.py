"""
Batch Feature Extraction for all races.

Processes all pose files and generates:
- all_features.csv: Features for all races
- quality_report.csv: Quality assessment per video
- problematic_videos.txt: List of videos needing manual review
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from speed_climbing.analysis.features import FeatureExtractor, save_features_json, save_features_csv
from speed_climbing.analysis.features.extractor import FeatureResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Quality assessment for a single extraction."""
    video_id: str
    lane: str
    competition: str
    extraction_quality: float
    race_segment_confidence: float
    total_frames: int
    valid_frames: int
    racing_frames: int
    fps: float
    status: str  # OK, WARNING, ERROR
    issues: str  # Comma-separated list of issues


# Quality thresholds
THRESHOLDS = {
    'extraction_quality_min': 0.7,
    'race_confidence_min': 0.6,
    'racing_frames_min': 90,  # 3 seconds at 30fps
    'racing_frames_max': 300,  # 10 seconds at 30fps
}


def assess_quality(result: FeatureResult, competition: str) -> QualityReport:
    """Assess the quality of an extraction result."""
    issues = []
    status = 'OK'

    # Check extraction quality
    if result.extraction_quality < THRESHOLDS['extraction_quality_min']:
        issues.append(f"low_extraction_quality({result.extraction_quality:.2f})")
        status = 'WARNING'

    # Check race segment confidence
    if result.race_segment_confidence < THRESHOLDS['race_confidence_min']:
        issues.append(f"low_confidence({result.race_segment_confidence:.2f})")
        status = 'WARNING'

    # Check racing frames
    if result.racing_frames < THRESHOLDS['racing_frames_min']:
        issues.append(f"too_short({result.racing_frames} frames)")
        status = 'WARNING'
    elif result.racing_frames > THRESHOLDS['racing_frames_max']:
        issues.append(f"too_long({result.racing_frames} frames)")
        status = 'WARNING'

    # Check for NaN values in features
    freq_features = result.frequency_features
    eff_features = result.efficiency_features
    post_features = result.posture_features

    for name, value in {**freq_features, **eff_features, **post_features}.items():
        if value is None or (isinstance(value, float) and (value != value)):  # NaN check
            issues.append(f"nan_{name}")
            status = 'ERROR'

    return QualityReport(
        video_id=result.video_id,
        lane=result.lane,
        competition=competition,
        extraction_quality=result.extraction_quality,
        race_segment_confidence=result.race_segment_confidence,
        total_frames=result.total_frames,
        valid_frames=result.valid_frames,
        racing_frames=result.racing_frames,
        fps=result.fps,
        status=status,
        issues=', '.join(issues) if issues else 'none'
    )


def find_all_pose_files() -> Dict[str, List[Path]]:
    """Find all pose files organized by competition."""
    poses_dir = project_root / 'data' / 'processed' / 'poses'

    competitions = {}

    for comp_dir in poses_dir.iterdir():
        if comp_dir.is_dir() and comp_dir.name != 'samples':
            pose_files = list(comp_dir.glob('*_poses.json'))
            if pose_files:
                competitions[comp_dir.name] = sorted(pose_files)
                logger.info(f"Found {len(pose_files)} pose files in {comp_dir.name}")

    return competitions


def process_all_races() -> Tuple[List[FeatureResult], List[QualityReport]]:
    """Process all pose files and extract features."""
    all_results = []
    all_quality = []

    # Find all pose files
    competitions = find_all_pose_files()

    if not competitions:
        logger.error("No pose files found!")
        return [], []

    total_files = sum(len(files) for files in competitions.values())
    logger.info(f"Processing {total_files} pose files from {len(competitions)} competitions")

    # Initialize extractor
    extractor = FeatureExtractor(fps=30.0, min_frames=30)

    processed = 0
    errors = 0

    for competition, pose_files in competitions.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {competition}: {len(pose_files)} files")
        logger.info('='*60)

        for pose_file in pose_files:
            processed += 1
            video_id = pose_file.stem.replace('_poses', '')

            try:
                # Extract features
                results = extractor.extract_from_file(pose_file)

                for result in results:
                    all_results.append(result)

                    # Assess quality
                    quality = assess_quality(result, competition)
                    all_quality.append(quality)

                    # Log status
                    status_emoji = {'OK': 'O', 'WARNING': 'W', 'ERROR': 'X'}[quality.status]
                    logger.info(f"  [{status_emoji}] {video_id} ({result.lane}): "
                              f"quality={result.extraction_quality:.2f}, "
                              f"conf={result.race_segment_confidence:.2f}, "
                              f"frames={result.racing_frames}")

            except Exception as e:
                errors += 1
                logger.error(f"  [X] {video_id}: {str(e)}")

                # Create error quality report
                all_quality.append(QualityReport(
                    video_id=video_id,
                    lane='unknown',
                    competition=competition,
                    extraction_quality=0.0,
                    race_segment_confidence=0.0,
                    total_frames=0,
                    valid_frames=0,
                    racing_frames=0,
                    fps=0.0,
                    status='ERROR',
                    issues=f'extraction_failed: {str(e)}'
                ))

    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {processed}")
    logger.info(f"Total extractions: {len(all_results)}")
    logger.info(f"Errors: {errors}")

    return all_results, all_quality


def generate_reports(results: List[FeatureResult], quality: List[QualityReport]):
    """Generate output files."""
    output_dir = project_root / 'data' / 'ml_dataset'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save all features
    if results:
        features_json = output_dir / 'all_features.json'
        features_csv = output_dir / 'all_features.csv'

        save_features_json(results, features_json)
        save_features_csv(results, features_csv)

        logger.info(f"\nFeatures saved to:")
        logger.info(f"  JSON: {features_json}")
        logger.info(f"  CSV:  {features_csv}")

    # Save quality report
    if quality:
        quality_df = pd.DataFrame([asdict(q) for q in quality])
        quality_csv = output_dir / 'quality_report.csv'
        quality_df.to_csv(quality_csv, index=False)
        logger.info(f"  Quality: {quality_csv}")

        # Generate problematic videos list
        problematic = [q for q in quality if q.status in ('WARNING', 'ERROR')]
        if problematic:
            problems_file = output_dir / 'problematic_videos.txt'
            with open(problems_file, 'w') as f:
                f.write(f"# Problematic Videos Report\n")
                f.write(f"# Generated: {timestamp}\n")
                f.write(f"# Total: {len(problematic)} videos need review\n\n")

                # Group by status
                errors = [q for q in problematic if q.status == 'ERROR']
                warnings = [q for q in problematic if q.status == 'WARNING']

                if errors:
                    f.write(f"## ERRORS ({len(errors)}):\n")
                    for q in errors:
                        f.write(f"  {q.video_id} ({q.lane}): {q.issues}\n")
                    f.write("\n")

                if warnings:
                    f.write(f"## WARNINGS ({len(warnings)}):\n")
                    for q in warnings:
                        f.write(f"  {q.video_id} ({q.lane}): {q.issues}\n")

            logger.info(f"  Problems: {problems_file}")

    # Print summary
    print_summary(quality)


def print_summary(quality: List[QualityReport]):
    """Print a summary of the extraction results."""
    if not quality:
        return

    # Count statuses
    ok_count = sum(1 for q in quality if q.status == 'OK')
    warning_count = sum(1 for q in quality if q.status == 'WARNING')
    error_count = sum(1 for q in quality if q.status == 'ERROR')

    # Group by competition
    by_comp = {}
    for q in quality:
        if q.competition not in by_comp:
            by_comp[q.competition] = {'OK': 0, 'WARNING': 0, 'ERROR': 0}
        by_comp[q.competition][q.status] += 1

    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print('='*60)

    print(f"\nOverall Status:")
    print(f"  OK:      {ok_count:3d} ({ok_count/len(quality)*100:.1f}%)")
    print(f"  WARNING: {warning_count:3d} ({warning_count/len(quality)*100:.1f}%)")
    print(f"  ERROR:   {error_count:3d} ({error_count/len(quality)*100:.1f}%)")
    print(f"  TOTAL:   {len(quality):3d}")

    print(f"\nBy Competition:")
    for comp, counts in sorted(by_comp.items()):
        total = sum(counts.values())
        print(f"  {comp:20s}: OK={counts['OK']:2d}, WARN={counts['WARNING']:2d}, ERR={counts['ERROR']:2d} (total: {total})")

    # Calculate average metrics
    valid_quality = [q for q in quality if q.status != 'ERROR']
    if valid_quality:
        avg_quality = sum(q.extraction_quality for q in valid_quality) / len(valid_quality)
        avg_conf = sum(q.race_segment_confidence for q in valid_quality) / len(valid_quality)
        avg_frames = sum(q.racing_frames for q in valid_quality) / len(valid_quality)

        print(f"\nAverage Metrics (excluding errors):")
        print(f"  Extraction Quality:     {avg_quality:.2%}")
        print(f"  Race Segment Confidence: {avg_conf:.2%}")
        print(f"  Racing Frames:          {avg_frames:.1f}")


def main():
    """Main entry point."""
    logger.info("Starting batch feature extraction...")
    logger.info(f"Project root: {project_root}")

    # Process all races
    results, quality = process_all_races()

    # Generate reports
    generate_reports(results, quality)

    logger.info("\nBatch extraction complete!")


if __name__ == '__main__':
    main()
