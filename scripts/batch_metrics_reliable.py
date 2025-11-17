"""
Batch Metrics Calculation - Reliable Races Only
===============================================
Calculate performance metrics for 114 verified races with calibration.

Features:
- Direct calculation from pose data (no PerformanceAnalyzer dependency)
- Applies camera calibration for measurements
- Validates metric quality

Usage:
    python scripts/batch_metrics_reliable.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import json
import numpy as np
from tqdm import tqdm
from datetime import datetime


def calculate_metrics_for_reliable_races():
    """Calculate metrics for all reliable races."""

    # Load reliable races list
    reliable_races_file = Path('data/processed/reliable_races_list.json')
    with open(reliable_races_file, 'r', encoding='utf-8') as f:
        reliable_data = json.load(f)

    race_ids = reliable_data['reliable_race_ids']

    print("="*70)
    print(f"Batch Metrics Calculation - Reliable Races")
    print("="*70)
    print(f"Total races to process: {len(race_ids)}")
    print(f"Features: Velocity, smoothness, efficiency")
    print(f"Estimated time: < 10 minutes")
    print(f"Output directory: data/processed/metrics/")
    print("="*70 + "\n")

    success_count = 0
    failed_races = []
    metrics_quality = []

    for race_id in tqdm(race_ids, desc="Calculating metrics"):
        try:
            # Find files
            pose_path = Path(f"data/processed/poses/{race_id}_pose.json")
            calib_path = Path(f"data/processed/calibration/{race_id}_calibration.json")
            metadata_path = find_metadata_path(race_id)

            if not pose_path.exists():
                raise FileNotFoundError(f"Pose file not found: {pose_path}")
            if not calib_path.exists():
                raise FileNotFoundError(f"Calibration file not found: {calib_path}")

            # Load data
            with open(pose_path, 'r', encoding='utf-8') as f:
                pose_data = json.load(f)

            with open(calib_path, 'r', encoding='utf-8') as f:
                calibration = json.load(f)

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Calculate metrics directly
            metrics = calculate_simple_metrics(pose_data, calibration, metadata)

            if metrics is None:
                raise ValueError("Failed to calculate metrics")

            # Save metrics
            output_path = Path(f"data/processed/metrics/{race_id}_metrics.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            # Track quality
            metrics_quality.append({
                'race_id': race_id,
                'avg_velocity_ms': metrics.get('average_velocity_ms', 0),
                'total_time_s': metrics.get('total_time_s', 0),
                'is_calibrated': True,
                'valid_velocity': 0.5 <= metrics.get('average_velocity_ms', 0) <= 3.0,
                'valid_duration': 4.0 <= metrics.get('total_time_s', 0) <= 20.0
            })

            success_count += 1

        except Exception as e:
            tqdm.write(f"Failed for {race_id}: {e}")
            failed_races.append({'race_id': race_id, 'error': str(e)})

    # Generate report
    valid_velocity_count = sum(1 for m in metrics_quality if m['valid_velocity'])
    valid_duration_count = sum(1 for m in metrics_quality if m['valid_duration'])

    velocities = [m['avg_velocity_ms'] for m in metrics_quality if m['valid_velocity']]
    durations = [m['total_time_s'] for m in metrics_quality if m['valid_duration']]

    report = {
        'total_races': len(race_ids),
        'successful': success_count,
        'failed': len(failed_races),
        'failed_races': failed_races,
        'quality_metrics': {
            'valid_velocity_count': valid_velocity_count,
            'valid_duration_count': valid_duration_count,
            'avg_velocity_ms': np.mean(velocities) if velocities else 0,
            'avg_duration_s': np.mean(durations) if durations else 0,
            'min_velocity_ms': np.min(velocities) if velocities else 0,
            'max_velocity_ms': np.max(velocities) if velocities else 0,
            'min_duration_s': np.min(durations) if durations else 0,
            'max_duration_s': np.max(durations) if durations else 0
        },
        'metrics_quality': metrics_quality,
        'completion_date': datetime.now().isoformat()
    }

    report_path = Path('data/processed/metrics_calculation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*70)
    print("Metrics calculation complete!")
    print("="*70)
    print(f"Successful: {success_count}/{len(race_ids)}")
    print(f"Failed: {len(failed_races)}")
    print(f"Valid velocity: {valid_velocity_count}/{len(metrics_quality)}")
    print(f"Valid duration: {valid_duration_count}/{len(metrics_quality)}")
    print(f"\nPerformance Statistics:")
    print(f"  Average velocity: {report['quality_metrics']['avg_velocity_ms']:.2f} m/s")
    print(f"  Average duration: {report['quality_metrics']['avg_duration_s']:.2f} s")
    print(f"\nReport saved to: {report_path}")
    print("="*70)

    if failed_races:
        print("\nFailed races:")
        for race in failed_races[:10]:
            print(f"   - {race['race_id']}: {race['error']}")
        if len(failed_races) > 10:
            print(f"   ... and {len(failed_races) - 10} more")


def calculate_simple_metrics(pose_data, calibration, metadata):
    """Calculate basic performance metrics from pose data."""

    frames = pose_data.get('frames', [])
    if len(frames) < 2:
        return None

    fps = pose_data.get('fps', 30.0)
    pixel_to_meter = calibration.get('pixel_to_meter_scale', 0.025)

    # Extract COM positions (simple average of visible landmarks)
    com_positions = []
    timestamps = []

    for i, frame in enumerate(frames):
        landmarks = frame.get('landmarks', [])
        if len(landmarks) < 10:  # Need minimum landmarks
            continue

        # Calculate COM as average of visible landmarks
        x_sum, y_sum, count = 0, 0, 0
        for lm in landmarks:
            if lm.get('visibility', 0) > 0.5:
                x_sum += lm['x']
                y_sum += lm['y']
                count += 1

        if count > 0:
            com_x = x_sum / count
            com_y = y_sum / count
            com_positions.append((com_x, com_y))
            timestamps.append(i / fps)

    if len(com_positions) < 2:
        return None

    # Calculate metrics
    com_array = np.array(com_positions)

    # Vertical movement (upward is negative y in image coords)
    vertical_displacement_px = com_array[0, 1] - com_array[-1, 1]  # Start y - End y
    vertical_displacement_m = abs(vertical_displacement_px * pixel_to_meter)

    # Total time
    total_time_s = timestamps[-1] - timestamps[0]

    # Average velocity
    if total_time_s > 0:
        avg_velocity_ms = vertical_displacement_m / total_time_s
    else:
        avg_velocity_ms = 0

    # Path length (total distance traveled)
    path_length_px = 0
    for i in range(1, len(com_array)):
        dx = com_array[i, 0] - com_array[i-1, 0]
        dy = com_array[i, 1] - com_array[i-1, 1]
        path_length_px += np.sqrt(dx**2 + dy**2)

    path_length_m = path_length_px * pixel_to_meter

    # Efficiency (straight line / actual path)
    straight_distance_px = np.sqrt(
        (com_array[-1, 0] - com_array[0, 0])**2 +
        (com_array[-1, 1] - com_array[0, 1])**2
    )
    straight_distance_m = straight_distance_px * pixel_to_meter

    efficiency = straight_distance_m / path_length_m if path_length_m > 0 else 0

    # Smoothness (standard deviation of velocity changes)
    velocities = []
    for i in range(1, len(com_array)):
        dt = timestamps[i] - timestamps[i-1]
        if dt > 0:
            dy = abs(com_array[i-1, 1] - com_array[i, 1]) * pixel_to_meter
            v = dy / dt
            velocities.append(v)

    smoothness_score = 1.0 / (1.0 + np.std(velocities)) if velocities else 0

    # Compile metrics
    metrics = {
        'race_id': metadata.get('race_id', 'unknown'),
        'competition': metadata.get('competition', 'unknown'),
        'total_time_s': float(total_time_s),
        'average_velocity_ms': float(avg_velocity_ms),
        'max_velocity_ms': float(max(velocities)) if velocities else 0,
        'vertical_displacement_m': float(vertical_displacement_m),
        'path_length_m': float(path_length_m),
        'straight_distance_m': float(straight_distance_m),
        'path_efficiency': float(efficiency),
        'smoothness_score': float(smoothness_score),
        'frames_analyzed': len(com_positions),
        'is_calibrated': True,
        'calibration_type': calibration.get('calibration_type', 'unknown'),
        'calculation_date': datetime.now().isoformat()
    }

    return metrics


def find_metadata_path(race_id: str) -> Path:
    """Find metadata path for a race."""
    race_segments_dir = Path("data/race_segments")

    for comp_dir in race_segments_dir.iterdir():
        if not comp_dir.is_dir():
            continue

        metadata_path = comp_dir / f"{race_id}_metadata.json"
        if metadata_path.exists():
            return metadata_path

    raise FileNotFoundError(f"Metadata not found for {race_id}")


if __name__ == "__main__":
    try:
        calculate_metrics_for_reliable_races()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        print("Partial results have been saved to data/processed/metrics/")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
