"""
Race Duration Validator
========================
Flags races with suspicious durations for manual review.

World Records (2024):
- Men: 5.00s
- Women: 6.53s

Reasonable range: 4.5s - 15s
(4.5s: slightly below WR, 15s: slower climbers + falls allowed)
"""

import json
from pathlib import Path
from typing import List, Dict
import sys

def validate_race_duration(metadata_path: Path) -> Dict:
    """Validate single race duration."""
    with open(metadata_path) as f:
        metadata = json.load(f)

    start = metadata['detected_start_frame']
    finish = metadata['detected_finish_frame']
    fps = 30.0  # Assume 30fps
    duration_seconds = (finish - start) / fps

    # Validation thresholds
    MIN_DURATION = 4.5  # Slightly below world record
    MAX_DURATION = 15.0  # Slower climbers + falls

    status = "OK"
    warnings = []

    if duration_seconds < MIN_DURATION:
        status = "SUSPICIOUS - TOO SHORT"
        warnings.append(f"Duration {duration_seconds:.2f}s < {MIN_DURATION}s (below world record!)")

    if duration_seconds > MAX_DURATION:
        status = "SUSPICIOUS - TOO LONG"
        warnings.append(f"Duration {duration_seconds:.2f}s > {MAX_DURATION}s (unusually slow)")

    return {
        'race_name': metadata_path.stem.replace('_metadata', ''),
        'duration_seconds': duration_seconds,
        'duration_frames': finish - start,
        'status': status,
        'warnings': warnings,
        'confidence_start': metadata.get('confidence_start', metadata.get('start_confidence', 'N/A')),
        'confidence_finish': metadata.get('confidence_finish', metadata.get('finish_confidence', 'N/A')),
        'manual_correction': metadata.get('manual_correction', False)
    }

def validate_all_races() -> None:
    """Validate all 188 races."""
    metadata_dir = Path("data/race_segments")
    results = []

    for comp_dir in metadata_dir.iterdir():
        if not comp_dir.is_dir():
            continue

        for metadata_file in comp_dir.glob("*_metadata.json"):
            result = validate_race_duration(metadata_file)
            results.append(result)

    # Print summary
    total = len(results)
    ok = sum(1 for r in results if r['status'] == 'OK')
    suspicious = total - ok

    print(f"=" * 80)
    print(f"RACE DURATION VALIDATION SUMMARY")
    print(f"=" * 80)
    print(f"Total races: {total}")
    print(f"OK: {ok} ({ok/total*100:.1f}%)")
    print(f"Suspicious: {suspicious} ({suspicious/total*100:.1f}%)")
    print()

    # Print suspicious races
    if suspicious > 0:
        print(f"SUSPICIOUS RACES (need manual review):")
        print(f"-" * 80)
        for result in results:
            if result['status'] != 'OK':
                correction_flag = " [MANUALLY CORRECTED]" if result['manual_correction'] else ""
                print(f"\n{result['race_name']}{correction_flag}")
                print(f"  Duration: {result['duration_seconds']:.2f}s ({result['duration_frames']} frames)")
                print(f"  Status: {result['status']}")
                print(f"  Confidence: start={result['confidence_start']}, finish={result['confidence_finish']}")
                for warning in result['warnings']:
                    print(f"  ! {warning}")

    # Save report
    report_path = Path("data/processed/race_duration_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump({
            'total': total,
            'ok_count': ok,
            'suspicious_count': suspicious,
            'results': results
        }, f, indent=2)

    print(f"\n" + "=" * 80)
    print(f"Report saved to: {report_path}")
    print(f"=" * 80)

if __name__ == "__main__":
    validate_all_races()
