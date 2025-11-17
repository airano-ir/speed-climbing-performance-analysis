"""
Run Complete Reliable Data Pipeline
====================================
Execute all 7 phases of the reliable races pipeline in sequence.

Phases:
1. Generate reliable races list (114 races)
2. Batch pose extraction (3-4 hours)
3. Batch calibration (2-3 hours)
4. Batch metrics calculation (3-4 hours)
5. Data aggregation & ML export
6. Pipeline validation
7. Dashboard generation

Total Estimated Time: 12-16 hours

Usage:
    # Run all phases
    python scripts/run_reliable_pipeline.py

    # Run specific phase
    python scripts/run_reliable_pipeline.py --phase 1

    # Skip completed phases (resume from checkpoint)
    python scripts/run_reliable_pipeline.py --resume

    # Dry run (show what would be executed)
    python scripts/run_reliable_pipeline.py --dry-run
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import argparse


class PipelineRunner:
    """Execute the reliable races pipeline."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.scripts = [
            {
                'phase': 1,
                'name': 'Generate Reliable Races List',
                'script': 'scripts/generate_reliable_races_list.py',
                'estimated_time': '5 minutes',
                'checkpoint': 'data/processed/reliable_races_list.json'
            },
            {
                'phase': 2,
                'name': 'Batch Pose Extraction',
                'script': 'scripts/batch_pose_extraction_reliable.py',
                'estimated_time': '3-4 hours',
                'checkpoint': 'data/processed/pose_extraction_report.json'
            },
            {
                'phase': 3,
                'name': 'Batch Calibration',
                'script': 'scripts/batch_calibration_reliable.py',
                'estimated_time': '2-3 hours',
                'checkpoint': 'data/processed/calibration_report.json'
            },
            {
                'phase': 4,
                'name': 'Batch Metrics Calculation',
                'script': 'scripts/batch_metrics_reliable.py',
                'estimated_time': '3-4 hours',
                'checkpoint': 'data/processed/metrics_calculation_report.json'
            },
            {
                'phase': 5,
                'name': 'Data Aggregation & ML Export',
                'script': 'scripts/aggregate_reliable_data.py',
                'estimated_time': '10 minutes',
                'checkpoint': 'data/processed/aggregated_metrics_reliable.csv'
            },
            {
                'phase': 6,
                'name': 'Pipeline Validation',
                'script': 'scripts/validate_pipeline_reliable.py',
                'estimated_time': '5 minutes',
                'checkpoint': 'data/processed/pipeline_validation_report.json'
            },
            {
                'phase': 7,
                'name': 'Dashboard Generation',
                'script': 'scripts/generate_dashboard_reliable.py',
                'estimated_time': '5 minutes',
                'checkpoint': 'data/processed/dashboard_reliable_races.html'
            }
        ]

        self.log_file = Path('data/processed/pipeline_execution_log.json')
        self.load_log()

    def load_log(self):
        """Load execution log."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.log = json.load(f)
        else:
            self.log = {
                'executions': [],
                'current_execution': {
                    'start_time': datetime.now().isoformat(),
                    'phases_completed': [],
                    'phases_failed': []
                }
            }

    def save_log(self):
        """Save execution log."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.log, f, indent=2)

    def check_phase_completed(self, phase_info):
        """Check if a phase has been completed."""
        checkpoint = Path(phase_info['checkpoint'])
        return checkpoint.exists()

    def run_phase(self, phase_info):
        """Execute a single phase."""
        phase_num = phase_info['phase']
        name = phase_info['name']
        script = phase_info['script']
        estimated_time = phase_info['estimated_time']

        print("\n" + "="*70)
        print(f"Phase {phase_num}: {name}")
        print("="*70)
        print(f"Script: {script}")
        print(f"Estimated time: {estimated_time}")

        # Check if already completed
        if self.check_phase_completed(phase_info):
            print(f"âœ“ Checkpoint found: {phase_info['checkpoint']}")
            response = input("Phase appears completed. Re-run anyway? (y/N): ")
            if response.lower() != 'y':
                print("Skipping phase (already completed)")
                return True

        if self.dry_run:
            print("[DRY RUN] Would execute: python", script)
            return True

        print(f"\nStarting at {datetime.now().strftime('%H:%M:%S')}...")

        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script],
                check=True,
                capture_output=False  # Show output in real-time
            )

            print(f"\nâœ… Phase {phase_num} completed successfully")
            self.log['current_execution']['phases_completed'].append({
                'phase': phase_num,
                'name': name,
                'completion_time': datetime.now().isoformat()
            })
            self.save_log()
            return True

        except subprocess.CalledProcessError as e:
            print(f"\nâŒ Phase {phase_num} failed with error code {e.returncode}")
            self.log['current_execution']['phases_failed'].append({
                'phase': phase_num,
                'name': name,
                'error': str(e),
                'failure_time': datetime.now().isoformat()
            })
            self.save_log()
            return False

        except KeyboardInterrupt:
            print(f"\nâš ï¸  Phase {phase_num} interrupted by user")
            raise

    def run_all(self, start_phase=1, resume=False):
        """Run all phases from start_phase onwards."""
        print("="*70)
        print("RELIABLE DATA PIPELINE - FULL EXECUTION")
        print("="*70)
        print(f"Total phases: 7")
        print(f"Estimated total time: 12-16 hours")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        if self.dry_run:
            print("\nâš ï¸  DRY RUN MODE - No scripts will be executed")

        if resume:
            print("\nðŸ“‹ Resume mode: Checking for completed phases...")

        for phase_info in self.scripts:
            if phase_info['phase'] < start_phase:
                continue

            # Skip if resuming and phase is completed
            if resume and self.check_phase_completed(phase_info):
                print(f"\nâœ“ Phase {phase_info['phase']} already completed, skipping")
                continue

            success = self.run_phase(phase_info)

            if not success:
                print(f"\nâŒ Pipeline stopped due to phase {phase_info['phase']} failure")
                return False

        # Mark execution as complete
        self.log['current_execution']['end_time'] = datetime.now().isoformat()
        self.log['current_execution']['status'] = 'completed'
        self.log['executions'].append(self.log['current_execution'])
        self.save_log()

        print("\n" + "="*70)
        print("ðŸŽ‰ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Phases completed: {len(self.log['current_execution']['phases_completed'])}/7")
        print("\nOutputs:")
        print("  - Reliable races list: data/processed/reliable_races_list.json")
        print("  - Pose files: data/processed/poses/ (114 files)")
        print("  - Calibration files: data/processed/calibration/ (114 files)")
        print("  - Metrics files: data/processed/metrics/ (114 files)")
        print("  - Aggregated data: data/processed/aggregated_metrics_reliable.csv")
        print("  - Validation report: data/processed/pipeline_validation_report.json")
        print("  - Dashboard: data/processed/dashboard_reliable_races.html")
        print("="*70)

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run the reliable races data pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases
  python scripts/run_reliable_pipeline.py

  # Start from phase 3
  python scripts/run_reliable_pipeline.py --phase 3

  # Resume from last checkpoint
  python scripts/run_reliable_pipeline.py --resume

  # Dry run
  python scripts/run_reliable_pipeline.py --dry-run
        """
    )
    parser.add_argument('--phase', type=int, default=1,
                       help='Start from this phase (1-7)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint, skip completed phases')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be executed without running')

    args = parser.parse_args()

    if args.phase < 1 or args.phase > 7:
        print("Error: Phase must be between 1 and 7")
        sys.exit(1)

    runner = PipelineRunner(dry_run=args.dry_run)

    try:
        success = runner.run_all(start_phase=args.phase, resume=args.resume)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nâš ï¸  Pipeline interrupted by user")
        print("Progress has been saved. Use --resume to continue from where you left off.")
        sys.exit(130)

    except Exception as e:
        print(f"\n\nâŒ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

