"""
Test Script for Review Interface Components
===========================================
اسکریپت تست برای کامپوننت‌های رابط بررسی

Run this script before using the interface to verify all components work correctly.

Usage:
    python scripts/review_interface/test_components.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.review_interface.config import ConfigManager
from scripts.review_interface.progress import ProgressTracker
from scripts.review_interface.metadata_manager import MetadataManager
from scripts.review_interface.validators import RaceValidator


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_success(text: str):
    """Print success message."""
    print(f"  ✓ {text}")


def print_error(text: str):
    """Print error message."""
    print(f"  ✗ {text}")


def test_config_manager():
    """Test ConfigManager component."""
    print_header("Testing ConfigManager / تست مدیر پیکربندی")

    try:
        # Initialize
        config = ConfigManager()
        print_success("ConfigManager initialized")

        # Test loading competitions
        comps = config.get_competitions()
        print_success(f"Loaded {len(comps)} competitions")

        for comp in comps:
            print(f"    - {comp.name} ({comp.key}): {comp.total_races} races @ {comp.fps} FPS")

        # Test validation rules
        rules = config.get_validation_rules()
        duration_min = rules['duration']['min']
        duration_max = rules['duration']['max']
        print_success(f"Duration validation range: {duration_min}s - {duration_max}s")

        # Test general settings
        settings = config.get_general_settings()
        default_fps = settings.get('default_fps', 'Not set')
        print_success(f"Default FPS: {default_fps}")

        return True

    except Exception as e:
        print_error(f"ConfigManager test failed: {e}")
        return False


def test_progress_tracker():
    """Test ProgressTracker component."""
    print_header("Testing ProgressTracker / تست ردیاب پیشرفت")

    try:
        # Initialize
        tracker = ProgressTracker()
        print_success("ProgressTracker initialized")

        # Load all races
        all_races = tracker.load_all_races()
        print_success(f"Loaded {len(all_races)} suspicious races")

        # Get pending races
        pending = tracker.get_pending_races()
        print_success(f"Pending races: {len(pending)}")

        # Get completed races
        completed = tracker.get_completed_races()
        print_success(f"Completed races: {len(completed)}")

        # Get statistics
        stats = tracker.get_statistics()
        print_success("Statistics:")
        print(f"    - Total: {stats['total']}")
        print(f"    - Completed: {stats['completed']}")
        print(f"    - Pending: {stats['pending']}")
        print(f"    - Skipped: {stats['skipped']}")
        print(f"    - Critical: {stats['critical']}")
        print(f"    - High: {stats['high']}")
        print(f"    - Medium: {stats['medium']}")

        # Get progress percentage
        progress = tracker.get_progress_percentage()
        print_success(f"Progress: {progress:.1f}%")

        return True

    except Exception as e:
        print_error(f"ProgressTracker test failed: {e}")
        return False


def test_metadata_manager():
    """Test MetadataManager component."""
    print_header("Testing MetadataManager / تست مدیر متادیتا")

    try:
        # Initialize
        mgr = MetadataManager()
        print_success("MetadataManager initialized")

        # Test with a known corrected race
        test_competition = 'chamonix_2024'
        test_race_id = 'Speed_finals_Chamonix_2024_race001'

        # Check if metadata exists
        if mgr.metadata_exists(test_competition, test_race_id):
            print_success(f"Metadata file exists for {test_race_id}")

            # Load metadata
            metadata = mgr.load_metadata(test_competition, test_race_id)
            print_success("Metadata loaded successfully")
            print(f"    - Duration: {metadata.get('race_duration', 'N/A')}s")
            print(f"    - Manual correction: {metadata.get('manual_correction', False)}")
            print(f"    - Start frame: {metadata.get('detected_start_frame', 'N/A')}")
            print(f"    - Finish frame: {metadata.get('detected_finish_frame', 'N/A')}")

            if metadata.get('manual_correction'):
                print_success("This race has been manually corrected")
                corr_meta = metadata.get('correction_metadata', {})
                print(f"    - Original duration: {corr_meta.get('original_detected_duration', 'N/A')}")
                print(f"    - Correction date: {corr_meta.get('correction_date', 'N/A')}")
                print(f"    - Reason: {corr_meta.get('correction_reason', 'N/A')}")

        else:
            print_error(f"Metadata file not found for {test_race_id}")
            print("    (This is expected if race has not been processed yet)")

        # Check if video exists
        if mgr.video_exists(test_competition, test_race_id):
            print_success(f"Video file exists for {test_race_id}")
            video_path = mgr.get_video_path(test_competition, test_race_id)
            print(f"    - Path: {video_path}")
        else:
            print_error(f"Video file not found for {test_race_id}")

        return True

    except Exception as e:
        print_error(f"MetadataManager test failed: {e}")
        print(f"    Error details: {type(e).__name__}: {e}")
        return False


def test_validators():
    """Test RaceValidator component."""
    print_header("Testing RaceValidator / تست اعتبارسنجی")

    try:
        # Initialize
        validator = RaceValidator()
        print_success("RaceValidator initialized")

        # Test cases
        test_cases = [
            ("Valid race", 100, 300, 30.0, True),
            ("Too short", 100, 120, 30.0, False),
            ("Too long", 100, 700, 30.0, False),
            ("Negative duration", 300, 100, 30.0, False),
        ]

        for name, start, finish, fps, should_pass in test_cases:
            results = validator.validate_all(start, finish, fps, 1000)
            is_valid = validator.is_all_valid(results)

            if is_valid == should_pass:
                print_success(f"Test '{name}': {'PASS' if is_valid else 'FAIL'} (expected)")
                duration = (finish - start) / fps
                print(f"    - Duration: {duration:.2f}s")
                for check_name, (valid, msg, severity) in results.items():
                    status = "✓" if valid else "✗"
                    print(f"    - {status} {check_name}: {msg}")
            else:
                print_error(f"Test '{name}': Unexpected result!")

        return True

    except Exception as e:
        print_error(f"RaceValidator test failed: {e}")
        return False


def run_all_tests():
    """Run all component tests."""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  MANUAL REVIEW INTERFACE - COMPONENT TESTS                       ║")
    print("║  تست کامپوننت‌های رابط بررسی دستی                                ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    results = {
        'ConfigManager': test_config_manager(),
        'ProgressTracker': test_progress_tracker(),
        'MetadataManager': test_metadata_manager(),
        'RaceValidator': test_validators(),
    }

    # Summary
    print_header("Test Summary / خلاصه تست‌ها")

    all_passed = all(results.values())
    passed = sum(results.values())
    total = len(results)

    for component, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {component}")

    print()
    print(f"  Results: {passed}/{total} tests passed")

    if all_passed:
        print()
        print("  ╔═══════════════════════════════════════════════════════════╗")
        print("  ║  ✓ ALL TESTS PASSED! / همه تست‌ها موفق!                  ║")
        print("  ║                                                           ║")
        print("  ║  You can now run the interface:                           ║")
        print("  ║  می‌توانید رابط را اجرا کنید:                              ║")
        print("  ║                                                           ║")
        print("  ║    cd scripts/review_interface                            ║")
        print("  ║    streamlit run app.py                                   ║")
        print("  ╚═══════════════════════════════════════════════════════════╝")
    else:
        print()
        print("  ╔═══════════════════════════════════════════════════════════╗")
        print("  ║  ✗ SOME TESTS FAILED / برخی تست‌ها شکست خوردند            ║")
        print("  ║                                                           ║")
        print("  ║  Please fix the errors before running the interface.     ║")
        print("  ║  لطفاً خطاها را قبل از اجرای رابط رفع کنید.               ║")
        print("  ╚═══════════════════════════════════════════════════════════╝")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
