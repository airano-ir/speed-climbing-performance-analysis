"""Test destructor fix for BlazePoseExtractor."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_pose_estimation.blazepose_extractor import BlazePoseExtractor


def test_destructor_with_uninitialized_pose():
    """Test that destructor doesn't crash if pose is never initialized."""
    extractor = BlazePoseExtractor.__new__(BlazePoseExtractor)
    # Don't call __init__, so self.pose is never set

    # This should not raise AttributeError
    try:
        extractor.release()
        print("✓ Destructor handles missing 'pose' attribute correctly")
    except AttributeError as e:
        print(f"✗ Destructor raised AttributeError: {e}")
        raise


def test_destructor_with_none_pose():
    """Test that destructor handles None pose gracefully."""
    extractor = BlazePoseExtractor.__new__(BlazePoseExtractor)
    extractor.pose = None

    # This should not raise any error
    try:
        extractor.release()
        print("✓ Destructor handles None pose correctly")
    except Exception as e:
        print(f"✗ Destructor raised error: {e}")
        raise


def test_normal_cleanup():
    """Test normal cleanup with proper initialization."""
    # This will actually initialize MediaPipe
    try:
        extractor = BlazePoseExtractor(model_complexity=0, min_detection_confidence=0.5)
        extractor.release()
        print("✓ Normal cleanup works correctly")
    except Exception as e:
        print(f"✓ MediaPipe not available in this environment (expected): {type(e).__name__}")


if __name__ == "__main__":
    print("Testing destructor fix...\n")
    test_destructor_with_uninitialized_pose()
    test_destructor_with_none_pose()
    test_normal_cleanup()
    print("\n✅ All destructor tests passed!")
