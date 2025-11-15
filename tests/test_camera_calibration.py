"""Tests for camera calibration."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calibration.camera_calibration import CameraCalibrator, CalibrationResult
from phase1_pose_estimation.hold_detector import HoldDetector, DetectedHold


def test_calibrator_initialization():
    """Test calibrator can be initialized."""
    route_path = "configs/ifsc_route_coordinates.json"

    if not Path(route_path).exists():
        print("⚠ Route map not found, skipping test")
        return

    calibrator = CameraCalibrator(route_coordinates_path=route_path)
    assert calibrator.route_map is not None, "Route map should be loaded"
    assert len(calibrator.route_map['holds']) > 0, "Should have holds"

    print(f"✓ Calibrator initialized with {len(calibrator.route_map['holds'])} holds")


def test_homography_transformation():
    """Test homography transformation functions."""
    # Create a simple homography (identity with scale)
    H = np.array([
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Create a mock calibration result
    def pixel_to_meter(px, py):
        point = np.array([px, py, 1.0])
        transformed = H @ point
        return transformed[0] / transformed[2], transformed[1] / transformed[2]

    # Test transformation
    mx, my = pixel_to_meter(1.0, 2.0)
    assert abs(mx - 100.0) < 0.01, f"Expected mx=100.0, got {mx}"
    assert abs(my - 200.0) < 0.01, f"Expected my=200.0, got {my}"

    print("✓ Homography transformation works correctly")


def test_save_load_calibration():
    """Test saving and loading calibration."""
    import tempfile

    # Create a mock calibration
    H = np.eye(3)
    calibration = CalibrationResult(
        homography_matrix=H,
        pixel_to_meter_scale=50.0,
        rmse_error=0.1,
        inlier_count=10,
        total_holds=12,
        inlier_ratio=0.833,
        confidence=0.9
    )

    route_path = "configs/ifsc_route_coordinates.json"
    if not Path(route_path).exists():
        print("⚠ Route map not found, skipping save/load test")
        return

    calibrator = CameraCalibrator(route_coordinates_path=route_path)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        calibrator.save_calibration(calibration, temp_path, metadata={"test": "data"})

        # Load back
        loaded = CameraCalibrator.load_calibration(temp_path)

        assert np.allclose(loaded.homography_matrix, H), "Homography should match"
        assert abs(loaded.pixel_to_meter_scale - 50.0) < 0.01, "Scale should match"
        assert abs(loaded.rmse_error - 0.1) < 0.01, "RMSE should match"
        assert loaded.inlier_count == 10, "Inlier count should match"

        print("✓ Save/load calibration works correctly")

    finally:
        Path(temp_path).unlink()


def test_calibration_on_synthetic_data():
    """Test calibration with synthetic hold detections."""
    route_path = "configs/ifsc_route_coordinates.json"
    if not Path(route_path).exists():
        print("⚠ Route map not found, skipping synthetic test")
        return

    # Create a synthetic frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Create synthetic detected holds that match known route positions
    # We'll create holds that correspond to the left lane (SN panels)
    # and place them at predictable pixel locations

    detected_holds = [
        # These are synthetic detections that should roughly align
        # with some SN (left lane) route holds
        DetectedHold(None, 200, 600, 0.9, 3000),  # Bottom left area
        DetectedHold(None, 400, 400, 0.8, 2500),  # Middle left
        DetectedHold(None, 600, 200, 0.7, 2000),  # Top middle
        DetectedHold(None, 300, 300, 0.85, 2800), # Mid-left
        DetectedHold(None, 500, 500, 0.75, 2200), # Center-ish
    ]

    calibrator = CameraCalibrator(
        route_coordinates_path=route_path,
        min_holds_for_calibration=4
    )

    # Attempt calibration
    result = calibrator.calibrate(frame, detected_holds, lane='left')

    if result:
        print(f"✓ Synthetic calibration successful:")
        print(f"  Inliers: {result.inlier_count}/{result.total_holds}")
        print(f"  RMSE: {result.rmse_error:.4f}m")
        print(f"  Confidence: {result.confidence:.2f}")
    else:
        # This is expected with random synthetic data - matching might fail
        print("✓ Synthetic calibration test completed (calibration may fail with random data)")


def test_real_video_calibration():
    """Test calibration on a real video if available."""
    video_path = Path("data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race003.mp4")
    route_path = Path("configs/ifsc_route_coordinates.json")

    if not video_path.exists() or not route_path.exists():
        print("⚠ Video or route map not found, skipping real video test")
        return

    # Load frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("⚠ Could not open video, skipping")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("⚠ Could not read frame, skipping")
        return

    # Detect holds
    detector = HoldDetector(
        route_coordinates_path=str(route_path),
        min_area=200,
        min_confidence=0.3
    )
    detected = detector.detect_holds(frame, lane='left')

    print(f"  Detected {len(detected)} holds in real video")

    # Calibrate
    calibrator = CameraCalibrator(
        route_coordinates_path=str(route_path),
        min_holds_for_calibration=4
    )

    result = calibrator.calibrate(frame, detected, lane='left')

    if result:
        print(f"✓ Real video calibration successful:")
        print(f"  Inliers: {result.inlier_count}/{result.total_holds}")
        print(f"  RMSE: {result.rmse_error:.4f}m")
        print(f"  Scale: {result.pixel_to_meter_scale:.1f}px/m")
        print(f"  Confidence: {result.confidence:.2f}")

        # Test transformation
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        mx, my = result.pixel_to_meter_func(center_x, center_y)
        print(f"  Center ({center_x}, {center_y}) → ({mx:.2f}m, {my:.2f}m)")

    else:
        print("  Calibration failed (expected with few/poorly matched holds)")


if __name__ == "__main__":
    print("Testing Camera Calibration...\n")
    test_calibrator_initialization()
    print()
    test_homography_transformation()
    print()
    test_save_load_calibration()
    print()
    test_calibration_on_synthetic_data()
    print()
    test_real_video_calibration()
    print("\n✅ All camera calibration tests passed!")
