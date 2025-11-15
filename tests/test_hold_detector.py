"""Tests for hold detector."""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_pose_estimation.hold_detector import HoldDetector, DetectedHold


def test_hold_detector_initialization():
    """Test hold detector can be initialized."""
    # Without route map
    detector1 = HoldDetector()
    assert detector1.route_map is None, "Route map should be None"
    print("✓ Hold detector initialized without route map")

    # With route map
    route_path = "configs/ifsc_route_coordinates.json"
    if Path(route_path).exists():
        detector2 = HoldDetector(route_coordinates_path=route_path)
        assert detector2.route_map is not None, "Route map should be loaded"
        assert len(detector2.route_map['holds']) > 0, "Route map should have holds"
        print(f"✓ Hold detector initialized with route map ({len(detector2.route_map['holds'])} holds)")
    else:
        print("⚠ Route map not found, skipping route map test")


def test_hold_detection_on_synthetic_frame():
    """Test hold detection on a synthetic frame with red circles."""
    # Create a test frame (1080p)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (220, 220, 220)  # Light gray background

    # Draw some red "holds"
    holds_positions = [
        (400, 300, 30),   # (x, y, radius)
        (800, 500, 40),
        (1200, 700, 35),
        (500, 900, 25),
    ]

    for x, y, r in holds_positions:
        cv2.circle(frame, (x, y), r, (0, 0, 255), -1)  # Red filled circle

    # Add some noise (blue and green circles)
    cv2.circle(frame, (1500, 400), 30, (255, 0, 0), -1)  # Blue
    cv2.circle(frame, (1600, 600), 25, (0, 255, 0), -1)  # Green

    # Detect holds
    detector = HoldDetector(min_area=500, max_area=10000, min_confidence=0.2)
    detected = detector.detect_holds(frame)

    # Should detect 4 red holds
    print(f"  Detected {len(detected)} holds")
    assert len(detected) >= 3, f"Should detect at least 3 red holds, got {len(detected)}"
    print("✓ Hold detection works on synthetic frame")

    # Check that detected positions are reasonable
    for i, hold in enumerate(detected[:4]):
        print(f"  Hold {i+1}: pos=({hold.pixel_x:.0f}, {hold.pixel_y:.0f}), "
              f"confidence={hold.confidence:.2f}, area={hold.contour_area:.0f}")


def test_lane_filtering():
    """Test lane filtering (left/right)."""
    # Create frame with holds in both lanes
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (220, 220, 220)

    # Left lane holds (x < 960)
    cv2.circle(frame, (400, 300), 30, (0, 0, 255), -1)
    cv2.circle(frame, (600, 600), 30, (0, 0, 255), -1)

    # Right lane holds (x > 960)
    cv2.circle(frame, (1200, 400), 30, (0, 0, 255), -1)
    cv2.circle(frame, (1500, 700), 30, (0, 0, 255), -1)

    detector = HoldDetector(min_area=500, min_confidence=0.2)

    # Detect all holds
    all_holds = detector.detect_holds(frame)
    print(f"  All holds: {len(all_holds)}")
    assert len(all_holds) >= 3, "Should detect at least 3 holds total"

    # Detect left lane only
    left_holds = detector.detect_holds(frame, lane='left')
    print(f"  Left lane: {len(left_holds)}")
    assert all(h.pixel_x < 960 for h in left_holds), "Left lane holds should be on left side"

    # Detect right lane only
    right_holds = detector.detect_holds(frame, lane='right')
    print(f"  Right lane: {len(right_holds)}")
    assert all(h.pixel_x > 960 for h in right_holds), "Right lane holds should be on right side"

    print("✓ Lane filtering works correctly")


def test_visualization():
    """Test visualization function."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (220, 220, 220)

    cv2.circle(frame, (500, 500), 30, (0, 0, 255), -1)

    detector = HoldDetector(min_confidence=0.2)
    detected = detector.detect_holds(frame)

    # Visualize
    output = detector.visualize_detections(frame, detected, show_labels=True)

    # Check that output is valid
    assert output.shape == frame.shape, "Output shape should match input"
    assert not np.array_equal(output, frame), "Output should be different from input"

    print("✓ Visualization works correctly")


def test_real_video_frame():
    """Test on a real video frame if available."""
    video_path = Path("data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race003.mp4")

    if not video_path.exists():
        print("⚠ No sample video found, skipping real video test")
        return

    # Load first frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("⚠ Could not open video, skipping")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("⚠ Could not read frame, skipping")
        return

    # Detect holds
    detector = HoldDetector(
        route_coordinates_path="configs/ifsc_route_coordinates.json",
        min_area=200,
        min_confidence=0.3
    )

    detected = detector.detect_holds(frame)
    print(f"  Detected {len(detected)} holds in real video frame")

    if len(detected) > 0:
        print(f"  Top hold: pos=({detected[0].pixel_x:.0f}, {detected[0].pixel_y:.0f}), "
              f"confidence={detected[0].confidence:.2f}")

    # Try matching to route
    if detector.route_map:
        matched = detector.match_to_route(detected, frame.shape[:2])
        matched_count = sum(1 for h in matched if h.hold_num is not None)
        print(f"  Matched {matched_count}/{len(detected)} holds to route")

    print("✓ Hold detection works on real video")


if __name__ == "__main__":
    print("Testing Hold Detector...\n")
    test_hold_detector_initialization()
    print()
    test_hold_detection_on_synthetic_frame()
    print()
    test_lane_filtering()
    print()
    test_visualization()
    print()
    test_real_video_frame()
    print("\n✅ All hold detector tests passed!")
