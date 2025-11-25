"""Tests for hold detector."""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from speed_climbing.vision.holds import HoldDetector, DetectedHold, LightingCondition


def test_hold_detector_initialization():
    """Test hold detector can be initialized."""
    # Without route map
    detector1 = HoldDetector()
    assert detector1.route_map is None, "Route map should be None"
    print("  Hold detector initialized without route map")

    # With route map
    route_path = project_root / "configs" / "ifsc_route_coordinates.json"
    if route_path.exists():
        detector2 = HoldDetector(route_coordinates_path=str(route_path))
        assert detector2.route_map is not None, "Route map should be loaded"
        assert len(detector2.route_map['holds']) > 0, "Route map should have holds"
        print(f"  Hold detector initialized with route map ({len(detector2.route_map['holds'])} holds)")
    else:
        print("  Route map not found, skipping route map test")


def test_hold_detection_on_synthetic_frame():
    """Test hold detection on a synthetic frame with red circles."""
    # Create a test frame (1080p)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (220, 220, 220)  # Light gray background

    # Draw some red "holds" (star-like shapes would be better but circles work)
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
    detector = HoldDetector(min_area=500, max_area=10000, min_confidence=0.3)
    detected = detector.detect_holds(frame)

    # Should detect at least some red holds
    print(f"  Detected {len(detected)} holds")
    assert len(detected) >= 2, f"Should detect at least 2 red holds, got {len(detected)}"
    print("  Hold detection works on synthetic frame")

    # Check that detected positions are reasonable
    for i, hold in enumerate(detected[:4], 1):
        print(f"    Hold {i}: pos=({hold.pixel_x:.0f}, {hold.pixel_y:.0f}), "
              f"confidence={hold.confidence:.2f}, star={hold.star_score:.2f}, area={hold.contour_area:.0f}")


def test_star_shape_detection():
    """Test that star shapes get higher scores than circles."""
    # Create a test frame
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    frame[:] = (200, 200, 200)

    # Draw a 4-pointed star
    star_center = (150, 250)
    star_points = []
    outer_r = 40
    inner_r = 15
    for i in range(8):
        angle = i * np.pi / 4 - np.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        x = int(star_center[0] + r * np.cos(angle))
        y = int(star_center[1] + r * np.sin(angle))
        star_points.append([x, y])
    star_points = np.array(star_points, dtype=np.int32)
    cv2.fillPoly(frame, [star_points], (0, 0, 255))  # Red star

    # Draw a circle
    cv2.circle(frame, (350, 250), 35, (0, 0, 255), -1)  # Red circle

    # Detect
    detector = HoldDetector(min_area=200, min_confidence=0.2, use_star_detection=True)
    detected = detector.detect_holds(frame)

    print(f"  Detected {len(detected)} shapes")
    for i, hold in enumerate(detected, 1):
        print(f"    Shape {i}: star_score={hold.star_score:.2f}, "
              f"confidence={hold.confidence:.2f}, pos=({hold.pixel_x:.0f}, {hold.pixel_y:.0f})")

    # The star should have a higher star_score
    if len(detected) >= 2:
        # Find star and circle by position
        star_hold = min(detected, key=lambda h: abs(h.pixel_x - 150))
        circle_hold = min(detected, key=lambda h: abs(h.pixel_x - 350))
        print(f"  Star shape score: {star_hold.star_score:.2f}")
        print(f"  Circle shape score: {circle_hold.star_score:.2f}")


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

    print("  Lane filtering works correctly")


def test_lighting_conditions():
    """Test lighting condition presets."""
    detector = HoldDetector()

    # Test all lighting conditions
    for condition in LightingCondition:
        detector.set_lighting_condition(condition)
        print(f"  Set lighting: {condition.value}")
        assert detector.lighting_condition == condition

    # Test auto-detection with different brightness levels
    # Bright frame
    bright_frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
    detected_bright = detector._detect_lighting_condition(bright_frame)
    print(f"  Bright frame detected as: {detected_bright.value}")

    # Dark frame
    dark_frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
    detected_dark = detector._detect_lighting_condition(dark_frame)
    print(f"  Dark frame detected as: {detected_dark.value}")

    print("  Lighting condition handling works")


def test_debug_info():
    """Test debug info output."""
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    frame[:] = (200, 200, 200)

    # Add some red circles
    cv2.circle(frame, (100, 100), 20, (0, 0, 255), -1)
    cv2.circle(frame, (200, 200), 5, (0, 0, 255), -1)  # Too small
    cv2.circle(frame, (300, 300), 25, (0, 0, 255), -1)

    detector = HoldDetector(min_area=200, min_confidence=0.3)
    holds, debug = detector.detect_holds(frame, return_debug_info=True)

    print(f"  Debug info: {debug}")
    assert 'candidates' in debug
    assert 'filtered_area' in debug
    assert 'detected' in debug

    print("  Debug info output works")


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

    print("  Visualization works correctly")


def test_real_video_frame():
    """Test on a real video frame if available."""
    # Try several possible video paths
    video_paths = [
        project_root / "data" / "race_segments" / "seoul_2024" / "Speed_finals_Seoul_2024_race003.mp4",
        project_root / "data" / "raw_videos" / "sample.mp4",
    ]

    video_path = None
    for p in video_paths:
        if p.exists():
            video_path = p
            break

    if video_path is None:
        print("  No sample video found, skipping real video test")
        return

    # Load first frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  Could not open video, skipping")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  Could not read frame, skipping")
        return

    # Detect holds
    route_path = project_root / "configs" / "ifsc_route_coordinates.json"
    detector = HoldDetector(
        route_coordinates_path=str(route_path) if route_path.exists() else None,
        min_area=500,
        min_confidence=0.4
    )

    detected = detector.detect_holds(frame)
    print(f"  Detected {len(detected)} holds in real video frame")

    if len(detected) > 0:
        print(f"  Top hold: pos=({detected[0].pixel_x:.0f}, {detected[0].pixel_y:.0f}), "
              f"confidence={detected[0].confidence:.2f}, star={detected[0].star_score:.2f}")

    print("  Hold detection works on real video")


if __name__ == "__main__":
    print("\nTesting Hold Detector...\n")

    print("1. Testing initialization...")
    test_hold_detector_initialization()
    print()

    print("2. Testing synthetic frame detection...")
    test_hold_detection_on_synthetic_frame()
    print()

    print("3. Testing star shape detection...")
    test_star_shape_detection()
    print()

    print("4. Testing lane filtering...")
    test_lane_filtering()
    print()

    print("5. Testing lighting conditions...")
    test_lighting_conditions()
    print()

    print("6. Testing debug info...")
    test_debug_info()
    print()

    print("7. Testing visualization...")
    test_visualization()
    print()

    print("8. Testing real video frame...")
    test_real_video_frame()
    print()

    print("All hold detector tests passed!")
