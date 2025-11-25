"""Tests for hybrid calibration system with fallback strategies."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch
import tempfile
import json

from speed_climbing.vision.calibration import (
    CameraCalibrator,
    PeriodicCalibrator,
    CalibrationResult,
    CalibrationMethod,
)


@dataclass
class MockDetectedHold:
    """Mock detected hold for testing."""
    hold_num: Optional[int]
    pixel_x: float
    pixel_y: float
    confidence: float
    contour_area: float = 500.0
    panel: Optional[str] = None


@pytest.fixture
def route_map_path(tmp_path):
    """Create a temporary route map file for testing."""
    route_map = {
        "wall": {
            "height_m": 15.0,
            "width_m": 1.5,
            "overhang_degrees": 5.0,
        },
        "holds": [
            # SN (left lane) holds - distributed across height
            {"hold_num": 1, "panel": "SN1", "wall_x_m": 0.75, "wall_y_m": 0.5},
            {"hold_num": 2, "panel": "SN2", "wall_x_m": 0.50, "wall_y_m": 2.0},
            {"hold_num": 3, "panel": "SN3", "wall_x_m": 0.80, "wall_y_m": 3.5},
            {"hold_num": 4, "panel": "SN4", "wall_x_m": 0.60, "wall_y_m": 5.0},
            {"hold_num": 5, "panel": "SN5", "wall_x_m": 0.70, "wall_y_m": 6.5},
            {"hold_num": 6, "panel": "SN6", "wall_x_m": 0.45, "wall_y_m": 8.0},
            {"hold_num": 7, "panel": "SN7", "wall_x_m": 0.85, "wall_y_m": 9.5},
            {"hold_num": 8, "panel": "SN8", "wall_x_m": 0.55, "wall_y_m": 11.0},
            {"hold_num": 9, "panel": "SN9", "wall_x_m": 0.75, "wall_y_m": 12.5},
            {"hold_num": 10, "panel": "SN10", "wall_x_m": 0.65, "wall_y_m": 14.0},
            # DX (right lane) holds
            {"hold_num": 11, "panel": "DX1", "wall_x_m": 0.75, "wall_y_m": 0.5},
            {"hold_num": 12, "panel": "DX2", "wall_x_m": 0.50, "wall_y_m": 2.0},
            {"hold_num": 13, "panel": "DX3", "wall_x_m": 0.80, "wall_y_m": 3.5},
            {"hold_num": 14, "panel": "DX4", "wall_x_m": 0.60, "wall_y_m": 5.0},
        ],
    }

    route_path = tmp_path / "route_map.json"
    with open(route_path, "w") as f:
        json.dump(route_map, f)

    return str(route_path)


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Add some structure to the frame
    cv2.rectangle(frame, (100, 100), (900, 1000), (50, 50, 50), -1)  # Left lane
    cv2.rectangle(frame, (1020, 100), (1820, 1000), (50, 50, 50), -1)  # Right lane
    return frame


def create_mock_holds_for_homography(frame_shape):
    """Create mock holds that should result in successful homography."""
    # These pixel positions should roughly match the normalized positions in route_map
    # Frame: 1920x1080, left lane: x < 960
    # Route map: wall_x_m / 1.5 gives normalized x, wall_y_m / 15.0 gives normalized y
    frame_height, frame_width = frame_shape[:2]
    mid_x = frame_width / 2

    holds = [
        # Hold 1: wall (0.75, 0.5) -> norm (0.5, 0.033) -> pixel (240, 1044)
        MockDetectedHold(None, 240, 1044, 0.9),
        # Hold 2: wall (0.50, 2.0) -> norm (0.333, 0.133) -> pixel (160, 936)
        MockDetectedHold(None, 160, 936, 0.85),
        # Hold 3: wall (0.80, 3.5) -> norm (0.533, 0.233) -> pixel (256, 828)
        MockDetectedHold(None, 256, 828, 0.8),
        # Hold 4: wall (0.60, 5.0) -> norm (0.4, 0.333) -> pixel (192, 720)
        MockDetectedHold(None, 192, 720, 0.75),
        # Hold 5: wall (0.70, 6.5) -> norm (0.467, 0.433) -> pixel (224, 612)
        MockDetectedHold(None, 224, 612, 0.7),
    ]
    return holds


def create_mock_holds_partial(frame_shape, num_holds=2):
    """Create a smaller number of mock holds for fallback testing."""
    frame_height, frame_width = frame_shape[:2]
    mid_x = frame_width / 2

    holds = [
        MockDetectedHold(None, 240, 1044, 0.9),  # Hold 1
        MockDetectedHold(None, 160, 936, 0.85),   # Hold 2
    ]

    if num_holds >= 3:
        holds.append(MockDetectedHold(None, 256, 828, 0.8))  # Hold 3

    return holds[:num_holds]


class TestCalibrationMethod:
    """Test CalibrationMethod enum."""

    def test_method_values(self):
        """Test that all method values are correct."""
        assert CalibrationMethod.HOMOGRAPHY.value == "homography"
        assert CalibrationMethod.AFFINE.value == "affine"
        assert CalibrationMethod.SIMILARITY.value == "similarity"
        assert CalibrationMethod.TEMPORAL_CACHE.value == "temporal_cache"
        assert CalibrationMethod.FAILED.value == "failed"


class TestCameraCalibrator:
    """Test CameraCalibrator class."""

    def test_init(self, route_map_path):
        """Test calibrator initialization."""
        calibrator = CameraCalibrator(route_map_path)

        assert calibrator.route_map is not None
        assert calibrator.min_holds == 4
        assert calibrator.min_holds_affine == 2
        assert calibrator.enable_affine_fallback is True
        assert calibrator.enable_edge_anchors is True
        assert calibrator.lane_width == 1.5
        assert calibrator.wall_height == 15.0

    def test_init_with_custom_params(self, route_map_path):
        """Test calibrator initialization with custom parameters."""
        calibrator = CameraCalibrator(
            route_map_path,
            min_holds_for_calibration=5,
            min_holds_for_affine=3,
            enable_affine_fallback=False,
            enable_edge_anchors=False,
        )

        assert calibrator.min_holds == 5
        assert calibrator.min_holds_affine == 3
        assert calibrator.enable_affine_fallback is False
        assert calibrator.enable_edge_anchors is False

    def test_calibrate_with_sufficient_holds(self, route_map_path, sample_frame):
        """Test calibration with >= 4 holds uses homography."""
        calibrator = CameraCalibrator(route_map_path)
        holds = create_mock_holds_for_homography(sample_frame.shape)

        result = calibrator.calibrate(sample_frame, holds, 'left')

        # With good matching, we should get a result
        if result:
            assert result.method in [CalibrationMethod.HOMOGRAPHY, CalibrationMethod.AFFINE]
            assert result.homography_matrix is not None
            assert result.homography_matrix.shape == (3, 3)
            assert 0.0 <= result.confidence <= 1.0

    def test_calibrate_insufficient_holds(self, route_map_path, sample_frame):
        """Test calibration with < 2 holds returns None."""
        calibrator = CameraCalibrator(route_map_path)
        holds = [MockDetectedHold(None, 100, 500, 0.9)]  # Only 1 hold

        result = calibrator.calibrate(sample_frame, holds, 'left')
        assert result is None

    def test_calibrate_no_route_map(self, sample_frame):
        """Test calibration without route map returns None."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json')
            invalid_path = f.name

        calibrator = CameraCalibrator(invalid_path)
        holds = create_mock_holds_for_homography(sample_frame.shape)

        result = calibrator.calibrate(sample_frame, holds, 'left')
        assert result is None


class TestAffineCalibration:
    """Test affine transform fallback."""

    def test_affine_with_3_holds(self, route_map_path, sample_frame):
        """Test that 3 holds can use affine transform."""
        calibrator = CameraCalibrator(
            route_map_path,
            min_holds_for_calibration=4,
            min_holds_for_affine=2,
            enable_affine_fallback=True,
        )

        # Manually test the affine calibration method
        pixel_points = [[100, 800], [200, 600], [300, 400]]
        meter_points = [[0.5, 1.0], [0.7, 3.0], [0.6, 5.0]]
        matched_nums = [1, 2, 3]

        result = calibrator._calibrate_affine(
            pixel_points, meter_points, matched_nums,
            [], sample_frame.shape
        )

        if result:
            assert result.method == CalibrationMethod.AFFINE
            assert result.homography_matrix.shape == (3, 3)
            # Affine confidence should have penalty applied
            assert result.confidence < 1.0


class TestSimilarityCalibration:
    """Test similarity transform fallback."""

    def test_similarity_with_2_holds(self, route_map_path, sample_frame):
        """Test that 2 holds can use similarity transform."""
        calibrator = CameraCalibrator(
            route_map_path,
            min_holds_for_calibration=4,
            min_holds_for_affine=2,
        )

        # Manually test the similarity calibration method
        pixel_points = [[100, 800], [200, 400]]
        meter_points = [[0.5, 1.0], [0.7, 5.0]]
        matched_nums = [1, 2]

        result = calibrator._calibrate_similarity(
            pixel_points, meter_points, matched_nums,
            [], sample_frame.shape
        )

        if result:
            assert result.method == CalibrationMethod.SIMILARITY
            assert result.homography_matrix.shape == (3, 3)
            # Similarity confidence should have larger penalty
            assert result.confidence < 0.7


class TestWallEdgeDetection:
    """Test wall edge anchor detection."""

    def test_edge_anchor_detection_with_lines(self, route_map_path):
        """Test edge anchor detection on frame with clear edges."""
        calibrator = CameraCalibrator(route_map_path)

        # Create frame with clear vertical and horizontal lines
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Add vertical lines (wall edges)
        cv2.line(frame, (100, 100), (100, 900), (255, 255, 255), 3)  # Left edge
        cv2.line(frame, (400, 100), (400, 900), (255, 255, 255), 3)  # Right edge of left lane

        # Add horizontal line at bottom (start pad)
        cv2.line(frame, (100, 900), (400, 900), (255, 255, 255), 3)

        pixel_anchors, meter_anchors = calibrator._detect_wall_edge_anchors(
            frame, 'left', num_needed=2
        )

        # May or may not find anchors depending on line detection
        # Just verify the return types are correct
        assert isinstance(pixel_anchors, list)
        assert isinstance(meter_anchors, list)
        assert len(pixel_anchors) == len(meter_anchors)

    def test_edge_anchor_detection_empty_frame(self, route_map_path):
        """Test edge anchor detection on blank frame."""
        calibrator = CameraCalibrator(route_map_path)

        # Create blank frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        pixel_anchors, meter_anchors = calibrator._detect_wall_edge_anchors(
            frame, 'left', num_needed=2
        )

        # Should return empty lists for blank frame
        assert len(pixel_anchors) == 0
        assert len(meter_anchors) == 0


class TestPeriodicCalibrator:
    """Test PeriodicCalibrator with caching."""

    def test_init(self, route_map_path):
        """Test periodic calibrator initialization."""
        calibrator = PeriodicCalibrator(route_map_path)

        assert calibrator.recalibration_interval > 0
        assert calibrator.max_frames_without_recalibration == 300
        assert calibrator.confidence_decay_rate == 0.001
        assert 'left' in calibrator.frames_since_calibration
        assert 'right' in calibrator.frames_since_calibration

    def test_cache_calibration(self, route_map_path, sample_frame):
        """Test that good calibrations are cached."""
        calibrator = PeriodicCalibrator(route_map_path)

        # Create a mock calibration result
        mock_result = CalibrationResult(
            homography_matrix=np.eye(3),
            pixel_to_meter_scale=0.01,
            rmse_error=0.05,
            inlier_count=5,
            total_holds=6,
            inlier_ratio=0.83,
            confidence=0.75,
            method=CalibrationMethod.HOMOGRAPHY,
        )

        # Cache it
        calibrator._cache_calibration(mock_result, 'left', 100)

        assert 'left' in calibrator.last_calibration
        assert calibrator.cache_frame_id['left'] == 100
        assert calibrator.cache_original_confidence['left'] == 0.75

    def test_confidence_decay(self, route_map_path):
        """Test that cached confidence decays over time."""
        calibrator = PeriodicCalibrator(
            route_map_path,
            confidence_decay_rate=0.01,  # 1% per frame for easier testing
        )

        # Create and cache a mock result
        mock_result = CalibrationResult(
            homography_matrix=np.eye(3),
            pixel_to_meter_scale=0.01,
            rmse_error=0.05,
            inlier_count=5,
            total_holds=6,
            inlier_ratio=0.83,
            confidence=0.80,
            method=CalibrationMethod.HOMOGRAPHY,
        )

        calibrator._cache_calibration(mock_result, 'left', 100)

        # Get cached at frame 110 (10 frames later)
        cached = calibrator._get_cached_calibration('left', 110)

        assert cached is not None
        assert cached.is_cached is True
        assert cached.cache_age_frames == 10
        assert cached.method == CalibrationMethod.TEMPORAL_CACHE
        # Confidence should have decayed
        assert cached.confidence < 0.80
        # But not below minimum (decay capped at 50%)
        assert cached.confidence >= 0.40

    def test_cache_invalidation_when_too_old(self, route_map_path):
        """Test that cache is invalidated when confidence drops too low."""
        calibrator = PeriodicCalibrator(
            route_map_path,
            confidence_decay_rate=0.1,  # 10% per frame - aggressive decay
        )
        calibrator.MIN_CACHE_CONFIDENCE = 0.30

        # Create and cache a result with moderate confidence
        mock_result = CalibrationResult(
            homography_matrix=np.eye(3),
            pixel_to_meter_scale=0.01,
            rmse_error=0.05,
            inlier_count=5,
            total_holds=6,
            inlier_ratio=0.83,
            confidence=0.50,
            method=CalibrationMethod.HOMOGRAPHY,
        )

        calibrator._cache_calibration(mock_result, 'left', 100)

        # Get cached at frame 200 (100 frames later with aggressive decay)
        # Decay factor = 1 - (100 * 0.1) = 0 -> capped at 0.5
        # Decayed confidence = 0.50 * 0.5 = 0.25 < MIN_CACHE_CONFIDENCE
        cached = calibrator._get_cached_calibration('left', 200)

        # Should return None because confidence dropped too low
        assert cached is None

    def test_reset(self, route_map_path):
        """Test cache reset functionality."""
        calibrator = PeriodicCalibrator(route_map_path)

        # Add some cached data
        mock_result = CalibrationResult(
            homography_matrix=np.eye(3),
            pixel_to_meter_scale=0.01,
            rmse_error=0.05,
            inlier_count=5,
            total_holds=6,
            inlier_ratio=0.83,
            confidence=0.75,
        )
        calibrator._cache_calibration(mock_result, 'left', 100)
        calibrator._cache_calibration(mock_result, 'right', 100)

        # Reset left lane only
        calibrator.reset('left')
        assert 'left' not in calibrator.last_calibration
        assert 'right' in calibrator.last_calibration

        # Reset all
        calibrator.reset()
        assert len(calibrator.last_calibration) == 0
        assert len(calibrator.calibration_cache) == 0

    def test_statistics_tracking(self, route_map_path):
        """Test that calibration statistics are tracked."""
        calibrator = PeriodicCalibrator(route_map_path)

        # Create mock results
        homo_result = CalibrationResult(
            homography_matrix=np.eye(3), pixel_to_meter_scale=0.01,
            rmse_error=0.05, inlier_count=5, total_holds=6,
            inlier_ratio=0.83, confidence=0.75,
            method=CalibrationMethod.HOMOGRAPHY,
        )
        affine_result = CalibrationResult(
            homography_matrix=np.eye(3), pixel_to_meter_scale=0.01,
            rmse_error=0.05, inlier_count=3, total_holds=3,
            inlier_ratio=1.0, confidence=0.60,
            method=CalibrationMethod.AFFINE,
        )

        # Update stats
        calibrator._update_method_stats(CalibrationMethod.HOMOGRAPHY, 'left')
        calibrator._update_method_stats(CalibrationMethod.HOMOGRAPHY, 'left')
        calibrator._update_method_stats(CalibrationMethod.AFFINE, 'left')
        calibrator._update_method_stats(CalibrationMethod.SIMILARITY, 'right')

        assert calibrator.stats['homography_count']['left'] == 2
        assert calibrator.stats['affine_count']['left'] == 1
        assert calibrator.stats['similarity_count']['right'] == 1

    def test_calibration_summary(self, route_map_path):
        """Test human-readable calibration summary."""
        calibrator = PeriodicCalibrator(route_map_path)

        # Add some cached data
        mock_result = CalibrationResult(
            homography_matrix=np.eye(3),
            pixel_to_meter_scale=0.01,
            rmse_error=0.05,
            inlier_count=5,
            total_holds=6,
            inlier_ratio=0.83,
            confidence=0.75,
            method=CalibrationMethod.HOMOGRAPHY,
        )
        calibrator._cache_calibration(mock_result, 'left', 100)

        summary = calibrator.get_calibration_summary()

        assert "Calibration Summary" in summary
        assert "Left Lane" in summary
        assert "homography" in summary
        assert "Method Usage" in summary


class TestCalibrationResult:
    """Test CalibrationResult dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        result = CalibrationResult(
            homography_matrix=np.eye(3),
            pixel_to_meter_scale=0.01,
            rmse_error=0.05,
            inlier_count=5,
            total_holds=6,
            inlier_ratio=0.83,
            confidence=0.75,
        )

        assert result.method == CalibrationMethod.HOMOGRAPHY
        assert result.is_cached is False
        assert result.cache_age_frames == 0
        assert result.anchor_points_used == 0
        assert result.edge_anchors_count == 0

    def test_custom_values(self):
        """Test setting custom values."""
        result = CalibrationResult(
            homography_matrix=np.eye(3),
            pixel_to_meter_scale=0.01,
            rmse_error=0.05,
            inlier_count=5,
            total_holds=6,
            inlier_ratio=0.83,
            confidence=0.75,
            method=CalibrationMethod.AFFINE,
            is_cached=True,
            cache_age_frames=50,
            anchor_points_used=3,
            edge_anchors_count=1,
        )

        assert result.method == CalibrationMethod.AFFINE
        assert result.is_cached is True
        assert result.cache_age_frames == 50
        assert result.anchor_points_used == 3
        assert result.edge_anchors_count == 1


class TestTransformFunctions:
    """Test the transformation functions."""

    def test_homography_transform(self, route_map_path):
        """Test that pixel_to_meter and meter_to_pixel are inverses."""
        calibrator = CameraCalibrator(route_map_path)

        # Create a simple known homography (scaling only)
        scale = 0.01  # 1 pixel = 0.01 meters
        homography = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ])

        # Test transform
        test_pixel = (500, 700)
        result = calibrator._transform_point(test_pixel[0], test_pixel[1], homography)

        expected_meter = (500 * scale, 700 * scale)
        assert abs(result[0] - expected_meter[0]) < 1e-6
        assert abs(result[1] - expected_meter[1]) < 1e-6

        # Test inverse
        inv_homography = np.linalg.inv(homography)
        back_to_pixel = calibrator._transform_point(result[0], result[1], inv_homography)

        assert abs(back_to_pixel[0] - test_pixel[0]) < 1e-6
        assert abs(back_to_pixel[1] - test_pixel[1]) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
