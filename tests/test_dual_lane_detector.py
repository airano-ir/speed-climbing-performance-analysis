"""
Unit Tests for Dual-Lane Detector
==================================

Tests for the dual-lane detection system.

Author: Speed Climbing Research Team
Date: 2025-11-12
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase1_pose_estimation"))

from dual_lane_detector import (
    DualLaneDetector,
    LaneBoundary,
    DualLaneResult,
    visualize_dual_lane
)


class TestLaneBoundary:
    """Test LaneBoundary class."""

    def test_creation(self):
        """Test boundary creation."""
        boundary = LaneBoundary(
            x_center=0.5,
            confidence=0.9,
            frame_width=1280,
            frame_height=720
        )

        assert boundary.x_center == 0.5
        assert boundary.confidence == 0.9
        assert boundary.x_pixel == 640  # 0.5 * 1280

    def test_is_left_lane_normalized(self):
        """Test lane assignment with normalized coordinates."""
        boundary = LaneBoundary(0.5, 0.9, 1280, 720)

        assert boundary.is_left_lane(0.3, normalized=True) is True
        assert boundary.is_left_lane(0.7, normalized=True) is False
        assert boundary.is_left_lane(0.5, normalized=True) is False  # Boundary is right side

    def test_is_left_lane_pixels(self):
        """Test lane assignment with pixel coordinates."""
        boundary = LaneBoundary(0.5, 0.9, 1280, 720)

        assert boundary.is_left_lane(300, normalized=False) is True
        assert boundary.is_left_lane(900, normalized=False) is False

    def test_get_lane_mask_left(self):
        """Test left lane mask generation."""
        boundary = LaneBoundary(0.5, 0.9, 100, 100)
        mask = boundary.get_lane_mask("left")

        assert mask.shape == (100, 100)
        assert mask[:, :50].sum() == 50 * 100  # Left side all 1s
        assert mask[:, 50:].sum() == 0          # Right side all 0s

    def test_get_lane_mask_right(self):
        """Test right lane mask generation."""
        boundary = LaneBoundary(0.5, 0.9, 100, 100)
        mask = boundary.get_lane_mask("right")

        assert mask.shape == (100, 100)
        assert mask[:, :50].sum() == 0          # Left side all 0s
        assert mask[:, 50:].sum() == 50 * 100  # Right side all 1s

    def test_invalid_lane(self):
        """Test invalid lane name."""
        boundary = LaneBoundary(0.5, 0.9, 100, 100)

        with pytest.raises(ValueError):
            boundary.get_lane_mask("middle")


class TestDualLaneDetector:
    """Test DualLaneDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return DualLaneDetector(
            model_complexity=0,  # Fastest for testing
            boundary_detection_method="fixed"
        )

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.model_complexity == 0
        assert detector.boundary_detection_method == "fixed"
        assert detector.total_frames == 0

    def test_fixed_boundary_detection(self, detector):
        """Test fixed boundary detection."""
        # Create dummy frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        boundary = detector.detect_lane_boundary(frame)

        assert boundary.x_center == 0.5
        assert boundary.confidence == 1.0
        assert boundary.frame_width == 1280
        assert boundary.frame_height == 720

    def test_edge_boundary_detection(self):
        """Test edge-based boundary detection."""
        detector = DualLaneDetector(boundary_detection_method="edge")

        # Create frame with clear vertical edge in middle
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :640] = 100  # Left side gray
        frame[:, 640:] = 200  # Right side brighter

        boundary = detector.detect_lane_boundary(frame)

        # Should detect edge near center
        assert 0.4 < boundary.x_center < 0.6
        assert boundary.confidence > 0.3

    def test_process_frame_returns_result(self, detector):
        """Test that process_frame returns DualLaneResult."""
        with detector:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            result = detector.process_frame(frame, 0, 0.0)

            assert isinstance(result, DualLaneResult)
            assert result.frame_id == 0
            assert result.timestamp == 0.0
            assert result.lane_boundary is not None

    def test_statistics_empty(self, detector):
        """Test statistics with no frames processed."""
        stats = detector.get_statistics()

        assert stats['total_frames'] == 0
        assert stats['left_detection_rate'] == 0.0
        assert stats['right_detection_rate'] == 0.0

    def test_statistics_after_processing(self, detector):
        """Test statistics after processing frames."""
        with detector:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            # Process 10 frames
            for i in range(10):
                detector.process_frame(frame, i, i / 30.0)

            stats = detector.get_statistics()

            assert stats['total_frames'] == 10
            assert 0.0 <= stats['left_detection_rate'] <= 1.0
            assert 0.0 <= stats['right_detection_rate'] <= 1.0

    def test_kalman_smoothing(self):
        """Test boundary smoothing with Kalman filter."""
        detector = DualLaneDetector(
            boundary_detection_method="fixed",
            enable_lane_smoothing=True
        )

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Process multiple frames
        boundaries = []
        for i in range(5):
            boundary = detector.detect_lane_boundary(frame)
            boundaries.append(boundary.x_center)

        # All should be close to 0.5
        for b in boundaries:
            assert 0.48 < b < 0.52

    def test_context_manager(self):
        """Test context manager usage."""
        detector = DualLaneDetector()

        with detector:
            assert detector.left_extractor is not None
            assert detector.right_extractor is not None

        # After exit, should be cleaned up
        assert detector.left_extractor is None
        assert detector.right_extractor is None


class TestVisualization:
    """Test visualization functions."""

    def test_visualize_dual_lane(self):
        """Test visualization function."""
        # Create dummy result
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        boundary = LaneBoundary(0.5, 0.9, 1280, 720)
        result = DualLaneResult(
            frame_id=0,
            timestamp=0.0,
            lane_boundary=boundary,
            left_climber=None,
            right_climber=None
        )

        annotated = visualize_dual_lane(frame, result)

        assert annotated.shape == frame.shape
        # Check that boundary line was drawn (not all black)
        assert annotated.sum() > 0


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline_synthetic_video(self):
        """Test full pipeline with synthetic video."""
        # Create synthetic video frames
        frames = []
        for i in range(30):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Add some noise to make it interesting
            frame += np.random.randint(0, 50, frame.shape, dtype=np.uint8)
            frames.append(frame)

        # Process with detector
        detector = DualLaneDetector(boundary_detection_method="fixed")

        results = []
        with detector:
            for i, frame in enumerate(frames):
                result = detector.process_frame(frame, i, i / 30.0)
                results.append(result)

        # Verify results
        assert len(results) == 30

        for i, result in enumerate(results):
            assert result.frame_id == i
            assert abs(result.timestamp - i / 30.0) < 0.001
            assert result.lane_boundary.x_center == 0.5

        # Check statistics
        stats = detector.get_statistics()
        assert stats['total_frames'] == 30


def test_create_test_frame_with_two_climbers():
    """
    Test creating a test frame with two simulated climbers.

    This verifies the helper function for creating synthetic test data.
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Draw left climber (simple stick figure)
    # Left lane: x < 640
    cv2.circle(frame, (300, 200), 20, (255, 255, 255), -1)  # Head
    cv2.line(frame, (300, 220), (300, 400), (255, 255, 255), 5)  # Body
    cv2.line(frame, (300, 250), (250, 350), (255, 255, 255), 5)  # Left arm
    cv2.line(frame, (300, 250), (350, 350), (255, 255, 255), 5)  # Right arm
    cv2.line(frame, (300, 400), (250, 550), (255, 255, 255), 5)  # Left leg
    cv2.line(frame, (300, 400), (350, 550), (255, 255, 255), 5)  # Right leg

    # Draw right climber
    # Right lane: x >= 640
    cv2.circle(frame, (900, 200), 20, (255, 255, 255), -1)  # Head
    cv2.line(frame, (900, 220), (900, 400), (255, 255, 255), 5)  # Body
    cv2.line(frame, (900, 250), (850, 350), (255, 255, 255), 5)  # Left arm
    cv2.line(frame, (900, 250), (950, 350), (255, 255, 255), 5)  # Right arm
    cv2.line(frame, (900, 400), (850, 550), (255, 255, 255), 5)  # Left leg
    cv2.line(frame, (900, 400), (950, 550), (255, 255, 255), 5)  # Right leg

    # Verify frame was created properly
    assert frame.shape == (720, 1280, 3)
    assert frame.sum() > 0  # Not all black


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
