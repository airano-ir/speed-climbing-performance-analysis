"""
Dual-Lane Detection for Speed Climbing Races.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

# Optional: Kalman filter for boundary smoothing
try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    KalmanFilter = None


@dataclass
class LaneBoundary:
    """Represents the boundary between left and right climbing lanes."""
    x_center: float  # Vertical line x-coordinate (normalized 0-1)
    confidence: float  # Detection confidence
    frame_width: int
    frame_height: int

    @property
    def x_pixel(self) -> int:
        """Get boundary x-coordinate in pixels."""
        return int(self.x_center * self.frame_width)

    def is_left_lane(self, x: float, normalized: bool = True) -> bool:
        """Check if a point is in the left lane."""
        if not normalized:
            x = x / self.frame_width
        return x < self.x_center

    def get_lane_mask(self, lane: str) -> np.ndarray:
        """Create a binary mask for a specific lane."""
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        if lane == "left":
            mask[:, :self.x_pixel] = 1
        elif lane == "right":
            mask[:, self.x_pixel:] = 1
        else:
            raise ValueError(f"Invalid lane: {lane}")
        return mask


class DualLaneDetector:
    """Detects and separates two climbers in a competitive speed climbing video."""

    def __init__(
        self,
        boundary_detection_method: str = "edge",
        enable_lane_smoothing: bool = True
    ):
        self.boundary_detection_method = boundary_detection_method
        self.enable_lane_smoothing = enable_lane_smoothing
        self.boundary_kf = None
        self.boundary_initialized = False
        self.frame_width = None
        self.frame_height = None

    def _initialize_kalman_filter(self):
        """Initialize Kalman filter for boundary position tracking."""
        if not FILTERPY_AVAILABLE:
            self.boundary_kf = None
            return

        self.boundary_kf = KalmanFilter(dim_x=2, dim_z=1)
        self.boundary_kf.F = np.array([[1., 1.], [0., 1.]])
        self.boundary_kf.H = np.array([[1., 0.]])
        self.boundary_kf.Q = np.array([[0.001, 0.], [0., 0.001]])
        self.boundary_kf.R = np.array([[0.01]])
        self.boundary_kf.x = np.array([[0.5], [0.]])
        self.boundary_kf.P = np.eye(2) * 1000
        self.boundary_initialized = True

    def detect_lane_boundary(self, frame: np.ndarray) -> LaneBoundary:
        """Detect the vertical boundary between left and right lanes."""
        h, w = frame.shape[:2]
        self.frame_height, self.frame_width = h, w

        if self.boundary_detection_method == "fixed":
            x_center = 0.5
            confidence = 1.0
        elif self.boundary_detection_method == "edge":
            x_center, confidence = self._detect_boundary_by_edges(frame)
        else:
            x_center = 0.5
            confidence = 0.5

        if self.enable_lane_smoothing and FILTERPY_AVAILABLE:
            if not self.boundary_initialized:
                self._initialize_kalman_filter()
                if self.boundary_kf:
                    self.boundary_kf.x[0] = x_center

            if self.boundary_kf:
                self.boundary_kf.predict()
                self.boundary_kf.update(np.array([[x_center]]))
                x_center = self.boundary_kf.x[0].item()

        return LaneBoundary(
            x_center=x_center,
            confidence=confidence,
            frame_width=w,
            frame_height=h
        )

    def _detect_boundary_by_edges(self, frame: np.ndarray) -> Tuple[float, float]:
        """Detect boundary using vertical edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.abs(sobel_x)
        vertical_profile = np.sum(sobel_x, axis=0)

        w = frame.shape[1]
        search_start = int(w * 0.2)
        search_end = int(w * 0.8)
        
        if search_end <= search_start:
             return 0.5, 0.0

        search_region = vertical_profile[search_start:search_end]
        peak_idx = np.argmax(search_region)
        peak_value = search_region[peak_idx]

        x_pixel = search_start + peak_idx
        x_center = x_pixel / w
        
        mean_val = np.mean(vertical_profile)
        confidence = min(1.0, peak_value / (mean_val * 3)) if mean_val > 0 else 0.0

        return x_center, confidence
