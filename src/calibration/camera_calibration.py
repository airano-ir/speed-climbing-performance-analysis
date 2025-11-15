"""Camera calibration using homography transformation.

This module implements camera calibration for speed climbing videos using
detected holds and the IFSC route map. It supports both static and dynamic
cameras through frame-by-frame calibration.

Key features:
- Homography-based pixel-to-meter conversion
- Handles camera movement (per-frame calibration)
- RANSAC for robust hold matching
- Calibration quality metrics (RMSE, inlier ratio)
- Supports partial wall visibility

Background:
In speed climbing videos, the camera often pans/zooms to follow the athlete.
To handle this, we detect holds in each frame and recompute the calibration
dynamically, allowing accurate tracking even with camera movement.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of camera calibration."""

    homography_matrix: np.ndarray  # 3x3 transformation matrix
    pixel_to_meter_scale: float  # Average scale (pixels per meter)
    rmse_error: float  # Root mean square error (meters)
    inlier_count: int  # Number of holds used for calibration
    total_holds: int  # Total holds detected
    inlier_ratio: float  # Inlier ratio (0-1)
    confidence: float  # Overall calibration confidence (0-1)

    # Transformation functions
    pixel_to_meter_func: Optional[Callable[[float, float], Tuple[float, float]]] = None
    meter_to_pixel_func: Optional[Callable[[float, float], Tuple[float, float]]] = None


class CameraCalibrator:
    """Calibrate camera using detected holds and IFSC route map.

    This calibrator computes a homography transformation from pixel coordinates
    to wall coordinates (in meters) using detected hold positions. It's designed
    to work with moving cameras by allowing per-frame calibration.
    """

    def __init__(
        self,
        route_coordinates_path: str,
        min_holds_for_calibration: int = 4,
        ransac_threshold: float = 0.05,  # meters
        min_inlier_ratio: float = 0.5
    ):
        """Initialize calibrator.

        Args:
            route_coordinates_path: Path to IFSC route coordinates JSON
            min_holds_for_calibration: Minimum holds needed for calibration
            ransac_threshold: RANSAC reprojection threshold (meters)
            min_inlier_ratio: Minimum ratio of inliers for valid calibration
        """
        self.min_holds = min_holds_for_calibration
        self.ransac_threshold = ransac_threshold
        self.min_inlier_ratio = min_inlier_ratio

        # Load route map
        with open(route_coordinates_path, 'r') as f:
            self.route_map = json.load(f)

        logger.info(f"Loaded route map with {len(self.route_map['holds'])} holds")

    def calibrate(
        self,
        frame: np.ndarray,
        detected_holds: List,  # List[DetectedHold] from hold_detector
        lane: str = 'left'
    ) -> Optional[CalibrationResult]:
        """Calibrate camera for a single frame.

        Args:
            frame: Video frame (used for dimensions)
            detected_holds: List of DetectedHold objects from HoldDetector
            lane: Which lane to calibrate ('left' or 'right')

        Returns:
            CalibrationResult if successful, None if calibration failed
        """
        if len(detected_holds) < self.min_holds:
            logger.warning(
                f"Not enough holds detected ({len(detected_holds)}) for calibration "
                f"(need at least {self.min_holds})"
            )
            return None

        # Filter route map by lane
        lane_prefix = 'SN' if lane == 'left' else 'DX'
        route_holds = [
            h for h in self.route_map['holds']
            if h['panel'].startswith(lane_prefix)
        ]

        if len(route_holds) == 0:
            logger.error(f"No route holds found for lane {lane}")
            return None

        # Match detected holds to route holds
        pixel_points, meter_points, matched_holds = self._match_holds_to_route(
            detected_holds, route_holds, frame.shape
        )

        if len(pixel_points) < self.min_holds:
            logger.warning(
                f"Not enough matched holds ({len(pixel_points)}) for calibration"
            )
            return None

        # Convert lists to numpy arrays
        pixel_points = np.array(pixel_points, dtype=np.float32)
        meter_points = np.array(meter_points, dtype=np.float32)

        # Compute homography with RANSAC
        homography, inliers = self._compute_homography_ransac(
            pixel_points, meter_points
        )

        if homography is None:
            logger.warning("Failed to compute homography")
            return None

        # Count inliers
        inlier_count = np.sum(inliers)
        inlier_ratio = inlier_count / len(pixel_points)

        if inlier_ratio < self.min_inlier_ratio:
            logger.warning(
                f"Inlier ratio too low ({inlier_ratio:.2f} < {self.min_inlier_ratio})"
            )
            return None

        # Calculate RMSE for inliers
        rmse = self._calculate_rmse(
            pixel_points[inliers == 1],
            meter_points[inliers == 1],
            homography
        )

        # Calculate average pixel-to-meter scale
        scale = self._calculate_scale(pixel_points, meter_points, homography)

        # Calculate overall confidence
        confidence = min(
            inlier_ratio * 1.5,  # Favor high inlier ratio
            1.0 - min(rmse / 0.5, 1.0)  # Penalize high RMSE
        )

        # Create transformation functions
        pixel_to_meter_func = lambda px, py: self._transform_point(
            px, py, homography
        )
        meter_to_pixel_func = lambda mx, my: self._transform_point(
            mx, my, np.linalg.inv(homography)
        )

        result = CalibrationResult(
            homography_matrix=homography,
            pixel_to_meter_scale=scale,
            rmse_error=rmse,
            inlier_count=inlier_count,
            total_holds=len(detected_holds),
            inlier_ratio=inlier_ratio,
            confidence=confidence,
            pixel_to_meter_func=pixel_to_meter_func,
            meter_to_pixel_func=meter_to_pixel_func
        )

        logger.info(
            f"Calibration successful: {inlier_count}/{len(pixel_points)} inliers, "
            f"RMSE={rmse:.4f}m, scale={scale:.1f}px/m, confidence={confidence:.2f}"
        )

        # Quality warnings
        if rmse > 0.10:  # 10cm threshold
            logger.warning(
                f"⚠️  High RMSE: {rmse:.3f}m (>{0.10:.2f}m) - "
                f"calibration may be inaccurate. Consider detecting more holds."
            )

        if inlier_count < 10:
            logger.warning(
                f"⚠️  Low hold count: {inlier_count} holds detected - "
                f"accuracy may be limited. Aim for 15-20 holds for best results."
            )

        if inlier_ratio < 0.7:
            logger.warning(
                f"⚠️  Low inlier ratio: {inlier_ratio:.2f} - "
                f"many holds rejected by RANSAC. Check for occlusions or mismatches."
            )

        if confidence < 0.6:
            logger.warning(
                f"⚠️  Low confidence: {confidence:.2f} - "
                f"calibration may be unreliable. Consider recalibration."
            )

        return result

    def _match_holds_to_route(
        self,
        detected_holds: List,
        route_holds: List[dict],
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[List, List, List]:
        """Match detected holds to route holds using nearest neighbor.

        Args:
            detected_holds: List of DetectedHold objects
            route_holds: List of route hold dicts from IFSC map
            frame_shape: Frame shape (height, width, channels)

        Returns:
            Tuple of (pixel_points, meter_points, matched_hold_nums)
        """
        frame_height, frame_width = frame_shape[:2]

        pixel_points = []
        meter_points = []
        matched_holds = []

        # Build a spatial index of route holds (simple list for now)
        # For better performance with many holds, use scipy.spatial.KDTree

        for detected in detected_holds:
            # Normalize detected position to 0-1
            norm_x = detected.pixel_x / frame_width
            norm_y = 1.0 - (detected.pixel_y / frame_height)  # Flip Y (image vs wall coords)

            # Find nearest route hold
            best_match = None
            best_distance = float('inf')

            for route_hold in route_holds:
                # Normalize route position to 0-1
                # Wall dimensions: 3m width × 15m height
                route_norm_x = route_hold['wall_x_m'] / 3.0
                route_norm_y = route_hold['wall_y_m'] / 15.0

                # Euclidean distance in normalized space
                dist = np.sqrt(
                    (norm_x - route_norm_x)**2 +
                    (norm_y - route_norm_y)**2
                )

                if dist < best_distance:
                    best_distance = dist
                    best_match = route_hold

            # Accept match if distance is reasonable
            # (< 0.2 in normalized space = within ~20% of frame)
            if best_distance < 0.2 and best_match:
                pixel_points.append([detected.pixel_x, detected.pixel_y])
                meter_points.append([
                    best_match['wall_x_m'],
                    best_match['wall_y_m']
                ])
                matched_holds.append(best_match['hold_num'])

        logger.debug(
            f"Matched {len(pixel_points)}/{len(detected_holds)} detected holds "
            f"to route holds"
        )

        return pixel_points, meter_points, matched_holds

    def _compute_homography_ransac(
        self,
        pixel_points: np.ndarray,
        meter_points: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute homography using RANSAC.

        Args:
            pixel_points: Nx2 array of pixel coordinates
            meter_points: Nx2 array of meter coordinates

        Returns:
            Tuple of (homography_matrix, inliers_mask) or (None, None) if failed
        """
        if len(pixel_points) < 4:
            return None, None

        try:
            homography, inliers = cv2.findHomography(
                pixel_points,
                meter_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=2000,
                confidence=0.995
            )

            if homography is None or inliers is None:
                return None, None

            return homography, inliers.ravel()

        except cv2.error as e:
            logger.error(f"Failed to compute homography: {e}")
            return None, None

    def _calculate_rmse(
        self,
        pixel_points: np.ndarray,
        meter_points: np.ndarray,
        homography: np.ndarray
    ) -> float:
        """Calculate RMSE of reprojection error in meters.

        Args:
            pixel_points: Nx2 array of pixel coordinates
            meter_points: Nx2 array of meter coordinates (ground truth)
            homography: 3x3 homography matrix

        Returns:
            RMSE in meters
        """
        if len(pixel_points) == 0:
            return float('inf')

        # Transform pixel points to meters
        ones = np.ones((len(pixel_points), 1))
        pixel_homogeneous = np.hstack([pixel_points, ones])

        transformed = (homography @ pixel_homogeneous.T).T

        # Convert from homogeneous coordinates
        transformed_meters = transformed[:, :2] / transformed[:, 2:3]

        # Calculate errors
        errors = np.linalg.norm(transformed_meters - meter_points, axis=1)

        # RMSE
        rmse = np.sqrt(np.mean(errors**2))

        return rmse

    def _calculate_scale(
        self,
        pixel_points: np.ndarray,
        meter_points: np.ndarray,
        homography: np.ndarray
    ) -> float:
        """Calculate average pixel-to-meter scale factor.

        Args:
            pixel_points: Nx2 array of pixel coordinates
            meter_points: Nx2 array of meter coordinates
            homography: 3x3 homography matrix

        Returns:
            Average scale in pixels per meter
        """
        if len(pixel_points) < 2:
            return 0.0

        # Calculate distances in pixel space
        pixel_dists = []
        meter_dists = []

        for i in range(len(pixel_points) - 1):
            for j in range(i + 1, min(i + 5, len(pixel_points))):  # Sample nearby points
                px_dist = np.linalg.norm(pixel_points[i] - pixel_points[j])
                m_dist = np.linalg.norm(meter_points[i] - meter_points[j])

                if m_dist > 0.01:  # Avoid division by zero
                    pixel_dists.append(px_dist)
                    meter_dists.append(m_dist)

        if not pixel_dists:
            return 0.0

        # Average pixels per meter
        scales = np.array(pixel_dists) / np.array(meter_dists)
        avg_scale = np.median(scales)  # Use median for robustness

        return avg_scale

    def _transform_point(
        self,
        x: float,
        y: float,
        homography: np.ndarray
    ) -> Tuple[float, float]:
        """Transform a point using homography.

        Args:
            x: X coordinate
            y: Y coordinate
            homography: 3x3 transformation matrix

        Returns:
            Tuple of transformed (x, y) coordinates
        """
        point = np.array([x, y, 1.0])
        transformed = homography @ point
        tx = transformed[0] / transformed[2]
        ty = transformed[1] / transformed[2]

        return tx, ty

    def save_calibration(
        self,
        calibration: CalibrationResult,
        output_path: str,
        metadata: Optional[dict] = None
    ):
        """Save calibration to JSON file.

        Args:
            calibration: CalibrationResult to save
            output_path: Path to output JSON file
            metadata: Optional metadata to include (e.g., video name, frame num)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "homography_matrix": calibration.homography_matrix.tolist(),
            "pixel_to_meter_scale": float(calibration.pixel_to_meter_scale),
            "rmse_error": float(calibration.rmse_error),
            "inlier_count": int(calibration.inlier_count),
            "total_holds": int(calibration.total_holds),
            "inlier_ratio": float(calibration.inlier_ratio),
            "confidence": float(calibration.confidence),
            "metadata": metadata or {}
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Calibration saved to {output_path}")

    @staticmethod
    def load_calibration(calibration_path: str) -> CalibrationResult:
        """Load calibration from JSON file.

        Args:
            calibration_path: Path to calibration JSON file

        Returns:
            CalibrationResult object with loaded data
        """
        with open(calibration_path, 'r') as f:
            data = json.load(f)

        homography = np.array(data['homography_matrix'])

        # Create transformation functions
        def pixel_to_meter(px, py):
            point = np.array([px, py, 1.0])
            transformed = homography @ point
            return transformed[0] / transformed[2], transformed[1] / transformed[2]

        def meter_to_pixel(mx, my):
            inv_homography = np.linalg.inv(homography)
            point = np.array([mx, my, 1.0])
            transformed = inv_homography @ point
            return transformed[0] / transformed[2], transformed[1] / transformed[2]

        return CalibrationResult(
            homography_matrix=homography,
            pixel_to_meter_scale=data['pixel_to_meter_scale'],
            rmse_error=data['rmse_error'],
            inlier_count=data['inlier_count'],
            total_holds=data['total_holds'],
            inlier_ratio=data['inlier_ratio'],
            confidence=data['confidence'],
            pixel_to_meter_func=pixel_to_meter,
            meter_to_pixel_func=meter_to_pixel
        )


class PeriodicCalibrator(CameraCalibrator):
    """Periodic calibrator that caches calibration results to reduce computation.

    Instead of calibrating every frame, this calibrator:
    1. Calibrates periodically (e.g., every 30 frames)
    2. Caches calibration results
    3. Returns the most recent valid calibration for intermediate frames

    This provides ~30x speedup while still adapting to camera movement.
    """

    def __init__(
        self,
        route_coordinates_path: str,
        recalibration_interval: int = 30,
        min_holds_for_calibration: int = 4,
        ransac_threshold: float = 0.05,
        min_inlier_ratio: float = 0.5,
        min_confidence_for_cache: float = 0.6
    ):
        """Initialize periodic calibrator.

        Args:
            route_coordinates_path: Path to IFSC route coordinates JSON
            recalibration_interval: Calibrate every N frames (default: 30 = 1 sec at 30fps)
            min_holds_for_calibration: Minimum holds needed for calibration
            ransac_threshold: RANSAC reprojection threshold (meters)
            min_inlier_ratio: Minimum ratio of inliers for valid calibration
            min_confidence_for_cache: Minimum confidence to cache a calibration
        """
        super().__init__(
            route_coordinates_path=route_coordinates_path,
            min_holds_for_calibration=min_holds_for_calibration,
            ransac_threshold=ransac_threshold,
            min_inlier_ratio=min_inlier_ratio
        )

        self.recalibration_interval = recalibration_interval
        self.min_confidence_for_cache = min_confidence_for_cache
        self.calibration_cache = {}  # frame_id -> CalibrationResult
        self.last_calibration = None
        self.frames_since_calibration = 0

        logger.info(
            f"Periodic calibrator initialized: recalibrate every {recalibration_interval} frames"
        )

    def calibrate_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        detected_holds: List,
        lane: str = 'left',
        force_recalibration: bool = False
    ) -> Optional[CalibrationResult]:
        """Calibrate for a specific frame with periodic updates.

        Args:
            frame: Video frame
            frame_id: Frame number (0-indexed)
            detected_holds: List of DetectedHold objects
            lane: Which lane to calibrate ('left' or 'right')
            force_recalibration: Force recalibration even if cached

        Returns:
            CalibrationResult (may be cached from recent frame) or None
        """
        # Check if we should recalibrate
        should_recalibrate = (
            force_recalibration or
            frame_id % self.recalibration_interval == 0 or
            self.last_calibration is None
        )

        if should_recalibration:
            # Perform calibration
            calibration = self.calibrate(frame, detected_holds, lane)

            if calibration and calibration.confidence >= self.min_confidence_for_cache:
                # Cache this calibration
                self.calibration_cache[frame_id] = calibration
                self.last_calibration = calibration
                self.frames_since_calibration = 0

                logger.debug(
                    f"Frame {frame_id}: New calibration cached "
                    f"(RMSE={calibration.rmse_error:.4f}m, conf={calibration.confidence:.2f})"
                )
            elif calibration:
                # Calibration succeeded but confidence too low - use but don't cache
                logger.warning(
                    f"Frame {frame_id}: Calibration confidence {calibration.confidence:.2f} "
                    f"below threshold {self.min_confidence_for_cache:.2f} - not cached"
                )
                return calibration
            else:
                # Calibration failed - fall back to last valid calibration
                logger.warning(
                    f"Frame {frame_id}: Calibration failed - using cached calibration"
                )
                if self.last_calibration:
                    self.frames_since_calibration += 1
                return self.last_calibration
        else:
            # Use cached calibration
            self.frames_since_calibration += 1

            # Warn if cached calibration is getting old
            if self.frames_since_calibration > self.recalibration_interval * 3:
                logger.warning(
                    f"Frame {frame_id}: Using calibration from "
                    f"{self.frames_since_calibration} frames ago - may be stale"
                )

        return self.last_calibration

    def get_nearest_calibration(self, frame_id: int) -> Optional[CalibrationResult]:
        """Get the nearest cached calibration for a given frame.

        Args:
            frame_id: Frame number

        Returns:
            Nearest calibration result or None if no cache exists
        """
        if not self.calibration_cache:
            return None

        # Find nearest frame ID in cache
        cached_frames = sorted(self.calibration_cache.keys())

        # Find closest frame
        nearest_frame = min(cached_frames, key=lambda f: abs(f - frame_id))

        return self.calibration_cache[nearest_frame]

    def get_cache_stats(self) -> dict:
        """Get statistics about calibration cache.

        Returns:
            Dictionary with cache statistics
        """
        if not self.calibration_cache:
            return {
                "total_calibrations": 0,
                "avg_rmse": 0.0,
                "avg_confidence": 0.0,
                "frame_coverage": 0.0
            }

        calibrations = list(self.calibration_cache.values())
        rmse_values = [c.rmse_error for c in calibrations]
        confidence_values = [c.confidence for c in calibrations]

        return {
            "total_calibrations": len(self.calibration_cache),
            "avg_rmse": np.mean(rmse_values),
            "min_rmse": np.min(rmse_values),
            "max_rmse": np.max(rmse_values),
            "std_rmse": np.std(rmse_values),
            "avg_confidence": np.mean(confidence_values),
            "min_confidence": np.min(confidence_values),
            "max_confidence": np.max(confidence_values),
            "cached_frames": sorted(self.calibration_cache.keys())
        }


def main():
    """Test camera calibration on a sample frame."""
    import argparse
    import sys
    from pathlib import Path

    # Add src to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from phase1_pose_estimation.hold_detector import HoldDetector

    parser = argparse.ArgumentParser(description="Calibrate camera from video frame")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--route-map",
        type=str,
        default="configs/ifsc_route_coordinates.json",
        help="Path to route map JSON"
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        default=30,
        help="Frame number to calibrate (default: 30)"
    )
    parser.add_argument(
        "--lane",
        type=str,
        default="left",
        choices=['left', 'right'],
        help="Lane to calibrate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/calibration/test_calibration.json",
        help="Output calibration JSON path"
    )

    args = parser.parse_args()

    # Load video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {args.frame_num}")
        return

    print(f"Processing frame {args.frame_num} from {args.video}")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

    # Detect holds
    print("\nDetecting holds...")
    hold_detector = HoldDetector(
        route_coordinates_path=args.route_map,
        min_area=200,
        min_confidence=0.2  # Lower threshold for better hold detection
    )
    detected_holds = hold_detector.detect_holds(frame, lane=args.lane)
    print(f"Detected {len(detected_holds)} holds")

    # Calibrate
    print("\nCalibrating camera...")
    calibrator = CameraCalibrator(
        route_coordinates_path=args.route_map,
        min_holds_for_calibration=4
    )

    calibration = calibrator.calibrate(frame, detected_holds, lane=args.lane)

    if calibration:
        print(f"\n✅ Calibration successful!")
        print(f"   Inliers: {calibration.inlier_count}/{calibration.total_holds}")
        print(f"   RMSE: {calibration.rmse_error:.4f} meters")
        print(f"   Scale: {calibration.pixel_to_meter_scale:.1f} pixels/meter")
        print(f"   Confidence: {calibration.confidence:.2f}")

        # Save calibration
        calibrator.save_calibration(
            calibration,
            args.output,
            metadata={
                "video": args.video,
                "frame_num": args.frame_num,
                "lane": args.lane
            }
        )

        # Test transformation
        print(f"\nTest transformation:")
        test_pixel = (frame.shape[1] // 2, frame.shape[0] // 2)  # Center of frame
        test_meter = calibration.pixel_to_meter_func(*test_pixel)
        print(f"   Center pixel {test_pixel} → {test_meter[0]:.2f}m, {test_meter[1]:.2f}m")

    else:
        print(f"\n❌ Calibration failed")


if __name__ == "__main__":
    main()
