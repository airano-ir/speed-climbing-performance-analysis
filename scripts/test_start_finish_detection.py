#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Start/Finish Detection
============================
Validates the StartFinishDetector on sample race videos.
Produces visualization and detailed validation report.
"""

import sys
import argparse
from pathlib import Path
import json
import cv2
import numpy as np
from typing import Dict, List, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.vision.pose import BlazePoseExtractor
from speed_climbing.vision.calibration import CameraCalibrator
from speed_climbing.vision.holds import HoldDetector
from speed_climbing.analysis.start_finish_detector import StartFinishDetector, StartEvent, FinishEvent
from speed_climbing.core.settings import IFSC_STANDARDS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class StartFinishValidator:
    """Validates start/finish detection on race videos."""

    def __init__(
        self,
        route_map_path: str,
        output_dir: Optional[Path] = None,
        visualize: bool = True
    ):
        self.route_map_path = route_map_path
        self.output_dir = output_dir or Path('data/processed/start_finish_tests')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualize = visualize

        # Load route map
        with open(route_map_path, 'r') as f:
            self.route_map = json.load(f)

        # Initialize components
        self.pose_extractor = BlazePoseExtractor(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hold_detector = HoldDetector()
        self.calibrator = CameraCalibrator()

    def test_video(self, video_path: str, race_name: Optional[str] = None) -> Dict:
        """
        Test start/finish detection on a single video.

        Args:
            video_path: Path to race video
            race_name: Optional race identifier

        Returns:
            Validation report dict
        """
        video_path = Path(video_path)
        race_name = race_name or video_path.stem

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {race_name}")
        logger.info(f"{'='*60}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {'error': 'Failed to open video'}

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = frame_count / fps if fps > 0 else 0

        logger.info(f"Video: {frame_count} frames @ {fps:.2f} fps ({duration_s:.2f}s)")

        # Initialize detector
        detector = StartFinishDetector()

        # Initialize calibration (using first frame)
        calibration_done = False
        homography = None

        # Data collection
        results = {
            'race_name': race_name,
            'video_path': str(video_path),
            'fps': fps,
            'frame_count': frame_count,
            'duration_s': duration_s,
            'frames_processed': 0,
            'start_event': None,
            'finish_event': None,
            'race_duration_s': None,
            'com_heights': [],
            'validation': {}
        }

        frame_id = 0
        visualization_frames = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_id / fps if fps > 0 else frame_id * 0.033

                # Pose extraction
                pose_result = self.pose_extractor.process_frame(frame, frame_id, timestamp)

                if not pose_result.has_detection:
                    frame_id += 1
                    continue

                # Calibration (one-time, using first good frame)
                if not calibration_done:
                    success, homography = self._calibrate_frame(frame)
                    if success:
                        calibration_done = True
                        logger.info(f"Calibration successful at frame {frame_id}")
                    else:
                        logger.warning(f"Calibration failed at frame {frame_id}")

                # Transform COM to world coordinates
                com_keypoint = pose_result.keypoints.get('COM')
                if com_keypoint and calibration_done and homography is not None:
                    com_world = self._transform_to_world(
                        com_keypoint, frame.shape, homography
                    )
                    com_height_m = com_world[1] if com_world is not None else 0.0
                else:
                    com_height_m = 0.0

                results['com_heights'].append({
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'com_height_m': com_height_m,
                    'confidence': pose_result.overall_confidence
                })

                # Start/Finish detection
                start_event, finish_event = detector.process_frame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    com_height_m=com_height_m,
                    pose_keypoints=pose_result.keypoints,
                    pose_confidence=pose_result.overall_confidence,
                    world_to_wall_transform=homography
                )

                if start_event:
                    results['start_event'] = self._serialize_event(start_event)
                    logger.info(f"✓ START detected: frame {frame_id}, t={timestamp:.2f}s, "
                               f"COM={com_height_m:.3f}m")

                if finish_event:
                    results['finish_event'] = self._serialize_event(finish_event)
                    logger.info(f"✓ FINISH detected: frame {frame_id}, t={timestamp:.2f}s")

                # Visualization (sample key frames)
                if self.visualize and (start_event or finish_event or frame_id % 30 == 0):
                    vis_frame = self._visualize_frame(
                        frame, pose_result, com_height_m, detector, frame_id, timestamp
                    )
                    visualization_frames.append(vis_frame)

                frame_id += 1

                # Stop after finish for efficiency (optional)
                # if finish_event:
                #     break

        finally:
            cap.release()

        results['frames_processed'] = frame_id
        results['race_duration_s'] = detector.get_race_duration()

        # Validation
        results['validation'] = self._validate_results(results)

        # Save results
        self._save_results(race_name, results, visualization_frames)

        return results

    def _calibrate_frame(self, frame: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """Attempt to calibrate using detected holds."""
        holds = self.hold_detector.detect_holds(frame)

        if len(holds) < 4:
            return False, None

        # Match to route map
        route_holds = self.route_map.get('holds', [])
        if not route_holds:
            return False, None

        # Extract positions
        detected_points = np.array([[h['center_x'], h['center_y']] for h in holds], dtype=np.float32)

        # Use first N holds from route map as reference
        n = min(len(holds), len(route_holds))
        reference_points = np.array([
            [h['wall_x_m'], h['wall_y_m']] for h in route_holds[:n]
        ], dtype=np.float32)

        # Compute homography
        success, H = self.calibrator.compute_homography(
            detected_points[:n],
            reference_points[:n]
        )

        return success, H

    def _transform_to_world(
        self,
        keypoint,
        frame_shape,
        homography: np.ndarray
    ) -> Optional[np.ndarray]:
        """Transform keypoint from pixel to world coordinates."""
        if homography is None:
            return None

        height, width = frame_shape[:2]
        pixel_x = int(keypoint.x * width)
        pixel_y = int(keypoint.y * height)

        point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), homography)

        return world_point[0][0]

    def _serialize_event(self, event) -> Dict:
        """Convert event dataclass to dict."""
        return {
            'frame_id': event.frame_id,
            'timestamp': event.timestamp,
            'height_m': getattr(event, 'com_height_m', None) or getattr(event, 'hand_height_m', None),
            'confidence': event.confidence,
            'method': event.method,
            'details': event.details
        }

    def _validate_results(self, results: Dict) -> Dict:
        """Validate detection results."""
        validation = {
            'start_detected': results['start_event'] is not None,
            'finish_detected': results['finish_event'] is not None,
            'race_complete': results['race_duration_s'] is not None,
            'checks': {}
        }

        # Check 1: Start COM height should be 0.8-1.2m
        if results['start_event']:
            start_com = results['start_event'].get('height_m', 0)
            validation['checks']['start_com_height_valid'] = 0.8 <= start_com <= 1.2
            validation['checks']['start_com_height_m'] = start_com

        # Check 2: Finish hand height should be 14.5-15.0m
        if results['finish_event']:
            finish_height = results['finish_event'].get('height_m', 0)
            validation['checks']['finish_height_valid'] = 14.5 <= finish_height <= 15.0
            validation['checks']['finish_height_m'] = finish_height

        # Check 3: Race duration should be reasonable (3-15 seconds for elite)
        if results['race_duration_s']:
            duration = results['race_duration_s']
            validation['checks']['duration_reasonable'] = 3.0 <= duration <= 15.0
            validation['checks']['duration_s'] = duration

        # Check 4: COM trajectory should show upward trend
        if len(results['com_heights']) > 10:
            heights = [h['com_height_m'] for h in results['com_heights'] if h['com_height_m'] > 0]
            if len(heights) > 2:
                trend = np.polyfit(range(len(heights)), heights, 1)[0]
                validation['checks']['upward_trajectory'] = trend > 0
                validation['checks']['trajectory_slope'] = float(trend)

        # Overall pass/fail
        checks = validation['checks']
        validation['overall_pass'] = all([
            validation['start_detected'],
            validation['finish_detected'],
            checks.get('start_com_height_valid', False),
            checks.get('upward_trajectory', False)
        ])

        return validation

    def _visualize_frame(
        self,
        frame: np.ndarray,
        pose_result,
        com_height_m: float,
        detector: StartFinishDetector,
        frame_id: int,
        timestamp: float
    ) -> np.ndarray:
        """Create annotated visualization frame."""
        vis = frame.copy()

        # Draw pose
        vis = self.pose_extractor.draw_landmarks(vis, pose_result)

        # Draw info overlay
        h, w = frame.shape[:2]

        # Status box
        status_text = []
        status_text.append(f"Frame: {frame_id} | Time: {timestamp:.2f}s")
        status_text.append(f"COM Height: {com_height_m:.3f}m")

        if detector.start_event:
            status_text.append(f"START: frame {detector.start_event.frame_id}")
        else:
            status_text.append("START: pending...")

        if detector.finish_event:
            status_text.append(f"FINISH: frame {detector.finish_event.frame_id}")
        else:
            status_text.append("FINISH: pending...")

        # Draw text
        y_offset = 30
        for text in status_text:
            cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)
            y_offset += 25

        return vis

    def _save_results(self, race_name: str, results: Dict, vis_frames: List[np.ndarray]):
        """Save results and visualization."""
        # Save JSON report
        json_path = self.output_dir / f"{race_name}_start_finish_report.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Report saved: {json_path}")

        # Save visualization video
        if vis_frames and len(vis_frames) > 0:
            video_path = self.output_dir / f"{race_name}_start_finish_viz.mp4"
            h, w = vis_frames[0].shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))

            for frame in vis_frames:
                out.write(frame)

            out.release()
            logger.info(f"Visualization saved: {video_path}")

        # Print summary
        self._print_summary(results)

    def _print_summary(self, results: Dict):
        """Print validation summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATION SUMMARY: {results['race_name']}")
        logger.info(f"{'='*60}")

        val = results['validation']

        logger.info(f"Start Detected: {'✓' if val['start_detected'] else '✗'}")
        logger.info(f"Finish Detected: {'✓' if val['finish_detected'] else '✗'}")

        if 'checks' in val:
            checks = val['checks']
            if 'start_com_height_valid' in checks:
                status = '✓' if checks['start_com_height_valid'] else '✗'
                height = checks.get('start_com_height_m', 0)
                logger.info(f"Start COM Height Valid: {status} ({height:.3f}m)")

            if 'duration_s' in checks:
                status = '✓' if checks.get('duration_reasonable', False) else '✗'
                duration = checks['duration_s']
                logger.info(f"Race Duration: {status} ({duration:.2f}s)")

        overall = '✓ PASS' if val.get('overall_pass', False) else '✗ FAIL'
        logger.info(f"\nOverall: {overall}")
        logger.info(f"{'='*60}\n")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'pose_extractor'):
            self.pose_extractor.release()


def main():
    parser = argparse.ArgumentParser(description='Test Start/Finish Detection')
    parser.add_argument('--video', type=str, required=True, help='Path to race video')
    parser.add_argument('--race-name', type=str, help='Race identifier')
    parser.add_argument('--route-map', type=str, default='configs/ifsc_route_coordinates.json',
                       help='Path to route map')
    parser.add_argument('--output-dir', type=str, default='data/processed/start_finish_tests',
                       help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')

    args = parser.parse_args()

    validator = StartFinishValidator(
        route_map_path=args.route_map,
        output_dir=Path(args.output_dir),
        visualize=not args.no_viz
    )

    results = validator.test_video(args.video, args.race_name)

    # Exit with error code if validation failed
    if not results.get('validation', {}).get('overall_pass', False):
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
