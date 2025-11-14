#!/usr/bin/env python3
"""
Manual Race Segmenter - Extract races using manual timestamps.

This module provides accurate race segmentation using manually provided
timestamps from YAML configuration files. It performs rough cuts with ffmpeg
and then refines the detection for precise start/finish times.
"""

import cv2
import yaml
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase1_pose_estimation.race_start_detector import RaceStartDetector
from phase1_pose_estimation.race_finish_detector import RaceFinishDetector


@dataclass
class RaceSegment:
    """Metadata for a segmented race."""
    race_id: str
    source_video: str
    round: str

    # Timestamps
    manual_start_time: str
    manual_end_time: str
    detected_start_frame: int
    detected_finish_frame: int
    detected_start_time: float
    detected_finish_time: float
    race_duration: float

    # Athletes
    left_athlete: Dict[str, str]
    right_athlete: Dict[str, str]

    # Detection results
    winner: Optional[str]  # 'left', 'right', or None
    time_difference: Optional[float]
    start_confidence: float
    finish_confidence: float

    # File info
    output_path: Optional[str]
    buffer_before: float
    buffer_after: float
    extraction_date: str

    # Optional notes
    notes: Optional[str] = None


class ManualRaceSegmenter:
    """
    Segment races using manual timestamps with refined detection.

    This segmenter uses manual timestamps for rough cuts, then applies
    detection algorithms to find precise start/finish times.
    """

    def __init__(
        self,
        buffer_before: float = 2.0,
        buffer_after: float = 2.0,
        refine_detection: bool = True
    ):
        """
        Initialize segmenter.

        Args:
            buffer_before: Seconds to include before manual start time
            buffer_after: Seconds to include after manual end time
            refine_detection: Whether to run detection for precise timing
        """
        self.buffer_before = buffer_before
        self.buffer_after = buffer_after
        self.refine_detection = refine_detection

        # Initialize detectors if refinement is enabled
        if self.refine_detection:
            self.start_detector = RaceStartDetector(method='motion')
            self.finish_detector = RaceFinishDetector(method='visual')

    def parse_timestamp_to_seconds(self, timestamp: str) -> float:
        """
        Convert timestamp string to seconds.

        Args:
            timestamp: Time in format "MM:SS" or "HH:MM:SS"

        Returns:
            Total seconds as float
        """
        parts = timestamp.strip().split(':')

        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp}")

    def extract_rough_clip(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
        buffer_before: Optional[float] = None,
        buffer_after: Optional[float] = None
    ) -> bool:
        """
        Extract rough clip using ffmpeg (frame-accurate).

        Args:
            video_path: Source video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output file path
            buffer_before: Buffer before start (uses default if None)
            buffer_after: Buffer after end (uses default if None)

        Returns:
            True if successful
        """
        # Use provided buffers or defaults
        if buffer_before is None:
            buffer_before = self.buffer_before
        if buffer_after is None:
            buffer_after = self.buffer_after

        # Calculate duration
        duration = end_time - start_time

        # Add buffers
        actual_start = max(0, start_time - buffer_before)
        actual_duration = duration + buffer_before + buffer_after

        # Build ffmpeg command (frame-accurate with re-encoding)
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-ss', str(actual_start),  # Seek to start
            '-i', str(video_path),  # Input file
            '-t', str(actual_duration),  # Duration
            '-c:v', 'libx264',  # Re-encode for accuracy
            '-preset', 'fast',  # Fast encoding
            '-crf', '23',  # Quality
            '-c:a', 'aac',  # Audio codec
            '-b:a', '128k',  # Audio bitrate
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: ffmpeg failed: {e.stderr.decode()[:200]}")
            return False
        except FileNotFoundError:
            print("  ERROR: ffmpeg not found. Please install ffmpeg.")
            return False

    def refine_start_time(
        self,
        video_path: Path,
        fps: float
    ) -> Optional[Tuple[int, float, float]]:
        """
        Refine start time using motion detector.

        Args:
            video_path: Path to rough clip
            fps: Video FPS

        Returns:
            (frame_id, timestamp, confidence) or None
        """
        try:
            result = self.start_detector.motion_detector.detect_start_from_video(
                video_path,
                max_search_frames=int(fps * 10)  # Search first 10 seconds
            )

            if result:
                frame_id, timestamp, confidence = result
                return (frame_id, timestamp, confidence)

            return None
        except Exception as e:
            print(f"  Warning: Start detection failed: {e}")
            return None

    def refine_finish_time(
        self,
        video_path: Path,
        start_frame: int,
        fps: float
    ) -> Optional[Tuple[int, float, float, Optional[str]]]:
        """
        Refine finish time using visual detector.

        Args:
            video_path: Path to rough clip
            start_frame: Start frame number
            fps: Video FPS

        Returns:
            (finish_frame, timestamp, confidence, winner) or None
        """
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            # Search for finish from start frame to end
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_search_frames = min(total_frames - start_frame, int(fps * 20))  # Max 20 seconds

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Detect finish (visual method)
            finish_frame = None
            finish_confidence = 0.0

            for i in range(max_search_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Check for button color change (simple approach)
                # Use top 20% of frame
                h, w = frame.shape[:2]
                top_region = frame[:int(h * 0.2), :]

                # Check for bright colors (button activation)
                hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)

                # Green button detection (common for finish)
                lower_green = (40, 50, 50)
                upper_green = (80, 255, 255)
                mask_green = cv2.inRange(hsv, lower_green, upper_green)

                green_ratio = cv2.countNonZero(mask_green) / (top_region.shape[0] * top_region.shape[1])

                if green_ratio > 0.02:  # 2% of top region is green
                    finish_frame = start_frame + i
                    finish_confidence = min(green_ratio * 20, 1.0)  # Scale to 0-1
                    break

            cap.release()

            if finish_frame is not None:
                finish_time = finish_frame / fps
                return (finish_frame, finish_time, finish_confidence, None)  # Winner detection TODO

            return None

        except Exception as e:
            print(f"  Warning: Finish detection failed: {e}")
            return None

    def process_single_race(
        self,
        video_path: Path,
        race_config: Dict[str, Any],
        video_config: Dict[str, Any],
        output_dir: Path,
        save_video: bool = True
    ) -> Optional[RaceSegment]:
        """
        Process a single race from config.

        Args:
            video_path: Source video file
            race_config: Race configuration dict
            video_config: Video configuration dict
            output_dir: Output directory
            save_video: Whether to save video clip

        Returns:
            RaceSegment metadata or None if failed
        """
        race_id = race_config['race_id']
        video_name = video_config['name']
        fps = video_config.get('fps', 30.0)

        # Parse timestamps
        start_time = self.parse_timestamp_to_seconds(race_config['start_time'])
        end_time = self.parse_timestamp_to_seconds(race_config['end_time'])

        # Check for late start flag (need more buffer before)
        buffer_before = self.buffer_before
        if race_config.get('late_start', False):
            buffer_before = 3.0  # Use 3s for late starts instead of default 1.5s
            late_start_note = " (LATE START - using 3s buffer)"
        else:
            late_start_note = ""

        # Generate output filename
        race_filename = f"{video_name}_race{race_id:03d}.mp4"
        race_output_path = output_dir / race_filename if save_video else None

        # Extract rough clip to temporary location for detection
        temp_clip_path = output_dir / f"temp_{race_filename}"

        print(f"\n  Race {race_id}: {race_config['round']}{late_start_note}")
        print(f"    Manual times: {race_config['start_time']} - {race_config['end_time']}")

        # Extract rough clip
        success = self.extract_rough_clip(
            video_path,
            start_time,
            end_time,
            temp_clip_path if self.refine_detection else race_output_path,
            buffer_before=buffer_before
        )

        if not success:
            return None

        # Initialize detection results
        detected_start_frame = int(start_time * fps)
        detected_start_time = start_time
        start_confidence = 1.0

        detected_finish_frame = int(end_time * fps)
        detected_finish_time = end_time
        finish_confidence = 1.0

        winner = None
        time_difference = None

        # Refine detection if enabled
        if self.refine_detection:
            print(f"    Refining detection...")

            # Refine start
            start_result = self.refine_start_time(temp_clip_path, fps)
            if start_result:
                rel_frame, rel_time, conf = start_result
                # Convert relative to absolute
                detected_start_frame = int((start_time - self.buffer_before + rel_time) * fps)
                detected_start_time = start_time - self.buffer_before + rel_time
                start_confidence = conf
                print(f"      Start refined: {detected_start_time:.2f}s (confidence: {start_confidence:.2f})")
            else:
                print(f"      Start detection failed, using manual time")

            # Refine finish (relative to detected start in the clip)
            clip_start_frame = int((start_time - buffer_before) * fps)
            relative_start_frame = int((detected_start_time - (start_time - buffer_before)) * fps)

            finish_result = self.refine_finish_time(temp_clip_path, relative_start_frame, fps)
            if finish_result:
                rel_frame, rel_time, conf, win = finish_result
                # Convert relative to absolute
                detected_finish_time = (start_time - buffer_before) + rel_time
                detected_finish_frame = int(detected_finish_time * fps)
                finish_confidence = conf
                winner = win
                print(f"      Finish refined: {detected_finish_time:.2f}s (confidence: {finish_confidence:.2f})")
            else:
                print(f"      Finish detection failed, using manual time")

            # If we want to save the final video, extract again with refined times
            if save_video and race_output_path:
                # Use refined times for final clip
                self.extract_rough_clip(
                    video_path,
                    detected_start_time,
                    detected_finish_time,
                    race_output_path,
                    buffer_before=buffer_before
                )

            # Remove temp clip
            if temp_clip_path.exists():
                temp_clip_path.unlink()

        # Calculate race duration
        race_duration = detected_finish_time - detected_start_time
        print(f"      Duration: {race_duration:.2f}s")

        # Create metadata
        segment = RaceSegment(
            race_id=f"{video_name}_race{race_id:03d}",
            source_video=str(video_path),
            round=race_config['round'],
            manual_start_time=race_config['start_time'],
            manual_end_time=race_config['end_time'],
            detected_start_frame=detected_start_frame,
            detected_finish_frame=detected_finish_frame,
            detected_start_time=detected_start_time,
            detected_finish_time=detected_finish_time,
            race_duration=race_duration,
            left_athlete=race_config['athletes']['left'],
            right_athlete=race_config['athletes']['right'],
            winner=winner,
            time_difference=time_difference,
            start_confidence=start_confidence,
            finish_confidence=finish_confidence,
            output_path=str(race_output_path) if race_output_path else None,
            buffer_before=buffer_before,
            buffer_after=self.buffer_after,
            extraction_date=datetime.now().isoformat(),
            notes=race_config.get('notes')
        )

        # Save metadata
        metadata_path = output_dir / f"{video_name}_race{race_id:03d}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(segment), f, indent=2, ensure_ascii=False)

        return segment

    def segment_from_config(
        self,
        config_path: Path,
        output_dir: Path,
        save_video: bool = True,
        max_races: Optional[int] = None
    ) -> List[RaceSegment]:
        """
        Segment all races from a YAML config file.

        Args:
            config_path: Path to YAML config file
            output_dir: Output directory for clips
            save_video: Whether to save video clips
            max_races: Maximum number of races to process (None = all)

        Returns:
            List of RaceSegment metadata
        """
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        video_config = config['video']
        races_config = config['races']

        # Resolve video path
        video_path = Path(video_config['path'])
        if not video_path.is_absolute():
            video_path = Path.cwd() / video_path

        if not video_path.exists():
            print(f"ERROR: Video file not found: {video_path}")
            return []

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process races
        segments = []
        total_races = len(races_config)
        if max_races:
            total_races = min(total_races, max_races)

        print(f"\nProcessing {video_config['name']}")
        print(f"  Video: {video_path}")
        print(f"  Location: {video_config.get('location', 'Unknown')}")
        print(f"  Total races: {total_races}")
        print(f"  Output: {output_dir}")
        print("="*70)

        for i, race_config in enumerate(races_config[:total_races]):
            segment = self.process_single_race(
                video_path,
                race_config,
                video_config,
                output_dir,
                save_video
            )

            if segment:
                segments.append(segment)

        # Save summary
        summary_path = output_dir / f"{video_config['name']}_summary.json"
        summary = {
            'source_video': str(video_path),
            'location': video_config.get('location', 'Unknown'),
            'event_type': video_config.get('event_type', 'Unknown'),
            'total_races': len(segments),
            'races': [asdict(s) for s in segments],
            'processing_date': datetime.now().isoformat()
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "="*70)
        print(f"Segmentation complete!")
        print(f"  Total races extracted: {len(segments)}")
        print(f"  Summary saved to: {summary_path}")
        print("="*70)

        return segments


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Segment races using manual timestamps'
    )
    parser.add_argument('config', type=str,
                       help='Path to YAML config file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for race clips')
    parser.add_argument('--buffer-before', type=float, default=2.0,
                       help='Buffer before start (seconds)')
    parser.add_argument('--buffer-after', type=float, default=2.0,
                       help='Buffer after finish (seconds)')
    parser.add_argument('--no-refine', action='store_true',
                       help='Disable detection refinement (use manual times only)')
    parser.add_argument('--metadata-only', action='store_true',
                       help='Generate metadata only (no video extraction)')
    parser.add_argument('--max-races', type=int, default=None,
                       help='Maximum number of races to process')

    args = parser.parse_args()

    # Create segmenter
    segmenter = ManualRaceSegmenter(
        buffer_before=args.buffer_before,
        buffer_after=args.buffer_after,
        refine_detection=not args.no_refine
    )

    # Process
    segments = segmenter.segment_from_config(
        Path(args.config),
        Path(args.output_dir),
        save_video=not args.metadata_only,
        max_races=args.max_races
    )

    print(f"\nSuccessfully extracted {len(segments)} race(s)")


if __name__ == '__main__':
    main()
