"""
Race Segmenter - Extracts individual race clips from long competition videos

This module:
1. Scans long competition videos (1-3 hours)
2. Detects all race starts using audio/motion detection
3. Detects corresponding finishes
4. Extracts and saves individual race clips (5-10 seconds each)
5. Generates metadata for each race

Usage:
    segmenter = RaceSegmenter()
    races = segmenter.segment_video('Seoul_2024.mp4', output_dir='data/race_segments/')

    # Results: 20-30 race clips saved as separate MP4 files

Author: Speed Climbing Analysis Project
Date: 2025-11-13
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase1_pose_estimation.race_start_detector import RaceStartDetector, RaceStartResult
from phase1_pose_estimation.race_finish_detector import RaceFinishDetector, RaceFinishResult


@dataclass
class RaceSegment:
    """Metadata for an extracted race segment"""
    race_id: str
    source_video: str
    start_frame: int
    finish_frame: int
    start_timestamp: float
    finish_timestamp: float
    duration: float
    start_confidence: float
    finish_confidence: float
    lane: str  # 'dual', 'left', 'right', 'unknown'
    output_path: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class RaceSegmenter:
    """
    Segments long competition videos into individual race clips.

    Workflow:
    1. Scan video for race starts (audio beep + motion)
    2. For each start, find corresponding finish
    3. Extract race segment (start - buffer → finish + buffer)
    4. Save as separate video file
    5. Generate metadata JSON
    """

    def __init__(
        self,
        start_detection_method: str = 'fusion',
        finish_detection_method: str = 'visual',
        buffer_before_sec: float = 1.0,
        buffer_after_sec: float = 1.0,
        min_race_duration: float = 3.0,
        max_race_duration: float = 15.0,
        search_window_after_start: float = 20.0
    ):
        """
        Args:
            start_detection_method: Method for start detection
            finish_detection_method: Method for finish detection
            buffer_before_sec: Buffer before start (seconds)
            buffer_after_sec: Buffer after finish (seconds)
            min_race_duration: Minimum valid race duration
            max_race_duration: Maximum valid race duration
            search_window_after_start: Window to search for finish after start
        """
        self.start_detector = RaceStartDetector(method=start_detection_method)
        self.finish_detector = RaceFinishDetector(method=finish_detection_method)

        self.buffer_before_sec = buffer_before_sec
        self.buffer_after_sec = buffer_after_sec
        self.min_race_duration = min_race_duration
        self.max_race_duration = max_race_duration
        self.search_window_after_start = search_window_after_start

    def detect_all_starts(
        self,
        video_path: Path,
        audio_path: Optional[Path] = None,
        max_search_time: float = None,
        max_starts: Optional[int] = None,
        min_gap_between_races: float = 30.0
    ) -> List[RaceStartResult]:
        """
        Detect all race starts in a video using sliding window approach.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            max_search_time: Maximum time to search (None = entire video)
            max_starts: Maximum number of starts to detect (None = all)
            min_gap_between_races: Minimum gap between consecutive races (seconds)

        Returns:
            List of RaceStartResult
        """
        # Get video duration
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        if max_search_time:
            duration = min(duration, max_search_time)

        starts = []
        current_time = 0.0
        search_window_size = 60.0  # Search 60 seconds at a time

        print(f"Searching for race starts using sliding window approach...")
        print(f"  Video duration: {duration:.1f}s")
        print(f"  Min gap between races: {min_gap_between_races}s")

        while current_time < duration:
            if max_starts and len(starts) >= max_starts:
                print(f"  Reached max starts limit ({max_starts})")
                break

            # Search in current window
            window_end = min(current_time + search_window_size, duration)

            print(f"  Searching window: {current_time:.1f}s - {window_end:.1f}s")

            # Use motion detector for this window
            result = self.start_detector.motion_detector.detect_start_from_video(
                video_path,
                max_search_frames=int(search_window_size * fps),
                skip_frames=int(current_time * fps)
            )

            if result:
                frame_id, timestamp, confidence = result

                # Check if this is far enough from previous starts
                is_new_race = True
                for prev_start in starts:
                    if abs(timestamp - prev_start.timestamp) < min_gap_between_races:
                        is_new_race = False
                        print(f"    Skipped: Too close to previous race (gap: {abs(timestamp - prev_start.timestamp):.1f}s)")
                        break

                if is_new_race:
                    start_result = RaceStartResult(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        confidence=confidence,
                        method='motion',
                        motion_confidence=confidence
                    )
                    starts.append(start_result)
                    print(f"    Found race start #{len(starts)} at {timestamp:.1f}s (confidence: {confidence:.2f})")

                    # Jump past this race
                    current_time = timestamp + min_gap_between_races
                else:
                    # Move forward a bit
                    current_time += search_window_size / 2
            else:
                # No start found in this window, move to next
                print(f"    No start found in this window")
                current_time += search_window_size / 2

            # Safety check: don't get stuck
            if current_time >= duration - 5.0:  # Stop if less than 5s remaining
                break

        print(f"\nTotal race starts found: {len(starts)}")
        return starts

    def extract_segment(
        self,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        output_path: Path,
        fps: float
    ) -> bool:
        """
        Extract video segment and save to file.

        Args:
            video_path: Source video path
            start_frame: Start frame (inclusive)
            end_frame: End frame (inclusive)
            output_path: Output video path
            fps: Video FPS

        Returns:
            Success boolean
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return False

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create writer
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Write frames
        frame_count = 0
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            frame_count += 1
            current_frame += 1

        cap.release()
        out.release()

        print(f"  Extracted {frame_count} frames to {output_path.name}")

        return frame_count > 0

    def segment_video(
        self,
        video_path: Path,
        output_dir: Path,
        audio_path: Optional[Path] = None,
        max_races: Optional[int] = None,
        save_video: bool = True,
        min_gap_between_races: float = 30.0
    ) -> List[RaceSegment]:
        """
        Segment video into individual race clips.

        Args:
            video_path: Path to video file
            output_dir: Directory to save race clips
            audio_path: Path to audio file (optional)
            max_races: Maximum number of races to extract (None = all)
            save_video: Whether to save video clips (False = metadata only)

        Returns:
            List of RaceSegment objects
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSegmenting video: {video_path.name}")
        print(f"Output directory: {output_dir}")

        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        cap.release()

        print(f"Video duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"FPS: {fps}, Total frames: {total_frames}")

        # Detect starts
        print("\nDetecting race starts...")
        starts = self.detect_all_starts(
            video_path,
            audio_path,
            max_starts=max_races,
            min_gap_between_races=min_gap_between_races
        )

        print(f"Found {len(starts)} race start(s)")

        if not starts:
            print("No races detected!")
            return []

        # Process each race
        race_segments = []

        for idx, start_result in enumerate(starts):
            if max_races and idx >= max_races:
                break

            print(f"\nProcessing race {idx + 1}/{len(starts)}")
            print(f"  Start: frame {start_result.frame_id}, time {start_result.timestamp:.2f}s")

            # Detect finish
            search_start_frame = start_result.frame_id
            search_end_frame = min(
                start_result.frame_id + int(self.search_window_after_start * fps),
                total_frames
            )

            print(f"  Searching for finish in frames {search_start_frame}-{search_end_frame}...")

            # Try both lanes
            finish_result = None
            lane = 'dual'  # Assume dual-lane by default

            # Try detecting finish (visual method doesn't require lane)
            finish_result = self.finish_detector.detect_from_video(
                video_path,
                lane='unknown',
                start_frame=search_start_frame,
                end_frame=search_end_frame
            )

            if finish_result:
                print(f"  Finish: frame {finish_result.frame_id}, time {finish_result.timestamp:.2f}s")

                # Validate duration
                duration = finish_result.timestamp - start_result.timestamp

                if duration < self.min_race_duration:
                    print(f"  WARNING: Duration {duration:.2f}s < minimum {self.min_race_duration}s. Skipping.")
                    continue

                if duration > self.max_race_duration:
                    print(f"  WARNING: Duration {duration:.2f}s > maximum {self.max_race_duration}s. Skipping.")
                    continue

                print(f"  Duration: {duration:.2f}s")

                # Calculate segment bounds with buffer
                segment_start_frame = max(
                    0,
                    start_result.frame_id - int(self.buffer_before_sec * fps)
                )
                segment_end_frame = min(
                    total_frames - 1,
                    finish_result.frame_id + int(self.buffer_after_sec * fps)
                )

                # Generate race ID
                race_id = f"{video_path.stem}_race{idx+1:03d}"

                # Create segment metadata
                segment = RaceSegment(
                    race_id=race_id,
                    source_video=str(video_path),
                    start_frame=segment_start_frame,
                    finish_frame=segment_end_frame,
                    start_timestamp=start_result.timestamp,
                    finish_timestamp=finish_result.timestamp,
                    duration=duration,
                    start_confidence=start_result.confidence,
                    finish_confidence=finish_result.confidence,
                    lane=lane,
                    metadata={
                        'start_method': start_result.method,
                        'finish_method': finish_result.method,
                        'buffer_before': self.buffer_before_sec,
                        'buffer_after': self.buffer_after_sec,
                        'extraction_date': datetime.now().isoformat()
                    }
                )

                # Extract video if requested
                if save_video:
                    output_video_path = output_dir / f"{race_id}.mp4"
                    success = self.extract_segment(
                        video_path,
                        segment_start_frame,
                        segment_end_frame,
                        output_video_path,
                        fps
                    )

                    if success:
                        segment.output_path = str(output_video_path)

                # Save metadata
                metadata_path = output_dir / f"{race_id}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(segment.to_dict(), f, indent=2)

                print(f"  Saved metadata: {metadata_path.name}")

                race_segments.append(segment)

            else:
                print(f"  WARNING: No finish detected for this race. Skipping.")

        # Save summary
        summary_path = output_dir / f"{video_path.stem}_summary.json"
        summary = {
            'source_video': str(video_path),
            'total_races': len(race_segments),
            'races': [seg.to_dict() for seg in race_segments],
            'processing_date': datetime.now().isoformat()
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Segmentation complete!")
        print(f"  Total races extracted: {len(race_segments)}")
        print(f"  Summary saved to: {summary_path}")
        print(f"{'='*60}")

        return race_segments


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract individual race clips from long competition videos'
    )
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--output-dir', type=str, default='data/race_segments',
                       help='Output directory for race clips')
    parser.add_argument('--audio', type=str, help='Path to audio file (optional)')
    parser.add_argument('--max-races', type=int, help='Maximum races to extract')
    parser.add_argument('--start-method', type=str, default='fusion',
                       choices=['audio', 'motion', 'fusion'],
                       help='Start detection method')
    parser.add_argument('--finish-method', type=str, default='visual',
                       choices=['pose', 'visual', 'combined'],
                       help='Finish detection method')
    parser.add_argument('--metadata-only', action='store_true',
                       help='Generate metadata only (no video extraction)')
    parser.add_argument('--buffer-before', type=float, default=1.0,
                       help='Buffer before start (seconds)')
    parser.add_argument('--buffer-after', type=float, default=1.0,
                       help='Buffer after finish (seconds)')
    parser.add_argument('--min-duration', type=float, default=3.0,
                       help='Minimum race duration (seconds)')
    parser.add_argument('--max-duration', type=float, default=15.0,
                       help='Maximum race duration (seconds)')
    parser.add_argument('--min-gap', type=float, default=30.0,
                       help='Minimum gap between races (seconds)')

    args = parser.parse_args()

    # Create segmenter
    segmenter = RaceSegmenter(
        start_detection_method=args.start_method,
        finish_detection_method=args.finish_method,
        buffer_before_sec=args.buffer_before,
        buffer_after_sec=args.buffer_after,
        min_race_duration=args.min_duration,
        max_race_duration=args.max_duration
    )

    # Segment video
    segments = segmenter.segment_video(
        Path(args.video),
        Path(args.output_dir),
        audio_path=Path(args.audio) if args.audio else None,
        max_races=args.max_races,
        save_video=not args.metadata_only,
        min_gap_between_races=args.min_gap
    )

    print(f"\nSuccessfully extracted {len(segments)} race(s)")
