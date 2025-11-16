"""
Video Extraction Integration
==============================
Integrates with Phase 1 race segmentation to extract new races manually.

Features:
- Manual timestamp entry (MM:SS or HH:MM:SS format)
- FFmpeg integration for segment extraction
- Metadata generation
- Integration with Video Library
- Progress tracking

⭐ Phase 1.5.1 Feature (2025-11-16)
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess
import json
from datetime import datetime
import re


class VideoExtractor:
    """Integrates with Phase 1 extraction scripts."""

    def __init__(self, config_manager):
        """
        Initialize video extractor.

        Args:
            config_manager: ConfigManager instance
        """
        self.config = config_manager
        self.project_root = Path(__file__).parent.parent.parent

    def validate_timestamp(self, timestamp: str) -> Tuple[bool, str, Optional[float]]:
        """
        Validate and parse timestamp string.

        Args:
            timestamp: Time string in MM:SS or HH:MM:SS format

        Returns:
            Tuple of (is_valid, error_message, seconds)
        """
        # Check format with regex
        pattern_mmss = r'^(\d{1,2}):(\d{2})$'
        pattern_hhmmss = r'^(\d{1,2}):(\d{2}):(\d{2})$'

        match_mmss = re.match(pattern_mmss, timestamp)
        match_hhmmss = re.match(pattern_hhmmss, timestamp)

        if match_mmss:
            minutes, seconds = map(int, match_mmss.groups())
            if seconds >= 60:
                return False, "Seconds must be 0-59", None
            total_seconds = minutes * 60 + seconds
            return True, "", total_seconds

        elif match_hhmmss:
            hours, minutes, seconds = map(int, match_hhmmss.groups())
            if minutes >= 60:
                return False, "Minutes must be 0-59", None
            if seconds >= 60:
                return False, "Seconds must be 0-59", None
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return True, "", total_seconds

        else:
            return False, "Invalid format. Use MM:SS or HH:MM:SS", None

    def extract_manual_segment(
        self,
        source_video: Path,
        competition_key: str,
        race_id: str,
        start_time: str,
        end_time: str,
        left_athlete: Dict,
        right_athlete: Dict,
        round_name: str = "",
        buffer_before: float = 3.0,
        buffer_after: float = 1.5
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Extract single race segment using manual timestamps.

        Args:
            source_video: Path to source video file
            competition_key: Competition identifier
            race_id: Unique race ID
            start_time: Manual start time (format: "MM:SS" or "HH:MM:SS")
            end_time: Manual end time
            left_athlete: Dict with name, country, bib_color
            right_athlete: Dict with name, country, bib_color
            round_name: Round description (e.g., "1/8 final - Men")
            buffer_before: Seconds before start
            buffer_after: Seconds after finish

        Returns:
            Tuple of (success, message, output_path)
        """
        # Validate timestamps
        valid_start, msg_start, start_seconds = self.validate_timestamp(start_time)
        if not valid_start:
            return False, f"Invalid start time: {msg_start}", None

        valid_end, msg_end, end_seconds = self.validate_timestamp(end_time)
        if not valid_end:
            return False, f"Invalid end time: {msg_end}", None

        # Check logical order
        if end_seconds <= start_seconds:
            return False, "End time must be after start time", None

        # Get competition config
        comp = self.config.get_competition(competition_key)
        if not comp:
            return False, f"Competition {competition_key} not found in config", None

        # Check source video exists
        if not source_video.exists():
            return False, f"Source video not found: {source_video}", None

        # Prepare output paths
        output_dir = Path(comp.race_segments_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_video = output_dir / f"{race_id}.mp4"
        output_metadata = output_dir / f"{race_id}_metadata.json"

        # Check if race already exists
        if output_video.exists():
            return False, f"Race {race_id} already exists. Choose different ID.", None

        # Calculate extraction parameters
        extract_start = max(0, start_seconds - buffer_before)
        extract_end = end_seconds + buffer_after
        duration = extract_end - extract_start
        race_duration = end_seconds - start_seconds

        # Prepare metadata
        metadata = {
            "race_id": race_id,
            "source_video": str(source_video),
            "round": round_name or "Manual Segmentation",
            "manual_start_time": start_time,
            "manual_end_time": end_time,
            "start_time_seconds": start_seconds,
            "end_time_seconds": end_seconds,
            "race_duration": race_duration,
            "extraction_start_seconds": extract_start,
            "extraction_end_seconds": extract_end,
            "extraction_duration": duration,
            "left_athlete": left_athlete,
            "right_athlete": right_athlete,
            "buffer_before": buffer_before,
            "buffer_after": buffer_after,
            "extraction_date": datetime.now().isoformat(),
            "extraction_method": "manual_ui",
            "notes": "Manually added via Review Interface",
            "manual_correction": True,  # Mark as manually verified
            "detected_start_frame": int(buffer_before * comp.fps),  # Relative to extracted video
            "detected_finish_frame": int((buffer_before + race_duration) * comp.fps)
        }

        try:
            # Use ffmpeg to extract segment
            ffmpeg_cmd = [
                'ffmpeg',
                '-ss', str(extract_start),      # Start time
                '-i', str(source_video),        # Input file
                '-t', str(duration),            # Duration
                '-c', 'copy',                   # Copy codec (fast, no re-encoding)
                '-avoid_negative_ts', '1',      # Fix timestamp issues
                '-y',                           # Overwrite output file
                str(output_video)
            ]

            # Run ffmpeg
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                return False, f"FFmpeg error: {result.stderr}", None

            # Verify output file was created
            if not output_video.exists():
                return False, "FFmpeg completed but output file not found", None

            # Save metadata
            with open(output_metadata, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            return True, f"Successfully extracted {race_id} ({race_duration:.1f}s)", output_video

        except subprocess.TimeoutExpired:
            return False, "FFmpeg timeout (>5 minutes). Video too large?", None

        except FileNotFoundError:
            return False, "FFmpeg not found. Please install ffmpeg first.", None

        except Exception as e:
            # Cleanup partial files
            if output_video.exists():
                output_video.unlink()
            if output_metadata.exists():
                output_metadata.unlink()

            return False, f"Extraction error: {str(e)}", None

    def check_ffmpeg_available(self) -> Tuple[bool, str]:
        """
        Check if ffmpeg is installed and accessible.

        Returns:
            Tuple of (is_available, version_or_error)
        """
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Extract version from first line
                version_line = result.stdout.split('\n')[0]
                return True, version_line

            return False, "FFmpeg found but not working properly"

        except FileNotFoundError:
            return False, "FFmpeg not found. Please install ffmpeg."

        except Exception as e:
            return False, f"Error checking ffmpeg: {str(e)}"

    def suggest_race_id(self, competition_key: str) -> str:
        """
        Suggest next race ID based on existing races.

        Args:
            competition_key: Competition identifier

        Returns:
            Suggested race ID (e.g., "Speed_finals_Paris_2026_race001")
        """
        comp = self.config.get_competition(competition_key)
        if not comp:
            return f"Speed_finals_{competition_key}_race001"

        # Find existing races
        race_dir = Path(comp.race_segments_path)
        if not race_dir.exists():
            return f"Speed_finals_{competition_key}_race001"

        # Count existing races
        existing_races = list(race_dir.glob("*.mp4"))
        next_number = len(existing_races) + 1

        return f"Speed_finals_{competition_key}_race{next_number:03d}"

    def get_video_info(self, video_path: Path) -> Dict:
        """
        Get video information using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with duration, fps, resolution, etc.
        """
        try:
            # Use ffprobe to get video info
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration,r_frame_rate,width,height',
                '-of', 'json',
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                stream = data['streams'][0]

                # Parse frame rate (e.g., "30/1" -> 30.0)
                fps_parts = stream['r_frame_rate'].split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1])

                return {
                    'duration': float(stream.get('duration', 0)),
                    'fps': fps,
                    'width': int(stream['width']),
                    'height': int(stream['height']),
                    'resolution': f"{stream['width']}x{stream['height']}"
                }

            return {}

        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}
