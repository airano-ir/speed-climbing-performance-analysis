"""
Video Library Manager
======================
Provides centralized view and management of all race videos across all competitions.

Features:
- List all videos (188+ races)
- Filter by competition, status, search query
- Statistics and summaries
- Remove videos (soft/hard delete)
- Quick navigation to video player

⭐ Phase 1.5.1 Enhanced Feature (2025-11-16)
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class VideoEntry:
    """Single video entry in library."""
    race_id: str
    competition: str
    video_path: Path
    metadata_path: Path
    duration: float
    status: str  # 'reviewed', 'suspicious', 'failed', 'pending'
    review_date: Optional[str] = None
    notes: str = ""
    left_athlete: str = ""
    right_athlete: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for display."""
        notes_text = self.notes if self.notes else ""
        notes_display = notes_text[:50] + '...' if len(notes_text) > 50 else notes_text

        return {
            'race_id': self.race_id,
            'competition': self.competition,
            'duration': self.duration,
            'status': self.status,
            'review_date': self.review_date or 'Not reviewed',
            'notes': notes_display,
            'athletes': f"{self.left_athlete} vs {self.right_athlete}"
        }


class VideoLibrary:
    """Manages video library across all competitions."""

    def __init__(self, config_manager):
        """
        Initialize video library.

        Args:
            config_manager: ConfigManager instance with competition data
        """
        self.config = config_manager
        self.competitions = config_manager.get_competitions()

    def get_all_videos(self) -> List[VideoEntry]:
        """
        Get all videos across all competitions.

        Returns:
            List of VideoEntry objects sorted by competition and race_id
        """
        videos = []

        for comp in self.competitions:
            race_segments_path = Path(comp.race_segments_path)

            # Find all video files in competition directory
            if not race_segments_path.exists():
                continue

            for video_file in race_segments_path.glob("*.mp4"):
                race_id = video_file.stem

                # Look for metadata file
                metadata_file = video_file.with_name(f"{race_id}_metadata.json")

                # Load metadata to get duration and status
                duration = 0.0
                status = 'pending'
                notes = ""
                review_date = None
                left_athlete = "Unknown"
                right_athlete = "Unknown"

                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)

                        duration = metadata.get('race_duration', 0.0)
                        notes = metadata.get('notes', '')
                        review_date = metadata.get('review_date', None)

                        # Extract athlete names
                        left_info = metadata.get('left_athlete', {})
                        right_info = metadata.get('right_athlete', {})

                        if isinstance(left_info, dict):
                            left_athlete = left_info.get('name', 'Unknown')
                        if isinstance(right_info, dict):
                            right_athlete = right_info.get('name', 'Unknown')

                        # Determine status
                        if metadata.get('manual_correction', False):
                            status = 'reviewed'
                        elif metadata.get('removed', False):
                            status = 'failed'  # Marked as removed
                        elif duration < 0 or duration < 4.5 or duration > 15.0:
                            status = 'suspicious'
                        else:
                            status = 'pending'

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Error reading metadata for {race_id}: {e}")
                        status = 'failed'

                entry = VideoEntry(
                    race_id=race_id,
                    competition=comp.key,
                    video_path=video_file,
                    metadata_path=metadata_file,
                    duration=duration,
                    status=status,
                    review_date=review_date,
                    notes=notes,
                    left_athlete=left_athlete,
                    right_athlete=right_athlete
                )
                videos.append(entry)

        # Sort by competition and race_id
        videos.sort(key=lambda x: (x.competition, x.race_id))
        return videos

    def filter_videos(
        self,
        videos: List[VideoEntry],
        competition: Optional[str] = None,
        status: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> List[VideoEntry]:
        """
        Filter videos by criteria.

        Args:
            videos: List of VideoEntry objects
            competition: Filter by competition key (case-insensitive)
            status: Filter by status
            search_query: Search in race_id, notes, or athlete names

        Returns:
            Filtered list of VideoEntry objects
        """
        filtered = videos

        if competition and competition != "all":
            filtered = [v for v in filtered if v.competition.lower() == competition.lower()]

        if status and status != "all":
            filtered = [v for v in filtered if v.status == status]

        if search_query:
            query_lower = search_query.lower()
            filtered = [
                v for v in filtered
                if query_lower in v.race_id.lower()
                or (v.notes and query_lower in v.notes.lower())
                or query_lower in v.left_athlete.lower()
                or query_lower in v.right_athlete.lower()
            ]

        return filtered

    def get_statistics(self, videos: List[VideoEntry]) -> Dict:
        """
        Get statistics about video library.

        Returns:
            Dictionary with counts by status and competition
        """
        stats = {
            'total': len(videos),
            'by_status': {},
            'by_competition': {}
        }

        for video in videos:
            # Count by status
            stats['by_status'][video.status] = stats['by_status'].get(video.status, 0) + 1

            # Count by competition
            stats['by_competition'][video.competition] = stats['by_competition'].get(video.competition, 0) + 1

        return stats

    def remove_video(
        self,
        video_entry: VideoEntry,
        remove_files: bool = False
    ) -> tuple[bool, str]:
        """
        Remove video from library.

        Args:
            video_entry: VideoEntry to remove
            remove_files: If True, delete actual files. If False, just mark as removed.

        Returns:
            Tuple of (success, message)
        """
        try:
            if remove_files:
                # Delete video file
                if video_entry.video_path.exists():
                    video_entry.video_path.unlink()

                # Delete metadata file
                if video_entry.metadata_path.exists():
                    video_entry.metadata_path.unlink()

                # Delete backup files
                backup_pattern = f"{video_entry.metadata_path.stem}_backup_*.json"
                backup_dir = video_entry.metadata_path.parent
                for backup_file in backup_dir.glob(backup_pattern):
                    backup_file.unlink()

                return True, f"Hard deleted {video_entry.race_id} and all related files"
            else:
                # Soft delete: mark as removed in metadata
                if video_entry.metadata_path.exists():
                    with open(video_entry.metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    metadata['removed'] = True
                    metadata['removal_date'] = datetime.now().isoformat()

                    with open(video_entry.metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                return True, f"Soft deleted {video_entry.race_id} (marked as removed in metadata)"

        except Exception as e:
            return False, f"Error removing video: {str(e)}"

    def get_video_by_race_id(self, race_id: str) -> Optional[VideoEntry]:
        """
        Find video entry by race ID.

        Args:
            race_id: Race identifier

        Returns:
            VideoEntry or None if not found
        """
        all_videos = self.get_all_videos()

        for video in all_videos:
            if video.race_id == race_id:
                return video

        return None

    def export_library_info(self, videos: List[VideoEntry], output_path: Path, format: str = 'json'):
        """
        Export library information to file.

        Args:
            videos: List of videos to export
            output_path: Output file path
            format: 'json', 'csv', or 'yaml'

        Returns:
            Tuple of (success, message)
        """
        try:
            if format == 'json':
                data = [v.to_dict() for v in videos]
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            elif format == 'csv':
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    if videos:
                        writer = csv.DictWriter(f, fieldnames=videos[0].to_dict().keys())
                        writer.writeheader()
                        for video in videos:
                            writer.writerow(video.to_dict())

            elif format == 'yaml':
                import yaml
                data = [v.to_dict() for v in videos]
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, allow_unicode=True)
            else:
                return False, f"Unsupported format: {format}"

            return True, f"Exported {len(videos)} videos to {output_path}"

        except Exception as e:
            return False, f"Export failed: {str(e)}"
