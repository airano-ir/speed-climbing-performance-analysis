"""
Metadata Manager
================
CRUD operations for race metadata JSON files.

Handles loading, updating, and saving race metadata with automatic backups
and correction tracking.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import shutil


class MetadataManager:
    """Manages race metadata JSON files."""

    def __init__(self, race_segments_dir: str = "data/race_segments"):
        """
        Initialize metadata manager.

        Args:
            race_segments_dir: Base directory for race segments
        """
        self.race_segments_dir = Path(race_segments_dir)

    def load_metadata(self, competition: str, race_id: str) -> Dict:
        """
        Load metadata for a race.

        Args:
            competition: Competition key (e.g., "chamonix_2024")
            race_id: Race ID (e.g., "Speed_finals_Chamonix_2024_race001")

        Returns:
            Metadata dictionary

        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        metadata_path = self._get_metadata_path(competition, race_id)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_metadata(
        self,
        competition: str,
        race_id: str,
        metadata: Dict,
        backup: bool = True
    ) -> None:
        """
        Save updated metadata (with optional backup).

        Args:
            competition: Competition key
            race_id: Race ID
            metadata: Metadata dictionary to save
            backup: Whether to create a backup of existing file
        """
        metadata_path = self._get_metadata_path(competition, race_id)

        # Create backup
        if backup and metadata_path.exists():
            backup_path = metadata_path.with_suffix('.json.bak')
            shutil.copy(metadata_path, backup_path)

        # Write updated metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def update_race_boundaries(
        self,
        competition: str,
        race_id: str,
        new_start_frame: int,
        new_finish_frame: int,
        fps: float,
        correction_reason: str,
        reviewer_notes: str = ""
    ) -> Dict:
        """
        Update race boundaries with manual correction.

        This method:
        - Preserves original detected values
        - Updates race boundaries
        - Recalculates duration and times
        - Sets manual_correction flag
        - Creates audit trail

        Args:
            competition: Competition key
            race_id: Race ID
            new_start_frame: Corrected start frame
            new_finish_frame: Corrected finish frame
            fps: Video FPS
            correction_reason: Reason for correction
            reviewer_notes: Additional reviewer notes

        Returns:
            Updated metadata dictionary
        """
        metadata = self.load_metadata(competition, race_id)

        # Preserve original values (first time correction only)
        if not metadata.get('manual_correction', False):
            # First time correction - save originals
            correction_metadata = {
                'original_detected_start_frame': metadata['detected_start_frame'],
                'original_detected_finish_frame': metadata['detected_finish_frame'],
                'original_detected_duration': f"{metadata['race_duration']}s (INVALID)",
                'correction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'correction_reason': correction_reason,
                'reviewer_notes': reviewer_notes
            }
            metadata['correction_metadata'] = correction_metadata
        else:
            # Update existing correction
            metadata['correction_metadata']['correction_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata['correction_metadata']['correction_reason'] = correction_reason
            metadata['correction_metadata']['reviewer_notes'] = reviewer_notes

        # Update detected values
        metadata['detected_start_frame'] = new_start_frame
        metadata['detected_finish_frame'] = new_finish_frame

        # Recalculate duration and times
        duration_frames = new_finish_frame - new_start_frame
        duration_seconds = duration_frames / fps

        metadata['race_duration'] = duration_seconds
        metadata['detected_start_time'] = new_start_frame / fps
        metadata['detected_finish_time'] = new_finish_frame / fps

        # Set manual correction flag
        metadata['manual_correction'] = True
        metadata['confidence_start'] = 1.0  # Manual review → perfect confidence
        metadata['confidence_finish'] = 1.0

        # Save with backup
        self.save_metadata(competition, race_id, metadata, backup=True)

        return metadata

    def _get_metadata_path(self, competition: str, race_id: str) -> Path:
        """
        Get path to metadata JSON file.

        Args:
            competition: Competition key
            race_id: Race ID

        Returns:
            Path to metadata file
        """
        return self.race_segments_dir / competition / f"{race_id}_metadata.json"

    def get_video_path(self, competition: str, race_id: str) -> Path:
        """
        Get path to race video file.

        Args:
            competition: Competition key
            race_id: Race ID

        Returns:
            Path to video file
        """
        return self.race_segments_dir / competition / f"{race_id}.mp4"

    def metadata_exists(self, competition: str, race_id: str) -> bool:
        """
        Check if metadata file exists.

        Args:
            competition: Competition key
            race_id: Race ID

        Returns:
            True if metadata exists, False otherwise
        """
        return self._get_metadata_path(competition, race_id).exists()

    def video_exists(self, competition: str, race_id: str) -> bool:
        """
        Check if video file exists.

        Args:
            competition: Competition key
            race_id: Race ID

        Returns:
            True if video exists, False otherwise
        """
        return self.get_video_path(competition, race_id).exists()
