"""
Progress Tracker
================
Manages CSV file tracking review progress for suspicious races.

The progress tracker maintains a CSV file with all races needing review,
their current status, and any corrections made.
"""

import csv
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RaceReviewStatus:
    """Status of a single race review."""
    priority: int
    race_id: str
    competition: str
    detected_duration_s: float
    duration_frames: int
    confidence_start: float
    confidence_finish: float
    status: str  # "SUSPICIOUS - TOO SHORT" etc.
    issue_description: str
    review_status: str  # "Pending" / "Completed" / "Skipped"
    corrected_duration_s: str  # Empty or float value
    corrected_start_frame: str  # Empty or int value
    corrected_finish_frame: str  # Empty or int value
    reviewer_notes: str
    review_date: str


class ProgressTracker:
    """Manages progress tracking CSV file."""

    def __init__(self, csv_path: str = "data/manual_review/progress_tracker.csv"):
        """
        Initialize progress tracker.

        Args:
            csv_path: Path to progress tracker CSV file

        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Progress tracker not found: {self.csv_path}")

    def load_all_races(self) -> List[RaceReviewStatus]:
        """
        Load all races from CSV.

        Returns:
            List of RaceReviewStatus objects
        """
        races = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                race = RaceReviewStatus(
                    priority=int(row['Priority']),
                    race_id=row['Race_ID'],
                    competition=row['Competition'],
                    detected_duration_s=float(row['Detected_Duration_s']),
                    duration_frames=int(row['Duration_Frames']),
                    confidence_start=float(row['Confidence_Start']),
                    confidence_finish=float(row['Confidence_Finish']),
                    status=row['Status'],
                    issue_description=row['Issue_Description'],
                    review_status=row['Review_Status'],
                    corrected_duration_s=row['Corrected_Duration_s'],
                    corrected_start_frame=row['Corrected_Start_Frame'],
                    corrected_finish_frame=row['Corrected_Finish_Frame'],
                    reviewer_notes=row['Reviewer_Notes'],
                    review_date=row['Review_Date']
                )
                races.append(race)
        return races

    def get_pending_races(self) -> List[RaceReviewStatus]:
        """
        Get only pending races.

        Returns:
            List of pending races
        """
        all_races = self.load_all_races()
        return [r for r in all_races if r.review_status == 'Pending']

    def get_completed_races(self) -> List[RaceReviewStatus]:
        """
        Get only completed races.

        Returns:
            List of completed races
        """
        all_races = self.load_all_races()
        return [r for r in all_races if r.review_status == 'Completed']

    def update_race(self, race: RaceReviewStatus) -> None:
        """
        Update a single race in CSV.

        Args:
            race: Updated RaceReviewStatus object
        """
        all_races = self.load_all_races()

        # Find and update
        for i, r in enumerate(all_races):
            if r.race_id == race.race_id:
                all_races[i] = race
                break

        # Write back to CSV
        self._write_all_races(all_races)

    def _write_all_races(self, races: List[RaceReviewStatus]) -> None:
        """
        Write all races to CSV.

        Args:
            races: List of all races to write
        """
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'Priority', 'Race_ID', 'Competition',
                'Detected_Duration_s', 'Duration_Frames',
                'Confidence_Start', 'Confidence_Finish',
                'Status', 'Issue_Description',
                'Review_Status', 'Corrected_Duration_s',
                'Corrected_Start_Frame', 'Corrected_Finish_Frame',
                'Reviewer_Notes', 'Review_Date'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for race in races:
                writer.writerow({
                    'Priority': race.priority,
                    'Race_ID': race.race_id,
                    'Competition': race.competition,
                    'Detected_Duration_s': race.detected_duration_s,
                    'Duration_Frames': race.duration_frames,
                    'Confidence_Start': race.confidence_start,
                    'Confidence_Finish': race.confidence_finish,
                    'Status': race.status,
                    'Issue_Description': race.issue_description,
                    'Review_Status': race.review_status,
                    'Corrected_Duration_s': race.corrected_duration_s,
                    'Corrected_Start_Frame': race.corrected_start_frame,
                    'Corrected_Finish_Frame': race.corrected_finish_frame,
                    'Reviewer_Notes': race.reviewer_notes,
                    'Review_Date': race.review_date
                })

    def get_statistics(self) -> Dict:
        """
        Get review statistics.

        Returns:
            Dictionary with statistics (total, completed, pending, etc.)
        """
        all_races = self.load_all_races()
        return {
            'total': len(all_races),
            'completed': sum(1 for r in all_races if r.review_status == 'Completed'),
            'pending': sum(1 for r in all_races if r.review_status == 'Pending'),
            'skipped': sum(1 for r in all_races if r.review_status == 'Skipped'),
            'critical': sum(1 for r in all_races if r.priority == 1),
            'high': sum(1 for r in all_races if r.priority == 2),
            'medium': sum(1 for r in all_races if r.priority == 3),
            'low': sum(1 for r in all_races if r.priority == 4)
        }

    def get_progress_percentage(self) -> float:
        """
        Get overall progress percentage.

        Returns:
            Progress percentage (0-100)
        """
        stats = self.get_statistics()
        if stats['total'] == 0:
            return 0.0
        return (stats['completed'] / stats['total']) * 100
