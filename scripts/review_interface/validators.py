"""
Validators
==========
Validation functions for race corrections.

Provides validation rules for race duration, frame order, and other
quality checks to ensure corrected data is valid.
"""

from typing import Dict, Tuple


class RaceValidator:
    """Validates race corrections."""

    def __init__(
        self,
        min_duration: float = 4.5,
        max_duration: float = 15.0,
        world_record_men: float = 5.00,
        world_record_women: float = 6.53
    ):
        """
        Initialize validator with thresholds.

        Args:
            min_duration: Minimum acceptable duration (seconds)
            max_duration: Maximum acceptable duration (seconds)
            world_record_men: Men's world record for reference
            world_record_women: Women's world record for reference
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.world_record_men = world_record_men
        self.world_record_women = world_record_women

    def validate_duration(self, duration_seconds: float) -> Tuple[bool, str, str]:
        """
        Validate race duration.

        Args:
            duration_seconds: Race duration in seconds

        Returns:
            Tuple of (is_valid, message, severity)
            severity: "critical", "warning", "info", or "success"
        """
        if duration_seconds < 0:
            return False, f"Negative duration: {duration_seconds:.2f}s (INVALID!)", "critical"

        if duration_seconds < 3.0:
            return False, f"Too short: {duration_seconds:.2f}s < 3s (impossible!)", "critical"

        if duration_seconds < self.min_duration:
            return False, f"Below minimum: {duration_seconds:.2f}s < {self.min_duration}s (faster than world record)", "warning"

        if duration_seconds > self.max_duration:
            return False, f"Above maximum: {duration_seconds:.2f}s > {self.max_duration}s (unusually slow or includes non-race footage)", "warning"

        if duration_seconds > 20.0:
            return False, f"Too long: {duration_seconds:.2f}s > 20s (likely includes pre/post-race)", "critical"

        # Valid duration
        return True, f"Valid duration: {duration_seconds:.2f}s", "success"

    def validate_frame_order(self, start_frame: int, finish_frame: int) -> Tuple[bool, str, str]:
        """
        Validate that finish_frame > start_frame.

        Args:
            start_frame: Start frame number
            finish_frame: Finish frame number

        Returns:
            Tuple of (is_valid, message, severity)
        """
        if finish_frame <= start_frame:
            return False, f"Finish frame ({finish_frame}) must be after start frame ({start_frame})", "critical"

        frame_diff = finish_frame - start_frame
        if frame_diff < 10:
            return False, f"Too few frames: {frame_diff} frames (likely detection error)", "warning"

        return True, f"Frame order valid: {frame_diff} frames", "success"

    def validate_frame_bounds(
        self,
        start_frame: int,
        finish_frame: int,
        total_frames: int
    ) -> Tuple[bool, str, str]:
        """
        Validate that frames are within video bounds.

        Args:
            start_frame: Start frame number
            finish_frame: Finish frame number
            total_frames: Total frames in video

        Returns:
            Tuple of (is_valid, message, severity)
        """
        if start_frame < 0:
            return False, f"Start frame ({start_frame}) cannot be negative", "critical"

        if finish_frame >= total_frames:
            return False, f"Finish frame ({finish_frame}) exceeds total frames ({total_frames})", "critical"

        if start_frame >= total_frames:
            return False, f"Start frame ({start_frame}) exceeds total frames ({total_frames})", "critical"

        return True, "Frame bounds valid", "success"

    def validate_all(
        self,
        start_frame: int,
        finish_frame: int,
        fps: float,
        total_frames: int = None
    ) -> Dict[str, Tuple[bool, str, str]]:
        """
        Run all validations.

        Args:
            start_frame: Start frame number
            finish_frame: Finish frame number
            fps: Video FPS
            total_frames: Total frames in video (optional)

        Returns:
            Dictionary of {validator_name: (is_valid, message, severity)}
        """
        results = {}

        # Frame order
        results['frame_order'] = self.validate_frame_order(start_frame, finish_frame)

        # Duration
        duration = (finish_frame - start_frame) / fps
        results['duration'] = self.validate_duration(duration)

        # Frame bounds (if total_frames provided)
        if total_frames is not None:
            results['frame_bounds'] = self.validate_frame_bounds(
                start_frame, finish_frame, total_frames
            )

        return results

    def is_all_valid(self, validation_results: Dict[str, Tuple[bool, str, str]]) -> bool:
        """
        Check if all validations passed.

        Args:
            validation_results: Results from validate_all()

        Returns:
            True if all validations passed, False otherwise
        """
        return all(result[0] for result in validation_results.values())

    def get_critical_errors(self, validation_results: Dict[str, Tuple[bool, str, str]]) -> list:
        """
        Get list of critical errors.

        Args:
            validation_results: Results from validate_all()

        Returns:
            List of critical error messages
        """
        return [
            result[1] for result in validation_results.values()
            if not result[0] and result[2] == "critical"
        ]

    def get_warnings(self, validation_results: Dict[str, Tuple[bool, str, str]]) -> list:
        """
        Get list of warnings.

        Args:
            validation_results: Results from validate_all()

        Returns:
            List of warning messages
        """
        return [
            result[1] for result in validation_results.values()
            if not result[0] and result[2] == "warning"
        ]
