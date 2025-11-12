"""
Video Processor for Speed Climbing Analysis
===========================================

Handles video loading, frame extraction, and preprocessing for pose estimation.
Supports various video formats and frame rates (60-240 fps).

Classes:
    VideoProcessor: Main class for video processing operations
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Process video files for speed climbing analysis.

    Features:
        - Multi-format support (mp4, avi, mov)
        - Frame-by-frame extraction with metadata
        - Automatic FPS detection
        - Optional frame skipping for performance
        - Video properties extraction

    Attributes:
        video_path: Path to input video file
        cap: OpenCV VideoCapture object
        fps: Frames per second of the video
        total_frames: Total number of frames
        width: Frame width in pixels
        height: Frame height in pixels

    Example:
        >>> processor = VideoProcessor("athlete_001.mp4")
        >>> for frame_data in processor.extract_frames():
        ...     frame_id = frame_data['frame_id']
        ...     image = frame_data['frame']
        ...     print(f"Processing frame {frame_id}")
    """

    def __init__(
        self,
        video_path: str,
        target_fps: Optional[int] = None
    ):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file
            target_fps: Optional target FPS for downsampling
                       If None, uses original video FPS

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Extract video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # Frame skip calculation for downsampling
        self.target_fps = target_fps if target_fps else self.fps
        self.frame_skip = max(1, int(self.fps / self.target_fps))

        logger.info(f"Video loaded: {self.video_path.name}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps:.2f} → {self.target_fps:.2f}")
        logger.info(f"  Duration: {self.duration:.2f}s ({self.total_frames} frames)")
        logger.info(f"  Frame skip: {self.frame_skip}")

    def extract_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None
    ) -> Generator[Dict, None, None]:
        """
        Extract frames from video as a generator.

        Args:
            start_frame: Starting frame index (0-based)
            end_frame: Ending frame index (exclusive). None = all frames
            resize: Optional (width, height) tuple for resizing frames

        Yields:
            Dictionary containing:
                - frame_id: Frame index in original video
                - timestamp: Time in seconds
                - frame: Numpy array (BGR format)
                - frame_number: Sequential number in extracted sequence

        Example:
            >>> for data in processor.extract_frames(resize=(640, 480)):
            ...     cv2.imshow('Frame', data['frame'])
        """
        if end_frame is None:
            end_frame = self.total_frames

        # Validate range
        start_frame = max(0, start_frame)
        end_frame = min(self.total_frames, end_frame)

        # Set start position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        current_frame = start_frame

        while current_frame < end_frame:
            ret, frame = self.cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {current_frame}")
                break

            # Apply frame skipping
            if (current_frame - start_frame) % self.frame_skip != 0:
                current_frame += 1
                continue

            # Resize if requested
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)

            # Calculate timestamp
            timestamp = current_frame / self.fps

            yield {
                'frame_id': current_frame,
                'timestamp': timestamp,
                'frame': frame,
                'frame_number': frame_count
            }

            frame_count += 1
            current_frame += 1

    def get_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """
        Extract a single frame at a specific timestamp.

        Args:
            time_seconds: Time in seconds

        Returns:
            Frame as numpy array, or None if time is out of bounds

        Example:
            >>> frame = processor.get_frame_at_time(2.5)  # Frame at 2.5 seconds
        """
        if time_seconds < 0 or time_seconds > self.duration:
            logger.warning(f"Time {time_seconds}s out of bounds [0, {self.duration}s]")
            return None

        frame_index = int(time_seconds * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame = self.cap.read()
        return frame if ret else None

    def get_properties(self) -> Dict:
        """
        Get video properties as a dictionary.

        Returns:
            Dictionary with video metadata
        """
        return {
            'filename': self.video_path.name,
            'path': str(self.video_path),
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'duration_seconds': self.duration,
            'target_fps': self.target_fps,
            'frame_skip': self.frame_skip
        }

    def save_frame(
        self,
        frame: np.ndarray,
        output_path: str,
        quality: int = 95
    ) -> bool:
        """
        Save a single frame as an image file.

        Args:
            frame: Frame to save (numpy array)
            output_path: Output file path (supports .jpg, .png)
            quality: JPEG quality (0-100) or PNG compression (0-9)

        Returns:
            True if successful, False otherwise
        """
        try:
            if output_path.endswith('.jpg') or output_path.endswith('.jpeg'):
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif output_path.endswith('.png'):
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, quality])
            else:
                cv2.imwrite(output_path, frame)

            logger.info(f"Frame saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return False

    def create_video_writer(
        self,
        output_path: str,
        fps: Optional[float] = None,
        codec: str = 'mp4v'
    ) -> cv2.VideoWriter:
        """
        Create a VideoWriter for saving processed video.

        Args:
            output_path: Output video file path
            fps: Output FPS (default: uses target_fps)
            codec: FourCC codec code (default: 'mp4v')

        Returns:
            OpenCV VideoWriter object

        Example:
            >>> writer = processor.create_video_writer("output.mp4")
            >>> for data in processor.extract_frames():
            ...     processed = some_processing(data['frame'])
            ...     writer.write(processed)
            >>> writer.release()
        """
        fps = fps if fps else self.target_fps
        fourcc = cv2.VideoWriter_fourcc(*codec)

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (self.width, self.height)
        )

        if not writer.isOpened():
            raise ValueError(f"Failed to create VideoWriter: {output_path}")

        logger.info(f"VideoWriter created: {output_path} ({fps} fps)")
        return writer

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resources on exit."""
        self.release()

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            logger.info("Video resources released")

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.release()


# ==================== Utility Functions ====================

def get_video_info(video_path: str) -> Dict:
    """
    Quick function to get video information without creating processor.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video properties

    Example:
        >>> info = get_video_info("athlete_001.mp4")
        >>> print(f"Duration: {info['duration_seconds']}s")
    """
    with VideoProcessor(video_path) as processor:
        return processor.get_properties()


def extract_thumbnail(
    video_path: str,
    output_path: str,
    time_seconds: float = 1.0
) -> bool:
    """
    Extract a thumbnail from video at specified time.

    Args:
        video_path: Input video path
        output_path: Output image path
        time_seconds: Time to extract thumbnail (default: 1.0s)

    Returns:
        True if successful

    Example:
        >>> extract_thumbnail("athlete.mp4", "thumbnail.jpg", time_seconds=2.5)
    """
    try:
        with VideoProcessor(video_path) as processor:
            frame = processor.get_frame_at_time(time_seconds)
            if frame is not None:
                return processor.save_frame(frame, output_path)
        return False
    except Exception as e:
        logger.error(f"Thumbnail extraction failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage and testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_processor.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    # Test video processor
    with VideoProcessor(video_path, target_fps=30) as processor:
        print("\n=== Video Properties ===")
        props = processor.get_properties()
        for key, value in props.items():
            print(f"{key:20s}: {value}")

        print("\n=== Extracting first 10 frames ===")
        for i, frame_data in enumerate(processor.extract_frames()):
            if i >= 10:
                break
            print(f"Frame {frame_data['frame_id']}: "
                  f"t={frame_data['timestamp']:.3f}s, "
                  f"shape={frame_data['frame'].shape}")
