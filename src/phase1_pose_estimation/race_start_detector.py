"""
Race Start Detector - Detects the start of a speed climbing race

This module provides multiple methods to detect race start:
1. Audio-based: Detects starting beep sound using librosa + FFT analysis
2. Motion-based: Detects sudden climber movement using optical flow
3. Fusion: Combines both methods for high accuracy

Usage:
    # Audio-only
    detector = RaceStartDetector(method='audio')
    start_time = detector.detect_from_video('video.mp4')

    # Motion-only
    detector = RaceStartDetector(method='motion')
    start_time = detector.detect_from_video('video.mp4')

    # Fusion (recommended)
    detector = RaceStartDetector(method='fusion')
    start_time = detector.detect_from_video('video.mp4')

Author: Speed Climbing Analysis Project
Date: 2025-11-13
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import json

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: librosa not available. Audio-based detection disabled.")


@dataclass
class RaceStartResult:
    """Result of race start detection"""
    frame_id: int
    timestamp: float  # seconds
    confidence: float  # 0.0 to 1.0
    method: str  # 'audio', 'motion', or 'fusion'
    audio_confidence: Optional[float] = None
    motion_confidence: Optional[float] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'method': self.method,
            'audio_confidence': self.audio_confidence,
            'motion_confidence': self.motion_confidence,
            'metadata': self.metadata
        }


class AudioBeepDetector:
    """
    Detects starting beep in audio using FFT analysis.

    IFSC speed climbing uses a standard starting beep:
    - Typically 800-1200 Hz tone
    - Duration: ~0.5-1.0 seconds
    - High amplitude spike
    """

    def __init__(
        self,
        target_freq_range: Tuple[float, float] = (800, 1200),
        min_duration: float = 0.3,
        amplitude_threshold_percentile: float = 95.0
    ):
        """
        Args:
            target_freq_range: Frequency range of beep in Hz
            min_duration: Minimum beep duration in seconds
            amplitude_threshold_percentile: Percentile for amplitude threshold
        """
        if not AUDIO_AVAILABLE:
            raise ImportError("librosa is required for audio detection")

        self.target_freq_range = target_freq_range
        self.min_duration = min_duration
        self.amplitude_threshold_percentile = amplitude_threshold_percentile

    def detect_beep(self, audio_path: Path, sample_window_sec: float = 0.1) -> List[Tuple[float, float]]:
        """
        Detect starting beep in audio file.

        Args:
            audio_path: Path to audio file (WAV)
            sample_window_sec: Window size for analysis in seconds

        Returns:
            List of (timestamp, confidence) tuples for detected beeps
        """
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None)

        # Compute STFT (Short-Time Fourier Transform)
        hop_length = int(sr * sample_window_sec)
        n_fft = 2048

        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Find bins in target frequency range
        freq_mask = (freqs >= self.target_freq_range[0]) & (freqs <= self.target_freq_range[1])
        freq_indices = np.where(freq_mask)[0]

        # Energy in target frequency range
        target_energy = np.sum(magnitude[freq_indices, :], axis=0)

        # Normalize
        target_energy = target_energy / (np.max(target_energy) + 1e-8)

        # Threshold
        threshold = np.percentile(target_energy, self.amplitude_threshold_percentile)

        # Find peaks above threshold
        peaks = target_energy > threshold

        # Group consecutive peaks
        beep_candidates = []
        in_beep = False
        beep_start = 0

        for i, is_peak in enumerate(peaks):
            if is_peak and not in_beep:
                # Start of beep
                beep_start = i
                in_beep = True
            elif not is_peak and in_beep:
                # End of beep
                beep_end = i
                duration = (beep_end - beep_start) * sample_window_sec

                if duration >= self.min_duration:
                    # Valid beep
                    timestamp = beep_start * sample_window_sec
                    confidence = float(np.mean(target_energy[beep_start:beep_end]))
                    beep_candidates.append((timestamp, confidence))

                in_beep = False

        return beep_candidates

    def detect_start_from_audio(self, audio_path: Path, max_search_time: float = 30.0) -> Optional[Tuple[float, float]]:
        """
        Detect race start from audio file.

        Args:
            audio_path: Path to audio file
            max_search_time: Maximum time to search for beep (seconds)

        Returns:
            (timestamp, confidence) or None if no beep found
        """
        beeps = self.detect_beep(audio_path)

        if not beeps:
            return None

        # Filter beeps within search window
        beeps_in_window = [(t, c) for t, c in beeps if t <= max_search_time]

        if not beeps_in_window:
            return None

        # Return first beep (highest confidence)
        beeps_sorted = sorted(beeps_in_window, key=lambda x: x[1], reverse=True)
        return beeps_sorted[0]


class MotionStartDetector:
    """
    Detects race start by detecting sudden climber movement.

    Uses dense optical flow to detect when climbers suddenly start moving.
    """

    def __init__(
        self,
        motion_threshold: float = 5.0,
        min_motion_frames: int = 3,
        roi_bottom_fraction: float = 0.7
    ):
        """
        Args:
            motion_threshold: Threshold for motion magnitude
            min_motion_frames: Minimum consecutive frames with motion
            roi_bottom_fraction: Focus on bottom fraction of frame (where climbers start)
        """
        self.motion_threshold = motion_threshold
        self.min_motion_frames = min_motion_frames
        self.roi_bottom_fraction = roi_bottom_fraction

    def compute_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two frames.

        Returns:
            Motion magnitude array
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Compute magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        return magnitude

    def detect_start_from_video(
        self,
        video_path: Path,
        max_search_frames: int = 900,  # 30 seconds at 30fps
        skip_frames: int = 0
    ) -> Optional[Tuple[int, float, float]]:
        """
        Detect race start from video using motion analysis.

        Args:
            video_path: Path to video file
            max_search_frames: Maximum frames to search
            skip_frames: Number of frames to skip at start

        Returns:
            (frame_id, timestamp, confidence) or None
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        # Skip initial frames
        for _ in range(skip_frames):
            cap.read()

        # Get first frame
        ret, prev_frame = cap.read()
        if not ret:
            return None

        height, width = prev_frame.shape[:2]

        # ROI: bottom portion where climbers start
        roi_y_start = int(height * (1 - self.roi_bottom_fraction))

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = prev_gray[roi_y_start:, :]

        motion_history = []
        frame_id = skip_frames + 1

        while frame_id < skip_frames + max_search_frames:
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = curr_gray[roi_y_start:, :]

            # Compute motion
            magnitude = self.compute_optical_flow(prev_gray, curr_gray)

            # Average motion in ROI
            avg_motion = np.mean(magnitude)
            motion_history.append(avg_motion)

            # Check for sudden motion increase
            if len(motion_history) >= self.min_motion_frames:
                recent_motion = motion_history[-self.min_motion_frames:]

                if all(m > self.motion_threshold for m in recent_motion):
                    # Motion detected!
                    timestamp = frame_id / fps
                    confidence = float(np.mean(recent_motion) / (self.motion_threshold + 1e-8))
                    confidence = min(confidence, 1.0)

                    cap.release()
                    return (frame_id, timestamp, confidence)

            prev_gray = curr_gray
            frame_id += 1

        cap.release()
        return None


class RaceStartDetector:
    """
    Main race start detector with multiple methods.

    Supports:
    - Audio-based detection (beep sound)
    - Motion-based detection (sudden movement)
    - Fusion (combines both for high accuracy)
    """

    def __init__(
        self,
        method: str = 'fusion',
        audio_weight: float = 0.6,
        motion_weight: float = 0.4,
        max_time_diff: float = 2.0
    ):
        """
        Args:
            method: Detection method ('audio', 'motion', 'fusion')
            audio_weight: Weight for audio in fusion mode
            motion_weight: Weight for motion in fusion mode
            max_time_diff: Maximum time difference for fusion (seconds)
        """
        if method not in ['audio', 'motion', 'fusion']:
            raise ValueError(f"Invalid method: {method}")

        self.method = method
        self.audio_weight = audio_weight
        self.motion_weight = motion_weight
        self.max_time_diff = max_time_diff

        if method in ['audio', 'fusion']:
            if not AUDIO_AVAILABLE:
                raise ImportError("librosa required for audio/fusion methods")
            self.audio_detector = AudioBeepDetector()
        else:
            self.audio_detector = None

        if method in ['motion', 'fusion']:
            self.motion_detector = MotionStartDetector()
        else:
            self.motion_detector = None

    def detect_from_video(
        self,
        video_path: Path,
        audio_path: Optional[Path] = None
    ) -> Optional[RaceStartResult]:
        """
        Detect race start from video file.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file (if separate). If None, tries {video_path}.wav

        Returns:
            RaceStartResult or None if no start detected
        """
        video_path = Path(video_path)

        if audio_path is None:
            # Try common audio extensions
            audio_path = video_path.with_suffix('.wav')
            if not audio_path.exists():
                audio_path = video_path.with_suffix('.mp3')
            if not audio_path.exists():
                audio_path = None

        # Get video FPS
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        audio_result = None
        motion_result = None

        # Audio detection
        if self.method in ['audio', 'fusion'] and audio_path and audio_path.exists():
            audio_detection = self.audio_detector.detect_start_from_audio(audio_path)
            if audio_detection:
                audio_time, audio_conf = audio_detection
                audio_frame = int(audio_time * fps)
                audio_result = (audio_frame, audio_time, audio_conf)

        # Motion detection
        if self.method in ['motion', 'fusion']:
            motion_result = self.motion_detector.detect_start_from_video(video_path)

        # Combine results
        if self.method == 'audio':
            if audio_result is None:
                return None
            frame_id, timestamp, confidence = audio_result
            return RaceStartResult(
                frame_id=frame_id,
                timestamp=timestamp,
                confidence=confidence,
                method='audio',
                audio_confidence=confidence
            )

        elif self.method == 'motion':
            if motion_result is None:
                return None
            frame_id, timestamp, confidence = motion_result
            return RaceStartResult(
                frame_id=frame_id,
                timestamp=timestamp,
                confidence=confidence,
                method='motion',
                motion_confidence=confidence
            )

        elif self.method == 'fusion':
            if audio_result is None and motion_result is None:
                return None

            if audio_result and motion_result:
                # Both available - check agreement
                audio_frame, audio_time, audio_conf = audio_result
                motion_frame, motion_time, motion_conf = motion_result

                time_diff = abs(audio_time - motion_time)

                if time_diff <= self.max_time_diff:
                    # Agreement - use weighted average
                    fused_time = (audio_time * self.audio_weight + motion_time * self.motion_weight)
                    fused_frame = int(fused_time * fps)
                    fused_conf = (audio_conf * self.audio_weight + motion_conf * self.motion_weight)

                    return RaceStartResult(
                        frame_id=fused_frame,
                        timestamp=fused_time,
                        confidence=fused_conf,
                        method='fusion',
                        audio_confidence=audio_conf,
                        motion_confidence=motion_conf,
                        metadata={'time_diff': time_diff, 'agreement': True}
                    )
                else:
                    # Disagreement - use higher confidence
                    if audio_conf > motion_conf:
                        return RaceStartResult(
                            frame_id=audio_frame,
                            timestamp=audio_time,
                            confidence=audio_conf,
                            method='fusion',
                            audio_confidence=audio_conf,
                            motion_confidence=motion_conf,
                            metadata={'time_diff': time_diff, 'agreement': False, 'used': 'audio'}
                        )
                    else:
                        return RaceStartResult(
                            frame_id=motion_frame,
                            timestamp=motion_time,
                            confidence=motion_conf,
                            method='fusion',
                            audio_confidence=audio_conf,
                            motion_confidence=motion_conf,
                            metadata={'time_diff': time_diff, 'agreement': False, 'used': 'motion'}
                        )

            elif audio_result:
                # Only audio available
                frame_id, timestamp, confidence = audio_result
                return RaceStartResult(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    confidence=confidence,
                    method='fusion',
                    audio_confidence=confidence,
                    metadata={'available': 'audio_only'}
                )
            else:
                # Only motion available
                frame_id, timestamp, confidence = motion_result
                return RaceStartResult(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    confidence=confidence,
                    method='fusion',
                    motion_confidence=confidence,
                    metadata={'available': 'motion_only'}
                )

        return None


# CLI interface
if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Detect race start in speed climbing video')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--audio', type=str, help='Path to audio file (optional)')
    parser.add_argument('--method', type=str, default='fusion',
                       choices=['audio', 'motion', 'fusion'],
                       help='Detection method')
    parser.add_argument('--output', type=str, help='Output JSON file for results')

    args = parser.parse_args()

    print(f"Detecting race start in: {args.video}")
    print(f"Method: {args.method}")

    detector = RaceStartDetector(method=args.method)
    result = detector.detect_from_video(
        Path(args.video),
        Path(args.audio) if args.audio else None
    )

    if result:
        print(f"\nRace start detected!")
        print(f"  Frame: {result.frame_id}")
        print(f"  Time: {result.timestamp:.2f}s")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Method: {result.method}")
        if result.audio_confidence:
            print(f"  Audio confidence: {result.audio_confidence:.2f}")
        if result.motion_confidence:
            print(f"  Motion confidence: {result.motion_confidence:.2f}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        print("\nNo race start detected")
        sys.exit(1)
