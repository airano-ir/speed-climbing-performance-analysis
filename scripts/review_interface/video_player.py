"""
Video Player Component
======================
Streamlit component for video playback with frame-by-frame navigation.

Provides interactive video player with:
- Frame slider
- Navigation buttons (±1, ±5, ±30 frames)
- Frame and time display
- Current frame image display
"""

import streamlit as st
import cv2
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


class VideoPlayer:
    """Video player with frame navigation for Streamlit."""

    def __init__(self, video_path: Path, fps: float):
        """
        Initialize video player.

        Args:
            video_path: Path to video file
            fps: Video FPS
        """
        self.video_path = video_path
        self.fps = fps
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.total_frames / fps if fps > 0 else 0

    def render(
        self,
        key_prefix: str = "video",
        language: str = "en"
    ) -> Tuple[int, float]:
        """
        Render video player with controls.

        Args:
            key_prefix: Unique prefix for widget keys
            language: "en" or "fa" for Persian/Farsi

        Returns:
            (current_frame, current_time_seconds)
        """
        # Language-specific labels
        labels = self._get_labels(language)

        st.subheader(labels['title'])

        # Display video info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(labels['total_frames'], self.total_frames)
        with col2:
            st.metric(labels['fps'], f"{self.fps:.1f}")
        with col3:
            st.metric(labels['duration'], f"{self.duration_seconds:.2f}s")

        # Initialize current frame from session state or default
        if f'{key_prefix}_current_frame' not in st.session_state:
            st.session_state[f'{key_prefix}_current_frame'] = 0

        current_frame = st.session_state[f'{key_prefix}_current_frame']

        # Frame slider
        current_frame = st.slider(
            labels['frame_label'],
            min_value=0,
            max_value=max(0, self.total_frames - 1),
            value=current_frame,
            step=1,
            key=f"{key_prefix}_frame_slider"
        )

        # Time display
        current_time = current_frame / self.fps if self.fps > 0 else 0
        st.text(f"{labels['time_label']}: {current_time:.3f}s ({labels['frame_text']} {current_frame})")

        # Navigation buttons
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            if st.button("⏮️ -30", key=f"{key_prefix}_minus30"):
                current_frame = max(0, current_frame - 30)
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        with col2:
            if st.button("⬅️ -5", key=f"{key_prefix}_minus5"):
                current_frame = max(0, current_frame - 5)
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        with col3:
            if st.button("◀️ -1", key=f"{key_prefix}_minus1"):
                current_frame = max(0, current_frame - 1)
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        with col4:
            if st.button("▶️ +1", key=f"{key_prefix}_plus1"):
                current_frame = min(self.total_frames - 1, current_frame + 1)
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        with col5:
            if st.button("➡️ +5", key=f"{key_prefix}_plus5"):
                current_frame = min(self.total_frames - 1, current_frame + 5)
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        with col6:
            if st.button("⏭️ +30", key=f"{key_prefix}_plus30"):
                current_frame = min(self.total_frames - 1, current_frame + 30)
                st.session_state[f'{key_prefix}_current_frame'] = current_frame
                st.rerun()

        # Display current frame
        frame_image = self._get_frame(current_frame)
        if frame_image is not None:
            st.image(frame_image, channels="BGR", use_container_width=True)
        else:
            st.error(f"{labels['error_loading']}: {current_frame}")

        # Store current frame in session state
        st.session_state[f'{key_prefix}_current_frame'] = current_frame

        return current_frame, current_time

    def _get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame from video.

        Args:
            frame_number: Frame index to retrieve

        Returns:
            Frame as numpy array or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def mark_frame_on_image(
        self,
        frame: np.ndarray,
        label: str,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw label on frame image.

        Args:
            frame: Frame image
            label: Text to draw
            color: BGR color tuple

        Returns:
            Frame with label drawn
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (50, 50), font, 1, color, 2, cv2.LINE_AA)
        return frame

    def close(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()

    def _get_labels(self, language: str) -> dict:
        """
        Get UI labels in specified language.

        Args:
            language: "en" or "fa"

        Returns:
            Dictionary of labels
        """
        if language == "fa":
            return {
                'title': '📹 پخش‌کننده ویدئو',
                'total_frames': 'مجموع فریم‌ها',
                'fps': 'فریم در ثانیه',
                'duration': 'مدت زمان',
                'frame_label': 'فریم',
                'time_label': 'زمان',
                'frame_text': 'فریم',
                'error_loading': 'خطا در بارگذاری فریم'
            }
        else:
            return {
                'title': '📹 Video Player',
                'total_frames': 'Total Frames',
                'fps': 'FPS',
                'duration': 'Duration',
                'frame_label': 'Frame',
                'time_label': 'Time',
                'frame_text': 'Frame',
                'error_loading': 'Failed to load frame'
            }


class ComparisonVideoPlayer:
    """Side-by-side video player for comparing two videos."""

    def __init__(self, video_path1: Path, video_path2: Path, fps: float):
        """
        Initialize comparison player.

        Args:
            video_path1: Path to first video
            video_path2: Path to second video
            fps: Video FPS
        """
        self.player1 = VideoPlayer(video_path1, fps)
        self.player2 = VideoPlayer(video_path2, fps)

    def render(self, key_prefix: str = "comparison", language: str = "en"):
        """
        Render side-by-side comparison.

        Args:
            key_prefix: Unique prefix for widget keys
            language: "en" or "fa"
        """
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Video 1")
            self.player1.render(f"{key_prefix}_vid1", language)

        with col2:
            st.markdown("### Video 2")
            self.player2.render(f"{key_prefix}_vid2", language)

    def close(self):
        """Release both video captures."""
        self.player1.close()
        self.player2.close()
