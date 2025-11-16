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

        # ═══════════════════════════════════════════════════════════
        # ⭐ ENHANCED UX: Fully Synchronized Frame Navigation
        # All controls (slider, text input, buttons) stay in perfect sync
        # ═══════════════════════════════════════════════════════════

        # Initialize current frame from session state or default
        if f'{key_prefix}_current_frame' not in st.session_state:
            st.session_state[f'{key_prefix}_current_frame'] = 0

        # Helper function to update frame and ensure sync
        def update_frame(new_frame: int):
            """Update frame and clamp to valid range."""
            clamped = max(0, min(self.total_frames - 1, new_frame))
            st.session_state[f'{key_prefix}_current_frame'] = clamped
            return clamped

        st.markdown(f"**{labels['frame_navigation']}**")

        # Create columns for slider and text input
        col_slider, col_text = st.columns([3, 1])

        with col_slider:
            # Frame slider - synced with session state
            slider_frame = st.slider(
                labels['frame_label'],
                min_value=0,
                max_value=max(0, self.total_frames - 1),
                value=st.session_state[f'{key_prefix}_current_frame'],
                step=1,
                key=f"{key_prefix}_frame_slider_widget",
                label_visibility="collapsed",
                help=labels.get('slider_help', 'Drag to navigate through frames')
            )

        with col_text:
            # ⭐ Direct text input for frame number - synced with session state
            text_input_frame = st.number_input(
                labels['frame_input_label'],
                min_value=0,
                max_value=max(0, self.total_frames - 1),
                value=st.session_state[f'{key_prefix}_current_frame'],
                step=1,
                key=f"{key_prefix}_frame_text_widget",
                help=labels['frame_input_help'],
                label_visibility="collapsed"
            )

        # Detect which input changed and update session state
        current_frame = st.session_state[f'{key_prefix}_current_frame']

        # Check if slider changed
        if slider_frame != current_frame:
            current_frame = update_frame(slider_frame)

        # Check if text input changed (takes priority over slider)
        if text_input_frame != current_frame:
            current_frame = update_frame(text_input_frame)

        # Time display with enhanced formatting and progress bar
        current_time = current_frame / self.fps if self.fps > 0 else 0
        progress_percentage = (current_frame / max(1, self.total_frames - 1)) * 100

        col_info, col_progress = st.columns([1, 2])

        with col_info:
            st.caption(
                f"⏱️ {labels['time_label']}: **{current_time:.3f}s** | "
                f"🎞️ {labels['frame_text']}: **{current_frame:,}** / **{self.total_frames - 1:,}**"
            )

        with col_progress:
            st.progress(
                progress_percentage / 100,
                text=f"📊 {progress_percentage:.1f}% through video"
            )

        # Navigation buttons with improved UX
        st.markdown(f"**{labels.get('quick_navigation', 'Quick Navigation')}**")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        # Helper for button navigation
        def navigate(delta: int):
            """Navigate by delta frames and trigger rerun for instant update."""
            new_frame = update_frame(current_frame + delta)
            st.rerun()

        with col1:
            if st.button("⏮️ Start", key=f"{key_prefix}_start", help="Jump to first frame (0)", use_container_width=True):
                update_frame(0)
                st.rerun()

        with col2:
            if st.button("⏪ -30", key=f"{key_prefix}_minus30", help="Go back 30 frames", use_container_width=True):
                navigate(-30)

        with col3:
            if st.button("◀️ -1", key=f"{key_prefix}_minus1", help="Previous frame", use_container_width=True):
                navigate(-1)

        with col4:
            if st.button("▶️ +1", key=f"{key_prefix}_plus1", help="Next frame", use_container_width=True):
                navigate(+1)

        with col5:
            if st.button("⏩ +30", key=f"{key_prefix}_plus30", help="Go forward 30 frames", use_container_width=True):
                navigate(+30)

        with col6:
            if st.button("⏭️ End", key=f"{key_prefix}_end", help=f"Jump to last frame ({self.total_frames - 1})", use_container_width=True):
                update_frame(self.total_frames - 1)
                st.rerun()

        with col7:
            # Custom jump input
            jump_frames = st.number_input(
                labels.get('custom_jump', 'Jump'),
                min_value=-self.total_frames,
                max_value=self.total_frames,
                value=0,
                step=10,
                key=f"{key_prefix}_custom_jump",
                help="Enter number of frames to jump (negative to go back)",
                label_visibility="collapsed"
            )
            if jump_frames != 0 and st.button("🎯", key=f"{key_prefix}_do_jump", help="Apply custom jump", use_container_width=True):
                navigate(jump_frames)

        # ═══════════════════════════════════════════════════════════
        # ⭐ Advanced Features: Bookmarks & Quick Access
        # ═══════════════════════════════════════════════════════════

        # Initialize bookmarks in session state
        if f'{key_prefix}_bookmarks' not in st.session_state:
            st.session_state[f'{key_prefix}_bookmarks'] = []

        with st.expander(f"🔖 {labels.get('bookmarks', 'Bookmarks')} ({len(st.session_state[f'{key_prefix}_bookmarks'])})", expanded=False):
            col_add, col_list = st.columns([1, 3])

            with col_add:
                if st.button(f"➕ {labels.get('add_bookmark', 'Add Current Frame')}", key=f"{key_prefix}_add_bookmark", use_container_width=True):
                    bookmarks = st.session_state[f'{key_prefix}_bookmarks']
                    if current_frame not in bookmarks:
                        bookmarks.append(current_frame)
                        bookmarks.sort()
                        st.success(f"Added frame {current_frame}")
                        st.rerun()
                    else:
                        st.info(f"Frame {current_frame} already bookmarked")

            with col_list:
                if st.session_state[f'{key_prefix}_bookmarks']:
                    st.caption(labels.get('saved_frames', 'Saved frames') + ": " + ", ".join([str(f) for f in st.session_state[f'{key_prefix}_bookmarks']]))

                    # Quick jump to bookmark
                    selected_bookmark = st.selectbox(
                        labels.get('jump_to_bookmark', 'Jump to bookmark'),
                        st.session_state[f'{key_prefix}_bookmarks'],
                        key=f"{key_prefix}_bookmark_select",
                        format_func=lambda x: f"Frame {x} ({x/self.fps:.2f}s)"
                    )

                    col_jump, col_clear = st.columns(2)
                    with col_jump:
                        if st.button("🎯 Jump", key=f"{key_prefix}_jump_bookmark", use_container_width=True):
                            update_frame(selected_bookmark)
                            st.rerun()

                    with col_clear:
                        if st.button("🗑️ Clear All", key=f"{key_prefix}_clear_bookmarks", use_container_width=True):
                            st.session_state[f'{key_prefix}_bookmarks'] = []
                            st.rerun()
                else:
                    st.info(labels.get('no_bookmarks', 'No bookmarks yet. Add current frame to create one.'))

        # Display current frame with overlay info
        frame_image = self._get_frame(current_frame)
        if frame_image is not None:
            # Add bookmark indicator if this frame is bookmarked
            if current_frame in st.session_state[f'{key_prefix}_bookmarks']:
                frame_image = self.mark_frame_on_image(
                    frame_image.copy(),
                    f"BOOKMARKED - Frame {current_frame}",
                    (0, 255, 255)  # Yellow
                )

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
                'frame_navigation': '🎯 انتخاب فریم',
                'frame_label': 'فریم',
                'frame_input_label': 'شماره فریم',
                'frame_input_help': 'مستقیم شماره فریم را وارد کنید - همزمان با اسلایدر sync می‌شود',
                'slider_help': 'اسلایدر را بکشید - همزمان با شماره فریم sync می‌شود',
                'quick_navigation': '⚡ ناوبری سریع',
                'custom_jump': 'پرش',
                'bookmarks': 'نشانک‌ها',
                'add_bookmark': 'افزودن فریم فعلی',
                'saved_frames': 'فریم‌های ذخیره شده',
                'jump_to_bookmark': 'پرش به نشانک',
                'no_bookmarks': 'هنوز نشانکی وجود ندارد. فریم فعلی را اضافه کنید.',
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
                'frame_navigation': '🎯 Frame Selection',
                'frame_label': 'Frame',
                'frame_input_label': 'Frame #',
                'frame_input_help': 'Type frame number directly - syncs instantly with slider',
                'slider_help': 'Drag slider - syncs instantly with frame number',
                'quick_navigation': '⚡ Quick Navigation',
                'custom_jump': 'Jump',
                'bookmarks': 'Bookmarks',
                'add_bookmark': 'Add Current Frame',
                'saved_frames': 'Saved frames',
                'jump_to_bookmark': 'Jump to bookmark',
                'no_bookmarks': 'No bookmarks yet. Add current frame to create one.',
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
