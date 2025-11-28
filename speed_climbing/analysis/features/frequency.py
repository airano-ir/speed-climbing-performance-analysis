"""
Frequency analysis for movement patterns.

Extracts rhythmic features using FFT analysis on keypoint time series.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy import signal
from scipy.fft import fft, fftfreq

from .base import extract_keypoint_series, interpolate_missing


class FrequencyAnalyzer:
    """
    Analyze movement frequency patterns from pose data.

    Extracts:
    - hand_frequency_hz: Dominant hand movement frequency
    - foot_frequency_hz: Dominant foot movement frequency
    - limb_sync_ratio: Coordination between hands and feet
    - movement_regularity: Consistency of rhythm
    """

    def __init__(self, fps: float = 30.0, min_frames: int = 30):
        """
        Args:
            fps: Video frame rate
            min_frames: Minimum frames required for analysis
        """
        self.fps = fps
        self.min_frames = min_frames

    def analyze(self, frames: List[Dict[str, Any]], lane: str = 'left') -> Dict[str, float]:
        """
        Extract frequency features from pose frames.

        Args:
            frames: List of frame dictionaries from pose JSON
            lane: 'left' or 'right' climber

        Returns:
            Dictionary of frequency features
        """
        # Extract keypoint series
        lw_x, lw_y, lw_valid = extract_keypoint_series(frames, 'left_wrist', lane)
        rw_x, rw_y, rw_valid = extract_keypoint_series(frames, 'right_wrist', lane)
        la_x, la_y, la_valid = extract_keypoint_series(frames, 'left_ankle', lane)
        ra_x, ra_y, ra_valid = extract_keypoint_series(frames, 'right_ankle', lane)

        # Check if we have enough valid data
        total_valid = sum([np.sum(lw_valid), np.sum(rw_valid), np.sum(la_valid), np.sum(ra_valid)])
        if total_valid < self.min_frames * 4 * 0.5:  # At least 50% valid
            return self._default_features()

        # Combine hand movements (use Y-coordinate for vertical motion)
        hand_y = self._combine_signals(lw_y, rw_y)
        foot_y = self._combine_signals(la_y, ra_y)

        # Interpolate missing values
        hand_y = interpolate_missing(hand_y)
        foot_y = interpolate_missing(foot_y)

        # Calculate features
        features = {
            'hand_frequency_hz': self._dominant_frequency(hand_y),
            'foot_frequency_hz': self._dominant_frequency(foot_y),
            'limb_sync_ratio': self._sync_ratio(hand_y, foot_y),
            'movement_regularity': self._regularity(hand_y),
            'hand_movement_amplitude': self._movement_amplitude(hand_y),
            'foot_movement_amplitude': self._movement_amplitude(foot_y),
        }

        return features

    def _combine_signals(self, sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
        """Combine two signals, preferring non-NaN values."""
        result = np.full_like(sig1, np.nan)

        for i in range(len(sig1)):
            if not np.isnan(sig1[i]) and not np.isnan(sig2[i]):
                # Both available - use average
                result[i] = (sig1[i] + sig2[i]) / 2
            elif not np.isnan(sig1[i]):
                result[i] = sig1[i]
            elif not np.isnan(sig2[i]):
                result[i] = sig2[i]

        return result

    def _dominant_frequency(self, y: np.ndarray) -> float:
        """
        Find dominant frequency using FFT.

        Returns frequency in Hz.
        """
        # Remove NaN and detrend
        valid = ~np.isnan(y)
        if np.sum(valid) < self.min_frames:
            return 0.0

        y_valid = y[valid].copy()

        # Detrend (remove linear trend)
        y_detrended = signal.detrend(y_valid)

        # Apply window to reduce spectral leakage
        window = signal.windows.hann(len(y_detrended))
        y_windowed = y_detrended * window

        # FFT
        n = len(y_windowed)
        yf = np.abs(fft(y_windowed))[:n//2]
        xf = fftfreq(n, 1/self.fps)[:n//2]

        # Find peak in reasonable frequency range (0.5 - 5 Hz for climbing)
        freq_mask = (xf >= 0.5) & (xf <= 5.0)

        if not np.any(freq_mask):
            return 0.0

        yf_masked = yf[freq_mask]
        xf_masked = xf[freq_mask]

        if len(yf_masked) == 0:
            return 0.0

        peak_idx = np.argmax(yf_masked)
        return float(xf_masked[peak_idx])

    def _sync_ratio(self, hand_y: np.ndarray, foot_y: np.ndarray) -> float:
        """
        Calculate synchronization ratio between hands and feet.

        Uses cross-correlation. Returns 0-1 (1 = perfectly synchronized).
        """
        # Get valid overlapping data
        valid = ~(np.isnan(hand_y) | np.isnan(foot_y))
        if np.sum(valid) < self.min_frames:
            return 0.0

        h = hand_y[valid]
        f = foot_y[valid]

        # Normalize
        h = (h - np.mean(h)) / (np.std(h) + 1e-10)
        f = (f - np.mean(f)) / (np.std(f) + 1e-10)

        # Cross-correlation (normalized)
        correlation = np.correlate(h, f, mode='full')
        max_corr = np.max(np.abs(correlation)) / len(h)

        # Clamp to 0-1
        return float(np.clip(max_corr, 0, 1))

    def _regularity(self, y: np.ndarray) -> float:
        """
        Calculate movement regularity using spectral entropy.

        Lower entropy = more regular (periodic) movement.
        Returns 0-1 (1 = most regular).
        """
        valid = ~np.isnan(y)
        if np.sum(valid) < self.min_frames:
            return 0.0

        y_valid = y[valid]
        y_detrended = signal.detrend(y_valid)

        # Power spectral density
        f, psd = signal.welch(y_detrended, fs=self.fps, nperseg=min(64, len(y_valid)//2))

        # Normalize to probability distribution
        psd_norm = psd / (np.sum(psd) + 1e-10)

        # Spectral entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

        # Max entropy for uniform distribution
        max_entropy = np.log2(len(psd))

        # Regularity = 1 - normalized entropy
        regularity = 1 - (entropy / max_entropy) if max_entropy > 0 else 0

        return float(np.clip(regularity, 0, 1))

    def _movement_amplitude(self, y: np.ndarray) -> float:
        """
        Calculate average movement amplitude.

        Returns standard deviation of detrended signal.
        """
        valid = ~np.isnan(y)
        if np.sum(valid) < self.min_frames:
            return 0.0

        y_valid = y[valid]
        y_detrended = signal.detrend(y_valid)

        return float(np.std(y_detrended))

    def _default_features(self) -> Dict[str, float]:
        """Return default features when analysis is not possible."""
        return {
            'hand_frequency_hz': 0.0,
            'foot_frequency_hz': 0.0,
            'limb_sync_ratio': 0.0,
            'movement_regularity': 0.0,
            'hand_movement_amplitude': 0.0,
            'foot_movement_amplitude': 0.0,
        }
