"""
YouTube Video Downloader Utility
================================

Downloads speed climbing videos from YouTube using yt-dlp.
Designed for IFSC World Cup and Olympic competition videos.

Usage:
    from src.utils.youtube_downloader import download_ifsc_video

    download_ifsc_video(
        url="https://youtube.com/watch?v=...",
        output_dir="data/raw_videos",
        quality="720p"
    )

Author: Speed Climbing Research Team
Date: 2025-11-12
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import yt_dlp


class IFSCVideoDownloader:
    """
    Downloads and manages IFSC speed climbing videos from YouTube.

    Features:
    - Automatic quality selection (720p/1080p)
    - Metadata extraction (duration, FPS, resolution)
    - Dual-lane race detection from title/description
    - Progress callback support
    """

    def __init__(self, output_dir: str = "data/raw_videos"):
        """
        Initialize downloader.

        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        url: str,
        quality: str = "720p",
        filename: Optional[str] = None,
        extract_audio: bool = False
    ) -> Dict[str, Any]:
        """
        Download a video from YouTube.

        Args:
            url: YouTube video URL
            quality: Desired quality ("720p", "1080p", "best")
            filename: Optional custom filename (without extension)
            extract_audio: If True, also extract audio as WAV

        Returns:
            Dictionary with download info:
            {
                'video_path': 'path/to/video.mp4',
                'audio_path': 'path/to/audio.wav',  # if extract_audio=True
                'metadata': {...}
            }
        """
        # Quality mapping
        quality_map = {
            "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "best": "bestvideo+bestaudio/best"
        }
        format_selector = quality_map.get(quality, quality_map["720p"])

        # Determine output filename
        if filename:
            output_template = str(self.output_dir / f"{filename}.%(ext)s")
        else:
            # Use video ID as filename
            output_template = str(self.output_dir / "%(id)s.%(ext)s")

        # yt-dlp options
        ydl_opts = {
            'format': format_selector,
            'outtmpl': output_template,
            'merge_output_format': 'mp4',
            'writeinfojson': True,  # Save metadata
            'progress_hooks': [self._progress_hook],
            'quiet': False,
            'no_warnings': False,
        }

        # Extract audio if requested
        if extract_audio:
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }]
            # Keep the video file after extracting audio
            ydl_opts['keepvideo'] = True

        print(f"[*] Downloading from: {url}")
        print(f"[*] Output directory: {self.output_dir}")
        print(f"[*] Quality: {quality}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info
            info = ydl.extract_info(url, download=True)

            # Get actual filename
            video_id = info['id']
            if filename:
                video_path = self.output_dir / f"{filename}.mp4"
                audio_path = self.output_dir / f"{filename}.wav" if extract_audio else None
                metadata_path = self.output_dir / f"{filename}.info.json"
            else:
                video_path = self.output_dir / f"{video_id}.mp4"
                audio_path = self.output_dir / f"{video_id}.wav" if extract_audio else None
                metadata_path = self.output_dir / f"{video_id}.info.json"

            # Extract metadata
            metadata = self._extract_metadata(info)

            # Save custom metadata
            with open(self.output_dir / f"{video_path.stem}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            result = {
                'video_path': str(video_path),
                'metadata': metadata,
                'youtube_id': video_id
            }

            if extract_audio and audio_path and audio_path.exists():
                result['audio_path'] = str(audio_path)

            print(f"[OK] Download complete: {video_path.name}")
            print(f"     Duration: {metadata['duration']:.1f}s")
            print(f"     Resolution: {metadata['width']}x{metadata['height']}")
            print(f"     FPS: {metadata['fps']}")

            return result

    def _progress_hook(self, d: Dict[str, Any]):
        """Progress callback for yt-dlp."""
        if d['status'] == 'downloading':
            # Extract progress info
            percent = d.get('_percent_str', 'N/A')
            speed = d.get('_speed_str', 'N/A')
            eta = d.get('_eta_str', 'N/A')
            print(f"  Progress: {percent} | Speed: {speed} | ETA: {eta}", end='\r')
        elif d['status'] == 'finished':
            print("\n  [+] Download finished, processing...")

    def _extract_metadata(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant metadata from yt-dlp info.

        Args:
            info: Full info dict from yt-dlp

        Returns:
            Cleaned metadata dict
        """
        # Detect if this is a dual-lane race
        title = info.get('title', '').lower()
        description = info.get('description', '').lower()

        is_dual_race = any(keyword in title or keyword in description
                          for keyword in ['final', 'semi-final', 'qualification',
                                         'race', 'vs', 'versus'])

        metadata = {
            'title': info.get('title', 'Unknown'),
            'youtube_id': info.get('id', 'Unknown'),
            'duration': info.get('duration', 0),
            'fps': info.get('fps', 30),
            'width': info.get('width', 0),
            'height': info.get('height', 0),
            'upload_date': info.get('upload_date', 'Unknown'),
            'uploader': info.get('uploader', 'Unknown'),
            'description': info.get('description', '')[:500],  # First 500 chars
            'is_dual_race': is_dual_race,
            'thumbnail': info.get('thumbnail', None),
        }

        return metadata

    def download_batch(self, urls: list[str], **kwargs) -> list[Dict[str, Any]]:
        """
        Download multiple videos.

        Args:
            urls: List of YouTube URLs
            **kwargs: Arguments to pass to download()

        Returns:
            List of download results
        """
        results = []

        for i, url in enumerate(urls, 1):
            print(f"\n{'='*60}")
            print(f"Video {i}/{len(urls)}")
            print(f"{'='*60}")

            try:
                result = self.download(url, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"❌ Error downloading {url}: {e}")
                results.append({'error': str(e), 'url': url})

        return results


# Convenience function
def download_ifsc_video(
    url: str,
    output_dir: str = "data/raw_videos",
    quality: str = "720p",
    filename: Optional[str] = None,
    extract_audio: bool = False
) -> Dict[str, Any]:
    """
    Download a single IFSC video from YouTube.

    Args:
        url: YouTube video URL
        output_dir: Directory to save video
        quality: Video quality ("720p", "1080p", "best")
        filename: Optional custom filename
        extract_audio: If True, also extract audio as WAV

    Returns:
        Download result dictionary

    Example:
        >>> result = download_ifsc_video(
        ...     url="https://youtube.com/watch?v=abc123",
        ...     quality="720p",
        ...     extract_audio=True
        ... )
        >>> print(result['video_path'])
        'data/raw_videos/abc123.mp4'
    """
    downloader = IFSCVideoDownloader(output_dir)
    return downloader.download(url, quality, filename, extract_audio)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python youtube_downloader.py <youtube_url> [quality] [filename]")
        print("\nExample:")
        print("  python youtube_downloader.py https://youtube.com/watch?v=abc123 720p ifsc_final_2024")
        sys.exit(1)

    url = sys.argv[1]
    quality = sys.argv[2] if len(sys.argv) > 2 else "720p"
    filename = sys.argv[3] if len(sys.argv) > 3 else None

    result = download_ifsc_video(url, quality=quality, filename=filename, extract_audio=True)
    print(f"\n✅ Success! Video saved to: {result['video_path']}")
