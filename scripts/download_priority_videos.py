"""
Download Priority IFSC Videos
==============================

Downloads the high-priority videos from YouTube for testing dual-climber analysis.

Usage:
    python scripts/download_priority_videos.py

Author: Speed Climbing Research Team
Date: 2025-11-12
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.youtube_downloader import IFSCVideoDownloader
import yaml


def main():
    """Download all priority videos from config."""

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "youtube_urls.yaml"

    print("=" * 70)
    print("IFSC Speed Climbing Video Downloader")
    print("=" * 70)
    print(f"\n[*] Loading config from: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Get priority videos
    priority_videos = config.get('priority', [])

    if not priority_videos:
        print("[ERROR] No priority videos found in config!")
        return

    print(f"\n[OK] Found {len(priority_videos)} priority videos to download\n")

    # Initialize downloader
    downloader = IFSCVideoDownloader(output_dir="data/raw_videos")

    # Download each video
    results = []

    for i, video in enumerate(priority_videos, 1):
        print(f"\n{'='*70}")
        print(f"[VIDEO {i}/{len(priority_videos)}]")
        print(f"{'='*70}")
        print(f"Title: {video['title']}")
        print(f"URL: {video['url']}")
        print(f"Description: {video['description']}")
        print(f"Dual Race: {'Yes' if video.get('dual_race') else 'No'}")

        # Create safe filename from title
        safe_filename = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_'
                               for c in video['title'])
        safe_filename = safe_filename.replace(' ', '_').lower()

        try:
            # Download video (with audio for start beep detection)
            result = downloader.download(
                url=video['url'],
                quality="720p",
                filename=safe_filename,
                extract_audio=True  # Extract audio for start beep detection
            )

            results.append({
                'title': video['title'],
                'success': True,
                'video_path': result['video_path'],
                'audio_path': result.get('audio_path', 'N/A'),
                'metadata': result['metadata']
            })

            print(f"\n[SUCCESS] Successfully downloaded: {safe_filename}")

        except Exception as e:
            print(f"\n[ERROR] Error downloading video: {e}")
            results.append({
                'title': video['title'],
                'success': False,
                'error': str(e)
            })

    # Summary
    print(f"\n\n{'='*70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*70}")

    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\n[OK] Successful: {successful}/{len(results)}")
    print(f"[ERROR] Failed: {failed}/{len(results)}")

    if successful > 0:
        print(f"\n[*] Videos saved to: data/raw_videos/")
        print("\nSuccessfully downloaded videos:")
        for r in results:
            if r['success']:
                print(f"  [+] {r['title']}")
                print(f"      Video: {Path(r['video_path']).name}")
                if r.get('audio_path') != 'N/A':
                    print(f"      Audio: {Path(r['audio_path']).name}")
                print(f"      Duration: {r['metadata']['duration']:.1f}s")
                print(f"      Resolution: {r['metadata']['width']}x{r['metadata']['height']}")
                print(f"      FPS: {r['metadata']['fps']}")
                print()

    if failed > 0:
        print("\n[WARNING] Failed downloads:")
        for r in results:
            if not r['success']:
                print(f"  [-] {r['title']}")
                print(f"      Error: {r['error']}")

    print(f"\n{'='*70}")
    print("Next steps:")
    print("  1. Verify downloaded videos play correctly")
    print("  2. Run dual-lane detection on videos")
    print("  3. Test pose estimation accuracy")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
