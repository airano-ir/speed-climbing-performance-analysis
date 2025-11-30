#!/usr/bin/env python3
"""
Speed Climbing Performance Analysis - Sample Data Downloader
دانلود داده‌های نمونه برای تحلیل عملکرد صعود سرعتی

This script downloads sample data for testing and demonstrations.
این اسکریپت داده‌های نمونه برای تست و نمایش را دانلود می‌کند.

Usage / استفاده:
    python scripts/download_sample_data.py                    # Download pose data
    python scripts/download_sample_data.py --include-video    # Also download videos
    python scripts/download_sample_data.py --list-races       # List available data

Note: Video files are optional and larger.
نکته: فایل‌های ویدیو اختیاری و حجیم‌تر هستند.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

# GitHub Release URL for sample data
GITHUB_REPO = "airano-ir/speed-climbing-performance-analysis"
GITHUB_RELEASE_BASE = f"https://github.com/{GITHUB_REPO}/releases/download"
SAMPLE_VERSION = "v0.1.0"

# Competition data - poses and videos
COMPETITIONS = {
    "seoul_2024": {
        "poses_url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/poses_seoul_2024.zip",
        "videos_url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/videos_seoul_2024.zip",
        "poses_size_mb": 32,
        "videos_size_mb": 120,
        "description": "Seoul 2024 IFSC World Cup",
        "races": 31
    },
    "chamonix_2024": {
        "poses_url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/poses_chamonix_2024.zip",
        "videos_url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/videos_chamonix_2024.zip",
        "poses_size_mb": 27,
        "videos_size_mb": 125,
        "description": "Chamonix 2024 IFSC World Cup",
        "races": 32
    },
    "innsbruck_2024": {
        "poses_url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/poses_innsbruck_2024.zip",
        "videos_url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/videos_innsbruck_2024.zip",
        "poses_size_mb": 42,
        "videos_size_mb": 125,
        "description": "Innsbruck 2024 IFSC World Cup",
        "races": 32
    },
    "villars_2024": {
        "poses_url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/poses_villars_2024.zip",
        "videos_url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/videos_villars_2024.zip",
        "poses_size_mb": 21,
        "videos_size_mb": 95,
        "description": "Villars 2024 IFSC World Cup",
        "races": 24
    },
    "zilina_2025": {
        "poses_url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/poses_zilina_2025.zip",
        "videos_url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/videos_zilina_2025.zip",
        "poses_size_mb": 102,
        "videos_size_mb": 270,
        "description": "Zilina 2025 IFSC Event",
        "races": 69
    }
}

# Sample video for quick demo
SAMPLE_VIDEO = {
    "url": f"{GITHUB_RELEASE_BASE}/{SAMPLE_VERSION}/sample_videos.zip",
    "size_mb": 15,
    "description": "Sample race videos for quick testing"
}


def download_file(url: str, dest_path: Path, expected_sha256: str = None) -> bool:
    """
    Download a file with progress indicator.

    Args:
        url: URL to download from
        dest_path: Local path to save file
        expected_sha256: Optional SHA256 hash to verify

    Returns:
        True if download successful, False otherwise
    """
    try:
        print(f"Downloading: {dest_path.name}")
        print(f"  From: {url}")

        # Download with progress
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 // total_size)
                print(f"\r  Progress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, reporthook)
        print()  # New line after progress

        # Verify hash if provided
        if expected_sha256:
            with open(dest_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash != expected_sha256:
                print(f"  WARNING: Hash mismatch!")
                print(f"    Expected: {expected_sha256}")
                print(f"    Got: {file_hash}")
                return False

        print(f"  Done: {dest_path}")
        return True

    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  ERROR: File not found (404)")
            print(f"  The release may not have been published yet.")
            print(f"  Use --offline to create sample files locally.")
        else:
            print(f"  ERROR: HTTP {e.code} - {e.reason}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip file to destination directory."""
    try:
        print(f"  Extracting to: {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  Extraction complete")
        # Optionally remove the zip file after extraction
        # zip_path.unlink()
        return True
    except Exception as e:
        print(f"  ERROR extracting: {e}")
        return False


def list_available_data():
    """List available data for download."""
    print("\nAvailable Competition Data:")
    print("=" * 70)

    total_poses_size = 0
    total_videos_size = 0
    total_races = 0

    for name, info in COMPETITIONS.items():
        print(f"\n  {name}:")
        print(f"    {info['description']} - {info['races']} races")
        print(f"    Poses:  ~{info['poses_size_mb']} MB")
        print(f"    Videos: ~{info['videos_size_mb']} MB")
        total_poses_size += info['poses_size_mb']
        total_videos_size += info['videos_size_mb']
        total_races += info['races']

    print()
    print("=" * 70)
    print(f"Total: {total_races} races")
    print(f"  Poses only:  ~{total_poses_size} MB")
    print(f"  With videos: ~{total_poses_size + total_videos_size} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Download sample data for Speed Climbing Performance Analysis"
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Also download video files (larger download)"
    )
    parser.add_argument(
        "--competition",
        choices=list(COMPETITIONS.keys()),
        action="append",
        help="Download specific competition (can be used multiple times)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all competitions"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available data"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data",
        help="Base directory for downloads (default: data)"
    )

    args = parser.parse_args()

    # List data and exit
    if args.list:
        list_available_data()
        return 0

    print("=" * 70)
    print("Speed Climbing Performance Analysis - Data Downloader")
    print("تحلیل عملکرد صعود سرعتی - دانلود داده‌ها")
    print("=" * 70)
    print()

    # Determine which competitions to download
    if args.all:
        competitions_to_download = list(COMPETITIONS.keys())
    elif args.competition:
        competitions_to_download = args.competition
    else:
        # Default: download sample videos only
        print("Downloading sample videos for quick testing...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        zip_path = output_dir / "sample_videos.zip"
        if download_file(SAMPLE_VIDEO['url'], zip_path):
            extract_zip(zip_path, output_dir / "samples")
            print("\nDone! Sample videos downloaded to data/samples/")
            print("\nFor full dataset, use:")
            print("  python scripts/download_sample_data.py --all")
            print("  python scripts/download_sample_data.py --all --include-video")
        return 0

    # Create output directories
    output_dir = Path(args.output_dir)
    poses_dir = output_dir / "processed" / "poses"
    videos_dir = output_dir / "race_segments"
    poses_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for comp_name in competitions_to_download:
        if comp_name not in COMPETITIONS:
            print(f"  Unknown competition: {comp_name}")
            continue

        info = COMPETITIONS[comp_name]
        print()
        print(f"Competition: {comp_name}")
        print(f"  {info['description']} - {info['races']} races")

        # Download poses
        poses_zip = poses_dir / f"poses_{comp_name}.zip"
        print(f"\n  Downloading poses (~{info['poses_size_mb']} MB)...")
        if download_file(info['poses_url'], poses_zip):
            if extract_zip(poses_zip, poses_dir):
                poses_zip.unlink()  # Remove zip after extraction
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1

        # Download videos if requested
        if args.include_video:
            videos_zip = videos_dir / f"videos_{comp_name}.zip"
            print(f"\n  Downloading videos (~{info['videos_size_mb']} MB)...")
            if download_file(info['videos_url'], videos_zip):
                if extract_zip(videos_zip, videos_dir):
                    videos_zip.unlink()  # Remove zip after extraction
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1

    # Summary
    print()
    print("=" * 70)
    print("Download complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print()
    print("Data locations:")
    print(f"  Poses:  {poses_dir.absolute()}")
    if args.include_video:
        print(f"  Videos: {videos_dir.absolute()}")
    print()

    if fail_count > 0:
        print("Note: Some downloads failed. You can:")
        print("  1. Run again to retry")
        print("  2. Download manually from GitHub releases:")
        print(f"     https://github.com/{GITHUB_REPO}/releases")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
