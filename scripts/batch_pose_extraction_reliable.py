"""
Batch Pose Extraction - Reliable Races Only
============================================
Extract BlazePose keypoints for 114 verified races.

Usage:
    python scripts/batch_pose_extraction_reliable.py

Output:
    data/processed/poses/<race_id>_pose.json (114 files)

Estimated Time: 3-4 hours (depends on hardware)
"""

import json
import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys


class BatchPoseExtractor:
    """Extract poses for reliable races in batch."""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Highest quality
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_race_poses(self, video_path: Path, metadata_path: Path) -> dict:
        """
        Extract poses for a single race.

        Args:
            video_path: Path to race video
            metadata_path: Path to race metadata JSON

        Returns:
            dict with frames, poses, quality metrics
        """
        # Load metadata for frame range
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Frame numbers in original video
        detected_start_frame = metadata.get('detected_start_frame', 0)
        detected_finish_frame = metadata.get('detected_finish_frame', 10000)
        fps = metadata.get('fps', 30.0)
        buffer_before = metadata.get('buffer_before', 1.5)

        # Calculate buffer in frames
        buffer_before_frames = int(buffer_before * fps)

        # Calculate where segment starts in original video
        segment_start_in_original = detected_start_frame - buffer_before_frames

        # Convert to segment-relative frame numbers
        start_frame = detected_start_frame - segment_start_in_original
        finish_frame = detected_finish_frame - segment_start_in_original

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Clamp finish frame to video bounds
        finish_frame = min(finish_frame, total_frames_in_video - 1)

        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_data = []
        missing_frames = 0

        for frame_idx in range(start_frame, finish_frame + 1):
            ret, frame = cap.read()
            if not ret:
                missing_frames += 1
                continue

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with BlazePose
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract landmarks
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })

                frames_data.append({
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / fps,
                    'landmarks': landmarks
                })
            else:
                missing_frames += 1

        cap.release()

        # Quality metrics
        total_frames = finish_frame - start_frame + 1
        success_rate = (total_frames - missing_frames) / total_frames * 100 if total_frames > 0 else 0

        return {
            'race_id': metadata.get('race_id', video_path.stem),
            'competition': video_path.parent.name,
            'frames': frames_data,
            'total_frames': total_frames,
            'extracted_frames': len(frames_data),
            'missing_frames': missing_frames,
            'success_rate': success_rate,
            'extraction_date': datetime.now().isoformat(),
            'start_frame': start_frame,
            'finish_frame': finish_frame,
            'fps': fps
        }

    def process_all_reliable_races(self, reliable_races_file: Path):
        """Process all reliable races."""
        # Load reliable races list
        with open(reliable_races_file, 'r', encoding='utf-8') as f:
            reliable_data = json.load(f)

        race_ids = reliable_data['reliable_race_ids']

        print("="*70)
        print(f"Batch Pose Extraction - Reliable Races")
        print("="*70)
        print(f"Total races to process: {len(race_ids)}")
        print(f"Estimated time: 3-4 hours (depends on hardware)")
        print(f"Output directory: data/processed/poses/")
        print("="*70 + "\n")

        success_count = 0
        failed_races = []
        quality_stats = []

        for race_id in tqdm(race_ids, desc="Extracting poses"):
            try:
                # Find video and metadata
                video_path, metadata_path = self._find_race_files(race_id)

                # Extract poses
                pose_data = self.extract_race_poses(video_path, metadata_path)

                # Save
                output_path = Path(f"data/processed/poses/{race_id}_pose.json")
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(pose_data, f, indent=2)

                # Track quality
                quality_stats.append({
                    'race_id': race_id,
                    'success_rate': pose_data['success_rate'],
                    'extracted_frames': pose_data['extracted_frames'],
                    'total_frames': pose_data['total_frames']
                })

                success_count += 1

            except Exception as e:
                tqdm.write(f"Failed for {race_id}: {e}")
                failed_races.append({'race_id': race_id, 'error': str(e)})

        # Calculate statistics
        avg_success_rate = sum(s['success_rate'] for s in quality_stats) / len(quality_stats) if quality_stats else 0
        total_frames_extracted = sum(s['extracted_frames'] for s in quality_stats)

        # Generate report
        report = {
            'total_races': len(race_ids),
            'successful': success_count,
            'failed': len(failed_races),
            'failed_races': failed_races,
            'quality_metrics': {
                'average_success_rate': avg_success_rate,
                'total_frames_extracted': total_frames_extracted,
                'races_with_high_quality': sum(1 for s in quality_stats if s['success_rate'] >= 95)
            },
            'quality_stats': quality_stats,
            'completion_date': datetime.now().isoformat()
        }

        report_path = Path('data/processed/pose_extraction_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print("\n" + "="*70)
        print("✅ Pose extraction complete!")
        print("="*70)
        print(f"Successful: {success_count}/{len(race_ids)}")
        print(f"Failed: {len(failed_races)}")
        print(f"Average success rate: {avg_success_rate:.1f}%")
        print(f"Total frames extracted: {total_frames_extracted:,}")
        print(f"Races with >95% quality: {report['quality_metrics']['races_with_high_quality']}")
        print(f"\nReport saved to: {report_path}")
        print("="*70)

        if failed_races:
            print("\n⚠️  Failed races:")
            for race in failed_races[:10]:  # Show first 10
                print(f"   - {race['race_id']}: {race['error']}")
            if len(failed_races) > 10:
                print(f"   ... and {len(failed_races) - 10} more")

    def _find_race_files(self, race_id: str):
        """Find video and metadata for a race."""
        # Search in all competition folders
        race_segments_dir = Path("data/race_segments")

        for comp_dir in race_segments_dir.iterdir():
            if not comp_dir.is_dir():
                continue

            video_path = comp_dir / f"{race_id}.mp4"
            metadata_path = comp_dir / f"{race_id}_metadata.json"

            if video_path.exists() and metadata_path.exists():
                return video_path, metadata_path

        raise FileNotFoundError(f"Files not found for {race_id}")

    def __del__(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


if __name__ == "__main__":
    extractor = BatchPoseExtractor()
    try:
        extractor.process_all_reliable_races(
            Path("data/processed/reliable_races_list.json")
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        print("Partial results have been saved to data/processed/poses/")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
