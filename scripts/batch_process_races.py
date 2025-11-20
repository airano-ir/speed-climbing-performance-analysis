#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Process Races
===================
Runs the speed_climbing pipeline on all videos in a directory.
Generates individual JSON results and a summary CSV.
"""

import sys
import argparse
from pathlib import Path
import json
import csv
import traceback
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.processing.pipeline import GlobalMapVideoProcessor

def main():
    parser = argparse.ArgumentParser(description='Batch Process Speed Climbing Videos')
    parser.add_argument('--input-dir', type=str, default='data/raw', help='Directory containing video files')
    parser.add_argument('--output-dir', type=str, default='data/processed/batch_results', help='Output directory for results')
    parser.add_argument('--route-map', type=str, default='configs/ifsc_route_coordinates.json', help='Path to route map')
    parser.add_argument('--extensions', nargs='+', default=['.mp4', '.mov', '.avi', '.mkv'], help='Video file extensions to look for')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find videos
    videos = []
    for ext in args.extensions:
        videos.extend(list(input_dir.glob(f"*{ext}")))
    
    if not videos:
        print(f"No videos found in {input_dir} with extensions {args.extensions}")
        return
        
    print(f"Found {len(videos)} videos to process.")
    
    # Initialize processor
    try:
        processor = GlobalMapVideoProcessor(route_map_path=args.route_map)
    except Exception as e:
        print(f"Failed to initialize processor: {e}")
        return 1

    summary_results = []
    
    for i, video_path in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] Processing {video_path.name}...")
        start_time = time.time()
        
        try:
            # TODO: In a real scenario, we might want to detect start/finish frames automatically
            # or load them from a sidecar file. For now, we process the whole video 
            # or rely on the pipeline's internal logic if it has auto-trimming (it doesn't yet fully).
            # We will pass empty metadata and let the processor handle it (or process full duration).
            metadata = {} 
            
            results = processor.process_race(str(video_path), metadata)
            
            # Save detailed JSON
            json_path = output_dir / f"{video_path.stem}_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            # Collect summary stats
            row = {
                'video': video_path.name,
                'status': 'success',
                'left_dist': results['left_climber']['summary']['total_distance_m'],
                'left_vel': results['left_climber']['summary']['avg_velocity_m_s'],
                'right_dist': results['right_climber']['summary']['total_distance_m'],
                'right_vel': results['right_climber']['summary']['avg_velocity_m_s'],
                'processing_time': round(time.time() - start_time, 2)
            }
            summary_results.append(row)
            print(f"✅ Success. Time: {row['processing_time']}s")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            # traceback.print_exc()
            summary_results.append({
                'video': video_path.name,
                'status': 'failed',
                'error': str(e),
                'processing_time': round(time.time() - start_time, 2)
            })
            
    # Save summary CSV
    csv_path = output_dir / 'batch_summary.csv'
    fieldnames = ['video', 'status', 'left_dist', 'left_vel', 'right_dist', 'right_vel', 'processing_time', 'error']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_results:
            # Fill missing keys with empty string
            full_row = {k: row.get(k, '') for k in fieldnames}
            writer.writerow(full_row)
            
    print(f"\nBatch processing complete. Summary saved to {csv_path}")

if __name__ == '__main__':
    main()
