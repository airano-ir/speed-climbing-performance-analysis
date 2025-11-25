#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run New Pipeline
================
Runs the fully refactored speed_climbing pipeline on a video.
"""

import sys
import argparse
from pathlib import Path
import json
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.processing.pipeline import GlobalMapVideoProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("calibration_debug.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


def main():
    parser = argparse.ArgumentParser(description='Run Speed Climbing Pipeline')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--route-map', type=str, default='configs/ifsc_route_coordinates.json', help='Path to route map')
    parser.add_argument('--output', type=str, default='data/processed/pipeline_test', help='Output directory')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {video_path}...")
    
    # Initialize processor
    processor = GlobalMapVideoProcessor(route_map_path=args.route_map)
    
    # Metadata (simplified for test)
    metadata = {
        'detected_start_frame': 0,
        'detected_finish_frame': 300 # Limit frames for testing
    }
    
    # Process
    try:
        results = processor.process_race(str(video_path), metadata)
        
        # Save results
        output_file = output_dir / f"{video_path.stem}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"✅ Processing complete. Results saved to {output_file}")
        
        # Print summary
        for lane in ['left_climber', 'right_climber']:
            data = results[lane]['summary']
            print(f"\n{lane.upper()} Summary:")
            print(f"  Total Time: {data['total_time_s']:.2f} s")
            print(f"  Total Distance: {data['total_distance_m']:.2f} m")
            print(f"  Avg Velocity: {data['avg_velocity_m_s']:.2f} m/s")
            
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == '__main__':
    sys.exit(main())
