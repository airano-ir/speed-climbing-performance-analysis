import sys
import json
from pathlib import Path
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.processing.pipeline import GlobalMapVideoProcessor

def test_pipeline(video_path, route_map_path, output_path):
    print(f"Testing pipeline on: {video_path}")
    
    processor = GlobalMapVideoProcessor(str(route_map_path))
    
    # Metadata (mocking start/end frames for test)
    # In real usage, these would come from a race detector or user input
    metadata = {
        'detected_start_frame': 0,
        'detected_finish_frame': 100 # Process first 100 frames for quick test
    }
    
    try:
        results = processor.process_race(video_path, metadata)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Success! Results saved to {output_path}")
        
        # Print summary
        left = results['left_climber']['summary']
        right = results['right_climber']['summary']
        print("\nLeft Climber Summary:")
        print(f"  Distance: {left['total_distance_m']:.2f}m")
        print(f"  Avg Velocity: {left['avg_velocity_m_s']:.2f}m/s")
        
        print("\nRight Climber Summary:")
        print(f"  Distance: {right['total_distance_m']:.2f}m")
        print(f"  Avg Velocity: {right['avg_velocity_m_s']:.2f}m/s")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use one of the existing videos
    video_file = Path("data/race_segments/seoul_2024/Speed_finals_Seoul_2024_race001.mp4")
    route_map = Path("configs/ifsc_route_coordinates.json")
    output = Path("data/processed/test_pipeline_result.json")
    
    if not video_file.exists():
        print(f"Video file not found: {video_file}")
        sys.exit(1)

    test_pipeline(video_file, route_map, output)
