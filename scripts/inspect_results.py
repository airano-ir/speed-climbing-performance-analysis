import json
import sys

try:
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    print("Top level keys:", list(data.keys()))
    for lane in ['left_climber', 'right_climber']:
        if lane in data:
            print(f"\n{lane} keys:", list(data[lane].keys()))
            if 'summary' in data[lane]:
                print(f"{lane} summary:", data[lane]['summary'])
            
            # Check first frame for calibration info
            if 'frame_data' in data[lane] and len(data[lane]['frame_data']) > 0:
                # Find a frame with calibration info (it might not be in the first one if detection failed)
                for i, frame in enumerate(data[lane]['frame_data'][:100]): # Check first 100 frames
                    if 'calibration' in frame:
                        print(f"{lane} frame {i} calibration:", frame['calibration'])
                        break
                else:
                    print(f"{lane}: No calibration info found in first 100 frames")
                    
except Exception as e:
    print(f"Error: {e}")
