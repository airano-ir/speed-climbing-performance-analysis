import re
from collections import Counter

log_file = "calibration_debug.log"
counts = Counter()

print("Scanning log file...")
try:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "Homography calibration" in line:
                counts['Homography'] += 1
            elif "Affine calibration" in line:
                counts['Affine'] += 1
            elif "Similarity calibration" in line:
                counts['Similarity'] += 1
            elif "Using cached calibration" in line:
                counts['Cached'] += 1
            elif "Calibration failed" in line:
                counts['Failed'] += 1
            
            if "Using cached calibration" in line and counts['Cached'] <= 5:
                print(f"Sample Cached: {line.strip()}")

    print("\nCalibration Method Counts:")
    for method, count in counts.items():
        print(f"{method}: {count}")
        
except Exception as e:
    print(f"Error: {e}")
