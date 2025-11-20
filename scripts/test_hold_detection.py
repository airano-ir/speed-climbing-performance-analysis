import cv2
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from speed_climbing.vision.holds import HoldDetector

def test_image(image_path, output_path):
    print(f"Testing image: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        print("Failed to load image")
        return

    detector = HoldDetector()
    holds = detector.detect_holds(frame)
    print(f"Detected {len(holds)} holds")
    
    # Visualize
    annotated = detector.visualize_detections(frame, holds)
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    test_image(Path("data/test_images/test_image_0.png"), Path("data/test_images/result_0.png"))
    test_image(Path("data/test_images/test_image_1.png"), Path("data/test_images/result_1.png"))
