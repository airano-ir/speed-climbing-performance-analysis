import cv2
import numpy as np
import argparse
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed_climbing.processing.tracking import WorldCoordinateTracker
from speed_climbing.core.settings import IFSC_STANDARDS

def visualize_calibration(video_path, frame_number=None, output_path="debug_calibration.jpg"):
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if frame_number is None:
        frame_number = total_frames // 2
    
    print(f"Analyzing frame {frame_number}/{total_frames}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return

    # Initialize tracker and pose extractor
    route_map_path = "configs/ifsc_route_coordinates.json"
    tracker = WorldCoordinateTracker(route_map_path)
    from speed_climbing.vision.pose import BlazePoseExtractor
    pose_extractor = BlazePoseExtractor()
    
    # We need to manually detect holds to visualize them, as process_frame doesn't return them
    # But process_frame does the calibration.
    # Let's do what process_frame does but keep the data.
    
    lane = 'left' # Default to left lane for visualization
    
    # 0. Detect Pose and Create Mask
    print("Detecting pose for masking...")
    pose_result = pose_extractor.extract_pose(frame)
    
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    if pose_result and pose_result.has_detection:
        print("Pose detected, creating mask...")
        # Collect all keypoints
        points = []
        h, w = frame.shape[:2]
        for kp in pose_result.keypoints.values():
            if kp.confidence > 0.3:
                points.append((int(kp.x * w), int(kp.y * h)))
        
        if points:
            # Create a convex hull around the climber
            hull = cv2.convexHull(np.array(points))
            # Dilate the hull to cover loose clothing/limbs
            cv2.fillConvexPoly(mask, hull, 0)
            # Dilate the black area (erode the white mask) to expand the exclusion zone
            kernel = np.ones((20, 20), np.uint8) # 20px margin
            mask = cv2.erode(mask, kernel, iterations=1)
            
    # Apply mask to frame for detection only
    masked_frame = frame.copy()
    masked_frame[mask == 0] = [0, 0, 0] # Black out the climber
    
    # 1. Detect Holds (on masked frame)
    print("Detecting holds...")
    detected_holds = tracker.hold_detector.detect_holds(masked_frame, lane=lane)
    print(f"Detected {len(detected_holds)} holds in {lane} lane.")
    
    # 2. Calibrate
    print("Calibrating...")
    
    # DEBUG: Check matching directly
    pixel_points, meter_points, matched_holds = tracker.calibrator._match_holds_to_route(
        detected_holds, 
        [h for h in tracker.calibrator.route_map['holds'] if h['panel'].startswith('SN' if lane == 'left' else 'DX')],
        frame.shape
    )
    print(f"DEBUG: Matched {len(pixel_points)} holds before RANSAC.")
    
    calibration = tracker.calibrator.calibrate_frame(frame, frame_number, detected_holds, lane=lane, force_recalibration=True)
    
    # Visualization
    vis_frame = frame.copy()
    
    # Draw Detected Holds (Red)
    for hold in detected_holds:
        cv2.circle(vis_frame, (int(hold.pixel_x), int(hold.pixel_y)), 5, (0, 0, 255), 2)
        # cv2.putText(vis_frame, f"{hold.area:.0f}", (int(hold.pixel_x)+5, int(hold.pixel_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

    if calibration:
        print(f"Calibration Successful! Confidence: {calibration.confidence:.2f}, RMSE: {calibration.rmse_error:.2f}")
        print(f"Homography:\n{calibration.homography_matrix}")
        
        # Draw Projected Route Map Holds (Green)
        route_map = tracker.calibrator.route_map
        lane_prefix = 'SN' if lane == 'left' else 'DX'
        
        for hold in route_map['holds']:
            if hold['panel'].startswith(lane_prefix):
                mx, my = hold['wall_x_m'], hold['wall_y_m']
                try:
                    px, py = calibration.meter_to_pixel_func(mx, my)
                    # Check for reasonable bounds before drawing
                    if -10000 < px < 10000 and -10000 < py < 10000:
                        cv2.circle(vis_frame, (int(px), int(py)), 4, (0, 255, 0), 2)
                except (OverflowError, ValueError):
                    continue
        
        # Draw Wall Boundaries
        corners_m = [
            (0, 0), (3, 0), (3, 15), (0, 15)
        ]
        corners_px = []
        for mx, my in corners_m:
            try:
                px, py = calibration.meter_to_pixel_func(mx, my)
                if -10000 < px < 10000 and -10000 < py < 10000:
                    corners_px.append((int(px), int(py)))
            except (OverflowError, ValueError):
                pass
        
        # Draw boundary lines
        if len(corners_px) == 4:
            for i in range(4):
                p1 = corners_px[i]
                p2 = corners_px[(i+1)%4]
                cv2.line(vis_frame, p1, p2, (255, 255, 0), 2)
            
        cv2.putText(vis_frame, f"Calibration OK (Conf: {calibration.confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("Calibration Failed!")
        cv2.putText(vis_frame, "Calibration FAILED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(output_path, vis_frame)
    print(f"Saved visualization to {output_path}")
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Calibration Visualization")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, default=None, help="Frame number to analyze (default: middle)")
    parser.add_argument("--output", type=str, default="debug_calibration.jpg", help="Output image path")
    
    args = parser.parse_args()
    visualize_calibration(args.video, args.frame, args.output)
