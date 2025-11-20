# MASTER CONTEXT: Speed Climbing Performance Analysis

## Project Overview
**Goal**: Build an automated video analysis system for speed climbing that tracks athletes, maps their position to the official IFSC wall (15m), and calculates performance metrics (velocity, splits, etc.).

**Current Status**: **Refactoring Complete & Testing (Phase 2.6)**
All core components have been migrated to the `speed_climbing` package. The full pipeline has been verified on a test video.

## Architecture (New)
The project is being migrated to a domain-driven structure:

### `speed_climbing` Package
*   **`core`**: Configuration and standards.
    *   `settings.py`: IFSC standards, default config.
*   **`vision`**: Computer Vision components.
    *   `holds.py`: `HoldDetector` (HSV-based).
    *   `lanes.py`: `DualLaneDetector` (Boundary detection).
    *   `pose.py`: `BlazePoseExtractor` (MediaPipe wrapper).
    *   `calibration.py`: `CameraCalibrator`, `PeriodicCalibrator` (Homography).
*   **`processing`**: Pipeline logic.
    *   `tracking.py`: `WorldCoordinateTracker` (Pixel -> Meter).
    *   `dropout.py`: `DropoutHandler` (Error recovery).
    *   `pipeline.py`: `GlobalMapVideoProcessor` (Orchestrator).
*   **`analysis`**: Data aggregation.
    *   `time_series.py`: `TimeSeriesBuilder`.

## Key Workflows
1.  **Global Map Registration**:
    *   Detect red holds.
    *   Match to IFSC route map (`configs/ifsc_route_coordinates.json`).
    *   Compute Homography (RANSAC).
    *   Transform Athlete COM (Center of Mass) to Wall Coordinates (Meters).

## Recent Updates
*   **2025-11-20**: 
    *   Completed migration of `TimeSeriesBuilder`, `DropoutHandler`, `WorldCoordinateTracker`, and `GlobalMapVideoProcessor`.
    *   Verified full pipeline with `scripts/run_new_pipeline.py`.
    *   Successful end-to-end test on `race001` (14.39m detected distance).

## Known Issues / Focus Areas
*   **Hold Detection**: Sensitivity to lighting/angle. (Mitigation: Tuning HSV, Area filters).
*   **Calibration**: Needs robust RANSAC to handle partial wall visibility.
*   **Physical Validation**: Initial prototype showed some height discrepancies (Physical Validation: FAIL in prototype), but full pipeline produced reasonable total distance. Needs fine-tuning.

## Next Steps
1.  Clean up legacy code in `src/`.
2.  Implement "Calibration Quality" visualization.
3.  Run batch processing on more races to validate robustness.
