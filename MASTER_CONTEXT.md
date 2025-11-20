# MASTER CONTEXT: Speed Climbing Performance Analysis

## Project Overview
**Goal**: Build an automated video analysis system for speed climbing that tracks athletes, maps their position to the official IFSC wall (15m), and calculates performance metrics (velocity, splits, etc.).

**Current Status**: **Refactoring & Migration (Phase 2.5)**
We are currently restructuring the project into a modular `speed_climbing` package to improve maintainability and testability.

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
*   **2025-11-20**: Started major refactoring. Created `speed_climbing` package. Migrated Vision and Processing components. Verified `HoldDetector` with user images.

## Known Issues / Focus Areas
*   **Hold Detection**: Sensitivity to lighting/angle. (Mitigation: Tuning HSV, Area filters).
*   **Calibration**: Needs robust RANSAC to handle partial wall visibility.
*   **Testing**: Need to run full pipeline test on video segments.

## Next Steps
1.  Complete migration of entry point scripts.
2.  Run full pipeline test on `race001` or `race003`.
3.  Implement "Calibration Quality" visualization.
