"""
Processing modules for Speed Climbing Performance Analysis.
"""

from speed_climbing.processing.tracking import WorldCoordinateTracker
from speed_climbing.processing.dropout import DropoutHandler
from speed_climbing.processing.pipeline import GlobalMapVideoProcessor
from speed_climbing.processing.athlete_centric import (
    AthleteCentricPipeline,
    RaceSegmentDetector,
    WallReferenceEstimator,
    RelativeMotionTracker,
    RacePhase,
    LaneState,
    FrameResult,
)

__all__ = [
    # Legacy pipeline
    'WorldCoordinateTracker',
    'DropoutHandler',
    'GlobalMapVideoProcessor',

    # New athlete-centric pipeline
    'AthleteCentricPipeline',
    'RaceSegmentDetector',
    'WallReferenceEstimator',
    'RelativeMotionTracker',
    'RacePhase',
    'LaneState',
    'FrameResult',
]
