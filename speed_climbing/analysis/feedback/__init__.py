"""
Fuzzy Logic Feedback System for Speed Climbing Analysis.

Provides personalized feedback based on extracted features.
"""

from .fuzzy_engine import FuzzyFeedbackEngine
from .feedback_generator import FeedbackGenerator
from .baseline import BaselineStatistics

__all__ = [
    'FuzzyFeedbackEngine',
    'FeedbackGenerator',
    'BaselineStatistics',
]
