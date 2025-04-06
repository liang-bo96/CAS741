"""Data Processing Module for EEG Analysis

This module handles the core data processing functionality for EEG analysis, including:
- Feature extraction
- Signal processing
- Statistical analysis
- Data transformation
"""

from .statistical_analyzer import compute_statistics

__all__ = ["compute_statistics"]
