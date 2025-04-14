"""Data Processing Module for EEG Analysis

This module handles the core data processing functionality for EEG analysis, including:
- Feature extraction
- Signal processing
- Statistical analysis
- Data transformation
"""

from .data_processing import compute_statistics, validate_input

__all__ = ["compute_statistics","validate_input",]
