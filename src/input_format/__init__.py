"""Input Format Module for EEG Data Processing

This module handles the input format processing for EEG data, including:
- Data loading and validation
- Format conversion
- Data preprocessing
"""

from .data_loader import load_eeg_data, validate_data_format
from .format_converter import convert_to_dataframe, convert_to_mne
from .preprocessor import preprocess_data

__all__ = [
    "load_eeg_data",
    "validate_data_format",
    "convert_to_dataframe",
    "convert_to_mne",
    "preprocess_data",
]
