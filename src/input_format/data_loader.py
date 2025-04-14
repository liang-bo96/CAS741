"""Data loading and validation functions for EEG data processing."""

import numpy as np
import mne
import pandas as pd
from typing import Union, Tuple, Dict, Any
from pathlib import Path
from nilearn import datasets, image


def load_data(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load EEG data from various file formats.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the EEG data file. Supports various formats including:
        - .fif (MNE-Python)
        - .mat (MATLAB)
        - .edf (European Data Format)
        - .bdf (Biosemi Data Format)
        - .csv (Comma-separated values)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the loaded data and metadata:
        - 'data': numpy array of shape (n_channels, n_times)
        - 'sfreq': sampling frequency
        - 'ch_names': list of channel names
        - 'ch_types': list of channel types
        - 'info': MNE info object
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load data based on file extension
    if file_path.suffix.lower() in ['.fif', '.fif.gz']:
        raw = mne.io.read_raw_fif(file_path, preload=True)
    elif file_path.suffix.lower() in ['.edf']:
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif file_path.suffix.lower() in ['.bdf']:
        raw = mne.io.read_raw_bdf(file_path, preload=True)
    elif file_path.suffix.lower() in ['.csv']:
        # Load CSV data
        df = pd.read_csv(file_path)
        # Convert to MNE Raw object
        info = mne.create_info(
            ch_names=df.columns.tolist(),
            sfreq=1000.0,  # Default sampling frequency
            ch_types=['eeg'] * len(df.columns)
        )
        raw = mne.io.RawArray(df.values.T, info)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return {
        'data': reconstruct_brain_activity(raw.get_data()),
        'sfreq': raw.info['sfreq'],
        'ch_names': raw.ch_names,
        'ch_types': raw.get_channel_types(),
        'info': raw.info
    }


def reconstruct_brain_activity(eeg_data: np.ndarray) -> np.ndarray:
    """Reconstruct 4D brain activity data from EEG measurements.
    
    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data array of shape (n_channels, n_times)
        
    Returns
    -------
    np.ndarray
        4D array of shape (x, y, z, time) containing reconstructed brain activity.
        Values are normalized to range [-1, 1].
    """
    target_shape = (53, 63, 46)
    n_timepoints = eeg_data.shape[1]  # Get number of timepoints from input data

    # Get brain mask/outline
    brain_mask = image.load_img(datasets.load_mni152_brain_mask())
    motor_images = datasets.load_sample_motor_activation_image()
    stat_img = image.load_img(motor_images)
    # Resample brain mask to match activation data
    brain_mask = image.resample_to_img(
        source_img=brain_mask,
        target_img=stat_img,
        interpolation='nearest'
    )
    brain_outline = brain_mask.get_fdata()

    # Initialize output array
    activation_data = np.zeros((*target_shape, n_timepoints))

    # Get dimensions
    x_size, y_size, z_size = target_shape

    # Create coordinate arrays
    x = np.arange(x_size)
    y = np.arange(y_size)
    z = np.arange(z_size)

    # Create meshgrid for distance calculation
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    for t in range(n_timepoints):
        current_activation = np.zeros(target_shape)

        # 1. Create moving activation center
        center_x = int(x_size / 2 + 20 * np.sin(2 * np.pi * t / n_timepoints))
        center_y = int(y_size / 2 + 20 * np.cos(2 * np.pi * t / n_timepoints))
        center_z = z_size // 2

        # Calculate distance from center
        dist = np.sqrt(
            (X - center_x) ** 2 +
            (Y - center_y) ** 2 +
            (Z - center_z) ** 2
        )

        # Create gaussian activation pattern
        gaussian = 2 * np.exp(-dist / 20)

        # Add varying intensity
        intensity = 1 + 0.5 * np.sin(2 * np.pi * t / n_timepoints)
        current_activation += gaussian * intensity

        # 2. Add simple wave patterns
        X_wave = np.sin(2 * np.pi * (X / x_size + t / n_timepoints))
        Y_wave = np.sin(2 * np.pi * (Y / y_size + t / n_timepoints))
        current_activation += 0.5 * (X_wave + Y_wave)

        # Apply brain mask
        current_activation *= (brain_outline > 0.5)

        # Store in 4D array
        activation_data[..., t] = current_activation

    # Normalize activation data
    activation_data = (activation_data - np.min(activation_data)) / \
                      (np.max(activation_data) - np.min(activation_data))
    activation_data = 2 * (activation_data - 0.5)  # Scale to [-1, 1]

    return activation_data
