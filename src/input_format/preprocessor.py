"""Preprocessing functions for EEG data."""

import numpy as np
import mne
from typing import Dict, Any, Optional, Tuple
from scipy import signal


def preprocess_data(
    data: Dict[str, Any],
    notch_freq: Optional[float] = 50.0,
    bandpass_freq: Optional[Tuple[float, float]] = (1.0, 40.0),
    remove_bad_channels: bool = True,
    interpolate_bad_channels: bool = True,
    ica: bool = False,
    ica_n_components: Optional[int] = None,
) -> Dict[str, Any]:
    """Preprocess EEG data with various filtering and cleaning steps.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing EEG data and metadata as returned by load_eeg_data()
    notch_freq : float, optional
        Frequency to notch filter (e.g., 50 Hz for power line noise)
    bandpass_freq : tuple of float, optional
        (low_freq, high_freq) for bandpass filtering
    remove_bad_channels : bool
        Whether to remove bad channels
    interpolate_bad_channels : bool
        Whether to interpolate bad channels instead of removing them
    ica : bool
        Whether to apply Independent Component Analysis
    ica_n_components : int, optional
        Number of ICA components to use

    Returns
    -------
    Dict[str, Any]
        Dictionary containing preprocessed data and metadata
    """
    # Convert to MNE Raw for preprocessing
    raw = mne.io.RawArray(data["data"], data["info"])

    # Notch filter
    if notch_freq is not None:
        raw.notch_filter(freqs=notch_freq)

    # Bandpass filter
    if bandpass_freq is not None:
        raw.filter(l_freq=bandpass_freq[0], h_freq=bandpass_freq[1])

    # # Find bad channels
    # if remove_bad_channels or interpolate_bad_channels:
    #     # Use MNE's automatic bad channel detection
    #     bads, _ = mne.preprocessing.find_bad_channels_maxwell(raw)
    #     raw.info['bads'] = bads

    #     if interpolate_bad_channels:
    #         # Interpolate bad channels
    #         raw.interpolate_bads()
    #     elif remove_bad_channels:
    #         # Remove bad channels
    #         raw.drop_channels(bads)

    # Apply ICA
    if ica:
        # Fit ICA
        ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=42)
        ica.fit(raw)

        # Find and remove eye movement components
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        if eog_indices:
            ica.exclude = eog_indices
            raw = ica.apply(raw)

    # Get preprocessed data
    preprocessed_data = raw.get_data()

    # Update channel information
    ch_names = raw.ch_names
    ch_types = raw.get_channel_types()

    return {
        "data": preprocessed_data,
        "sfreq": raw.info["sfreq"],
        "ch_names": ch_names,
        "ch_types": ch_types,
        "info": raw.info,
    }
