"""Format conversion functions for EEG data processing."""

import numpy as np
import pandas as pd
import mne
from typing import Dict, Any, Union


def convert_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert EEG data to pandas DataFrame format.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing EEG data and metadata as returned by load_eeg_data()

    Returns
    -------
    pd.DataFrame
        DataFrame with channels as columns and time points as index
    """
    # Create time index
    n_times = data["data"].shape[1]
    time_index = pd.Index(np.arange(n_times) / data["sfreq"], name="time")

    # Create DataFrame
    df = pd.DataFrame(data["data"].T, index=time_index, columns=data["ch_names"])

    return df


def convert_to_mne(data: Dict[str, Any]) -> mne.io.Raw:
    """Convert EEG data to MNE-Python Raw format.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing EEG data and metadata as returned by load_eeg_data()

    Returns
    -------
    mne.io.Raw
        MNE-Python Raw object containing the EEG data
    """
    # Create info object
    info = mne.create_info(
        ch_names=data["ch_names"], sfreq=data["sfreq"], ch_types=data["ch_types"]
    )

    # Create Raw object
    raw = mne.io.RawArray(data["data"], info)

    return raw
