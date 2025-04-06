"""Butterfly plotting functions for EEG data visualization using Plotly."""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Union, Tuple
import pandas as pd


def plot_butterfly(
    data: Union[Dict[str, Any], pd.DataFrame],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xlim: Optional[Union[float, Tuple[float, float]]] = None,
    w: float = 5,
    h: float = 2.5,
    name: Optional[str] = None,
    color: str = "black",
    ylabel: Optional[list] = None,
    axtitle: bool = True,
    **kwargs
) -> go.Figure:
    """Create a butterfly plot of EEG data using Plotly.

    Parameters
    ----------
    data : Dict[str, Any] or pd.DataFrame
        EEG data dictionary or DataFrame
    vmin : float, optional
        Minimum value for y-axis
    vmax : float, optional
        Maximum value for y-axis
    xlim : float or tuple, optional
        X-axis limits
    w : float
        Plot width in inches
    h : float
        Plot height in inches
    name : str, optional
        Window title
    color : str
        Line color
    ylabel : list, optional
        Y-axis labels
    axtitle : bool
        Whether to show axis titles
    **kwargs
        Additional arguments passed to Plotly

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Convert data to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(
            data["data"].T,
            index=np.arange(data["data"].shape[1]) / data["sfreq"],
            columns=data["ch_names"],
        )
    else:
        df = data

    # Create figure
    fig = go.Figure()

    # Add traces for each channel
    for col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                mode="lines",
                line=dict(width=1, color=color),
                showlegend=False,
            )
        )

    # Update layout
    fig.update_layout(
        title=name or "Butterfly Plot",
        height=h * 800,  # Convert inches to pixels
        width=w * 800,
        showlegend=False,
    )

    # Update axes
    if xlim is not None:
        if isinstance(xlim, tuple):
            fig.update_xaxes(range=xlim)
        else:
            fig.update_xaxes(range=[0, xlim])

    if vmin is not None:
        fig.update_yaxes(range=[vmin, vmax])

    if axtitle:
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Amplitude")

    if ylabel is not None:
        fig.update_yaxes(ticktext=ylabel)

    return fig
