"""Topographic map plotting functions for EEG data visualization using Plotly."""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Union, Tuple
import pandas as pd


def plot_topo(
    data: Union[Dict[str, Any], pd.DataFrame],
    times: Optional[Union[float, list[float]]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    n_bins: int = 6,
    name: Optional[str] = None,
    **kwargs,
) -> go.Figure:
    """Create a topographic map of EEG data using Plotly.

    Parameters
    ----------
    data : Dict[str, Any] or pd.DataFrame
        EEG data dictionary or DataFrame
    times : float or list[float], optional
        Time point or time window to plot
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap
    cmap : str, optional
        Colormap name
    n_bins : int
        Number of time bins for TopomapBins
    name : str, optional
        Window title
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

    if times is None:
        # Create time bins
        time_bins = np.linspace(df.index[0], df.index[-1], n_bins + 1)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

        # Create subplots for each time bin
        fig = go.Figure()

        for i, t in enumerate(bin_centers):
            # Get data for this time bin
            mask = (df.index >= time_bins[i]) & (df.index < time_bins[i + 1])
            bin_data = df[mask].mean()

            # Add heatmap for this time bin
            fig.add_trace(
                go.Heatmap(
                    z=bin_data.values.reshape(1, -1),
                    x=bin_data.index,
                    y=[f"{t:.2f}s"],
                    colorscale=cmap or "RdBu",
                    zmin=vmin,
                    zmax=vmax,
                    showscale=True if i == 0 else False,
                    name=f"{t:.2f}s",
                )
            )

        # Update layout
        fig.update_layout(
            title=name or "Topographic Maps",
            height=200 * n_bins,
            width=800,
            showlegend=False,
        )

        # Update axes
        fig.update_xaxes(title_text="Channels")
        fig.update_yaxes(title_text="Time")

    else:
        # Single time point or window
        if isinstance(times, list):
            plot_data = df[df.index.isin(times)].mean()
        else:
            # Find closest time point
            idx = np.abs(df.index - times).argmin()
            plot_data = df.iloc[idx]

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=plot_data.values.reshape(1, -1),
                x=plot_data.index,
                y=[""],
                colorscale=cmap or "RdBu",
                zmin=vmin,
                zmax=vmax,
                showscale=True,
            )
        )

        # Update layout
        fig.update_layout(
            title=name or "Topographic Map", height=200, width=800, showlegend=False
        )

        # Update axes
        fig.update_xaxes(title_text="Channels")
        fig.update_yaxes(title_text="")

    return fig
