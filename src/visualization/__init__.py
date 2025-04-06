"""Visualization Module for EEG Data

This module handles the visualization of EEG data, including:
- GlassBrain plots
- Butterfly plots
- Topographic maps
- Time series plots
"""

from .glassbrain_plotter import GlassBrainPlotter
from .butterfly_plotter import plot_butterfly
from .topo_plotter import plot_topo
from .time_series_plotter import TimeSeriesPlotter

__all__ = ["GlassBrainPlotter", "TimeSeriesPlotter", "plot_butterfly", "plot_topo"]
