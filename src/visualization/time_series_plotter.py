"""Time series plotting functions for EEG data visualization using Plotly."""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Union, Tuple, List
import pandas as pd

class TimeSeriesPlotter:
    def __init__(
        self,
        activation_data: np.ndarray,
        time_data: np.ndarray,
        channels: Optional[List[str]] = None,
        times: Optional[Union[float, Tuple[float, float]]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        name: Optional[str] = None,
        color: str = 'black',
        title: Optional[str] = None,
        **kwargs
    ):
        """Initialize TimeSeriesPlotter.
        
        Parameters
        ----------
        activation_data : np.ndarray
            Brain activation data array. Can be either:
            - 2D array of shape (n_channels, n_times) for EEG data
            - 4D array of shape (x, y, z, time) for volumetric data
        time_data : np.ndarray
            Array of time points corresponding to the activation data
        channels : Optional[List[str]]
            List of channel names to plot
        times : Optional[Union[float, Tuple[float, float]]]
            Time point or time window to plot
        vmin : Optional[float]
            Minimum value for y-axis
        vmax : Optional[float]
            Maximum value for y-axis
        name : Optional[str]
            Window title
        color : str
            Line color
        title : Optional[str]
            Plot title
        **kwargs
            Additional arguments passed to Plotly
        """
        self.activation_data = activation_data
        self.time_data = time_data
        self.channels = channels
        self.times = times
        self.vmin = vmin
        self.vmax = vmax
        self.name = name
        self.color = color
        self.title = title
        self.kwargs = kwargs
        
    def plot_time_series(self) -> go.Figure:
        """Create a time series plot of EEG data using Plotly."""
        # Create DataFrame from numpy array if time_data is not already a DataFrame
        if not isinstance(self.time_data, pd.DataFrame):
            # Create default channel names if none provided
            if self.channels is None:
                channel_names = [f'Channel {i}' for i in range(self.activation_data.shape[0])]
                data_to_plot = self.activation_data
            else:
                # If channels are specified, use only the first len(channels) channels
                channel_names = self.channels
                data_to_plot = self.activation_data[:len(channel_names)]
                
            # Create DataFrame with time points as index and channels as columns
            df = pd.DataFrame(
                data_to_plot.T,  # Transpose to get (time_points, channels)
                index=self.time_data,
                columns=channel_names
            )
        else:
            df = self.time_data

        # Create figure
        fig = go.Figure()
        
        # Add traces for each channel
        for col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                mode='lines',
                line=dict(width=1, color=self.color)
            ))
        
        # Update layout
        fig.update_layout(
            title_text=self.title or self.name or 'Time Series Plot',
            height=600,
            width=800,
            showlegend=True
        )
        
        # Update axes
        if self.vmin is not None:
            fig.update_yaxes(range=[self.vmin, self.vmax])
        
        fig.update_xaxes(title_text='Time (s)')
        fig.update_yaxes(title_text='Amplitude')
        
        return fig
    
    def update_time_series(self, time_point: int, selected_channel: Optional[Tuple[int, int]] = None) -> go.Figure:
        """Update time series plot with current time point highlighted.
        
        Parameters
        ----------
        time_point : int
            Current time point to highlight
        selected_channel : Optional[Tuple[int, int]]
            Optional tuple of (channel_index, value) for highlighting specific channels
            
        Returns
        -------
        go.Figure
            Plotly figure object containing the time series visualization
        """
        fig = go.Figure()
        
        # Handle EEG data format (channels, time) instead of (x, y, z, time)
        if len(self.activation_data.shape) == 2:  # EEG data format
            # Plot each channel
            for i in range(self.activation_data.shape[0]):
                fig.add_trace(go.Scatter(
                    x=self.time_data,
                    y=self.activation_data[i, :],
                    mode='lines',
                    name=f'Channel {i}',
                    line=dict(width=1)
                ))
        else:  # Original 4D format
            # Create time series from activation data
            # Take mean across spatial dimensions for each time point
            mean_activation = np.mean(self.activation_data.reshape(-1, self.activation_data.shape[-1]), axis=0)
            
            # Add time series data
            fig.add_trace(go.Scatter(
                x=self.time_data,
                y=mean_activation,
                mode='lines',
                name='Mean Brain Activity'
            ))
        
        # Add vertical line for current time
        fig.add_vline(
            x=time_point,
            line_width=2,
            line_dash="dash",
            line_color="red"
        )
        
        # Highlight selected channel if any
        if selected_channel is not None:
            ch_idx, ch_value = selected_channel
            fig.add_trace(go.Scatter(
                x=[time_point],
                y=[ch_value],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='circle'
                ),
                name='Selected Channel'
            ))
        
        # Update layout
        fig.update_layout(
            title='Brain Activity Over Time',
            xaxis_title='Time Point',
            yaxis_title='Activation',
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig

    def plot_time_statistics(
        data: Union[Dict[str, Any], pd.DataFrame],
        stat: str = 'mean',
        channels: Optional[List[str]] = None,
        times: Optional[Union[float, Tuple[float, float]]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        name: Optional[str] = None,
        color: str = 'black',
        **kwargs
    ) -> go.Figure:
        """Create a time series statistics plot of EEG data using Plotly.
        
        Parameters
        ----------
        data : Dict[str, Any] or pd.DataFrame
            EEG data dictionary or DataFrame
        stat : str
            Statistical measure to plot ('mean', 'std', 'sem', 'ci')
        channels : list of str, optional
            Channels to plot
        times : float or tuple, optional
            Time point or time window to plot
        vmin : float, optional
            Minimum value for y-axis
        vmax : float, optional
            Maximum value for y-axis
        name : str, optional
            Window title
        color : str
            Line color
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
                data['data'].T,
                index=np.arange(data['data'].shape[1]) / data['sfreq'],
                columns=data['ch_names']
            )
        else:
            df = data
        
        # Select channels if specified
        if channels is not None:
            df = df[channels]
        
        # Select time window if specified
        if times is not None:
            if isinstance(times, tuple):
                mask = (df.index >= times[0]) & (df.index <= times[1])
                df = df[mask]
            else:
                # Find closest time point
                idx = np.abs(df.index - times).argmin()
                df = df.iloc[:idx + 1]
        
        # Calculate statistics
        if stat == 'mean':
            stat_data = df.mean(axis=1)
            stat_name = 'Mean'
        elif stat == 'std':
            stat_data = df.std(axis=1)
            stat_name = 'Standard Deviation'
        elif stat == 'sem':
            stat_data = df.sem(axis=1)
            stat_name = 'Standard Error of Mean'
        elif stat == 'ci':
            # Calculate 95% confidence interval
            mean = df.mean(axis=1)
            std = df.std(axis=1)
            n = len(df.columns)
            ci = 1.96 * std / np.sqrt(n)
            stat_data = pd.DataFrame({
                'lower': mean - ci,
                'upper': mean + ci
            })
            stat_name = '95% Confidence Interval'
        else:
            raise ValueError(f"Unknown statistical measure: {stat}")
        
        # Create figure
        fig = go.Figure()
        
        if stat == 'ci':
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=df.index,
                y=stat_data['upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Upper CI'
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=stat_data['lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Lower CI'
            ))
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df.mean(axis=1),
                mode='lines',
                line=dict(width=1, color=color),
                name='Mean'
            ))
        else:
            # Add statistical measure
            fig.add_trace(go.Scatter(
                x=df.index,
                y=stat_data,
                mode='lines',
                line=dict(width=1, color=color),
                name=stat_name
            ))
        
        # Update layout
        fig.update_layout(
            title=name or f'Time Series {stat_name}',
            height=600,
            width=800,
            showlegend=True
        )
        
        # Update axes
        if vmin is not None:
            fig.update_yaxes(range=[vmin, vmax])
        
        fig.update_xaxes(title_text='Time (s)')
        fig.update_yaxes(title_text='Amplitude')
        
        return fig 