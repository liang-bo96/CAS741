"""GlassBrain plotting functions for EEG data visualization using Plotly."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
from scipy import ndimage
from nilearn import image


class GlassBrainPlotter:
    def __init__(
        self,
        activation_data: np.ndarray,
        brain_outline: np.ndarray,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        dest: str = "mri",
        title: Optional[str] = None,
        mri_resolution: bool = False,
        mni305: Optional[bool] = None,
        black_bg: bool = False,
        display_mode: Optional[str] = None,
        threshold: Optional[Union[float, str]] = None,
        colorbar: bool = False,
        draw_cross: bool = True,
        annotate: bool = True,
        alpha: float = 0.7,
        plot_abs: bool = False,
        draw_arrows: bool = True,
        symmetric_cbar: Union[bool, str] = "auto",
        **kwargs,
    ):
        """Initialize the GlassBrainPlotter.

        Parameters
        ----------
        activation_data : np.ndarray
            Brain activation data array. Can be either:
            - 2D array of shape (n_channels, n_times) for EEG data
            - 4D array of shape (x, y, z, time) for volumetric data
        brain_outline : np.ndarray
            3D array containing the brain mask/outline
        cmap : str, optional
            Colormap name
        vmin : float, optional
            Minimum value for colormap
        vmax : float, optional
            Maximum value for colormap
        dest : str
            Coordinate system ('mri' or 'surf')
        title : str, optional
            Plot title
        mri_resolution : bool
            Whether to use MRI resolution
        mni305 : bool, optional
            Whether to project from MNI-305 to MNI-152 space
        black_bg : bool
            Whether to use black background
        display_mode : str, optional
            Display mode for cuts
        threshold : float or str, optional
            Threshold for transparency
        colorbar : bool
            Whether to show colorbar
        draw_cross : bool
            Whether to draw crosshair
        annotate : bool
            Whether to add annotations
        alpha : float
            Transparency for brain overlay
        plot_abs : bool
            Whether to plot absolute values
        draw_arrows : bool
            Whether to draw direction arrows
        symmetric_cbar : bool or str
            Whether to use symmetric colorbar
        **kwargs
            Additional arguments passed to Plotly
        """
        self.activation_data = activation_data
        self.brain_outline = brain_outline
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.dest = dest
        self.title = title
        self.mri_resolution = mri_resolution
        self.mni305 = mni305
        self.black_bg = black_bg
        self.display_mode = display_mode
        self.threshold = threshold
        self.colorbar = colorbar
        self.draw_cross = draw_cross
        self.annotate = annotate
        self.alpha = alpha
        self.plot_abs = plot_abs
        self.draw_arrows = draw_arrows
        self.symmetric_cbar = symmetric_cbar
        self.kwargs = kwargs

    def plot_glassbrain(self) -> go.Figure:
        """Create a GlassBrain plot of EEG data using Plotly.

        Parameters
        ----------
        data : Dict[str, Any] or pd.DataFrame
            EEG data dictionary or DataFrame
        cmap : str, optional
            Colormap name
        vmin : float, optional
            Minimum value for colormap
        vmax : float, optional
            Maximum value for colormap
        dest : str
            Coordinate system ('mri' or 'surf')
        mri_resolution : bool
            Whether to use MRI resolution
        mni305 : bool, optional
            Whether to project from MNI-305 to MNI-152 space
        black_bg : bool
            Whether to use black background
        display_mode : str, optional
            Display mode for cuts
        threshold : float or str, optional
            Threshold for transparency
        colorbar : bool
            Whether to show colorbar
        draw_cross : bool
            Whether to draw crosshair
        annotate : bool
            Whether to add annotations
        alpha : float
            Transparency for brain overlay
        plot_abs : bool
            Whether to plot absolute values
        draw_arrows : bool
            Whether to draw direction arrows
        symmetric_cbar : bool or str
            Whether to use symmetric colorbar
        interpolation : str
            Interpolation method
        show_time : bool
            Whether to show time label
        **kwargs
            Additional arguments passed to Plotly

        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Convert data to DataFrame if needed
        if isinstance(self.activation_data, dict):
            df = pd.DataFrame(
                self.activation_data["data"].T,
                index=np.arange(self.activation_data["data"].shape[1])
                / self.activation_data["sfreq"],
                columns=self.activation_data["ch_names"],
            )
        else:
            df = pd.DataFrame(
                self.activation_data.T, columns=range(self.activation_data.shape[0])
            )

        # Create figure
        fig = go.Figure()

        # Add traces for each channel
        for col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[col], name=col, mode="lines", line=dict(width=1)
                )
            )

        # Update layout
        fig.update_layout(
            title=self.title,
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            showlegend=True,
            template="plotly_dark" if self.black_bg else "plotly_white",
        )

        return fig

    def update_glass_brain(
        self, time_point: int, selected_channel: Optional[Tuple[int, int]] = None
    ) -> go.Figure:
        """Update glass brain plot with outline and internal heatmap.

        Parameters
        ----------
        time_point : int
            Current time point to display
        selected_channel : Optional[Tuple[int, int]]
            Optional tuple of (channel_index, value) for highlighting specific channels

        Returns
        -------
        go.Figure
            Plotly figure object containing the glass brain visualization
        """
        fig = go.Figure()

        # Get maximum intensity projections
        outline_proj = np.max(self.brain_outline, axis=2)

        # Handle EEG data format (channels, time) instead of (x, y, z, time)
        if len(self.activation_data.shape) == 2:  # EEG data format
            # Create a 2D projection for EEG data
            activation_proj = np.zeros_like(outline_proj)

            # Map EEG channels to 2D space
            n_channels = self.activation_data.shape[0]
            grid_size = int(np.ceil(np.sqrt(n_channels)))

            # Create a grid for channel placement
            for i in range(min(n_channels, grid_size * grid_size)):
                row = i // grid_size
                col = i % grid_size

                # Scale to fit brain outline
                x = int(col * (outline_proj.shape[1] - 1) / (grid_size - 1))
                y = int(row * (outline_proj.shape[0] - 1) / (grid_size - 1))

                # Place channel value
                activation_proj[y, x] = self.activation_data[i, time_point]
        else:  # Original 4D format
            # Select the current time point for activation data
            current_activation = self.activation_data[..., time_point]
            activation_proj = np.max(current_activation, axis=2)

        # Create brain outline (only the edges)
        # Get edges using gradient
        edges_x = ndimage.sobel(outline_proj, axis=0)
        edges_y = ndimage.sobel(outline_proj, axis=1)
        edges = np.sqrt(edges_x**2 + edges_y**2)

        # Add brain outline
        fig.add_trace(
            go.Contour(
                z=edges,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgb(0,0,0)"]],  # Black outline
                showscale=False,
                contours=dict(
                    start=0.1,  # Adjust threshold for edge detection
                    end=0.1,
                    size=0,
                    coloring="lines",  # Only show lines
                ),
                line=dict(width=2, color="black"),
                name="Brain Outline",
            )
        )

        # Mask activation data with brain outline
        masked_activation = np.where(outline_proj > 0.5, activation_proj, np.nan)

        # Add activation data as heatmap
        fig.add_trace(
            go.Heatmap(
                z=masked_activation,
                colorscale="RdBu",
                zmid=0,
                showscale=True,
                opacity=1.0,  # Full opacity for clear visualization
                colorbar=dict(
                    title="Activation",
                    thickness=15,
                    len=0.9,
                    tickmode="auto",
                    nticks=10,
                ),
                name="Neural Activity",
            )
        )

        # Update layout
        fig.update_layout(
            title=f"Brain Activity at Time Point {time_point}",
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        return fig

    def plot_glassbrain_butterfly(
        data: Union[Dict[str, Any], pd.DataFrame],
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        dest: str = "mri",
        mri_resolution: bool = False,
        mni305: Optional[bool] = None,
        black_bg: bool = False,
        display_mode: Optional[str] = None,
        threshold: Optional[Union[float, str]] = None,
        colorbar: bool = False,
        alpha: float = 0.7,
        plot_abs: bool = False,
        draw_arrows: bool = True,
        symmetric_cbar: Union[bool, str] = "auto",
        interpolation: str = "nearest",
        w: float = 5,
        h: float = 2.5,
        xlim: Optional[Union[float, Tuple[float, float]]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """Create a butterfly plot with time-linked GlassBrain plot using Plotly.

        Parameters
        ----------
        data : Dict[str, Any] or pd.DataFrame
            EEG data dictionary or DataFrame
        cmap : str, optional
            Colormap name
        vmin : float, optional
            Minimum value for colormap
        vmax : float, optional
            Maximum value for colormap
        dest : str
            Coordinate system ('mri' or 'surf')
        mri_resolution : bool
            Whether to use MRI resolution
        mni305 : bool, optional
            Whether to project from MNI-305 to MNI-152 space
        black_bg : bool
            Whether to use black background
        display_mode : str, optional
            Display mode for cuts
        threshold : float or str, optional
            Threshold for transparency
        colorbar : bool
            Whether to show colorbar
        alpha : float
            Transparency for brain overlay
        plot_abs : bool
            Whether to plot absolute values
        draw_arrows : bool
            Whether to draw direction arrows
        symmetric_cbar : bool or str
            Whether to use symmetric colorbar
        interpolation : str
            Interpolation method
        w : float
            Plot width in inches
        h : float
            Plot height in inches
        xlim : float or tuple, optional
            X-axis limits
        name : str, optional
            Window title
        **kwargs
            Additional arguments passed to Plotly

        Returns
        -------
        go.Figure
            Plotly figure object with butterfly plot and GlassBrain visualization
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

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Butterfly Plot", "GlassBrain Visualization"),
            row_heights=[0.6, 0.4],
        )

        # Add butterfly plot
        for col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    mode="lines",
                    line=dict(width=1),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Add GlassBrain visualization
        fig.add_trace(
            go.Heatmap(
                z=df.values,
                x=df.index,
                y=df.columns,
                colorscale=cmap or "RdBu",
                zmin=vmin,
                zmax=vmax,
                showscale=colorbar,
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=name or "EEG Data Visualization",
            height=h * 800,  # Convert inches to pixels
            width=w * 800,
            template="plotly_dark" if black_bg else "plotly_white",
            showlegend=False,
        )

        # Update axes
        if xlim is not None:
            if isinstance(xlim, tuple):
                fig.update_xaxes(range=xlim, row=1, col=1)
                fig.update_xaxes(range=xlim, row=2, col=1)
            else:
                fig.update_xaxes(range=[0, xlim], row=1, col=1)
                fig.update_xaxes(range=[0, xlim], row=2, col=1)

        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Channels", row=2, col=1)

        return fig
