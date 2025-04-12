from nilearn import datasets, image, plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Dict, Any
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback_context
from scipy import ndimage
import socket
import nibabel as nib
import mne
from pathlib import Path
import os
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.input_format.data_loader import load_eeg_data
from src.visualization.glassbrain_plotter import GlassBrainPlotter
from src.visualization.time_series_plotter import TimeSeriesPlotter


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port
        s.listen(1)
        port = s.getsockname()[1]
    return port


class BrainViewerConnection:
    def __init__(self):
        self.app = Dash(__name__)
        self.time_data = None
        self.brain_data = None
        self.current_time = 0
        self.selected_channel = None

        # Load sample data
        self.load_sample_data()

        # Initialize plotters
        self.glass_brain_plotter = GlassBrainPlotter(
            activation_data=self.activation_data, brain_outline=self.brain_outline
        )
        self.time_series_plotter = TimeSeriesPlotter(
            activation_data=self.activation_data, time_data=self.time_data
        )

        self.init_layout()
        self.init_callbacks()
        self.port = find_free_port()

    def load_sample_data(self):
        """Load sample data from nilearn with enhanced spatial-temporal variations."""
        # Get sample motor activation data
        motor_images = datasets.load_sample_motor_activation_image()
        self.stat_img = image.load_img(motor_images)

        # Get brain mask/outline
        self.brain_mask = image.load_img(datasets.load_mni152_brain_mask())

        # Resample brain mask to match activation data
        self.brain_mask = image.resample_to_img(
            source_img=self.brain_mask,
            target_img=self.stat_img,
            interpolation="nearest",
        )

        # Convert to numpy arrays
        self.brain_outline = self.brain_mask.get_fdata()
        base_activation = self.stat_img.get_fdata()

        # Create time-varying activation data
        n_timepoints = 100
        self.activation_data = np.zeros((*base_activation.shape, n_timepoints))

        # Load EEG data
        current_dir = Path(__file__).parent
        sample_data_path = current_dir / "sample_data.fif"
        eeg_data = load_eeg_data(sample_data_path)

        # Store the EEG data
        self.activation_data = eeg_data["data"]

        # Create time series data
        self.time_data = np.arange(self.activation_data.shape[1])

    def update_glass_brain(
        self, time_point: int, selected_channel: Optional[Tuple[int, int]] = None
    ) -> go.Figure:
        """Update glass brain plot with outline and internal heatmap."""
        return self.glass_brain_plotter.update_glass_brain(
            time_point=time_point, selected_channel=selected_channel
        )

    def create_multi_view_brain(self, time_point: int) -> go.Figure:
        """Create multi-view brain visualization."""
        fig = make_subplots(
            rows=1, cols=3, subplot_titles=["Sagittal", "Coronal", "Axial"]
        )

        from scipy import ndimage

        # Handle EEG data format (channels, time) instead of (x, y, z, time)
        if len(self.activation_data.shape) == 2:  # EEG data format
            # Create 3D projections for EEG data
            n_channels = self.activation_data.shape[0]
            grid_size = int(np.ceil(np.cbrt(n_channels)))

            # Create a 3D grid for channel placement
            x_size, y_size, z_size = self.brain_outline.shape

            # Create empty 3D arrays for each view
            sagittal_data = np.zeros((y_size, z_size))
            coronal_data = np.zeros((x_size, z_size))
            axial_data = np.zeros((x_size, y_size))

            # Map EEG channels to 3D space
            for i in range(min(n_channels, grid_size * grid_size * grid_size)):
                x = i % grid_size
                y = (i // grid_size) % grid_size
                z = i // (grid_size * grid_size)

                # Scale to fit brain outline
                x_scaled = int(x * (x_size - 1) / (grid_size - 1))
                y_scaled = int(y * (y_size - 1) / (grid_size - 1))
                z_scaled = int(z * (z_size - 1) / (grid_size - 1))

                # Place channel value in each view
                sagittal_data[y_scaled, z_scaled] = self.activation_data[i, time_point]
                coronal_data[x_scaled, z_scaled] = self.activation_data[i, time_point]
                axial_data[x_scaled, y_scaled] = self.activation_data[i, time_point]

            # Create projections for each view
            sagittal_outline = np.max(self.brain_outline, axis=0)
            coronal_outline = np.max(self.brain_outline, axis=1)
            axial_outline = np.max(self.brain_outline, axis=2)

            # Create edges for each view
            sagittal_edges_x = ndimage.sobel(sagittal_outline, axis=0)
            sagittal_edges_y = ndimage.sobel(sagittal_outline, axis=1)
            sagittal_edges = np.sqrt(sagittal_edges_x**2 + sagittal_edges_y**2)

            coronal_edges_x = ndimage.sobel(coronal_outline, axis=0)
            coronal_edges_y = ndimage.sobel(coronal_outline, axis=1)
            coronal_edges = np.sqrt(coronal_edges_x**2 + coronal_edges_y**2)

            axial_edges_x = ndimage.sobel(axial_outline, axis=0)
            axial_edges_y = ndimage.sobel(axial_outline, axis=1)
            axial_edges = np.sqrt(axial_edges_x**2 + axial_edges_y**2)

            # Add brain outlines
            fig.add_trace(
                go.Contour(
                    z=sagittal_edges,
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgb(0,0,0)"]],
                    showscale=False,
                    contours=dict(start=0.1, end=0.1, size=0, coloring="lines"),
                    line=dict(width=2, color="black"),
                    name="sagittal outline",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Contour(
                    z=coronal_edges,
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgb(0,0,0)"]],
                    showscale=False,
                    contours=dict(start=0.1, end=0.1, size=0, coloring="lines"),
                    line=dict(width=2, color="black"),
                    name="coronal outline",
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Contour(
                    z=axial_edges,
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgb(0,0,0)"]],
                    showscale=False,
                    contours=dict(start=0.1, end=0.1, size=0, coloring="lines"),
                    line=dict(width=2, color="black"),
                    name="axial outline",
                ),
                row=1,
                col=3,
            )

            # Mask activation data
            sagittal_masked = np.where(sagittal_outline > 0.5, sagittal_data, np.nan)
            coronal_masked = np.where(coronal_outline > 0.5, coronal_data, np.nan)
            axial_masked = np.where(axial_outline > 0.5, axial_data, np.nan)

            # Add activation data
            fig.add_trace(
                go.Heatmap(
                    z=sagittal_masked,
                    colorscale="RdBu",
                    zmid=0,
                    showscale=(1 == 3),
                    opacity=1.0,
                    name="sagittal activation",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Heatmap(
                    z=coronal_masked,
                    colorscale="RdBu",
                    zmid=0,
                    showscale=(2 == 3),
                    opacity=1.0,
                    name="coronal activation",
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Heatmap(
                    z=axial_masked,
                    colorscale="RdBu",
                    zmid=0,
                    showscale=(3 == 3),
                    opacity=1.0,
                    name="axial activation",
                ),
                row=1,
                col=3,
            )

        else:  # Original 4D format
            # Get current time point data
            current_activation = self.activation_data[..., time_point]

            # Define projections for each view
            views = ["sagittal", "coronal", "axial"]
            for idx, view in enumerate(views, 1):
                if view == "sagittal":
                    outline_proj = np.max(self.brain_outline, axis=0)
                    activation_proj = np.max(current_activation, axis=0)
                elif view == "coronal":
                    outline_proj = np.max(self.brain_outline, axis=1)
                    activation_proj = np.max(current_activation, axis=1)
                else:  # axial
                    outline_proj = np.max(self.brain_outline, axis=2)
                    activation_proj = np.max(current_activation, axis=2)

                # Create edges
                edges_x = ndimage.sobel(outline_proj, axis=0)
                edges_y = ndimage.sobel(outline_proj, axis=1)
                edges = np.sqrt(edges_x**2 + edges_y**2)

                # Add brain outline
                fig.add_trace(
                    go.Contour(
                        z=edges,
                        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgb(0,0,0)"]],
                        showscale=False,
                        contours=dict(start=0.1, end=0.1, size=0, coloring="lines"),
                        line=dict(width=2, color="black"),
                        name=f"{view} outline",
                    ),
                    row=1,
                    col=idx,
                )

                # Mask activation data
                masked_activation = np.where(
                    outline_proj > 0.5, activation_proj, np.nan
                )

                # Add activation data
                fig.add_trace(
                    go.Heatmap(
                        z=masked_activation,
                        colorscale="RdBu",
                        zmid=0,
                        showscale=(idx == 3),
                        opacity=1.0,
                        name=f"{view} activation",
                    ),
                    row=1,
                    col=idx,
                )

        # Update layout
        fig.update_layout(
            height=400,
            width=1200,
            title=f"Multi-view Brain Activity at Time Point {time_point}",
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        # Update axes
        for i in range(1, 4):
            fig.update_xaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor=f"y{i}",
                scaleratio=1,
                row=1,
                col=i,
            )
            fig.update_yaxes(
                showgrid=False, zeroline=False, showticklabels=False, row=1, col=i
            )

        return fig
    def init_layout(self):
        """Initialize the dashboard layout."""
        self.app.layout = html.Div(
            [
                html.H1("Brain Activity Visualization"),
                # Info display for clicked points
                html.Div(
                    [
                        html.Div(id="click-data-time", style={"margin": "10px"}),
                        html.Div(id="click-data-brain", style={"margin": "10px"}),
                    ],
                    style={"padding": "10px", "backgroundColor": "#f0f0f0"},
                ),
                # Control Panel
                html.Div(
                    [
                        html.Button("Play/Pause", id="play-button"),
                        dcc.Slider(
                            id="time-slider",
                            min=0,
                            max=100,
                            step=1,
                            value=0,
                            marks={i: f"{i/100:.1f}s" for i in range(0, 101, 10)},
                        ),
                        dcc.Interval(
                            id="animation-interval", interval=100, disabled=True
                        ),
                    ],
                    style={"padding": "20px"},
                ),
                # Plots Container
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id="time-series-plot",
                                    config={"displayModeBar": True},
                                )
                            ],
                            style={"width": "50%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="glass-brain-plot",
                                    config={"displayModeBar": True},
                                )
                            ],
                            style={"width": "50%", "display": "inline-block"},
                        ),
                    ]
                ),
            ]
        )

    def init_callbacks(self):
        """Initialize callbacks including click interactions."""

        @self.app.callback(
            Output("animation-interval", "disabled"),
            Input("play-button", "n_clicks"),
            State("animation-interval", "disabled"),
        )
        def toggle_animation(n_clicks, current_state):
            if n_clicks is None:
                return True
            return not current_state

        @self.app.callback(
            [
                Output("time-series-plot", "figure"),
                Output("glass-brain-plot", "figure"),
                Output("click-data-time", "children"),
                Output("click-data-brain", "children"),
            ],
            [
                Input("time-slider", "value"),
                Input("animation-interval", "n_intervals"),
                Input("time-series-plot", "clickData"),
                Input("glass-brain-plot", "clickData"),
            ],
        )
        def update_all(slider_value, n_intervals, time_click_data, brain_click_data):
            ctx = callback_context
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # Handle time point selection
            if n_intervals is None:
                time_point = slider_value
            else:
                time_point = (slider_value + n_intervals) % len(self.time_data)

            # Update info text
            time_info = "No time point selected"
            brain_info = "No brain location selected"

            if trigger_id == "time-series-plot" and time_click_data:
                time_point = int(time_click_data["points"][0]["x"])
                time_info = f"Selected time: {time_point/100:.2f}s"

            if trigger_id == "glass-brain-plot" and brain_click_data:
                point = brain_click_data["points"][0]
                brain_info = f"Selected location: ({point['y']}, {point['x']})"

            # Update visualizations
            time_fig = self.update_time_series(time_point)
            brain_fig = self.create_multi_view_brain(time_point)

            return time_fig, brain_fig, time_info, brain_info

    def update_time_series(
        self, time_point: int, selected_channel: Optional[Tuple[int, int]] = None
    ) -> go.Figure:
        """Update time series plot with current time point highlighted."""
        return self.time_series_plotter.update_time_series(
            time_point=time_point, selected_channel=selected_channel
        )

    def run_server(self, debug: bool = True):
        """Run the Dash server."""
        print(f"Starting server on port: {self.port}")
        try:
            self.app.run(debug=debug, port=self.port)
        except Exception as e:
            print(f"Failed to start on port {self.port}. Error: {str(e)}")
            self.port = find_free_port()
            print(f"Retrying with port: {self.port}")
            self.app.run(debug=debug, port=self.port)


# Example usage
if __name__ == "__main__":
    viewer = BrainViewerConnection()

    viewer.run_server()

