"""Test cases for visualization module."""

import unittest
import numpy as np
import plotly.graph_objects as go
import mne
import pandas as pd


from src.visualization.topo_plotter import plot_topo
from src.visualization.time_series_plotter import TimeSeriesPlotter
from src.visualization.glassbrain_plotter import GlassBrainPlotter
from src.visualization.butterfly_plotter import plot_butterfly


class TestVisualization(unittest.TestCase):
    """Test cases for visualization module."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create sample EEG data
        n_channels = 32  # More realistic number of channels
        n_times = 1000
        sfreq = 1000.0

        # Create random EEG data
        cls.test_data = np.random.randn(n_channels, n_times)

        # Create realistic channel names
        cls.ch_names = [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "FC5",
            "FC1",
            "FC2",
            "FC6",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "TP9",
            "CP5",
            "CP1",
            "CP2",
            "CP6",
            "TP10",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "PO9",
            "O1",
            "Oz",
            "O2",
            "PO10",
        ]
        cls.ch_types = ["eeg"] * n_channels

        # Create MNE info object with montage
        cls.info = mne.create_info(
            ch_names=cls.ch_names, sfreq=sfreq, ch_types=cls.ch_types
        )

        # Set montage
        montage = mne.channels.make_standard_montage("standard_1020")
        cls.info.set_montage(montage)

        # Create time points array
        cls.times = np.arange(n_times) / sfreq

        # Create data dictionary
        cls.data = {
            "data": cls.test_data,
            "sfreq": sfreq,
            "ch_names": cls.ch_names,
            "ch_types": cls.ch_types,
            "info": cls.info,
            "times": cls.times,
        }

        # Create brain data for glassbrain plotter
        cls.brain_data = np.random.randn(100, 100, 100)  # 3D brain data
        cls.glassbrain_plotter = GlassBrainPlotter(
            activation_data=cls.test_data,  # Using test_data as activation data
            brain_outline=cls.brain_data,  # Using brain_data as outline
            title="Test GlassBrain",
        )

        # Create time series plotter
        cls.time_series_plotter = TimeSeriesPlotter(
            activation_data=cls.test_data,  # Using test_data as activation data
            time_data=cls.times,  # Using times array for time points
            channels=cls.ch_names,  # Using channel names
            title="Test Time Series",
        )

    def test_plot_topography(self):
        """Test topographic plotting function."""
        # Test single time point
        fig = plot_topo(self.data, time_point=0.2, title="Test Topography")
        fig.show()
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, "Topographic Maps")

        # Test multiple time points
        fig = plot_topo(
            self.data, times=[0.1, 0.2, 0.3], vmin=-2, vmax=2, colorscale="RdBu"
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)  # One heatmap for each time point

        # Test with channel selection
        fig = plot_topo(self.data, time_point=0.2, channels=["Fz", "Cz", "Pz"])
        self.assertIsInstance(fig, go.Figure)

        # Test with no times specified (using bins)
        fig = plot_topo(self.data, n_bins=4, vmin=-2, vmax=2, cmap="RdBu")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 4)  # One heatmap for each bin

    def test_plot_time_series(self):
        """Test time series plotting function."""
        # Test basic plot
        fig = self.time_series_plotter.plot_time_series()
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, "Test Time Series")

        # Test with channel selection
        plotter = TimeSeriesPlotter(
            activation_data=self.test_data,
            time_data=self.times,
            channels=["Fz", "Cz", "Pz"],
            title="Channel Selection Test",
        )
        fig = plotter.plot_time_series()
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 3)  # At least 3 channels

        # Test with time window and baseline
        plotter = TimeSeriesPlotter(
            activation_data=self.test_data,
            time_data=self.times,
            channels=self.ch_names,
            times=(0.1, 0.5),
            vmin=-10,
            vmax=10,
            title="Time Window Test",
        )
        fig = plotter.plot_time_series()
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(tuple(fig.layout.yaxis.range), (-10, 10))

        # Test with single time point
        plotter = TimeSeriesPlotter(
            activation_data=self.test_data,
            time_data=self.times,
            channels=self.ch_names,
            times=0.2,
            title="Single Time Point Test",
        )
        fig = plotter.plot_time_series()
        self.assertIsInstance(fig, go.Figure)

    def test_update_time_series(self):
        """Test time series update function."""
        # Test without selected channel
        time_data = np.mean(self.test_data, axis=0)
        plotter = TimeSeriesPlotter(
            activation_data=time_data, time_data=self.times, title="Update Test"
        )
        fig = plotter.update_time_series(time_point=50)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)  # Average signal
        self.assertEqual(len(fig.layout.shapes), 1)  # Vertical line

        # Test with selected channel
        fig = plotter.update_time_series(time_point=50, selected_channel=(10, 20))
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)  # Signal + selected channel marker

    def test_plot_time_statistics(self):
        """Test time series statistics plotting function."""
        # Test mean statistics
        plotter = TimeSeriesPlotter(
            activation_data=self.test_data,
            time_data=self.times,
            channels=["Fz", "Cz", "Pz"],
            title="Statistics Test",
        )
        fig = plotter.plot_time_series()
        self.assertIsInstance(fig, go.Figure)

        # Test with different statistics
        for stat in ["mean", "std", "sem", "ci"]:
            plotter = TimeSeriesPlotter(
                activation_data=self.test_data,
                time_data=self.times,
                channels=["Fz", "Cz", "Pz"],
                title=f"{stat.upper()} Test",
            )
            fig = plotter.plot_time_series()
            self.assertIsInstance(fig, go.Figure)

    def test_plot_glassbrain(self):
        """Test glassbrain plotting function."""
        # Test basic plot
        fig = self.glassbrain_plotter.plot_glassbrain()
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, "Test GlassBrain")

        # Test with different parameters
        self.glassbrain_plotter.threshold = 1.0
        self.glassbrain_plotter.cmap = "Hot"
        fig = self.glassbrain_plotter.plot_glassbrain()
        self.assertIsInstance(fig, go.Figure)

        # Test update with time point
        fig = self.glassbrain_plotter.update_glass_brain(time_point=50)
        self.assertIsInstance(fig, go.Figure)

        # Test update with selected channel
        fig = self.glassbrain_plotter.update_glass_brain(
            time_point=50, selected_channel=(10, 20)
        )
        self.assertIsInstance(fig, go.Figure)

        # Test creating a new instance with different parameters
        new_plotter = GlassBrainPlotter(
            activation_data=self.test_data,
            brain_outline=self.brain_data,
            cmap="RdBu",
            vmin=-2,
            vmax=2,
            black_bg=True,
            colorbar=True,
            title="Test GlassBrain New",
        )
        fig = new_plotter.plot_glassbrain()
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, "Test GlassBrain New")

    def test_plot_butterfly(self):
        """Test butterfly plotting function."""
        # Test basic plot
        fig = plot_butterfly(self.data, name="Test Butterfly")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.title.text, "Test Butterfly")

        # Test with averaging
        fig = plot_butterfly(self.data, average=True, show_std=True)
        self.assertIsInstance(fig, go.Figure)
        fig.show()
        self.assertTrue(len(fig.data) > 1)  # Mean line + std band

        # Test with channel groups
        channel_groups = {
            "Frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
            "Central": ["C3", "Cz", "C4"],
            "Parietal": ["P3", "Pz", "P4"],
        }
        fig = plot_butterfly(
            self.data, channel_groups=channel_groups, colors=["red", "green", "blue"]
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 32)

        # Test with DataFrame input and additional parameters
        df = pd.DataFrame(self.test_data.T, columns=self.ch_names)
        fig = plot_butterfly(
            df,
            vmin=-2,
            vmax=2,
            xlim=(0, 0.5),
            w=8,
            h=6,
            ylabel=["Channel 1", "Channel 2"],
            axtitle=True,
        )
        self.assertIsInstance(fig, go.Figure)


if __name__ == "__main__":
    unittest.main()
