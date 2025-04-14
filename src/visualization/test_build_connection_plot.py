"""Test cases for BrainViewerConnection class."""

import unittest
import importlib.util
import pytest
import numpy as np
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Check if required modules are available
plotly_available = importlib.util.find_spec("plotly") is not None
mne_available = importlib.util.find_spec("mne") is not None
nilearn_available = importlib.util.find_spec("nilearn") is not None
dash_available = importlib.util.find_spec("dash") is not None

# Import modules only if available
if plotly_available:
    import plotly.graph_objects as go

if mne_available:
    import mne

# Import with error handling
try:
    from src.visualization.build_connection_plot import BrainViewerConnection, find_free_port
    build_connection_available = True
except ImportError:
    build_connection_available = False


@pytest.mark.skipif(not all([plotly_available, mne_available, nilearn_available, 
                             dash_available, build_connection_available]),
                    reason="Required dependencies are missing")
class TestBrainViewerConnection(unittest.TestCase):
    """Test cases for BrainViewerConnection class."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create sample EEG data
        n_channels = 32
        n_times = 1000
        sfreq = 1000.0

        # Create random EEG data
        cls.test_data = np.random.randn(n_channels, n_times)

        # Create channel names
        cls.ch_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
            "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10",
        ]
        cls.ch_types = ["eeg"] * n_channels

        # Create MNE info object with montage
        if mne_available:
            cls.info = mne.create_info(
                ch_names=cls.ch_names, sfreq=sfreq, ch_types=cls.ch_types
            )

            # Set montage
            montage = mne.channels.make_standard_montage("standard_1020")
            cls.info.set_montage(montage)
        else:
            cls.info = {}

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

        # Create 3D brain data for testing
        cls.brain_data = np.random.randn(53, 63, 46)

    def test_find_free_port(self):
        """Test the find_free_port function."""
        port = find_free_port()
        self.assertIsInstance(port, int)
        self.assertTrue(1024 <= port <= 65535)  # Valid port range

    @patch('src.visualization.build_connection_plot.load_data')
    @patch('src.visualization.build_connection_plot.datasets')
    @patch('src.visualization.build_connection_plot.image')
    def test_initialization(self, mock_image, mock_datasets, mock_load_data):
        """Test BrainViewerConnection initialization."""
        # Setup mocks
        mock_load_data.return_value = {
            "data": self.test_data,
            "sfreq": 1000.0,
            "ch_names": self.ch_names,
            "ch_types": self.ch_types,
            "info": self.info,
            "times": self.times,
        }
        
        mock_stat_img = MagicMock()
        mock_image.load_img.return_value = mock_stat_img
        
        mock_brain_mask = MagicMock()
        mock_brain_mask.get_fdata.return_value = self.brain_data
        mock_image.resample_to_img.return_value = mock_brain_mask

        # Create instance
        viewer = BrainViewerConnection()
        
        # Verify initialization
        self.assertIsNotNone(viewer.app)
        self.assertIsNotNone(viewer.port)
        self.assertIsInstance(viewer.port, int)
        self.assertEqual(viewer.current_time, 0)
        self.assertIsNone(viewer.selected_channel)
        self.assertIsNotNone(viewer.glass_brain_plotter)
        self.assertIsNotNone(viewer.time_series_plotter)
        
        # Verify data was loaded
        self.assertIsNotNone(viewer.activation_data)
        self.assertIsNotNone(viewer.time_data)
        self.assertIsNotNone(viewer.brain_outline)

    @patch('src.visualization.build_connection_plot.load_data')
    @patch('src.visualization.build_connection_plot.datasets')
    @patch('src.visualization.build_connection_plot.image')
    def test_update_methods(self, mock_image, mock_datasets, mock_load_data):
        """Test update methods of BrainViewerConnection."""
        # Setup mocks
        mock_load_data.return_value = {
            "data": self.test_data,
            "sfreq": 1000.0,
            "ch_names": self.ch_names,
            "ch_types": self.ch_types,
            "info": self.info,
            "times": self.times,
        }
        
        mock_stat_img = MagicMock()
        mock_image.load_img.return_value = mock_stat_img
        
        mock_brain_mask = MagicMock()
        mock_brain_mask.get_fdata.return_value = self.brain_data
        mock_image.resample_to_img.return_value = mock_brain_mask

        # Create instance
        viewer = BrainViewerConnection()
        
        # Test update_glass_brain
        mock_glass_brain = MagicMock(return_value=go.Figure())
        viewer.glass_brain_plotter.update_glass_brain = mock_glass_brain
        
        fig = viewer.update_glass_brain(time_point=50)
        self.assertIsInstance(fig, go.Figure)
        mock_glass_brain.assert_called_once_with(time_point=50, selected_channel=None)
        
        # Test update with selected channel
        mock_glass_brain.reset_mock()
        fig = viewer.update_glass_brain(time_point=50, selected_channel=(10, 20))
        self.assertIsInstance(fig, go.Figure)
        mock_glass_brain.assert_called_once_with(time_point=50, selected_channel=(10, 20))
        
        # Test update_time_series
        mock_time_series = MagicMock(return_value=go.Figure())
        viewer.time_series_plotter.update_time_series = mock_time_series
        
        fig = viewer.update_time_series(time_point=50)
        self.assertIsInstance(fig, go.Figure)
        mock_time_series.assert_called_once_with(time_point=50, selected_channel=None)
        
        # Test with selected channel
        mock_time_series.reset_mock()
        fig = viewer.update_time_series(time_point=50, selected_channel=(10, 20))
        self.assertIsInstance(fig, go.Figure)
        mock_time_series.assert_called_once_with(time_point=50, selected_channel=(10, 20))

    @patch('src.visualization.build_connection_plot.load_data')
    @patch('src.visualization.build_connection_plot.datasets')
    @patch('src.visualization.build_connection_plot.image')
    def test_create_multi_view_brain(self, mock_image, mock_datasets, mock_load_data):
        """Test create_multi_view_brain method."""
        # Setup mocks
        mock_load_data.return_value = {
            "data": self.test_data,
            "sfreq": 1000.0,
            "ch_names": self.ch_names,
            "ch_types": self.ch_types,
            "info": self.info,
            "times": self.times,
        }
        
        mock_stat_img = MagicMock()
        mock_image.load_img.return_value = mock_stat_img
        
        mock_brain_mask = MagicMock()
        mock_brain_mask.get_fdata.return_value = self.brain_data
        mock_image.resample_to_img.return_value = mock_brain_mask

        # Create instance
        viewer = BrainViewerConnection()
        
        # Test create_multi_view_brain with 2D data format
        fig = viewer.create_multi_view_brain(time_point=50)
        self.assertIsInstance(fig, go.Figure)
        
        # Verify figure has expected structure
        self.assertEqual(len(fig.data), 6)  # 3 outlines + 3 heatmaps
        self.assertEqual(fig.layout.height, 400)
        self.assertEqual(fig.layout.width, 1200)
        self.assertIn("Multi-view Brain Activity", fig.layout.title.text)
        
        # Test with 4D data format
        viewer.activation_data = np.random.randn(53, 63, 46, 100)  # 4D format
        fig = viewer.create_multi_view_brain(time_point=50)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 6)  # 3 outlines + 3 heatmaps

    @patch('src.visualization.build_connection_plot.load_data')
    @patch('src.visualization.build_connection_plot.datasets')
    @patch('src.visualization.build_connection_plot.image')
    def test_callbacks(self, mock_image, mock_datasets, mock_load_data):
        """Test callbacks in BrainViewerConnection."""
        # Setup mocks
        mock_load_data.return_value = {
            "data": self.test_data,
            "sfreq": 1000.0,
            "ch_names": self.ch_names,
            "ch_types": self.ch_types,
            "info": self.info,
            "times": self.times,
        }
        
        mock_stat_img = MagicMock()
        mock_image.load_img.return_value = mock_stat_img
        
        mock_brain_mask = MagicMock()
        mock_brain_mask.get_fdata.return_value = self.brain_data
        mock_image.resample_to_img.return_value = mock_brain_mask

        # Create instance
        viewer = BrainViewerConnection()
        
        # Verify that callbacks were initialized
        self.assertIsNotNone(viewer.app.callback_map)
        self.assertTrue(len(viewer.app.callback_map) > 0, "No callbacks were registered")
        
        # Check for callbacks related to key components
        callback_names = str(viewer.app.callback_map.keys())
        self.assertIn("time-series-plot.figure", callback_names, "Time series plot callback not found")
        self.assertIn("glass-brain-plot.figure", callback_names, "Glass brain plot callback not found")
        self.assertIn("animation-interval", callback_names, "Animation interval callback not found")

    @patch('src.visualization.build_connection_plot.load_data')
    @patch('src.visualization.build_connection_plot.datasets')
    @patch('src.visualization.build_connection_plot.image')
    def test_run_server(self, mock_image, mock_datasets, mock_load_data):
        """Test run_server method."""
        # Setup mocks
        mock_load_data.return_value = {
            "data": self.test_data,
            "sfreq": 1000.0,
            "ch_names": self.ch_names,
            "ch_types": self.ch_types,
            "info": self.info,
            "times": self.times,
        }

        mock_stat_img = MagicMock()
        mock_image.load_img.return_value = mock_stat_img

        mock_brain_mask = MagicMock()
        mock_brain_mask.get_fdata.return_value = self.brain_data
        mock_image.resample_to_img.return_value = mock_brain_mask

        # Create instance
        viewer = BrainViewerConnection()

        # Mock the app.run method to prevent actual server start
        with patch.object(viewer.app, 'run') as mock_run:
            viewer.run_server(debug=False)
            mock_run.assert_called_once_with(port=viewer.port, debug=False)

            # Test retry logic by simulating a port conflict
            mock_run.side_effect = [OSError("Port in use"), None]
            mock_run.reset_mock()

            # Create OSError to trigger retry logic
            with patch('src.visualization.build_connection_plot.find_free_port', return_value=viewer.port + 1):
                viewer.run_server(debug=True)

                # Since we mocked find_free_port to return a different port,
                # a second call should happen with the new port
                self.assertEqual(mock_run.call_count, 2)


if __name__ == "__main__":
    unittest.main() 