"""Test cases for core data processing functions."""

import unittest
import numpy as np
import pandas as pd
from scipy import stats
import mne
from typing import Dict, Any

from .data_processing import compute_statistics, validate_input

class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create sample EEG data
        n_channels = 4
        n_times = 1000
        sfreq = 1000.0
        
        # Create random EEG data
        cls.test_data = np.random.randn(n_channels, n_times)
        cls.ch_names = [f'ch{i+1}' for i in range(n_channels)]
        cls.ch_types = ['eeg'] * n_channels
        
        # Create MNE info object
        cls.info = mne.create_info(
            ch_names=cls.ch_names,
            sfreq=sfreq,
            ch_types=cls.ch_types
        )
        
        # Create data dictionary
        cls.data = {
            'data': cls.test_data,
            'sfreq': sfreq,
            'ch_names': cls.ch_names,
            'ch_types': cls.ch_types,
            'info': cls.info
        }
    
    def test_analyze_statistics(self):
        """Test statistical analysis function."""
        # Test with default parameters
        stats_results = compute_statistics(self.data, methods=['descriptive', 'ttest', 'anova', 'correlation', 'permutation'])
        
        # Check if all required statistics are present
        self.assertIn('descriptive', stats_results)
        
        # Check descriptive statistics
        desc_stats = stats_results['descriptive']
        self.assertIn('mean', desc_stats)
        self.assertIn('std', desc_stats)
        self.assertIn('median', desc_stats)
        
        # Check shapes
        self.assertEqual(desc_stats['mean'].shape, (self.test_data.shape[0],))
        self.assertEqual(desc_stats['std'].shape, (self.test_data.shape[0],))
        
        # Test with specific time window
        window_stats = compute_statistics(
            self.data,
            time_window=(0.1, 0.3),
            metrics=['mean', 'variance']
        )
        self.assertIn('mean', window_stats['descriptive'])
        self.assertIn('skewness', window_stats['descriptive'])
        self.assertIn('kurtosis', window_stats['descriptive'])

    def test_validate_data_format(self):
        """Test data format validation."""
        # Create valid data dictionary with 4D brain data
        valid_data = {
            'data': np.random.randn(53, 63, 46, 1000),  # Using 4D brain activity shape
            'sfreq': self.info['sfreq'],
            'ch_names': self.ch_names,
            'ch_types': self.ch_types,
            'info': self.info
        }

        # For the data validation test, we need to use 2D format temporarily
        # since that's what the validate_data_format function expects
        test_data_2d = self.test_data.copy()

        # Test valid data with 2D format for validation function
        valid_data_2d = valid_data.copy()
        valid_data_2d['data'] = test_data_2d
        is_valid, message = validate_input(valid_data_2d)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Data format is valid")

        # Test missing key
        invalid_data = valid_data_2d.copy()
        del invalid_data['data']
        is_valid, message = validate_input(invalid_data)
        self.assertFalse(is_valid)
        self.assertEqual(message, "Missing required key: data")

        # Test invalid data shape - using 3D instead of 2D
        invalid_data = valid_data_2d.copy()
        invalid_data['data'] = np.random.randn(3, 4, 5)  # 3D array
        is_valid, message = validate_input(invalid_data)
        self.assertFalse(is_valid)
        self.assertIn("Data must be 2D", message)
if __name__ == '__main__':
    unittest.main() 