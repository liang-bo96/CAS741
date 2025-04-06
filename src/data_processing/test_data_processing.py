"""Test cases for core data processing functions."""

import unittest
import numpy as np
import pandas as pd
from scipy import stats
import mne
from typing import Dict, Any

from statistical_analyzer import compute_statistics

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
        

if __name__ == '__main__':
    unittest.main() 