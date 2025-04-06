"""Test cases for the input_format module."""

import unittest
import numpy as np
import pandas as pd
import mne
import os
from pathlib import Path
import tempfile
import shutil

from data_loader import load_eeg_data, validate_data_format
from format_converter import convert_to_dataframe, convert_to_mne
from preprocessor import preprocess_data

class TestInputFormat(unittest.TestCase):
    """Test cases for input_format module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        
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
        
        # Create MNE Raw object
        cls.raw = mne.io.RawArray(cls.test_data, cls.info)
        
        # Save test files
        cls.fif_path = os.path.join(cls.temp_dir, 'test.fif')
        cls.csv_path = os.path.join(cls.temp_dir, 'test.csv')
        
        # Save as FIF
        cls.raw.save(cls.fif_path, overwrite=True)
        
        # Save as CSV
        df = pd.DataFrame(cls.test_data.T, columns=cls.ch_names)
        df.to_csv(cls.csv_path, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        shutil.rmtree(cls.temp_dir)
    
    def test_load_eeg_data_fif(self):
        """Test loading EEG data from FIF file."""
        data = load_eeg_data(self.fif_path)
        
        # Check data structure
        self.assertIn('data', data)
        self.assertIn('sfreq', data)
        self.assertIn('ch_names', data)
        self.assertIn('ch_types', data)
        self.assertIn('info', data)
        
        # Check data shape
        self.assertEqual(data['data'].shape, self.test_data.shape)
        
        # Check channel names
        self.assertEqual(data['ch_names'], self.ch_names)
        
        # Check sampling frequency
        self.assertEqual(data['sfreq'], self.info['sfreq'])
    
    def test_load_eeg_data_csv(self):
        """Test loading EEG data from CSV file."""
        data = load_eeg_data(self.csv_path)
        
        # Check data structure
        self.assertIn('data', data)
        self.assertIn('sfreq', data)
        self.assertIn('ch_names', data)
        self.assertIn('ch_types', data)
        self.assertIn('info', data)
        
        # Check data shape
        self.assertEqual(data['data'].shape, self.test_data.shape)
        
        # Check channel names
        self.assertEqual(data['ch_names'], self.ch_names)
    
    def test_load_eeg_data_invalid_file(self):
        """Test loading EEG data from invalid file."""
        invalid_path = os.path.join(self.temp_dir, 'invalid.txt')
        with open(invalid_path, 'w') as f:
            f.write('invalid data')
        
        with self.assertRaises(ValueError):
            load_eeg_data(invalid_path)
    
    def test_validate_data_format(self):
        """Test data format validation."""
        # Create valid data dictionary
        valid_data = {
            'data': self.test_data,
            'sfreq': self.info['sfreq'],
            'ch_names': self.ch_names,
            'ch_types': self.ch_types,
            'info': self.info
        }
        
        # Test valid data
        is_valid, message = validate_data_format(valid_data)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Data format is valid")
        
        # Test missing key
        invalid_data = valid_data.copy()
        del invalid_data['data']
        is_valid, message = validate_data_format(invalid_data)
        self.assertFalse(is_valid)
        self.assertEqual(message, "Missing required key: data")
        
        # Test invalid data shape
        invalid_data = valid_data.copy()
        invalid_data['data'] = np.random.randn(3, 4, 5)  # 3D array
        is_valid, message = validate_data_format(invalid_data)
        self.assertFalse(is_valid)
        self.assertIn("Data must be 2D", message)
    
    def test_convert_to_dataframe(self):
        """Test conversion to DataFrame format."""
        data = {
            'data': self.test_data,
            'sfreq': self.info['sfreq'],
            'ch_names': self.ch_names,
            'ch_types': self.ch_types,
            'info': self.info
        }
        
        df = convert_to_dataframe(data)
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (self.test_data.shape[1], self.test_data.shape[0]))
        self.assertEqual(list(df.columns), self.ch_names)
        self.assertEqual(df.index.name, 'time')
    
    def test_convert_to_mne(self):
        """Test conversion to MNE format."""
        data = {
            'data': self.test_data,
            'sfreq': self.info['sfreq'],
            'ch_names': self.ch_names,
            'ch_types': self.ch_types,
            'info': self.info
        }
        
        raw = convert_to_mne(data)
        
        # Check MNE Raw object
        self.assertIsInstance(raw, mne.io.RawArray)
        self.assertEqual(raw.ch_names, self.ch_names)
        self.assertEqual(raw.info['sfreq'], self.info['sfreq'])
        np.testing.assert_array_equal(raw.get_data(), self.test_data)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        data = {
            'data': self.test_data,
            'sfreq': self.info['sfreq'],
            'ch_names': self.ch_names,
            'ch_types': self.ch_types,
            'info': self.info
        }
        
        # Test notch filtering
        processed = preprocess_data(data, notch_freq=50.0)
        self.assertEqual(processed['data'].shape, self.test_data.shape)
        
        # Test bandpass filtering
        processed = preprocess_data(data, bandpass_freq=(1.0, 40.0))
        self.assertEqual(processed['data'].shape, self.test_data.shape)
        
        # Test ICA
        # processed = preprocess_data(data, ica=True, ica_n_components=3)
        # self.assertEqual(processed['data'].shape[0], 3)  # Number of components
        
        # Test bad channel handling
        # Add some bad channels
        # bad_data = data.copy()
        # bad_data['data'][0, :] = np.nan
        # processed = preprocess_data(bad_data, remove_bad_channels=True)
        # self.assertEqual(processed['data'].shape[0], self.test_data.shape[0] - 1)

if __name__ == '__main__':
    unittest.main() 