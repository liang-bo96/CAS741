# McMaster EEG Visualization Project

This project provides tools for EEG data visualization and analysis, including various plotting capabilities for brain data, time series, and topographic maps.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment support (venv)

## Installation

### Automatic Installation (Recommended)

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Run the installation script:
```bash
./install.sh
```

This script will:
- Check Python version
- Create a virtual environment
- Install all dependencies
- Run tests to verify installation

### Manual Installation

If you prefer to install manually:

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install the package:
```bash
pip install -e .
```

## Usage

After installation, activate the virtual environment:
```bash
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

The package provides various visualization tools:
- Glass brain plotting
- Time series visualization
- Topographic maps
- Butterfly plots

## Project Structure

The project is organized as follows:
- `src/visualization/`: Contains visualization modules
- `src/data_processing/`: Data processing utilities
- `src/input_format/`: Input format handlers

## Testing

To run the tests:
```bash
python -m pytest src/visualization/test_visualization.py -v
```

## Dependencies

The project requires the following main dependencies:
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 0.24.0
- mne >= 0.24.0
- plotly >= 5.18.0
- dash >= 2.14.0
- pandas >= 1.5.0
- pywavelets >= 1.1.1
