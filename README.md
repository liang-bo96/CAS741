# McMaster EEG Visualization Project

A Python package for visualizing and analyzing EEG data using modern visualization techniques.

## Features

- Interactive time series plots
- Topographic maps
- Glass brain visualizations
- Butterfly plots
- Statistical analysis
- Connection analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/liang-bo96/mcmaster-eeg-visualization.git
cd mcmaster-eeg-visualization
```

2. Run the installation script:
```bash
./install.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

## Project Structure

```
mcmaster-eeg-visualization/
├── src/
│   ├── input_format/      # Data loading and preprocessing
│   │   ├── data_loader.py     # Functions for loading EEG data
│   │   ├── format_converter.py # Convert between data formats
│   │   └── preprocessor.py     # Data preprocessing functions
│   ├── visualization/     # Visualization components
│   │   ├── butterfly_plotter.py    # Butterfly plot visualization
│   │   ├── glassbrain_plotter.py   # Glass brain visualization
│   │   ├── time_series_plotter.py  # Time series visualization
│   │   ├── topo_plotter.py         # Topographic map visualization
│   │   └── build_connection_plot.py # Connection analysis visualization
│   └── analysis/         # Statistical analysis
│       ├── statistical_analyzer.py  # Statistical analysis functions
│       └── connectivity_analyzer.py # Connectivity analysis functions
├── tests/                # Test files
│   ├── test_input_format.py    # Tests for input format module
│   ├── test_visualization.py   # Tests for visualization module
│   └── test_analysis.py        # Tests for analysis module
├── install.sh           # Installation script
├── setup.py            # Package configuration
└── README.md           # This file
```

## Usage Examples

### Basic Visualization

```python
from src.visualization import plot_butterfly, plot_topography, plot_glassbrain
from src.input_format import load_eeg_data

# Load EEG data
data = load_eeg_data('path_to_eeg_file')

# Create visualizations
butterfly_plot = plot_butterfly(data)
topo_plot = plot_topography(data)
glass_brain = plot_glassbrain(data)
```
### Building Connection
visualization.build_connection_plot.main()

## Testing

Run the test suite:
```bash
python -m pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNE-Python for EEG data handling
- Plotly for interactive visualizations
- Nilearn for brain visualization tools