# McMaster EEG Visualization Project

[![Python Tests](https://github.com/liang-bo96/CAS741/actions/workflows/python-tests.yml/badge.svg)](https://github.com/liang-bo96/CAS741/actions/workflows/python-tests.yml)

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
# Remove existing directory if it exists
rm -rf CAS741
# Clone the repository
git clone git@github.com:liang-bo96/CAS741.git
cd CAS741
```

2. Run the installation script:
```bash
./install.sh
```
3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. Install additional required package:
```bash
pip install nilearn==0.11.0
```



5. To automatically run the visualization function after activating the virtual environment, add the following to your shell's configuration file (e.g., ~/.bashrc, ~/.zshrc, etc.):
```bash
# Run EEG visualization after activating virtual environment
python src/visualization/build_connection_plot.py
```

6. you can see the running result in terminal, just click the address, you will see the plotting result in browser
```
Starting server on port: 64406(a random port)
Dash is running on http://127.0.0.1:64406/
```

## Project Structure

```
CAS741/
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


## Continuous Integration

This project uses GitHub Actions for continuous integration. Every commit to the main or master branch, as well as every pull request, triggers the test suite to run automatically.

### CI/CD Pipeline Features

- Automatic test execution using pytest
- Code coverage reporting
- Integration with Codecov for coverage visualization
- Testing across different Python versions

### Status

The current status of the CI pipeline is shown by the badge at the top of this README. Click on it to see detailed test reports.

### Running Tests Manually

To run the tests and generate coverage reports locally:

```bash
# Run all tests
python -m pytest src

# Run with coverage report
python -m pytest src --cov=src
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