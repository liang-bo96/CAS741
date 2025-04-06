#!/bin/bash

# Exit on error
set -e

# Print error message and exit
error_msg() {
    echo "Error: $1" >&2
    exit 1
}

# Print status message
status_msg() {
    echo "$1"
}

# Check Python version
status_msg "Installing McMaster EEG Visualization Project..."
if ! command -v python3 &> /dev/null; then
    error_msg "Python 3 is required but not installed"
fi

python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    status_msg "Python version $python_version detected"
else
    error_msg "Python 3.8 or higher is required. Found version $python_version"
fi

# Create virtual environment
status_msg "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
status_msg "Upgrading pip..."
python -m pip install --upgrade pip

# Install package dependencies
status_msg "Installing package dependencies..."
pip install -e .

# Install development dependencies
status_msg "Installing development dependencies..."
pip install pytest

status_msg "Installation completed successfully!"
status_msg "To activate the virtual environment, run: source venv/bin/activate" 