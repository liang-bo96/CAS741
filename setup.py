from setuptools import setup, find_packages

setup(
    name="mcmaster-eeg-visualization",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "mne>=0.24.0",
        "plotly>=5.18.0",
        "dash>=2.14.0",
        "pandas>=1.5.0",
        "pywavelets>=1.1.1",
    ],
    python_requires=">=3.8",
    author="McMaster EEG Visualization Team",
    description="EEG data visualization and analysis tools",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 