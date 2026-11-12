# TerraSpectra - Day 3 Data Loading Pipeline

## Objective
Build a reusable pipeline to load and validate hyperspectral data.

## Supported Formats
- HDF5
- GeoTIFF

## Features
- Load hyperspectral image cubes.
- Extract spectral wavelengths.
- Extract GIS metadata such as CRS and image dimensions.
- Load healthy and stressed crop labels.
- Validate image dimensions and spectral bands.
- Check missing or invalid values.
- Compare HDF5 and GeoTIFF data for consistency.

## Expected Dataset
- Height: 128 pixels
- Width: 128 pixels
- Spectral bands: 224
- Wavelength range: 400-1000 nm
- Healthy label: 0
- Stressed label: 1

## Main File
backend/data_pipeline/load_hyperspectral_data.py

## Note
The current dataset is synthetic and intended for development and testing.
