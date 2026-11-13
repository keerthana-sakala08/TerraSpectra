# TerraSpectra - Day 4

## Hyperspectral Data Preprocessing

### Objective
Clean and prepare raw hyperspectral data for machine learning.

### Implemented Features
1. Missing value handling using per-band median.
2. Spectral noise reduction using moving-average smoothing.
3. Per-band Min-Max normalization.
4. Data validation.
5. Saving processed data in HDF5 format.

### Input
datasets/sample/hyperspectral_sample.h5

### Output
datasets/processed/processed_hyperspectral.h5

### Dataset Specifications
- Image dimensions: 128 x 128
- Spectral bands: 224
- Data type: Float32
- Normalization range: 0 to 1
- Classes: Healthy (0), Stressed (1)

### Note
The dataset is synthetic and intended for development and testing.
