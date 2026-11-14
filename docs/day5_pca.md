# TerraSpectra - Day 5

## Principal Component Analysis

### Objective
Reduce hyperspectral spectral dimensions while preserving important information.

### Implemented Features
1. Load preprocessed hyperspectral data.
2. Convert the data cube into a pixel matrix.
3. Apply PCA with three principal components.
4. Calculate explained variance.
5. Save the reduced data in HDF5 format.
6. Validate the output.

### Input
datasets/processed/processed_hyperspectral.h5

### Output
datasets/processed/pca_hyperspectral.h5

### Dimensions
- Original bands: 224
- Reduced components: 3
- Original image: 128 x 128
- Reduced image: 128 x 128 x 3

### Note
The dataset is synthetic and intended for development and testing.
