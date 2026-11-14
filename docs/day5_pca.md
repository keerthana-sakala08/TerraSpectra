# 

# \# Day 5 - Principal Component Analysis (PCA)

# 

# \## Objective

# Reduce the spectral dimensions of the hyperspectral

# dataset using Principal Component Analysis.

# 

# \## Work Completed

# 1\. Loaded the preprocessed hyperspectral dataset.

# 2\. Converted image pixels into a feature matrix.

# 3\. Applied PCA using scikit-learn.

# 4\. Reduced 224 spectral bands to 3 components.

# 5\. Calculated explained variance ratios.

# 6\. Validated the PCA output.

# 7\. Saved PCA features and model information.

# 

# \## Input Dataset

# \- Image dimensions: 128 x 128

# \- Original spectral bands: 224

# \- Pixel feature matrix: 16384 x 224

# 

# \## PCA Output

# \- Reduced dimensions: 128 x 128 x 3

# \- Number of components: 3

# \- Labels preserved: Healthy (0), Stressed (1)

# 

# \## Output File

# datasets/processed/pca\_hyperspectral.h5

# 

# \## Outcome

# The hyperspectral data was transformed from 224

# spectral features into 3 principal components.

# 

# The explained variance ratio indicates how much

# of the original data variation is represented

# by the selected components.

# 

# Note: PCA is a dimensionality-reduction technique.

# It does not independently identify crop diseases.

# The current dataset is synthetic.

# 

