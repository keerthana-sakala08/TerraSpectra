

\# TerraSpectra - Day 6

\## Labelled Dataset Preparation for 3D-CNN



\### Objective

Prepare labelled hyperspectral image patches for training and testing a 3D-CNN model.



\### Work Completed

1\. Loaded the preprocessed hyperspectral dataset.

2\. Validated cube dimensions and labels.

3\. Extracted 5 x 5 spatial patches.

4\. Prepared Healthy (0) and Stressed (1) classes.

5\. Balanced the dataset with 1500 samples per class.

6\. Split the dataset into 80% training and 20% testing.

7\. Saved the prepared dataset in HDF5 format.



\### Dataset Information



\- Original cube: 128 x 128 x 224

\- Patch size: 5 x 5

\- Spectral bands: 224

\- Total samples: 3000

\- Training samples: 2400

\- Testing samples: 600



\### Class Distribution



| Dataset | Healthy | Stressed |

|---|---:|---:|

| Training | 1200 | 1200 |

| Testing | 300 | 300 |



\### Tensor Format

Samples x Channels x Spectral Bands x Height x Width



Training shape: (2400, 1, 224, 5, 5)



Testing shape: (600, 1, 224, 5, 5)



\### Output

datasets/processed/cnn\_training\_dataset.h5



\### Important Note

This is a synthetic development dataset.

It has not been validated using real-world crop observations.

The random patch split may include spatially related samples

in both training and testing sets. Spatially separated

validation is needed for reliable real-world evaluation.



\### Status

Day 6 completed successfully.



