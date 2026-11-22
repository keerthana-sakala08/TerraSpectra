
# TerraSpectra – Day 13
## Hybrid 3D-CNN + Vision Transformer Integration

### 1. Objective

The objective of Day 13 is to integrate the existing
3D-CNN architecture with Vision Transformer (ViT)
components to create a hybrid deep learning model.

The hybrid model combines spectral-spatial feature
extraction and attention-based feature learning.

### 2. Model Architecture

The hybrid model contains two parallel branches:

**Branch 1 – 3D-CNN**

- Uses the existing TerraSpectra 3D-CNN architecture.
- Extracts spectral and spatial features from hyperspectral
  crop patches.
- Produces a 64-dimensional feature vector.

**Branch 2 – Vision Transformer**

- Uses hyperspectral patch embedding.
- Adds positional encoding.
- Uses a Transformer Encoder with two layers.
- Extracts the CLS token representation.
- Produces a 128-dimensional feature vector.

### 3. Hybrid Architecture Flow

Input Hyperspectral Patch
        |
        |-------------------------|
        |                         |
      3D-CNN                    ViT Branch
        |                         |
  Feature Extraction          Patch Embedding
        |                         |
   64 Features              Positional Encoding
                                  |
                            Transformer Encoder
                                  |
                             CLS Token
                                  |
                            128 Features
        |                         |
        |-------------------------|
                    |
               Feature Fusion
                    |
              192 Features
                    |
              Classification
                    |
             Healthy / Stressed

### 4. Input Specifications

| Parameter | Value |
|---|---|
| Input channels | 1 |
| Spectral bands | 224 |
| Patch height | 5 |
| Patch width | 5 |
| Number of classes | 2 |
| Input tensor | (B, 1, 224, 5, 5) |

### 5. Feature Fusion

The CNN and ViT feature vectors are concatenated.

- CNN feature dimension: 64
- ViT feature dimension: 128
- Combined feature dimension: 192

The combined features are passed through a classification
head to produce two class scores.

### 6. Implementation

File created:

`backend/models/hybrid_cnn_vit.py`

Main model class:

`TerraSpectraHybridCNNViT`

Main components:

1. 3D-CNN feature extractor
2. Hyperspectral patch embedding
3. Positional encoding
4. Transformer Encoder
5. Feature fusion layer
6. Classification layer

### 7. Testing and Validation

The Day 13 test checks:

- Loading hyperspectral training samples.
- CNN feature extraction.
- ViT feature extraction.
- Feature concatenation.
- Classification output dimensions.
- Finite model outputs.
- Gradient flow through CNN and ViT branches.

Expected dimensions:

| Stage | Output shape |
|---|---|
| Input | (4, 1, 224, 5, 5) |
| CNN features | (4, 64) |
| ViT features | (4, 128) |
| Combined features | (4, 192) |
| Classification | (4, 2) |

### 8. Dataset Information

The current training dataset contains synthetic
hyperspectral samples with two classes:

- Healthy (0)
- Stressed (1)

The model is currently an architecture prototype.
It has not yet been trained as a hybrid model.

### 9. Limitations

- The current dataset is synthetic.
- Only Healthy and Stressed classes are included.
- Real-world crop disease detection has not been validated.
- Model performance must be evaluated after training.

### 10. Day 13 Outcome

The hybrid architecture integrates 3D-CNN and ViT
feature extraction into a single classification model.

The next development stage is hybrid model training,
testing, and comparison with the standalone 3D-CNN.

**Status:** Architecture implementation and testing.
