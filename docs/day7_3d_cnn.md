

*# TerraSpectra - Day 7*

*## 3D-CNN Model Architecture*



*### Objective*

*Build a 3D Convolutional Neural Network using PyTorch*

*for hyperspectral crop classification.*



*### Work Completed*

*1. Created the TerraSpectra3DCNN model.*

*2. Implemented three 3D convolution blocks.*

*3. Added Batch Normalization and ReLU.*

*4. Added Max Pooling layers.*

*5. Implemented Adaptive Average Pooling.*

*6. Added Dropout and Linear classification layers.*

*7. Tested the model using Day 6 sample patches.*



*### Model Input*

*- Channels: 1*

*- Spectral bands: 224*

*- Patch height: 5*

*- Patch width: 5*



*### Model Output*

*- Healthy: Class 0*

*- Stressed: Class 1*

*- Two output scores per sample.*



*### Architecture*

*Input -> Conv3D -> BatchNorm -> ReLU ->*

*MaxPool -> Conv3D -> BatchNorm -> ReLU ->*

*MaxPool -> Conv3D -> Global Average Pooling ->*

*Dropout -> Linear Classification*



*### Dataset*

*datasets/processed/cnn\_training\_dataset.h5*



*### Model File*

*backend/models/cnn3d\_model.py*



*### Important Note*

*This is an initial model architecture tested using*

*synthetic hyperspectral data. Model training and*

*real-world validation have not yet been completed.*



*### Status*

*Day 7 architecture implementation completed.*



