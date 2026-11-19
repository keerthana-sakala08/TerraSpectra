

*# TerraSpectra - Day 9*

*## Crop Health Prediction and Heatmap Generation*



*### Objective*

*Use the trained 3D-CNN model to classify hyperspectral*

*image pixels into Healthy and Stressed categories.*



*### Work Completed*

*1. Loaded the preprocessed hyperspectral cube.*

*2. Loaded the trained 3D-CNN model.*

*3. Extracted 5 x 5 patches for each image pixel.*

*4. Predicted Healthy (0) and Stressed (1) classes.*

*5. Generated a pixel-level prediction map.*

*6. Calculated predicted healthy and stressed percentages.*

*7. Generated a stress probability heatmap.*

*8. Saved prediction results in HDF5 format.*



*### Input Information*

*- Hyperspectral cube: 128 x 128 x 224*

*- Spectral bands: 224*

*- Patch size: 5 x 5*

*- Total pixels: 16,384*



*### Generated Outputs*

*- outputs/day9/crop\_health\_predictions.h5*

*- outputs/day9/crop\_health\_map.png*

*- outputs/day9/stress\_probability\_map.png*



*### Model*

*TerraSpectra3DCNN (PyTorch)*



*### Class Labels*

*- 0: Healthy*

*- 1: Stressed*



*### Important Note*

*The predictions are based on synthetic development data.*

*They are not validated real-world crop disease diagnoses*

*or measured crop damage percentages.*



*### Status*

*Day 9 prediction and heatmap generation completed.*



