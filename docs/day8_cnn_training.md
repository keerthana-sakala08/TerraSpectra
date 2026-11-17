

*# TerraSpectra - Day 8*

*## Training the 3D-CNN Model*



*### Objective*

*Train the 3D-CNN model using the prepared hyperspectral*

*image patches with PyTorch.*



*### Work Completed*

*1. Loaded the Day 6 labelled dataset.*

*2. Converted NumPy arrays into PyTorch tensors.*

*3. Split training data into training and validation sets.*

*4. Used Cross Entropy Loss for classification.*

*5. Used the Adam optimizer.*

*6. Trained the model for 8 epochs.*

*7. Monitored training and validation loss and accuracy.*

*8. Saved the best model based on validation accuracy.*

*9. Evaluated the model using the separate test dataset.*



*### Dataset Information*

*- Total original patches: 3000*

*- Training samples: 1920*

*- Validation samples: 480*

*- Testing samples: 600*

*- Classes: Healthy (0), Stressed (1)*



*### Model Configuration*

*- Architecture: 3D-CNN*

*- Framework: PyTorch*

*- Optimizer: Adam*

*- Learning rate: 0.001*

*- Batch size: 16*

*- Epochs: 8*

*- Loss function: Cross Entropy Loss*



*### Evaluation Metrics*

*Accuracy, Precision, Recall, F1 Score and Confusion Matrix*

*are calculated and saved in the training metrics JSON file.*



*### Generated Files*

*- models/terraspectra\_3dcnn.pth*

*- outputs/day8\_training\_metrics.json*



*### Important Note*

*The model was trained using synthetic hyperspectral data.*

*The results are preliminary and do not represent validated*

*real-world crop disease detection performance.*



*### Status*

*Day 8 training and test evaluation completed.*



