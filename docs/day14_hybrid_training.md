# Day 14: Hybrid CNN-ViT Model Training and Evaluation

## Objective
To train and evaluate the Hybrid 3D-CNN + Vision Transformer model and compare its performance with the standalone 3D-CNN model.

## Work Completed
- Implemented the hybrid CNN-ViT training pipeline.
- Loaded the prepared hyperspectral training and testing datasets.
- Used a stratified training and validation split.
- Trained the hybrid model using the Adam optimizer.
- Evaluated the hybrid model on the test dataset.
- Compared its performance with the existing 3D-CNN model.

## Model Architecture
The hybrid model combines:
- **3D-CNN branch:** Extracts spatial and spectral features (64 features).
- **ViT branch:** Captures global relationships using transformer attention (128 features).
- **Feature fusion:** Combines both branches into a 192-dimensional feature representation.
- **Classifier:** Predicts the crop condition.

## Training Configuration
- Optimizer: Adam
- Learning rate: 0.0005
- Batch size: 16
- Epochs: 8
- Loss function: CrossEntropyLoss
- Validation split: 20%
- Random seed: 42

## Evaluation
The hybrid model and standalone 3D-CNN were evaluated using the same test dataset. Accuracy, precision, recall, and F1-score were calculated. The evaluation results are stored in `outputs/day14_hybrid_metrics.json`.

## Output Files
- `models/terraspectra_hybrid_cnn_vit.pth`
- `outputs/day14_hybrid_metrics.json`

## Limitations
The current dataset is synthetic and contains healthy and stressed crop classes. The results do not represent real-world disease detection performance. Real labelled agricultural data and field validation are required.

## Conclusion
The hybrid CNN-ViT training and evaluation pipeline was implemented. The project is ready for the next stage: crop-health analytics and dashboard development.