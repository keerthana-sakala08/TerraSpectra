
# TerraSpectra - Day 11
## Vision Transformer (ViT) Architecture

### Objective
Develop a Vision Transformer model for hyperspectral crop analysis.

### Work Completed
1. Created a Vision Transformer architecture using PyTorch.
2. Converted spatial hyperspectral patches into tokens.
3. Implemented linear token embedding.
4. Added learnable classification and positional embeddings.
5. Implemented Transformer encoder layers.
6. Added a classification head for crop health classification.
7. Tested the architecture using sample training data.

### Model Configuration
- Input bands: 224
- Spatial patch size: 5 x 5
- Number of spatial tokens: 25
- Embedding dimension: 128
- Attention heads: 4
- Transformer encoder layers: 2
- Output classes: 2

### Input Shape
(Batch, 1, 224, 5, 5)

### Output Shape
(Batch, 2)

### Classes
- 0: Healthy
- 1: Stressed

### Important Note
This architecture is an initial development model.
Its performance must be evaluated through training
and testing before making real-world claims.

### Status
Day 11 Vision Transformer architecture completed.
