
# TerraSpectra - Day 12
## ViT Patch Embedding, Positional Encoding and Self-Attention

### Objective
Implement and test the core components of a Vision Transformer.

### Work Completed
1. Implemented hyperspectral patch embedding.
2. Converted 5 x 5 spatial patches into 25 tokens.
3. Projected spectral features into 128-dimensional embeddings.
4. Added a learnable classification token.
5. Implemented positional embeddings.
6. Implemented four-head self-attention.
7. Added residual connection and layer normalization.
8. Tested tensor dimensions and attention weights.

### Model Configuration
- Spectral bands: 224
- Spatial patch size: 5 x 5
- Spatial tokens: 25
- Embedding dimension: 128
- Classification token: 1
- Attention heads: 4

### Verified Output Shapes
- Input: (4, 1, 224, 5, 5)
- Patch embedding: (4, 25, 128)
- Positional encoding: (4, 26, 128)
- Self-attention output: (4, 26, 128)
- Attention weights: (4, 4, 26, 26)

### Status
Day 12 ViT components implemented and tested.
