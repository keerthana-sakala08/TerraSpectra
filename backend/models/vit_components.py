
from pathlib import Path

import h5py
import torch
import torch.nn as nn


# --------------------------------------------------
# 1. Patch Embedding
# --------------------------------------------------

class HyperspectralPatchEmbedding(nn.Module):

    def __init__(
        self,
        num_bands=224,
        patch_size=5,
        embedding_dim=128
    ):
        super().__init__()

        self.num_bands = num_bands
        self.patch_size = patch_size
        self.num_tokens = patch_size * patch_size

        self.projection = nn.Linear(
            num_bands,
            embedding_dim
        )

    def forward(self, x):

        # Input: (B, 1, 224, 5, 5)

        if x.ndim != 5:
            raise ValueError("Expected a 5-dimensional input.")

        batch_size, channels, bands, height, width = x.shape

        if channels != 1:
            raise ValueError("Expected one input channel.")

        if bands != self.num_bands:
            raise ValueError("Incorrect number of spectral bands.")

        if height != self.patch_size or width != self.patch_size:
            raise ValueError("Incorrect spatial patch size.")

        # Rearrange to (B, 5, 5, 224)
        x = x.squeeze(1).permute(0, 2, 3, 1)

        # Flatten spatial positions into 25 tokens
        x = x.reshape(
            batch_size,
            self.num_tokens,
            self.num_bands
        )

        # Project each token to 128 dimensions
        x = self.projection(x)

        return x


# --------------------------------------------------
# 2. Positional Encoding
# --------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        num_tokens=25,
        embedding_dim=128
    ):
        super().__init__()

        # Learnable classification token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embedding_dim)
        )

        # 25 patch tokens + 1 classification token
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_tokens + 1, embedding_dim)
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):

        batch_size = x.shape[0]

        # Add classification token
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        # Add positional information
        x = x + self.position_embedding

        return x


# --------------------------------------------------
# 3. Multi-Head Self-Attention
# --------------------------------------------------

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim=128,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Self-attention: query, key and value are the same tokens
        attention_output, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False
        )

        # Residual connection and normalization
        output = self.layer_norm(
            x + self.dropout(attention_output)
        )

        return output, attention_weights


# --------------------------------------------------
# 4. Test all three components
# --------------------------------------------------

def test_components():

    print("\nTesting TerraSpectra ViT Components")

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "datasets/processed/cnn_training_dataset.h5"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}"
        )

    # Load four real training patches
    with h5py.File(dataset_path, "r") as file:

        x = torch.tensor(
            file["X_train"][:4],
            dtype=torch.float32
        )

    print("\nOriginal input shape:", tuple(x.shape))

    # Component 1: Patch embedding
    patch_embedding = HyperspectralPatchEmbedding()

    tokens = patch_embedding(x)

    print("After patch embedding:", tuple(tokens.shape))

    assert tokens.shape == (4, 25, 128)

    # Component 2: Positional encoding
    positional_encoding = PositionalEncoding()

    positioned_tokens = positional_encoding(tokens)

    print("After positional encoding:",
          tuple(positioned_tokens.shape))

    assert positioned_tokens.shape == (4, 26, 128)

    # Component 3: Multi-head self-attention
    self_attention = MultiHeadSelfAttention()

    attention_output, attention_weights = self_attention(
        positioned_tokens
    )

    print("After self-attention:",
          tuple(attention_output.shape))

    print("Attention weights shape:",
          tuple(attention_weights.shape))

    assert attention_output.shape == (4, 26, 128)
    assert attention_weights.shape == (4, 4, 26, 26)

    assert torch.isfinite(attention_output).all()
    assert torch.isfinite(attention_weights).all()

    print("\nAll three components passed!")

    print("\nDAY 12 TEST COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    test_components()
