
from pathlib import Path

import h5py
import torch
import torch.nn as nn


class TerraSpectraViT(nn.Module):

    def __init__(
        self,
        num_bands=224,
        patch_size=5,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()

        self.num_bands = num_bands
        self.patch_size = patch_size
        self.num_tokens = patch_size * patch_size
        self.embedding_dim = embedding_dim

        # 1. Convert each spatial token into an embedding
        self.patch_embedding = nn.Linear(
            num_bands,
            embedding_dim
        )

        # 2. Learnable classification token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embedding_dim)
        )

        # 3. Positional embeddings
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens + 1, embedding_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 5. Final classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.positional_embedding,
            std=0.02
        )

        nn.init.xavier_uniform_(
            self.patch_embedding.weight
        )

        nn.init.zeros_(
            self.patch_embedding.bias
        )

    def forward(self, x):

        # Expected input:
        # (Batch, 1, 224, 5, 5)

        if x.ndim != 5:
            raise ValueError(
                "Input must have 5 dimensions: "
                "(batch, 1, bands, height, width)"
            )

        batch_size, channels, bands, height, width = x.shape

        if channels != 1:
            raise ValueError("Expected one input channel.")

        if bands != self.num_bands:
            raise ValueError("Unexpected number of spectral bands.")

        if height != self.patch_size or width != self.patch_size:
            raise ValueError("Unexpected spatial patch dimensions.")

        # Convert to:
        # (Batch, Height, Width, Bands)
        x = x.squeeze(1).permute(0, 2, 3, 1)

        # Convert spatial positions into tokens:
        # (Batch, 25, 224)
        x = x.reshape(
            batch_size,
            self.num_tokens,
            self.num_bands
        )

        # Project spectral features:
        # (Batch, 25, 128)
        x = self.patch_embedding(x)

        # Add classification token:
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        # Add positional information
        x = x + self.positional_embedding

        x = self.dropout(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Extract classification token
        cls_output = x[:, 0]

        # Predict class scores
        output = self.classifier(cls_output)

        return output


def test_model():

    print("Testing TerraSpectra Vision Transformer")

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "datasets/processed/cnn_training_dataset.h5"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}"
        )

    with h5py.File(dataset_path, "r") as file:

        x = torch.tensor(
            file["X_train"][:4],
            dtype=torch.float32
        )

        labels = torch.tensor(
            file["y_train"][:4],
            dtype=torch.long
        )

    print("Input shape:", tuple(x.shape))
    print("Labels shape:", tuple(labels.shape))

    model = TerraSpectraViT()

    # Forward pass
    predictions = model(x)

    print("Output shape:", tuple(predictions.shape))

    # Calculate sample loss
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(predictions, labels)

    print("Sample loss:", round(loss.item(), 4))

    # Verify outputs
    assert predictions.shape == (4, 2)
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(loss)

    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters()
              if p.requires_grad))

    print("\nDAY 11 ViT TEST PASSED!")


if __name__ == "__main__":
    test_model()
