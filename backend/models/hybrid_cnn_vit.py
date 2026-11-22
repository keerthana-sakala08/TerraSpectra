
from pathlib import Path

import h5py
import torch
import torch.nn as nn

import sys
from pathlib import Path

# Add the project root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.cnn3d_model import TerraSpectra3DCNN

from backend.models.vit_components import (
    HyperspectralPatchEmbedding,
    PositionalEncoding,
)
# ==========================================
# TERRASPECTRA - DAY 13
# HYBRID 3D-CNN + VISION TRANSFORMER
# ==========================================

NUM_CLASSES = 2
NUM_BANDS = 224
PATCH_SIZE = 5
EMBEDDING_DIM = 128


class TerraSpectraHybridCNNViT(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        # ======================================
        # BRANCH 1: 3D-CNN FEATURE EXTRACTOR
        # ======================================

        self.cnn = TerraSpectra3DCNN(
            num_classes=num_classes
        )

        # Remove the original classification layer.
        # Keep the CNN feature extraction layers.
        self.cnn.classifier = nn.Identity()

        # ======================================
        # BRANCH 2: VISION TRANSFORMER
        # ======================================

        self.patch_embedding = HyperspectralPatchEmbedding(
            num_bands=NUM_BANDS,
            patch_size=PATCH_SIZE,
            embedding_dim=EMBEDDING_DIM
        )

        self.positional_encoding = PositionalEncoding(
            num_tokens=PATCH_SIZE * PATCH_SIZE,
            embedding_dim=EMBEDDING_DIM
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIM,
            nhead=4,
            dim_feedforward=EMBEDDING_DIM * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.vit_norm = nn.LayerNorm(EMBEDDING_DIM)

        # ======================================
        # FEATURE FUSION
        # CNN features = 64
        # ViT features = 128
        # Combined = 192
        # ======================================

        self.fusion = nn.Sequential(

            nn.Linear(64 + EMBEDDING_DIM, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    # ==========================================
    # FORWARD PASS
    # ==========================================

    def forward(self, x):

        if x.ndim != 5:
            raise ValueError(
                "Expected input shape (B, 1, 224, 5, 5)."
            )

        if tuple(x.shape[1:]) != (
            1, NUM_BANDS, PATCH_SIZE, PATCH_SIZE
        ):
            raise ValueError(
                "Incorrect input dimensions. "
                "Expected (B, 1, 224, 5, 5)."
            )

        # ======================================
        # CNN FEATURE EXTRACTION
        # ======================================

        cnn_features = self.cnn.conv_block1(x)

        cnn_features = self.cnn.conv_block2(
            cnn_features
        )

        cnn_features = self.cnn.conv_block3(
            cnn_features
        )

        cnn_features = self.cnn.global_pool(
            cnn_features
        )

        cnn_features = torch.flatten(
            cnn_features,
            start_dim=1
        )

        # Output: (B, 64)

        # ======================================
        # VIT FEATURE EXTRACTION
        # ======================================

        tokens = self.patch_embedding(x)

        tokens = self.positional_encoding(tokens)

        tokens = self.transformer(tokens)

        tokens = self.vit_norm(tokens)

        # CLS token represents the complete patch.
        vit_features = tokens[:, 0, :]

        # Output: (B, 128)

        # ======================================
        # FEATURE FUSION
        # ======================================

        combined_features = torch.cat(
            [cnn_features, vit_features],
            dim=1
        )

        # Output: (B, 192)

        # ======================================
        # CLASSIFICATION
        # ======================================

        output = self.fusion(combined_features)

        # Output: (B, 2)

        return output


# ==========================================
# DAY 13 MODEL TESTING
# ==========================================

def test_hybrid_model():

    print("\n======================================")
    print(" TERRASPECTRA DAY 13")
    print(" HYBRID 3D-CNN + VIT MODEL")
    print("======================================")

    # Locate project root.
    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "datasets"
        / "processed"
        / "cnn_training_dataset.h5"
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}"
        )

    # ======================================
    # LOAD SAMPLE DATA
    # ======================================

    print("\nStep 1: Loading training samples...")

    with h5py.File(dataset_path, "r") as file:

        sample_data = file["X_train"][:4]

        sample_labels = file["y_train"][:4]

    x = torch.tensor(
        sample_data,
        dtype=torch.float32
    )

    y = torch.tensor(
        sample_labels,
        dtype=torch.long
    ).reshape(-1)

    print("Input shape:", tuple(x.shape))
    print("Labels shape:", tuple(y.shape))

    # ======================================
    # CREATE HYBRID MODEL
    # ======================================

    print("\nStep 2: Creating hybrid model...")

    model = TerraSpectraHybridCNNViT()

    model.eval()

    # ======================================
    # TEST CNN BRANCH
    # ======================================

    print("\nStep 3: Testing CNN branch...")

    with torch.no_grad():

        cnn_features = model.cnn.conv_block1(x)

        cnn_features = model.cnn.conv_block2(
            cnn_features
        )

        cnn_features = model.cnn.conv_block3(
            cnn_features
        )

        cnn_features = model.cnn.global_pool(
            cnn_features
        )

        cnn_features = torch.flatten(
            cnn_features,
            start_dim=1
        )

    print(
        "CNN feature shape:",
        tuple(cnn_features.shape)
    )

    assert cnn_features.shape == (4, 64)

    # ======================================
    # TEST VIT BRANCH
    # ======================================

    print("\nStep 4: Testing ViT branch...")

    with torch.no_grad():

        tokens = model.patch_embedding(x)

        positioned_tokens = model.positional_encoding(
            tokens
        )

        transformed_tokens = model.transformer(
            positioned_tokens
        )

        transformed_tokens = model.vit_norm(
            transformed_tokens
        )

        vit_features = transformed_tokens[:, 0, :]

    print(
        "ViT feature shape:",
        tuple(vit_features.shape)
    )

    assert vit_features.shape == (4, 128)

    # ======================================
    # TEST FEATURE FUSION AND OUTPUT
    # ======================================

    print("\nStep 5: Testing feature fusion...")

    combined_features = torch.cat(
        [cnn_features, vit_features],
        dim=1
    )

    print(
        "Combined feature shape:",
        tuple(combined_features.shape)
    )

    assert combined_features.shape == (4, 192)

    with torch.no_grad():

        output = model(x)

    print(
        "Hybrid model output shape:",
        tuple(output.shape)
    )

    assert output.shape == (4, NUM_CLASSES)

    assert torch.isfinite(output).all()

    # ======================================
    # TEST GRADIENT FLOW
    # ======================================

    print("\nStep 6: Testing gradient flow...")

    model.train()

    model.zero_grad()

    train_output = model(x)

    loss = nn.CrossEntropyLoss()(
        train_output,
        y
    )

    loss.backward()

    # CNN branch gradient
    cnn_gradient = (
        model.cnn.conv_block1[0].weight.grad
    )

    # ViT branch gradient
    vit_gradient = (
        model.patch_embedding.projection.weight.grad
    )

    assert cnn_gradient is not None

    assert vit_gradient is not None

    assert torch.isfinite(cnn_gradient).all()

    assert torch.isfinite(vit_gradient).all()

    print("CNN branch gradient: PASSED")

    print("ViT branch gradient: PASSED")

    # ======================================
    # MODEL INFORMATION
    # ======================================

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nTraining loss:", round(loss.item(), 4))

    print(
        "Trainable parameters:",
        total_parameters
    )

    # ======================================
    # FINAL VALIDATION
    # ======================================

    print("\n======================================")
    print(" HYBRID MODEL VALIDATION PASSED")
    print("======================================")

    print("CNN feature extraction: PASSED")

    print("ViT feature extraction: PASSED")

    print("Feature fusion: PASSED")

    print("Classification output: PASSED")

    print("Gradient flow: PASSED")

    print("\nDAY 13 COMPLETED SUCCESSFULLY!")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    test_hybrid_model()
