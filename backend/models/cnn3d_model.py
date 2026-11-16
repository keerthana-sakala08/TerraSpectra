
import os
import h5py
import torch
import torch.nn as nn


# ==========================================
# TERRASPECTRA - DAY 7
# 3D-CNN MODEL ARCHITECTURE
# ==========================================

DATASET_PATH = "datasets/processed/cnn_training_dataset.h5"

NUM_CLASSES = 2
INPUT_CHANNELS = 1
SPECTRAL_BANDS = 224
PATCH_SIZE = 5


class TerraSpectra3DCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super(TerraSpectra3DCNN, self).__init__()

        # First 3D convolution block
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=(7, 3, 3),
                padding=(3, 1, 1)
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1))
        )

        # Second 3D convolution block
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=(5, 3, 3),
                padding=(2, 1, 1)
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1))
        )

        # Third 3D convolution block
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        # Convert feature maps into one feature vector
        self.global_pool = nn.AdaptiveAvgPool3d(
            output_size=(1, 1, 1)
        )

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.conv_block1(x)

        x = self.conv_block2(x)

        x = self.conv_block3(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x


def test_model():

    print("\nLoading Day 6 training dataset...")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "Day 6 dataset not found: " + DATASET_PATH
        )

    with h5py.File(DATASET_PATH, "r") as file:

        # Load four sample patches
        sample_data = file["X_train"][:4]

    input_tensor = torch.tensor(
        sample_data,
        dtype=torch.float32
    )

    print("Input tensor shape:", tuple(input_tensor.shape))

    print("\nCreating TerraSpectra 3D-CNN model...")

    model = TerraSpectra3DCNN()

    model.eval()

    # Test the forward pass
    with torch.no_grad():

        output = model(input_tensor)

    print("Model output shape:", tuple(output.shape))

    print(
        "Total trainable parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )

    if output.shape != (4, NUM_CLASSES):
        raise ValueError("Unexpected model output shape.")

    if not torch.isfinite(output).all():
        raise ValueError("Model produced invalid output values.")

    print("\n3D-CNN model validation passed!")

    print("Input: 4 hyperspectral patches")

    print("Output: 4 samples with 2 class scores each")

    print("\nDAY 7 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":

    print("\n========== TERRASPECTRA DAY 7 ==========")

    test_model()
