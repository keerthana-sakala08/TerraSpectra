
import os
import sys
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path

# Find the TerraSpectra project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.cnn3d_model import TerraSpectra3DCNN


# ==========================================
# TERRASPECTRA - DAY 9
# CROP HEALTH PREDICTION AND HEATMAP
# ==========================================

INPUT_FILE = "datasets/processed/processed_hyperspectral.h5"
MODEL_FILE = "models/terraspectra_3dcnn.pth"

OUTPUT_DIR = "outputs/day9"
PREDICTION_FILE = os.path.join(
    OUTPUT_DIR,
    "crop_health_predictions.h5"
)

PREDICTION_IMAGE = os.path.join(
    OUTPUT_DIR,
    "crop_health_map.png"
)

PROBABILITY_IMAGE = os.path.join(
    OUTPUT_DIR,
    "stress_probability_map.png"
)

PATCH_SIZE = 5
BATCH_SIZE = 64


def load_cube():

    print("\nLoading preprocessed hyperspectral cube...")

    with h5py.File(INPUT_FILE, "r") as file:
        cube = file["processed_cube"][:]

    if cube.ndim != 3:
        raise ValueError("Expected a 3D hyperspectral cube.")

    if not np.isfinite(cube).all():
        raise ValueError("Input cube contains invalid values.")

    print("Cube shape:", cube.shape)

    return cube


def load_model(device):

    print("\nLoading trained 3D-CNN model...")

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "Trained model not found: " + MODEL_FILE
        )

    model = TerraSpectra3DCNN().to(device)

    state_dict = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("Trained model loaded successfully.")

    return model


def predict_crop_health(cube, model, device):

    print("\nPreparing image patches...")

    height, width, bands = cube.shape

    radius = PATCH_SIZE // 2

    padded_cube = np.pad(
        cube,
        (
            (radius, radius),
            (radius, radius),
            (0, 0)
        ),
        mode="reflect"
    )

    total_pixels = height * width

    prediction_map = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    stress_probability_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    print("Total pixels to predict:", total_pixels)

    model.eval()

    batch_patches = []
    batch_positions = []

    processed = 0

    def predict_batch():

        nonlocal processed

        if not batch_patches:
            return

        batch_array = np.stack(
            batch_patches
        ).astype(np.float32)

        # Shape:
        # Batch, Channel, Spectral Bands, Height, Width
        input_tensor = torch.from_numpy(
            batch_array[:, np.newaxis, :, :, :]
        ).float().to(device)

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            stress_probabilities = probabilities[:, 1]

        predictions = predictions.cpu().numpy()
        stress_probabilities = (
            stress_probabilities.cpu().numpy()
        )

        for index, (row, col) in enumerate(batch_positions):

            prediction_map[row, col] = predictions[index]

            stress_probability_map[row, col] = (
                stress_probabilities[index]
            )

        processed += len(batch_patches)

        if processed % 1024 == 0 or processed == total_pixels:
            print(
                f"Processed {processed}/{total_pixels} pixels"
            )

        batch_patches.clear()
        batch_positions.clear()

    print("\nStarting crop health prediction...")

    for row in range(height):

        for col in range(width):

            patch = padded_cube[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
                :
            ]

            patch = np.transpose(
                patch,
                (2, 0, 1)
            )

            batch_patches.append(patch)
            batch_positions.append((row, col))

            if len(batch_patches) >= BATCH_SIZE:
                predict_batch()

    # Process remaining patches
    predict_batch()

    print("\nPrediction completed.")

    return prediction_map, stress_probability_map


def calculate_statistics(prediction_map):

    print("\nCalculating crop health statistics...")

    total_pixels = prediction_map.size

    healthy_pixels = int(
        np.sum(prediction_map == 0)
    )

    stressed_pixels = int(
        np.sum(prediction_map == 1)
    )

    healthy_percentage = (
        healthy_pixels / total_pixels
    ) * 100

    stressed_percentage = (
        stressed_pixels / total_pixels
    ) * 100

    print("Total pixels:", total_pixels)
    print("Healthy pixels:", healthy_pixels)
    print("Stressed pixels:", stressed_pixels)

    print(
        f"Predicted healthy area: {healthy_percentage:.2f}%"
    )

    print(
        f"Predicted stressed area: {stressed_percentage:.2f}%"
    )

    return {
        "total_pixels": total_pixels,
        "healthy_pixels": healthy_pixels,
        "stressed_pixels": stressed_pixels,
        "healthy_percentage": healthy_percentage,
        "stressed_percentage": stressed_percentage
    }


def save_prediction(
    prediction_map,
    stress_probability_map,
    statistics
):

    print("\nSaving prediction results...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with h5py.File(PREDICTION_FILE, "w") as file:

        file.create_dataset(
            "prediction_map",
            data=prediction_map,
            compression="gzip"
        )

        file.create_dataset(
            "stress_probability_map",
            data=stress_probability_map,
            compression="gzip"
        )

        for key, value in statistics.items():
            file.attrs[key] = value

        file.attrs["project"] = "TerraSpectra"
        file.attrs["model"] = "TerraSpectra3DCNN"
        file.attrs["patch_size"] = PATCH_SIZE
        file.attrs["class_0"] = "Healthy"
        file.attrs["class_1"] = "Stressed"
        file.attrs["dataset_type"] = "Synthetic"
        file.attrs["real_world_validation"] = False

    print("Prediction data saved:", PREDICTION_FILE)


def create_heatmaps(
    prediction_map,
    stress_probability_map
):

    print("\nGenerating crop health heatmaps...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Green = Healthy, Red = Stressed
    from matplotlib.colors import ListedColormap

    class_colors = ListedColormap([
        "#16A34A",
        "#DC2626"
    ])

    plt.figure(figsize=(8, 6))

    plt.imshow(
        prediction_map,
        cmap=class_colors,
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    plt.title("TerraSpectra - Crop Health Prediction Map")
    plt.xlabel("Image Column")
    plt.ylabel("Image Row")

    colorbar = plt.colorbar(
        ticks=[0, 1]
    )

    colorbar.ax.set_yticklabels([
        "Healthy",
        "Stressed"
    ])

    plt.tight_layout()
    plt.savefig(
        PREDICTION_IMAGE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Continuous stress probability heatmap
    plt.figure(figsize=(8, 6))

    image = plt.imshow(
        stress_probability_map,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    plt.title("TerraSpectra - Stress Probability Map")
    plt.xlabel("Image Column")
    plt.ylabel("Image Row")

    plt.colorbar(
        image,
        label="Predicted stress probability"
    )

    plt.tight_layout()
    plt.savefig(
        PROBABILITY_IMAGE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Crop health map saved:", PREDICTION_IMAGE)
    print("Stress probability map saved:", PROBABILITY_IMAGE)


def main():

    print("\n========== TERRASPECTRA DAY 9 ==========")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    cube = load_cube()

    model = load_model(device)

    prediction_map, stress_probability_map = (
        predict_crop_health(
            cube,
            model,
            device
        )
    )

    statistics = calculate_statistics(
        prediction_map
    )

    save_prediction(
        prediction_map,
        stress_probability_map,
        statistics
    )

    create_heatmaps(
        prediction_map,
        stress_probability_map
    )

    print("\nDAY 9 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
