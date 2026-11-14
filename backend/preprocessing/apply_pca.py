
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "datasets" / "processed"
    / "processed_hyperspectral.h5"
)

OUTPUT_DIR = ROOT / "datasets" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "pca_hyperspectral.h5"

N_COMPONENTS = 3


def load_processed_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_FILE}. "
            "Complete Day 4 preprocessing first."
        )

    with h5py.File(INPUT_FILE, "r") as file:
        cube = file["processed_cube"][:]
        labels = file["labels"][:]

    print("Processed data loaded.")
    print("Original shape:", cube.shape)

    return cube, labels


def apply_pca(cube):
    height, width, bands = cube.shape

    # Convert image pixels into rows and spectral bands into columns.
    pixels = cube.reshape(-1, bands)

    print("Applying PCA...")
    print("Input pixel matrix:", pixels.shape)

    pca = PCA(
        n_components=N_COMPONENTS,
        svd_solver="randomized",
        random_state=42
    )

    transformed = pca.fit_transform(pixels)

    pca_cube = transformed.reshape(
        height, width, N_COMPONENTS
    )

    print("PCA completed.")
    print("Reduced shape:", pca_cube.shape)

    return pca_cube, pca


def save_results(pca_cube, labels, pca):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(OUTPUT_FILE, "w") as file:
        file.create_dataset(
            "pca_cube",
            data=pca_cube.astype(np.float32),
            compression="gzip"
        )

        file.create_dataset("labels", data=labels)

        file.create_dataset(
            "explained_variance_ratio",
            data=pca.explained_variance_ratio_
        )

        file.attrs["project"] = "TerraSpectra"
        file.attrs["method"] = "Principal Component Analysis"
        file.attrs["number_of_components"] = N_COMPONENTS

    print("PCA output saved:", OUTPUT_FILE)


def main():
    print("\n========== TERRASPECTRA DAY 5 ==========\n")

    cube, labels = load_processed_data()

    pca_cube, pca = apply_pca(cube)

    if not np.isfinite(pca_cube).all():
        raise ValueError("PCA output contains invalid values.")

    if pca_cube.shape[:2] != labels.shape:
        raise ValueError("PCA output and labels do not match.")

    save_results(pca_cube, labels, pca)

    print("\nExplained variance ratio:")
    for index, value in enumerate(
        pca.explained_variance_ratio_, start=1
    ):
        print(f"Component {index}: {value:.4f}")

    print(
        "Total variance preserved:",
        f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )

    print("\nDay 5 PCA completed successfully!")


if __name__ == "__main__":
    main()
