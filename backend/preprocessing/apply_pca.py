
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "datasets" / "processed" /
    "processed_hyperspectral.h5"
)

OUTPUT_DIR = ROOT / "datasets" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "pca_hyperspectral.h5"

N_COMPONENTS = 3


def load_processed_data():
    """Load the normalized hyperspectral dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_FILE}. "
            "Complete Day 4 first."
        )

    with h5py.File(INPUT_FILE, "r") as file:

        cube = file["processed_cube"][:]
        labels = file["labels"][:]
        wavelengths = file["wavelengths_nm"][:]

        metadata = {
            key: value
            for key, value in file.attrs.items()
        }

    if cube.ndim != 3:
        raise ValueError("Expected a 3D hyperspectral cube.")

    if cube.shape[:2] != labels.shape:
        raise ValueError("Image and label dimensions do not match.")

    if cube.shape[2] != len(wavelengths):
        raise ValueError("Band and wavelength counts do not match.")

    if not np.isfinite(cube).all():
        raise ValueError("Input contains invalid values.")

    print("Processed dataset loaded successfully.")
    print("Original cube shape:", cube.shape)

    return cube, labels, wavelengths, metadata


def apply_pca(cube):
    """Reduce spectral dimensions using PCA."""

    height, width, bands = cube.shape

    # Convert each pixel into a row of spectral features.
    pixels = cube.reshape(-1, bands)

    print("\nApplying PCA...")
    print("Pixel matrix shape:", pixels.shape)

    pca = PCA(
        n_components=N_COMPONENTS,
        svd_solver="randomized",
        random_state=42
    )

    transformed = pca.fit_transform(pixels)

    # Restore the image's spatial dimensions.
    pca_cube = transformed.reshape(
        height,
        width,
        N_COMPONENTS
    )

    print("PCA completed.")
    print("Reduced cube shape:", pca_cube.shape)

    return pca_cube, pca


def validate_pca(pca_cube, labels, pca):
    """Validate PCA results."""

    if pca_cube.ndim != 3:
        raise ValueError("PCA output must be 3D.")

    if pca_cube.shape[:2] != labels.shape:
        raise ValueError("PCA output and labels do not match.")

    if pca_cube.shape[2] != N_COMPONENTS:
        raise ValueError("Unexpected number of PCA components.")

    if not np.isfinite(pca_cube).all():
        raise ValueError("PCA output contains invalid values.")

    if len(pca.explained_variance_ratio_) != N_COMPONENTS:
        raise ValueError("Explained variance information is missing.")

    if not np.isfinite(pca.explained_variance_ratio_).all():
        raise ValueError("Invalid explained variance values.")

    if np.any(pca.explained_variance_ratio_ < 0):
        raise ValueError("Negative explained variance found.")

    print("PCA validation passed.")


def save_results(pca_cube, labels, wavelengths, pca, metadata):
    """Save PCA features and model information."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with h5py.File(OUTPUT_FILE, "w") as file:

        file.create_dataset(
            "pca_cube",
            data=pca_cube.astype(np.float32),
            compression="gzip"
        )

        file.create_dataset(
            "labels",
            data=labels,
            compression="gzip"
        )

        file.create_dataset(
            "wavelengths_nm",
            data=wavelengths
        )

        file.create_dataset(
            "explained_variance_ratio",
            data=pca.explained_variance_ratio_
        )

        file.create_dataset(
            "pca_components",
            data=pca.components_
        )

        file.create_dataset(
            "pca_mean",
            data=pca.mean_
        )

        for key, value in metadata.items():
            file.attrs[key] = value

        file.attrs["method"] = "Principal Component Analysis"
        file.attrs["number_of_components"] = N_COMPONENTS
        file.attrs["original_number_of_bands"] = len(wavelengths)

    print("\nPCA results saved:")
    print(OUTPUT_FILE)


def main():

    print("\n========== TERRASPECTRA DAY 5 ==========\n")

    cube, labels, wavelengths, metadata = load_processed_data()

    pca_cube, pca = apply_pca(cube)

    validate_pca(
        pca_cube,
        labels,
        pca
    )

    save_results(
        pca_cube,
        labels,
        wavelengths,
        pca,
        metadata
    )

    print("\nExplained variance ratio:")

    for index, value in enumerate(
        pca.explained_variance_ratio_,
        start=1
    ):
        print(f"Component {index}: {value:.4f}")

    total_variance = (
        pca.explained_variance_ratio_.sum() * 100
    )

    print(
        "Total variance preserved:",
        f"{total_variance:.2f}%"
    )

    print("\nOriginal shape:", cube.shape)
    print("Reduced shape:", pca_cube.shape)

    print("\nDAY 5 PCA COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
