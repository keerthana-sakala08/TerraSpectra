
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "datasets" / "sample" / "hyperspectral_sample.h5"
OUTPUT_DIR = ROOT / "datasets" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "processed_hyperspectral.h5"

SMOOTHING_WINDOW = 5


def load_data(file_path):
    """Load the raw hyperspectral cube and related information."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with h5py.File(file_path, "r") as file:
        cube = file["hyperspectral_cube"][:]
        labels = file["labels"][:]
        wavelengths = file["wavelengths_nm"][:]

    print("Raw dataset loaded successfully.")
    return cube, labels, wavelengths


def handle_missing_values(cube):
    """Replace NaN and infinite values using per-band median."""

    cleaned = cube.copy()

    for band in range(cleaned.shape[2]):
        band_data = cleaned[:, :, band]
        valid_values = band_data[np.isfinite(band_data)]

        if valid_values.size == 0:
            replacement = 0.0
        else:
            replacement = np.median(valid_values)

        band_data[~np.isfinite(band_data)] = replacement

    print("Missing values handled.")
    return cleaned


def reduce_spectral_noise(cube, window=SMOOTHING_WINDOW):
    """Apply moving-average smoothing along the spectral dimension."""

    if window < 1 or window % 2 == 0:
        raise ValueError("Smoothing window must be a positive odd number.")

    padding = window // 2

    padded = np.pad(
        cube,
        ((0, 0), (0, 0), (padding, padding)),
        mode="edge"
    )

    smoothed = np.zeros_like(cube, dtype=np.float32)

    for index in range(window):
        smoothed += padded[:, :, index:index + cube.shape[2]]

    smoothed /= window

    print(f"Spectral noise reduction completed (window={window}).")
    return smoothed


def normalize_data(cube):
    """Normalize each spectral band to the range 0 to 1."""

    minimum = np.min(cube, axis=(0, 1), keepdims=True)
    maximum = np.max(cube, axis=(0, 1), keepdims=True)

    difference = maximum - minimum

    normalized = np.divide(
        cube - minimum,
        difference,
        out=np.zeros_like(cube, dtype=np.float32),
        where=difference > 1e-8
    )

    normalized = np.clip(normalized, 0, 1)

    print("Spectral normalization completed.")
    return normalized.astype(np.float32)


def validate_processed_data(cube, labels, wavelengths):
    """Check the processed cube before saving."""

    if cube.ndim != 3:
        raise ValueError("Cube must have three dimensions.")

    if cube.shape[:2] != labels.shape:
        raise ValueError("Image and label dimensions do not match.")

    if cube.shape[2] != len(wavelengths):
        raise ValueError("Spectral band and wavelength counts differ.")

    if not np.isfinite(cube).all():
        raise ValueError("Processed data contains invalid values.")

    if cube.min() < 0 or cube.max() > 1:
        raise ValueError("Normalized values must be between 0 and 1.")

    if not np.isin(np.unique(labels), [0, 1]).all():
        raise ValueError("Unexpected crop labels found.")

    print("Processed data validation successful.")


def save_processed_data(cube, labels, wavelengths):
    """Save the processed cube and metadata into an HDF5 file."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(OUTPUT_FILE, "w") as file:
        file.create_dataset(
            "processed_cube",
            data=cube,
            compression="gzip"
        )

        file.create_dataset("labels", data=labels)
        file.create_dataset("wavelengths_nm", data=wavelengths)

        file.attrs["project"] = "TerraSpectra"
        file.attrs["processing"] = (
            "Missing value handling, spectral smoothing, "
            "per-band min-max normalization"
        )

        file.attrs["normalization_range"] = "0 to 1"

    print(f"Processed dataset saved: {OUTPUT_FILE}")


def main():
    print("\n========== TERRASPECTRA DAY 4 ==========\n")

    cube, labels, wavelengths = load_data(INPUT_FILE)

    print("Original cube shape:", cube.shape)
    print("Original minimum:", float(cube.min()))
    print("Original maximum:", float(cube.max()))

    # Test missing-value handling without changing the raw dataset.
    test_cube = cube.copy()
    test_cube[0, 0, 0] = np.nan
    test_result = handle_missing_values(test_cube)

    if not np.isfinite(test_result).all():
        raise ValueError("Missing-value handling test failed.")

    print("Missing-value handling test: Passed")

    # Actual preprocessing pipeline.
    cleaned_cube = handle_missing_values(cube)

    smoothed_cube = reduce_spectral_noise(cleaned_cube)

    processed_cube = normalize_data(smoothed_cube)

    validate_processed_data(
        processed_cube, labels, wavelengths
    )

    save_processed_data(
        processed_cube, labels, wavelengths
    )

    print("\n========== PREPROCESSING SUMMARY ==========")
    print("Original shape:", cube.shape)
    print("Processed shape:", processed_cube.shape)
    print("Missing-value test: Passed")
    print("Noise reduction: Completed")
    print("Normalization: 0 to 1")
    print("Final minimum:", float(processed_cube.min()))
    print("Final maximum:", float(processed_cube.max()))

    print("\nDay 4 preprocessing completed successfully!")


if __name__ == "__main__":
    main()
