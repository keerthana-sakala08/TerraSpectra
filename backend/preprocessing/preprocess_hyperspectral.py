
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "datasets" / "sample" /
    "hyperspectral_sample.h5"
)

OUTPUT_DIR = ROOT / "datasets" / "processed"

OUTPUT_FILE = (
    OUTPUT_DIR / "processed_hyperspectral.h5"
)

SMOOTHING_WINDOW = 5


def load_dataset():
    """Load the original hyperspectral dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    with h5py.File(INPUT_FILE, "r") as file:

        cube = file["hyperspectral_cube"][:]
        labels = file["labels"][:]
        wavelengths = file["wavelengths_nm"][:]

        metadata = {
            key: value
            for key, value in file.attrs.items()
        }

    print("Original dataset loaded.")
    print("Cube shape:", cube.shape)

    return cube, labels, wavelengths, metadata


def handle_missing_values(cube):
    """Replace NaN and infinite values using band medians."""

    cleaned = cube.astype(np.float32, copy=True)

    for band in range(cleaned.shape[2]):

        band_data = cleaned[:, :, band]

        valid_values = band_data[
            np.isfinite(band_data)
        ]

        if valid_values.size == 0:
            replacement = 0.0
        else:
            replacement = float(
                np.median(valid_values)
            )

        band_data[~np.isfinite(band_data)] = replacement

    print("Missing-value handling completed.")

    return cleaned


def reduce_spectral_noise(cube, window=SMOOTHING_WINDOW):
    """Apply moving-average smoothing across spectral bands."""

    if window < 1 or window % 2 == 0:
        raise ValueError(
            "Smoothing window must be a positive odd number."
        )

    if window > cube.shape[2]:
        raise ValueError(
            "Smoothing window cannot exceed the band count."
        )

    padding = window // 2

    padded = np.pad(
        cube,
        ((0, 0), (0, 0), (padding, padding)),
        mode="edge"
    )

    smoothed = np.zeros(
        cube.shape,
        dtype=np.float32
    )

    for index in range(window):

        smoothed += padded[
            :,
            :,
            index:index + cube.shape[2]
        ]

    smoothed /= window

    print(
        f"Spectral smoothing completed. Window: {window}"
    )

    return smoothed


def normalize_data(cube):
    """Normalize each spectral band to the range 0 to 1."""

    minimum = np.min(
        cube,
        axis=(0, 1),
        keepdims=True
    )

    maximum = np.max(
        cube,
        axis=(0, 1),
        keepdims=True
    )

    difference = maximum - minimum

    normalized = np.divide(
        cube - minimum,
        difference,
        out=np.zeros_like(cube, dtype=np.float32),
        where=difference > 1e-8
    )

    normalized = np.clip(normalized, 0, 1)

    print("Normalization completed.")

    return normalized.astype(np.float32)


def validate_processed_data(cube, labels, wavelengths):
    """Check the processed dataset before saving."""

    if cube.ndim != 3:
        raise ValueError("Processed cube must be 3D.")

    if labels.ndim != 2:
        raise ValueError("Labels must be 2D.")

    if cube.shape[:2] != labels.shape:
        raise ValueError(
            "Processed image and label dimensions differ."
        )

    if cube.shape[2] != len(wavelengths):
        raise ValueError(
            "Band count and wavelength count differ."
        )

    if not np.isfinite(cube).all():
        raise ValueError(
            "Processed cube contains invalid values."
        )

    if cube.min() < 0 or cube.max() > 1:
        raise ValueError(
            "Normalized values must be between 0 and 1."
        )

    if not np.isin(np.unique(labels), [0, 1]).all():
        raise ValueError(
            "Unexpected labels found."
        )

    print("Processed dataset validation passed.")


def save_processed_data(
    cube,
    labels,
    wavelengths,
    metadata
):
    """Save processed data and preserve original metadata."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with h5py.File(OUTPUT_FILE, "w") as file:

        file.create_dataset(
            "processed_cube",
            data=cube,
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

        for key, value in metadata.items():
            file.attrs[key] = value

        file.attrs["preprocessing"] = (
            "Missing-value handling, spectral smoothing, "
            "per-band min-max normalization"
        )

        file.attrs["normalization_range"] = "0 to 1"

        file.attrs["smoothing_window"] = SMOOTHING_WINDOW

    print("Processed dataset saved:")
    print(OUTPUT_FILE)


def main():

    print("\n========== TERRASPECTRA DAY 4 ==========\n")

    cube, labels, wavelengths, metadata = load_dataset()

    print("\nOriginal minimum:", float(np.min(cube)))
    print("Original maximum:", float(np.max(cube)))

    # Test missing-value handling separately.
    test_cube = cube.copy()
    test_cube[0, 0, 0] = np.nan

    test_result = handle_missing_values(test_cube)

    if not np.isfinite(test_result).all():
        raise ValueError(
            "Missing-value handling test failed."
        )

    print("Missing-value test: PASSED")

    # Actual preprocessing.
    cleaned_cube = handle_missing_values(cube)

    smoothed_cube = reduce_spectral_noise(
        cleaned_cube
    )

    processed_cube = normalize_data(
        smoothed_cube
    )

    validate_processed_data(
        processed_cube,
        labels,
        wavelengths
    )

    save_processed_data(
        processed_cube,
        labels,
        wavelengths,
        metadata
    )

    print("\nProcessed shape:", processed_cube.shape)
    print(
        "Processed minimum:",
        float(processed_cube.min())
    )
    print(
        "Processed maximum:",
        float(processed_cube.max())
    )

    print("\nDAY 4 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
