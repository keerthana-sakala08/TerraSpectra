
from pathlib import Path
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    ROOT
    / "datasets"
    / "sample"
    / "hyperspectral_sample.h5"
)


def load_dataset():
    """Load hyperspectral data and metadata."""

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    with h5py.File(DATASET_FILE, "r") as file:

        required_datasets = [
            "hyperspectral_cube",
            "labels",
            "wavelengths_nm"
        ]

        for name in required_datasets:
            if name not in file:
                raise ValueError(
                    f"Missing dataset: {name}"
                )

        cube = file["hyperspectral_cube"][:]
        labels = file["labels"][:]
        wavelengths = file["wavelengths_nm"][:]

        metadata = {
            key: value
            for key, value in file.attrs.items()
        }

    print("Hyperspectral dataset loaded successfully.")

    return cube, labels, wavelengths, metadata


def validate_image(cube, labels, wavelengths):
    """Validate image dimensions and spectral information."""

    print("\n--- IMAGE VALIDATION ---")

    if cube.ndim != 3:
        raise ValueError(
            "Hyperspectral cube must have 3 dimensions."
        )

    if labels.ndim != 2:
        raise ValueError(
            "Labels must have 2 dimensions."
        )

    if cube.shape[:2] != labels.shape:
        raise ValueError(
            "Image and label dimensions do not match."
        )

    if cube.shape[2] != len(wavelengths):
        raise ValueError(
            "Band count and wavelength count do not match."
        )

    if not np.isfinite(cube).all():
        raise ValueError(
            "Hyperspectral cube contains invalid values."
        )

    if not np.isfinite(wavelengths).all():
        raise ValueError(
            "Wavelength information contains invalid values."
        )

    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError(
            "Wavelengths must be in increasing order."
        )

    unique_labels = np.unique(labels)

    if not np.isin(unique_labels, [0, 1]).all():
        raise ValueError(
            "Unexpected class labels found."
        )

    print("Image dimensions:", cube.shape)
    print("Label dimensions:", labels.shape)
    print("Number of bands:", cube.shape[2])
    print(
        "Wavelength range:",
        float(wavelengths.min()),
        "to",
        float(wavelengths.max()),
        "nm"
    )
    print("Available classes:", unique_labels.tolist())

    print("Image validation: PASSED")


def validate_metadata(metadata):
    """Validate crop and sensor metadata."""

    print("\n--- METADATA VALIDATION ---")

    required_fields = [
        "project",
        "crop_type",
        "scanning_method",
        "number_of_bands",
        "wavelength_range_nm",
        "class_0",
        "class_1",
        "data_status",
        "real_world_validation"
    ]

    for field in required_fields:
        if field not in metadata:
            raise ValueError(
                f"Missing metadata field: {field}"
            )

    if metadata["scanning_method"] != "Hyperspectral Imaging":
        raise ValueError(
            "Unexpected scanning method."
        )

    if int(metadata["number_of_bands"]) != 224:
        raise ValueError(
            "Expected 224 spectral bands."
        )

    if metadata["class_0"] != "Healthy":
        raise ValueError("Invalid class 0 description.")

    if metadata["class_1"] != "Stressed":
        raise ValueError("Invalid class 1 description.")

    if metadata["real_world_validation"] not in (False, 0):
        raise ValueError(
            "This development dataset should be marked "
            "as not validated in the real world."
        )

    print("Project:", metadata["project"])
    print("Crop type:", metadata["crop_type"])
    print("Scanning method:", metadata["scanning_method"])
    print("Sensor:", metadata.get("sensor_type", "Not provided"))
    print("Number of bands:", metadata["number_of_bands"])
    print("Class 0:", metadata["class_0"])
    print("Class 1:", metadata["class_1"])
    print("Data status:", metadata["data_status"])

    print("Metadata validation: PASSED")


def generate_report(cube, labels, wavelengths, metadata):
    """Display a final validation report."""

    print("\n========== TERRASPECTRA DAY 3 REPORT ==========")

    print("Dataset: Valid")
    print("Crop:", metadata["crop_type"])
    print("Image height:", cube.shape[0])
    print("Image width:", cube.shape[1])
    print("Spectral bands:", cube.shape[2])
    print("Labels:", np.unique(labels).tolist())
    print("Wavelength count:", len(wavelengths))

    print("Scanning:", metadata["scanning_method"])
    print("Synthetic dataset:", "Yes")

    print("\nFINAL RESULT: ALL VALIDATIONS PASSED")


def main():
    print("\n========== TERRASPECTRA DAY 3 ==========\n")

    cube, labels, wavelengths, metadata = load_dataset()

    validate_image(
        cube,
        labels,
        wavelengths
    )

    validate_metadata(metadata)

    generate_report(
        cube,
        labels,
        wavelengths,
        metadata
    )


if __name__ == "__main__":
    main()
