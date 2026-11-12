
from pathlib import Path

import h5py
import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "sample"

H5_FILE = DATA_DIR / "hyperspectral_sample.h5"
TIF_FILE = DATA_DIR / "hyperspectral_sample.tif"
LABEL_FILE = DATA_DIR / "crop_labels.tif"

EXPECTED_BANDS = 224


def load_hdf5_data(file_path):
    """Load hyperspectral cube, labels and wavelengths from HDF5."""

    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    with h5py.File(file_path, "r") as file:
        cube = file["hyperspectral_cube"][:]
        labels = file["labels"][:]
        wavelengths = file["wavelengths_nm"][:]

        metadata = {
            "format": "HDF5",
            "project": file.attrs.get("project", "Unknown"),
            "dataset_type": file.attrs.get("dataset_type", "Unknown"),
            "bands": int(file.attrs.get("bands", cube.shape[-1])),
            "cube_shape": cube.shape,
            "label_shape": labels.shape,
            "wavelength_range_nm": (
                float(wavelengths.min()),
                float(wavelengths.max())
            )
        }

    return cube, labels, wavelengths, metadata


def load_geotiff_data(file_path):
    """Load a hyperspectral GeoTIFF and extract GIS metadata."""

    if not file_path.exists():
        raise FileNotFoundError(f"GeoTIFF file not found: {file_path}")

    with rasterio.open(file_path) as dataset:
        cube = dataset.read()
        cube = np.moveaxis(cube, 0, -1)

        metadata = {
            "format": "GeoTIFF",
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "crs": str(dataset.crs),
            "transform": str(dataset.transform),
            "data_type": dataset.dtypes[0],
            "cube_shape": cube.shape
        }

    return cube, metadata


def load_label_geotiff(file_path):
    """Load crop classification labels from GeoTIFF."""

    if not file_path.exists():
        raise FileNotFoundError(f"Label file not found: {file_path}")

    with rasterio.open(file_path) as dataset:
        labels = dataset.read(1)

    return labels


def validate_hyperspectral_data(cube, labels, wavelengths):
    """Validate dimensions, bands, labels and data values."""

    errors = []

    if cube.ndim != 3:
        errors.append("Hyperspectral cube must have 3 dimensions.")

    if cube.shape[-1] != EXPECTED_BANDS:
        errors.append(
            f"Expected {EXPECTED_BANDS} bands, got {cube.shape[-1]}."
        )

    if labels.shape != cube.shape[:2]:
        errors.append("Label dimensions do not match the image.")

    if len(wavelengths) != cube.shape[-1]:
        errors.append("Wavelength count does not match spectral bands.")

    if not np.isfinite(cube).all():
        errors.append("Cube contains NaN or infinite values.")

    if not np.isfinite(wavelengths).all():
        errors.append("Wavelength data contains invalid values.")

    if not np.isin(np.unique(labels), [0, 1]).all():
        errors.append("Unexpected label values found.")

    if errors:
        raise ValueError("\n".join(errors))

    return True


def main():
    print("\n========== TERRASPECTRA DATA PIPELINE ==========\n")

    h5_cube, labels, wavelengths, h5_metadata = load_hdf5_data(
        H5_FILE
    )

    tif_cube, tif_metadata = load_geotiff_data(TIF_FILE)

    tif_labels = load_label_geotiff(LABEL_FILE)

    validate_hyperspectral_data(
        h5_cube, labels, wavelengths
    )

    validate_hyperspectral_data(
        tif_cube, tif_labels, wavelengths
    )

    if not np.allclose(h5_cube, tif_cube, atol=1e-6):
        raise ValueError("HDF5 and GeoTIFF cube values do not match.")

    if not np.array_equal(labels, tif_labels):
        raise ValueError("HDF5 and GeoTIFF labels do not match.")

    print("HDF5 METADATA")
    for key, value in h5_metadata.items():
        print(f"{key}: {value}")

    print("\nGEOTIFF METADATA")
    for key, value in tif_metadata.items():
        print(f"{key}: {value}")

    print("\nVALIDATION RESULTS")
    print("HDF5 data: Valid")
    print("GeoTIFF data: Valid")
    print("Spectral bands: 224")
    print("Labels: Healthy (0), Stressed (1)")
    print("HDF5 and GeoTIFF values: Matching")

    print("\nDay 3 data-loading pipeline completed successfully!")


if __name__ == "__main__":
    main()
