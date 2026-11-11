
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "datasets" / "sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT = 128
WIDTH = 128
BANDS = 224
RANDOM_SEED = 42


def create_spectral_signatures(wavelengths):
    """Create synthetic healthy and stressed vegetation signatures."""
    red_edge = 1 / (1 + np.exp(-(wavelengths - 710) / 18))

    healthy = (
        0.06
        + 0.48 * red_edge
        - 0.025 * np.exp(-0.5 * ((wavelengths - 670) / 25) ** 2)
    )

    stressed = healthy * 0.86 + 0.015

    return healthy, stressed


def create_labels():
    """Create healthy and disease-stressed spatial regions."""
    labels = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    y, x = np.ogrid[:HEIGHT, :WIDTH]

    # Elliptical stressed crop region
    ellipse = (
        ((x - 78) / 25) ** 2
        + ((y - 75) / 18) ** 2
    ) <= 1

    labels[ellipse] = 1

    # Additional rectangular stressed region
    labels[25:48, 20:48] = 1

    return labels


def generate_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    wavelengths = np.linspace(
        400, 1000, BANDS, dtype=np.float32
    )

    healthy, stressed = create_spectral_signatures(wavelengths)
    labels = create_labels()

    # Assign spectral signatures according to pixel labels
    cube = healthy[None, None, :] + (
        labels[:, :, None]
        * (stressed - healthy)[None, None, :]
    )

    # Add realistic synthetic sensor noise
    brightness = rng.normal(
        1.0, 0.025, size=(HEIGHT, WIDTH, 1)
    )

    noise = rng.normal(
        0, 0.008, size=(HEIGHT, WIDTH, BANDS)
    )

    cube = np.clip(
        cube * brightness + noise, 0, 1
    ).astype(np.float32)

    h5_path = OUTPUT_DIR / "hyperspectral_sample.h5"
    tif_path = OUTPUT_DIR / "hyperspectral_sample.tif"
    label_path = OUTPUT_DIR / "crop_labels.tif"

    # Save HDF5 dataset
    with h5py.File(h5_path, "w") as file:
        file.create_dataset(
            "hyperspectral_cube",
            data=cube,
            compression="gzip"
        )

        file.create_dataset("labels", data=labels)
        file.create_dataset("wavelengths_nm", data=wavelengths)

        file.attrs["project"] = "TerraSpectra"
        file.attrs["dataset_type"] = "Synthetic"
        file.attrs["bands"] = BANDS
        file.attrs["classes"] = "0=Healthy, 1=Stressed"

    # Save hyperspectral GeoTIFF (bands, height, width)
    profile = {
        "driver": "GTiff",
        "height": HEIGHT,
        "width": WIDTH,
        "count": BANDS,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(80.0, 17.0, 0.0001, 0.0001),
        "compress": "deflate"
    }

    with rasterio.open(tif_path, "w", **profile) as dataset:
        dataset.write(np.moveaxis(cube, -1, 0))

    # Save crop labels as a separate GeoTIFF
    label_profile = profile.copy()
    label_profile.update(count=1, dtype="uint8")

    with rasterio.open(label_path, "w", **label_profile) as dataset:
        dataset.write(labels, 1)

    print("TerraSpectra synthetic dataset created successfully!")
    print(f"Cube shape: {cube.shape}")
    print(f"Number of bands: {BANDS}")
    print(f"Spatial dimensions: {HEIGHT} x {WIDTH}")
    print(f"Classes: Healthy (0), Stressed (1)")
    print(f"HDF5: {h5_path}")
    print(f"GeoTIFF: {tif_path}")
    print(f"Labels: {label_path}")


if __name__ == "__main__":
    generate_dataset()
