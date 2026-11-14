
from pathlib import Path
import h5py

ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    ROOT / "datasets" / "sample" /
    "hyperspectral_sample.h5"
)


def add_metadata():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    # Open the existing file without deleting its data
    with h5py.File(DATASET_FILE, "r+") as file:

        # General project information
        file.attrs["project"] = "TerraSpectra"
        file.attrs["dataset_type"] = "Synthetic Hyperspectral Dataset"
        file.attrs["sample_id"] = "TS_SAMPLE_001"

        # Crop information
        file.attrs["crop_type"] = "Tomato"
        file.attrs["field_location"] = (
            "Synthetic test field - no real coordinates"
        )

        # Scanning information
        file.attrs["scanning_method"] = "Hyperspectral Imaging"
        file.attrs["sensor_type"] = "Synthetic Hyperspectral Sensor"
        file.attrs["number_of_bands"] = 224
        file.attrs["wavelength_range_nm"] = "400-1000"

        # Class information
        file.attrs["class_0"] = "Healthy"
        file.attrs["class_1"] = "Stressed"

        # Dataset limitations
        file.attrs["data_status"] = (
            "Synthetic data for development and testing"
        )
        file.attrs["real_world_validation"] = False

        print("Crop metadata added successfully.")

        print("\nDataset information:")
        for key, value in file.attrs.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    print("\n===== TERRASPECTRA DAY 2 =====\n")
    add_metadata()
    print("\nDay 2 metadata update completed!")
