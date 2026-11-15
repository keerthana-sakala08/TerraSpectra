
import os
import h5py
import numpy as np

from sklearn.model_selection import train_test_split


# ==========================================
# TERRASPECTRA - DAY 6
# Prepare labelled dataset for 3D-CNN
# ==========================================

INPUT_FILE = (
    "datasets/processed/processed_hyperspectral.h5"
)

OUTPUT_FILE = (
    "datasets/processed/cnn_training_dataset.h5"
)

PATCH_SIZE = 5
MAX_SAMPLES_PER_CLASS = 1500
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset():

    print("\nLoading preprocessed dataset...")

    with h5py.File(INPUT_FILE, "r") as file:

        cube = file["processed_cube"][:]
        labels = file["labels"][:]

    print("Cube shape:", cube.shape)
    print("Labels shape:", labels.shape)

    return cube, labels


def validate_dataset(cube, labels):

    print("\nValidating dataset...")

    if cube.ndim != 3:
        raise ValueError("Cube must be 3-dimensional.")

    if labels.ndim != 2:
        raise ValueError("Labels must be 2-dimensional.")

    if cube.shape[:2] != labels.shape:
        raise ValueError("Cube and labels shapes do not match.")

    if not np.isfinite(cube).all():
        raise ValueError("Cube contains invalid values.")

    classes = np.unique(labels)

    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(
            f"Expected classes [0, 1], found {classes}"
        )

    print("Dataset validation passed.")
    print("Classes:", classes)


def extract_patches(cube, labels):

    print("\nExtracting image patches...")

    height, width, bands = cube.shape

    radius = PATCH_SIZE // 2

    # Add padding to handle image boundaries
    padded_cube = np.pad(
        cube,
        (
            (radius, radius),
            (radius, radius),
            (0, 0)
        ),
        mode="reflect"
    )

    rng = np.random.default_rng(RANDOM_STATE)

    selected_pixels = []

    for class_id in [0, 1]:

        positions = np.argwhere(labels == class_id)

        sample_count = min(
            MAX_SAMPLES_PER_CLASS,
            len(positions)
        )

        selected = rng.choice(
            len(positions),
            size=sample_count,
            replace=False
        )

        class_positions = positions[selected]

        for row, col in class_positions:

            selected_pixels.append(
                (row, col, class_id)
            )

    rng.shuffle(selected_pixels)

    number_of_samples = len(selected_pixels)

    # PyTorch Conv3D format:
    # Samples, Channels, Spectral Bands, Height, Width

    X = np.empty(
        (
            number_of_samples,
            1,
            bands,
            PATCH_SIZE,
            PATCH_SIZE
        ),
        dtype=np.float32
    )

    y = np.empty(
        number_of_samples,
        dtype=np.int64
    )

    for index, (row, col, class_id) in enumerate(
        selected_pixels
    ):

        patch = padded_cube[
            row:row + PATCH_SIZE,
            col:col + PATCH_SIZE,
            :
        ]

        X[index, 0] = np.transpose(
            patch,
            (2, 0, 1)
        )

        y[index] = class_id

    print("Patch extraction completed.")
    print("Patch dataset shape:", X.shape)
    print("Label dataset shape:", y.shape)

    return X, y


def split_dataset(X, y):

    print("\nSplitting training and testing data...")

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train, X_test, y_train, y_test


def save_dataset(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\nSaving prepared dataset...")

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with h5py.File(OUTPUT_FILE, "w") as file:

        file.create_dataset(
            "X_train",
            data=X_train,
            compression="gzip",
            compression_opts=4
        )

        file.create_dataset(
            "X_test",
            data=X_test,
            compression="gzip",
            compression_opts=4
        )

        file.create_dataset(
            "y_train",
            data=y_train,
            compression="gzip"
        )

        file.create_dataset(
            "y_test",
            data=y_test,
            compression="gzip"
        )

        file.attrs["project"] = "TerraSpectra"
        file.attrs["day"] = 6
        file.attrs["patch_size"] = PATCH_SIZE
        file.attrs["number_of_bands"] = X_train.shape[2]
        file.attrs["class_0"] = "Healthy"
        file.attrs["class_1"] = "Stressed"
        file.attrs["dataset_type"] = "Synthetic"
        file.attrs["real_world_validation"] = False

    print("Dataset saved successfully:")
    print(OUTPUT_FILE)


def main():

    print("\n========== TERRASPECTRA DAY 6 ==========")

    cube, labels = load_dataset()

    validate_dataset(cube, labels)

    X, y = extract_patches(cube, labels)

    X_train, X_test, y_train, y_test = (
        split_dataset(X, y)
    )

    save_dataset(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n========== DAY 6 COMPLETED ==========")


if __name__ == "__main__":
    main()
