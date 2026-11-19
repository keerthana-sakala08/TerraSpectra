from pathlib import Path
import json

import h5py
import numpy as np
import rasterio
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_TIF = PROJECT_ROOT / "datasets/sample/hyperspectral_sample.tif"
LABEL_TIF = PROJECT_ROOT / "datasets/sample/crop_labels.tif"
PROCESSED_H5 = PROJECT_ROOT / "datasets/processed/processed_hyperspectral.h5"
PREDICTION_H5 = PROJECT_ROOT / "outputs/day9/crop_health_predictions.h5"

OUTPUT_DIR = PROJECT_ROOT / "outputs/day10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_TIF = OUTPUT_DIR / "predicted_crop_health.tif"
PROBABILITY_TIF = OUTPUT_DIR / "stress_probability.tif"
REPORT_JSON = OUTPUT_DIR / "day10_validation_report.json"


# --------------------------------------------------
# 2. Load and validate the source GeoTIFF
# --------------------------------------------------

def inspect_source_geotiff():
    print("\nSTEP 1: Inspecting source GeoTIFF")

    with rasterio.open(SOURCE_TIF) as src:
        source_shape = (src.height, src.width)
        source_crs = src.crs
        source_transform = src.transform
        source_bounds = src.bounds
        band_count = src.count

        print("Source image shape:", source_shape)
        print("Number of bands:", band_count)
        print("CRS:", source_crs)
        print("Transform:", source_transform)
        print("Bounds:", source_bounds)

        if source_crs is None:
            raise ValueError("Source GeoTIFF has no CRS.")

        if source_transform.is_identity:
            raise ValueError("Source GeoTIFF has an identity transform.")

        if band_count != 224:
            raise ValueError(
                f"Expected 224 bands, but found {band_count}."
            )

    return {
        "shape": source_shape,
        "crs": source_crs,
        "transform": source_transform,
        "bounds": source_bounds,
        "band_count": band_count,
    }


# --------------------------------------------------
# 3. Check the label GeoTIFF
# --------------------------------------------------

def validate_label_geotiff(source_info):
    print("\nSTEP 2: Validating label GeoTIFF")

    with rasterio.open(LABEL_TIF) as labels_src:
        label_shape = (labels_src.height, labels_src.width)
        label_crs = labels_src.crs
        label_transform = labels_src.transform
        label_array = labels_src.read(1)

        print("Label shape:", label_shape)
        print("Label CRS:", label_crs)
        print("Label classes:", np.unique(label_array))

        if label_shape != source_info["shape"]:
            raise ValueError("Source and label dimensions do not match.")

        if label_crs != source_info["crs"]:
            raise ValueError("Source and label CRS do not match.")

        if not label_transform.almost_equals(source_info["transform"]):
            raise ValueError("Source and label transforms do not match.")

        if not np.isin(np.unique(label_array), [0, 1]).all():
            raise ValueError("Unexpected label values. Expected 0 and 1.")

    print("Source and label spatial alignment: PASSED")

    return label_array


# --------------------------------------------------
# 4. Validate processed HDF5 data
# --------------------------------------------------

def validate_processed_data(source_info):
    print("\nSTEP 3: Validating processed HDF5 data")

    with h5py.File(PROCESSED_H5, "r") as file:
        cube = file["processed_cube"][:]
        labels = file["labels"][:]

        print("Processed cube shape:", cube.shape)
        print("Processed labels shape:", labels.shape)

        if cube.shape[:2] != source_info["shape"]:
            raise ValueError("Processed cube dimensions do not match GeoTIFF.")

        if labels.shape != source_info["shape"]:
            raise ValueError("Processed labels dimensions do not match GeoTIFF.")

        if not np.isfinite(cube).all():
            raise ValueError("Processed cube contains invalid values.")

        if not np.isin(np.unique(labels), [0, 1]).all():
            raise ValueError("Processed labels contain unexpected classes.")

    print("Processed dataset validation: PASSED")

    return labels


# --------------------------------------------------
# 5. Load Day 9 predictions
# --------------------------------------------------

def load_predictions(expected_shape):
    print("\nSTEP 4: Loading Day 9 predictions")

    with h5py.File(PREDICTION_H5, "r") as file:
        print("Available prediction datasets:", list(file.keys()))

        if "prediction_map" not in file:
            raise KeyError(
                "The prediction_map dataset was not found in Day 9 HDF5."
            )

        if "stress_probability_map" not in file:
            raise KeyError(
                "The stress_probability_map dataset was not found in Day 9 HDF5."
            )

        prediction_map = file["prediction_map"][:]
        probability_map = file["stress_probability_map"][:]

    print("Prediction shape:", prediction_map.shape)
    print("Probability shape:", probability_map.shape)

    if prediction_map.shape != expected_shape:
        raise ValueError("Prediction map dimensions do not match source image.")

    if probability_map.shape != expected_shape:
        raise ValueError("Probability map dimensions do not match source image.")

    if not np.isfinite(probability_map).all():
        raise ValueError("Probability map contains invalid values.")

    if not np.isin(np.unique(prediction_map), [0, 1]).all():
        raise ValueError("Prediction map contains unexpected classes.")

    if probability_map.min() < 0 or probability_map.max() > 1:
        raise ValueError("Stress probabilities must be between 0 and 1.")

    print("Prediction validation: PASSED")

    return prediction_map.astype(np.uint8), probability_map.astype(np.float32)


# --------------------------------------------------
# 6. Evaluate model predictions
# --------------------------------------------------

def evaluate_predictions(true_labels, predictions):
    print("\nSTEP 5: Evaluating model predictions")

    y_true = true_labels.flatten()
    y_pred = predictions.flatten()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Healthy", "Stressed"],
        output_dict=True,
        zero_division=0,
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


# --------------------------------------------------
# 7. Export prediction maps as GeoTIFFs
# --------------------------------------------------

def export_geotiffs(source_info, predictions, probabilities):
    print("\nSTEP 6: Exporting GIS prediction files")

    with rasterio.open(SOURCE_TIF) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        height=predictions.shape[0],
        width=predictions.shape[1],
        count=1,
        dtype="uint8",
        compress="lzw",
        nodata=None,
    )

    with rasterio.open(PREDICTION_TIF, "w", **profile) as dst:
        dst.write(predictions, 1)
        dst.set_band_description(1, "Predicted crop health class")
        dst.update_tags(
            project="TerraSpectra",
            data_type="synthetic development data",
            class_0="Healthy",
            class_1="Stressed",
            real_world_validation="False",
        )

    probability_profile = profile.copy()
    probability_profile.update(dtype="float32")

    with rasterio.open(PROBABILITY_TIF, "w", **probability_profile) as dst:
        dst.write(probabilities, 1)
        dst.set_band_description(1, "Model-predicted stress probability")
        dst.update_tags(
            project="TerraSpectra",
            data_type="synthetic development data",
            real_world_validation="False",
        )

    print("Prediction GeoTIFF saved:", PREDICTION_TIF)
    print("Probability GeoTIFF saved:", PROBABILITY_TIF)


# --------------------------------------------------
# 8. Save validation report
# --------------------------------------------------

def save_report(source_info, metrics):
    transform = source_info["transform"]
    bounds = source_info["bounds"]

    report = {
        "project": "TerraSpectra",
        "day": 10,
        "validation_type": "GIS alignment and synthetic model evaluation",
        "dataset_type": "Synthetic development data",
        "real_world_validation": False,
        "source_image": {
            "shape": list(source_info["shape"]),
            "bands": source_info["band_count"],
            "crs": str(source_info["crs"]),
            "transform": list(transform)[:6],
            "bounds": list(bounds),
            "pixel_width": abs(transform.a),
            "pixel_height": abs(transform.e),
        },
        "model_metrics": metrics,
        "generated_files": [
            str(PREDICTION_TIF.relative_to(PROJECT_ROOT)),
            str(PROBABILITY_TIF.relative_to(PROJECT_ROOT)),
        ],
        "limitations": [
            "The dataset is synthetic.",
            "The coordinates are not verified real farm coordinates.",
            "Pixel predictions are not validated crop disease diagnoses.",
            "Stress probabilities are not calibrated real-world risk estimates.",
            "Physical acreage cannot be calculated without verified spatial resolution.",
        ],
    }

    # A one-degree pixel size in EPSG:4326 is a warning sign for
    # synthetic coordinates, not a measurement of a real farm.
    if source_info["crs"].is_geographic and (
        abs(transform.a) >= 0.1 or abs(transform.e) >= 0.1
    ):
        report["spatial_warning"] = (
            "The geographic pixel size is unusually large for field-level "
            "mapping. Treat this transform as synthetic, not real farm location data."
        )

    with open(REPORT_JSON, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nValidation report saved:", REPORT_JSON)


# --------------------------------------------------
# 9. Main execution
# --------------------------------------------------

def main():
    required_files = [
        SOURCE_TIF,
        LABEL_TIF,
        PROCESSED_H5,
        PREDICTION_H5,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    source_info = inspect_source_geotiff()

    label_tif_array = validate_label_geotiff(source_info)
    processed_labels = validate_processed_data(source_info)

    if not np.array_equal(label_tif_array, processed_labels):
        raise ValueError(
            "Label GeoTIFF and processed HDF5 labels do not match."
        )

    predictions, probabilities = load_predictions(source_info["shape"])

    metrics = evaluate_predictions(processed_labels, predictions)

    export_geotiffs(source_info, predictions, probabilities)
    save_report(source_info, metrics)

    print("\n========================================")
    print("DAY 10 VALIDATION COMPLETED SUCCESSFULLY")
    print("========================================")


if __name__ == "__main__":
    main()