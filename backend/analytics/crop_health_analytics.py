
from pathlib import Path
import json

import numpy as np
import rasterio


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
    PROJECT_ROOT / "outputs" / "day10" / "predicted_crop_health.tif"
)
PROBABILITY_FILE = (
    PROJECT_ROOT / "outputs" / "day10" / "stress_probability.tif"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "day15"
OUTPUT_FILE = OUTPUT_DIR / "crop_health_analytics.json"


def calculate_crop_health_analytics():
    print("\nTERRASPECTRA - DAY 15")
    print("CROP HEALTH ANALYTICS")
    print("-" * 45)

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found: {PREDICTION_FILE}")

    if not PROBABILITY_FILE.exists():
        raise FileNotFoundError(f"Probability file not found: {PROBABILITY_FILE}")

    # Read prediction and probability rasters
    with rasterio.open(PREDICTION_FILE) as prediction_src:
        prediction = prediction_src.read(1, masked=True)
        prediction_shape = prediction_src.shape
        prediction_crs = prediction_src.crs
        prediction_transform = prediction_src.transform

    with rasterio.open(PROBABILITY_FILE) as probability_src:
        probability = probability_src.read(1, masked=True)

        if probability_src.shape != prediction_shape:
            raise ValueError("Prediction and probability shapes do not match.")

        if probability_src.crs != prediction_crs:
            raise ValueError("Prediction and probability CRS do not match.")

        if probability_src.transform != prediction_transform:
            raise ValueError("Prediction and probability transforms do not match.")

    prediction_data = np.asarray(prediction.data)
    probability_data = np.asarray(probability.data)

    # Exclude invalid or nodata pixels
    valid_mask = (
        ~np.ma.getmaskarray(prediction)
        & ~np.ma.getmaskarray(probability)
        & np.isfinite(prediction_data)
        & np.isfinite(probability_data)
    )

    valid_predictions = prediction_data[valid_mask].astype(int)
    valid_probabilities = probability_data[valid_mask].astype(float)

    if valid_predictions.size == 0:
        raise ValueError("No valid pixels available for analytics.")

    if not np.isin(valid_predictions, [0, 1]).all():
        raise ValueError("Unexpected prediction classes. Expected 0 and 1.")

    if (
        np.any(valid_probabilities < 0)
        or np.any(valid_probabilities > 1)
    ):
        raise ValueError("Stress probabilities must be between 0 and 1.")

    # Class definitions: 0 = Healthy, 1 = Stressed
    total_pixels = int(valid_predictions.size)
    healthy_pixels = int(np.sum(valid_predictions == 0))
    stressed_pixels = int(np.sum(valid_predictions == 1))

    healthy_percentage = (healthy_pixels / total_pixels) * 100
    stressed_percentage = (stressed_pixels / total_pixels) * 100

    average_stress_probability = float(np.mean(valid_probabilities))

    # Probability-based analytical categories
    high_risk_pixels = int(np.sum(valid_probabilities >= 0.70))
    medium_risk_pixels = int(
        np.sum(
            (valid_probabilities >= 0.40)
            & (valid_probabilities < 0.70)
        )
    )
    low_risk_pixels = int(np.sum(valid_probabilities < 0.40))

    # Prepare report
    report = {
        "project": "TerraSpectra",
        "day": 15,
        "analytics_type": "Crop Health Analytics",
        "dataset_type": "synthetic",
        "real_world_validation": False,
        "class_definitions": {
            "0": "Healthy",
            "1": "Stressed"
        },
        "total_valid_pixels": total_pixels,
        "healthy_pixels": healthy_pixels,
        "stressed_pixels": stressed_pixels,
        "healthy_percentage": round(healthy_percentage, 2),
        "stressed_percentage": round(stressed_percentage, 2),
        "average_stress_probability": round(
            average_stress_probability, 4
        ),
        "risk_distribution": {
            "low_risk_pixels": low_risk_pixels,
            "medium_risk_pixels": medium_risk_pixels,
            "high_risk_pixels": high_risk_pixels
        },
        "risk_thresholds": {
            "low": "Below 0.40",
            "medium": "0.40 to below 0.70",
            "high": "0.70 and above"
        },
        "limitations": [
            "The dataset is synthetic.",
            "Results are not validated for real agricultural fields.",
            "Affected acreage is not calculated because verified "
            "ground resolution is unavailable."
        ]
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Total valid pixels:", total_pixels)
    print("Healthy pixels:", healthy_pixels)
    print("Stressed pixels:", stressed_pixels)
    print("Healthy percentage:", round(healthy_percentage, 2), "%")
    print("Stressed percentage:", round(stressed_percentage, 2), "%")
    print(
        "Average stress probability:",
        round(average_stress_probability, 4)
    )
    print("High-risk pixels:", high_risk_pixels)
    print("\nAnalytics report saved:", OUTPUT_FILE)
    print("\nDAY 15 ANALYTICS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    calculate_crop_health_analytics()
