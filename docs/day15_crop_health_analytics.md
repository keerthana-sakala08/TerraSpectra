# Day 15 – Crop Health Analytics and Dashboard

## 1. Objective

The objective of Day 15 was to develop a Crop Health Analytics module for TerraSpectra. It analyzes hyperspectral crop-health prediction outputs and generates statistical information about healthy and stressed pixels.

The analytics results are made available through a FastAPI endpoint and displayed on a React dashboard.

## 2. Implementation

### 2.1 Input Data

The analytics module uses the following outputs generated during previous development stages:

- `outputs/day10/predicted_crop_health.tif`
- `outputs/day10/stress_probability.tif`

The prediction classes are:

| Class | Meaning |
|---|---|
| 0 | Healthy |
| 1 | Stressed |

### 2.2 Analytics Module

Created:

`backend/analytics/crop_health_analytics.py`

The module performs the following operations:

1. Loads crop-health prediction and stress-probability raster files.
2. Validates raster dimensions, coordinate reference system, and transformations.
3. Checks valid pixel values and probability ranges.
4. Counts healthy and stressed pixels.
5. Calculates healthy and stressed percentages.
6. Calculates average stress probability.
7. Categorizes pixels into low-, medium-, and high-risk groups.
8. Saves the analytics report as JSON.

### 2.3 Analytics Output

Generated report:

`outputs/day15/crop_health_analytics.json`

The following results were obtained:

| Metric | Result |
|---|---:|
| Total valid pixels | 16,384 |
| Healthy pixels | 14,157 |
| Stressed pixels | 2,227 |
| Healthy percentage | 86.41% |
| Stressed percentage | 13.59% |
| Average stress probability | 0.1615 |

### 2.4 Risk Distribution

Risk categories were calculated using stress-probability thresholds.

| Risk category | Probability range | Pixel count |
|---|---|---:|
| Low risk | Below 0.40 | 14,134 |
| Medium risk | 0.40 to below 0.70 | 122 |
| High risk | 0.70 and above | 2,128 |

### 2.5 Backend API Integration

Updated:

`backend/app.py`

Added the following endpoint:

`GET /api/analytics/crop-health`

The endpoint reads the generated JSON analytics report and returns the crop-health statistics to the frontend.

The backend uses FastAPI and includes CORS configuration for local frontend development.

### 2.6 Frontend Dashboard

Updated the following React files:

- `frontend/src/App.jsx`
- `frontend/src/App.css`
- `frontend/src/index.css`

The dashboard is designed to display:

- Total valid pixels
- Healthy crop percentage
- Stressed crop percentage
- High-risk pixel count
- Crop-health progress indicators
- Risk distribution
- Dataset information
- Analytics limitations

The frontend retrieves analytics data from the FastAPI backend.

## 3. Technologies Used

- Python
- Rasterio
- NumPy
- JSON
- FastAPI
- React
- JavaScript
- CSS

## 4. Testing and Validation

The following checks were performed:

- Verified that the Day 10 prediction raster exists.
- Verified that the stress-probability raster exists.
- Executed the crop-health analytics Python module successfully.
- Generated the Day 15 JSON analytics report.
- Verified that the generated report exists.
- Tested the crop-health analytics API endpoint and received the expected JSON response.

## 5. Limitations

- The current dataset is synthetic.
- The results have not been validated using real agricultural field data.
- The results should not be treated as confirmed real-world crop diagnoses.
- Affected acreage is not calculated because verified ground resolution is unavailable.
- Real-world reliability requires testing with labelled agricultural datasets.

## 6. Outcome

Day 15 established the crop-health analytics pipeline and its backend API integration. It provides statistical summaries of healthy and stressed pixels and makes those results available to a React dashboard.

The module forms a foundation for future crop-health visualization, risk monitoring, and integration with additional agricultural analysis features.

**Status:** Day 15 analytics implementation and documentation prepared.
