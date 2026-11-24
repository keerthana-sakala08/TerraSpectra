
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "day15"
    / "crop_health_analytics.json"
)


app = FastAPI(
    title="TerraSpectra API",
    description=(
        "AI-Powered Multilingual Smart Agriculture "
        "and Crop Management Platform"
    ),
    version="2.0.0"
)


# Allow the React development server to access FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "project": "TerraSpectra",
        "description": (
            "AI-Powered Multilingual Smart Agriculture Platform"
        ),
        "features": [
            "Mobile Camera Crop Scanning",
            "Hyperspectral Crop Scanning",
            "Crop Health Detection",
            "Crop Damage Analysis",
            "Disease Warning System",
            "Treatment Guidance",
            "Soil Analysis",
            "Weather Information",
            "Irrigation Guidance",
            "Crop Recommendations",
            "Agricultural Marketplace",
            "Multilingual Assistance"
        ],
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TerraSpectra Backend",
        "version": "2.0.0"
    }


@app.get("/api/languages")
def get_languages():
    return {
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "te", "name": "Telugu"},
            {"code": "hi", "name": "Hindi"},
            {"code": "ta", "name": "Tamil"},
            {"code": "kn", "name": "Kannada"},
            {"code": "ml", "name": "Malayalam"}
        ]
    }


@app.get("/api/analytics/crop-health")
def get_crop_health_analytics():
    if not ANALYTICS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Crop health analytics report not found. "
                   "Run the Day 15 analytics script first."
        )

    try:
        with ANALYTICS_FILE.open("r", encoding="utf-8") as file:
            analytics = json.load(file)

        return analytics

    except (json.JSONDecodeError, OSError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read analytics report: {error}"
        )
