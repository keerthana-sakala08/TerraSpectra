from fastapi import FastAPI

app = FastAPI(
    title="TerraSpectra API",
    description="Hyperspectral Crop Disease Forecasting Platform",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "project": "TerraSpectra",
        "message": "Hyperspectral Crop Disease Forecasting API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TerraSpectra Backend"
    }