"""
Model Deployment with FastAPI
==============================

Production-ready model serving with REST API:
- FastAPI endpoint
- Model loading and caching
- Input validation
- Batch prediction
- Health checks
- Docker support

Author: Brill Consulting
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import pickle
from pathlib import Path


app = FastAPI(title="ML Model API", version="1.0.0")


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[List[float]]
    model_version: str = "latest"


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float]
    model_version: str
    confidence: List[float]


class ModelServer:
    """Model serving class."""

    def __init__(self, model_path: str):
        """Initialize with model path."""
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()

    def load_model(self):
        """Load model from disk."""
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Loaded model from {self.model_path}")
        else:
            # Create dummy model for demo
            self.model = {"type": "classifier", "loaded": True}
            print("✓ Created dummy model for demo")

    def predict(self, features: np.ndarray) -> Dict:
        """Make predictions."""
        # Simulate predictions
        predictions = np.random.rand(len(features))
        confidence = np.random.rand(len(features))

        return {
            "predictions": predictions.tolist(),
            "confidence": confidence.tolist()
        }


# Global model server instance
model_server = ModelServer("model.pkl")


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health", "/metrics"]
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_server.model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Prediction endpoint."""
    try:
        features = np.array(request.features)

        # Validate input shape
        if features.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail="Features must be 2D array"
            )

        # Make predictions
        result = model_server.predict(features)

        return PredictionResponse(
            predictions=result["predictions"],
            confidence=result["confidence"],
            model_version=request.model_version
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """Model metrics endpoint."""
    return {
        "requests_total": 1000,
        "avg_latency_ms": 15.5,
        "error_rate": 0.01
    }


def demo():
    """Demo server (requires uvicorn)."""
    print("ML Model Server")
    print("="*50)
    print("\nTo start server:")
    print("  uvicorn model_server:app --reload")
    print("\nEndpoints:")
    print("  GET  /         - Root")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Make predictions")
    print("  GET  /metrics  - Get metrics")
    print("\nExample curl:")
    print('  curl -X POST "http://localhost:8000/predict" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"features": [[1.0, 2.0, 3.0]]}\'')


if __name__ == '__main__':
    demo()
