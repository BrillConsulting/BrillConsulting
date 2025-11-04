"""
ML Training Pipeline
====================

Automated end-to-end training pipeline for ML models:
- Data loading and preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Model validation
- Artifact management

Author: Brill Consulting
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class TrainingPipeline:
    """End-to-end ML training pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration."""
        self.config = config
        self.artifacts_dir = Path(config.get("artifacts_dir", "./artifacts"))
        self.artifacts_dir.mkdir(exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_data(self, data_path: str) -> Dict:
        """Load and validate data."""
        print(f"Loading data from {data_path}...")
        # Simulate data loading
        data = {
            "X_train": np.random.randn(1000, 10),
            "y_train": np.random.randint(0, 2, 1000),
            "X_test": np.random.randn(200, 10),
            "y_test": np.random.randint(0, 2, 200)
        }
        print(f"✓ Loaded {len(data['X_train'])} training samples")
        return data

    def preprocess(self, data: Dict) -> Dict:
        """Preprocess data."""
        print("Preprocessing data...")
        # Normalize
        mean = data["X_train"].mean(axis=0)
        std = data["X_train"].std(axis=0)

        data["X_train"] = (data["X_train"] - mean) / (std + 1e-8)
        data["X_test"] = (data["X_test"] - mean) / (std + 1e-8)

        self.save_artifact("preprocessing", {"mean": mean, "std": std})
        print("✓ Preprocessing complete")
        return data

    def train_model(self, data: Dict) -> Any:
        """Train model."""
        print("Training model...")
        # Simulate training
        model = {"type": "classifier", "trained": True}
        print("✓ Training complete")
        return model

    def evaluate_model(self, model: Any, data: Dict) -> Dict:
        """Evaluate model performance."""
        print("Evaluating model...")
        # Simulate evaluation
        metrics = {
            "accuracy": 0.85 + np.random.uniform(-0.05, 0.05),
            "precision": 0.83 + np.random.uniform(-0.05, 0.05),
            "recall": 0.86 + np.random.uniform(-0.05, 0.05),
            "f1": 0.84 + np.random.uniform(-0.05, 0.05)
        }
        print(f"✓ Evaluation: Accuracy={metrics['accuracy']:.4f}")
        return metrics

    def save_artifact(self, name: str, artifact: Any):
        """Save pipeline artifact."""
        artifact_path = self.artifacts_dir / f"{name}_{self.run_id}.pkl"
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f)
        print(f"✓ Saved {name} to {artifact_path}")

    def save_metadata(self, metrics: Dict):
        """Save run metadata."""
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "metrics": metrics
        }

        metadata_path = self.artifacts_dir / f"metadata_{self.run_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_path}")

    def run(self, data_path: str) -> Dict:
        """Run complete pipeline."""
        print(f"\n{'='*50}")
        print(f"Starting Training Pipeline - Run {self.run_id}")
        print(f"{'='*50}\n")

        # Load data
        data = self.load_data(data_path)

        # Preprocess
        data = self.preprocess(data)

        # Train
        model = self.train_model(data)

        # Evaluate
        metrics = self.evaluate_model(model, data)

        # Save artifacts
        self.save_artifact("model", model)
        self.save_metadata(metrics)

        print(f"\n{'='*50}")
        print("Pipeline Complete!")
        print(f"{'='*50}\n")

        return {
            "run_id": self.run_id,
            "metrics": metrics,
            "artifacts_dir": str(self.artifacts_dir)
        }


def demo():
    """Demo training pipeline."""
    config = {
        "artifacts_dir": "./ml_artifacts",
        "model_type": "classifier"
    }

    pipeline = TrainingPipeline(config)
    result = pipeline.run("data.csv")

    print("\nPipeline Result:")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    demo()
