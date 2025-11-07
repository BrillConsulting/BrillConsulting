"""
Model Registry & Versioning
============================

MLflow-based registry with A/B testing

Author: Brill Consulting
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelVersion:
    """Model version metadata."""
    name: str
    version: int
    metrics: Dict[str, float]
    stage: str = "staging"
    registered_at: str = ""


class ModelRegistry:
    """Model registry with versioning."""

    def __init__(self, tracking_uri: str = "localhost:5000"):
        """Initialize registry."""
        self.tracking_uri = tracking_uri
        self.models: Dict[str, List[ModelVersion]] = {}

        print(f"📚 Model Registry initialized")
        print(f"   URI: {tracking_uri}")

    def register_model(
        self,
        name: str,
        model_path: str,
        metrics: Dict[str, float],
        tags: Dict[str, str] = None
    ) -> int:
        """Register new model version."""
        if name not in self.models:
            self.models[name] = []

        version = len(self.models[name]) + 1

        model_version = ModelVersion(
            name=name,
            version=version,
            metrics=metrics,
            stage="staging",
            registered_at=datetime.now().isoformat()
        )

        self.models[name].append(model_version)

        print(f"   ✓ Registered: {name} v{version}")
        print(f"      Metrics: {metrics}")

        return version

    def promote_to_production(self, model_name: str, version: int) -> None:
        """Promote model to production."""
        model = self.models[model_name][version - 1]
        model.stage = "production"

        print(f"   🚀 Promoted: {model_name} v{version} → production")

    def create_ab_test(
        self,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.1
    ) -> Dict:
        """Create A/B test."""
        print(f"\n🧪 A/B Test created")
        print(f"   Model A: {model_a} ({(1-traffic_split)*100:.0f}%)")
        print(f"   Model B: {model_b} ({traffic_split*100:.0f}%)")

        return {
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "status": "running"
        }


def demo():
    """Demonstrate model registry."""
    print("=" * 60)
    print("Model Registry Demo")
    print("=" * 60)

    registry = ModelRegistry()

    # Register versions
    v1 = registry.register_model(
        "llama2-7b-chat",
        "s3://models/v1",
        {"perplexity": 5.47, "accuracy": 0.856}
    )

    v2 = registry.register_model(
        "llama2-7b-chat",
        "s3://models/v2",
        {"perplexity": 5.32, "accuracy": 0.871}
    )

    # Promote
    registry.promote_to_production("llama2-7b-chat", v2)

    # A/B test
    registry.create_ab_test(
        f"llama2-7b-chat:v{v1}",
        f"llama2-7b-chat:v{v2}",
        traffic_split=0.1
    )


if __name__ == "__main__":
    demo()
