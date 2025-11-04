"""
Azure Machine Learning
======================

Azure ML workspace and model management:
- Workspace management
- Dataset registration
- Model training and registration
- Model deployment
- Experiment tracking

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json


class AzureMLWorkspace:
    """Azure ML Workspace."""

    def __init__(self, name: str, subscription_id: str, resource_group: str, location: str = "eastus"):
        self.name = name
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.location = location
        self.datasets = {}
        self.experiments = {}
        self.models = {}
        self.endpoints = {}

    def get_workspace_info(self) -> Dict:
        """Get workspace information."""
        return {
            "name": self.name,
            "subscription_id": self.subscription_id,
            "resource_group": self.resource_group,
            "location": self.location,
            "datasets": len(self.datasets),
            "experiments": len(self.experiments),
            "models": len(self.models),
            "endpoints": len(self.endpoints)
        }


class AzureMLDataset:
    """Azure ML Dataset."""

    def __init__(self, workspace: AzureMLWorkspace, name: str, description: str = ""):
        self.workspace = workspace
        self.name = name
        self.description = description
        self.version = 1
        self.created_at = datetime.now()
        self.data = []

    def register(self, data_path: str, tags: Optional[Dict] = None) -> Dict:
        """Register dataset."""
        print(f"\n📊 Registering dataset: {self.name}")

        dataset_info = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "data_path": data_path,
            "tags": tags or {},
            "registered_at": self.created_at.isoformat()
        }

        self.workspace.datasets[self.name] = dataset_info
        print(f"✓ Dataset registered: {self.name} v{self.version}")

        return dataset_info

    def load_data(self, sample_size: int = 100) -> List[Dict]:
        """Load dataset (simulated)."""
        print(f"📥 Loading dataset: {self.name}")

        # Simulate loading data
        self.data = [
            {"feature_1": i, "feature_2": i * 2, "label": i % 2}
            for i in range(sample_size)
        ]

        print(f"✓ Loaded {len(self.data)} samples")
        return self.data


class AzureMLExperiment:
    """Azure ML Experiment for tracking."""

    def __init__(self, workspace: AzureMLWorkspace, name: str):
        self.workspace = workspace
        self.name = name
        self.runs = []
        self.created_at = datetime.now()

        workspace.experiments[name] = self

    def start_run(self, run_name: Optional[str] = None) -> 'AzureMLRun':
        """Start new experiment run."""
        run_name = run_name or f"run_{len(self.runs) + 1}"
        print(f"\n🏃 Starting run: {run_name}")

        run = AzureMLRun(self, run_name)
        self.runs.append(run)

        return run

    def get_runs(self) -> List['AzureMLRun']:
        """Get all runs."""
        return self.runs


class AzureMLRun:
    """Azure ML Run."""

    def __init__(self, experiment: AzureMLExperiment, name: str):
        self.experiment = experiment
        self.name = name
        self.status = "Running"
        self.metrics = {}
        self.parameters = {}
        self.tags = {}
        self.start_time = datetime.now()
        self.end_time = None

    def log_metric(self, name: str, value: float):
        """Log metric."""
        self.metrics[name] = value
        print(f"   📈 Metric logged: {name} = {value:.4f}")

    def log_parameter(self, name: str, value: Any):
        """Log parameter."""
        self.parameters[name] = value
        print(f"   ⚙️  Parameter logged: {name} = {value}")

    def tag(self, key: str, value: str):
        """Add tag."""
        self.tags[key] = value

    def complete(self):
        """Complete run."""
        self.status = "Completed"
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"\n✓ Run completed in {duration:.2f}s")

    def get_metrics(self) -> Dict:
        """Get all metrics."""
        return self.metrics


class AzureMLModel:
    """Azure ML Model."""

    def __init__(self, workspace: AzureMLWorkspace, name: str, version: int = 1):
        self.workspace = workspace
        self.name = name
        self.version = version
        self.tags = {}
        self.properties = {}
        self.registered_at = None

    def register(self, model_path: str, description: str = "", tags: Optional[Dict] = None) -> Dict:
        """Register model."""
        print(f"\n🤖 Registering model: {self.name}")

        self.registered_at = datetime.now()
        self.tags = tags or {}

        model_info = {
            "name": self.name,
            "version": self.version,
            "description": description,
            "model_path": model_path,
            "tags": self.tags,
            "registered_at": self.registered_at.isoformat()
        }

        self.workspace.models[f"{self.name}_v{self.version}"] = model_info
        print(f"✓ Model registered: {self.name} v{self.version}")

        return model_info

    def download(self, target_path: str):
        """Download model."""
        print(f"⬇️  Downloaded model to: {target_path}")


class AzureMLEndpoint:
    """Azure ML Endpoint for deployment."""

    def __init__(self, workspace: AzureMLWorkspace, name: str, model: AzureMLModel):
        self.workspace = workspace
        self.name = name
        self.model = model
        self.state = "Creating"
        self.scoring_uri = None
        self.swagger_uri = None

    def deploy(self, compute_type: str = "ACI", instance_type: str = "Standard_DS2_v2") -> Dict:
        """Deploy model to endpoint."""
        print(f"\n🚀 Deploying endpoint: {self.name}")
        print(f"   Model: {self.model.name} v{self.model.version}")
        print(f"   Compute: {compute_type} ({instance_type})")

        self.state = "Healthy"
        self.scoring_uri = f"https://{self.name}.{self.workspace.location}.inference.ml.azure.com/score"
        self.swagger_uri = f"https://{self.name}.{self.workspace.location}.inference.ml.azure.com/swagger.json"

        endpoint_info = {
            "name": self.name,
            "state": self.state,
            "scoring_uri": self.scoring_uri,
            "swagger_uri": self.swagger_uri,
            "model": f"{self.model.name}_v{self.model.version}",
            "deployed_at": datetime.now().isoformat()
        }

        self.workspace.endpoints[self.name] = endpoint_info
        print(f"✓ Endpoint deployed successfully")
        print(f"   Scoring URI: {self.scoring_uri}")

        return endpoint_info

    def predict(self, data: List[Dict]) -> List[Dict]:
        """Make prediction."""
        print(f"\n🔮 Making prediction with {len(data)} samples")

        # Simulate prediction
        predictions = [{"prediction": i % 2, "probability": 0.85} for i in range(len(data))]

        print(f"✓ Predictions completed")
        return predictions


def demo():
    """Demo Azure Machine Learning."""
    print("Azure Machine Learning Demo")
    print("=" * 60)

    # 1. Create Workspace
    print("\n1. Create ML Workspace")
    print("-" * 60)

    workspace = AzureMLWorkspace(
        name="ml-workspace-demo",
        subscription_id="12345678-1234-1234-1234-123456789abc",
        resource_group="rg-ml-demo",
        location="eastus"
    )

    print(f"✓ Workspace created: {workspace.name}")

    # 2. Register Dataset
    print("\n2. Register Dataset")
    print("-" * 60)

    dataset = AzureMLDataset(workspace, "customer-churn", "Customer churn prediction dataset")
    dataset.register("azureml://datasets/customer-churn.csv", tags={"type": "training", "version": "1.0"})

    data = dataset.load_data(sample_size=1000)

    # 3. Run Experiment
    print("\n3. Run Training Experiment")
    print("-" * 60)

    experiment = AzureMLExperiment(workspace, "churn-prediction")
    run = experiment.start_run("run_001")

    # Log parameters
    run.log_parameter("learning_rate", 0.01)
    run.log_parameter("batch_size", 32)
    run.log_parameter("epochs", 10)

    # Simulate training and log metrics
    run.log_metric("accuracy", 0.92)
    run.log_metric("precision", 0.89)
    run.log_metric("recall", 0.91)
    run.log_metric("f1_score", 0.90)

    run.tag("framework", "sklearn")
    run.tag("algorithm", "RandomForest")

    run.complete()

    # 4. Register Model
    print("\n4. Register Model")
    print("-" * 60)

    model = AzureMLModel(workspace, "churn-predictor", version=1)
    model.register(
        model_path="outputs/model.pkl",
        description="Random Forest model for churn prediction",
        tags={"accuracy": "0.92", "framework": "sklearn"}
    )

    # 5. Deploy Model
    print("\n5. Deploy Model to Endpoint")
    print("-" * 60)

    endpoint = AzureMLEndpoint(workspace, "churn-predictor-endpoint", model)
    deployment_info = endpoint.deploy(compute_type="ACI", instance_type="Standard_DS2_v2")

    # 6. Make Predictions
    print("\n6. Make Predictions")
    print("-" * 60)

    test_data = [
        {"feature_1": 10, "feature_2": 20},
        {"feature_1": 15, "feature_2": 30}
    ]

    predictions = endpoint.predict(test_data)
    print(f"\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: Class {pred['prediction']} (probability: {pred['probability']})")

    # 7. Workspace Summary
    print("\n7. Workspace Summary")
    print("-" * 60)

    info = workspace.get_workspace_info()
    print(f"  Workspace: {info['name']}")
    print(f"  Location: {info['location']}")
    print(f"  Datasets: {info['datasets']}")
    print(f"  Experiments: {info['experiments']}")
    print(f"  Models: {info['models']}")
    print(f"  Endpoints: {info['endpoints']}")

    print(f"\n  Registered Models:")
    for model_key, model_info in workspace.models.items():
        print(f"    • {model_info['name']} v{model_info['version']}")

    print(f"\n  Active Endpoints:")
    for endpoint_name, endpoint_info in workspace.endpoints.items():
        print(f"    • {endpoint_name} ({endpoint_info['state']})")

    print("\n✓ Azure Machine Learning Demo Complete!")


if __name__ == '__main__':
    demo()
