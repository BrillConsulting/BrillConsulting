"""
MLflow ML Lifecycle Management
Author: BrillConsulting
Description: Advanced ML experiment tracking, model registry, and deployment with MLflow
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class ExperimentTracker:
    """Advanced MLflow experiment tracking"""

    def __init__(self, tracking_uri: str = 'http://localhost:5000'):
        """
        Initialize experiment tracker

        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        self.runs = []
        self.experiments = []

    def create_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new MLflow experiment

        Args:
            config: Experiment configuration

        Returns:
            Experiment details
        """
        print(f"\n{'='*60}")
        print("Creating MLflow Experiment")
        print(f"{'='*60}")

        experiment_name = config.get('name', 'default_experiment')
        tags = config.get('tags', {})

        code = f"""
import mlflow

mlflow.set_tracking_uri("{self.tracking_uri}")

# Create experiment
experiment_id = mlflow.create_experiment(
    "{experiment_name}",
    tags={{
        "team": "data-science",
        "project": "production",
        "version": "1.0"
    }}
)

# Or set existing experiment
mlflow.set_experiment("{experiment_name}")
"""

        result = {
            'experiment_name': experiment_name,
            'experiment_id': len(self.experiments) + 1,
            'tags': tags,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.experiments.append(result)

        print(f"✓ Experiment created: {experiment_name}")
        print(f"  ID: {result['experiment_id']}")
        print(f"{'='*60}")

        return result

    def log_training_run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log comprehensive training run

        Args:
            config: Training configuration

        Returns:
            Run details
        """
        print(f"\n{'='*60}")
        print("Logging Training Run")
        print(f"{'='*60}")

        run_name = config.get('run_name', 'training_run')
        params = config.get('params', {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        })

        # Simulate training metrics
        num_epochs = params.get('epochs', 100)
        metrics_history = []

        for epoch in range(0, num_epochs, 10):
            train_loss = 2.0 - 1.5 * (epoch / num_epochs) + np.random.uniform(-0.05, 0.05)
            val_loss = 2.1 - 1.4 * (epoch / num_epochs) + np.random.uniform(-0.05, 0.05)
            accuracy = 0.5 + 0.45 * (epoch / num_epochs) + np.random.uniform(-0.02, 0.02)

            metrics_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': accuracy
            })

        code = f"""
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

mlflow.set_experiment("{config.get('experiment_name', 'default')}")

with mlflow.start_run(run_name="{run_name}"):
    # Log parameters
    params = {params}
    mlflow.log_params(params)

    # Train model and log metrics per epoch
    for epoch in range({num_epochs}):
        # Training step
        train_loss = train_model(epoch)
        val_loss = validate_model(epoch)
        accuracy = evaluate_model(epoch)

        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)

    # Log final metrics
    mlflow.log_metric("final_accuracy", accuracy)
    mlflow.log_metric("final_val_loss", val_loss)

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=mlflow.models.infer_signature(X_train, model.predict(X_train))
    )

    # Log training plots
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    mlflow.log_artifact('training_curve.png')

    # Log model artifacts
    mlflow.log_artifact('model_config.json')
    mlflow.log_dict(params, 'parameters.json')

    print(f"Run ID: {{mlflow.active_run().info.run_id}}")
"""

        result = {
            'run_name': run_name,
            'run_id': f"run_{len(self.runs) + 1}",
            'params': params,
            'final_metrics': {
                'train_loss': metrics_history[-1]['train_loss'],
                'val_loss': metrics_history[-1]['val_loss'],
                'accuracy': metrics_history[-1]['accuracy']
            },
            'metrics_history': metrics_history,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.runs.append(result)

        print(f"✓ Run logged: {run_name}")
        print(f"  Final accuracy: {result['final_metrics']['accuracy']:.4f}")
        print(f"  Final val_loss: {result['final_metrics']['val_loss']:.4f}")
        print(f"{'='*60}")

        return result

    def log_hyperparameter_sweep(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log hyperparameter tuning sweep

        Args:
            config: Sweep configuration

        Returns:
            Sweep results
        """
        print(f"\n{'='*60}")
        print("Hyperparameter Sweep")
        print(f"{'='*60}")

        param_grid = config.get('param_grid', {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64],
            'optimizer': ['adam', 'sgd']
        })

        # Simulate sweep
        sweep_results = []
        best_score = 0
        best_params = {}

        for lr in param_grid.get('learning_rate', [0.01]):
            for bs in param_grid.get('batch_size', [32]):
                for opt in param_grid.get('optimizer', ['adam']):
                    # Simulate training
                    score = 0.7 + 0.2 * np.random.rand()

                    params = {
                        'learning_rate': lr,
                        'batch_size': bs,
                        'optimizer': opt
                    }

                    sweep_results.append({
                        'params': params,
                        'score': score
                    })

                    if score > best_score:
                        best_score = score
                        best_params = params
                        print(f"  New best: {score:.4f} - LR={lr}, BS={bs}, Opt={opt}")

        code = """
import mlflow
from itertools import product

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'optimizer': ['adam', 'sgd', 'rmsprop']
}

# Generate all combinations
keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in product(*values)]

# Run experiments
for params in experiments:
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = train_model(**params)
        score = evaluate_model(model)

        # Log metrics
        mlflow.log_metric("accuracy", score)
        mlflow.log_metric("val_loss", loss)

        # Log model if best
        if score > best_score:
            mlflow.sklearn.log_model(model, "best_model")
            mlflow.set_tag("best_run", "true")
"""

        result = {
            'param_grid': param_grid,
            'num_runs': len(sweep_results),
            'best_params': best_params,
            'best_score': best_score,
            'all_results': sweep_results,
            'code': code
        }

        print(f"\n✓ Sweep completed!")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Best params: {best_params}")
        print(f"{'='*60}")

        return result


class ModelRegistry:
    """MLflow Model Registry management"""

    def __init__(self):
        """Initialize model registry"""
        self.models = []
        self.versions = {}

    def register_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register model in registry

        Args:
            config: Model configuration

        Returns:
            Registration details
        """
        print(f"\n{'='*60}")
        print("Model Registration")
        print(f"{'='*60}")

        model_name = config.get('name', 'my_model')
        run_id = config.get('run_id', 'run_12345')

        # Increment version
        if model_name not in self.versions:
            self.versions[model_name] = 0
        self.versions[model_name] += 1
        version = self.versions[model_name]

        code = f"""
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(
    model_uri,
    "{model_name}",
    tags={{
        "framework": "sklearn",
        "task": "classification"
    }}
)

print(f"Model {{result.name}} version {{result.version}} registered")

# Add description
client.update_model_version(
    name="{model_name}",
    version=result.version,
    description="Production model for fraud detection"
)
"""

        result = {
            'model_name': model_name,
            'version': version,
            'run_id': run_id,
            'stage': 'None',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.models.append(result)

        print(f"✓ Model registered: {model_name} v{version}")
        print(f"  Run ID: {run_id}")
        print(f"{'='*60}")

        return result

    def transition_model_stage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transition model to different stage

        Args:
            config: Transition configuration

        Returns:
            Transition details
        """
        print(f"\n{'='*60}")
        print("Model Stage Transition")
        print(f"{'='*60}")

        model_name = config.get('name', 'my_model')
        version = config.get('version', 1)
        stage = config.get('stage', 'Production')

        code = f"""
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition to staging
client.transition_model_version_stage(
    name="{model_name}",
    version={version},
    stage="{stage}",
    archive_existing_versions=True
)

# Get model version details
model_version = client.get_model_version(
    name="{model_name}",
    version={version}
)

print(f"Model {{model_version.name}} v{{model_version.version}}")
print(f"Stage: {{model_version.current_stage}}")
print(f"Status: {{model_version.status}}")
"""

        result = {
            'model_name': model_name,
            'version': version,
            'new_stage': stage,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Model transitioned: {model_name} v{version}")
        print(f"  New stage: {stage}")
        print(f"{'='*60}")

        return result

    def load_production_model(self, model_name: str) -> str:
        """
        Get code to load production model

        Args:
            model_name: Name of the model

        Returns:
            Loading code
        """
        code = f"""
import mlflow
from mlflow.tracking import MlflowClient

# Load latest production model
model_name = "{model_name}"
model_version_uri = f"models:/{{model_name}}/Production"

# Load model
loaded_model = mlflow.pyfunc.load_model(model_version_uri)

# Or load specific version
model_version = 3
model_uri = f"models:/{{model_name}}/{{model_version}}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Make predictions
predictions = loaded_model.predict(X_test)

# Get model metadata
client = MlflowClient()
model_version_details = client.get_model_version(
    name=model_name,
    version=model_version
)

print(f"Loaded model: {{model_version_details.name}}")
print(f"Version: {{model_version_details.version}}")
print(f"Stage: {{model_version_details.current_stage}}")
print(f"Run ID: {{model_version_details.run_id}}")
"""

        print(f"\n✓ Production model loading code generated for: {model_name}")
        return code


class MLflowManager:
    """Comprehensive MLflow management"""

    def __init__(self, tracking_uri: str = 'http://localhost:5000'):
        """
        Initialize MLflow manager

        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        self.tracker = ExperimentTracker(tracking_uri)
        self.registry = ModelRegistry()

    def setup_autolog(self, framework: str = 'sklearn') -> str:
        """
        Setup MLflow autologging

        Args:
            framework: ML framework name

        Returns:
            Autolog setup code
        """
        print(f"\n{'='*60}")
        print(f"MLflow Autologging - {framework}")
        print(f"{'='*60}")

        if framework == 'sklearn':
            code = """
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Enable autologging
mlflow.sklearn.autolog()

# Train model - everything logged automatically
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Predictions and metrics logged automatically
    predictions = model.predict(X_test)

# Logged automatically:
# - Model parameters
# - Training metrics
# - Model artifact
# - Feature importance
# - Training dataset signature
"""
        elif framework == 'tensorflow':
            code = """
import mlflow
import mlflow.tensorflow
import tensorflow as tf

# Enable autologging
mlflow.tensorflow.autolog()

with mlflow.start_run():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Everything logged automatically
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Logged automatically:
# - Optimizer parameters
# - Loss and metrics per epoch
# - Model architecture
# - TensorBoard logs
"""
        elif framework == 'pytorch':
            code = """
import mlflow
import mlflow.pytorch
import torch

# Enable autologging
mlflow.pytorch.autolog()

with mlflow.start_run():
    model = YourPyTorchModel()

    for epoch in range(10):
        # Training loop
        loss = train_epoch(model)

        # Log manually within autolog
        mlflow.log_metric("epoch_loss", loss, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, "model")
"""
        else:
            code = "# Unsupported framework"

        print(f"✓ Autologging code generated for {framework}")
        print(f"{'='*60}")

        return code

    def compare_runs(self, run_ids: List[str]) -> str:
        """
        Generate code to compare multiple runs

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Comparison code
        """
        code = """
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()

# Get runs
run_ids = ['run1', 'run2', 'run3']
runs = [client.get_run(run_id) for run_id in run_ids]

# Compare metrics
comparison = []
for run in runs:
    comparison.append({
        'run_id': run.info.run_id,
        'accuracy': run.data.metrics.get('accuracy'),
        'val_loss': run.data.metrics.get('val_loss'),
        'learning_rate': run.data.params.get('learning_rate'),
        'batch_size': run.data.params.get('batch_size')
    })

df = pd.DataFrame(comparison)
print(df.sort_values('accuracy', ascending=False))

# Search runs by metrics
runs = client.search_runs(
    experiment_ids=['1'],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)
"""

        print(f"\n✓ Run comparison code generated")
        return code

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'tracking_uri': self.tracking_uri,
            'experiments': len(self.tracker.experiments),
            'runs': len(self.tracker.runs),
            'registered_models': len(self.registry.models),
            'features': ['experiment_tracking', 'model_registry', 'autologging', 'deployment'],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate MLflow"""
    print("=" * 60)
    print("MLflow ML Lifecycle Management Demo")
    print("=" * 60)

    mgr = MLflowManager()

    # Create experiment
    print("\n1. Creating experiment...")
    exp_result = mgr.tracker.create_experiment({
        'name': 'fraud_detection_v2',
        'tags': {'team': 'ml-ops', 'project': 'fraud'}
    })

    # Log training run
    print("\n2. Logging training run...")
    run_result = mgr.tracker.log_training_run({
        'run_name': 'baseline_model',
        'experiment_name': 'fraud_detection_v2',
        'params': {
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 50
        }
    })

    # Hyperparameter sweep
    print("\n3. Hyperparameter tuning sweep...")
    sweep_result = mgr.tracker.log_hyperparameter_sweep({
        'param_grid': {
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'optimizer': ['adam', 'sgd']
        }
    })

    # Register model
    print("\n4. Registering model...")
    model_result = mgr.registry.register_model({
        'name': 'fraud_detector',
        'run_id': run_result['run_id']
    })

    # Transition model stage
    print("\n5. Transitioning model to production...")
    stage_result = mgr.registry.transition_model_stage({
        'name': 'fraud_detector',
        'version': 1,
        'stage': 'Production'
    })

    # Autologging
    print("\n6. Autologging setup for sklearn...")
    autolog_code = mgr.setup_autolog('sklearn')
    print(autolog_code[:300] + "...\n")

    # Load production model
    print("\n7. Production model loading code...")
    load_code = mgr.registry.load_production_model('fraud_detector')
    print(load_code[:300] + "...\n")

    # Manager info
    print("\n8. Manager summary:")
    info = mgr.get_manager_info()
    print(f"  Experiments: {info['experiments']}")
    print(f"  Runs: {info['runs']}")
    print(f"  Registered models: {info['registered_models']}")
    print(f"  Features: {', '.join(info['features'])}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
