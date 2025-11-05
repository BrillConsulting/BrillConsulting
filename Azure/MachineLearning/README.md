# Azure Machine Learning Integration

Comprehensive implementation of Azure Machine Learning for workspace management, model training, deployment, and MLOps.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a complete Python implementation for Azure Machine Learning, featuring workspace management, dataset registration, experiment tracking, model training and deployment, and MLOps automation. Built for enterprise machine learning workflows with Azure's ML platform.

## Features

### Workspace Management
- **Workspace Creation**: Configure ML workspaces
- **Compute Management**: CPU and GPU compute clusters
- **Environment Management**: Custom Python environments
- **Datastore Configuration**: Connect to data sources
- **Resource Monitoring**: Track resource usage

### Dataset & Data Management
- **Dataset Registration**: Register and version datasets
- **Data Profiling**: Understand data characteristics
- **Data Validation**: Quality checks and validation
- **Feature Engineering**: Transform and prepare data
- **Data Lineage**: Track data provenance

### Experiment Tracking
- **Run Management**: Track training runs
- **Metric Logging**: Record performance metrics
- **Parameter Tracking**: Log hyperparameters
- **Artifact Management**: Save models and outputs
- **Visualization**: Compare experiments

### Model Management
- **Model Registration**: Version and catalog models
- **Model Packaging**: Package for deployment
- **Model Validation**: Test model quality
- **Model Versioning**: Track model evolution
- **Model Metadata**: Tags and descriptions

### Deployment
- **Online Endpoints**: Real-time inference
- **Batch Endpoints**: Batch scoring
- **Managed Endpoints**: Fully managed deployment
- **Container Deployment**: Custom containers
- **Scaling**: Auto-scaling configuration

### MLOps Features
- **Pipeline Automation**: End-to-end ML pipelines
- **CI/CD Integration**: DevOps for ML
- **Model Monitoring**: Performance tracking
- **Data Drift Detection**: Monitor data changes
- **Automated Retraining**: Trigger on drift

## Architecture

```
MachineLearning/
├── azure_ml.py                # Main implementation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/MachineLearning

# Install dependencies
pip install -r requirements.txt
```

## Configuration

```python
from azure_ml import AzureMLWorkspace, AzureMLDataset, AzureMLExperiment

workspace = AzureMLWorkspace(
    subscription_id="your-subscription-id",
    resource_group="rg-ml",
    workspace_name="ml-workspace",
    location="eastus"
)
```

## Usage Examples

### Workspace Setup

```python
# Create workspace
workspace = AzureMLWorkspace(
    subscription_id="xxx",
    resource_group="rg-ml",
    workspace_name="ml-workspace"
)

# Create compute cluster
workspace.create_compute_cluster(
    name="cpu-cluster",
    vm_size="Standard_D3_v2",
    min_nodes=0,
    max_nodes=4
)
```

### Dataset Management

```python
# Register dataset
dataset = AzureMLDataset(workspace, "customer-data")
dataset.register(
    datastore="workspaceblobstore",
    path="data/customers.csv"
)

# Load dataset
data = dataset.load()
print(f"Dataset shape: {data.shape}")
```

### Training Experiment

```python
# Create experiment
experiment = AzureMLExperiment(workspace, "churn-prediction")

# Start run
run = experiment.start_run()

# Log metrics
run.log_metric("accuracy", 0.92)
run.log_metric("precision", 0.89)
run.log_metric("recall", 0.91)

# Log parameters
run.log_parameter("learning_rate", 0.01)
run.log_parameter("epochs", 100)
run.log_parameter("batch_size", 32)

# Save model
run.save_model("model.pkl", model_object)

# Complete run
run.complete()
```

### Model Registration

```python
from azure_ml import AzureMLModel

model = AzureMLModel(workspace, "churn-predictor")

# Register model
model.register(
    model_path="outputs/model.pkl",
    description="Customer churn prediction model",
    tags={"type": "classification", "framework": "sklearn"}
)

# List models
models = model.list_versions()
for version in models:
    print(f"Version {version}: {version.accuracy}")
```

### Model Deployment

```python
# Deploy to managed endpoint
endpoint = model.deploy_to_endpoint(
    endpoint_name="churn-prediction-ep",
    instance_type="Standard_DS2_v2",
    instance_count=2
)

# Test endpoint
sample_data = {
    "age": 35,
    "tenure": 24,
    "monthly_charges": 75.50
}

prediction = endpoint.predict(sample_data)
print(f"Prediction: {prediction}")
```

### ML Pipeline

```python
from azure_ml import MLPipeline

pipeline = MLPipeline(workspace, "training-pipeline")

# Add pipeline steps
pipeline.add_step("data_prep", script="prepare_data.py")
pipeline.add_step("training", script="train_model.py",
                  depends_on="data_prep")
pipeline.add_step("evaluation", script="evaluate_model.py",
                  depends_on="training")
pipeline.add_step("deployment", script="deploy_model.py",
                  depends_on="evaluation")

# Run pipeline
pipeline_run = pipeline.submit()
pipeline_run.wait_for_completion()
```

### Automated ML

```python
from azure_ml import AutoMLConfig

automl_config = AutoMLConfig(
    task="classification",
    primary_metric="accuracy",
    training_data=dataset,
    label_column="churn",
    n_cross_validations=5,
    max_concurrent_iterations=4,
    experiment_timeout_hours=1
)

# Run AutoML
automl_run = experiment.submit(automl_config)
best_model = automl_run.get_best_model()
```

## Running Demos

```bash
# Run all demo functions
python azure_ml.py
```

## API Reference

### AzureMLWorkspace

**`create_compute_cluster(name, vm_size, min_nodes, max_nodes)`** - Create compute

**`create_environment(name, conda_file)`** - Create environment

**`get_datastore(name)`** - Get datastore

### AzureMLDataset

**`register(datastore, path)`** - Register dataset

**`load()`** - Load dataset into memory

**`profile()`** - Generate data profile

### AzureMLExperiment

**`start_run()`** - Start new run

**`list_runs()`** - List all runs

**`get_run(run_id)`** - Get specific run

### AzureMLModel

**`register(model_path, description, tags)`** - Register model

**`deploy_to_endpoint(endpoint_name, instance_type, instance_count)`** - Deploy

**`list_versions()`** - List model versions

## Best Practices

### 1. Use Managed Compute
```python
# Auto-scaling compute cluster
workspace.create_compute_cluster(
    name="training-cluster",
    vm_size="Standard_NC6",  # GPU
    min_nodes=0,  # Scale to zero
    max_nodes=4
)
```

### 2. Version Everything
```python
# Version datasets
dataset.register(version="1.0", tags={"source": "production"})

# Version models
model.register(version="2.1", description="Improved accuracy")
```

### 3. Track Experiments
```python
# Comprehensive logging
run.log_metric("train_accuracy", 0.95)
run.log_metric("val_accuracy", 0.92)
run.log_parameter("optimizer", "adam")
run.log_artifact("confusion_matrix.png")
```

### 4. Use Environments
```python
# Custom environment
env = workspace.create_environment(
    name="sklearn-env",
    conda_file="environment.yml"
)
```

### 5. Enable Monitoring
```python
# Deploy with monitoring
endpoint = model.deploy_to_endpoint(
    endpoint_name="prod-endpoint",
    enable_app_insights=True,
    collect_model_data=True
)
```

## Use Cases

### 1. Predictive Maintenance
```python
# Train model
experiment = AzureMLExperiment(workspace, "predictive-maintenance")
run = experiment.start_run()
# Train and log...
run.complete()

# Deploy for real-time predictions
model.deploy_to_endpoint("maintenance-prediction")
```

### 2. Demand Forecasting
```python
# AutoML for time series
automl_config = AutoMLConfig(
    task="forecasting",
    time_column_name="date",
    forecast_horizon=30
)
```

### 3. Churn Prediction
```python
# Classification pipeline
pipeline = MLPipeline(workspace, "churn-pipeline")
pipeline.add_step("feature_engineering")
pipeline.add_step("model_training")
pipeline.add_step("model_evaluation")
```

## Performance Optimization

### 1. Use GPU for Deep Learning
```python
workspace.create_compute_cluster(
    name="gpu-cluster",
    vm_size="Standard_NC12",
    max_nodes=2
)
```

### 2. Parallel Hyperparameter Tuning
```python
from azure_ml import HyperDriveConfig

hyperdrive_config = HyperDriveConfig(
    max_concurrent_runs=4,
    max_total_runs=20
)
```

### 3. Cache Intermediate Results
```python
# Enable pipeline caching
pipeline.enable_cache(allow_reuse=True)
```

## Security Considerations

1. **Access Control**: Use RBAC for workspace access
2. **Credential Management**: Use Azure Key Vault
3. **Data Encryption**: Enable at-rest and in-transit
4. **Network Isolation**: Use private endpoints
5. **Audit Logging**: Track all operations

## Troubleshooting

**Issue**: Training job fails
**Solution**: Check compute availability and logs

**Issue**: Deployment errors
**Solution**: Verify environment dependencies

**Issue**: Slow training
**Solution**: Use GPU compute or increase cluster size

## Deployment

### Azure Setup
```bash
# Create ML workspace
az ml workspace create \
    --name ml-workspace \
    --resource-group rg-ml \
    --location eastus

# Create compute
az ml compute create \
    --name cpu-cluster \
    --type AmlCompute \
    --size Standard_D3_v2
```

## Monitoring

### Key Metrics
- Training job success rate
- Model accuracy trends
- Endpoint latency
- Prediction throughput
- Data drift score

## Dependencies

```
Python >= 3.8
azureml-core >= 1.40.0
azureml-train-automl >= 1.40.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
```

## Support

For questions or support:
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure AI Services](../AzureAI/)
- [Azure OpenAI](../AzureOpenAI/)
- [Data Services](../DataServices/)

---

**Built with Azure Machine Learning** | **Brill Consulting © 2024**
