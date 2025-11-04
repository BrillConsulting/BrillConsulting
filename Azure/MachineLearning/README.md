# Azure Machine Learning

Azure ML workspace and model management.

## Features

- Workspace management
- Dataset registration and management
- Experiment tracking with runs
- Metric and parameter logging
- Model registration and versioning
- Model deployment to endpoints
- Online inference

## Usage

```python
from azure_ml import AzureMLWorkspace, AzureMLDataset, AzureMLExperiment

# Create workspace
workspace = AzureMLWorkspace("ml-workspace", "subscription-id", "rg-ml")

# Register dataset
dataset = AzureMLDataset(workspace, "customer-data")
dataset.register("azureml://datasets/customers.csv")

# Run experiment
experiment = AzureMLExperiment(workspace, "churn-prediction")
run = experiment.start_run()
run.log_metric("accuracy", 0.92)
run.log_parameter("learning_rate", 0.01)
run.complete()

# Register and deploy model
model = AzureMLModel(workspace, "churn-predictor")
model.register("outputs/model.pkl")
```

## Demo

```bash
python azure_ml.py
```
