# Experiment Tracking

Track ML experiments, parameters, metrics, and artifacts with MLflow-like interface.

## Features

- Parameter logging
- Metric tracking over time
- Artifact management
- Experiment comparison
- Best model selection
- Persistent storage

## Usage

```python
from experiment_tracker import ExperimentTracker

# Create experiment
tracker = ExperimentTracker("my_experiment")

# Log parameters
tracker.log_params({"lr": 0.01, "batch_size": 32})

# Log metrics
for epoch in range(10):
    tracker.log_metric("loss", loss_value, step=epoch)

# Log artifacts
tracker.log_artifact("model", model)

# Save
tracker.save_experiment()

# Compare experiments
comparison = ExperimentTracker.compare_experiments([
    "exp1/experiment.json",
    "exp2/experiment.json"
])
```

## Demo

```bash
python experiment_tracker.py
```
