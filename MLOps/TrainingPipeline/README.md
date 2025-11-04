# ML Training Pipeline

Automated end-to-end training pipeline for reproducible ML model development.

## Features

- Data loading and validation
- Preprocessing and feature engineering
- Model training and tuning
- Evaluation metrics
- Artifact management
- Metadata tracking
- Configurable pipelines

## Usage

```python
from training_pipeline import TrainingPipeline

config = {
    "artifacts_dir": "./artifacts",
    "model_type": "classifier"
}

pipeline = TrainingPipeline(config)
result = pipeline.run("data.csv")
```

## Demo

```bash
python training_pipeline.py
```
