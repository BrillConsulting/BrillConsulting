# Data Versioning & Pipeline Management

Production data versioning, ML pipeline management, and experiment tracking using DVC, Git, and cloud storage backends.

## Features

- **Data Versioning** - Track datasets with DVC (Git for data)
- **Pipeline Management** - Reproducible ML pipelines with DVC
- **Experiment Tracking** - Version models, metrics, and parameters
- **Remote Storage** - S3, GCS, Azure Blob, HDFS backends
- **Collaboration** - Share datasets and models across teams
- **Lineage Tracking** - Data and model provenance
- **CI/CD Integration** - Automated pipeline execution
- **Large File Support** - Handle datasets from MB to TB

## Architecture

```
[Local Data] → [DVC Track] → [Git Commit] → [DVC Push] → [Remote Storage]
                     ↓                                          ↓
              [.dvc files]                              [S3/GCS/Azure]
                     ↓
             [Git Repository]
```

## Installation

```bash
pip install dvc[all]  # All backends
# or
pip install dvc[s3]   # S3 only
pip install dvc[gs]   # GCS only
pip install dvc[azure] # Azure only
```

## Usage

### Initialize DVC

```bash
# Initialize in Git repo
git init
dvc init

# Configure remote storage
dvc remote add -d storage s3://my-bucket/dvc-storage
dvc remote modify storage region us-west-2

# Or for GCS
dvc remote add -d storage gs://my-bucket/dvc-storage

# Or for Azure
dvc remote add -d storage azure://mycontainer/path
```

### Track Data

```python
from data_versioning import DVCTracker

tracker = DVCTracker(
    project_dir=".",
    remote="storage"
)

# Track dataset
tracker.track_data(
    path="data/training.csv",
    message="Add training dataset v1.0"
)

# Track model
tracker.track_model(
    path="models/yolov8n.pt",
    metrics={"accuracy": 0.95, "f1": 0.93},
    message="Train YOLOv8n model"
)
```

### Version Control

```bash
# Track data file
dvc add data/training.csv

# Commit .dvc file to Git
git add data/training.csv.dvc data/.gitignore
git commit -m "Add training data v1.0"

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Checkout specific version
git checkout v1.0
dvc checkout
```

### ML Pipelines

#### Define Pipeline

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/dataset.csv
    outs:
      - data/processed/features.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/features.csv
    params:
      - train.epochs
      - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      - metrics/train.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/test/test.csv
    metrics:
      - metrics/eval.json:
          cache: false
```

#### Run Pipeline

```python
from data_versioning import PipelineManager

pipeline = PipelineManager(pipeline_file="dvc.yaml")

# Run entire pipeline
results = pipeline.run()

# Run specific stage
pipeline.run_stage("train")

# Reproduce pipeline
pipeline.reproduce()
```

### Experiment Tracking

```python
from data_versioning import ExperimentTracker

tracker = ExperimentTracker()

# Start experiment
with tracker.start_run(name="yolov8_training"):
    # Log parameters
    tracker.log_params({
        "model": "yolov8n",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001
    })

    # Train model
    model = train_model()

    # Log metrics
    tracker.log_metrics({
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1": 0.935
    })

    # Log model
    tracker.log_model(
        model=model,
        path="models/yolov8n.pt"
    )

# Compare experiments
comparison = tracker.compare_experiments(
    experiment_ids=["exp1", "exp2", "exp3"]
)
```

## Data Versioning Patterns

### Large Dataset Versioning

```python
# Track large dataset
tracker = DVCTracker()

# Add to DVC (creates .dvc file)
tracker.add("data/imagenet.tar.gz")

# Commit metadata to Git
tracker.commit("Add ImageNet dataset")

# Push data to S3 (large file)
tracker.push()

# Others can pull with:
# dvc pull data/imagenet.tar.gz.dvc
```

### Incremental Versioning

```python
# Version 1
tracker.track_data("data/v1/train.csv", message="Initial dataset")

# Version 2 (add more samples)
tracker.track_data("data/v2/train.csv", message="Add 10k samples")

# Version 3 (fix labels)
tracker.track_data("data/v3/train.csv", message="Fix label errors")

# Switch between versions
tracker.checkout_version(tag="v1.0")
```

### Model Versioning

```python
# Version model with metadata
tracker.track_model(
    path="models/resnet50_v1.pt",
    metrics={
        "accuracy": 0.95,
        "f1": 0.94,
        "inference_time_ms": 15
    },
    params={
        "architecture": "resnet50",
        "pretrained": True,
        "epochs": 100
    },
    tags=["production", "v1.0"]
)
```

## Pipeline Management

### Parameterized Pipelines

```yaml
# params.yaml
train:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

model:
  architecture: yolov8n
  pretrained: true
```

```python
# Load and use parameters
from data_versioning import load_params

params = load_params("params.yaml")

model = train_model(
    epochs=params["train"]["epochs"],
    batch_size=params["train"]["batch_size"]
)
```

### Pipeline Execution

```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro train

# Run with different params
dvc repro -P train.epochs=200

# Show pipeline DAG
dvc dag
```

### Metrics Comparison

```python
# Compare metrics across runs
from data_versioning import MetricsComparator

comparator = MetricsComparator()

# Show metrics diff
diff = comparator.diff(
    baseline="HEAD~1",
    comparison="HEAD"
)

print(f"Accuracy: {diff['accuracy']:+.2%}")
print(f"F1 Score: {diff['f1']:+.2%}")
```

## Remote Storage Backends

### AWS S3

```bash
# Configure S3
dvc remote add -d s3remote s3://my-bucket/dvc-storage
dvc remote modify s3remote region us-west-2
dvc remote modify s3remote profile myprofile

# Or with credentials
dvc remote modify s3remote access_key_id YOUR_ACCESS_KEY
dvc remote modify s3remote secret_access_key YOUR_SECRET_KEY
```

### Google Cloud Storage

```bash
# Configure GCS
dvc remote add -d gcsremote gs://my-bucket/dvc-storage
dvc remote modify gcsremote projectname myproject

# Authenticate
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

### Azure Blob Storage

```bash
# Configure Azure
dvc remote add -d azureremote azure://mycontainer/path
dvc remote modify azureremote account_name myaccount
dvc remote modify azureremote connection_string "..."
```

### HDFS

```bash
# Configure HDFS
dvc remote add -d hdfsremote hdfs://namenode:8020/dvc-storage
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/dvc-pipeline.yml
name: DVC Pipeline

on: [push, pull_request]

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          pip install dvc[s3]
          pip install -r requirements.txt

      - name: Pull data
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: dvc pull

      - name: Run pipeline
        run: dvc repro

      - name: Push results
        run: dvc push
```

### GitLab CI

```yaml
# .gitlab-ci.yml
dvc_pipeline:
  image: python:3.9
  before_script:
    - pip install dvc[s3]
    - dvc pull
  script:
    - dvc repro
  after_script:
    - dvc push
```

## Collaboration Workflows

### Share Dataset

```bash
# Person A: Track and share
dvc add data/dataset.csv
git add data/dataset.csv.dvc
git commit -m "Add dataset"
git push
dvc push

# Person B: Get dataset
git pull
dvc pull
```

### Model Registry

```python
from data_versioning import ModelRegistry

registry = ModelRegistry(remote="s3://models")

# Register model
registry.register(
    name="yolov8_production",
    version="1.0.0",
    model_path="models/yolov8n.pt",
    metrics={"accuracy": 0.95},
    stage="production"
)

# Load model
model = registry.load_model(
    name="yolov8_production",
    version="1.0.0"
)

# Promote model
registry.transition(
    name="yolov8_production",
    version="1.1.0",
    stage="production"
)
```

## Best Practices

### Data Organization

```
data/
  raw/           # Original, immutable data
    dataset.csv.dvc
  processed/     # Processed features
    features.csv.dvc
  interim/       # Intermediate transformations
    temp.csv.dvc

models/          # Trained models
  yolov8n.pt.dvc
  resnet50.pt.dvc

metrics/         # Evaluation metrics
  train.json
  eval.json
```

### Version Tagging

```bash
# Tag important versions
git tag -a v1.0 -m "Production model v1.0"
git push origin v1.0

# Tag data versions
git tag -a data-v1.0 -m "Training dataset v1.0"
```

### Pipeline Best Practices

✅ Make pipelines deterministic (set random seeds)
✅ Use parameters for hyperparameters
✅ Cache intermediate results
✅ Track all dependencies
✅ Version code, data, and models together
✅ Use meaningful commit messages
✅ Document data sources and transformations

## Monitoring & Lineage

### Data Lineage

```python
from data_versioning import LineageTracker

tracker = LineageTracker()

# Track data lineage
lineage = tracker.get_lineage(
    artifact="models/yolov8n.pt"
)

print(f"Training data: {lineage.training_data}")
print(f"Code version: {lineage.code_commit}")
print(f"Parameters: {lineage.params}")
```

### Audit Trail

```python
# Get audit trail for dataset
audit = tracker.get_audit_trail(
    path="data/training.csv"
)

for entry in audit:
    print(f"{entry.timestamp}: {entry.message} by {entry.author}")
```

## Performance

### Storage Efficiency

| Backend | Throughput | Latency | Cost |
|---------|------------|---------|------|
| S3 | 100MB/s | 50ms | $0.023/GB/month |
| GCS | 120MB/s | 40ms | $0.020/GB/month |
| Azure | 90MB/s | 60ms | $0.018/GB/month |
| HDFS | 500MB/s | 10ms | On-premise |

### Caching

```bash
# DVC cache size
du -sh .dvc/cache

# Clean unused cache
dvc gc --workspace

# Cloud cache for team
dvc cache dir /shared/dvc-cache
```

## Technologies

- **Version Control**: DVC, Git, Git-LFS
- **Storage**: S3, GCS, Azure Blob, HDFS, SSH
- **Experiment Tracking**: MLflow, Weights & Biases (W&B)
- **Pipeline**: DVC pipelines, Airflow, Kubeflow
- **Metadata**: SQLite, PostgreSQL

## Use Cases

### Dataset Versioning

```python
# Track evolving dataset
tracker.track_data("data/train_v1.csv", message="Initial 10k samples")
# ... add more data
tracker.track_data("data/train_v2.csv", message="Added 50k samples")
# ... fix labels
tracker.track_data("data/train_v3.csv", message="Fixed label errors")
```

### Model Registry

```python
# Version models in production
registry.register("fraud_detector", "1.0.0", stage="production")
registry.register("fraud_detector", "1.1.0", stage="staging")
registry.register("fraud_detector", "1.2.0", stage="development")
```

### Reproducible Research

```bash
# Reproduce exact experiment
git checkout experiment-branch
dvc checkout
dvc repro
```

## References

- DVC Documentation: https://dvc.org/doc
- DVC Tutorial: https://dvc.org/doc/start
- DVC with S3: https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3
- DVC Pipelines: https://dvc.org/doc/user-guide/pipelines
- MLOps with DVC: https://dvc.org/doc/use-cases/versioning-data-and-model-files
