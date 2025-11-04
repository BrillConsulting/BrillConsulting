# CI/CD Pipeline for ML

Automated testing, validation, and deployment pipeline for ML models.

## Features

- Automated unit and integration testing
- Model performance validation
- Performance benchmarking
- Docker image building
- Multi-environment deployment
- Rollback capability

## Usage

```python
from ci_cd_pipeline import MLCICDPipeline

config = {
    "min_accuracy": 0.85,
    "max_latency_ms": 50
}

pipeline = MLCICDPipeline(config)
result = pipeline.run_pipeline("model.pkl", environment="production")
```

## GitHub Actions Example

```yaml
name: ML CI/CD
on: [push]
jobs:
  test-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: python ci_cd_pipeline.py
```

## Demo

```bash
python ci_cd_pipeline.py
```
