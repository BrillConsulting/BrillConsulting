# Model Packaging

Containerization and packaging of ML models for reproducible deployment across environments.

## Overview

Model packaging ensures ML models can be deployed consistently across development, staging, and production environments. This includes containerization (Docker), dependency management, configuration handling, and artifact bundling.

## Key Concepts

### Containerization
Package model with all dependencies in a Docker container for:
- Environment consistency
- Dependency isolation
- Reproducible deployments
- Easy scaling and orchestration

### Artifact Management
Bundle all necessary components:
- Trained model files (.pkl, .h5, .pt, .onnx)
- Preprocessing pipelines
- Configuration files
- Requirements and dependencies
- Model metadata

### Multi-Stage Builds
Optimize container size:
- Build stage: Install dependencies, compile code
- Runtime stage: Copy only necessary artifacts
- Result: Smaller, faster containers

## Docker Best Practices

### 1. Multi-Stage Dockerfile
```dockerfile
# Build stage
FROM python:3.9-slim as builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY model.pkl .
COPY app.py .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

### 2. Optimize Image Size
- Use slim base images
- Remove build dependencies
- Use .dockerignore
- Layer caching
- Multi-stage builds

### 3. Security
- Don't run as root
- Scan for vulnerabilities
- Use specific version tags
- Minimize attack surface

## Packaging Formats

### 1. Docker Image
Standard containerization:
```bash
docker build -t my-model:v1.0 .
docker run -p 8000:8000 my-model:v1.0
```

### 2. ONNX
Cross-platform inference:
```python
import torch.onnx

# Export PyTorch model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 3. TorchScript
Optimized PyTorch deployment:
```python
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model.pt")
```

### 4. SavedModel (TensorFlow)
TensorFlow serving format:
```python
model.save('saved_model/my_model')
```

## Example Package Structure

```
model-package/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── model/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── config.json
├── src/
│   ├── __init__.py
│   ├── predictor.py
│   └── utils.py
├── tests/
│   └── test_predictor.py
└── README.md
```

## Deployment Patterns

### 1. REST API Container
FastAPI + model in single container:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Batch Processing Container
Process data in batches:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "batch_processor.py"]
```

### 3. Sidecar Pattern
Model server + application containers:
```yaml
# docker-compose.yml
services:
  model:
    image: my-model:v1.0
    ports:
      - "8000:8000"

  app:
    image: my-app:v1.0
    environment:
      MODEL_URL: http://model:8000
```

## Model Versioning

### Semantic Versioning
```bash
my-model:1.0.0   # Major.Minor.Patch
my-model:1.0.1   # Bug fix
my-model:1.1.0   # New features
my-model:2.0.0   # Breaking changes
```

### Tagging Strategy
```bash
# Multiple tags for same image
docker tag my-model:1.0.0 my-model:latest
docker tag my-model:1.0.0 my-model:stable
docker tag my-model:1.0.0 my-model:prod
```

## Registry Management

### Push to Registry
```bash
# Tag for registry
docker tag my-model:v1.0 registry.company.com/my-model:v1.0

# Push
docker push registry.company.com/my-model:v1.0
```

### Private Registry
```bash
# Docker Hub
docker login

# AWS ECR
aws ecr get-login-password | docker login --username AWS --password-stdin

# GCP GCR
gcloud auth configure-docker

# Azure ACR
az acr login --name myregistry
```

## CI/CD Integration

### Build Pipeline
```yaml
# .github/workflows/build.yml
name: Build and Push
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build image
        run: docker build -t my-model:${{ github.sha }} .
      - name: Push image
        run: docker push my-model:${{ github.sha }}
```

## Best Practices

### 1. Include Health Checks
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1
```

### 2. Environment Configuration
```python
# Use environment variables
import os

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.pkl")
PORT = int(os.getenv("PORT", "8000"))
```

### 3. Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Model loaded successfully")
```

### 4. Resource Limits
```yaml
# docker-compose.yml
services:
  model:
    image: my-model:v1.0
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### 5. Documentation
Include in package:
- Model card
- API documentation
- Example requests
- Performance benchmarks
- Resource requirements

## Tools and Technologies

- **Docker**: Container platform
- **Docker Compose**: Multi-container orchestration
- **Kubernetes**: Container orchestration at scale
- **ONNX**: Cross-platform model format
- **TorchServe/TFServing**: Framework-specific serving
- **MLflow**: Model packaging and registry
- **BentoML**: ML model serving framework

## Security Considerations

- **Image Scanning**: Use Trivy, Clair, or Snyk
- **Secret Management**: Use environment variables, not hardcoded
- **Minimal Base Images**: Reduce attack surface
- **User Permissions**: Don't run as root
- **Version Pinning**: Pin exact dependency versions

## Monitoring

### Container Metrics
- CPU and memory usage
- Request latency
- Error rates
- Container restarts

### Model Metrics
- Prediction latency
- Throughput (requests/sec)
- Model version in use
- Input/output statistics

## Status

📝 **Note**: This is a conceptual guide. Full implementation would include complete Dockerfiles, CI/CD pipelines, and deployment configurations for various platforms.

## References

- Docker Best Practices
- Kubernetes Documentation
- MLflow Model Registry
- ONNX Runtime
- Model Serving Patterns

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
