# Vertex AI - Unified Machine Learning Platform

Comprehensive Google Cloud Vertex AI implementation for building, training, and deploying machine learning models at scale with AutoML, custom training, and MLOps capabilities.

## Features

### Dataset Management
- **Multiple Dataset Types**: Tabular, image, text, video datasets
- **Data Import**: Import from Cloud Storage (CSV, JSON, images, videos)
- **Data Splitting**: Automatic train/validation/test splits (80/10/10)
- **Data Labeling**: Support for data annotation and labeling
- **Dataset Versioning**: Track dataset versions and changes

### AutoML Training
- **AutoML Tables**: Automated ML for tabular data (classification, regression)
- **AutoML Vision**: Image classification and object detection
- **AutoML Natural Language**: Text classification and sentiment analysis
- **Optimization Objectives**: minimize-rmse, minimize-mae, minimize-log-loss, maximize-au-prc
- **Budget Control**: Flexible training budget (1-72000 node hours)

### Custom Training
- **Custom Containers**: Train with custom Docker containers
- **GPU/TPU Support**: NVIDIA Tesla K80, T4, V100, P100, TPU v3
- **Distributed Training**: Multi-worker and multi-GPU training
- **Hyperparameter Tuning**: Automated hyperparameter optimization (1-100 trials)
- **Training Arguments**: Pass custom arguments to training scripts

### Model Management
- **Model Upload**: Upload TensorFlow, PyTorch, scikit-learn models
- **Model Versioning**: Create and manage multiple model versions
- **Model Evaluation**: Evaluate models on test datasets
- **Model Metadata**: Store descriptions, labels, and metrics
- **Model Export**: Export models in various formats

### Endpoint Deployment
- **Endpoint Creation**: Create prediction endpoints
- **Model Deployment**: Deploy models to endpoints with autoscaling
- **Traffic Splitting**: A/B testing with traffic percentage control
- **Autoscaling**: Min-max replica configuration (1-100 replicas)
- **Machine Types**: n1-standard-2/4/8, n1-highmem-2/4/8, custom types
- **Online Predictions**: Real-time predictions via REST API

### Batch Predictions
- **Batch Jobs**: Large-scale batch inference
- **Multiple Formats**: JSONL, CSV, BigQuery input/output
- **Scalable Processing**: Process millions of predictions
- **Output to Storage**: Write results to Cloud Storage or BigQuery

### Feature Store
- **Feature Management**: Create and manage ML features
- **Entity Types**: Organize features by entity (user, product, etc.)
- **Feature Serving**: Online and batch feature serving
- **Feature Versioning**: Track feature versions and lineage
- **Value Types**: BOOL, DOUBLE, INT64, STRING, BYTES

## Usage Example

```python
from vertex_ai import VertexAIManager

# Initialize manager
mgr = VertexAIManager(
    project_id='my-gcp-project',
    location='us-central1'
)

# 1. Create and import dataset
dataset = mgr.datasets.create_dataset({
    'display_name': 'customer-churn-data',
    'dataset_type': 'tabular',
    'labels': {'env': 'production'}
})

mgr.datasets.import_data({
    'dataset_name': 'customer-churn-data',
    'source_uris': ['gs://my-bucket/data/churn-data.csv']
})

mgr.datasets.split_dataset('customer-churn-data', 0.8, 0.1, 0.1)

# 2. AutoML training
automl_job = mgr.automl.create_automl_tabular_training({
    'display_name': 'churn-prediction-automl',
    'dataset_name': 'customer-churn-data',
    'target_column': 'churn',
    'optimization_objective': 'maximize-au-prc',
    'budget_milli_node_hours': 2000
})

status = mgr.automl.get_training_status('churn-prediction-automl')

# 3. Custom training with GPU
custom_job = mgr.custom_training.create_custom_training_job({
    'display_name': 'custom-tensorflow-training',
    'container_uri': 'gcr.io/my-project/training:latest',
    'machine_type': 'n1-standard-8',
    'accelerator_type': 'NVIDIA_TESLA_V100',
    'accelerator_count': 2,
    'replica_count': 1,
    'args': ['--epochs=50', '--batch-size=32']
})

# 4. Hyperparameter tuning
tuning_job = mgr.custom_training.create_hyperparameter_tuning_job({
    'display_name': 'hyperparameter-tuning',
    'container_uri': 'gcr.io/my-project/training:latest',
    'hyperparameter_spec': {
        'learning_rate': {'min': 0.001, 'max': 0.1},
        'batch_size': [16, 32, 64],
        'dropout': {'min': 0.1, 'max': 0.5}
    },
    'max_trial_count': 20,
    'parallel_trial_count': 4
})

# 5. Upload and version model
model = mgr.models.upload_model({
    'display_name': 'churn-predictor',
    'artifact_uri': 'gs://my-bucket/models/churn-model-v1/',
    'serving_container_image_uri': 'gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest',
    'description': 'Customer churn prediction model'
})

# Create new version
mgr.models.create_model_version('churn-predictor', {
    'artifact_uri': 'gs://my-bucket/models/churn-model-v2/'
})

# Evaluate model
evaluation = mgr.models.evaluate_model('churn-predictor', 'test-dataset')

# 6. Create endpoint and deploy
endpoint = mgr.endpoints.create_endpoint({
    'display_name': 'churn-prediction-endpoint',
    'description': 'Production endpoint for churn predictions'
})

deployment = mgr.endpoints.deploy_model({
    'endpoint_name': 'churn-prediction-endpoint',
    'model_name': 'churn-predictor',
    'machine_type': 'n1-standard-4',
    'min_replica_count': 2,
    'max_replica_count': 10,
    'traffic_percentage': 100
})

# 7. Online predictions
test_instances = [
    {'tenure': 12, 'monthly_charges': 50.5, 'total_charges': 606.0},
    {'tenure': 24, 'monthly_charges': 85.0, 'total_charges': 2040.0}
]

predictions = mgr.endpoints.predict('churn-prediction-endpoint', test_instances)

# 8. Batch predictions
batch_job = mgr.batch_predictions.create_batch_prediction_job({
    'display_name': 'monthly-churn-predictions',
    'model_name': 'churn-predictor',
    'input_uri': 'gs://my-bucket/batch-input/customers.jsonl',
    'output_uri': 'gs://my-bucket/batch-output/',
    'instances_format': 'jsonl',
    'predictions_format': 'jsonl'
})

# 9. Feature Store
featurestore = mgr.feature_store.create_featurestore({
    'featurestore_id': 'customer-features'
})

entity_type = mgr.feature_store.create_entity_type('customer-features', {
    'entity_type_id': 'customer',
    'description': 'Customer entity'
})

feature = mgr.feature_store.create_feature('customer-features', 'customer', {
    'feature_id': 'lifetime_value',
    'value_type': 'DOUBLE',
    'description': 'Customer lifetime value'
})
```

## AutoML Optimization Objectives

### Classification
- **maximize-au-prc**: Maximize area under precision-recall curve (imbalanced datasets)
- **maximize-au-roc**: Maximize area under ROC curve
- **minimize-log-loss**: Minimize log loss (probabilistic predictions)

### Regression
- **minimize-rmse**: Minimize root mean squared error
- **minimize-mae**: Minimize mean absolute error
- **minimize-rmsle**: Minimize root mean squared logarithmic error

## GPU and TPU Options

### NVIDIA Tesla GPUs
- **NVIDIA_TESLA_K80**: Entry-level GPU (12GB memory)
- **NVIDIA_TESLA_T4**: Efficient inference and training (16GB)
- **NVIDIA_TESLA_V100**: High-performance training (16GB/32GB)
- **NVIDIA_TESLA_P100**: Powerful training (16GB)
- **NVIDIA_TESLA_P4**: Inference-optimized (8GB)

### TPU (Tensor Processing Units)
- **TPU_V2**: 2nd generation TPU (8 cores)
- **TPU_V3**: 3rd generation TPU (8 cores, faster)
- **TPU_V4**: Latest generation TPU

## Machine Types for Deployment

### Standard
- **n1-standard-2**: 2 vCPUs, 7.5GB RAM
- **n1-standard-4**: 4 vCPUs, 15GB RAM
- **n1-standard-8**: 8 vCPUs, 30GB RAM
- **n1-standard-16**: 16 vCPUs, 60GB RAM

### High Memory
- **n1-highmem-2**: 2 vCPUs, 13GB RAM
- **n1-highmem-4**: 4 vCPUs, 26GB RAM
- **n1-highmem-8**: 8 vCPUs, 52GB RAM

### High CPU
- **n1-highcpu-4**: 4 vCPUs, 3.6GB RAM
- **n1-highcpu-8**: 8 vCPUs, 7.2GB RAM

## Pricing Estimates

### Training (per node hour)
- **AutoML Tables**: $19.32/node hour
- **AutoML Vision**: $3.465/node hour
- **AutoML NLP**: $3.00/node hour
- **Custom Training (n1-standard-4)**: $0.196/hour
- **Custom Training (n1-standard-4 + V100)**: $2.696/hour

### Prediction (per node hour)
- **n1-standard-2**: $0.098/hour
- **n1-standard-4**: $0.196/hour
- **n1-highmem-2**: $0.130/hour

### Storage
- **Model Storage**: $0.10/GB/month
- **Dataset Storage**: $0.023/GB/month

## Best Practices

1. **Use AutoML** for rapid prototyping and baseline models
2. **Split datasets** properly (80/10/10) for reliable evaluation
3. **Start with small budgets** for AutoML, increase if needed
4. **Use GPUs/TPUs** for deep learning and large datasets
5. **Enable autoscaling** on endpoints for variable traffic
6. **Use traffic splitting** for safe model rollouts
7. **Monitor model performance** continuously in production
8. **Use Feature Store** for consistent feature engineering
9. **Leverage batch predictions** for large-scale inference
10. **Version models** to track improvements over time

## Common Use Cases

### AutoML Tables
- Customer churn prediction
- Fraud detection
- Demand forecasting
- Credit scoring

### AutoML Vision
- Image classification
- Object detection
- Product defect detection
- Medical image analysis

### AutoML NLP
- Sentiment analysis
- Document classification
- Intent detection
- Entity extraction

### Custom Training
- Deep learning models (TensorFlow, PyTorch)
- Transfer learning
- Multi-modal models
- Large-scale training

## Requirements

```
google-cloud-aiplatform
google-cloud-storage
```

## Configuration

Set up authentication:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

## Model Serving Containers

### TensorFlow
- `gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest`
- `gcr.io/cloud-aiplatform/prediction/tf2-gpu.2-8:latest`

### PyTorch
- `gcr.io/cloud-aiplatform/prediction/pytorch-cpu.1-10:latest`
- `gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-10:latest`

### Scikit-learn
- `gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest`

### XGBoost
- `gcr.io/cloud-aiplatform/prediction/xgboost-cpu.1-4:latest`

## MLOps Integration

- **Pipelines**: Kubeflow Pipelines for ML workflows
- **Experiments**: Track and compare training runs
- **Metadata**: Store artifact and execution metadata
- **Model Monitoring**: Monitor prediction quality and drift
- **Explainability**: Feature attributions and example-based explanations

## Author

BrillConsulting - Enterprise Cloud Solutions
