# AWS SageMaker Management

**Production-ready ML operations with training, deployment, and inference capabilities.**

## 🎯 Overview

Comprehensive SageMaker management for end-to-end machine learning workflows:
- Training job management with hyperparameter tuning
- Model creation and deployment
- Real-time inference endpoints
- Batch transform jobs
- Model monitoring

## ✨ Features

- **Training Jobs**: Create and monitor training with custom algorithms
- **Model Management**: Deploy models from training outputs
- **Endpoints**: Real-time prediction APIs with auto-scaling
- **Batch Inference**: Large-scale batch transform jobs
- **Monitoring**: Track training metrics and endpoint performance

## 📋 Prerequisites

1. **AWS Account** with SageMaker permissions
2. **IAM Role** with SageMakerFullAccess
3. **S3 Bucket** for training data and model artifacts
4. **Python 3.8+** and **boto3**

## 🚀 Quick Start

```python
from aws_sagemaker import SageMakerManager

# Initialize
sm = SageMakerManager(region='us-east-1')

# Train model
job = sm.create_training_job(
    job_name='my-training-job',
    role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
    algorithm_specification={
        'TrainingImage': 'xgboost-image',
        'TrainingInputMode': 'File'
    },
    input_data_config=[...],
    output_data_config={'S3OutputPath': 's3://bucket/output/'},
    resource_config={
        'InstanceType': 'ml.m5.xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 30
    },
    stopping_condition={'MaxRuntimeInSeconds': 3600}
)

# Deploy model
model = sm.create_model(
    model_name='my-model',
    role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
    primary_container={
        'Image': 'xgboost-image',
        'ModelDataUrl': 's3://bucket/output/model.tar.gz'
    }
)

# Create endpoint
endpoint = sm.create_endpoint('my-endpoint', 'my-config')

# Make predictions
response = sm.invoke_endpoint(
    endpoint_name='my-endpoint',
    payload=json.dumps({'instances': [[1, 2, 3]]}).encode()
)
```

## 💻 Usage Examples

### Training with XGBoost

```python
job = sm.create_training_job(
    job_name='xgboost-classifier',
    role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
    algorithm_specification={
        'TrainingImage': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1',
        'TrainingInputMode': 'File'
    },
    input_data_config=[{
        'ChannelName': 'train',
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': 's3://my-bucket/train/',
                'S3DataDistributionType': 'FullyReplicated'
            }
        }
    }],
    output_data_config={'S3OutputPath': 's3://my-bucket/output/'},
    resource_config={
        'InstanceType': 'ml.m5.2xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 50
    },
    stopping_condition={'MaxRuntimeInSeconds': 7200},
    hyperparameters={
        'max_depth': '5',
        'eta': '0.2',
        'objective': 'binary:logistic',
        'num_round': '100'
    }
)

# Monitor training
status = sm.describe_training_job('xgboost-classifier')
print(f"Status: {status['status']}")
print(f"Metrics: {status.get('metrics', {})}")
```

### Deploy and Invoke

```python
# Create model
model = sm.create_model(
    model_name='xgboost-model-v1',
    role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
    primary_container={
        'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1',
        'ModelDataUrl': 's3://my-bucket/output/model.tar.gz'
    }
)

# Create endpoint config
config = sm.create_endpoint_config(
    config_name='xgboost-config-v1',
    production_variants=[{
        'VariantName': 'AllTraffic',
        'ModelName': 'xgboost-model-v1',
        'InitialInstanceCount': 2,
        'InstanceType': 'ml.m5.large',
        'InitialVariantWeight': 1.0
    }]
)

# Deploy endpoint
endpoint = sm.create_endpoint('xgboost-endpoint', 'xgboost-config-v1')

# Wait for deployment (check status)
status = sm.describe_endpoint('xgboost-endpoint')
while status['status'] == 'Creating':
    time.sleep(30)
    status = sm.describe_endpoint('xgboost-endpoint')

# Invoke for predictions
data = {'instances': [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]}
payload = json.dumps(data).encode('utf-8')

response = sm.invoke_endpoint(
    endpoint_name='xgboost-endpoint',
    payload=payload,
    content_type='application/json'
)

print(f"Predictions: {response['predictions']}")
```

### Batch Transform

```python
# Create batch transform job
transform_job = sm.create_transform_job(
    job_name='batch-inference-job',
    model_name='xgboost-model-v1',
    transform_input={
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': 's3://my-bucket/batch-input/'
            }
        },
        'ContentType': 'text/csv'
    },
    transform_output={
        'S3OutputPath': 's3://my-bucket/batch-output/'
    },
    transform_resources={
        'InstanceType': 'ml.m5.xlarge',
        'InstanceCount': 2
    }
)
```

## 🏗️ Architecture

```
SageMakerManager
├── Training Jobs
│   ├── create_training_job()
│   ├── describe_training_job()
│   ├── list_training_jobs()
│   └── stop_training_job()
│
├── Models
│   ├── create_model()
│   ├── describe_model()
│   └── delete_model()
│
├── Endpoints
│   ├── create_endpoint_config()
│   ├── create_endpoint()
│   ├── describe_endpoint()
│   ├── update_endpoint()
│   └── delete_endpoint()
│
├── Inference
│   ├── invoke_endpoint()          # Real-time
│   └── create_transform_job()     # Batch
│
└── Monitoring
    └── get_summary()
```

## 🔒 Best Practices

1. **IAM Roles**: Use least-privilege roles for SageMaker
2. **Data Security**: Encrypt S3 buckets and enable VPC for endpoints
3. **Cost Optimization**: Use Spot instances for training, auto-scaling for endpoints
4. **Model Versioning**: Tag models with version numbers
5. **Monitoring**: Enable CloudWatch metrics and model monitoring
6. **Instance Selection**: Choose appropriate instance types for workload

## 📊 Common Use Cases

### Binary Classification

```python
# Train binary classifier
job = sm.create_training_job(
    job_name='churn-prediction',
    hyperparameters={
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': '6',
        'eta': '0.3'
    },
    ...
)
```

### Multi-class Classification

```python
hyperparameters={
    'objective': 'multi:softmax',
    'num_class': '10',
    'max_depth': '5'
}
```

### Regression

```python
hyperparameters={
    'objective': 'reg:squarederror',
    'max_depth': '7',
    'eta': '0.1'
}
```

## 🐛 Troubleshooting

**Issue: Training job fails with "No space left on device"**
- **Solution**: Increase `VolumeSizeInGB` in resource_config

**Issue: Endpoint invocation timeout**
- **Solution**: Increase instance count or use larger instance type

**Issue: "AccessDenied" errors**
- **Solution**: Verify IAM role has required S3 and SageMaker permissions

## 📚 Supported Algorithms

- **XGBoost**: Gradient boosting (classification, regression)
- **Linear Learner**: Linear models
- **Random Cut Forest**: Anomaly detection
- **Neural Networks**: Custom deep learning models
- **Custom Containers**: Bring your own algorithm (BYOA)

## 🔗 Related Services

- **S3**: Data storage
- **CloudWatch**: Monitoring and logging
- **ECR**: Container registry for custom algorithms
- **Lambda**: Event-driven inference triggers
- **Step Functions**: ML workflow orchestration

## 📞 Support

- **Email**: clientbrill@gmail.com
- **LinkedIn**: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

**Author**: Brill Consulting | **Last Updated**: November 2025
