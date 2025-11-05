# AWS Lambda Management

**Comprehensive serverless function management with event triggers, layers, VPC integration, and monitoring.**

## 🎯 Overview

Production-ready Lambda management system that provides:
- Function creation and deployment with multiple code sources
- Event source mappings (S3, DynamoDB, SQS, SNS, Kinesis, API Gateway)
- Layer management for shared dependencies
- Environment variables and configuration management
- VPC and security group integration
- Alias and version control for blue/green deployments
- CloudWatch Logs integration and monitoring

## ✨ Features

### Function Management
- **Create Functions**: Deploy with inline code, ZIP files, or S3
- **Update Code**: Seamlessly update function code
- **Configuration**: Manage timeout, memory, environment variables
- **Versions**: Immutable function versions
- **Aliases**: Mutable pointers to versions (prod, dev, etc.)
- **Delete**: Clean function removal

### Event Integration
- **DynamoDB Streams**: Process database changes
- **SQS Queues**: Handle asynchronous messages
- **SNS Topics**: React to notifications
- **Kinesis Streams**: Process real-time data
- **S3 Events**: Respond to object changes
- **API Gateway**: HTTP API endpoints

### Layer Management
- **Publish Layers**: Share code and dependencies
- **Version Control**: Layer versioning
- **Runtime Compatibility**: Multi-runtime support

### Invocation
- **Synchronous**: Request-response pattern
- **Asynchronous**: Event-driven invocation
- **Test Events**: DryRun mode for testing
- **Log Tailing**: Include logs in response

### Monitoring
- **CloudWatch Logs**: Function execution logs
- **Metrics**: Invocation counts, duration, errors
- **Summary Reports**: Aggregate function statistics

## 📋 Prerequisites

1. **AWS Account**: Active AWS account
2. **IAM Permissions**:
   - `lambda:CreateFunction`
   - `lambda:InvokeFunction`
   - `lambda:UpdateFunctionCode`
   - `lambda:UpdateFunctionConfiguration`
   - `lambda:GetFunction`
   - `lambda:ListFunctions`
   - `lambda:DeleteFunction`
   - `lambda:CreateEventSourceMapping`
   - `lambda:PublishLayerVersion`
   - `lambda:CreateAlias`
   - `iam:PassRole` (to assign execution role)
   - `logs:GetLogEvents` (for log retrieval)

3. **IAM Execution Role**: Lambda needs an execution role with:
   - `AWSLambdaBasicExecutionRole` (minimum)
   - Additional permissions based on function needs

4. **Python**: Version 3.8+
5. **boto3**: AWS SDK for Python

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

```bash
aws configure
```

### 3. Create IAM Execution Role

```bash
# Create trust policy
cat > lambda-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create role
aws iam create-role \
  --role-name lambda-execution-role \
  --assume-role-policy-document file://lambda-trust-policy.json

# Attach basic execution policy
aws iam attach-role-policy \
  --role-name lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

## 💻 Usage Examples

### Initialize Lambda Manager

```python
from aws_lambda import LambdaManager

# Initialize with default region
lambda_manager = LambdaManager(region="us-east-1")

# Or use a specific AWS profile
lambda_manager = LambdaManager(region="us-west-2", profile="production")
```

### Create Lambda Function

```python
# Method 1: Inline code (for simple functions)
function = lambda_manager.create_function(
    function_name='hello-world',
    runtime='python3.11',
    handler='index.handler',
    role_arn='arn:aws:iam::123456789012:role/lambda-execution-role',
    code_content='''
def handler(event, context):
    name = event.get('name', 'World')
    return {
        'statusCode': 200,
        'body': f'Hello, {name}!'
    }
    ''',
    timeout=30,
    memory_size=256,
    environment_variables={
        'ENV': 'production',
        'LOG_LEVEL': 'INFO'
    },
    tags={'Project': 'Demo', 'Owner': 'DevOps'}
)

print(f"Function ARN: {function['function_arn']}")
```

```python
# Method 2: From ZIP file
function = lambda_manager.create_function(
    function_name='process-data',
    runtime='python3.11',
    handler='app.handler',
    role_arn='arn:aws:iam::123456789012:role/lambda-execution-role',
    zip_file_path='./function.zip',
    timeout=300,
    memory_size=512
)
```

```python
# Method 3: From S3
function = lambda_manager.create_function(
    function_name='api-handler',
    runtime='python3.11',
    handler='lambda_function.handler',
    role_arn='arn:aws:iam::123456789012:role/lambda-execution-role',
    s3_bucket='my-lambda-deployments',
    s3_key='functions/api-handler-v1.0.0.zip'
)
```

### Invoke Lambda Function

```python
# Synchronous invocation with logs
result = lambda_manager.invoke_function(
    function_name='hello-world',
    payload={'name': 'Alice', 'action': 'greet'},
    log_type='Tail'
)

print(f"Status: {result['status_code']}")
print(f"Response: {result['payload']}")
print(f"Logs:\n{result['logs']}")
```

```python
# Asynchronous invocation (fire-and-forget)
lambda_manager.invoke_async(
    function_name='process-data',
    payload={'records': [1, 2, 3, 4, 5]}
)
print("Function invoked asynchronously")
```

### Update Function Code

```python
# Update from ZIP file
lambda_manager.update_function_code(
    function_name='my-function',
    zip_file_path='./updated-function.zip',
    publish=True  # Publish as new version
)

# Update from S3
lambda_manager.update_function_code(
    function_name='my-function',
    s3_bucket='my-bucket',
    s3_key='functions/my-function-v2.0.0.zip'
)
```

### Update Function Configuration

```python
lambda_manager.update_function_configuration(
    function_name='my-function',
    timeout=600,  # 10 minutes
    memory_size=1024,  # 1 GB
    environment_variables={
        'DB_HOST': 'prod-db.example.com',
        'CACHE_ENABLED': 'true'
    }
)
```

### Event Source Mappings

```python
# Connect to SQS queue
sqs_mapping = lambda_manager.create_event_source_mapping(
    function_name='process-orders',
    event_source_arn='arn:aws:sqs:us-east-1:123456789012:order-queue',
    batch_size=10
)

# Connect to DynamoDB Stream
dynamodb_mapping = lambda_manager.create_event_source_mapping(
    function_name='sync-to-elasticsearch',
    event_source_arn='arn:aws:dynamodb:us-east-1:123456789012:table/Orders/stream/...',
    batch_size=100,
    starting_position='LATEST'
)

# Connect to Kinesis Stream
kinesis_mapping = lambda_manager.create_event_source_mapping(
    function_name='process-clickstream',
    event_source_arn='arn:aws:kinesis:us-east-1:123456789012:stream/clickstream',
    batch_size=500,
    starting_position='TRIM_HORIZON'
)
```

### Lambda Layers

```python
# Publish layer with shared dependencies
layer = lambda_manager.publish_layer_version(
    layer_name='common-dependencies',
    description='Requests, boto3, pandas libraries',
    zip_file_path='./layers/dependencies.zip',
    compatible_runtimes=['python3.11', 'python3.10', 'python3.9']
)

print(f"Layer ARN: {layer['layer_version_arn']}")

# Create function using the layer
function = lambda_manager.create_function(
    function_name='data-processor',
    runtime='python3.11',
    handler='index.handler',
    role_arn='arn:aws:iam::123456789012:role/lambda-role',
    code_content='...',
    layers=[layer['layer_version_arn']]
)
```

### Aliases for Blue/Green Deployment

```python
# Create production alias pointing to version 1
prod_alias = lambda_manager.create_alias(
    function_name='api-handler',
    alias_name='prod',
    function_version='1',
    description='Production environment'
)

# Create dev alias pointing to latest ($LATEST)
dev_alias = lambda_manager.create_alias(
    function_name='api-handler',
    alias_name='dev',
    function_version='$LATEST'
)

# Invoke using alias
result = lambda_manager.invoke_function(
    function_name='api-handler:prod',
    payload={'endpoint': '/users', 'method': 'GET'}
)
```

### VPC Configuration

```python
# Deploy function in VPC
function = lambda_manager.create_function(
    function_name='database-processor',
    runtime='python3.11',
    handler='index.handler',
    role_arn='arn:aws:iam::123456789012:role/lambda-vpc-role',
    code_content='...',
    vpc_config={
        'SubnetIds': ['subnet-12345', 'subnet-67890'],
        'SecurityGroupIds': ['sg-abcdef123']
    }
)
```

### Monitor Functions

```python
# List all functions
functions = lambda_manager.list_functions()
for func in functions:
    print(f"{func['function_name']}: {func['runtime']} ({func['memory_size']} MB)")

# Get function details
details = lambda_manager.get_function('my-function')
print(f"Timeout: {details['timeout']}s")
print(f"Memory: {details['memory_size']} MB")
print(f"Last Modified: {details['last_modified']}")

# Get recent logs
logs = lambda_manager.get_function_logs('my-function', limit=50)
for log in logs:
    print(log)

# Get summary
summary = lambda_manager.get_summary()
print(f"Total Functions: {summary['total_functions']}")
print(f"Runtimes: {summary['runtimes']}")
```

## 🏗️ Architecture

```
LambdaManager
├── Function Management
│   ├── create_function()              # Deploy new function
│   ├── update_function_code()         # Update code
│   ├── update_function_configuration() # Update settings
│   ├── get_function()                 # Get details
│   ├── list_functions()               # List all functions
│   └── delete_function()              # Remove function
│
├── Invocation
│   ├── invoke_function()              # Sync/async invocation
│   └── invoke_async()                 # Async shorthand
│
├── Event Sources
│   ├── create_event_source_mapping()   # Connect event source
│   └── list_event_source_mappings()    # List mappings
│
├── Layers
│   └── publish_layer_version()         # Publish layer
│
├── Aliases
│   └── create_alias()                  # Create alias
│
└── Monitoring
    ├── get_function_logs()             # CloudWatch logs
    └── get_summary()                   # Statistics
```

## 📦 Packaging Lambda Functions

### Simple Function

```bash
# Create inline code (handled automatically)
# Just pass code_content parameter
```

### Function with Dependencies

```bash
# Create function directory
mkdir my-function
cd my-function

# Install dependencies
pip install requests pandas -t .

# Add your code
cat > lambda_function.py <<EOF
import requests
import pandas as pd

def handler(event, context):
    # Your code here
    return {'statusCode': 200}
EOF

# Create ZIP
zip -r ../my-function.zip .
cd ..

# Deploy
python -c "
from aws_lambda import LambdaManager
manager = LambdaManager()
manager.create_function(
    function_name='my-function',
    runtime='python3.11',
    handler='lambda_function.handler',
    role_arn='arn:aws:iam::123456789012:role/lambda-role',
    zip_file_path='./my-function.zip'
)
"
```

### Lambda Layer

```bash
# Create layer structure
mkdir -p layer/python
cd layer/python

# Install dependencies
pip install requests boto3 pandas -t .
cd ../..

# Create ZIP (must include python/ directory)
cd layer && zip -r ../dependencies-layer.zip . && cd ..

# Publish layer
python -c "
from aws_lambda import LambdaManager
manager = LambdaManager()
manager.publish_layer_version(
    layer_name='common-libs',
    description='Common dependencies',
    zip_file_path='./dependencies-layer.zip',
    compatible_runtimes=['python3.11']
)
"
```

## 🔒 Security Best Practices

1. **Least Privilege**: Grant minimal IAM permissions
2. **Environment Variables**: Use for configuration, not secrets
3. **Secrets Manager**: Store sensitive data in AWS Secrets Manager
4. **VPC**: Deploy in VPC for database/internal resource access
5. **Resource Policies**: Control who can invoke functions
6. **Encryption**: Enable encryption at rest for environment variables
7. **X-Ray**: Enable for distributed tracing
8. **Dead Letter Queues**: Configure DLQ for failed async invocations

## 📊 Common Use Cases

### API Backend

```python
# Create REST API handler
lambda_manager.create_function(
    function_name='api-users-handler',
    runtime='python3.11',
    handler='app.handler',
    role_arn='arn:aws:iam::123456789012:role/api-lambda-role',
    zip_file_path='./api.zip',
    timeout=30,
    memory_size=512,
    environment_variables={
        'DB_HOST': 'prod-db.rds.amazonaws.com',
        'API_KEY_SECRET': 'prod/api-keys'
    }
)
```

### Data Processing Pipeline

```python
# Process S3 uploads
lambda_manager.create_function(
    function_name='process-uploads',
    runtime='python3.11',
    handler='processor.handle_s3_event',
    role_arn='arn:aws:iam::123456789012:role/s3-processor-role',
    zip_file_path='./processor.zip',
    timeout=900,  # 15 minutes
    memory_size=3008  # 3 GB
)
```

### Scheduled Jobs

```python
# Daily report generation (triggered by EventBridge/CloudWatch Events)
lambda_manager.create_function(
    function_name='daily-report-generator',
    runtime='python3.11',
    handler='reports.generate_daily',
    role_arn='arn:aws:iam::123456789012:role/report-lambda-role',
    zip_file_path='./reports.zip',
    timeout=600,
    memory_size=1024
)
```

## 🐛 Troubleshooting

### Issue: "NoCredentialsError"
**Solution**: Configure AWS credentials using `aws configure`

### Issue: "InvalidParameterValueException: The role defined for the function cannot be assumed by Lambda"
**Solution**: Verify IAM role trust policy allows `lambda.amazonaws.com`

### Issue: "ResourceConflictException: Function already exists"
**Solution**: Use `update_function_code()` instead of `create_function()`

### Issue: Timeout errors
**Solution**: Increase function timeout or optimize code

### Issue: Out of memory
**Solution**: Increase memory allocation or optimize memory usage

## 📚 API Reference

### LambdaManager Class

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `create_function()` | function_name, runtime, handler, role_arn, ... | Dict | Create new function |
| `invoke_function()` | function_name, payload, invocation_type, log_type | Dict | Invoke function |
| `invoke_async()` | function_name, payload | Dict | Async invocation |
| `update_function_code()` | function_name, zip_file_path/s3_*, publish | Dict | Update code |
| `update_function_configuration()` | function_name, timeout, memory_size, ... | Dict | Update config |
| `create_event_source_mapping()` | function_name, event_source_arn, ... | Dict | Connect event source |
| `publish_layer_version()` | layer_name, description, zip_file_path, runtimes | Dict | Publish layer |
| `create_alias()` | function_name, alias_name, function_version | Dict | Create alias |
| `get_function()` | function_name | Dict | Get function details |
| `list_functions()` | - | List[Dict] | List all functions |
| `get_function_logs()` | function_name, limit | List[str] | Get CloudWatch logs |
| `get_summary()` | - | Dict | Get summary stats |

## 🔗 Related AWS Services

- **API Gateway**: HTTP APIs for Lambda
- **EventBridge**: Scheduled and event-driven triggers
- **DynamoDB Streams**: Database change events
- **SQS**: Message queuing
- **SNS**: Pub/sub notifications
- **S3**: Object storage events
- **CloudWatch**: Logging and monitoring
- **X-Ray**: Distributed tracing

## 📞 Support

For questions or issues:
- **Email**: clientbrill@gmail.com
- **LinkedIn**: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## 📄 License

**Author**: Brill Consulting

---

**Last Updated**: November 2025
