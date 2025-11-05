# AWS Secrets Manager

Secure secrets management with automatic rotation and versioning.

## Features

- **Secret Storage**: Store database credentials, API keys, tokens
- **Automatic Rotation**: Lambda-based rotation for RDS, Redshift
- **Versioning**: Track secret changes with version IDs
- **Encryption**: KMS encryption at rest
- **Access Control**: IAM and resource-based policies
- **Cross-Region Replication**: Replicate secrets for disaster recovery

## Quick Start

```python
from aws_secretsmanager import SecretsManagerManager
import json

# Initialize
sm = SecretsManagerManager(region='us-east-1')

# Create database secret
secret = sm.create_secret(
    name='prod/db/credentials',
    secret_string=json.dumps({
        'username': 'admin',
        'password': 'secure-password-123',
        'host': 'db.example.com',
        'port': 5432
    }),
    description='Production database credentials'
)

# Retrieve secret
secret_value = sm.get_secret_value('prod/db/credentials')
credentials = json.loads(secret_value['secret_string'])

# Update secret
sm.update_secret(
    'prod/db/credentials',
    secret_string=json.dumps({
        'username': 'admin',
        'password': 'new-secure-password-456',
        'host': 'db.example.com',
        'port': 5432
    })
)

# Enable automatic rotation
sm.rotate_secret(
    'prod/db/credentials',
    rotation_lambda_arn='arn:aws:lambda:us-east-1:123456789012:function:rotate-db'
)
```

## Use Cases

- **Database Credentials**: RDS, Aurora, DynamoDB
- **API Keys**: Third-party service authentication
- **SSH Keys**: Secure server access
- **OAuth Tokens**: Application credentials
- **Encryption Keys**: Application-level encryption

## Security Best Practices

- Use IAM policies to restrict access
- Enable CloudTrail logging for audit trails
- Implement automatic rotation where possible
- Use resource-based policies for cross-account access
- Tag secrets for organization and billing

## Author

Brill Consulting
