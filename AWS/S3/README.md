# AWS S3 Management

**Production-ready S3 bucket and object management with versioning, lifecycle policies, presigned URLs, and encryption.**

## 🎯 Overview

Comprehensive S3 management system featuring:
- Bucket operations (create, list, delete)
- Object management (upload, download, copy, delete)
- Versioning and lifecycle policies
- Presigned URLs for temporary access
- Encryption (SSE-S3, SSE-KMS)
- Bucket policies and ACLs

## ✨ Features

### Bucket Management
- Create/delete buckets with region selection
- List all buckets
- Configure ACLs (private, public-read, etc.)
- Force delete (remove all objects first)

### Object Operations
- Upload files or binary data
- Download objects
- List objects with prefix filtering
- Copy objects between buckets
- Delete objects
- Custom metadata and tags

### Advanced Features
- **Versioning**: Enable version control
- **Lifecycle Policies**: Auto-transition to Glacier, auto-delete old versions
- **Presigned URLs**: Temporary access without credentials
- **Encryption**: SSE-S3 and SSE-KMS support
- **Bucket Policies**: Fine-grained access control

## 📋 Prerequisites

1. **AWS Account** with S3 permissions
2. **IAM Permissions**: `s3:*` or specific permissions (CreateBucket, PutObject, GetObject, etc.)
3. **Python 3.8+**
4. **boto3**: AWS SDK

## 🚀 Installation

```bash
pip install -r requirements.txt
aws configure  # Set up credentials
```

## 💻 Usage Examples

### Initialize S3 Manager

```python
from aws_s3 import S3Manager

s3_manager = S3Manager(region='us-east-1')
```

### Bucket Operations

```python
# Create bucket
s3_manager.create_bucket('my-data-bucket', acl='private')

# List all buckets
buckets = s3_manager.list_buckets()

# Delete bucket (with force option to delete all objects)
s3_manager.delete_bucket('old-bucket', force=True)
```

### Upload and Download

```python
# Upload file
s3_manager.upload_file(
    file_path='./data.csv',
    bucket_name='my-data-bucket',
    object_key='datasets/data.csv',
    metadata={'source': 'prod'},
    tags={'Environment': 'production'}
)

# Upload from memory
s3_manager.upload_object(
    bucket_name='my-data-bucket',
    object_key='config.json',
    data=b'{"key": "value"}',
    content_type='application/json'
)

# Download file
s3_manager.download_file(
    bucket_name='my-data-bucket',
    object_key='datasets/data.csv',
    file_path='./local_data.csv'
)

# Get object content
content = s3_manager.get_object('my-data-bucket', 'config.json')
```

### List and Manage Objects

```python
# List all objects
objects = s3_manager.list_objects('my-data-bucket')

# List with prefix filter
logs = s3_manager.list_objects('my-data-bucket', prefix='logs/')

# Copy object
s3_manager.copy_object(
    source_bucket='bucket-a',
    source_key='file.txt',
    dest_bucket='bucket-b',
    dest_key='backup/file.txt'
)

# Delete object
s3_manager.delete_object('my-data-bucket', 'old-file.txt')
```

### Versioning

```python
# Enable versioning
s3_manager.enable_versioning('my-data-bucket')

# Check versioning status
status = s3_manager.get_versioning_status('my-data-bucket')
print(f"Versioning: {status}")  # Output: Enabled or Disabled
```

### Presigned URLs

```python
# Generate presigned URL for downloads (1 hour expiry)
url = s3_manager.generate_presigned_url(
    bucket_name='my-data-bucket',
    object_key='private/report.pdf',
    expiration=3600
)
print(f"Share this URL: {url}")

# Generate presigned URL for uploads
upload_url = s3_manager.generate_presigned_url(
    bucket_name='my-data-bucket',
    object_key='uploads/new-file.pdf',
    expiration=300,
    http_method='put_object'
)
```

### Lifecycle Policies

```python
# Auto-delete old versions after 90 days
# Move to Glacier after 30 days
rules = [
    {
        'ID': 'DeleteOldVersions',
        'Status': 'Enabled',
        'NoncurrentVersionExpiration': {'NoncurrentDays': 90}
    },
    {
        'ID': 'TransitionToGlacier',
        'Status': 'Enabled',
        'Filter': {'Prefix': 'archives/'},
        'Transitions': [{'Days': 30, 'StorageClass': 'GLACIER'}]
    }
]
s3_manager.put_lifecycle_policy('my-data-bucket', rules)
```

### Security

```python
# Enable default encryption (SSE-S3)
s3_manager.enable_default_encryption(
    bucket_name='my-data-bucket',
    sse_algorithm='AES256'
)

# Enable encryption with KMS
s3_manager.enable_default_encryption(
    bucket_name='my-data-bucket',
    sse_algorithm='aws:kms',
    kms_key_id='arn:aws:kms:us-east-1:123456789012:key/...'
)

# Set bucket policy
policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"AWS": "arn:aws:iam::123456789012:user/alice"},
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::my-data-bucket/*"
    }]
}
s3_manager.put_bucket_policy('my-data-bucket', policy)
```

### Monitoring

```python
# Get bucket size and object count
stats = s3_manager.get_bucket_size('my-data-bucket')
print(f"Objects: {stats['object_count']}")
print(f"Total Size: {stats['total_size_mb']} MB")

# Get summary
summary = s3_manager.get_summary()
print(f"Total buckets: {summary['total_buckets']}")
```

## 🏗️ Architecture

```
S3Manager
├── Bucket Operations
│   ├── create_bucket()
│   ├── list_buckets()
│   └── delete_bucket()
│
├── Object Operations
│   ├── upload_file()
│   ├── upload_object()
│   ├── download_file()
│   ├── get_object()
│   ├── list_objects()
│   ├── delete_object()
│   └── copy_object()
│
├── Versioning
│   ├── enable_versioning()
│   └── get_versioning_status()
│
├── Advanced Features
│   ├── generate_presigned_url()
│   ├── put_lifecycle_policy()
│   ├── put_bucket_policy()
│   └── enable_default_encryption()
│
└── Monitoring
    ├── get_bucket_size()
    └── get_summary()
```

## 🔒 Security Best Practices

1. **Private by Default**: Always use `acl='private'` unless public access is required
2. **Encryption**: Enable default encryption for sensitive data
3. **Bucket Policies**: Use least-privilege access policies
4. **Versioning**: Enable for critical data protection
5. **Presigned URLs**: Use short expiration times (≤1 hour)
6. **Block Public Access**: Enable S3 Block Public Access settings
7. **Access Logs**: Enable S3 access logging for audit trails
8. **MFA Delete**: Enable for versioned buckets with critical data

## 📊 Common Use Cases

### Static Website Hosting

```python
# Upload website files
s3_manager.upload_file('./index.html', 'my-website-bucket', 'index.html')
s3_manager.upload_file('./styles.css', 'my-website-bucket', 'styles.css')

# Set bucket policy for public read
policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::my-website-bucket/*"
    }]
}
s3_manager.put_bucket_policy('my-website-bucket', policy)
```

### Data Lake Storage

```python
# Upload with organized structure
s3_manager.upload_file('./sales_2024.csv', 'datalake-bucket', 'raw/sales/2024/sales_2024.csv')

# Set lifecycle to archive old data
rules = [{
    'ID': 'ArchiveOldData',
    'Status': 'Enabled',
    'Filter': {'Prefix': 'raw/'},
    'Transitions': [
        {'Days': 90, 'StorageClass': 'STANDARD_IA'},
        {'Days': 365, 'StorageClass': 'GLACIER'}
    ]
}]
s3_manager.put_lifecycle_policy('datalake-bucket', rules)
```

### Backup and Disaster Recovery

```python
# Enable versioning for data protection
s3_manager.enable_versioning('backup-bucket')

# Upload backup with metadata
s3_manager.upload_file(
    file_path='./database_backup.sql',
    bucket_name='backup-bucket',
    object_key='backups/daily/2024-11-05.sql',
    metadata={'backup_type': 'full', 'source': 'prod-db'}
)

# Auto-delete old backups after 30 days
rules = [{
    'ID': 'DeleteOldBackups',
    'Status': 'Enabled',
    'Expiration': {'Days': 30}
}]
s3_manager.put_lifecycle_policy('backup-bucket', rules)
```

## 🐛 Troubleshooting

**Issue: "NoSuchBucket"**
- **Solution**: Verify bucket name and ensure it exists

**Issue: "AccessDenied"**
- **Solution**: Check IAM permissions and bucket policies

**Issue: "BucketAlreadyExists"**
- **Solution**: Bucket names are globally unique; choose a different name

**Issue: "NoCredentials"**
- **Solution**: Run `aws configure` to set up credentials

## 📚 API Reference

| Method | Description |
|--------|-------------|
| `create_bucket()` | Create new S3 bucket |
| `list_buckets()` | List all buckets |
| `upload_file()` | Upload file to S3 |
| `upload_object()` | Upload binary data |
| `download_file()` | Download object to file |
| `get_object()` | Get object content as bytes |
| `list_objects()` | List objects in bucket |
| `delete_object()` | Delete object |
| `copy_object()` | Copy object |
| `enable_versioning()` | Enable bucket versioning |
| `generate_presigned_url()` | Create temporary access URL |
| `put_lifecycle_policy()` | Configure lifecycle rules |
| `enable_default_encryption()` | Enable encryption |
| `get_bucket_size()` | Get bucket statistics |

## 🔗 Related AWS Services

- **CloudFront**: CDN for S3 content delivery
- **Lambda**: Process S3 events
- **Athena**: Query S3 data with SQL
- **Glue**: ETL for S3 data lakes
- **Glacier**: Long-term archival storage

## 📞 Support

- **Email**: clientbrill@gmail.com
- **LinkedIn**: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

**Author**: Brill Consulting | **Last Updated**: November 2025
