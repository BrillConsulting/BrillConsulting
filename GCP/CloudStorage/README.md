# Cloud Storage - Advanced Object Storage

Comprehensive Cloud Storage implementation for managing buckets, objects, lifecycle policies, IAM, versioning, and signed URLs.

## Features

### Bucket Management
- **Bucket Creation**: Create buckets with location and storage class
- **Versioning**: Enable object versioning for data protection
- **CORS Configuration**: Configure cross-origin resource sharing
- **Storage Classes**: STANDARD, NEARLINE, COLDLINE, ARCHIVE

### Object Operations
- **Upload/Download**: Transfer objects to/from buckets
- **Signed URLs**: Generate temporary access URLs (v4 signing)
- **Custom Metadata**: Set key-value metadata on objects
- **Batch Operations**: Upload multiple files in parallel

### Lifecycle Policies
- **Age-Based Deletion**: Auto-delete old objects
- **Storage Class Transitions**: Move to cheaper storage over time
- **Version Management**: Delete non-current versions

### IAM & Access Control
- **IAM Policies**: Grant fine-grained access to buckets
- **Public Access**: Make buckets publicly readable
- **Service Account Access**: Configure service account permissions

### Event Notifications
- **Pub/Sub Integration**: Trigger events on object changes
- **Event Types**: OBJECT_FINALIZE, OBJECT_DELETE, OBJECT_ARCHIVE
- **Custom Attributes**: Add metadata to notifications

## Usage Example

```python
from gcp_storage import CloudStorageManager

mgr = CloudStorageManager(project_id='my-gcp-project')

# Create bucket with lifecycle
bucket = mgr.buckets.create_bucket({
    'name': 'my-data-bucket',
    'location': 'US',
    'storage_class': 'STANDARD'
})

# Upload and generate signed URL
mgr.objects.upload_object({
    'bucket_name': 'my-data-bucket',
    'object_name': 'file.txt',
    'source_file': '/path/to/file.txt'
})

signed_url = mgr.objects.generate_signed_url({
    'bucket_name': 'my-data-bucket',
    'object_name': 'file.txt',
    'expiration_minutes': 15
})
```

## Storage Classes

- **STANDARD**: $0.020/GB/month - Hot data
- **NEARLINE**: $0.010/GB/month - < once/month access
- **COLDLINE**: $0.004/GB/month - < once/quarter
- **ARCHIVE**: $0.0012/GB/month - Long-term archival

## Best Practices

1. **Enable versioning** for important data
2. **Set lifecycle policies** to reduce costs
3. **Use signed URLs** for temporary access
4. **Configure CORS** for web applications
5. **Use IAM** for fine-grained access control
6. **Enable notifications** for event-driven workflows

## Requirements

```
google-cloud-storage
```

## Author

BrillConsulting - Enterprise Cloud Solutions
