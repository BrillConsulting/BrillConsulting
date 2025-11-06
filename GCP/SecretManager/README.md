# Secret Manager - Secure Secret Storage

Comprehensive Secret Manager implementation for storing, managing, and accessing sensitive data with automatic rotation and version control.

## Features

### Secret Creation
- **Secret Storage**: Create secrets with automatic replication
- **Version Management**: Multiple versions per secret
- **Labels**: Organize secrets with custom labels
- **Replication**: Automatic or user-managed replication

### Secret Access
- **Version Access**: Access specific or latest secret versions
- **Secure Retrieval**: Encrypted secret value retrieval
- **Version Listing**: List all versions of a secret
- **Audit Logging**: Track all secret access

### Version Lifecycle
- **Enable/Disable**: Control secret version availability
- **Version Deletion**: Disable old secret versions
- **Destroy Versions**: Permanently delete secret versions
- **State Management**: ENABLED, DISABLED, DESTROYED states

### Automatic Rotation
- **Scheduled Rotation**: Automatic secret rotation with Cloud Scheduler
- **Cloud Functions Integration**: Custom rotation logic
- **Rotation Policies**: Configure rotation frequency (daily, weekly, monthly)
- **Password Generation**: Automatic secure password generation

### IAM and Access Control
- **IAM Policies**: Grant secret access to service accounts
- **Role Management**: secretAccessor, secretVersionManager roles
- **Custom Roles**: Create fine-grained access roles
- **Access Conditions**: Conditional IAM bindings

## Usage Example

```python
from secretmanager import SecretManager

# Initialize manager
mgr = SecretManager(project_id='my-gcp-project')

# Create secret
secret = mgr.creation.create_secret({
    'secret_id': 'database-password',
    'replication': 'automatic',
    'labels': {'env': 'production', 'type': 'database'}
})

# Add secret version
version = mgr.creation.add_secret_version({
    'secret_id': 'database-password',
    'payload': 'super-secure-password-123'
})

# Access secret
value_code = mgr.access.access_secret_version('database-password', 'latest')

# Configure automatic rotation
rotation = mgr.rotation.configure_rotation({
    'secret_id': 'database-password',
    'rotation_period_days': 30,
    'rotation_function': 'rotate-db-password'
})

# Grant access
mgr.iam.grant_secret_access({
    'secret_id': 'database-password',
    'member': 'serviceAccount:app@project.iam.gserviceaccount.com',
    'role': 'roles/secretmanager.secretAccessor'
})

# Disable old version
mgr.versioning.disable_secret_version('database-password', '1')
```

## Best Practices

1. **Enable automatic rotation** for sensitive secrets
2. **Use service accounts** with least privilege
3. **Enable audit logging** for all secret access
4. **Use labels** for secret organization
5. **Destroy old versions** after rotation
6. **Never log** actual secret values

## Requirements

```
google-cloud-secret-manager
google-cloud-scheduler
```

## Author

BrillConsulting - Enterprise Cloud Solutions
