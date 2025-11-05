# Azure Key Vault Service Integration

Advanced implementation of Azure Key Vault with secrets, keys, certificates, and comprehensive security management.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a comprehensive Python implementation for Azure Key Vault, featuring secret management, cryptographic key operations, certificate lifecycle management, access policies, and enterprise security features. Built for production applications requiring secure credential storage and encryption key management.

## Features

### Core Capabilities
- **Secret Management**: CRUD operations, versioning, rotation
- **Key Operations**: Encryption, decryption, signing, verification
- **Certificate Management**: Creation, import, auto-renewal
- **Access Policies**: RBAC and policy-based access control
- **Soft Delete**: Recoverable deletion with purge protection
- **Backup & Restore**: Full backup and recovery capabilities

### Advanced Features
- **Secret Versioning**: Track all secret versions with rollback
- **Key Rotation**: Automated and manual key rotation
- **Managed Identities**: Azure AD integration
- **Network Security**: IP filtering and virtual network rules
- **Audit Logging**: Comprehensive activity tracking
- **Purge Protection**: Irreversible deletion prevention

## Architecture

```
KeyVault/
├── keyvault.py               # Main implementation
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

### Key Components

1. **SecretManager**: Secret lifecycle management
   - Create and update secrets
   - Version management
   - Backup and restore
   - Soft delete and recovery

2. **KeyManager**: Cryptographic key operations
   - RSA and EC key creation
   - Encryption/decryption
   - Digital signatures
   - Key rotation

3. **CertificateManager**: Certificate lifecycle
   - Certificate creation
   - Import existing certificates
   - Auto-renewal policies
   - Thumbprint management

4. **KeyVaultManager**: Unified vault management
   - Access policy configuration
   - Network security rules
   - Audit logging
   - Vault properties

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/KeyVault

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your Azure Key Vault credentials:

```python
from keyvault import KeyVaultManager

manager = KeyVaultManager(
    vault_name="my-keyvault",
    vault_url="https://my-keyvault.vault.azure.net/",
    tenant_id="your-tenant-id",
    subscription_id="your-subscription-id"
)
```

### Environment Variables (Recommended)

```bash
export AZURE_KEYVAULT_NAME="my-keyvault"
export AZURE_KEYVAULT_URL="https://my-keyvault.vault.azure.net/"
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
```

## Usage Examples

### Secret Management

```python
from keyvault import KeyVaultManager, SecretContentType
from datetime import datetime, timedelta

manager = KeyVaultManager("my-keyvault", "https://my-keyvault.vault.azure.net/", 
                         "tenant-id", "subscription-id")
secret_mgr = manager.secret_manager

# Create a secret
secret = secret_mgr.set_secret(
    "database-password",
    "P@ssw0rd123!",
    content_type=SecretContentType.PASSWORD,
    expires_on=datetime.now() + timedelta(days=90),
    tags={"environment": "production"}
)

# Get secret value
retrieved = secret_mgr.get_secret("database-password")
print(f"Secret value: {retrieved.value}")

# List all secrets
for secret in secret_mgr.list_secrets():
    print(f"{secret['name']}: {secret['enabled']}")

# Rotate secret
new_version = secret_mgr.set_secret("database-password", "NewP@ssw0rd456!")

# List versions
versions = secret_mgr.list_secret_versions("database-password")
for v in versions:
    print(f"Version: {v['version']}, Created: {v['created_on']}")
```

### Key Operations

```python
from keyvault import KeyManager, KeyType, KeyOperation

key_mgr = manager.key_manager

# Create RSA key
key = key_mgr.create_key(
    "encryption-key",
    KeyType.RSA,
    key_size=2048,
    tags={"purpose": "data-encryption"}
)

# Encrypt data
plaintext = "Sensitive information"
encrypted = key_mgr.encrypt("encryption-key", plaintext)
print(f"Ciphertext: {encrypted['ciphertext']}")

# Decrypt data
decrypted = key_mgr.decrypt("encryption-key", encrypted['ciphertext'])
print(f"Plaintext: {decrypted['plaintext']}")

# Sign data
digest = "SHA256:abc123"
signature = key_mgr.sign("encryption-key", digest)

# Verify signature
verification = key_mgr.verify("encryption-key", digest, signature['signature'])
print(f"Valid signature: {verification['is_valid']}")

# Rotate key
new_key = key_mgr.rotate_key("encryption-key")
print(f"New key version: {new_key.version}")
```

### Certificate Management

```python
from keyvault import CertificateManager

cert_mgr = manager.certificate_manager

# Create certificate
cert = cert_mgr.create_certificate(
    "ssl-cert",
    subject="CN=example.com",
    validity_months=12,
    issuer="DigiCert",
    tags={"domain": "example.com"}
)

print(f"Certificate: {cert.name}")
print(f"Thumbprint: {cert.thumbprint}")
print(f"Expires: {cert.expires_on}")

# Update certificate policy
policy = cert_mgr.update_certificate_policy(
    "ssl-cert",
    auto_renew=True,
    renew_days_before_expiry=30
)

# Import certificate
imported = cert_mgr.import_certificate(
    "imported-cert",
    certificate_data="<base64-encoded-cert>",
    password="cert-password"
)
```

### Access Policies

```python
# Create access policy for user
user_policy = manager.create_access_policy(
    object_id="user-object-id",
    secret_permissions=["get", "list", "set"],
    key_permissions=["get", "list", "create", "encrypt", "decrypt"],
    certificate_permissions=["get", "list", "create"]
)

# Create policy for service principal
sp_policy = manager.create_access_policy(
    object_id="sp-object-id",
    secret_permissions=["get"],
    application_id="app-id"
)

# List all policies
for policy in manager.list_access_policies():
    print(f"Object ID: {policy['object_id']}")
    print(f"Permissions: {policy['permissions']}")
```

### Security Features

```python
# Enable soft delete
soft_delete = manager.enable_soft_delete(retention_days=90)
print(f"Soft delete enabled: {soft_delete['retention_days']} days")

# Enable purge protection
purge = manager.enable_purge_protection()
print(f"Purge protection: {purge['purge_protection_enabled']}")

# Configure network rules
network = manager.configure_network_rules(
    default_action="Deny",
    allowed_ip_ranges=["203.0.113.0/24"],
    bypass="AzureServices"
)
```

### Backup and Restore

```python
# Backup secret
backup_data = secret_mgr.backup_secret("important-secret")
print(f"Backup size: {len(backup_data)} bytes")

# Restore from backup
restored = secret_mgr.restore_secret(backup_data)
print(f"Restored: {restored.name}")
```

### Soft Delete and Recovery

```python
# Delete secret (soft delete)
delete_result = secret_mgr.delete_secret("old-secret")
print(f"Scheduled purge: {delete_result['scheduled_purge_date']}")

# Recover deleted secret
recovered = secret_mgr.recover_deleted_secret("old-secret")
print(f"Recovered: {recovered.name}")

# Permanently delete (purge)
purge_result = secret_mgr.purge_deleted_secret("old-secret")
print(f"Purged: {purge_result['status']}")
```

## Running Demos

```bash
# Run all demo functions
python keyvault.py
```

Demo output includes:
- Secret CRUD operations and versioning
- Key encryption/decryption and signing
- Certificate management
- Access policy configuration
- Security features demonstration
- Vault management operations

## Best Practices

### 1. Secret Management
- Use content type hints for better organization
- Set expiration dates for sensitive secrets
- Regularly rotate secrets and keys
- Use tags for environment and application tracking

### 2. Access Control
- Follow principle of least privilege
- Use separate policies for users and applications
- Regularly audit access policies
- Use managed identities when possible

### 3. Security
- Always enable soft delete for production vaults
- Enable purge protection for critical vaults
- Configure network rules to restrict access
- Monitor audit logs for suspicious activity

### 4. Key Management
- Use HSM-backed keys for sensitive operations
- Rotate keys according to compliance requirements
- Use appropriate key sizes (2048-bit minimum for RSA)
- Implement key versioning strategy

### 5. Certificate Management
- Configure auto-renewal before expiration
- Monitor certificate expiration dates
- Use reputable certificate authorities
- Implement certificate rotation procedures

## Use Cases

### 1. Database Connection Strings
```python
secret_mgr.set_secret(
    "db-connection",
    "Server=sql.example.com;Database=prod;User=admin;Password=secret",
    content_type=SecretContentType.CONNECTION_STRING,
    tags={"database": "production"}
)
```

### 2. API Key Storage
```python
secret_mgr.set_secret(
    "external-api-key",
    "sk-1234567890abcdef",
    content_type=SecretContentType.API_KEY,
    expires_on=datetime.now() + timedelta(days=365)
)
```

### 3. Data Encryption
```python
# Create encryption key
key_mgr.create_key("data-encryption-key", KeyType.RSA, key_size=4096)

# Encrypt sensitive data
encrypted_data = key_mgr.encrypt("data-encryption-key", sensitive_info)

# Store encrypted data, decrypt when needed
decrypted_data = key_mgr.decrypt("data-encryption-key", encrypted_data['ciphertext'])
```

### 4. SSL/TLS Certificate Management
```python
cert_mgr.create_certificate(
    "web-ssl-cert",
    subject="CN=www.example.com",
    validity_months=12,
    issuer="LetsEncrypt"
)

cert_mgr.update_certificate_policy(
    "web-ssl-cert",
    auto_renew=True,
    renew_days_before_expiry=30
)
```

## API Reference

### SecretManager Methods

- `set_secret(name, value, content_type, enabled, expires_on, tags)`: Create/update secret
- `get_secret(name, version)`: Retrieve secret value
- `list_secrets()`: List all secrets
- `list_secret_versions(name)`: List versions of a secret
- `update_secret_properties(name, enabled, expires_on, tags)`: Update metadata
- `delete_secret(name)`: Soft delete secret
- `recover_deleted_secret(name)`: Recover deleted secret
- `purge_deleted_secret(name)`: Permanently delete secret
- `backup_secret(name)`: Backup secret
- `restore_secret(backup_data)`: Restore from backup

### KeyManager Methods

- `create_key(name, key_type, key_size, operations, expires_on, tags)`: Create key
- `get_key(name, version)`: Retrieve key
- `list_keys()`: List all keys
- `encrypt(key_name, plaintext, algorithm)`: Encrypt data
- `decrypt(key_name, ciphertext, algorithm)`: Decrypt data
- `sign(key_name, digest, algorithm)`: Sign data
- `verify(key_name, digest, signature, algorithm)`: Verify signature
- `rotate_key(name)`: Create new key version
- `delete_key(name)`: Soft delete key

### CertificateManager Methods

- `create_certificate(name, subject, validity_months, issuer, tags)`: Create certificate
- `import_certificate(name, certificate_data, password, tags)`: Import certificate
- `get_certificate(name, version)`: Retrieve certificate
- `list_certificates()`: List all certificates
- `update_certificate_policy(name, auto_renew, renew_days_before_expiry)`: Update policy
- `delete_certificate(name)`: Soft delete certificate

### KeyVaultManager Methods

- `create_access_policy(object_id, permissions, application_id)`: Create access policy
- `list_access_policies()`: List all policies
- `enable_soft_delete(retention_days)`: Enable soft delete
- `enable_purge_protection()`: Enable purge protection
- `configure_network_rules(default_action, allowed_ips, vnets)`: Configure network
- `get_vault_properties()`: Get vault information
- `audit_log_entry(operation, resource_type, resource_name, principal_id, result)`: Log activity
- `get_audit_logs(start_time, end_time, operation)`: Retrieve audit logs

## Performance Optimization

### Caching
```python
# Cache frequently accessed secrets
secret_cache = {}
def get_cached_secret(name):
    if name not in secret_cache:
        secret_cache[name] = secret_mgr.get_secret(name)
    return secret_cache[name]
```

### Batch Operations
```python
# Retrieve multiple secrets efficiently
secret_names = ["secret1", "secret2", "secret3"]
secrets = [secret_mgr.get_secret(name) for name in secret_names]
```

## Security Considerations

1. **Credential Protection**: Never log or print secret values
2. **Network Security**: Use private endpoints in production
3. **Access Control**: Implement RBAC and Azure AD authentication
4. **Audit Logging**: Enable diagnostic settings in Azure
5. **Key Rotation**: Automate rotation policies
6. **Backup**: Regular backups for critical secrets

## Monitoring

### Key Metrics
- Secret/key access frequency
- Failed authentication attempts
- Secret/key expiration dates
- Rotation compliance
- Audit log volume

### Audit Logging
```python
# Add audit entry
manager.audit_log_entry(
    operation="SecretGet",
    resource_type="secret",
    resource_name="api-key",
    principal_id="user@example.com",
    result="success"
)

# Query audit logs
logs = manager.get_audit_logs(
    start_time=datetime.now() - timedelta(hours=24),
    operation="SecretGet"
)
```

## Troubleshooting

### Common Issues

**Issue**: Access denied errors
**Solution**: Verify access policies and Azure AD permissions

**Issue**: Secret not found
**Solution**: Check secret name and version, verify vault URL

**Issue**: Key operation failed
**Solution**: Verify key operations are enabled for the key type

**Issue**: Certificate expiration
**Solution**: Configure auto-renewal policies

## Dependencies

```
Python >= 3.8
dataclasses
typing
json
base64
```

See `requirements.txt` for complete list.

## Version History

- **v1.0.0**: Initial release with secret management
- **v2.0.0**: Added key operations and certificate management
- **v2.1.0**: Enhanced security features and audit logging

## Contributing

Contributions are welcome! Please submit pull requests or open issues on GitHub.

## License

This project is part of the Brill Consulting portfolio.

## Support

For questions or support:
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure Monitor](../AzureMonitor/)
- [Azure Identity](../AzureIdentity/)
- [Azure Security](../AzureSecurity/)

---

**Built with Azure Key Vault** | **Brill Consulting © 2024**
