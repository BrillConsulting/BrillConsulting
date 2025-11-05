"""
Azure Key Vault Service Integration
Author: BrillConsulting
Contact: clientbrill@gmail.com
LinkedIn: brillconsulting
Description: Advanced Key Vault implementation with secrets, keys, certificates, and security management
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import base64


class SecretContentType(Enum):
    """Content types for secrets"""
    TEXT = "text/plain"
    JSON = "application/json"
    PASSWORD = "application/x-password"
    CONNECTION_STRING = "application/x-connection-string"
    API_KEY = "application/x-api-key"


class KeyType(Enum):
    """Key types supported by Key Vault"""
    RSA = "RSA"
    RSA_HSM = "RSA-HSM"
    EC = "EC"
    EC_HSM = "EC-HSM"
    OCT = "oct"
    OCT_HSM = "oct-HSM"


class KeyOperation(Enum):
    """Cryptographic operations"""
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    SIGN = "sign"
    VERIFY = "verify"
    WRAP_KEY = "wrapKey"
    UNWRAP_KEY = "unwrapKey"


class CertificateContentType(Enum):
    """Certificate content types"""
    PEM = "application/x-pem-file"
    PKCS12 = "application/x-pkcs12"


class DeletionRecoveryLevel(Enum):
    """Deletion recovery levels"""
    PURGEABLE = "Purgeable"
    RECOVERABLE = "Recoverable"
    RECOVERABLE_PROTECTED = "Recoverable+ProtectedSubscription"
    CUSTOMIZED_RECOVERABLE = "CustomizedRecoverable"
    CUSTOMIZED_RECOVERABLE_PROTECTED = "CustomizedRecoverable+ProtectedSubscription"


@dataclass
class SecretProperties:
    """Secret properties and metadata"""
    name: str
    value: str
    version: str
    content_type: Optional[str] = None
    enabled: bool = True
    expires_on: Optional[str] = None
    not_before: Optional[str] = None
    created_on: Optional[str] = None
    updated_on: Optional[str] = None
    recovery_level: str = "Recoverable"
    tags: Optional[Dict[str, str]] = None


@dataclass
class KeyProperties:
    """Key properties and metadata"""
    name: str
    key_type: KeyType
    key_size: int
    version: str
    enabled: bool = True
    expires_on: Optional[str] = None
    not_before: Optional[str] = None
    created_on: Optional[str] = None
    updated_on: Optional[str] = None
    operations: Optional[List[str]] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class CertificateProperties:
    """Certificate properties and metadata"""
    name: str
    version: str
    thumbprint: str
    enabled: bool = True
    expires_on: Optional[str] = None
    not_before: Optional[str] = None
    created_on: Optional[str] = None
    updated_on: Optional[str] = None
    subject: Optional[str] = None
    issuer: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class AccessPolicy:
    """Key Vault access policy"""
    tenant_id: str
    object_id: str
    application_id: Optional[str] = None
    permissions: Optional[Dict[str, List[str]]] = None


class SecretManager:
    """
    Manage secrets in Azure Key Vault

    Features:
    - Secret CRUD operations
    - Secret versioning
    - Secret rotation
    - Backup and restore
    - Soft delete and recovery
    """

    def __init__(self, vault_url: str):
        """
        Initialize secret manager

        Args:
            vault_url: Key Vault URL (e.g., https://myvault.vault.azure.net/)
        """
        self.vault_url = vault_url
        self.secrets: Dict[str, Dict[str, SecretProperties]] = {}
        self.deleted_secrets: Dict[str, SecretProperties] = {}

    def set_secret(
        self,
        name: str,
        value: str,
        content_type: Optional[SecretContentType] = None,
        enabled: bool = True,
        expires_on: Optional[datetime] = None,
        not_before: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> SecretProperties:
        """
        Create or update a secret

        Args:
            name: Secret name
            value: Secret value
            content_type: Content type hint
            enabled: Whether secret is enabled
            expires_on: Expiration time
            not_before: Not valid before time
            tags: Secret tags

        Returns:
            SecretProperties object
        """
        version = f"v{datetime.now().timestamp()}"

        secret = SecretProperties(
            name=name,
            value=value,
            version=version,
            content_type=content_type.value if content_type else None,
            enabled=enabled,
            expires_on=expires_on.isoformat() if expires_on else None,
            not_before=not_before.isoformat() if not_before else None,
            created_on=datetime.now().isoformat(),
            updated_on=datetime.now().isoformat(),
            tags=tags or {}
        )

        # Store with version history
        if name not in self.secrets:
            self.secrets[name] = {}
        self.secrets[name][version] = secret

        return secret

    def get_secret(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[SecretProperties]:
        """
        Get a secret value

        Args:
            name: Secret name
            version: Specific version (latest if not specified)

        Returns:
            SecretProperties or None if not found
        """
        if name not in self.secrets:
            return None

        versions = self.secrets[name]
        if not versions:
            return None

        if version:
            return versions.get(version)

        # Return latest version
        latest_version = max(versions.keys())
        return versions[latest_version]

    def list_secrets(self) -> List[Dict[str, Any]]:
        """
        List all secrets (without values)

        Returns:
            List of secret metadata
        """
        results = []
        for name, versions in self.secrets.items():
            if versions:
                latest_version = max(versions.keys())
                secret = versions[latest_version]
                results.append({
                    "name": secret.name,
                    "version": secret.version,
                    "enabled": secret.enabled,
                    "created_on": secret.created_on,
                    "updated_on": secret.updated_on,
                    "tags": secret.tags
                })
        return results

    def list_secret_versions(self, name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a secret

        Args:
            name: Secret name

        Returns:
            List of secret versions
        """
        if name not in self.secrets:
            return []

        versions = []
        for version, secret in self.secrets[name].items():
            versions.append({
                "version": version,
                "enabled": secret.enabled,
                "created_on": secret.created_on,
                "expires_on": secret.expires_on
            })

        return sorted(versions, key=lambda x: x["created_on"], reverse=True)

    def update_secret_properties(
        self,
        name: str,
        enabled: Optional[bool] = None,
        expires_on: Optional[datetime] = None,
        content_type: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        version: Optional[str] = None
    ) -> SecretProperties:
        """
        Update secret properties without changing value

        Args:
            name: Secret name
            enabled: Enable/disable secret
            expires_on: Expiration time
            content_type: Content type
            tags: Tags to update
            version: Specific version (latest if not specified)

        Returns:
            Updated SecretProperties
        """
        secret = self.get_secret(name, version)
        if not secret:
            raise ValueError(f"Secret '{name}' not found")

        if enabled is not None:
            secret.enabled = enabled
        if expires_on is not None:
            secret.expires_on = expires_on.isoformat()
        if content_type is not None:
            secret.content_type = content_type
        if tags is not None:
            secret.tags = tags

        secret.updated_on = datetime.now().isoformat()

        return secret

    def delete_secret(self, name: str) -> Dict[str, Any]:
        """
        Delete a secret (soft delete)

        Args:
            name: Secret name

        Returns:
            Deletion information
        """
        if name not in self.secrets:
            raise ValueError(f"Secret '{name}' not found")

        # Get latest version for deleted secrets tracking
        latest_version = max(self.secrets[name].keys())
        deleted_secret = self.secrets[name][latest_version]

        # Move to deleted secrets
        self.deleted_secrets[name] = deleted_secret
        del self.secrets[name]

        return {
            "name": name,
            "deleted_date": datetime.now().isoformat(),
            "scheduled_purge_date": (datetime.now() + timedelta(days=90)).isoformat(),
            "recovery_id": f"{self.vault_url}/deletedsecrets/{name}"
        }

    def recover_deleted_secret(self, name: str) -> SecretProperties:
        """
        Recover a deleted secret

        Args:
            name: Secret name

        Returns:
            Recovered SecretProperties
        """
        if name not in self.deleted_secrets:
            raise ValueError(f"Deleted secret '{name}' not found")

        secret = self.deleted_secrets[name]

        # Restore to active secrets
        if name not in self.secrets:
            self.secrets[name] = {}
        self.secrets[name][secret.version] = secret

        del self.deleted_secrets[name]

        return secret

    def purge_deleted_secret(self, name: str) -> Dict[str, Any]:
        """
        Permanently delete a secret

        Args:
            name: Secret name

        Returns:
            Purge confirmation
        """
        if name not in self.deleted_secrets:
            raise ValueError(f"Deleted secret '{name}' not found")

        del self.deleted_secrets[name]

        return {
            "name": name,
            "purged_date": datetime.now().isoformat(),
            "status": "permanently_deleted"
        }

    def backup_secret(self, name: str) -> str:
        """
        Backup a secret

        Args:
            name: Secret name

        Returns:
            Base64 encoded backup data
        """
        if name not in self.secrets:
            raise ValueError(f"Secret '{name}' not found")

        backup_data = {
            "name": name,
            "versions": {
                version: asdict(secret)
                for version, secret in self.secrets[name].items()
            },
            "backup_date": datetime.now().isoformat()
        }

        # Encode as base64
        backup_json = json.dumps(backup_data)
        backup_b64 = base64.b64encode(backup_json.encode()).decode()

        return backup_b64

    def restore_secret(self, backup_data: str) -> SecretProperties:
        """
        Restore a secret from backup

        Args:
            backup_data: Base64 encoded backup data

        Returns:
            Restored SecretProperties
        """
        # Decode backup
        backup_json = base64.b64decode(backup_data.encode()).decode()
        backup = json.loads(backup_json)

        name = backup["name"]

        # Restore all versions
        self.secrets[name] = {}
        for version, secret_dict in backup["versions"].items():
            secret = SecretProperties(**secret_dict)
            self.secrets[name][version] = secret

        # Return latest version
        latest_version = max(self.secrets[name].keys())
        return self.secrets[name][latest_version]


class KeyManager:
    """
    Manage cryptographic keys in Azure Key Vault

    Features:
    - Key creation and management
    - Encryption/decryption
    - Signing and verification
    - Key rotation
    - Key wrapping
    """

    def __init__(self, vault_url: str):
        """
        Initialize key manager

        Args:
            vault_url: Key Vault URL
        """
        self.vault_url = vault_url
        self.keys: Dict[str, Dict[str, KeyProperties]] = {}
        self.deleted_keys: Dict[str, KeyProperties] = {}

    def create_key(
        self,
        name: str,
        key_type: KeyType,
        key_size: int = 2048,
        enabled: bool = True,
        operations: Optional[List[KeyOperation]] = None,
        expires_on: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> KeyProperties:
        """
        Create a new cryptographic key

        Args:
            name: Key name
            key_type: Type of key (RSA, EC, etc.)
            key_size: Key size in bits
            enabled: Whether key is enabled
            operations: Allowed key operations
            expires_on: Expiration time
            tags: Key tags

        Returns:
            KeyProperties object
        """
        version = f"v{datetime.now().timestamp()}"

        # Default operations based on key type
        if operations is None:
            if key_type in [KeyType.RSA, KeyType.RSA_HSM]:
                operations = [
                    KeyOperation.ENCRYPT,
                    KeyOperation.DECRYPT,
                    KeyOperation.SIGN,
                    KeyOperation.VERIFY,
                    KeyOperation.WRAP_KEY,
                    KeyOperation.UNWRAP_KEY
                ]
            else:
                operations = [KeyOperation.SIGN, KeyOperation.VERIFY]

        key = KeyProperties(
            name=name,
            key_type=key_type,
            key_size=key_size,
            version=version,
            enabled=enabled,
            expires_on=expires_on.isoformat() if expires_on else None,
            created_on=datetime.now().isoformat(),
            updated_on=datetime.now().isoformat(),
            operations=[op.value for op in operations],
            tags=tags or {}
        )

        if name not in self.keys:
            self.keys[name] = {}
        self.keys[name][version] = key

        return key

    def get_key(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[KeyProperties]:
        """
        Get a key

        Args:
            name: Key name
            version: Specific version (latest if not specified)

        Returns:
            KeyProperties or None if not found
        """
        if name not in self.keys:
            return None

        versions = self.keys[name]
        if not versions:
            return None

        if version:
            return versions.get(version)

        latest_version = max(versions.keys())
        return versions[latest_version]

    def list_keys(self) -> List[Dict[str, Any]]:
        """
        List all keys

        Returns:
            List of key metadata
        """
        results = []
        for name, versions in self.keys.items():
            if versions:
                latest_version = max(versions.keys())
                key = versions[latest_version]
                results.append({
                    "name": key.name,
                    "key_type": key.key_type.value,
                    "key_size": key.key_size,
                    "enabled": key.enabled,
                    "created_on": key.created_on,
                    "operations": key.operations
                })
        return results

    def encrypt(
        self,
        key_name: str,
        plaintext: str,
        algorithm: str = "RSA-OAEP-256"
    ) -> Dict[str, Any]:
        """
        Encrypt data with a key

        Args:
            key_name: Key name
            plaintext: Data to encrypt
            algorithm: Encryption algorithm

        Returns:
            Encryption result with ciphertext
        """
        key = self.get_key(key_name)
        if not key:
            raise ValueError(f"Key '{key_name}' not found")

        if not key.enabled:
            raise ValueError(f"Key '{key_name}' is disabled")

        # Simulate encryption
        ciphertext = base64.b64encode(plaintext.encode()).decode()

        return {
            "key_id": f"{self.vault_url}/keys/{key_name}/{key.version}",
            "ciphertext": ciphertext,
            "algorithm": algorithm,
            "encrypted_at": datetime.now().isoformat()
        }

    def decrypt(
        self,
        key_name: str,
        ciphertext: str,
        algorithm: str = "RSA-OAEP-256"
    ) -> Dict[str, Any]:
        """
        Decrypt data with a key

        Args:
            key_name: Key name
            ciphertext: Data to decrypt
            algorithm: Decryption algorithm

        Returns:
            Decryption result with plaintext
        """
        key = self.get_key(key_name)
        if not key:
            raise ValueError(f"Key '{key_name}' not found")

        # Simulate decryption
        plaintext = base64.b64decode(ciphertext.encode()).decode()

        return {
            "key_id": f"{self.vault_url}/keys/{key_name}/{key.version}",
            "plaintext": plaintext,
            "algorithm": algorithm,
            "decrypted_at": datetime.now().isoformat()
        }

    def sign(
        self,
        key_name: str,
        digest: str,
        algorithm: str = "RS256"
    ) -> Dict[str, Any]:
        """
        Sign data with a key

        Args:
            key_name: Key name
            digest: Data digest to sign
            algorithm: Signing algorithm

        Returns:
            Signature result
        """
        key = self.get_key(key_name)
        if not key:
            raise ValueError(f"Key '{key_name}' not found")

        # Simulate signing
        signature = base64.b64encode(f"SIGNATURE:{digest}".encode()).decode()

        return {
            "key_id": f"{self.vault_url}/keys/{key_name}/{key.version}",
            "signature": signature,
            "algorithm": algorithm,
            "signed_at": datetime.now().isoformat()
        }

    def verify(
        self,
        key_name: str,
        digest: str,
        signature: str,
        algorithm: str = "RS256"
    ) -> Dict[str, Any]:
        """
        Verify a signature

        Args:
            key_name: Key name
            digest: Original data digest
            signature: Signature to verify
            algorithm: Signing algorithm

        Returns:
            Verification result
        """
        key = self.get_key(key_name)
        if not key:
            raise ValueError(f"Key '{key_name}' not found")

        # Simulate verification
        expected_sig = base64.b64encode(f"SIGNATURE:{digest}".encode()).decode()
        is_valid = signature == expected_sig

        return {
            "key_id": f"{self.vault_url}/keys/{key_name}/{key.version}",
            "is_valid": is_valid,
            "algorithm": algorithm,
            "verified_at": datetime.now().isoformat()
        }

    def rotate_key(self, name: str) -> KeyProperties:
        """
        Create a new version of a key (rotation)

        Args:
            name: Key name

        Returns:
            New KeyProperties
        """
        current_key = self.get_key(name)
        if not current_key:
            raise ValueError(f"Key '{name}' not found")

        # Create new version with same properties
        new_key = self.create_key(
            name=name,
            key_type=current_key.key_type,
            key_size=current_key.key_size,
            enabled=current_key.enabled,
            operations=[KeyOperation(op) for op in current_key.operations],
            tags=current_key.tags
        )

        return new_key

    def delete_key(self, name: str) -> Dict[str, Any]:
        """
        Delete a key (soft delete)

        Args:
            name: Key name

        Returns:
            Deletion information
        """
        if name not in self.keys:
            raise ValueError(f"Key '{name}' not found")

        latest_version = max(self.keys[name].keys())
        deleted_key = self.keys[name][latest_version]

        self.deleted_keys[name] = deleted_key
        del self.keys[name]

        return {
            "name": name,
            "deleted_date": datetime.now().isoformat(),
            "scheduled_purge_date": (datetime.now() + timedelta(days=90)).isoformat(),
            "recovery_id": f"{self.vault_url}/deletedkeys/{name}"
        }


class CertificateManager:
    """
    Manage certificates in Azure Key Vault

    Features:
    - Certificate creation and import
    - Certificate policies
    - Auto-renewal
    - Certificate operations
    """

    def __init__(self, vault_url: str):
        """
        Initialize certificate manager

        Args:
            vault_url: Key Vault URL
        """
        self.vault_url = vault_url
        self.certificates: Dict[str, Dict[str, CertificateProperties]] = {}
        self.deleted_certificates: Dict[str, CertificateProperties] = {}

    def create_certificate(
        self,
        name: str,
        subject: str,
        validity_months: int = 12,
        enabled: bool = True,
        issuer: str = "Self",
        key_type: str = "RSA",
        key_size: int = 2048,
        tags: Optional[Dict[str, str]] = None
    ) -> CertificateProperties:
        """
        Create a new certificate

        Args:
            name: Certificate name
            subject: Subject name (e.g., CN=example.com)
            validity_months: Validity period in months
            enabled: Whether certificate is enabled
            issuer: Certificate issuer
            key_type: Key type (RSA, EC)
            key_size: Key size in bits
            tags: Certificate tags

        Returns:
            CertificateProperties object
        """
        version = f"v{datetime.now().timestamp()}"

        # Generate thumbprint
        thumbprint = base64.b64encode(f"{name}:{version}".encode()).decode()[:40]

        cert = CertificateProperties(
            name=name,
            version=version,
            thumbprint=thumbprint,
            enabled=enabled,
            subject=subject,
            issuer=issuer,
            created_on=datetime.now().isoformat(),
            updated_on=datetime.now().isoformat(),
            not_before=datetime.now().isoformat(),
            expires_on=(datetime.now() + timedelta(days=validity_months * 30)).isoformat(),
            tags=tags or {}
        )

        if name not in self.certificates:
            self.certificates[name] = {}
        self.certificates[name][version] = cert

        return cert

    def import_certificate(
        self,
        name: str,
        certificate_data: str,
        password: Optional[str] = None,
        enabled: bool = True,
        tags: Optional[Dict[str, str]] = None
    ) -> CertificateProperties:
        """
        Import a certificate

        Args:
            name: Certificate name
            certificate_data: Base64 encoded certificate (PEM or PKCS12)
            password: Password for PKCS12
            enabled: Whether certificate is enabled
            tags: Certificate tags

        Returns:
            CertificateProperties object
        """
        version = f"v{datetime.now().timestamp()}"
        thumbprint = base64.b64encode(f"{name}:{version}".encode()).decode()[:40]

        cert = CertificateProperties(
            name=name,
            version=version,
            thumbprint=thumbprint,
            enabled=enabled,
            subject="CN=imported",
            issuer="Unknown",
            created_on=datetime.now().isoformat(),
            updated_on=datetime.now().isoformat(),
            tags=tags or {}
        )

        if name not in self.certificates:
            self.certificates[name] = {}
        self.certificates[name][version] = cert

        return cert

    def get_certificate(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[CertificateProperties]:
        """
        Get a certificate

        Args:
            name: Certificate name
            version: Specific version (latest if not specified)

        Returns:
            CertificateProperties or None if not found
        """
        if name not in self.certificates:
            return None

        versions = self.certificates[name]
        if not versions:
            return None

        if version:
            return versions.get(version)

        latest_version = max(versions.keys())
        return versions[latest_version]

    def list_certificates(self) -> List[Dict[str, Any]]:
        """
        List all certificates

        Returns:
            List of certificate metadata
        """
        results = []
        for name, versions in self.certificates.items():
            if versions:
                latest_version = max(versions.keys())
                cert = versions[latest_version]
                results.append({
                    "name": cert.name,
                    "thumbprint": cert.thumbprint,
                    "subject": cert.subject,
                    "issuer": cert.issuer,
                    "enabled": cert.enabled,
                    "expires_on": cert.expires_on
                })
        return results

    def update_certificate_policy(
        self,
        name: str,
        auto_renew: bool = True,
        renew_days_before_expiry: int = 30,
        issuer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update certificate policy

        Args:
            name: Certificate name
            auto_renew: Enable auto-renewal
            renew_days_before_expiry: Days before expiry to renew
            issuer: Certificate issuer

        Returns:
            Updated policy
        """
        cert = self.get_certificate(name)
        if not cert:
            raise ValueError(f"Certificate '{name}' not found")

        policy = {
            "certificate_name": name,
            "auto_renew": auto_renew,
            "renew_days_before_expiry": renew_days_before_expiry,
            "issuer": issuer or cert.issuer,
            "key_type": "RSA",
            "key_size": 2048,
            "updated_at": datetime.now().isoformat()
        }

        return policy

    def delete_certificate(self, name: str) -> Dict[str, Any]:
        """
        Delete a certificate (soft delete)

        Args:
            name: Certificate name

        Returns:
            Deletion information
        """
        if name not in self.certificates:
            raise ValueError(f"Certificate '{name}' not found")

        latest_version = max(self.certificates[name].keys())
        deleted_cert = self.certificates[name][latest_version]

        self.deleted_certificates[name] = deleted_cert
        del self.certificates[name]

        return {
            "name": name,
            "deleted_date": datetime.now().isoformat(),
            "scheduled_purge_date": (datetime.now() + timedelta(days=90)).isoformat(),
            "recovery_id": f"{self.vault_url}/deletedcertificates/{name}"
        }


class KeyVaultManager:
    """
    Comprehensive Azure Key Vault manager

    Features:
    - Unified interface for secrets, keys, and certificates
    - Access policies and RBAC
    - Managed identities
    - Monitoring and auditing
    - Soft delete and purge protection
    """

    def __init__(
        self,
        vault_name: str,
        vault_url: str,
        tenant_id: str,
        subscription_id: str
    ):
        """
        Initialize Key Vault manager

        Args:
            vault_name: Key Vault name
            vault_url: Key Vault URL
            tenant_id: Azure AD tenant ID
            subscription_id: Azure subscription ID
        """
        self.vault_name = vault_name
        self.vault_url = vault_url
        self.tenant_id = tenant_id
        self.subscription_id = subscription_id

        self.secret_manager = SecretManager(vault_url)
        self.key_manager = KeyManager(vault_url)
        self.certificate_manager = CertificateManager(vault_url)

        self.access_policies: List[AccessPolicy] = []
        self.audit_logs: List[Dict[str, Any]] = []

    def create_access_policy(
        self,
        object_id: str,
        secret_permissions: Optional[List[str]] = None,
        key_permissions: Optional[List[str]] = None,
        certificate_permissions: Optional[List[str]] = None,
        application_id: Optional[str] = None
    ) -> AccessPolicy:
        """
        Create access policy for a principal

        Args:
            object_id: User/service principal object ID
            secret_permissions: Secret permissions
            key_permissions: Key permissions
            certificate_permissions: Certificate permissions
            application_id: Application ID (optional)

        Returns:
            AccessPolicy object
        """
        permissions = {}

        if secret_permissions:
            permissions["secrets"] = secret_permissions
        if key_permissions:
            permissions["keys"] = key_permissions
        if certificate_permissions:
            permissions["certificates"] = certificate_permissions

        policy = AccessPolicy(
            tenant_id=self.tenant_id,
            object_id=object_id,
            application_id=application_id,
            permissions=permissions
        )

        self.access_policies.append(policy)

        return policy

    def list_access_policies(self) -> List[Dict[str, Any]]:
        """
        List all access policies

        Returns:
            List of access policies
        """
        return [asdict(policy) for policy in self.access_policies]

    def enable_soft_delete(
        self,
        retention_days: int = 90
    ) -> Dict[str, Any]:
        """
        Enable soft delete protection

        Args:
            retention_days: Retention period for deleted items

        Returns:
            Configuration status
        """
        return {
            "vault_name": self.vault_name,
            "soft_delete_enabled": True,
            "retention_days": retention_days,
            "enabled_at": datetime.now().isoformat()
        }

    def enable_purge_protection(self) -> Dict[str, Any]:
        """
        Enable purge protection (cannot be disabled once enabled)

        Returns:
            Configuration status
        """
        return {
            "vault_name": self.vault_name,
            "purge_protection_enabled": True,
            "enabled_at": datetime.now().isoformat(),
            "note": "Purge protection cannot be disabled once enabled"
        }

    def configure_network_rules(
        self,
        default_action: str = "Deny",
        allowed_ip_ranges: Optional[List[str]] = None,
        virtual_networks: Optional[List[str]] = None,
        bypass: str = "AzureServices"
    ) -> Dict[str, Any]:
        """
        Configure network access rules

        Args:
            default_action: Default action (Allow/Deny)
            allowed_ip_ranges: Allowed IP address ranges
            virtual_networks: Allowed virtual network subnets
            bypass: Services that can bypass firewall

        Returns:
            Network configuration
        """
        return {
            "vault_name": self.vault_name,
            "default_action": default_action,
            "ip_rules": allowed_ip_ranges or [],
            "virtual_network_rules": virtual_networks or [],
            "bypass": bypass,
            "configured_at": datetime.now().isoformat()
        }

    def get_vault_properties(self) -> Dict[str, Any]:
        """
        Get Key Vault properties

        Returns:
            Vault properties
        """
        return {
            "vault_name": self.vault_name,
            "vault_url": self.vault_url,
            "tenant_id": self.tenant_id,
            "subscription_id": self.subscription_id,
            "secrets_count": len(self.secret_manager.secrets),
            "keys_count": len(self.key_manager.keys),
            "certificates_count": len(self.certificate_manager.certificates),
            "access_policies_count": len(self.access_policies),
            "sku": "standard",
            "location": "eastus"
        }

    def audit_log_entry(
        self,
        operation: str,
        resource_type: str,
        resource_name: str,
        principal_id: str,
        result: str
    ):
        """
        Add audit log entry

        Args:
            operation: Operation performed
            resource_type: Resource type (secret/key/certificate)
            resource_name: Resource name
            principal_id: Principal performing operation
            result: Operation result (success/failure)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "resource_type": resource_type,
            "resource_name": resource_name,
            "principal_id": principal_id,
            "result": result,
            "vault_name": self.vault_name
        }

        self.audit_logs.append(log_entry)

    def get_audit_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit logs

        Args:
            start_time: Start time filter
            end_time: End time filter
            operation: Operation filter

        Returns:
            List of audit log entries
        """
        logs = self.audit_logs.copy()

        if start_time:
            logs = [
                log for log in logs
                if datetime.fromisoformat(log["timestamp"]) >= start_time
            ]

        if end_time:
            logs = [
                log for log in logs
                if datetime.fromisoformat(log["timestamp"]) <= end_time
            ]

        if operation:
            logs = [log for log in logs if log["operation"] == operation]

        return logs


def demo_secret_operations():
    """Demonstrate secret management operations"""
    print("=== Secret Operations Demo ===\n")

    vault_manager = KeyVaultManager(
        vault_name="my-keyvault",
        vault_url="https://my-keyvault.vault.azure.net/",
        tenant_id="tenant-id",
        subscription_id="subscription-id"
    )

    secret_mgr = vault_manager.secret_manager

    # Create secrets
    print("Creating secrets...")
    db_secret = secret_mgr.set_secret(
        "database-connection",
        "Server=myserver;Database=mydb;User=admin;Password=secret123",
        content_type=SecretContentType.CONNECTION_STRING,
        tags={"environment": "production", "app": "api"}
    )
    print(f"Created secret: {db_secret.name} (version: {db_secret.version})")

    api_key = secret_mgr.set_secret(
        "api-key",
        "sk-1234567890abcdef",
        content_type=SecretContentType.API_KEY,
        expires_on=datetime.now() + timedelta(days=365)
    )
    print(f"Created secret: {api_key.name} (expires: {api_key.expires_on})\n")

    # List secrets
    print("Listing all secrets:")
    for secret in secret_mgr.list_secrets():
        print(f"  - {secret['name']} (created: {secret['created_on']})")
    print()

    # Get secret
    retrieved = secret_mgr.get_secret("database-connection")
    print(f"Retrieved secret: {retrieved.name}")
    print(f"Value: {retrieved.value[:30]}...")
    print(f"Content type: {retrieved.content_type}\n")

    # Update secret (rotation)
    print("Rotating secret...")
    new_version = secret_mgr.set_secret(
        "database-connection",
        "Server=myserver;Database=mydb;User=admin;Password=newsecret456",
        content_type=SecretContentType.CONNECTION_STRING
    )
    print(f"New version created: {new_version.version}\n")

    # List versions
    print("Secret versions:")
    versions = secret_mgr.list_secret_versions("database-connection")
    for v in versions:
        print(f"  - {v['version']} (created: {v['created_on']})")
    print()

    # Backup and restore
    print("Backing up secret...")
    backup = secret_mgr.backup_secret("api-key")
    print(f"Backup size: {len(backup)} bytes")

    # Delete and restore
    print("\nDeleting secret...")
    delete_result = secret_mgr.delete_secret("api-key")
    print(f"Deleted: {delete_result['name']}")
    print(f"Scheduled purge: {delete_result['scheduled_purge_date']}")

    print("\nRecovering secret...")
    recovered = secret_mgr.recover_deleted_secret("api-key")
    print(f"Recovered: {recovered.name}\n")


def demo_key_operations():
    """Demonstrate key management operations"""
    print("=== Key Operations Demo ===\n")

    vault_manager = KeyVaultManager(
        vault_name="my-keyvault",
        vault_url="https://my-keyvault.vault.azure.net/",
        tenant_id="tenant-id",
        subscription_id="subscription-id"
    )

    key_mgr = vault_manager.key_manager

    # Create keys
    print("Creating keys...")
    rsa_key = key_mgr.create_key(
        "encryption-key",
        KeyType.RSA,
        key_size=2048,
        tags={"purpose": "data-encryption"}
    )
    print(f"Created RSA key: {rsa_key.name}")
    print(f"Key size: {rsa_key.key_size} bits")
    print(f"Operations: {', '.join(rsa_key.operations)}\n")

    # Encrypt and decrypt
    print("Encrypting data...")
    plaintext = "Sensitive data that needs encryption"
    encrypted = key_mgr.encrypt("encryption-key", plaintext)
    print(f"Ciphertext: {encrypted['ciphertext'][:50]}...")

    print("\nDecrypting data...")
    decrypted = key_mgr.decrypt("encryption-key", encrypted['ciphertext'])
    print(f"Plaintext: {decrypted['plaintext']}\n")

    # Sign and verify
    print("Signing data...")
    digest = "SHA256:1234567890abcdef"
    signature = key_mgr.sign("encryption-key", digest)
    print(f"Signature: {signature['signature'][:50]}...")

    print("\nVerifying signature...")
    verification = key_mgr.verify("encryption-key", digest, signature['signature'])
    print(f"Valid: {verification['is_valid']}\n")

    # Key rotation
    print("Rotating key...")
    new_key = key_mgr.rotate_key("encryption-key")
    print(f"New key version: {new_key.version}")

    print("\nKey versions:")
    for key_name, versions in key_mgr.keys.items():
        if key_name == "encryption-key":
            for version in versions:
                print(f"  - {version}")
    print()


def demo_certificate_operations():
    """Demonstrate certificate management operations"""
    print("=== Certificate Operations Demo ===\n")

    vault_manager = KeyVaultManager(
        vault_name="my-keyvault",
        vault_url="https://my-keyvault.vault.azure.net/",
        tenant_id="tenant-id",
        subscription_id="subscription-id"
    )

    cert_mgr = vault_manager.certificate_manager

    # Create certificate
    print("Creating certificate...")
    cert = cert_mgr.create_certificate(
        "ssl-certificate",
        subject="CN=example.com",
        validity_months=12,
        issuer="DigiCert",
        key_size=2048,
        tags={"domain": "example.com", "type": "ssl"}
    )
    print(f"Created certificate: {cert.name}")
    print(f"Subject: {cert.subject}")
    print(f"Issuer: {cert.issuer}")
    print(f"Thumbprint: {cert.thumbprint}")
    print(f"Expires: {cert.expires_on}\n")

    # List certificates
    print("Listing certificates:")
    for c in cert_mgr.list_certificates():
        print(f"  - {c['name']} (expires: {c['expires_on']})")
    print()

    # Update policy
    print("Updating certificate policy...")
    policy = cert_mgr.update_certificate_policy(
        "ssl-certificate",
        auto_renew=True,
        renew_days_before_expiry=30
    )
    print(f"Auto-renew: {policy['auto_renew']}")
    print(f"Renew days before expiry: {policy['renew_days_before_expiry']}\n")


def demo_access_policies():
    """Demonstrate access policy management"""
    print("=== Access Policies Demo ===\n")

    vault_manager = KeyVaultManager(
        vault_name="my-keyvault",
        vault_url="https://my-keyvault.vault.azure.net/",
        tenant_id="tenant-id",
        subscription_id="subscription-id"
    )

    # Create access policies
    print("Creating access policies...")

    policy1 = vault_manager.create_access_policy(
        object_id="user-object-id-1",
        secret_permissions=["get", "list", "set"],
        key_permissions=["get", "list", "create"],
        certificate_permissions=["get", "list"]
    )
    print(f"Created policy for: {policy1.object_id}")

    policy2 = vault_manager.create_access_policy(
        object_id="sp-object-id-1",
        secret_permissions=["get"],
        application_id="app-id-1"
    )
    print(f"Created policy for service principal: {policy2.object_id}\n")

    # List policies
    print("Access policies:")
    for policy in vault_manager.list_access_policies():
        print(f"  Object ID: {policy['object_id']}")
        print(f"  Permissions: {policy['permissions']}\n")


def demo_security_features():
    """Demonstrate security features"""
    print("=== Security Features Demo ===\n")

    vault_manager = KeyVaultManager(
        vault_name="my-keyvault",
        vault_url="https://my-keyvault.vault.azure.net/",
        tenant_id="tenant-id",
        subscription_id="subscription-id"
    )

    # Enable soft delete
    print("Enabling soft delete...")
    soft_delete = vault_manager.enable_soft_delete(retention_days=90)
    print(f"Soft delete enabled: {soft_delete['soft_delete_enabled']}")
    print(f"Retention: {soft_delete['retention_days']} days\n")

    # Enable purge protection
    print("Enabling purge protection...")
    purge_protection = vault_manager.enable_purge_protection()
    print(f"Purge protection enabled: {purge_protection['purge_protection_enabled']}\n")

    # Configure network rules
    print("Configuring network rules...")
    network = vault_manager.configure_network_rules(
        default_action="Deny",
        allowed_ip_ranges=["203.0.113.0/24", "198.51.100.0/24"],
        bypass="AzureServices"
    )
    print(f"Default action: {network['default_action']}")
    print(f"Allowed IPs: {', '.join(network['ip_rules'])}")
    print(f"Bypass: {network['bypass']}\n")

    # Audit logging
    print("Creating audit log entries...")
    vault_manager.audit_log_entry(
        operation="SecretGet",
        resource_type="secret",
        resource_name="database-connection",
        principal_id="user@example.com",
        result="success"
    )
    vault_manager.audit_log_entry(
        operation="KeyCreate",
        resource_type="key",
        resource_name="encryption-key",
        principal_id="sp@example.com",
        result="success"
    )

    print("Recent audit logs:")
    for log in vault_manager.get_audit_logs()[-5:]:
        print(f"  [{log['timestamp']}] {log['operation']} - {log['resource_name']} by {log['principal_id']}")
    print()


def demo_vault_management():
    """Demonstrate vault management"""
    print("=== Vault Management Demo ===\n")

    vault_manager = KeyVaultManager(
        vault_name="my-keyvault",
        vault_url="https://my-keyvault.vault.azure.net/",
        tenant_id="tenant-id",
        subscription_id="subscription-id"
    )

    # Add some test data
    vault_manager.secret_manager.set_secret("test-secret-1", "value1")
    vault_manager.secret_manager.set_secret("test-secret-2", "value2")
    vault_manager.key_manager.create_key("test-key", KeyType.RSA)
    vault_manager.certificate_manager.create_certificate("test-cert", "CN=test.com")

    # Get vault properties
    print("Vault properties:")
    props = vault_manager.get_vault_properties()
    print(f"  Name: {props['vault_name']}")
    print(f"  URL: {props['vault_url']}")
    print(f"  Secrets: {props['secrets_count']}")
    print(f"  Keys: {props['keys_count']}")
    print(f"  Certificates: {props['certificates_count']}")
    print(f"  Access policies: {props['access_policies_count']}")
    print(f"  SKU: {props['sku']}")
    print(f"  Location: {props['location']}\n")


if __name__ == "__main__":
    print("Azure Key Vault - Advanced Implementation")
    print("=" * 60)
    print()

    # Run all demos
    demo_secret_operations()
    demo_key_operations()
    demo_certificate_operations()
    demo_access_policies()
    demo_security_features()
    demo_vault_management()

    print("=" * 60)
    print("All demos completed successfully!")
