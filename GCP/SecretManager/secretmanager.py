"""
Secret Manager - Secure Secret Storage and Management
Author: BrillConsulting
Description: Comprehensive secret management with versioning, rotation, and IAM
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import base64
import hashlib


class SecretCreation:
    """Create and manage secrets"""

    def __init__(self, project_id: str):
        """
        Initialize secret creation

        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.secrets = {}

    def create_secret(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new secret

        Args:
            config: Secret configuration

        Returns:
            Secret creation result
        """
        print(f"\n{'='*60}")
        print("Creating Secret")
        print(f"{'='*60}")

        secret_id = config.get('secret_id', 'my-secret')
        replication = config.get('replication', 'automatic')
        labels = config.get('labels', {})

        code = f"""
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()

# Define secret
parent = f"projects/{self.project_id}"
secret_id = "{secret_id}"

# Create secret
secret = {{
    'replication': {{
        '{replication}': {{}},
    }},
    'labels': {labels}
}}

response = client.create_secret(
    request={{
        "parent": parent,
        "secret_id": secret_id,
        "secret": secret,
    }}
)

print(f"Created secret: {{response.name}}")
"""

        result = {
            'secret_id': secret_id,
            'project_id': self.project_id,
            'name': f"projects/{self.project_id}/secrets/{secret_id}",
            'replication': replication,
            'labels': labels,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.secrets[secret_id] = result

        print(f"✓ Secret created: {secret_id}")
        print(f"  Replication: {replication}")
        print(f"  Resource name: {result['name']}")
        print(f"{'='*60}")

        return result

    def add_secret_version(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new version to existing secret

        Args:
            config: Version configuration

        Returns:
            Version creation result
        """
        print(f"\n{'='*60}")
        print("Adding Secret Version")
        print(f"{'='*60}")

        secret_id = config.get('secret_id', 'my-secret')
        payload = config.get('payload', 'secret-value')

        code = f"""
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()

# Add secret version
parent = client.secret_path("{self.project_id}", "{secret_id}")

# Payload as bytes
payload_bytes = "{payload}".encode('UTF-8')

# Add the secret version
version = client.add_secret_version(
    request={{
        "parent": parent,
        "payload": {{"data": payload_bytes}},
    }}
)

print(f"Added secret version: {{version.name}}")
"""

        # Hash payload for tracking (not storing actual secret)
        payload_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]

        result = {
            'secret_id': secret_id,
            'version_name': f"projects/{self.project_id}/secrets/{secret_id}/versions/1",
            'payload_hash': payload_hash,
            'state': 'ENABLED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Secret version added: {secret_id}")
        print(f"  Version: {result['version_name']}")
        print(f"  State: {result['state']}")
        print(f"{'='*60}")

        return result


class SecretAccess:
    """Access and retrieve secret values"""

    def __init__(self, project_id: str):
        """Initialize secret access"""
        self.project_id = project_id

    def access_secret_version(self, secret_id: str, version: str = 'latest') -> str:
        """
        Access a secret version

        Args:
            secret_id: Secret ID
            version: Version ID or 'latest'

        Returns:
            Code to access secret
        """
        code = f"""
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()

# Build the resource name
name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"

# Access the secret version
response = client.access_secret_version(request={{"name": name}})

# Decode the secret payload
payload = response.payload.data.decode('UTF-8')
print(f"Secret value retrieved ({{len(payload)}} chars)")

# Use the secret
# payload contains the actual secret value
"""

        print(f"\n✓ Secret access code generated for: {secret_id}")
        print(f"  Version: {version}")
        return code

    def list_secret_versions(self, secret_id: str) -> str:
        """
        List all versions of a secret

        Args:
            secret_id: Secret ID

        Returns:
            Code to list versions
        """
        code = f"""
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()

# List all secret versions
parent = client.secret_path("{self.project_id}", "{secret_id}")

versions = []
for version in client.list_secret_versions(request={{"parent": parent}}):
    versions.append({{
        'name': version.name,
        'state': secretmanager.SecretVersion.State(version.state).name,
        'created': version.create_time,
    }})
    print(f"Version: {{version.name}} - State: {{version.state}}")

print(f"Total versions: {{len(versions)}}")
"""

        print(f"\n✓ Version listing code generated for: {secret_id}")
        return code


class SecretVersioning:
    """Manage secret versions and lifecycle"""

    def __init__(self, project_id: str):
        """Initialize version management"""
        self.project_id = project_id

    def disable_secret_version(self, secret_id: str, version_id: str) -> Dict[str, Any]:
        """
        Disable a secret version

        Args:
            secret_id: Secret ID
            version_id: Version ID

        Returns:
            Disable operation result
        """
        print(f"\n{'='*60}")
        print("Disabling Secret Version")
        print(f"{'='*60}")

        code = f"""
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()

# Build the resource name
name = client.secret_version_path(
    "{self.project_id}",
    "{secret_id}",
    "{version_id}"
)

# Disable the secret version
response = client.disable_secret_version(request={{"name": name}})
print(f"Disabled secret version: {{response.name}}")
"""

        result = {
            'secret_id': secret_id,
            'version_id': version_id,
            'state': 'DISABLED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Version disabled: {secret_id}/versions/{version_id}")
        print(f"{'='*60}")

        return result

    def destroy_secret_version(self, secret_id: str, version_id: str) -> Dict[str, Any]:
        """
        Permanently destroy a secret version

        Args:
            secret_id: Secret ID
            version_id: Version ID

        Returns:
            Destroy operation result
        """
        print(f"\n{'='*60}")
        print("Destroying Secret Version")
        print(f"{'='*60}")

        code = f"""
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()

# Build the resource name
name = client.secret_version_path(
    "{self.project_id}",
    "{secret_id}",
    "{version_id}"
)

# Destroy the secret version (irreversible!)
response = client.destroy_secret_version(request={{"name": name}})
print(f"Destroyed secret version: {{response.name}}")
print("This operation is irreversible!")
"""

        result = {
            'secret_id': secret_id,
            'version_id': version_id,
            'state': 'DESTROYED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Version destroyed: {secret_id}/versions/{version_id}")
        print(f"  ⚠️  This operation is irreversible!")
        print(f"{'='*60}")

        return result


class SecretRotation:
    """Automatic secret rotation"""

    def __init__(self, project_id: str):
        """Initialize rotation management"""
        self.project_id = project_id
        self.rotation_schedules = []

    def configure_rotation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure automatic secret rotation

        Args:
            config: Rotation configuration

        Returns:
            Rotation configuration result
        """
        print(f"\n{'='*60}")
        print("Configuring Secret Rotation")
        print(f"{'='*60}")

        secret_id = config.get('secret_id', 'database-password')
        rotation_period_days = config.get('rotation_period_days', 30)
        function_name = config.get('rotation_function', 'rotate-secret')

        code = f"""
from google.cloud import secretmanager
from google.cloud import scheduler_v1
from google.cloud import functions_v1

# Create rotation function (Cloud Function)
# This function will be triggered by Cloud Scheduler

def rotate_secret(request):
    '''Rotate secret by creating new version'''
    import os
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    project_id = os.environ['PROJECT_ID']
    secret_id = os.environ['SECRET_ID']

    # Generate new secret value (example: database password)
    import secrets
    import string
    alphabet = string.ascii_letters + string.digits
    new_password = ''.join(secrets.choice(alphabet) for i in range(32))

    # Add new secret version
    parent = client.secret_path(project_id, secret_id)
    payload_bytes = new_password.encode('UTF-8')

    version = client.add_secret_version(
        request={{
            "parent": parent,
            "payload": {{"data": payload_bytes}},
        }}
    )

    # Update database with new password
    # update_database_password(new_password)

    return f"Rotated: {{version.name}}"

# Configure Cloud Scheduler to trigger rotation
scheduler_client = scheduler_v1.CloudSchedulerClient()

parent = f"projects/{self.project_id}/locations/us-central1"

job = {{
    "name": f"{{parent}}/jobs/{secret_id}-rotation",
    "http_target": {{
        "uri": f"https://us-central1-{self.project_id}.cloudfunctions.net/{function_name}",
        "http_method": scheduler_v1.HttpMethod.POST,
    }},
    "schedule": "0 0 */{rotation_period_days} * *",  # Every {rotation_period_days} days
    "time_zone": "UTC",
}}

scheduler_client.create_job(parent=parent, job=job)
print(f"Rotation configured: every {rotation_period_days} days")
"""

        result = {
            'secret_id': secret_id,
            'rotation_period_days': rotation_period_days,
            'rotation_function': function_name,
            'schedule': f"0 0 */{rotation_period_days} * *",
            'next_rotation': (datetime.now() + timedelta(days=rotation_period_days)).isoformat(),
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.rotation_schedules.append(result)

        print(f"✓ Rotation configured: {secret_id}")
        print(f"  Period: Every {rotation_period_days} days")
        print(f"  Function: {function_name}")
        print(f"  Next rotation: {result['next_rotation'][:10]}")
        print(f"{'='*60}")

        return result


class SecretIAM:
    """Secret access control and IAM"""

    def __init__(self, project_id: str):
        """Initialize IAM management"""
        self.project_id = project_id

    def grant_secret_access(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grant access to a secret

        Args:
            config: IAM configuration

        Returns:
            IAM policy update result
        """
        print(f"\n{'='*60}")
        print("Granting Secret Access")
        print(f"{'='*60}")

        secret_id = config.get('secret_id', 'my-secret')
        member = config.get('member', 'serviceAccount:my-app@project.iam.gserviceaccount.com')
        role = config.get('role', 'roles/secretmanager.secretAccessor')

        code = f"""
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()

# Get the secret
name = client.secret_path("{self.project_id}", "{secret_id}")

# Get the current IAM policy
policy = client.get_iam_policy(request={{"resource": name}})

# Add the member to the policy
policy.bindings.add(
    role="{role}",
    members=["{member}"]
)

# Update the IAM policy
updated_policy = client.set_iam_policy(
    request={{"resource": name, "policy": policy}}
)

print(f"Granted {{'{role}'}} to {{'{member}'}}")
"""

        result = {
            'secret_id': secret_id,
            'member': member,
            'role': role,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Access granted to: {secret_id}")
        print(f"  Member: {member}")
        print(f"  Role: {role}")
        print(f"{'='*60}")

        return result

    def create_secret_accessor_role(self) -> str:
        """
        Create custom role for secret access

        Returns:
            Custom role creation code
        """
        code = f"""
from google.cloud import iam_admin_v1

client = iam_admin_v1.IAMClient()

# Define custom role
parent = f"projects/{self.project_id}"

role = {{
    "role_id": "customSecretAccessor",
    "role": {{
        "title": "Custom Secret Accessor",
        "description": "Custom role for accessing specific secrets",
        "included_permissions": [
            "secretmanager.secrets.get",
            "secretmanager.versions.access",
            "secretmanager.versions.list",
        ],
        "stage": "GA",
    }}
}}

created_role = client.create_role(parent=parent, **role)
print(f"Created custom role: {{created_role.name}}")
"""

        print("\n✓ Custom secret accessor role code generated")
        return code


class SecretManager:
    """Comprehensive secret management"""

    def __init__(self, project_id: str = 'my-project'):
        """
        Initialize Secret Manager

        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.creation = SecretCreation(project_id)
        self.access = SecretAccess(project_id)
        self.versioning = SecretVersioning(project_id)
        self.rotation = SecretRotation(project_id)
        self.iam = SecretIAM(project_id)

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'secrets': len(self.creation.secrets),
            'rotation_schedules': len(self.rotation.rotation_schedules),
            'features': [
                'secret_creation',
                'secret_versioning',
                'automatic_rotation',
                'iam_control',
                'audit_logging',
                'encryption_at_rest'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Secret Manager capabilities"""
    print("=" * 60)
    print("Secret Manager Comprehensive Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'

    # Initialize manager
    mgr = SecretManager(project_id)

    # Create secret
    secret_result = mgr.creation.create_secret({
        'secret_id': 'database-password',
        'replication': 'automatic',
        'labels': {'env': 'production', 'type': 'database'}
    })

    # Add secret version
    version_result = mgr.creation.add_secret_version({
        'secret_id': 'database-password',
        'payload': 'super-secure-password-123'
    })

    # Access secret
    access_code = mgr.access.access_secret_version('database-password', 'latest')

    # List versions
    list_code = mgr.access.list_secret_versions('database-password')

    # Configure rotation
    rotation_result = mgr.rotation.configure_rotation({
        'secret_id': 'database-password',
        'rotation_period_days': 30,
        'rotation_function': 'rotate-db-password'
    })

    # Grant access
    iam_result = mgr.iam.grant_secret_access({
        'secret_id': 'database-password',
        'member': 'serviceAccount:app@my-gcp-project.iam.gserviceaccount.com',
        'role': 'roles/secretmanager.secretAccessor'
    })

    # Disable old version
    disable_result = mgr.versioning.disable_secret_version('database-password', '1')

    # Manager info
    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Secret Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Secrets: {info['secrets']}")
    print(f"Rotation schedules: {info['rotation_schedules']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")
    print("\nSecret Manager Best Practices:")
    print("  1. Enable automatic rotation for sensitive secrets")
    print("  2. Use service accounts with least privilege")
    print("  3. Enable audit logging for all secret access")
    print("  4. Use labels for secret organization")
    print("  5. Destroy old versions after rotation")
    print("  6. Never log actual secret values")


if __name__ == "__main__":
    demo()
