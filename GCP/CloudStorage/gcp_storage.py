"""
Google Cloud Storage - Advanced Object Storage
Author: Brill Consulting
Description: Comprehensive cloud storage with lifecycle, CORS, IAM, versioning, and signed URLs
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import base64


class BucketManager:
    """Manage storage buckets"""

    def __init__(self, project_id: str):
        """Initialize bucket manager"""
        self.project_id = project_id
        self.buckets = {}

    def create_bucket(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create storage bucket"""
        print(f"\n{'='*60}")
        print("Creating Storage Bucket")
        print(f"{'='*60}")

        name = config.get('name', 'my-bucket')
        location = config.get('location', 'US')
        storage_class = config.get('storage_class', 'STANDARD')

        code = f"""
from google.cloud import storage

storage_client = storage.Client()

# Create bucket
bucket = storage_client.bucket('{name}')
bucket.storage_class = '{storage_class}'

new_bucket = storage_client.create_bucket(
    bucket,
    location='{location}'
)

print(f"Bucket {{new_bucket.name}} created in {{new_bucket.location}}")
"""

        result = {
            'name': name,
            'location': location,
            'storage_class': storage_class,
            'versioning': False,
            'lifecycle_rules': [],
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.buckets[name] = result

        print(f"✓ Bucket created: gs://{name}")
        print(f"  Location: {location}")
        print(f"  Storage class: {storage_class}")
        print(f"{'='*60}")

        return result

    def enable_versioning(self, bucket_name: str) -> Dict[str, Any]:
        """Enable bucket versioning"""
        print(f"\n{'='*60}")
        print("Enabling Versioning")
        print(f"{'='*60}")

        code = f"""
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('{bucket_name}')

bucket.versioning_enabled = True
bucket.patch()

print(f"Versioning enabled for {{bucket.name}}")
"""

        result = {
            'bucket': bucket_name,
            'versioning_enabled': True,
            'code': code
        }

        if bucket_name in self.buckets:
            self.buckets[bucket_name]['versioning'] = True

        print(f"✓ Versioning enabled: {bucket_name}")
        print(f"{'='*60}")

        return result

    def set_cors(self, bucket_name: str, cors_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Configure CORS for bucket"""
        print(f"\n{'='*60}")
        print("Configuring CORS")
        print(f"{'='*60}")

        code = f"""
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('{bucket_name}')

# CORS configuration
cors_configuration = {cors_config}

bucket.cors = cors_configuration
bucket.patch()

print(f"CORS configured for {{bucket.name}}")
"""

        result = {
            'bucket': bucket_name,
            'cors_rules': len(cors_config),
            'code': code
        }

        print(f"✓ CORS configured: {bucket_name}")
        print(f"  Rules: {len(cors_config)}")
        print(f"{'='*60}")

        return result


class ObjectManager:
    """Manage storage objects"""

    def __init__(self, project_id: str):
        """Initialize object manager"""
        self.project_id = project_id

    def upload_object(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Upload object to bucket"""
        print(f"\n{'='*60}")
        print("Uploading Object")
        print(f"{'='*60}")

        bucket_name = config.get('bucket_name', 'my-bucket')
        object_name = config.get('object_name', 'file.txt')
        source_file = config.get('source_file', '/path/to/file')

        code = f"""
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('{bucket_name}')
blob = bucket.blob('{object_name}')

# Upload from file
blob.upload_from_filename('{source_file}')

# Or upload from string
# blob.upload_from_string('data content')

print(f"File uploaded to gs://{bucket_name}/{object_name}")
"""

        result = {
            'bucket': bucket_name,
            'object': object_name,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Object uploaded: gs://{bucket_name}/{object_name}")
        print(f"{'='*60}")

        return result

    def generate_signed_url(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signed URL for object access"""
        print(f"\n{'='*60}")
        print("Generating Signed URL")
        print(f"{'='*60}")

        bucket_name = config.get('bucket_name', 'my-bucket')
        object_name = config.get('object_name', 'file.txt')
        expiration_minutes = config.get('expiration_minutes', 15)

        code = f"""
from google.cloud import storage
from datetime import timedelta

storage_client = storage.Client()
bucket = storage_client.bucket('{bucket_name}')
blob = bucket.blob('{object_name}')

# Generate signed URL
url = blob.generate_signed_url(
    version="v4",
    expiration=timedelta(minutes={expiration_minutes}),
    method="GET"
)

print(f"Signed URL generated (expires in {expiration_minutes} minutes)")
print(f"URL: {{url}}")
"""

        expiration = datetime.now() + timedelta(minutes=expiration_minutes)

        result = {
            'bucket': bucket_name,
            'object': object_name,
            'expiration': expiration.isoformat(),
            'code': code
        }

        print(f"✓ Signed URL generated: {object_name}")
        print(f"  Expires: {expiration.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        return result

    def set_object_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set custom metadata for object"""
        print(f"\n{'='*60}")
        print("Setting Object Metadata")
        print(f"{'='*60}")

        bucket_name = config.get('bucket_name', 'my-bucket')
        object_name = config.get('object_name', 'file.txt')
        metadata = config.get('metadata', {})

        code = f"""
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('{bucket_name}')
blob = bucket.blob('{object_name}')

# Set custom metadata
blob.metadata = {metadata}
blob.patch()

print(f"Metadata set for {{blob.name}}")
"""

        result = {
            'bucket': bucket_name,
            'object': object_name,
            'metadata': metadata,
            'code': code
        }

        print(f"✓ Metadata set: {object_name}")
        print(f"  Keys: {', '.join(metadata.keys())}")
        print(f"{'='*60}")

        return result


class LifecycleManager:
    """Manage bucket lifecycle policies"""

    def __init__(self, project_id: str):
        """Initialize lifecycle manager"""
        self.project_id = project_id

    def set_lifecycle_policy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set lifecycle policy for bucket"""
        print(f"\n{'='*60}")
        print("Setting Lifecycle Policy")
        print(f"{'='*60}")

        bucket_name = config.get('bucket_name', 'my-bucket')
        rules = config.get('rules', [])

        code = f"""
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('{bucket_name}')

# Define lifecycle rules
rules = {json.dumps(rules, indent=4)}

bucket.lifecycle_rules = rules
bucket.patch()

print(f"Lifecycle policy set for {{bucket.name}}")
print(f"Rules: {{len(rules)}}")
"""

        result = {
            'bucket': bucket_name,
            'rules': rules,
            'rules_count': len(rules),
            'code': code
        }

        print(f"✓ Lifecycle policy set: {bucket_name}")
        print(f"  Rules: {len(rules)}")
        for i, rule in enumerate(rules, 1):
            action = rule.get('action', {}).get('type', 'Unknown')
            print(f"    {i}. {action}")
        print(f"{'='*60}")

        return result


class IAMManager:
    """Manage IAM policies for buckets"""

    def __init__(self, project_id: str):
        """Initialize IAM manager"""
        self.project_id = project_id

    def make_bucket_public(self, bucket_name: str) -> Dict[str, Any]:
        """Make bucket publicly readable"""
        print(f"\n{'='*60}")
        print("Making Bucket Public")
        print(f"{'='*60}")

        code = f"""
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('{bucket_name}')

# Make public
policy = bucket.get_iam_policy(requested_policy_version=3)
policy.bindings.append(
    {{
        "role": "roles/storage.objectViewer",
        "members": {{"allUsers"}},
    }}
)

bucket.set_iam_policy(policy)

print(f"Bucket {{bucket.name}} is now public")
"""

        result = {
            'bucket': bucket_name,
            'public': True,
            'code': code
        }

        print(f"✓ Bucket made public: {bucket_name}")
        print(f"  Warning: All objects are publicly accessible")
        print(f"{'='*60}")

        return result

    def grant_access(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Grant access to bucket"""
        print(f"\n{'='*60}")
        print("Granting Bucket Access")
        print(f"{'='*60}")

        bucket_name = config.get('bucket_name', 'my-bucket')
        member = config.get('member', 'user:user@example.com')
        role = config.get('role', 'roles/storage.objectViewer')

        code = f"""
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('{bucket_name}')

# Grant access
policy = bucket.get_iam_policy(requested_policy_version=3)
policy.bindings.append(
    {{
        "role": "{role}",
        "members": {{"{member}"}},
    }}
)

bucket.set_iam_policy(policy)

print(f"Access granted to {{'{member}'}}")
"""

        result = {
            'bucket': bucket_name,
            'member': member,
            'role': role,
            'code': code
        }

        print(f"✓ Access granted: {bucket_name}")
        print(f"  Member: {member}")
        print(f"  Role: {role}")
        print(f"{'='*60}")

        return result


class NotificationManager:
    """Manage Pub/Sub notifications"""

    def __init__(self, project_id: str):
        """Initialize notification manager"""
        self.project_id = project_id

    def create_notification(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Pub/Sub notification for bucket"""
        print(f"\n{'='*60}")
        print("Creating Pub/Sub Notification")
        print(f"{'='*60}")

        bucket_name = config.get('bucket_name', 'my-bucket')
        topic_name = config.get('topic_name', 'projects/project/topics/storage-events')

        code = f"""
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('{bucket_name}')

# Create notification
notification = bucket.notification(
    topic_name='{topic_name}',
    custom_attributes={{'source': 'cloud-storage'}}
)

notification.event_types = [
    'OBJECT_FINALIZE',
    'OBJECT_DELETE',
]

notification.create()

print(f"Notification created: {{notification.notification_id}}")
"""

        result = {
            'bucket': bucket_name,
            'topic': topic_name,
            'events': ['OBJECT_FINALIZE', 'OBJECT_DELETE'],
            'code': code
        }

        print(f"✓ Notification created: {bucket_name}")
        print(f"  Topic: {topic_name}")
        print(f"  Events: OBJECT_FINALIZE, OBJECT_DELETE")
        print(f"{'='*60}")

        return result


class TransferManager:
    """Manage storage transfer operations"""

    def __init__(self, project_id: str):
        """Initialize transfer manager"""
        self.project_id = project_id

    def parallel_upload(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Upload multiple files in parallel"""
        print(f"\n{'='*60}")
        print("Parallel Upload")
        print(f"{'='*60}")

        bucket_name = config.get('bucket_name', 'my-bucket')
        files = config.get('files', [])

        code = f"""
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor

storage_client = storage.Client()
bucket = storage_client.bucket('{bucket_name}')

files_to_upload = {files}

def upload_file(file_path):
    blob = bucket.blob(file_path.split('/')[-1])
    blob.upload_from_filename(file_path)
    return file_path

# Parallel upload
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(upload_file, files_to_upload))

print(f"Uploaded {{len(results)}} files to {{bucket.name}}")
"""

        result = {
            'bucket': bucket_name,
            'files': len(files),
            'code': code
        }

        print(f"✓ Parallel upload configured: {len(files)} files")
        print(f"  Bucket: {bucket_name}")
        print(f"{'='*60}")

        return result


class CloudStorageManager:
    """Comprehensive Cloud Storage management"""

    def __init__(self, project_id: str = 'my-project'):
        """
        Initialize Cloud Storage manager

        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.buckets = BucketManager(project_id)
        self.objects = ObjectManager(project_id)
        self.lifecycle = LifecycleManager(project_id)
        self.iam = IAMManager(project_id)
        self.notifications = NotificationManager(project_id)
        self.transfer = TransferManager(project_id)

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'buckets': len(self.buckets.buckets),
            'features': [
                'bucket_management',
                'object_operations',
                'lifecycle_policies',
                'versioning',
                'signed_urls',
                'cors_configuration',
                'iam_policies',
                'pubsub_notifications',
                'parallel_transfers'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Cloud Storage capabilities"""
    print("=" * 60)
    print("Cloud Storage Comprehensive Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'

    # Initialize manager
    mgr = CloudStorageManager(project_id)

    # Create bucket
    bucket = mgr.buckets.create_bucket({
        'name': 'my-data-bucket',
        'location': 'US',
        'storage_class': 'STANDARD'
    })

    # Enable versioning
    versioning = mgr.buckets.enable_versioning('my-data-bucket')

    # Configure CORS
    cors = mgr.buckets.set_cors('my-data-bucket', [
        {
            'origin': ['https://example.com'],
            'method': ['GET', 'POST'],
            'responseHeader': ['Content-Type'],
            'maxAgeSeconds': 3600
        }
    ])

    # Upload object
    upload = mgr.objects.upload_object({
        'bucket_name': 'my-data-bucket',
        'object_name': 'data/file.txt',
        'source_file': '/tmp/file.txt'
    })

    # Generate signed URL
    signed_url = mgr.objects.generate_signed_url({
        'bucket_name': 'my-data-bucket',
        'object_name': 'data/file.txt',
        'expiration_minutes': 15
    })

    # Set lifecycle policy
    lifecycle = mgr.lifecycle.set_lifecycle_policy({
        'bucket_name': 'my-data-bucket',
        'rules': [
            {
                'action': {'type': 'Delete'},
                'condition': {'age': 30}
            },
            {
                'action': {'type': 'SetStorageClass', 'storageClass': 'NEARLINE'},
                'condition': {'age': 90}
            }
        ]
    })

    # Grant access
    access = mgr.iam.grant_access({
        'bucket_name': 'my-data-bucket',
        'member': 'user:user@example.com',
        'role': 'roles/storage.objectViewer'
    })

    # Create notification
    notification = mgr.notifications.create_notification({
        'bucket_name': 'my-data-bucket',
        'topic_name': 'projects/my-project/topics/storage-events'
    })

    # Manager info
    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Cloud Storage Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Buckets: {info['buckets']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")
    print("\nCloud Storage Best Practices:")
    print("  1. Enable versioning for important data")
    print("  2. Set lifecycle policies to reduce costs")
    print("  3. Use signed URLs for temporary access")
    print("  4. Configure CORS for web applications")
    print("  5. Use IAM for fine-grained access control")
    print("  6. Enable notifications for event-driven workflows")


if __name__ == "__main__":
    demo()
