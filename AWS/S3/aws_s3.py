"""
AWS S3 Management
=================

Comprehensive S3 bucket and object management with versioning, lifecycle policies,
presigned URLs, encryption, and advanced features.

Author: Brill Consulting
"""

import boto3
import logging
import json
import os
from typing import Dict, List, Optional, Any, BinaryIO
from datetime import datetime, timedelta
from botocore.exceptions import ClientError, NoCredentialsError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3Manager:
    """
    Advanced AWS S3 Management System

    Provides comprehensive S3 operations including:
    - Bucket management (create, list, delete)
    - Object operations (upload, download, delete, copy)
    - Versioning and lifecycle policies
    - Presigned URLs for temporary access
    - Encryption (SSE-S3, SSE-KMS)
    - Bucket policies and ACLs
    - Object tagging
    - Multipart upload for large files
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """
        Initialize S3 Manager.

        Args:
            region: AWS region (default: us-east-1)
            profile: AWS CLI profile name (optional)
        """
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.s3_client = session.client('s3', region_name=region)
            self.s3_resource = session.resource('s3', region_name=region)
            self.region = region
            logger.info(f"S3 Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing S3 Manager: {e}")
            raise

    # ==================== Bucket Operations ====================

    def create_bucket(
        self,
        bucket_name: str,
        region: Optional[str] = None,
        acl: str = "private"
    ) -> Dict[str, Any]:
        """
        Create S3 bucket.

        Args:
            bucket_name: Bucket name (must be globally unique)
            region: AWS region (uses manager's region if not specified)
            acl: Access control list (private, public-read, etc.)

        Returns:
            Bucket creation details
        """
        try:
            logger.info(f"Creating bucket: {bucket_name}")

            region = region or self.region
            params = {'Bucket': bucket_name, 'ACL': acl}

            # LocationConstraint not needed for us-east-1
            if region != 'us-east-1':
                params['CreateBucketConfiguration'] = {'LocationConstraint': region}

            self.s3_client.create_bucket(**params)

            logger.info(f"✓ Bucket created: s3://{bucket_name}")

            return {
                'bucket_name': bucket_name,
                'region': region,
                'acl': acl
            }

        except ClientError as e:
            logger.error(f"Error creating bucket: {e}")
            raise

    def list_buckets(self) -> List[Dict[str, Any]]:
        """List all S3 buckets."""
        try:
            response = self.s3_client.list_buckets()

            buckets = []
            for bucket in response.get('Buckets', []):
                buckets.append({
                    'name': bucket['Name'],
                    'creation_date': bucket['CreationDate'].isoformat()
                })

            logger.info(f"Found {len(buckets)} bucket(s)")
            return buckets

        except ClientError as e:
            logger.error(f"Error listing buckets: {e}")
            raise

    def delete_bucket(self, bucket_name: str, force: bool = False) -> None:
        """
        Delete S3 bucket.

        Args:
            bucket_name: Bucket to delete
            force: If True, delete all objects first
        """
        try:
            if force:
                # Delete all objects first
                bucket = self.s3_resource.Bucket(bucket_name)
                bucket.objects.all().delete()
                bucket.object_versions.all().delete()

            self.s3_client.delete_bucket(Bucket=bucket_name)
            logger.info(f"✓ Bucket deleted: {bucket_name}")

        except ClientError as e:
            logger.error(f"Error deleting bucket: {e}")
            raise

    # ==================== Object Operations ====================

    def upload_file(
        self,
        file_path: str,
        bucket_name: str,
        object_key: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload file to S3.

        Args:
            file_path: Local file path
            bucket_name: Destination bucket
            object_key: S3 object key (uses filename if not specified)
            metadata: Custom metadata
            tags: Object tags

        Returns:
            Upload details
        """
        try:
            object_key = object_key or os.path.basename(file_path)
            logger.info(f"Uploading {file_path} to s3://{bucket_name}/{object_key}")

            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            if tags:
                tag_string = '&'.join([f"{k}={v}" for k, v in tags.items()])
                extra_args['Tagging'] = tag_string

            self.s3_client.upload_file(file_path, bucket_name, object_key, ExtraArgs=extra_args or None)

            # Get file size
            file_size = os.path.getsize(file_path)

            logger.info(f"✓ File uploaded ({file_size} bytes)")

            return {
                'bucket': bucket_name,
                'key': object_key,
                'size': file_size,
                'url': f"s3://{bucket_name}/{object_key}"
            }

        except ClientError as e:
            logger.error(f"Error uploading file: {e}")
            raise

    def upload_object(
        self,
        bucket_name: str,
        object_key: str,
        data: bytes,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload data directly to S3.

        Args:
            bucket_name: Destination bucket
            object_key: S3 object key
            data: Binary data to upload
            content_type: Content type (e.g., 'application/json')

        Returns:
            Upload details
        """
        try:
            logger.info(f"Uploading object to s3://{bucket_name}/{object_key}")

            params = {'Bucket': bucket_name, 'Key': object_key, 'Body': data}
            if content_type:
                params['ContentType'] = content_type

            response = self.s3_client.put_object(**params)

            logger.info(f"✓ Object uploaded ({len(data)} bytes)")

            return {
                'bucket': bucket_name,
                'key': object_key,
                'etag': response['ETag'].strip('"'),
                'size': len(data)
            }

        except ClientError as e:
            logger.error(f"Error uploading object: {e}")
            raise

    def download_file(
        self,
        bucket_name: str,
        object_key: str,
        file_path: str
    ) -> Dict[str, Any]:
        """Download file from S3."""
        try:
            logger.info(f"Downloading s3://{bucket_name}/{object_key} to {file_path}")

            self.s3_client.download_file(bucket_name, object_key, file_path)

            file_size = os.path.getsize(file_path)
            logger.info(f"✓ File downloaded ({file_size} bytes)")

            return {'file_path': file_path, 'size': file_size}

        except ClientError as e:
            logger.error(f"Error downloading file: {e}")
            raise

    def get_object(self, bucket_name: str, object_key: str) -> bytes:
        """Get object content."""
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=object_key)
            content = response['Body'].read()
            logger.info(f"✓ Retrieved object: {object_key} ({len(content)} bytes)")
            return content

        except ClientError as e:
            logger.error(f"Error getting object: {e}")
            raise

    def list_objects(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List objects in bucket.

        Args:
            bucket_name: Bucket to list
            prefix: Filter by prefix
            max_keys: Maximum number of keys to return

        Returns:
            List of object details
        """
        try:
            params = {'Bucket': bucket_name, 'MaxKeys': max_keys}
            if prefix:
                params['Prefix'] = prefix

            response = self.s3_client.list_objects_v2(**params)

            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag'].strip('"')
                })

            logger.info(f"Found {len(objects)} object(s)")
            return objects

        except ClientError as e:
            logger.error(f"Error listing objects: {e}")
            raise

    def delete_object(self, bucket_name: str, object_key: str) -> None:
        """Delete object from S3."""
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=object_key)
            logger.info(f"✓ Object deleted: s3://{bucket_name}/{object_key}")

        except ClientError as e:
            logger.error(f"Error deleting object: {e}")
            raise

    def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str
    ) -> Dict[str, Any]:
        """Copy object within S3."""
        try:
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key
            )

            logger.info(f"✓ Object copied to s3://{dest_bucket}/{dest_key}")

            return {'source': f"{source_bucket}/{source_key}", 'destination': f"{dest_bucket}/{dest_key}"}

        except ClientError as e:
            logger.error(f"Error copying object: {e}")
            raise

    # ==================== Versioning ====================

    def enable_versioning(self, bucket_name: str) -> None:
        """Enable bucket versioning."""
        try:
            self.s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            logger.info(f"✓ Versioning enabled for {bucket_name}")

        except ClientError as e:
            logger.error(f"Error enabling versioning: {e}")
            raise

    def get_versioning_status(self, bucket_name: str) -> str:
        """Get bucket versioning status."""
        try:
            response = self.s3_client.get_bucket_versioning(Bucket=bucket_name)
            status = response.get('Status', 'Disabled')
            logger.info(f"Versioning status for {bucket_name}: {status}")
            return status

        except ClientError as e:
            logger.error(f"Error getting versioning status: {e}")
            raise

    # ==================== Presigned URLs ====================

    def generate_presigned_url(
        self,
        bucket_name: str,
        object_key: str,
        expiration: int = 3600,
        http_method: str = 'get_object'
    ) -> str:
        """
        Generate presigned URL for temporary access.

        Args:
            bucket_name: Bucket name
            object_key: Object key
            expiration: URL expiration in seconds (default: 1 hour)
            http_method: 'get_object' or 'put_object'

        Returns:
            Presigned URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                http_method,
                Params={'Bucket': bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )

            logger.info(f"✓ Presigned URL generated (expires in {expiration}s)")
            return url

        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise

    # ==================== Lifecycle Policies ====================

    def put_lifecycle_policy(
        self,
        bucket_name: str,
        rules: List[Dict[str, Any]]
    ) -> None:
        """
        Add lifecycle policy to bucket.

        Example rules:
        [
            {
                'ID': 'DeleteOldVersions',
                'Status': 'Enabled',
                'NoncurrentVersionExpiration': {'NoncurrentDays': 90}
            },
            {
                'ID': 'TransitionToGlacier',
                'Status': 'Enabled',
                'Transitions': [{'Days': 30, 'StorageClass': 'GLACIER'}]
            }
        ]
        """
        try:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket_name,
                LifecycleConfiguration={'Rules': rules}
            )
            logger.info(f"✓ Lifecycle policy configured for {bucket_name}")

        except ClientError as e:
            logger.error(f"Error setting lifecycle policy: {e}")
            raise

    # ==================== Bucket Policy ====================

    def put_bucket_policy(self, bucket_name: str, policy: Dict[str, Any]) -> None:
        """
        Set bucket policy.

        Args:
            bucket_name: Bucket name
            policy: Policy document as dictionary
        """
        try:
            policy_string = json.dumps(policy)
            self.s3_client.put_bucket_policy(Bucket=bucket_name, Policy=policy_string)
            logger.info(f"✓ Bucket policy applied to {bucket_name}")

        except ClientError as e:
            logger.error(f"Error setting bucket policy: {e}")
            raise

    # ==================== Encryption ====================

    def enable_default_encryption(
        self,
        bucket_name: str,
        sse_algorithm: str = "AES256",
        kms_key_id: Optional[str] = None
    ) -> None:
        """
        Enable default encryption for bucket.

        Args:
            bucket_name: Bucket name
            sse_algorithm: 'AES256' (SSE-S3) or 'aws:kms' (SSE-KMS)
            kms_key_id: KMS key ID (required for SSE-KMS)
        """
        try:
            rule = {'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': sse_algorithm}}
            if kms_key_id and sse_algorithm == 'aws:kms':
                rule['ApplyServerSideEncryptionByDefault']['KMSMasterKeyID'] = kms_key_id

            self.s3_client.put_bucket_encryption(
                Bucket=bucket_name,
                ServerSideEncryptionConfiguration={'Rules': [rule]}
            )

            logger.info(f"✓ Default encryption enabled for {bucket_name} ({sse_algorithm})")

        except ClientError as e:
            logger.error(f"Error enabling encryption: {e}")
            raise

    # ==================== Monitoring ====================

    def get_bucket_size(self, bucket_name: str) -> Dict[str, Any]:
        """Get total size and object count for bucket."""
        try:
            total_size = 0
            object_count = 0

            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name):
                for obj in page.get('Contents', []):
                    total_size += obj['Size']
                    object_count += 1

            return {
                'bucket': bucket_name,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'object_count': object_count
            }

        except ClientError as e:
            logger.error(f"Error getting bucket size: {e}")
            raise

    def get_summary(self) -> Dict[str, Any]:
        """Get S3 summary."""
        try:
            buckets = self.list_buckets()

            return {
                'region': self.region,
                'total_buckets': len(buckets),
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of S3 Manager capabilities."""
    print("AWS S3 Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    # Example usage
    print("\n1️⃣  Bucket Operations:")
    print("""
    s3_manager = S3Manager(region='us-east-1')

    # Create bucket
    s3_manager.create_bucket('my-data-bucket', acl='private')

    # List buckets
    buckets = s3_manager.list_buckets()
    for bucket in buckets:
        print(f"{bucket['name']} (created: {bucket['creation_date']})")
    """)

    print("\n2️⃣  Upload and Download:")
    print("""
    # Upload file
    s3_manager.upload_file(
        file_path='./data.csv',
        bucket_name='my-data-bucket',
        object_key='datasets/data.csv',
        metadata={'source': 'production'},
        tags={'Environment': 'prod', 'Team': 'data'}
    )

    # Upload from memory
    s3_manager.upload_object(
        bucket_name='my-data-bucket',
        object_key='config.json',
        data=b'{"setting": "value"}',
        content_type='application/json'
    )

    # Download file
    s3_manager.download_file(
        bucket_name='my-data-bucket',
        object_key='datasets/data.csv',
        file_path='./downloaded_data.csv'
    )
    """)

    print("\n3️⃣  Versioning and Presigned URLs:")
    print("""
    # Enable versioning
    s3_manager.enable_versioning('my-data-bucket')

    # Generate presigned URL (temporary access)
    url = s3_manager.generate_presigned_url(
        bucket_name='my-data-bucket',
        object_key='private/report.pdf',
        expiration=3600  # 1 hour
    )
    print(f"Share this URL: {url}")
    """)

    print("\n4️⃣  Lifecycle Policies:")
    print("""
    # Auto-delete old versions after 90 days
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
    """)

    print("\n5️⃣  Security:")
    print("""
    # Enable default encryption
    s3_manager.enable_default_encryption(
        bucket_name='my-data-bucket',
        sse_algorithm='AES256'
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
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")
    print("\n⚠️  Setup: Configure AWS credentials with `aws configure`")


if __name__ == '__main__':
    demo()
