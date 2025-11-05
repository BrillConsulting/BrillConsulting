"""
AWS Secrets Manager
===================

Secure secrets management with rotation and versioning.

Author: Brill Consulting
"""

import boto3
import logging
import json
from typing import Dict, Optional, Any
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecretsManagerManager:
    """AWS Secrets Manager operations."""

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.secrets_client = session.client('secretsmanager', region_name=region)
            self.region = region
            logger.info(f"Secrets Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise

    def create_secret(
        self,
        name: str,
        secret_string: Optional[str] = None,
        secret_binary: Optional[bytes] = None,
        description: str = "",
        kms_key_id: Optional[str] = None,
        tags: Optional[list] = None
    ) -> Dict[str, Any]:
        """Create secret."""
        try:
            params = {'Name': name, 'Description': description}

            if secret_string:
                params['SecretString'] = secret_string
            elif secret_binary:
                params['SecretBinary'] = secret_binary

            if kms_key_id:
                params['KmsKeyId'] = kms_key_id
            if tags:
                params['Tags'] = tags

            response = self.secrets_client.create_secret(**params)
            logger.info(f"✓ Secret created: {name}")

            return {
                'arn': response['ARN'],
                'name': response['Name'],
                'version_id': response['VersionId']
            }

        except ClientError as e:
            logger.error(f"Error creating secret: {e}")
            raise

    def get_secret_value(self, secret_id: str, version_id: Optional[str] = None) -> Dict[str, Any]:
        """Get secret value."""
        try:
            params = {'SecretId': secret_id}
            if version_id:
                params['VersionId'] = version_id

            response = self.secrets_client.get_secret_value(**params)

            result = {
                'arn': response['ARN'],
                'name': response['Name'],
                'version_id': response['VersionId']
            }

            if 'SecretString' in response:
                result['secret_string'] = response['SecretString']
            if 'SecretBinary' in response:
                result['secret_binary'] = response['SecretBinary']

            logger.info(f"✓ Retrieved secret: {secret_id}")
            return result

        except ClientError as e:
            logger.error(f"Error getting secret: {e}")
            raise

    def update_secret(
        self,
        secret_id: str,
        secret_string: Optional[str] = None,
        secret_binary: Optional[bytes] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update secret value."""
        try:
            params = {'SecretId': secret_id}

            if secret_string:
                params['SecretString'] = secret_string
            elif secret_binary:
                params['SecretBinary'] = secret_binary

            if description:
                params['Description'] = description

            response = self.secrets_client.update_secret(**params)
            logger.info(f"✓ Secret updated: {secret_id}")

            return {
                'arn': response['ARN'],
                'name': response['Name'],
                'version_id': response['VersionId']
            }

        except ClientError as e:
            logger.error(f"Error updating secret: {e}")
            raise

    def delete_secret(
        self,
        secret_id: str,
        recovery_window_in_days: int = 30,
        force_delete: bool = False
    ) -> Dict[str, Any]:
        """Delete secret (with recovery window)."""
        try:
            params = {'SecretId': secret_id}

            if force_delete:
                params['ForceDeleteWithoutRecovery'] = True
            else:
                params['RecoveryWindowInDays'] = recovery_window_in_days

            response = self.secrets_client.delete_secret(**params)
            logger.info(f"✓ Secret scheduled for deletion: {secret_id}")

            return {
                'arn': response['ARN'],
                'name': response['Name'],
                'deletion_date': response.get('DeletionDate', 'N/A')
            }

        except ClientError as e:
            logger.error(f"Error deleting secret: {e}")
            raise

    def list_secrets(self, max_results: int = 100) -> list:
        """List all secrets."""
        try:
            response = self.secrets_client.list_secrets(MaxResults=max_results)
            secrets = [
                {
                    'arn': s['ARN'],
                    'name': s['Name'],
                    'last_changed': s.get('LastChangedDate', datetime.now()).isoformat()
                }
                for s in response.get('SecretList', [])
            ]
            logger.info(f"Found {len(secrets)} secret(s)")
            return secrets

        except ClientError as e:
            logger.error(f"Error listing secrets: {e}")
            raise

    def rotate_secret(self, secret_id: str, rotation_lambda_arn: str) -> None:
        """Enable automatic secret rotation."""
        try:
            self.secrets_client.rotate_secret(
                SecretId=secret_id,
                RotationLambdaARN=rotation_lambda_arn,
                RotationRules={'AutomaticallyAfterDays': 30}
            )
            logger.info(f"✓ Rotation enabled for: {secret_id}")

        except ClientError as e:
            logger.error(f"Error rotating secret: {e}")
            raise


def demo():
    """Demo Secrets Manager."""
    print("AWS Secrets Manager - Demo")
    print("=" * 70)
    print("""
    sm = SecretsManagerManager(region='us-east-1')

    # Create secret
    secret = sm.create_secret(
        name='db-password',
        secret_string=json.dumps({'username': 'admin', 'password': 'secret123'}),
        description='Database credentials'
    )

    # Get secret
    value = sm.get_secret_value('db-password')
    credentials = json.loads(value['secret_string'])

    # Update secret
    sm.update_secret(
        'db-password',
        secret_string=json.dumps({'username': 'admin', 'password': 'newsecret456'})
    )

    # Delete secret (30-day recovery)
    sm.delete_secret('db-password', recovery_window_in_days=30)
    """)
    print("\n✓ Demo Complete!")


if __name__ == '__main__':
    demo()
