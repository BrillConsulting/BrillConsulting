"""
AWS DynamoDB Management
=======================

Comprehensive NoSQL database operations with DynamoDB.

Author: Brill Consulting
"""

import boto3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
from botocore.exceptions import ClientError, NoCredentialsError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamoDBManager:
    """
    Advanced AWS DynamoDB Management System

    Provides comprehensive DynamoDB operations including:
    - Table creation and management
    - Item CRUD operations (Create, Read, Update, Delete)
    - Batch operations
    - Query and Scan with filters
    - Global and Local Secondary Indexes
    - Streams and backups
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize DynamoDB Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.dynamodb_client = session.client('dynamodb', region_name=region)
            self.dynamodb_resource = session.resource('dynamodb', region_name=region)
            self.region = region
            logger.info(f"DynamoDB Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing DynamoDB Manager: {e}")
            raise

    # ==================== Table Management ====================

    def create_table(
        self,
        table_name: str,
        key_schema: List[Dict[str, str]],
        attribute_definitions: List[Dict[str, str]],
        billing_mode: str = "PAY_PER_REQUEST",
        provisioned_throughput: Optional[Dict[str, int]] = None,
        global_secondary_indexes: Optional[List[Dict[str, Any]]] = None,
        stream_enabled: bool = False
    ) -> Dict[str, Any]:
        """
        Create DynamoDB table.

        Args:
            table_name: Table name
            key_schema: [{'AttributeName': 'id', 'KeyType': 'HASH'}]
            attribute_definitions: [{'AttributeName': 'id', 'AttributeType': 'S'}]
            billing_mode: PAY_PER_REQUEST or PROVISIONED
            provisioned_throughput: {'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
            global_secondary_indexes: GSI configurations
            stream_enabled: Enable DynamoDB Streams
        """
        try:
            logger.info(f"Creating table: {table_name}")

            params = {
                'TableName': table_name,
                'KeySchema': key_schema,
                'AttributeDefinitions': attribute_definitions,
                'BillingMode': billing_mode
            }

            if billing_mode == 'PROVISIONED' and provisioned_throughput:
                params['ProvisionedThroughput'] = provisioned_throughput

            if global_secondary_indexes:
                params['GlobalSecondaryIndexes'] = global_secondary_indexes

            if stream_enabled:
                params['StreamSpecification'] = {
                    'StreamEnabled': True,
                    'StreamViewType': 'NEW_AND_OLD_IMAGES'
                }

            response = self.dynamodb_client.create_table(**params)

            logger.info(f"✓ Table created: {table_name}")

            return {
                'table_name': response['TableDescription']['TableName'],
                'table_arn': response['TableDescription']['TableArn'],
                'status': response['TableDescription']['TableStatus']
            }

        except ClientError as e:
            logger.error(f"Error creating table: {e}")
            raise

    def describe_table(self, table_name: str) -> Dict[str, Any]:
        """Get table details."""
        try:
            response = self.dynamodb_client.describe_table(TableName=table_name)

            table = response['Table']
            return {
                'table_name': table['TableName'],
                'table_arn': table['TableArn'],
                'status': table['TableStatus'],
                'item_count': table['ItemCount'],
                'table_size_bytes': table['TableSizeBytes'],
                'creation_date': table['CreationDateTime'].isoformat()
            }

        except ClientError as e:
            logger.error(f"Error describing table: {e}")
            raise

    def list_tables(self) -> List[str]:
        """List all tables."""
        try:
            response = self.dynamodb_client.list_tables()
            tables = response.get('TableNames', [])
            logger.info(f"Found {len(tables)} table(s)")
            return tables

        except ClientError as e:
            logger.error(f"Error listing tables: {e}")
            raise

    def delete_table(self, table_name: str) -> None:
        """Delete table."""
        try:
            self.dynamodb_client.delete_table(TableName=table_name)
            logger.info(f"✓ Table deleted: {table_name}")
        except ClientError as e:
            logger.error(f"Error deleting table: {e}")
            raise

    # ==================== Item Operations ====================

    def put_item(self, table_name: str, item: Dict[str, Any]) -> None:
        """Insert or replace item."""
        try:
            table = self.dynamodb_resource.Table(table_name)
            table.put_item(Item=item)
            logger.info(f"✓ Item inserted into {table_name}")

        except ClientError as e:
            logger.error(f"Error putting item: {e}")
            raise

    def get_item(
        self,
        table_name: str,
        key: Dict[str, Any],
        consistent_read: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get item by key."""
        try:
            table = self.dynamodb_resource.Table(table_name)
            response = table.get_item(Key=key, ConsistentRead=consistent_read)

            item = response.get('Item')
            if item:
                logger.info(f"✓ Item retrieved from {table_name}")
            else:
                logger.info(f"Item not found in {table_name}")

            return item

        except ClientError as e:
            logger.error(f"Error getting item: {e}")
            raise

    def update_item(
        self,
        table_name: str,
        key: Dict[str, Any],
        update_expression: str,
        expression_attribute_values: Dict[str, Any],
        expression_attribute_names: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Update item attributes."""
        try:
            table = self.dynamodb_resource.Table(table_name)

            params = {
                'Key': key,
                'UpdateExpression': update_expression,
                'ExpressionAttributeValues': expression_attribute_values,
                'ReturnValues': 'ALL_NEW'
            }

            if expression_attribute_names:
                params['ExpressionAttributeNames'] = expression_attribute_names

            response = table.update_item(**params)

            logger.info(f"✓ Item updated in {table_name}")
            return response['Attributes']

        except ClientError as e:
            logger.error(f"Error updating item: {e}")
            raise

    def delete_item(self, table_name: str, key: Dict[str, Any]) -> None:
        """Delete item."""
        try:
            table = self.dynamodb_resource.Table(table_name)
            table.delete_item(Key=key)
            logger.info(f"✓ Item deleted from {table_name}")

        except ClientError as e:
            logger.error(f"Error deleting item: {e}")
            raise

    # ==================== Batch Operations ====================

    def batch_write_items(
        self,
        table_name: str,
        items: List[Dict[str, Any]]
    ) -> None:
        """Batch write items (up to 25 at a time)."""
        try:
            table = self.dynamodb_resource.Table(table_name)

            with table.batch_writer() as batch:
                for item in items:
                    batch.put_item(Item=item)

            logger.info(f"✓ Batch wrote {len(items)} items to {table_name}")

        except ClientError as e:
            logger.error(f"Error batch writing items: {e}")
            raise

    def batch_get_items(
        self,
        table_name: str,
        keys: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Batch get items (up to 100 at a time)."""
        try:
            response = self.dynamodb_resource.batch_get_item(
                RequestItems={
                    table_name: {'Keys': keys}
                }
            )

            items = response.get('Responses', {}).get(table_name, [])
            logger.info(f"✓ Retrieved {len(items)} items from {table_name}")

            return items

        except ClientError as e:
            logger.error(f"Error batch getting items: {e}")
            raise

    # ==================== Query and Scan ====================

    def query(
        self,
        table_name: str,
        key_condition_expression: str,
        expression_attribute_values: Dict[str, Any],
        filter_expression: Optional[str] = None,
        index_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query items."""
        try:
            table = self.dynamodb_resource.Table(table_name)

            params = {
                'KeyConditionExpression': key_condition_expression,
                'ExpressionAttributeValues': expression_attribute_values
            }

            if filter_expression:
                params['FilterExpression'] = filter_expression
            if index_name:
                params['IndexName'] = index_name
            if limit:
                params['Limit'] = limit

            response = table.query(**params)

            items = response.get('Items', [])
            logger.info(f"Query returned {len(items)} item(s)")

            return items

        except ClientError as e:
            logger.error(f"Error querying table: {e}")
            raise

    def scan(
        self,
        table_name: str,
        filter_expression: Optional[str] = None,
        expression_attribute_values: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Scan table."""
        try:
            table = self.dynamodb_resource.Table(table_name)

            params = {}
            if filter_expression:
                params['FilterExpression'] = filter_expression
            if expression_attribute_values:
                params['ExpressionAttributeValues'] = expression_attribute_values
            if limit:
                params['Limit'] = limit

            response = table.scan(**params)

            items = response.get('Items', [])
            logger.info(f"Scan returned {len(items)} item(s)")

            return items

        except ClientError as e:
            logger.error(f"Error scanning table: {e}")
            raise

    # ==================== Monitoring ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get DynamoDB summary."""
        try:
            tables = self.list_tables()

            return {
                'region': self.region,
                'total_tables': len(tables),
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of DynamoDB Manager capabilities."""
    print("AWS DynamoDB Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    print("\n1️⃣  Create Table:")
    print("""
    db = DynamoDBManager(region='us-east-1')

    table = db.create_table(
        table_name='Users',
        key_schema=[
            {'AttributeName': 'user_id', 'KeyType': 'HASH'}
        ],
        attribute_definitions=[
            {'AttributeName': 'user_id', 'AttributeType': 'S'}
        ],
        billing_mode='PAY_PER_REQUEST'
    )
    """)

    print("\n2️⃣  Item Operations:")
    print("""
    # Put item
    db.put_item('Users', {
        'user_id': 'user123',
        'name': 'John Doe',
        'email': 'john@example.com',
        'age': 30
    })

    # Get item
    item = db.get_item('Users', {'user_id': 'user123'})

    # Update item
    db.update_item(
        'Users',
        key={'user_id': 'user123'},
        update_expression='SET age = :age',
        expression_attribute_values={':age': 31}
    )

    # Delete item
    db.delete_item('Users', {'user_id': 'user123'})
    """)

    print("\n3️⃣  Batch Operations:")
    print("""
    # Batch write
    items = [
        {'user_id': f'user{i}', 'name': f'User {i}'}
        for i in range(10)
    ]
    db.batch_write_items('Users', items)

    # Batch get
    keys = [{'user_id': f'user{i}'} for i in range(5)]
    items = db.batch_get_items('Users', keys)
    """)

    print("\n4️⃣  Query and Scan:")
    print("""
    # Query
    users = db.query(
        'Users',
        key_condition_expression='user_id = :uid',
        expression_attribute_values={':uid': 'user123'}
    )

    # Scan with filter
    active_users = db.scan(
        'Users',
        filter_expression='age > :min_age',
        expression_attribute_values={':min_age': 18}
    )
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
