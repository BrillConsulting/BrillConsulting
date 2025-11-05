"""
AWS SQS (Simple Queue Service)
===============================

Fully managed message queuing for microservices and distributed systems.

Author: Brill Consulting
"""

import boto3
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SQSManager:
    """
    Advanced AWS SQS Management System

    Provides comprehensive SQS operations including:
    - Queue creation and management (Standard and FIFO)
    - Message sending and receiving
    - Batch operations (send/receive/delete)
    - Message attributes and system attributes
    - Visibility timeout management
    - Dead-letter queues
    - Long polling for cost optimization
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize SQS Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.sqs_client = session.client('sqs', region_name=region)
            self.sqs_resource = session.resource('sqs', region_name=region)
            self.region = region
            logger.info(f"SQS Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing SQS Manager: {e}")
            raise

    # ==================== Queue Management ====================

    def create_queue(
        self,
        queue_name: str,
        fifo_queue: bool = False,
        content_based_deduplication: bool = False,
        visibility_timeout: int = 30,
        message_retention_period: int = 345600,  # 4 days
        receive_message_wait_time: int = 0,
        delay_seconds: int = 0,
        dead_letter_queue_arn: Optional[str] = None,
        max_receive_count: int = 5,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create SQS queue.

        Args:
            queue_name: Queue name (must end with .fifo for FIFO queues)
            fifo_queue: Create FIFO queue
            content_based_deduplication: Enable content-based deduplication (FIFO only)
            visibility_timeout: Visibility timeout in seconds (0-43200)
            message_retention_period: Message retention in seconds (60-1209600)
            receive_message_wait_time: Long polling wait time (0-20 seconds)
            delay_seconds: Delivery delay (0-900 seconds)
            dead_letter_queue_arn: DLQ ARN for failed messages
            max_receive_count: Max receives before sending to DLQ
            tags: Resource tags
        """
        try:
            logger.info(f"Creating queue: {queue_name}")

            attributes = {
                'VisibilityTimeout': str(visibility_timeout),
                'MessageRetentionPeriod': str(message_retention_period),
                'ReceiveMessageWaitTimeSeconds': str(receive_message_wait_time),
                'DelaySeconds': str(delay_seconds)
            }

            if fifo_queue:
                attributes['FifoQueue'] = 'true'

            if content_based_deduplication:
                attributes['ContentBasedDeduplication'] = 'true'

            if dead_letter_queue_arn:
                redrive_policy = {
                    'deadLetterTargetArn': dead_letter_queue_arn,
                    'maxReceiveCount': max_receive_count
                }
                attributes['RedrivePolicy'] = json.dumps(redrive_policy)

            params = {
                'QueueName': queue_name,
                'Attributes': attributes
            }

            if tags:
                params['tags'] = tags

            response = self.sqs_client.create_queue(**params)

            logger.info(f"✓ Queue created: {queue_name}")
            return {
                'queue_url': response['QueueUrl']
            }

        except ClientError as e:
            logger.error(f"Error creating queue: {e}")
            raise

    def get_queue_url(self, queue_name: str) -> str:
        """Get queue URL by name."""
        try:
            response = self.sqs_client.get_queue_url(QueueName=queue_name)
            return response['QueueUrl']

        except ClientError as e:
            logger.error(f"Error getting queue URL: {e}")
            raise

    def get_queue_attributes(self, queue_url: str) -> Dict[str, Any]:
        """Get queue attributes."""
        try:
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=['All']
            )

            attributes = response['Attributes']
            return {
                'queue_url': queue_url,
                'queue_arn': attributes.get('QueueArn'),
                'messages_available': int(attributes.get('ApproximateNumberOfMessages', 0)),
                'messages_in_flight': int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0)),
                'messages_delayed': int(attributes.get('ApproximateNumberOfMessagesDelayed', 0)),
                'visibility_timeout': int(attributes.get('VisibilityTimeout', 0)),
                'created_timestamp': attributes.get('CreatedTimestamp')
            }

        except ClientError as e:
            logger.error(f"Error getting queue attributes: {e}")
            raise

    def set_queue_attributes(self, queue_url: str, attributes: Dict[str, str]) -> None:
        """Set queue attributes."""
        try:
            self.sqs_client.set_queue_attributes(
                QueueUrl=queue_url,
                Attributes=attributes
            )
            logger.info(f"✓ Queue attributes updated")

        except ClientError as e:
            logger.error(f"Error setting queue attributes: {e}")
            raise

    def list_queues(self, queue_name_prefix: Optional[str] = None) -> List[str]:
        """List queues."""
        try:
            params = {}
            if queue_name_prefix:
                params['QueueNamePrefix'] = queue_name_prefix

            response = self.sqs_client.list_queues(**params)

            queue_urls = response.get('QueueUrls', [])
            logger.info(f"Found {len(queue_urls)} queue(s)")

            return queue_urls

        except ClientError as e:
            logger.error(f"Error listing queues: {e}")
            raise

    def delete_queue(self, queue_url: str) -> None:
        """Delete queue."""
        try:
            self.sqs_client.delete_queue(QueueUrl=queue_url)
            logger.info(f"✓ Queue deleted: {queue_url}")

        except ClientError as e:
            logger.error(f"Error deleting queue: {e}")
            raise

    def purge_queue(self, queue_url: str) -> None:
        """Purge all messages from queue."""
        try:
            self.sqs_client.purge_queue(QueueUrl=queue_url)
            logger.info(f"✓ Queue purged: {queue_url}")

        except ClientError as e:
            logger.error(f"Error purging queue: {e}")
            raise

    # ==================== Message Operations ====================

    def send_message(
        self,
        queue_url: str,
        message_body: str,
        delay_seconds: int = 0,
        message_attributes: Optional[Dict[str, Any]] = None,
        message_group_id: Optional[str] = None,
        message_deduplication_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send message to queue.

        Args:
            queue_url: Queue URL
            message_body: Message content
            delay_seconds: Delivery delay (0-900)
            message_attributes: Custom message attributes
            message_group_id: Message group ID (FIFO only)
            message_deduplication_id: Deduplication ID (FIFO only)
        """
        try:
            params = {
                'QueueUrl': queue_url,
                'MessageBody': message_body,
                'DelaySeconds': delay_seconds
            }

            if message_attributes:
                formatted_attributes = {}
                for key, value in message_attributes.items():
                    if isinstance(value, str):
                        formatted_attributes[key] = {
                            'DataType': 'String',
                            'StringValue': value
                        }
                    elif isinstance(value, (int, float)):
                        formatted_attributes[key] = {
                            'DataType': 'Number',
                            'StringValue': str(value)
                        }
                params['MessageAttributes'] = formatted_attributes

            if message_group_id:
                params['MessageGroupId'] = message_group_id

            if message_deduplication_id:
                params['MessageDeduplicationId'] = message_deduplication_id

            response = self.sqs_client.send_message(**params)

            logger.info(f"✓ Message sent: {response['MessageId']}")
            return {
                'message_id': response['MessageId'],
                'md5_of_body': response['MD5OfMessageBody']
            }

        except ClientError as e:
            logger.error(f"Error sending message: {e}")
            raise

    def send_message_batch(
        self,
        queue_url: str,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Send batch messages (up to 10 messages).

        Args:
            queue_url: Queue URL
            messages: List of message entries with 'Id', 'MessageBody', etc.
        """
        try:
            logger.info(f"Sending {len(messages)} messages in batch")

            response = self.sqs_client.send_message_batch(
                QueueUrl=queue_url,
                Entries=messages
            )

            successful = len(response.get('Successful', []))
            failed = len(response.get('Failed', []))

            logger.info(f"✓ Batch sent: {successful} successful, {failed} failed")
            return {
                'successful': successful,
                'failed': failed,
                'failed_entries': response.get('Failed', [])
            }

        except ClientError as e:
            logger.error(f"Error sending batch: {e}")
            raise

    def receive_messages(
        self,
        queue_url: str,
        max_number_of_messages: int = 1,
        wait_time_seconds: int = 0,
        visibility_timeout: Optional[int] = None,
        message_attribute_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Receive messages from queue.

        Args:
            queue_url: Queue URL
            max_number_of_messages: Max messages to receive (1-10)
            wait_time_seconds: Long polling wait time (0-20)
            visibility_timeout: Override visibility timeout
            message_attribute_names: List of attribute names to retrieve
        """
        try:
            params = {
                'QueueUrl': queue_url,
                'MaxNumberOfMessages': max_number_of_messages,
                'WaitTimeSeconds': wait_time_seconds,
                'AttributeNames': ['All']
            }

            if visibility_timeout is not None:
                params['VisibilityTimeout'] = visibility_timeout

            if message_attribute_names:
                params['MessageAttributeNames'] = message_attribute_names
            else:
                params['MessageAttributeNames'] = ['All']

            response = self.sqs_client.receive_message(**params)

            messages = response.get('Messages', [])
            logger.info(f"Received {len(messages)} message(s)")

            return [
                {
                    'message_id': msg['MessageId'],
                    'receipt_handle': msg['ReceiptHandle'],
                    'body': msg['Body'],
                    'attributes': msg.get('Attributes', {}),
                    'message_attributes': msg.get('MessageAttributes', {})
                }
                for msg in messages
            ]

        except ClientError as e:
            logger.error(f"Error receiving messages: {e}")
            raise

    def delete_message(self, queue_url: str, receipt_handle: str) -> None:
        """Delete message from queue."""
        try:
            self.sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.info(f"✓ Message deleted")

        except ClientError as e:
            logger.error(f"Error deleting message: {e}")
            raise

    def delete_message_batch(
        self,
        queue_url: str,
        receipt_handles: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Delete batch messages (up to 10 messages).

        Args:
            queue_url: Queue URL
            receipt_handles: List of {'Id': '1', 'ReceiptHandle': 'xxx'}
        """
        try:
            logger.info(f"Deleting {len(receipt_handles)} messages in batch")

            response = self.sqs_client.delete_message_batch(
                QueueUrl=queue_url,
                Entries=receipt_handles
            )

            successful = len(response.get('Successful', []))
            failed = len(response.get('Failed', []))

            logger.info(f"✓ Batch deleted: {successful} successful, {failed} failed")
            return {
                'successful': successful,
                'failed': failed,
                'failed_entries': response.get('Failed', [])
            }

        except ClientError as e:
            logger.error(f"Error deleting batch: {e}")
            raise

    def change_message_visibility(
        self,
        queue_url: str,
        receipt_handle: str,
        visibility_timeout: int
    ) -> None:
        """Change message visibility timeout."""
        try:
            self.sqs_client.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout
            )
            logger.info(f"✓ Visibility timeout updated: {visibility_timeout}s")

        except ClientError as e:
            logger.error(f"Error changing visibility: {e}")
            raise

    # ==================== Monitoring ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get SQS summary."""
        try:
            queues = self.list_queues()

            total_messages = 0
            for queue_url in queues:
                attrs = self.get_queue_attributes(queue_url)
                total_messages += attrs['messages_available']

            return {
                'region': self.region,
                'total_queues': len(queues),
                'total_messages': total_messages,
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of SQS Manager capabilities."""
    print("AWS SQS Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    print("\n1️⃣  Queue Management:")
    print("""
    sqs = SQSManager(region='us-east-1')

    # Create standard queue
    queue = sqs.create_queue(
        queue_name='my-queue',
        visibility_timeout=60,
        receive_message_wait_time=10  # Long polling
    )

    # Create FIFO queue
    fifo_queue = sqs.create_queue(
        queue_name='orders.fifo',
        fifo_queue=True,
        content_based_deduplication=True
    )

    # Create queue with DLQ
    dlq = sqs.create_queue('my-dlq')
    dlq_attrs = sqs.get_queue_attributes(dlq['queue_url'])

    queue_with_dlq = sqs.create_queue(
        queue_name='my-queue-with-dlq',
        dead_letter_queue_arn=dlq_attrs['queue_arn'],
        max_receive_count=3
    )
    """)

    print("\n2️⃣  Send Messages:")
    print("""
    # Send single message
    sqs.send_message(
        queue_url='https://sqs.us-east-1.amazonaws.com/123456789012/my-queue',
        message_body='Hello from SQS!',
        message_attributes={
            'priority': 'high',
            'customer_id': '12345'
        }
    )

    # Send FIFO message
    sqs.send_message(
        queue_url='https://sqs.us-east-1.amazonaws.com/123456789012/orders.fifo',
        message_body='Order #12345',
        message_group_id='order-processing',
        message_deduplication_id='order-12345'
    )

    # Send batch
    messages = [
        {'Id': '1', 'MessageBody': 'Message 1'},
        {'Id': '2', 'MessageBody': 'Message 2', 'DelaySeconds': 10},
        {'Id': '3', 'MessageBody': 'Message 3'}
    ]
    sqs.send_message_batch(queue_url, messages)
    """)

    print("\n3️⃣  Receive and Process Messages:")
    print("""
    # Receive messages (long polling)
    messages = sqs.receive_messages(
        queue_url='https://sqs.us-east-1.amazonaws.com/123456789012/my-queue',
        max_number_of_messages=10,
        wait_time_seconds=20  # Long polling
    )

    # Process and delete
    for msg in messages:
        # Process message
        print(f"Processing: {msg['body']}")

        # Delete after successful processing
        sqs.delete_message(
            queue_url='...',
            receipt_handle=msg['receipt_handle']
        )
    """)

    print("\n4️⃣  Visibility Timeout:")
    print("""
    # Extend processing time
    sqs.change_message_visibility(
        queue_url='...',
        receipt_handle=msg['receipt_handle'],
        visibility_timeout=120  # 2 more minutes
    )
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
