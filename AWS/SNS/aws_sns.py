"""
AWS SNS (Simple Notification Service)
======================================

Pub/Sub messaging for distributed systems and mobile notifications.

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


class SNSManager:
    """
    Advanced AWS SNS Management System

    Provides comprehensive SNS operations including:
    - Topic creation and management (Standard and FIFO)
    - Subscriptions (Email, SMS, HTTP/HTTPS, SQS, Lambda)
    - Message publishing with attributes and filtering
    - Platform applications for mobile push
    - SMS sending
    - Dead-letter queues
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize SNS Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.sns_client = session.client('sns', region_name=region)
            self.region = region
            logger.info(f"SNS Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing SNS Manager: {e}")
            raise

    # ==================== Topic Management ====================

    def create_topic(
        self,
        name: str,
        display_name: Optional[str] = None,
        fifo_topic: bool = False,
        content_based_deduplication: bool = False,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create SNS topic.

        Args:
            name: Topic name (must end with .fifo for FIFO topics)
            display_name: Display name for SMS notifications
            fifo_topic: Create FIFO topic
            content_based_deduplication: Enable content-based deduplication (FIFO only)
            tags: Resource tags
        """
        try:
            logger.info(f"Creating topic: {name}")

            params = {'Name': name}

            attributes = {}
            if display_name:
                attributes['DisplayName'] = display_name
            if fifo_topic:
                attributes['FifoTopic'] = 'true'
            if content_based_deduplication:
                attributes['ContentBasedDeduplication'] = 'true'

            if attributes:
                params['Attributes'] = attributes

            if tags:
                params['Tags'] = [{'Key': k, 'Value': v} for k, v in tags.items()]

            response = self.sns_client.create_topic(**params)

            logger.info(f"✓ Topic created: {name}")
            return {
                'topic_arn': response['TopicArn']
            }

        except ClientError as e:
            logger.error(f"Error creating topic: {e}")
            raise

    def get_topic_attributes(self, topic_arn: str) -> Dict[str, Any]:
        """Get topic attributes."""
        try:
            response = self.sns_client.get_topic_attributes(TopicArn=topic_arn)

            return {
                'topic_arn': topic_arn,
                'attributes': response['Attributes']
            }

        except ClientError as e:
            logger.error(f"Error getting topic attributes: {e}")
            raise

    def set_topic_attributes(
        self,
        topic_arn: str,
        attribute_name: str,
        attribute_value: str
    ) -> None:
        """Set topic attribute."""
        try:
            self.sns_client.set_topic_attributes(
                TopicArn=topic_arn,
                AttributeName=attribute_name,
                AttributeValue=attribute_value
            )
            logger.info(f"✓ Topic attribute updated: {attribute_name}")

        except ClientError as e:
            logger.error(f"Error setting topic attribute: {e}")
            raise

    def list_topics(self) -> List[str]:
        """List all topics."""
        try:
            response = self.sns_client.list_topics()

            topic_arns = [topic['TopicArn'] for topic in response.get('Topics', [])]
            logger.info(f"Found {len(topic_arns)} topic(s)")

            return topic_arns

        except ClientError as e:
            logger.error(f"Error listing topics: {e}")
            raise

    def delete_topic(self, topic_arn: str) -> None:
        """Delete topic."""
        try:
            self.sns_client.delete_topic(TopicArn=topic_arn)
            logger.info(f"✓ Topic deleted: {topic_arn}")

        except ClientError as e:
            logger.error(f"Error deleting topic: {e}")
            raise

    # ==================== Subscriptions ====================

    def subscribe(
        self,
        topic_arn: str,
        protocol: str,
        endpoint: str,
        filter_policy: Optional[Dict[str, Any]] = None,
        return_subscription_arn: bool = True
    ) -> Dict[str, Any]:
        """
        Subscribe to topic.

        Args:
            topic_arn: Topic ARN
            protocol: 'email', 'sms', 'http', 'https', 'sqs', 'lambda', 'application'
            endpoint: Email, phone, URL, queue ARN, or Lambda ARN
            filter_policy: Message filtering policy
            return_subscription_arn: Return subscription ARN immediately
        """
        try:
            logger.info(f"Creating subscription: {protocol} - {endpoint}")

            params = {
                'TopicArn': topic_arn,
                'Protocol': protocol,
                'Endpoint': endpoint,
                'ReturnSubscriptionArn': return_subscription_arn
            }

            if filter_policy:
                params['Attributes'] = {
                    'FilterPolicy': json.dumps(filter_policy)
                }

            response = self.sns_client.subscribe(**params)

            logger.info(f"✓ Subscription created: {response['SubscriptionArn']}")
            return {
                'subscription_arn': response['SubscriptionArn']
            }

        except ClientError as e:
            logger.error(f"Error creating subscription: {e}")
            raise

    def confirm_subscription(
        self,
        topic_arn: str,
        token: str,
        authenticate_on_unsubscribe: bool = False
    ) -> Dict[str, Any]:
        """Confirm subscription (for HTTP/HTTPS endpoints)."""
        try:
            response = self.sns_client.confirm_subscription(
                TopicArn=topic_arn,
                Token=token,
                AuthenticateOnUnsubscribe='true' if authenticate_on_unsubscribe else 'false'
            )

            logger.info(f"✓ Subscription confirmed")
            return {
                'subscription_arn': response['SubscriptionArn']
            }

        except ClientError as e:
            logger.error(f"Error confirming subscription: {e}")
            raise

    def list_subscriptions_by_topic(self, topic_arn: str) -> List[Dict[str, Any]]:
        """List subscriptions for topic."""
        try:
            response = self.sns_client.list_subscriptions_by_topic(TopicArn=topic_arn)

            subscriptions = [
                {
                    'subscription_arn': sub['SubscriptionArn'],
                    'protocol': sub['Protocol'],
                    'endpoint': sub['Endpoint']
                }
                for sub in response.get('Subscriptions', [])
            ]

            logger.info(f"Found {len(subscriptions)} subscription(s)")
            return subscriptions

        except ClientError as e:
            logger.error(f"Error listing subscriptions: {e}")
            raise

    def unsubscribe(self, subscription_arn: str) -> None:
        """Unsubscribe from topic."""
        try:
            self.sns_client.unsubscribe(SubscriptionArn=subscription_arn)
            logger.info(f"✓ Unsubscribed: {subscription_arn}")

        except ClientError as e:
            logger.error(f"Error unsubscribing: {e}")
            raise

    # ==================== Publishing ====================

    def publish(
        self,
        message: str,
        topic_arn: Optional[str] = None,
        target_arn: Optional[str] = None,
        phone_number: Optional[str] = None,
        subject: Optional[str] = None,
        message_attributes: Optional[Dict[str, Any]] = None,
        message_structure: Optional[str] = None,
        message_group_id: Optional[str] = None,
        message_deduplication_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Publish message.

        Args:
            message: Message body
            topic_arn: Topic ARN (for topic publishing)
            target_arn: Target ARN (for mobile push)
            phone_number: Phone number (for SMS)
            subject: Message subject (for email)
            message_attributes: Message attributes for filtering
            message_structure: 'json' for platform-specific messages
            message_group_id: Message group ID (FIFO topics)
            message_deduplication_id: Deduplication ID (FIFO topics)
        """
        try:
            params = {'Message': message}

            if topic_arn:
                params['TopicArn'] = topic_arn
            elif target_arn:
                params['TargetArn'] = target_arn
            elif phone_number:
                params['PhoneNumber'] = phone_number
            else:
                raise ValueError("Must specify topic_arn, target_arn, or phone_number")

            if subject:
                params['Subject'] = subject

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

            if message_structure:
                params['MessageStructure'] = message_structure

            if message_group_id:
                params['MessageGroupId'] = message_group_id

            if message_deduplication_id:
                params['MessageDeduplicationId'] = message_deduplication_id

            response = self.sns_client.publish(**params)

            logger.info(f"✓ Message published: {response['MessageId']}")
            return {
                'message_id': response['MessageId']
            }

        except ClientError as e:
            logger.error(f"Error publishing message: {e}")
            raise

    def publish_batch(
        self,
        topic_arn: str,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Publish batch messages (up to 10 messages).

        Args:
            topic_arn: Topic ARN
            messages: List of message entries with 'Id', 'Message', and optional fields
        """
        try:
            logger.info(f"Publishing {len(messages)} messages in batch")

            response = self.sns_client.publish_batch(
                TopicArn=topic_arn,
                PublishBatchRequestEntries=messages
            )

            successful = len(response.get('Successful', []))
            failed = len(response.get('Failed', []))

            logger.info(f"✓ Batch published: {successful} successful, {failed} failed")
            return {
                'successful': successful,
                'failed': failed,
                'failed_entries': response.get('Failed', [])
            }

        except ClientError as e:
            logger.error(f"Error publishing batch: {e}")
            raise

    # ==================== SMS ====================

    def send_sms(
        self,
        phone_number: str,
        message: str,
        sender_id: Optional[str] = None,
        sms_type: str = "Transactional"
    ) -> Dict[str, Any]:
        """
        Send SMS message.

        Args:
            phone_number: Phone number (E.164 format, e.g., +1234567890)
            message: Message text
            sender_id: Sender ID (up to 11 alphanumeric characters)
            sms_type: 'Promotional' or 'Transactional'
        """
        try:
            logger.info(f"Sending SMS to: {phone_number}")

            attributes = {
                'AWS.SNS.SMS.SMSType': {
                    'DataType': 'String',
                    'StringValue': sms_type
                }
            }

            if sender_id:
                attributes['AWS.SNS.SMS.SenderID'] = {
                    'DataType': 'String',
                    'StringValue': sender_id
                }

            response = self.sns_client.publish(
                PhoneNumber=phone_number,
                Message=message,
                MessageAttributes=attributes
            )

            logger.info(f"✓ SMS sent: {response['MessageId']}")
            return {
                'message_id': response['MessageId']
            }

        except ClientError as e:
            logger.error(f"Error sending SMS: {e}")
            raise

    # ==================== Monitoring ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get SNS summary."""
        try:
            topics = self.list_topics()

            return {
                'region': self.region,
                'total_topics': len(topics),
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of SNS Manager capabilities."""
    print("AWS SNS Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    print("\n1️⃣  Topic Management:")
    print("""
    sns = SNSManager(region='us-east-1')

    # Create standard topic
    topic = sns.create_topic(
        name='notifications',
        display_name='App Notifications'
    )

    # Create FIFO topic
    fifo_topic = sns.create_topic(
        name='orders.fifo',
        fifo_topic=True,
        content_based_deduplication=True
    )
    """)

    print("\n2️⃣  Subscriptions:")
    print("""
    # Email subscription
    sns.subscribe(
        topic_arn='arn:aws:sns:us-east-1:123456789012:notifications',
        protocol='email',
        endpoint='alerts@example.com'
    )

    # Lambda subscription with filter
    sns.subscribe(
        topic_arn='arn:aws:sns:us-east-1:123456789012:notifications',
        protocol='lambda',
        endpoint='arn:aws:lambda:us-east-1:123456789012:function:process',
        filter_policy={
            'event_type': ['order_placed', 'order_shipped']
        }
    )

    # SQS subscription
    sns.subscribe(
        topic_arn='arn:aws:sns:us-east-1:123456789012:notifications',
        protocol='sqs',
        endpoint='arn:aws:sqs:us-east-1:123456789012:my-queue'
    )
    """)

    print("\n3️⃣  Publishing Messages:")
    print("""
    # Publish to topic
    sns.publish(
        topic_arn='arn:aws:sns:us-east-1:123456789012:notifications',
        message='New order received!',
        subject='Order Notification',
        message_attributes={
            'event_type': 'order_placed',
            'priority': 'high'
        }
    )

    # Publish to FIFO topic
    sns.publish(
        topic_arn='arn:aws:sns:us-east-1:123456789012:orders.fifo',
        message='Order #12345',
        message_group_id='order-processing',
        message_deduplication_id='order-12345'
    )
    """)

    print("\n4️⃣  SMS Messaging:")
    print("""
    # Send SMS
    sns.send_sms(
        phone_number='+1234567890',
        message='Your verification code is: 123456',
        sender_id='MyApp',
        sms_type='Transactional'
    )
    """)

    print("\n5️⃣  Batch Publishing:")
    print("""
    # Publish multiple messages
    messages = [
        {'Id': '1', 'Message': 'Message 1', 'Subject': 'Subject 1'},
        {'Id': '2', 'Message': 'Message 2', 'Subject': 'Subject 2'},
        {'Id': '3', 'Message': 'Message 3', 'Subject': 'Subject 3'}
    ]

    result = sns.publish_batch(
        topic_arn='arn:aws:sns:us-east-1:123456789012:notifications',
        messages=messages
    )
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
