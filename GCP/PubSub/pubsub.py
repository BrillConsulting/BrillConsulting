"""
Cloud Pub/Sub Messaging Service
Author: BrillConsulting
Description: Advanced Pub/Sub implementation with patterns, ordering, and error handling
"""

import json
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime
import time


class PubSubTopic:
    """Pub/Sub topic management"""

    def __init__(self, project_id: str):
        """
        Initialize topic manager

        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.topics = []

    def create_topic(self, topic_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create Pub/Sub topic

        Args:
            topic_id: Topic ID
            config: Topic configuration

        Returns:
            Topic creation result
        """
        print(f"\n{'='*60}")
        print("Creating Pub/Sub Topic")
        print(f"{'='*60}")

        config = config or {}
        message_retention = config.get('message_retention_duration', '7d')

        code = f"""
from google.cloud import pubsub_v1

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('{self.project_id}', '{topic_id}')

# Create topic
topic = publisher.create_topic(request={{"name": topic_path}})

# Set message retention
topic.message_retention_duration.seconds = 7 * 24 * 60 * 60  # 7 days

print(f"Created topic: {{topic.name}}")
"""

        result = {
            'topic_id': topic_id,
            'topic_path': f"projects/{self.project_id}/topics/{topic_id}",
            'message_retention': message_retention,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.topics.append(result)

        print(f"✓ Topic created: {topic_id}")
        print(f"  Path: {result['topic_path']}")
        print(f"  Retention: {message_retention}")
        print(f"{'='*60}")

        return result

    def create_ordered_topic(self, topic_id: str) -> Dict[str, Any]:
        """
        Create topic with message ordering enabled

        Args:
            topic_id: Topic ID

        Returns:
            Topic configuration
        """
        code = f"""
from google.cloud import pubsub_v1

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('{self.project_id}', '{topic_id}')

# Create topic with ordering enabled
topic = publisher.create_topic(
    request={{
        "name": topic_path,
        "message_storage_policy": {{
            "allowed_persistence_regions": ["us-central1", "us-east1"]
        }}
    }}
)

print(f"Created ordered topic: {{topic.name}}")
"""

        result = {
            'topic_id': topic_id,
            'ordering_enabled': True,
            'code': code
        }

        print(f"\n✓ Ordered topic configured: {topic_id}")
        return result


class PubSubSubscription:
    """Pub/Sub subscription management"""

    def __init__(self, project_id: str):
        """Initialize subscription manager"""
        self.project_id = project_id
        self.subscriptions = []

    def create_pull_subscription(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create pull subscription

        Args:
            config: Subscription configuration

        Returns:
            Subscription details
        """
        print(f"\n{'='*60}")
        print("Creating Pull Subscription")
        print(f"{'='*60}")

        subscription_id = config.get('subscription_id')
        topic_id = config.get('topic_id')
        ack_deadline = config.get('ack_deadline_seconds', 60)

        code = f"""
from google.cloud import pubsub_v1

subscriber = pubsub_v1.SubscriberClient()
topic_path = subscriber.topic_path('{self.project_id}', '{topic_id}')
subscription_path = subscriber.subscription_path('{self.project_id}', '{subscription_id}')

# Create subscription
subscription = subscriber.create_subscription(
    request={{
        "name": subscription_path,
        "topic": topic_path,
        "ack_deadline_seconds": {ack_deadline},
        "retain_acked_messages": False,
        "message_retention_duration": {{"seconds": 7 * 24 * 60 * 60}}  # 7 days
    }}
)

print(f"Created subscription: {{subscription.name}}")
"""

        result = {
            'subscription_id': subscription_id,
            'topic_id': topic_id,
            'subscription_path': f"projects/{self.project_id}/subscriptions/{subscription_id}",
            'ack_deadline_seconds': ack_deadline,
            'type': 'pull',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.subscriptions.append(result)

        print(f"✓ Pull subscription created: {subscription_id}")
        print(f"  Topic: {topic_id}")
        print(f"  Ack deadline: {ack_deadline}s")
        print(f"{'='*60}")

        return result

    def create_push_subscription(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create push subscription with endpoint

        Args:
            config: Subscription configuration

        Returns:
            Subscription details
        """
        print(f"\n{'='*60}")
        print("Creating Push Subscription")
        print(f"{'='*60}")

        subscription_id = config.get('subscription_id')
        topic_id = config.get('topic_id')
        push_endpoint = config.get('push_endpoint')

        code = f"""
from google.cloud import pubsub_v1

subscriber = pubsub_v1.SubscriberClient()
topic_path = subscriber.topic_path('{self.project_id}', '{topic_id}')
subscription_path = subscriber.subscription_path('{self.project_id}', '{subscription_id}')

# Create push subscription
subscription = subscriber.create_subscription(
    request={{
        "name": subscription_path,
        "topic": topic_path,
        "push_config": {{
            "push_endpoint": "{push_endpoint}",
            "oidc_token": {{
                "service_account_email": "pubsub@{self.project_id}.iam.gserviceaccount.com"
            }}
        }},
        "ack_deadline_seconds": 600
    }}
)

print(f"Created push subscription: {{subscription.name}}")
"""

        result = {
            'subscription_id': subscription_id,
            'topic_id': topic_id,
            'push_endpoint': push_endpoint,
            'type': 'push',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Push subscription created: {subscription_id}")
        print(f"  Endpoint: {push_endpoint}")
        print(f"{'='*60}")

        return result

    def create_dead_letter_subscription(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create subscription with dead letter topic

        Args:
            config: Subscription configuration

        Returns:
            Subscription with DLQ
        """
        subscription_id = config.get('subscription_id')
        topic_id = config.get('topic_id')
        dead_letter_topic = config.get('dead_letter_topic')
        max_delivery_attempts = config.get('max_delivery_attempts', 5)

        code = f"""
from google.cloud import pubsub_v1

subscriber = pubsub_v1.SubscriberClient()
topic_path = subscriber.topic_path('{self.project_id}', '{topic_id}')
subscription_path = subscriber.subscription_path('{self.project_id}', '{subscription_id}')
dead_letter_topic_path = subscriber.topic_path('{self.project_id}', '{dead_letter_topic}')

# Create subscription with dead letter topic
subscription = subscriber.create_subscription(
    request={{
        "name": subscription_path,
        "topic": topic_path,
        "dead_letter_policy": {{
            "dead_letter_topic": dead_letter_topic_path,
            "max_delivery_attempts": {max_delivery_attempts}
        }},
        "retry_policy": {{
            "minimum_backoff": {{"seconds": 10}},
            "maximum_backoff": {{"seconds": 600}}
        }}
    }}
)

print(f"Created subscription with DLQ: {{subscription.name}}")
"""

        result = {
            'subscription_id': subscription_id,
            'dead_letter_topic': dead_letter_topic,
            'max_delivery_attempts': max_delivery_attempts,
            'code': code
        }

        print(f"\n✓ Dead letter subscription created: {subscription_id}")
        print(f"  Max attempts: {max_delivery_attempts}")
        print(f"  DLQ topic: {dead_letter_topic}")

        return result


class PubSubPublisher:
    """Pub/Sub message publishing"""

    def __init__(self, project_id: str, topic_id: str):
        """Initialize publisher"""
        self.project_id = project_id
        self.topic_id = topic_id
        self.published_count = 0

    def publish_message(self, data: Dict[str, Any], attributes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Publish single message

        Args:
            data: Message data
            attributes: Message attributes

        Returns:
            Publish result
        """
        message_data = json.dumps(data).encode('utf-8')
        attributes = attributes or {}

        code = f"""
from google.cloud import pubsub_v1
import json

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('{self.project_id}', '{self.topic_id}')

# Publish message
data = json.dumps({data}).encode('utf-8')
future = publisher.publish(
    topic_path,
    data,
    **{attributes}
)

message_id = future.result()
print(f"Published message: {{message_id}}")
"""

        # Simulate publish
        message_id = f"msg_{self.published_count}_{int(time.time())}"
        self.published_count += 1

        result = {
            'message_id': message_id,
            'data': data,
            'attributes': attributes,
            'size_bytes': len(message_data),
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"\n✓ Message published: {message_id}")
        print(f"  Size: {result['size_bytes']} bytes")

        return result

    def publish_batch(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Publish batch of messages

        Args:
            messages: List of messages to publish

        Returns:
            Batch publish result
        """
        print(f"\n{'='*60}")
        print("Publishing Message Batch")
        print(f"{'='*60}")

        code = f"""
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1 import types
import json

publisher = pubsub_v1.PublisherClient(
    batch_settings=types.BatchSettings(
        max_messages=100,
        max_bytes=1024 * 1024,  # 1 MB
        max_latency=0.1  # 100ms
    )
)

topic_path = publisher.topic_path('{self.project_id}', '{self.topic_id}')

# Publish messages in batch
futures = []
for msg in messages:
    data = json.dumps(msg).encode('utf-8')
    future = publisher.publish(topic_path, data)
    futures.append(future)

# Wait for all messages
message_ids = [f.result() for f in futures]
print(f"Published {{len(message_ids)}} messages")
"""

        # Simulate batch publish
        published_ids = []
        total_bytes = 0

        for i, msg in enumerate(messages):
            msg_data = json.dumps(msg).encode('utf-8')
            message_id = f"batch_msg_{i}_{int(time.time())}"
            published_ids.append(message_id)
            total_bytes += len(msg_data)
            self.published_count += 1

        result = {
            'messages_published': len(published_ids),
            'message_ids': published_ids[:5],  # First 5
            'total_bytes': total_bytes,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Batch published: {len(published_ids)} messages")
        print(f"  Total size: {total_bytes} bytes")
        print(f"{'='*60}")

        return result

    def publish_with_ordering(self, messages: List[Dict[str, Any]], ordering_key: str) -> str:
        """
        Publish messages with ordering key

        Args:
            messages: Messages to publish
            ordering_key: Ordering key for message sequence

        Returns:
            Publishing code
        """
        code = f"""
from google.cloud import pubsub_v1
import json

# Enable message ordering
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('{self.project_id}', '{self.topic_id}')

# Publish with ordering key
for msg in messages:
    data = json.dumps(msg).encode('utf-8')
    future = publisher.publish(
        topic_path,
        data,
        ordering_key='{ordering_key}'
    )
    message_id = future.result()
    print(f"Published ordered message: {{message_id}}")

print("All messages published in order")
"""

        print(f"\n✓ Ordered publishing configured with key: {ordering_key}")
        return code


class PubSubSubscriber:
    """Pub/Sub message consumption"""

    def __init__(self, project_id: str, subscription_id: str):
        """Initialize subscriber"""
        self.project_id = project_id
        self.subscription_id = subscription_id
        self.messages_received = 0

    def pull_messages(self, max_messages: int = 10) -> str:
        """
        Pull messages synchronously

        Args:
            max_messages: Maximum messages to pull

        Returns:
            Pull code
        """
        code = f"""
from google.cloud import pubsub_v1

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path('{self.project_id}', '{self.subscription_id}')

# Pull messages
response = subscriber.pull(
    request={{
        "subscription": subscription_path,
        "max_messages": {max_messages}
    }},
    timeout=5.0
)

# Process messages
for received_message in response.received_messages:
    print(f"Received: {{received_message.message.data}}")

    # Acknowledge message
    subscriber.acknowledge(
        request={{
            "subscription": subscription_path,
            "ack_ids": [received_message.ack_id]
        }}
    )

print(f"Processed {{len(response.received_messages)}} messages")
"""

        print(f"\n✓ Pull subscriber configured (max {max_messages} messages)")
        return code

    def streaming_pull(self) -> str:
        """
        Set up streaming pull subscription

        Returns:
            Streaming pull code
        """
        code = f"""
from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path('{self.project_id}', '{self.subscription_id}')

def callback(message):
    print(f"Received message: {{message.data}}")

    # Process message
    try:
        # Your processing logic here
        process_message(message.data)

        # Acknowledge
        message.ack()
    except Exception as e:
        print(f"Error: {{e}}")
        message.nack()  # Requeue for retry

# Start streaming pull
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

print(f"Listening for messages on {{subscription_path}}...")

# Keep running
try:
    streaming_pull_future.result()
except TimeoutError:
    streaming_pull_future.cancel()
"""

        print(f"\n✓ Streaming pull configured for: {self.subscription_id}")
        return code


class PubSubManager:
    """Comprehensive Pub/Sub management"""

    def __init__(self, project_id: str = 'my-project'):
        """Initialize Pub/Sub manager"""
        self.project_id = project_id
        self.topics = []
        self.subscriptions = []

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'topics': len(self.topics),
            'subscriptions': len(self.subscriptions),
            'features': [
                'topics',
                'subscriptions',
                'push_pull',
                'message_ordering',
                'dead_letter_queues',
                'batch_publishing'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Pub/Sub capabilities"""
    print("=" * 60)
    print("Cloud Pub/Sub Messaging Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'

    # Create topic
    topic_mgr = PubSubTopic(project_id)
    topic = topic_mgr.create_topic('user-events', {
        'message_retention_duration': '7d'
    })

    # Create pull subscription
    sub_mgr = PubSubSubscription(project_id)
    pull_sub = sub_mgr.create_pull_subscription({
        'subscription_id': 'user-events-pull',
        'topic_id': 'user-events',
        'ack_deadline_seconds': 60
    })

    # Create push subscription
    push_sub = sub_mgr.create_push_subscription({
        'subscription_id': 'user-events-push',
        'topic_id': 'user-events',
        'push_endpoint': 'https://myapp.com/pubsub/receive'
    })

    # Create dead letter subscription
    dlq_sub = sub_mgr.create_dead_letter_subscription({
        'subscription_id': 'user-events-with-dlq',
        'topic_id': 'user-events',
        'dead_letter_topic': 'user-events-dlq',
        'max_delivery_attempts': 5
    })

    # Publish messages
    publisher = PubSubPublisher(project_id, 'user-events')

    # Single message
    msg = publisher.publish_message(
        data={'user_id': '123', 'event': 'login'},
        attributes={'source': 'web', 'version': 'v1'}
    )

    # Batch messages
    batch_messages = [
        {'user_id': f'user_{i}', 'event': 'action'}
        for i in range(10)
    ]
    batch_result = publisher.publish_batch(batch_messages)

    # Ordered publishing
    ordered_code = publisher.publish_with_ordering(
        messages=batch_messages,
        ordering_key='user_123'
    )

    # Pull messages
    subscriber = PubSubSubscriber(project_id, 'user-events-pull')
    pull_code = subscriber.pull_messages(max_messages=10)
    streaming_code = subscriber.streaming_pull()

    # Manager info
    mgr = PubSubManager(project_id)
    mgr.topics.append(topic)
    mgr.subscriptions.extend([pull_sub, push_sub, dlq_sub])

    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Pub/Sub Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Topics: {info['topics']}")
    print(f"Subscriptions: {info['subscriptions']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    demo()
