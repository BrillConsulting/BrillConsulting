# Pub/Sub - Messaging and Event Streaming

Comprehensive Google Cloud Pub/Sub implementation for reliable, asynchronous messaging and event-driven architectures.

## Features

### Topic Management
- **Topic Creation**: Create topics with message retention policies
- **Topic Configuration**: Configure message storage and encryption
- **Schema Management**: Define and enforce message schemas
- **Topic Deletion**: Clean up unused topics

### Subscription Types
- **Pull Subscriptions**: Application-controlled message retrieval
- **Push Subscriptions**: HTTP endpoint message delivery
- **Dead Letter Queues**: Automatic handling of undeliverable messages
- **Snapshot Management**: Point-in-time message replay

### Publishing
- **Single Message Publishing**: Publish individual messages
- **Batch Publishing**: Efficient bulk message publishing (100 msgs, 1MB, 100ms)
- **Message Ordering**: Guaranteed ordering with ordering keys
- **Message Attributes**: Custom metadata with messages

### Consumption
- **Pull Message Retrieval**: On-demand message fetching
- **Acknowledgment Management**: Manual message acknowledgment
- **Batch Processing**: Process multiple messages efficiently
- **Flow Control**: Configure message processing rate

### Reliability Features
- **Dead Letter Topics**: Handle repeatedly failing messages
- **Retry Policies**: Configurable retry attempts (max 100)
- **Message Expiration**: Automatic message TTL
- **Exactly-Once Delivery**: Optional delivery guarantee

## Usage Example

```python
from pubsub import PubSubManager

# Initialize manager
mgr = PubSubManager(project_id='my-gcp-project')

# Create topic
topic = mgr.topic.create_topic('user-events', {
    'message_retention_duration': 86400  # 24 hours
})

# Create pull subscription
subscription = mgr.subscription.create_pull_subscription({
    'subscription_id': 'user-events-processor',
    'topic_id': 'user-events',
    'ack_deadline_seconds': 60
})

# Create dead letter subscription
dlq_subscription = mgr.subscription.create_dead_letter_subscription({
    'subscription_id': 'events-with-dlq',
    'topic_id': 'user-events',
    'dead_letter_topic_id': 'failed-events',
    'max_delivery_attempts': 5
})

# Publish messages with ordering
mgr.publisher.publish_with_ordering(
    messages=[{'user_id': '123', 'event': 'login'}],
    ordering_key='user-123'
)
```

## Best Practices

1. **Use batch publishing** for high-throughput scenarios
2. **Configure dead letter queues** for mission-critical messages
3. **Set appropriate acknowledgment deadlines** based on processing time
4. **Use message ordering** when sequence matters
5. **Monitor subscription backlogs** to prevent message buildup
6. **Implement idempotent consumers** for at-least-once delivery

## Requirements

```
google-cloud-pubsub
```

## Author

BrillConsulting - Enterprise Cloud Solutions
