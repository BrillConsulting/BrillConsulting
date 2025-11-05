# AWS SNS (Simple Notification Service)

Fully managed pub/sub messaging service for application-to-application (A2A) and application-to-person (A2P) communication.

## Features

- **Topics**: Standard and FIFO topics for pub/sub messaging
- **Subscriptions**: Email, SMS, HTTP/HTTPS, SQS, Lambda, mobile push
- **Message Filtering**: Subscriber-based filtering with filter policies
- **Batch Publishing**: Send up to 10 messages in a single request
- **SMS Messaging**: Send text messages globally
- **Mobile Push**: iOS, Android, and other platforms
- **Dead-Letter Queues**: Handle message delivery failures
- **Message Attributes**: Custom metadata for routing and filtering

## Quick Start

```python
from aws_sns import SNSManager

# Initialize
sns = SNSManager(region='us-east-1')

# Create topic
topic = sns.create_topic(name='notifications', display_name='App Notifications')

# Subscribe email
sns.subscribe(
    topic_arn=topic['topic_arn'],
    protocol='email',
    endpoint='alerts@example.com'
)

# Subscribe Lambda with filter
sns.subscribe(
    topic_arn=topic['topic_arn'],
    protocol='lambda',
    endpoint='arn:aws:lambda:us-east-1:123456789012:function:processor',
    filter_policy={'event_type': ['order_placed']}
)

# Publish message
sns.publish(
    topic_arn=topic['topic_arn'],
    message='New order received!',
    subject='Order Alert',
    message_attributes={'event_type': 'order_placed', 'priority': 'high'}
)

# Send SMS
sns.send_sms(
    phone_number='+1234567890',
    message='Your code: 123456',
    sms_type='Transactional'
)
```

## Use Cases

- **Application Alerts**: System notifications and alerts
- **Fan-Out Pattern**: Broadcast to multiple subscribers (SQS, Lambda)
- **User Notifications**: Email and SMS to end users
- **Mobile Push**: Push notifications to mobile devices
- **Event-Driven Architectures**: Decouple microservices
- **Monitoring Alerts**: CloudWatch alarms and health checks

## Message Filtering

Filter messages at subscription level:

```python
filter_policy = {
    'event_type': ['order_placed', 'order_shipped'],
    'price': [{'numeric': ['>', 100]}],
    'region': ['us-east-1', 'us-west-2']
}
```

## FIFO Topics

Ordered message delivery with deduplication:

```python
sns.create_topic(
    name='orders.fifo',
    fifo_topic=True,
    content_based_deduplication=True
)
```

## Author

Brill Consulting
