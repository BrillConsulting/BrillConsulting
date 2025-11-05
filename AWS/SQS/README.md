# AWS SQS (Simple Queue Service)

Fully managed message queuing service for decoupling and scaling microservices, distributed systems, and serverless applications.

## Features

- **Queue Types**: Standard (high throughput) and FIFO (ordered delivery)
- **At-Least-Once Delivery**: Reliable message delivery (Standard queues)
- **Exactly-Once Processing**: Deduplication (FIFO queues)
- **Batch Operations**: Send/receive/delete up to 10 messages at once
- **Dead-Letter Queues**: Handle failed messages automatically
- **Long Polling**: Reduce costs with wait-time-based message retrieval
- **Visibility Timeout**: Prevent duplicate processing
- **Message Attributes**: Custom metadata for routing and filtering
- **Delay Queues**: Postpone message delivery

## Quick Start

```python
from aws_sqs import SQSManager

# Initialize
sqs = SQSManager(region='us-east-1')

# Create queue with DLQ
dlq = sqs.create_queue('my-dlq')
dlq_attrs = sqs.get_queue_attributes(dlq['queue_url'])

queue = sqs.create_queue(
    queue_name='my-queue',
    visibility_timeout=60,
    receive_message_wait_time=10,  # Long polling
    dead_letter_queue_arn=dlq_attrs['queue_arn'],
    max_receive_count=3
)

# Send message
sqs.send_message(
    queue_url=queue['queue_url'],
    message_body='Hello from SQS!',
    message_attributes={'priority': 'high', 'customer_id': '12345'}
)

# Receive and process messages
messages = sqs.receive_messages(
    queue_url=queue['queue_url'],
    max_number_of_messages=10,
    wait_time_seconds=20  # Long polling
)

for msg in messages:
    print(f"Processing: {msg['body']}")
    # Delete after processing
    sqs.delete_message(queue['queue_url'], msg['receipt_handle'])
```

## Use Cases

- **Application Decoupling**: Separate components for independent scaling
- **Work Queues**: Distribute tasks across workers
- **Load Leveling**: Buffer requests during traffic spikes
- **Event Buffering**: Queue events for batch processing
- **Microservices Communication**: Async messaging between services
- **Order Processing**: FIFO queues for ordered workflows

## Queue Types

### Standard Queue
- Unlimited throughput
- At-least-once delivery
- Best-effort ordering
- Use for high-throughput scenarios

### FIFO Queue
- 300 TPS (3000 with batching)
- Exactly-once processing
- Strict ordering
- Use for ordered workflows

## Long Polling

Save costs by reducing empty receives:

```python
messages = sqs.receive_messages(
    queue_url=queue_url,
    wait_time_seconds=20  # Wait up to 20 seconds
)
```

## Dead-Letter Queues

Automatically handle failed messages:

```python
sqs.create_queue(
    queue_name='my-queue',
    dead_letter_queue_arn=dlq_arn,
    max_receive_count=3  # Move to DLQ after 3 failures
)
```

## Author

Brill Consulting
