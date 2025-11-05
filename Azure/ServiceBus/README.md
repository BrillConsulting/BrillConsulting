# Azure Service Bus Integration

Advanced implementation of Azure Service Bus with message queuing, publish-subscribe patterns, and enterprise messaging capabilities.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a comprehensive Python implementation for Azure Service Bus, featuring reliable message queuing, topic-based publish-subscribe messaging, dead-letter handling, scheduled messages, and session management. Built for enterprise applications requiring decoupled, reliable, and scalable messaging between distributed systems.

## Features

### Core Capabilities
- **Message Queuing**: FIFO message delivery with competing consumers
- **Publish-Subscribe**: Topic-based message distribution to multiple subscribers
- **Message Sessions**: Ordered message processing and state management
- **Dead-Letter Queues**: Automatic handling of problematic messages
- **Scheduled Messages**: Delay message delivery to specific times
- **Message Deferral**: Postpone message processing
- **Transactions**: Atomic message operations
- **Duplicate Detection**: Automatic duplicate message elimination

### Advanced Features
- **Auto-Forwarding**: Chain queues and topics
- **Message Batching**: Efficient batch send and receive
- **Peek-Lock Mode**: At-least-once delivery with visibility timeout
- **Receive-Delete Mode**: At-most-once delivery for performance
- **Message Properties**: Custom metadata and filtering
- **Correlation Filters**: Content-based routing
- **Time-to-Live (TTL)**: Automatic message expiration
- **Auto-Delete on Idle**: Automatic cleanup of unused entities

## Architecture

```
ServiceBus/
├── service_bus.py             # Main implementation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

### Key Components

1. **ServiceBusManager**: Main service interface
   - Queue and topic management
   - Message sending and receiving
   - Entity configuration

2. **Queue Operations**:
   - Send and receive messages
   - Batch operations
   - Session handling

3. **Topic Operations**:
   - Publish messages to topics
   - Manage subscriptions
   - Filter configuration

4. **Message Processing**:
   - Receive modes (Peek-Lock, Receive-Delete)
   - Dead-letter handling
   - Message deferral

5. **Session Management**:
   - Session-aware queues and subscriptions
   - State management
   - Ordered processing

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/ServiceBus

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your Azure Service Bus credentials:

```python
from service_bus import ServiceBusManager

manager = ServiceBusManager(
    connection_string="Endpoint=sb://your-namespace.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=your-key"
)
```

### Environment Variables (Recommended)

```bash
export SERVICE_BUS_CONNECTION_STRING="Endpoint=sb://your-namespace.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=your-key"
export SERVICE_BUS_QUEUE_NAME="your-queue-name"
export SERVICE_BUS_TOPIC_NAME="your-topic-name"
```

## Usage Examples

### 1. Queue Operations - Send and Receive

```python
from service_bus import ServiceBusManager

manager = ServiceBusManager(
    connection_string="your-connection-string"
)

# Create a queue
queue = manager.create_queue(
    queue_name="orders",
    max_size_in_megabytes=1024,
    default_message_time_to_live_in_seconds=3600,
    enable_dead_lettering=True
)

print(f"Queue created: {queue['name']}")

# Send a message
message = {
    "order_id": "12345",
    "customer": "John Doe",
    "items": ["Product A", "Product B"],
    "total": 99.99
}

message_id = manager.send_message(
    queue_name="orders",
    message_body=message,
    properties={
        "priority": "high",
        "region": "us-west"
    }
)

print(f"Message sent: {message_id}")

# Receive messages
received_messages = manager.receive_messages(
    queue_name="orders",
    max_messages=10,
    max_wait_time_seconds=5
)

for msg in received_messages:
    print(f"Order ID: {msg.body['order_id']}")
    print(f"Customer: {msg.body['customer']}")

    # Complete the message (remove from queue)
    manager.complete_message(msg)
    print("Message completed")
```

### 2. Batch Message Operations

```python
# Send messages in batch
messages = [
    {"order_id": f"order-{i}", "amount": 100 + i}
    for i in range(1, 101)
]

batch_result = manager.send_batch_messages(
    queue_name="orders",
    messages=messages
)

print(f"Sent {batch_result['success_count']} messages")
print(f"Failed: {batch_result['failure_count']}")

# Receive messages in batch
batch = manager.receive_message_batch(
    queue_name="orders",
    max_messages=50,
    max_wait_time_seconds=10
)

for message in batch:
    process_order(message.body)
    manager.complete_message(message)
```

### 3. Publish-Subscribe with Topics

```python
# Create a topic
topic = manager.create_topic(
    topic_name="notifications",
    max_size_in_megabytes=2048,
    enable_batched_operations=True
)

# Create subscriptions with filters
email_subscription = manager.create_subscription(
    topic_name="notifications",
    subscription_name="email-notifications",
    filter_rule={
        "type": "CorrelationFilter",
        "properties": {"channel": "email"}
    }
)

sms_subscription = manager.create_subscription(
    topic_name="notifications",
    subscription_name="sms-notifications",
    filter_rule={
        "type": "CorrelationFilter",
        "properties": {"channel": "sms"}
    }
)

# Publish messages to topic
email_message = {
    "type": "notification",
    "title": "Order Shipped",
    "body": "Your order has been shipped"
}

manager.publish_to_topic(
    topic_name="notifications",
    message_body=email_message,
    properties={"channel": "email", "priority": "normal"}
)

sms_message = {
    "type": "notification",
    "text": "Order shipped - Track: ABC123"
}

manager.publish_to_topic(
    topic_name="notifications",
    message_body=sms_message,
    properties={"channel": "sms", "priority": "high"}
)

# Receive from subscription
messages = manager.receive_from_subscription(
    topic_name="notifications",
    subscription_name="email-notifications",
    max_messages=10
)

for msg in messages:
    print(f"Email notification: {msg.body['title']}")
    manager.complete_message(msg)
```

### 4. Scheduled Messages

```python
from datetime import datetime, timedelta

# Schedule message for future delivery
scheduled_time = datetime.utcnow() + timedelta(hours=2)

message = {
    "reminder": "Meeting in 2 hours",
    "meeting_id": "meeting-123"
}

sequence_number = manager.schedule_message(
    queue_name="reminders",
    message_body=message,
    scheduled_enqueue_time=scheduled_time
)

print(f"Message scheduled with sequence number: {sequence_number}")

# Cancel scheduled message
manager.cancel_scheduled_message(
    queue_name="reminders",
    sequence_number=sequence_number
)

print("Scheduled message cancelled")
```

### 5. Dead-Letter Queue Handling

```python
# Receive messages from queue
messages = manager.receive_messages("orders", max_messages=10)

for msg in messages:
    try:
        # Process message
        process_order(msg.body)
        manager.complete_message(msg)
    except Exception as e:
        # Move to dead-letter queue with reason
        manager.dead_letter_message(
            message=msg,
            reason="ProcessingError",
            error_description=str(e)
        )
        print(f"Message moved to dead-letter queue: {e}")

# Process dead-letter messages
dead_letter_messages = manager.receive_dead_letter_messages(
    queue_name="orders",
    max_messages=10
)

for dlq_msg in dead_letter_messages:
    print(f"Dead-letter reason: {dlq_msg.dead_letter_reason}")
    print(f"Error: {dlq_msg.dead_letter_error_description}")

    # Attempt to reprocess or log for manual intervention
    if can_reprocess(dlq_msg):
        # Resend to main queue
        manager.send_message("orders", dlq_msg.body)
        manager.complete_message(dlq_msg)
```

### 6. Message Sessions

```python
# Create session-enabled queue
session_queue = manager.create_queue(
    queue_name="order-processing",
    requires_session=True,
    max_size_in_megabytes=1024
)

# Send messages with session ID
for i in range(1, 6):
    message = {
        "order_id": f"order-{i}",
        "customer_id": "customer-123",
        "step": i,
        "data": f"Step {i} data"
    }

    manager.send_message(
        queue_name="order-processing",
        message_body=message,
        session_id="customer-123"  # Group by customer
    )

# Receive messages from session
session_messages = manager.receive_session_messages(
    queue_name="order-processing",
    session_id="customer-123",
    max_messages=10
)

for msg in session_messages:
    print(f"Processing step {msg.body['step']} for {msg.body['customer_id']}")
    manager.complete_message(msg)

# Get and set session state
state = {"last_processed_step": 5, "timestamp": datetime.utcnow().isoformat()}
manager.set_session_state(
    queue_name="order-processing",
    session_id="customer-123",
    state=state
)

# Retrieve session state
retrieved_state = manager.get_session_state(
    queue_name="order-processing",
    session_id="customer-123"
)
print(f"Last processed step: {retrieved_state['last_processed_step']}")
```

### 7. Message Deferral and Peek

```python
# Peek at messages without receiving
peeked_messages = manager.peek_messages(
    queue_name="orders",
    max_messages=5
)

for msg in peeked_messages:
    print(f"Peeked message: {msg.body['order_id']}")

# Receive and defer messages
messages = manager.receive_messages("orders", max_messages=10)

for msg in messages:
    if requires_later_processing(msg):
        # Defer message for later
        sequence_number = manager.defer_message(msg)
        print(f"Message deferred: {sequence_number}")
    else:
        process_message(msg)
        manager.complete_message(msg)

# Receive deferred messages later
deferred_message = manager.receive_deferred_message(
    queue_name="orders",
    sequence_number=sequence_number
)

if deferred_message:
    process_message(deferred_message)
    manager.complete_message(deferred_message)
```

## Running Demos

```bash
# Run the implementation
python service_bus.py
```

Demo output includes:
- Queue creation and management
- Send and receive operations
- Topic and subscription setup
- Dead-letter queue handling
- Session management

## Receive Modes

### Peek-Lock Mode (Default)
```python
# Messages are locked for processing
# Must explicitly complete, abandon, or dead-letter
messages = manager.receive_messages(
    queue_name="orders",
    max_messages=10,
    receive_mode="PeekLock"
)

for msg in messages:
    try:
        process(msg)
        manager.complete_message(msg)
    except Exception:
        manager.abandon_message(msg)  # Return to queue
```

### Receive-Delete Mode
```python
# Messages are immediately removed upon receipt
# Higher performance, but no retry on failure
messages = manager.receive_messages(
    queue_name="orders",
    max_messages=10,
    receive_mode="ReceiveAndDelete"
)

for msg in messages:
    process(msg)  # No need to complete
```

## API Reference

### ServiceBusManager

#### Queue Methods

**`create_queue(queue_name, max_size_in_megabytes, default_message_time_to_live_in_seconds, enable_dead_lettering, requires_session)`**
- Creates a new queue
- **Parameters**: queue_name (str), max_size_in_megabytes (int), ttl (int), enable_dead_lettering (bool), requires_session (bool)
- **Returns**: `Dict[str, Any]`

**`delete_queue(queue_name)`**
- Deletes a queue
- **Returns**: `None`

**`list_queues()`**
- Lists all queues
- **Returns**: `List[Dict[str, Any]]`

**`get_queue_properties(queue_name)`**
- Gets queue properties and metrics
- **Returns**: `Dict[str, Any]`

#### Message Methods

**`send_message(queue_name, message_body, properties, session_id, time_to_live)`**
- Sends a message to a queue
- **Returns**: `str` (message_id)

**`send_batch_messages(queue_name, messages)`**
- Sends multiple messages in batch
- **Returns**: `Dict[str, int]`

**`receive_messages(queue_name, max_messages, max_wait_time_seconds, receive_mode)`**
- Receives messages from queue
- **Returns**: `List[Message]`

**`complete_message(message)`**
- Completes a message (removes from queue)
- **Returns**: `None`

**`abandon_message(message)`**
- Abandons a message (returns to queue)
- **Returns**: `None`

**`dead_letter_message(message, reason, error_description)`**
- Moves message to dead-letter queue
- **Returns**: `None`

**`defer_message(message)`**
- Defers message for later processing
- **Returns**: `int` (sequence_number)

**`receive_deferred_message(queue_name, sequence_number)`**
- Receives a deferred message
- **Returns**: `Message`

#### Topic Methods

**`create_topic(topic_name, max_size_in_megabytes, enable_batched_operations)`**
- Creates a new topic
- **Returns**: `Dict[str, Any]`

**`delete_topic(topic_name)`**
- Deletes a topic
- **Returns**: `None`

**`create_subscription(topic_name, subscription_name, filter_rule)`**
- Creates a subscription with optional filter
- **Returns**: `Dict[str, Any]`

**`delete_subscription(topic_name, subscription_name)`**
- Deletes a subscription
- **Returns**: `None`

**`publish_to_topic(topic_name, message_body, properties)`**
- Publishes message to topic
- **Returns**: `str` (message_id)

**`receive_from_subscription(topic_name, subscription_name, max_messages)`**
- Receives messages from subscription
- **Returns**: `List[Message]`

#### Scheduled Message Methods

**`schedule_message(queue_name, message_body, scheduled_enqueue_time)`**
- Schedules a message for future delivery
- **Returns**: `int` (sequence_number)

**`cancel_scheduled_message(queue_name, sequence_number)`**
- Cancels a scheduled message
- **Returns**: `None`

#### Session Methods

**`receive_session_messages(queue_name, session_id, max_messages)`**
- Receives messages from a specific session
- **Returns**: `List[Message]`

**`set_session_state(queue_name, session_id, state)`**
- Sets session state
- **Returns**: `None`

**`get_session_state(queue_name, session_id)`**
- Gets session state
- **Returns**: `Dict[str, Any]`

#### Dead-Letter Methods

**`receive_dead_letter_messages(queue_name, max_messages)`**
- Receives messages from dead-letter queue
- **Returns**: `List[Message]`

#### Peek Methods

**`peek_messages(queue_name, max_messages)`**
- Peeks at messages without receiving
- **Returns**: `List[Message]`

## Best Practices

### 1. Choose Appropriate Entity Type
```python
# Use queues for point-to-point communication
queue = manager.create_queue("orders")

# Use topics for publish-subscribe patterns
topic = manager.create_topic("notifications")
```

### 2. Implement Idempotent Message Handlers
```python
def process_message(msg):
    message_id = msg.message_id

    # Check if already processed
    if is_processed(message_id):
        return

    # Process message
    perform_operation(msg.body)

    # Mark as processed
    mark_processed(message_id)
```

### 3. Handle Poison Messages
```python
for msg in messages:
    try:
        if msg.delivery_count > 3:
            # Too many retries, move to dead-letter
            manager.dead_letter_message(
                msg,
                reason="MaxDeliveryCountExceeded",
                error_description="Failed after 3 attempts"
            )
        else:
            process_message(msg)
            manager.complete_message(msg)
    except Exception as e:
        manager.abandon_message(msg)
```

### 4. Use Message Properties for Routing
```python
# Send with properties
manager.send_message(
    queue_name="orders",
    message_body=order,
    properties={
        "priority": "high",
        "region": "us-west",
        "customer_type": "premium"
    }
)

# Filter in subscription
manager.create_subscription(
    topic_name="orders",
    subscription_name="premium-orders",
    filter_rule={
        "type": "CorrelationFilter",
        "properties": {"customer_type": "premium"}
    }
)
```

### 5. Implement Batching for High Throughput
```python
# Accumulate messages
message_batch = []
for order in orders:
    message_batch.append({"order_id": order.id, "data": order.data})

    if len(message_batch) >= 100:
        manager.send_batch_messages("orders", message_batch)
        message_batch = []

# Send remaining
if message_batch:
    manager.send_batch_messages("orders", message_batch)
```

### 6. Set Appropriate TTL
```python
# Short TTL for time-sensitive messages
manager.send_message(
    queue_name="notifications",
    message_body=notification,
    time_to_live_seconds=300  # 5 minutes
)

# Longer TTL for durable operations
manager.send_message(
    queue_name="orders",
    message_body=order,
    time_to_live_seconds=86400  # 24 hours
)
```

### 7. Monitor Queue Metrics
```python
# Get queue statistics
props = manager.get_queue_properties("orders")
print(f"Active messages: {props['active_message_count']}")
print(f"Dead-letter count: {props['dead_letter_message_count']}")
print(f"Scheduled messages: {props['scheduled_message_count']}")

# Alert if queue is growing
if props['active_message_count'] > 10000:
    send_alert("Queue backlog detected")
```

## Use Cases

### 1. Order Processing
Decouple order submission from order processing for scalability and resilience.

### 2. Event Distribution
Publish events to multiple subscribers for event-driven architectures.

### 3. Load Leveling
Queue requests during peak periods and process during off-peak hours.

### 4. Request-Reply Pattern
Implement asynchronous request-reply with correlation IDs and reply queues.

### 5. Workflow Orchestration
Chain multiple processing steps using auto-forwarding and session-aware queues.

### 6. Notification System
Distribute notifications across multiple channels (email, SMS, push) using topics.

## Troubleshooting

### Common Issues

**Issue**: Messages not being received
**Solution**: Check queue name, verify connection string, ensure messages haven't expired

**Issue**: Messages going to dead-letter queue
**Solution**: Check dead-letter reason, verify processing logic, increase max delivery count

**Issue**: High latency
**Solution**: Use batching, enable prefetch, increase concurrent receivers

**Issue**: Duplicate messages
**Solution**: Implement idempotent handlers, enable duplicate detection, use unique message IDs

**Issue**: Session lock timeout
**Solution**: Increase lock duration, process messages faster, renew session lock periodically

**Issue**: Throttling
**Solution**: Implement exponential backoff retry, use batching, upgrade service tier

## Deployment

### Azure CLI Deployment
```bash
# Create Service Bus namespace
az servicebus namespace create \
    --name my-servicebus \
    --resource-group my-resource-group \
    --location eastus \
    --sku Standard

# Create queue
az servicebus queue create \
    --name orders \
    --namespace-name my-servicebus \
    --resource-group my-resource-group \
    --max-size 1024 \
    --default-message-time-to-live P1D \
    --enable-dead-lettering-on-message-expiration true

# Create topic
az servicebus topic create \
    --name notifications \
    --namespace-name my-servicebus \
    --resource-group my-resource-group \
    --max-size 2048

# Create subscription
az servicebus topic subscription create \
    --name email-subscription \
    --topic-name notifications \
    --namespace-name my-servicebus \
    --resource-group my-resource-group

# Get connection string
az servicebus namespace authorization-rule keys list \
    --name RootManageSharedAccessKey \
    --namespace-name my-servicebus \
    --resource-group my-resource-group \
    --query primaryConnectionString -o tsv
```

### Infrastructure as Code
```python
# Terraform example
resource "azurerm_servicebus_namespace" "example" {
  name                = "my-servicebus"
  location            = azurerm_resource_group.example.location
  resource_group_name = azurerm_resource_group.example.name
  sku                 = "Standard"
}

resource "azurerm_servicebus_queue" "example" {
  name                = "orders"
  namespace_id        = azurerm_servicebus_namespace.example.id
  max_size_in_megabytes = 1024
  default_message_ttl = "P1D"
}
```

### Container Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY service_bus.py .
CMD ["python", "service_bus.py"]
```

## Monitoring

### Key Metrics
- Active message count
- Dead-letter message count
- Incoming and outgoing messages per second
- Message processing time
- Delivery attempt count
- Queue/topic size

### Azure Monitor Integration
```bash
# Enable diagnostic logs
az servicebus namespace update \
    --name my-servicebus \
    --resource-group my-resource-group \
    --enable-diagnostic-settings

# Configure alerts
az monitor metrics alert create \
    --name HighMessageCount \
    --resource-group my-resource-group \
    --scopes /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.ServiceBus/namespaces/{namespace}/queues/{queue} \
    --condition "avg ActiveMessages > 10000" \
    --window-size 5m \
    --evaluation-frequency 1m
```

### Application Insights Integration
```python
from applicationinsights import TelemetryClient

tc = TelemetryClient('instrumentation-key')

# Track message processing
tc.track_event('MessageProcessed', {
    'queue': 'orders',
    'processing_time': 150,
    'success': True
})

tc.flush()
```

## Dependencies

```
Python >= 3.8
azure-core >= 1.26.0
azure-servicebus >= 7.8.0
typing
datetime
json
```

See `requirements.txt` for complete list.

## Version History

- **v1.0.0**: Initial release with queue operations
- **v1.1.0**: Added topic and subscription support
- **v1.2.0**: Enhanced session management and dead-letter handling
- **v2.0.0**: Added scheduled messages and batch operations

## Contributing

Contributions are welcome! Please submit pull requests or open issues on GitHub.

## License

This project is part of the Brill Consulting portfolio.

## Support

For questions or support:
- **Email**: clientbrill@gmail.com
- **LinkedIn**: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure Event Hubs](../EventHubs/)
- [Azure Event Grid](../EventGrid/)
- [Azure Storage Queues](../StorageQueues/)

---

**Built with Azure Service Bus** | **Brill Consulting © 2024**
