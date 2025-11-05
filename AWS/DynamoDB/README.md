# AWS DynamoDB

Fully managed NoSQL database service with single-digit millisecond performance at any scale.

## Features

- **Tables**: Create and manage DynamoDB tables
- **Items**: CRUD operations for items
- **Queries**: Efficient queries using partition and sort keys
- **Scans**: Full table scans with filtering
- **Indexes**: Global and Local Secondary Indexes
- **Batch Operations**: Batch read/write for efficiency
- **Streams**: Capture table changes in real-time
- **Transactions**: ACID transactions across multiple items
- **Auto Scaling**: Automatic capacity adjustment
- **Point-in-Time Recovery**: Continuous backups

## Quick Start

```python
from aws_dynamodb import DynamoDBManager

# Initialize
dynamo = DynamoDBManager(region='us-east-1')

# Create table
dynamo.create_table(
    table_name='Users',
    partition_key=('user_id', 'S'),
    sort_key=('created_at', 'N'),
    billing_mode='PAY_PER_REQUEST'
)

# Put item
dynamo.put_item(
    table_name='Users',
    item={
        'user_id': 'user123',
        'created_at': 1699200000,
        'email': 'user@example.com',
        'name': 'John Doe',
        'status': 'active'
    }
)

# Get item
user = dynamo.get_item(
    table_name='Users',
    key={'user_id': 'user123', 'created_at': 1699200000}
)

# Query by partition key
users = dynamo.query(
    table_name='Users',
    key_condition_expression='user_id = :uid',
    expression_attribute_values={':uid': 'user123'}
)

# Update item
dynamo.update_item(
    table_name='Users',
    key={'user_id': 'user123', 'created_at': 1699200000},
    update_expression='SET #status = :status',
    expression_attribute_names={'#status': 'status'},
    expression_attribute_values={':status': 'inactive'}
)
```

## Use Cases

- **User Profiles**: Store user data with flexible schema
- **Session Management**: Fast session lookups
- **Gaming Leaderboards**: Real-time ranking with sort keys
- **Shopping Carts**: Temporary cart data storage
- **IoT Data**: Time-series data from devices
- **Mobile Backends**: Serverless mobile app data layer
- **Caching**: High-performance caching layer

## Table Design Patterns

### Single Table Design
Store multiple entity types in one table for optimal performance:
```python
# Users: PK=USER#123, SK=PROFILE
# Orders: PK=USER#123, SK=ORDER#456
# Products: PK=PRODUCT#789, SK=METADATA
```

### Composite Keys
```python
# Partition Key: customer_id
# Sort Key: order_date (for time-range queries)
```

### Secondary Indexes

**Global Secondary Index (GSI)**: Different partition/sort keys
```python
dynamo.create_table(
    table_name='Orders',
    partition_key=('order_id', 'S'),
    global_secondary_indexes=[{
        'IndexName': 'CustomerIndex',
        'Keys': [('customer_id', 'S'), ('order_date', 'N')],
        'Projection': {'ProjectionType': 'ALL'}
    }]
)
```

**Local Secondary Index (LSI)**: Same partition key, different sort key

## Batch Operations

```python
# Batch write (up to 25 items)
dynamo.batch_write_items(
    table_name='Users',
    items=[
        {'user_id': 'user1', 'name': 'Alice'},
        {'user_id': 'user2', 'name': 'Bob'},
        {'user_id': 'user3', 'name': 'Charlie'}
    ]
)

# Batch get (up to 100 items)
users = dynamo.batch_get_items(
    table_name='Users',
    keys=[
        {'user_id': 'user1'},
        {'user_id': 'user2'}
    ]
)
```

## Transactions

```python
# Transactional write
dynamo.transact_write_items([
    {
        'Put': {
            'TableName': 'Orders',
            'Item': {'order_id': '123', 'amount': 100}
        }
    },
    {
        'Update': {
            'TableName': 'Users',
            'Key': {'user_id': 'user123'},
            'UpdateExpression': 'SET balance = balance - :amount',
            'ExpressionAttributeValues': {':amount': 100}
        }
    }
])
```

## DynamoDB Streams

Capture changes to track data modifications:
```python
# Enable streams
dynamo.update_table(
    table_name='Users',
    stream_specification={
        'StreamEnabled': True,
        'StreamViewType': 'NEW_AND_OLD_IMAGES'
    }
)
```

## Best Practices

- Use partition keys with high cardinality
- Design tables to avoid hot partitions
- Use batch operations for multiple items
- Enable auto-scaling for variable workloads
- Use sparse indexes to save storage
- Implement TTL for automatic data expiration
- Use projection expressions to fetch only needed attributes
- Consider reserved capacity for predictable workloads

## Author

Brill Consulting
