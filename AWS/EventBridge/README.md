# AWS EventBridge

Serverless event bus for building scalable event-driven architectures and connecting applications.

## Features

- **Event Buses**: Default and custom event buses for routing
- **Rules**: Event pattern matching and routing logic
- **Targets**: Lambda, SNS, SQS, Step Functions, Kinesis, and more
- **Event Patterns**: Filter events based on content
- **Scheduled Rules**: Cron and rate-based event generation
- **Archives**: Store and replay events
- **API Destinations**: Send events to HTTP endpoints
- **Schema Registry**: Discover and validate event schemas
- **Cross-Account Events**: Send events across AWS accounts
- **SaaS Integration**: Connect to 90+ SaaS applications

## Quick Start

```python
from aws_eventbridge import EventBridgeManager
import json

# Initialize
eb = EventBridgeManager(region='us-east-1')

# Create custom event bus
bus = eb.create_event_bus('my-app-events')

# Create rule with event pattern
rule = eb.put_rule(
    name='user-signup-rule',
    event_pattern={
        'source': ['my.application'],
        'detail-type': ['User Action'],
        'detail': {
            'action': ['signup']
        }
    },
    event_bus_name='my-app-events',
    description='Trigger on user signup'
)

# Add Lambda target
eb.put_targets(
    rule_name='user-signup-rule',
    event_bus_name='my-app-events',
    targets=[{
        'Id': '1',
        'Arn': 'arn:aws:lambda:us-east-1:123456789012:function:process-signup',
        'Input': json.dumps({'environment': 'production'})
    }]
)

# Send custom event
eb.put_events([{
    'Source': 'my.application',
    'DetailType': 'User Action',
    'Detail': json.dumps({
        'action': 'signup',
        'user': 'john@example.com',
        'timestamp': '2024-11-05T10:30:00Z'
    }),
    'EventBusName': 'my-app-events'
}])

# Create scheduled rule (every 5 minutes)
eb.put_rule(
    name='backup-rule',
    schedule_expression='rate(5 minutes)',
    description='Run backup every 5 minutes'
)
```

## Use Cases

- **Microservices Decoupling**: Async communication between services
- **Application Integration**: Connect AWS services and SaaS apps
- **Event-Driven Workflows**: Trigger actions based on events
- **Scheduled Tasks**: Run jobs on cron schedules
- **Audit Logging**: Capture and route system events
- **Data Pipelines**: Trigger ETL workflows on data changes
- **Monitoring**: React to CloudWatch alarms and metrics

## Event Patterns

Match events based on content:

### Exact Match
```python
{
    'source': ['my.application'],
    'detail-type': ['User Action']
}
```

### Prefix Match
```python
{
    'source': [{'prefix': 'aws.'}]  # Match all AWS services
}
```

### Numeric Match
```python
{
    'detail': {
        'price': [{'numeric': ['>', 100]}]  # Price > 100
    }
}
```

### Array Match
```python
{
    'detail': {
        'tags': [{'exists': True}]  # Tags field exists
    }
}
```

### Complex Patterns
```python
{
    'source': ['my.application'],
    'detail-type': ['Order'],
    'detail': {
        'status': ['completed', 'shipped'],
        'amount': [{'numeric': ['>=', 1000]}],
        'region': ['us-east-1', 'us-west-2']
    }
}
```

## Scheduled Rules

### Rate Expressions
- `rate(5 minutes)` - Every 5 minutes
- `rate(1 hour)` - Every hour
- `rate(1 day)` - Every day

### Cron Expressions
- `cron(0 9 * * ? *)` - 9 AM UTC daily
- `cron(0 12 ? * MON-FRI *)` - Noon on weekdays
- `cron(0/15 * * * ? *)` - Every 15 minutes

## Targets

EventBridge can send events to:
- AWS Lambda functions
- SNS topics
- SQS queues
- Step Functions state machines
- Kinesis streams
- ECS tasks
- CodePipeline
- API Gateway
- HTTP endpoints (API Destinations)
- And 20+ other AWS services

## Input Transformation

Transform event data before sending to target:

```python
eb.put_targets(
    rule_name='my-rule',
    targets=[{
        'Id': '1',
        'Arn': 'arn:aws:lambda:...:function:processor',
        'InputTransformer': {
            'InputPathsMap': {
                'user': '$.detail.user',
                'action': '$.detail.action'
            },
            'InputTemplate': '{"username": <user>, "event": <action>}'
        }
    }]
)
```

## Event Archives

Store and replay events:

```python
# Create archive
eb.create_archive(
    archive_name='user-events-archive',
    source_arn='arn:aws:events:us-east-1:123456789012:event-bus/default',
    retention_days=30
)

# Replay events
eb.start_replay(
    replay_name='user-events-replay',
    source_arn='archive-arn',
    start_time=datetime(2024, 11, 1),
    end_time=datetime(2024, 11, 5)
)
```

## Best Practices

- Use meaningful source and detail-type values
- Design event patterns to be specific
- Use dead-letter queues for failed targets
- Enable archives for critical event buses
- Monitor rule execution metrics
- Use input transformers to reduce payload size
- Implement idempotent targets
- Tag resources for organization

## Author

Brill Consulting
