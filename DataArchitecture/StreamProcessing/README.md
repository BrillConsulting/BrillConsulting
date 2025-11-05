# Stream Processing
Real-time stream processing with windowing and stateful operations

## Overview

A production-grade stream processing framework providing real-time data processing capabilities with windowing, stateful operations, and event-time handling. Features tumbling and sliding windows, watermarks for late data handling, stream joins, and comprehensive state management for building robust real-time data pipelines.

## Features

### Core Capabilities
- **Stream Creation**: Register and manage event streams
- **Event Publishing**: Publish events to streams with timestamps
- **Stream Processors**: Register custom processing functions
- **Windowing Operations**: Tumbling and sliding windows
- **Window Aggregations**: Aggregate events within windows
- **State Management**: Persist and retrieve stateful data
- **Event Buffering**: Buffer events for batch processing

### Advanced Features
- **Event Time Processing**: Use event timestamps vs processing time
- **Watermarks**: Handle late-arriving events gracefully
- **Stream Joins**: Join events from multiple streams
- **Stateful Processing**: Maintain state across events
- **Batch Processing**: Process events in configurable batches
- **Stream Metrics**: Track processing statistics and performance
- **Window Management**: Automatic window lifecycle management

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/DataArchitecture.git
cd DataArchitecture/StreamProcessing

# Install dependencies
pip install pandas

# Run the implementation
python streamprocessing.py
```

## Usage Examples

### Create Stream

```python
from streamprocessing import StreamProcessing, Event
from datetime import datetime

# Initialize stream processing engine
sp = StreamProcessing()

# Create event stream
sp.create_stream("user-events", {
    "type": "user_activity"
})

print("Stream created: user-events")
```

### Register Processor

```python
def count_events(event: Event) -> dict:
    """Count events by type."""
    current = sp.get_state(event.event_type, namespace="counters") or 0
    sp.update_state(event.event_type, current + 1, namespace="counters")
    return {"event_type": event.event_type, "count": current + 1}

# Register processor
sp.register_processor(
    processor_id="event_counter",
    stream_id="user-events",
    process_fn=count_events
)

print("Processor registered: event_counter")
```

### Publish Events

```python
from datetime import timedelta

# Publish events with event time
base_time = datetime.now()

for i in range(10):
    event = Event(
        event_id=f"event_{i}",
        event_type="page_view" if i % 2 == 0 else "button_click",
        data={"user_id": f"user_{i % 3}", "page": f"/page{i}"},
        event_time=base_time + timedelta(seconds=i * 30)
    )
    sp.publish_event("user-events", event)

print("Published 10 events")
```

### Create Tumbling Window

```python
# Create 5-minute tumbling window
sp.create_tumbling_window(
    window_id="5min_tumbling",
    stream_id="user-events",
    window_size_seconds=300
)

print("Tumbling window created: 5 minutes, non-overlapping")
```

### Create Sliding Window

```python
# Create 10-minute sliding window with 5-minute slide
sp.create_sliding_window(
    window_id="10min_sliding",
    stream_id="user-events",
    window_size_seconds=600,
    slide_size_seconds=300
)

print("Sliding window created: 10 min window, 5 min slide")
```

### Assign Events to Windows

```python
# Get event from buffer
if sp.event_buffer["user-events"]:
    event = sp.event_buffer["user-events"][0]

    # Assign to tumbling window
    windows = sp.assign_to_window("5min_tumbling", event)

    print(f"Event assigned to {len(windows)} window(s)")
```

### Window Aggregation

```python
from collections import defaultdict

def count_by_type(events: list) -> dict:
    """Count events by type in window."""
    counts = defaultdict(int)
    for event in events:
        counts[event.event_type] += 1
    return dict(counts)

# Get current window
tumbling_config = sp.windows["5min_tumbling"]
if tumbling_config.get("current_window"):
    window = tumbling_config["current_window"]

    # Aggregate events in window
    result = sp.aggregate_window(window, count_by_type)

    print(f"Window aggregation result: {result}")
```

### State Management

```python
# Update state
sp.update_state("total_events", 1000, namespace="stats")
sp.update_state("total_users", 50, namespace="stats")

# Retrieve state
total_events = sp.get_state("total_events", namespace="stats")
total_users = sp.get_state("total_users", namespace="stats")

print(f"Total events: {total_events}")
print(f"Total users: {total_users}")
```

### Set Watermarks

```python
# Set watermark for handling late data
watermark_time = datetime.now() - timedelta(minutes=5)
sp.set_watermark("user-events", watermark_time)

print(f"Watermark set: {watermark_time.isoformat()}")
print("Events before watermark are considered late")
```

### Stream Joins

```python
# Create second stream
sp.create_stream("transaction-events", {
    "type": "transactions"
})

# Create stream join
join = sp.join_streams(
    join_id="user_transaction_join",
    stream1_id="user-events",
    stream2_id="transaction-events",
    join_key="user_id",
    window_seconds=300  # 5-minute join window
)

print(f"Stream join created: {join['join_id']}")
print(f"Join key: {join['join_key']}")
print(f"Window: {join['window_seconds']}s")
```

### Batch Processing

```python
# Process events in batches
batch_result = sp.process_stream_batch(
    stream_id="user-events",
    batch_size=5
)

print(f"Processed: {batch_result['processed_count']} events")
print(f"Remaining: {batch_result['remaining_count']} events")
```

### Get Stream Metrics

```python
# Get processing metrics
metrics = sp.get_stream_metrics("user-events")

print(f"Stream Metrics:")
print(f"  Total events: {metrics['total_events']}")
print(f"  Buffer size: {metrics['buffer_size']}")
print(f"  Processors: {metrics['processor_count']}")

print("\nProcessor Details:")
for proc in metrics["processors"]:
    print(f"  {proc['processor_id']}: {proc['processed_count']} events")
```

### Generate Processing Report

```python
# Get comprehensive processing report
report = sp.generate_processing_report()

print("Stream Processing Report:")
print(f"  Total Streams: {report['summary']['total_streams']}")
print(f"  Total Processors: {report['summary']['total_processors']}")
print(f"  Total Windows: {report['summary']['total_windows']}")
print(f"  Processed Events: {report['summary']['total_processed_events']}")

print("\nStream Details:")
for stream in report["streams"]:
    print(f"  {stream['stream_id']}:")
    print(f"    Events: {stream['total_events']}")
    print(f"    Buffered: {stream['buffer_size']}")
```

## Demo Instructions

Run the included demonstration:

```bash
python streamprocessing.py
```

The demo showcases:
1. Creating multiple event streams
2. Registering stream processors
3. Creating tumbling and sliding windows
4. Publishing events to streams
5. Assigning events to windows
6. Window aggregation operations
7. State management and retrieval
8. Setting watermarks for late data
9. Creating stream joins
10. Batch processing
11. Stream metrics and monitoring
12. Comprehensive processing reports

## Key Concepts

### Windows

Time-based grouping of events:
- **Tumbling Windows**: Fixed-size, non-overlapping windows
  - Example: 5-minute windows [0-5, 5-10, 10-15...]
  - Each event belongs to exactly one window
  - Good for aggregations over discrete time periods

- **Sliding Windows**: Fixed-size, overlapping windows
  - Example: 10-min window, 5-min slide [0-10, 5-15, 10-20...]
  - Events can belong to multiple windows
  - Good for moving averages and rolling statistics

- **Session Windows**: Gap-based windows (future feature)
  - Windows defined by inactivity gaps
  - Good for user session analysis

### Event Time vs Processing Time

- **Event Time**: When the event actually occurred
  - More accurate for analysis
  - Requires watermarks for completeness
  - Handles out-of-order events correctly

- **Processing Time**: When event is processed
  - Simpler to implement
  - Lower latency
  - May miss late events

### Watermarks

Mechanism for handling late data:
- Track progress of event time
- Signal when window can be closed
- Balance completeness vs. latency
- Allow late data within tolerance

### Stateful Processing

Maintain state across events:
- **Per-Key State**: Different state per key
- **Global State**: Shared across all events
- **Window State**: State within windows
- **Persistent**: Survives failures

## Architecture

```
┌────────────────────────────────────────────┐
│     Stream Processing Engine               │
│                                            │
│  ┌─────────────────────────────────┐      │
│  │  Event Streams                  │      │
│  │  - user-events                  │      │
│  │  - transaction-events           │      │
│  └──────────┬──────────────────────┘      │
│             │                              │
│             ▼                              │
│  ┌─────────────────────────────────┐      │
│  │  Stream Processors              │      │
│  │  - Event Counter                │      │
│  │  - Filter                       │      │
│  │  - Transformer                  │      │
│  └──────────┬──────────────────────┘      │
│             │                              │
│             ▼                              │
│  ┌─────────────────────────────────┐      │
│  │  Windowing Engine               │      │
│  │  - Tumbling: [0-5] [5-10]       │      │
│  │  - Sliding: [0-10] [5-15]       │      │
│  └──────────┬──────────────────────┘      │
│             │                              │
│             ▼                              │
│  ┌─────────────────────────────────┐      │
│  │  Aggregation & State            │      │
│  │  - Window Aggregations          │      │
│  │  - State Stores                 │      │
│  │  - Watermarks                   │      │
│  └──────────┬──────────────────────┘      │
│             │                              │
│             ▼                              │
│  ┌─────────────────────────────────┐      │
│  │  Output Streams                 │      │
│  │  - Processed Events             │      │
│  │  - Aggregation Results          │      │
│  └─────────────────────────────────┘      │
└────────────────────────────────────────────┘
```

## Use Cases

- **Real-Time Analytics**: Live dashboards and metrics
- **Fraud Detection**: Detect suspicious patterns instantly
- **User Behavior Analysis**: Track user activity in real-time
- **IoT Processing**: Process sensor data streams
- **Log Aggregation**: Aggregate and analyze logs
- **Alerting**: Trigger alerts on specific patterns
- **Recommendation Engines**: Real-time recommendations
- **A/B Testing**: Live experiment analysis

## Best Practices

- Use event time for accurate analysis
- Set appropriate window sizes for use case
- Configure watermarks based on late data tolerance
- Implement idempotent processors
- Monitor state store size
- Use appropriate parallelism
- Handle backpressure properly
- Test with out-of-order events

## Performance Considerations

- Window size affects memory usage
- State stores require proper sizing
- Batch processing reduces overhead
- Watermark intervals impact completeness
- Consider partitioning for scalability
- Monitor buffer sizes
- Tune checkpoint intervals

## Window Selection Guide

### Use Tumbling Windows When:
- Discrete time period analysis needed
- Aggregating metrics (hourly counts, daily totals)
- Each event should belong to one window
- Simpler semantics required

### Use Sliding Windows When:
- Moving averages needed
- Rolling statistics required
- Recent activity analysis
- Smoothed metrics desired

### Use Session Windows When:
- User sessions to be tracked
- Activity bursts to be grouped
- Gap-based grouping needed
- Variable-length windows required

## Integration

Integrate with:
- Apache Kafka for event streaming
- Apache Flink for distributed processing
- Apache Spark Streaming
- Time-series databases (InfluxDB, TimescaleDB)
- Monitoring systems (Prometheus, Grafana)
- Alert platforms (PagerDuty)

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [linkedin.com/in/brillconsulting](https://linkedin.com/in/brillconsulting)
- Specialization: Data Architecture & Engineering Solutions
