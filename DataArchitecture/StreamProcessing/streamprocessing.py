"""
Stream Processing Framework
===========================

Real-time stream processing with windowing and state management:
- Event stream ingestion and processing
- Windowing (tumbling, sliding, session)
- Stateful processing
- Stream aggregations
- Event time processing
- Watermarks and late data handling
- Stream joins

Author: Brill Consulting
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
from enum import Enum
import time


class WindowType(Enum):
    """Window types for stream processing."""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"


class Event:
    """Represents a stream event."""

    def __init__(self, event_id: str, event_type: str, data: Dict,
                 event_time: Optional[datetime] = None):
        """Initialize event."""
        self.event_id = event_id
        self.event_type = event_type
        self.data = data
        self.event_time = event_time or datetime.now()
        self.processing_time = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "event_time": self.event_time.isoformat(),
            "processing_time": self.processing_time.isoformat()
        }


class Window:
    """Represents a processing window."""

    def __init__(self, window_id: str, start_time: datetime,
                 end_time: datetime, window_type: str):
        """Initialize window."""
        self.window_id = window_id
        self.start_time = start_time
        self.end_time = end_time
        self.window_type = window_type
        self.events = []
        self.state = {}
        self.is_closed = False

    def add_event(self, event: Event):
        """Add event to window."""
        if not self.is_closed:
            self.events.append(event)

    def close(self):
        """Close the window."""
        self.is_closed = True

    def get_summary(self) -> Dict:
        """Get window summary."""
        return {
            "window_id": self.window_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "window_type": self.window_type,
            "event_count": len(self.events),
            "is_closed": self.is_closed,
            "state": self.state
        }


class StreamProcessing:
    """Stream processing engine with windowing and state management."""

    def __init__(self):
        """Initialize stream processing engine."""
        self.streams = {}
        self.windows = {}
        self.processors = {}
        self.state_stores = defaultdict(dict)
        self.watermarks = {}
        self.event_buffer = defaultdict(deque)
        self.processed_events = []

    def create_stream(self, stream_id: str, config: Optional[Dict] = None) -> Dict:
        """Create a new event stream."""
        print(f"Creating stream: {stream_id}")

        stream = {
            "stream_id": stream_id,
            "config": config or {},
            "created_at": datetime.now().isoformat(),
            "event_count": 0,
            "processors": []
        }

        self.streams[stream_id] = stream

        print(f"✓ Stream created: {stream_id}")

        return stream

    def publish_event(self, stream_id: str, event: Event) -> Dict:
        """Publish event to stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")

        # Add to buffer
        self.event_buffer[stream_id].append(event)

        # Update stream stats
        self.streams[stream_id]["event_count"] += 1

        # Process event through registered processors
        for processor_id in self.streams[stream_id]["processors"]:
            self._process_event(processor_id, event)

        return {
            "stream_id": stream_id,
            "event_id": event.event_id,
            "published_at": datetime.now().isoformat()
        }

    def register_processor(self, processor_id: str, stream_id: str,
                          process_fn: Callable, config: Optional[Dict] = None) -> Dict:
        """Register a stream processor."""
        print(f"Registering processor: {processor_id}")

        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")

        processor = {
            "processor_id": processor_id,
            "stream_id": stream_id,
            "process_fn": process_fn,
            "config": config or {},
            "processed_count": 0,
            "created_at": datetime.now().isoformat()
        }

        self.processors[processor_id] = processor
        self.streams[stream_id]["processors"].append(processor_id)

        print(f"✓ Processor registered: {processor_id}")

        return processor

    def _process_event(self, processor_id: str, event: Event):
        """Process event through processor."""
        processor = self.processors[processor_id]
        process_fn = processor["process_fn"]

        try:
            # Execute processing function
            result = process_fn(event)

            processor["processed_count"] += 1
            self.processed_events.append({
                "processor_id": processor_id,
                "event_id": event.event_id,
                "result": result,
                "processed_at": datetime.now().isoformat()
            })

        except Exception as e:
            print(f"Error processing event {event.event_id}: {str(e)}")

    def create_tumbling_window(self, window_id: str, stream_id: str,
                              window_size_seconds: int) -> Dict:
        """Create tumbling window (fixed-size, non-overlapping)."""
        print(f"Creating tumbling window: {window_id}")

        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")

        window_config = {
            "window_id": window_id,
            "stream_id": stream_id,
            "window_type": WindowType.TUMBLING.value,
            "window_size_seconds": window_size_seconds,
            "current_window": None,
            "completed_windows": []
        }

        self.windows[window_id] = window_config

        print(f"✓ Tumbling window created: {window_size_seconds}s")

        return window_config

    def create_sliding_window(self, window_id: str, stream_id: str,
                             window_size_seconds: int,
                             slide_size_seconds: int) -> Dict:
        """Create sliding window (fixed-size, overlapping)."""
        print(f"Creating sliding window: {window_id}")

        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")

        window_config = {
            "window_id": window_id,
            "stream_id": stream_id,
            "window_type": WindowType.SLIDING.value,
            "window_size_seconds": window_size_seconds,
            "slide_size_seconds": slide_size_seconds,
            "windows": []
        }

        self.windows[window_id] = window_config

        print(f"✓ Sliding window created: {window_size_seconds}s window, {slide_size_seconds}s slide")

        return window_config

    def assign_to_window(self, window_id: str, event: Event) -> List[Window]:
        """Assign event to window(s)."""
        if window_id not in self.windows:
            raise ValueError(f"Window {window_id} not found")

        window_config = self.windows[window_id]
        window_type = window_config["window_type"]
        assigned_windows = []

        if window_type == WindowType.TUMBLING.value:
            window = self._get_or_create_tumbling_window(window_config, event.event_time)
            window.add_event(event)
            assigned_windows.append(window)

        elif window_type == WindowType.SLIDING.value:
            windows = self._get_sliding_windows(window_config, event.event_time)
            for window in windows:
                window.add_event(event)
                assigned_windows.append(window)

        return assigned_windows

    def _get_or_create_tumbling_window(self, config: Dict,
                                       event_time: datetime) -> Window:
        """Get or create tumbling window for event time."""
        window_size = timedelta(seconds=config["window_size_seconds"])

        # Calculate window boundaries
        window_start = datetime(
            event_time.year, event_time.month, event_time.day,
            event_time.hour, event_time.minute,
            (event_time.second // config["window_size_seconds"]) *
            config["window_size_seconds"]
        )
        window_end = window_start + window_size

        # Check if current window exists
        current = config.get("current_window")
        if current and current.start_time == window_start:
            return current

        # Close current window if it exists
        if current:
            current.close()
            config["completed_windows"].append(current)

        # Create new window
        window_id = f"{config['window_id']}_{window_start.isoformat()}"
        new_window = Window(
            window_id=window_id,
            start_time=window_start,
            end_time=window_end,
            window_type=WindowType.TUMBLING.value
        )

        config["current_window"] = new_window
        return new_window

    def _get_sliding_windows(self, config: Dict, event_time: datetime) -> List[Window]:
        """Get sliding windows for event time."""
        window_size = timedelta(seconds=config["window_size_seconds"])
        slide_size = timedelta(seconds=config["slide_size_seconds"])

        # Find all windows that should contain this event
        matching_windows = []

        # Generate window start times that could contain this event
        num_windows = config["window_size_seconds"] // config["slide_size_seconds"]

        for i in range(num_windows):
            potential_start = event_time - timedelta(seconds=i * config["slide_size_seconds"])
            potential_start = potential_start.replace(microsecond=0)
            potential_end = potential_start + window_size

            if potential_start <= event_time < potential_end:
                # Find or create window
                window = self._find_or_create_sliding_window(
                    config, potential_start, potential_end
                )
                matching_windows.append(window)

        return matching_windows

    def _find_or_create_sliding_window(self, config: Dict,
                                       start_time: datetime,
                                       end_time: datetime) -> Window:
        """Find or create sliding window."""
        # Look for existing window
        for window in config["windows"]:
            if window.start_time == start_time:
                return window

        # Create new window
        window_id = f"{config['window_id']}_{start_time.isoformat()}"
        new_window = Window(
            window_id=window_id,
            start_time=start_time,
            end_time=end_time,
            window_type=WindowType.SLIDING.value
        )

        config["windows"].append(new_window)
        return new_window

    def aggregate_window(self, window: Window, aggregation_fn: Callable) -> Any:
        """Apply aggregation function to window."""
        print(f"Aggregating window: {window.window_id}")

        if not window.events:
            return None

        result = aggregation_fn(window.events)

        window.state["aggregation_result"] = result
        window.state["aggregated_at"] = datetime.now().isoformat()

        print(f"✓ Aggregated {len(window.events)} events")

        return result

    def update_state(self, key: str, value: Any, namespace: str = "default"):
        """Update state store."""
        self.state_stores[namespace][key] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }

    def get_state(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from state store."""
        state = self.state_stores[namespace].get(key)
        return state["value"] if state else None

    def set_watermark(self, stream_id: str, watermark: datetime):
        """Set watermark for stream (for handling late data)."""
        self.watermarks[stream_id] = watermark
        print(f"✓ Watermark set for {stream_id}: {watermark.isoformat()}")

    def get_watermark(self, stream_id: str) -> Optional[datetime]:
        """Get current watermark for stream."""
        return self.watermarks.get(stream_id)

    def join_streams(self, join_id: str, stream1_id: str, stream2_id: str,
                    join_key: str, window_seconds: int) -> Dict:
        """Join two streams within a time window."""
        print(f"Creating stream join: {join_id}")

        if stream1_id not in self.streams or stream2_id not in self.streams:
            raise ValueError("One or both streams not found")

        join_config = {
            "join_id": join_id,
            "stream1_id": stream1_id,
            "stream2_id": stream2_id,
            "join_key": join_key,
            "window_seconds": window_seconds,
            "matches": [],
            "created_at": datetime.now().isoformat()
        }

        # Store in state for tracking
        self.update_state(f"join_{join_id}", join_config, namespace="joins")

        print(f"✓ Stream join created")
        print(f"  Streams: {stream1_id} ⋈ {stream2_id}")
        print(f"  Window: {window_seconds}s")

        return join_config

    def process_stream_batch(self, stream_id: str, batch_size: int = 100) -> Dict:
        """Process a batch of events from stream buffer."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")

        buffer = self.event_buffer[stream_id]
        processed = 0
        batch_events = []

        while buffer and processed < batch_size:
            event = buffer.popleft()
            batch_events.append(event)
            processed += 1

        result = {
            "stream_id": stream_id,
            "processed_count": processed,
            "remaining_count": len(buffer),
            "processed_at": datetime.now().isoformat()
        }

        return result

    def get_stream_metrics(self, stream_id: str) -> Dict:
        """Get metrics for a stream."""
        if stream_id not in self.streams:
            return {"error": "Stream not found"}

        stream = self.streams[stream_id]

        metrics = {
            "stream_id": stream_id,
            "total_events": stream["event_count"],
            "buffer_size": len(self.event_buffer[stream_id]),
            "processor_count": len(stream["processors"]),
            "watermark": self.watermarks.get(stream_id)
        }

        # Add processor metrics
        processor_metrics = []
        for processor_id in stream["processors"]:
            processor = self.processors[processor_id]
            processor_metrics.append({
                "processor_id": processor_id,
                "processed_count": processor["processed_count"]
            })

        metrics["processors"] = processor_metrics

        return metrics

    def generate_processing_report(self) -> Dict:
        """Generate comprehensive stream processing report."""
        print("\nGenerating Stream Processing Report...")
        print("="*50)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_streams": len(self.streams),
                "total_processors": len(self.processors),
                "total_windows": len(self.windows),
                "total_processed_events": len(self.processed_events)
            },
            "streams": []
        }

        # Stream details
        for stream_id, stream in self.streams.items():
            stream_info = {
                "stream_id": stream_id,
                "total_events": stream["event_count"],
                "buffer_size": len(self.event_buffer[stream_id]),
                "processors": len(stream["processors"])
            }
            report["streams"].append(stream_info)

        # Processing stats
        if self.processed_events:
            processor_counts = defaultdict(int)
            for event in self.processed_events:
                processor_counts[event["processor_id"]] += 1

            report["top_processors"] = sorted(
                processor_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

        print(f"Total Streams: {report['summary']['total_streams']}")
        print(f"Total Processors: {report['summary']['total_processors']}")
        print(f"Total Processed Events: {report['summary']['total_processed_events']}")

        return report


def demo():
    """Demo Stream Processing."""
    print("Stream Processing Demo")
    print("="*50)

    sp = StreamProcessing()

    # 1. Create streams
    print("\n1. Creating Event Streams")
    print("-"*50)

    sp.create_stream("user-events", {"type": "user_activity"})
    sp.create_stream("transaction-events", {"type": "transactions"})

    # 2. Register processors
    print("\n2. Registering Stream Processors")
    print("-"*50)

    def count_events(event: Event) -> Dict:
        """Count events by type."""
        current = sp.get_state(event.event_type, namespace="counters") or 0
        sp.update_state(event.event_type, current + 1, namespace="counters")
        return {"event_type": event.event_type, "count": current + 1}

    def filter_high_value(event: Event) -> Optional[Dict]:
        """Filter high-value transactions."""
        amount = event.data.get("amount", 0)
        if amount > 1000:
            return {"event_id": event.event_id, "amount": amount, "high_value": True}
        return None

    sp.register_processor("event_counter", "user-events", count_events)
    sp.register_processor("high_value_filter", "transaction-events", filter_high_value)

    # 3. Create windows
    print("\n3. Creating Processing Windows")
    print("-"*50)

    sp.create_tumbling_window("5min_tumbling", "user-events", window_size_seconds=300)
    sp.create_sliding_window("10min_sliding", "user-events",
                            window_size_seconds=600, slide_size_seconds=300)

    # 4. Publish events
    print("\n4. Publishing Events to Streams")
    print("-"*50)

    # Publish user events
    base_time = datetime.now()
    for i in range(10):
        event = Event(
            event_id=f"user_event_{i}",
            event_type="page_view" if i % 2 == 0 else "button_click",
            data={"user_id": f"user_{i % 3}", "page": f"/page{i}"},
            event_time=base_time + timedelta(seconds=i * 30)
        )
        sp.publish_event("user-events", event)

    print(f"✓ Published 10 user events")

    # Publish transaction events
    for i in range(5):
        event = Event(
            event_id=f"txn_{i}",
            event_type="purchase",
            data={"amount": 500 + i * 300, "user_id": f"user_{i % 3}"},
            event_time=base_time + timedelta(seconds=i * 60)
        )
        sp.publish_event("transaction-events", event)

    print(f"✓ Published 5 transaction events")

    # 5. Assign events to windows
    print("\n5. Assigning Events to Windows")
    print("-"*50)

    # Get some events from buffer to assign to windows
    for _ in range(5):
        if sp.event_buffer["user-events"]:
            event = sp.event_buffer["user-events"][0]
            windows = sp.assign_to_window("5min_tumbling", event)
            print(f"  Event {event.event_id} assigned to {len(windows)} window(s)")

    # 6. Window aggregation
    print("\n6. Window Aggregation")
    print("-"*50)

    def count_by_type(events: List[Event]) -> Dict:
        """Count events by type."""
        counts = defaultdict(int)
        for event in events:
            counts[event.event_type] += 1
        return dict(counts)

    tumbling_config = sp.windows["5min_tumbling"]
    if tumbling_config.get("current_window"):
        window = tumbling_config["current_window"]
        result = sp.aggregate_window(window, count_by_type)
        print(f"Aggregation result: {result}")

    # 7. State management
    print("\n7. State Management")
    print("-"*50)

    # Get state from counter processor
    page_view_count = sp.get_state("page_view", namespace="counters")
    button_click_count = sp.get_state("button_click", namespace="counters")

    print(f"Page views: {page_view_count}")
    print(f"Button clicks: {button_click_count}")

    # 8. Watermarks
    print("\n8. Setting Watermarks")
    print("-"*50)

    watermark_time = datetime.now() - timedelta(minutes=5)
    sp.set_watermark("user-events", watermark_time)

    # 9. Stream joins
    print("\n9. Creating Stream Join")
    print("-"*50)

    join = sp.join_streams(
        "user_transaction_join",
        "user-events",
        "transaction-events",
        join_key="user_id",
        window_seconds=300
    )

    # 10. Batch processing
    print("\n10. Batch Processing")
    print("-"*50)

    batch_result = sp.process_stream_batch("user-events", batch_size=5)
    print(f"Processed: {batch_result['processed_count']} events")
    print(f"Remaining in buffer: {batch_result['remaining_count']}")

    # 11. Stream metrics
    print("\n11. Stream Metrics")
    print("-"*50)

    metrics = sp.get_stream_metrics("user-events")
    print(f"Total events: {metrics['total_events']}")
    print(f"Buffer size: {metrics['buffer_size']}")
    print(f"Processors:")
    for proc in metrics["processors"]:
        print(f"  {proc['processor_id']}: {proc['processed_count']} events")

    # 12. Generate report
    print("\n12. Processing Report")
    print("-"*50)

    report = sp.generate_processing_report()

    print(f"\nStream Details:")
    for stream in report["streams"]:
        print(f"  {stream['stream_id']}: {stream['total_events']} events, "
              f"{stream['buffer_size']} buffered")

    print("\n✓ Stream Processing Demo Complete!")


if __name__ == '__main__':
    demo()
