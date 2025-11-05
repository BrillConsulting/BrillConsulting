# Linux Process Management

Complete process monitoring, management, and control toolkit.

## Features

- **Process Listing**: List and filter running processes
- **Process Information**: Detailed process metrics (CPU, memory, I/O)
- **Process Control**: Kill processes with signals (TERM, KILL, HUP)
- **Priority Management**: Set process nice values
- **Top Processes**: Find top CPU/memory consumers
- **Process Monitoring**: Track resource usage with thresholds
- **Process Tree**: Visualize parent-child relationships
- **Resource Tracking**: Detailed CPU, memory, I/O statistics
- **Script Generation**: Generate monitoring scripts

## Technologies

- psutil
- Linux /proc filesystem
- Process signals
- nice/renice

## Usage

```python
from process_manager import ProcessManager

# Initialize manager
manager = ProcessManager()

# List processes
processes = manager.list_processes({'user': 'www-data'})

# Get process info
info = manager.get_process_info(1234)

# Get top CPU consumers
top_cpu = manager.get_top_processes(sort_by='cpu', limit=10)

# Monitor process
monitor = manager.monitor_process({
    'pid': 1234,
    'cpu_threshold': 80,
    'memory_threshold': 85
})

# Kill process
result = manager.kill_process(9999, signal='TERM')
```

## Demo

```bash
python process_manager.py
```
