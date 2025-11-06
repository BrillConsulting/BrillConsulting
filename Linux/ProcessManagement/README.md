# Linux Process Management System

**Version:** 2.0.0
**Author:** BrillConsulting
**Status:** Production Ready

A comprehensive, production-grade Linux process monitoring, management, and control toolkit with real-time alerting, resource tracking, and automated management capabilities.

## Overview

This system provides enterprise-level process management capabilities for Linux environments, featuring real-time monitoring with automated alerts, detailed resource tracking, process tree visualization, priority management, and comprehensive signal handling. Built on `psutil` with thread-safe operations and robust error handling.

## Features

### Core Process Management
- **Process Listing & Filtering**: List all processes with flexible filtering by user, name, status, CPU, and memory usage
- **Detailed Process Information**: Comprehensive metrics including CPU times, memory breakdown, I/O counters, connections, and file descriptors
- **Signal Handling**: Send any standard Unix signal (TERM, KILL, HUP, INT, QUIT, USR1, USR2, STOP, CONT)
- **Priority Management**: Set and modify process priority using nice values (-20 to 19)
- **Top Process Identification**: Find top resource consumers sorted by CPU or memory usage

### Advanced Monitoring
- **Real-time Process Monitoring**: Thread-safe monitoring system with configurable thresholds
- **Automated Alerts**: Alert generation with severity levels, cooldown periods, and custom callbacks
- **Resource Tracking**: Historical tracking of CPU, memory, I/O, threads, and file descriptors over time
- **Process Trees**: Recursive process tree visualization with resource usage
- **System Statistics**: Overall system health including CPU, memory, swap, and process counts

### Resource Management
- **Resource Limits**: View and set process resource limits (CPU time, file size, memory, file descriptors, etc.)
- **Performance Analysis**: Identify resource bottlenecks and optimize process performance
- **Alert Management**: Filter and retrieve alerts by severity with configurable limits

## Architecture

```
ProcessManager (Main Interface)
├── ProcessMonitor (Real-time Monitoring)
│   ├── Alert System (Thread-safe)
│   ├── Threshold Checking
│   └── Monitoring Loop
├── Process Operations
│   ├── List & Filter
│   ├── Get Info
│   ├── Signal Handling
│   └── Priority Management
└── Resource Management
    ├── Limits
    ├── Tracking
    └── Statistics
```

## Installation

### Requirements
- Python 3.8+
- Linux operating system
- Root privileges (for some operations)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python process_manager.py
```

## Usage

### Quick Start

```python
from process_manager import ProcessManager

# Initialize
manager = ProcessManager()

# Get system overview
stats = manager.get_system_stats()
print(f"CPU Usage: {stats['cpu']['percent_total']}%")
print(f"Memory: {stats['memory']['percent']}%")

# List processes with high CPU usage
processes = manager.list_processes({'min_cpu': 50.0})

# Get top CPU consumers
top_cpu = manager.get_top_processes(sort_by='cpu', limit=10)
for proc in top_cpu:
    print(f"PID {proc['pid']}: {proc['name']} - {proc['cpu_percent']}%")
```

### Process Information

```python
# Get detailed process information
info = manager.get_process_info(1234)
print(f"Name: {info['name']}")
print(f"CPU: {info['cpu_percent']}%")
print(f"Memory: {info['memory_info']['rss_mb']} MB")
print(f"Threads: {info['num_threads']}")
print(f"I/O Read: {info['io_counters']['read_bytes']} bytes")
```

### Process Control

```python
# Send signals
manager.send_signal(1234, 'TERM')  # Graceful shutdown
manager.send_signal(1234, 'HUP')   # Reload config
manager.send_signal(1234, 'KILL')  # Force kill

# Set priority (nice value)
result = manager.set_priority(1234, 10)  # Lower priority
result = manager.set_priority(1234, -5)  # Higher priority (requires root)
```

### Real-time Monitoring

```python
# Configure monitoring
monitors = [
    {
        'pid': 1234,
        'name': 'webapp',
        'cpu_threshold': 80.0,      # Alert if CPU > 80%
        'memory_threshold': 85.0,   # Alert if memory > 85%
        'check_interval': 5,        # Check every 5 seconds
        'alert_cooldown': 60        # Min 60s between alerts
    }
]

# Start monitoring
result = manager.start_monitoring(monitors)
print(f"Monitoring {result['monitors']} processes")

# Get alerts
alerts = manager.get_alerts(severity='warning', limit=10)
for alert in alerts:
    print(f"{alert['alert_type']}: PID {alert['pid']} - {alert['message']}")

# Stop monitoring
manager.stop_monitoring()
```

### Resource Tracking

```python
# Track process resources over time
snapshots = manager.track_process_resources(
    pid=1234,
    duration=60,   # Track for 60 seconds
    interval=1     # Sample every second
)

# Analyze trends
for snapshot in snapshots:
    print(f"{snapshot['timestamp']}: CPU {snapshot['cpu_percent']}%, "
          f"Memory {snapshot['memory_rss_mb']} MB")
```

### Process Trees

```python
# Get process tree
tree = manager.get_process_tree(root_pid=1)

# Recursive tree structure
def print_tree(node, indent=0):
    print("  " * indent + f"PID {node['pid']}: {node['name']}")
    for child in node['children']:
        print_tree(child, indent + 1)

print_tree(tree)
```

### Resource Limits

```python
# Get current limits
limits = manager.get_resource_limits()
print(f"Max open files: {limits['limits']['nofile']}")

# Set limits (current process only)
result = manager.set_resource_limits(os.getpid(), [
    {
        'resource_type': 'nofile',
        'soft_limit': 4096,
        'hard_limit': 8192
    }
])
```

### Advanced Filtering

```python
# Filter by multiple criteria
processes = manager.list_processes({
    'user': 'www-data',          # Specific user
    'name': 'nginx',             # Process name contains 'nginx'
    'status': 'running',         # Only running processes
    'min_cpu': 10.0,             # CPU > 10%
    'min_memory': 5.0            # Memory > 5%
})

# Get processes from specific user
user_procs = manager.list_processes({'user': 'postgres'})

# Find memory-intensive processes
memory_hogs = manager.list_processes({'min_memory': 20.0})
```

## API Reference

### ProcessManager

#### `list_processes(filter_config=None)`
List all running processes with optional filtering.

**Parameters:**
- `filter_config` (dict): Filter options
  - `user` (str): Filter by username
  - `name` (str): Filter by process name (partial match)
  - `status` (str): Filter by status (running, sleeping, etc.)
  - `min_cpu` (float): Minimum CPU percentage
  - `min_memory` (float): Minimum memory percentage

**Returns:** List of process dictionaries

#### `get_process_info(pid)`
Get comprehensive information about a specific process.

**Parameters:**
- `pid` (int): Process ID

**Returns:** Process information dictionary or None

#### `send_signal(pid, sig='TERM')`
Send signal to process.

**Parameters:**
- `pid` (int): Process ID
- `sig` (str): Signal name (TERM, KILL, HUP, INT, QUIT, USR1, USR2, STOP, CONT)

**Returns:** Result dictionary

#### `set_priority(pid, nice_value)`
Set process priority (nice value).

**Parameters:**
- `pid` (int): Process ID
- `nice_value` (int): Nice value (-20 to 19, lower = higher priority)

**Returns:** Result dictionary

#### `get_top_processes(sort_by='cpu', limit=10)`
Get top processes by resource usage.

**Parameters:**
- `sort_by` (str): Sort by 'cpu' or 'memory'
- `limit` (int): Number of processes to return

**Returns:** List of top processes

#### `get_process_tree(root_pid=None)`
Get process tree structure.

**Parameters:**
- `root_pid` (int): Root process ID (None for all root processes)

**Returns:** Tree dictionary with recursive children

#### `track_process_resources(pid, duration=60, interval=1)`
Track process resources over time.

**Parameters:**
- `pid` (int): Process ID
- `duration` (int): Tracking duration in seconds
- `interval` (int): Sample interval in seconds

**Returns:** List of resource snapshots

#### `start_monitoring(monitors)`
Start real-time process monitoring.

**Parameters:**
- `monitors` (list): List of monitor configurations

**Returns:** Monitoring status dictionary

#### `stop_monitoring()`
Stop all process monitoring.

**Returns:** Status dictionary

#### `get_alerts(severity=None, limit=50)`
Get recent alerts.

**Parameters:**
- `severity` (str): Filter by severity (info, warning, critical)
- `limit` (int): Maximum alerts to return

**Returns:** List of alert dictionaries

#### `get_system_stats()`
Get overall system statistics.

**Returns:** System statistics dictionary

#### `get_resource_limits(pid=None)`
Get process resource limits.

**Parameters:**
- `pid` (int): Process ID (None for current process)

**Returns:** Resource limits dictionary

## Configuration

### Monitor Configuration

```python
monitor_config = {
    'pid': 1234,                    # Required: Process ID
    'name': 'myapp',                # Optional: Display name
    'cpu_threshold': 80.0,          # CPU alert threshold (%)
    'memory_threshold': 85.0,       # Memory alert threshold (%)
    'check_interval': 5,            # Check interval (seconds)
    'alert_cooldown': 60            # Cooldown between alerts (seconds)
}
```

### Alert Severity Levels
- **info**: Informational alerts
- **warning**: Resource thresholds exceeded
- **critical**: Process died or critical errors

## Performance Considerations

- **CPU Usage**: Process iteration is optimized but may impact performance on systems with thousands of processes
- **Memory**: Monitoring thread uses minimal memory; resource tracking stores snapshots in memory
- **Threading**: Monitor runs in separate daemon thread; multiple managers can coexist
- **Permissions**: Some operations require root privileges (negative nice values, accessing other users' processes)

## Error Handling

All methods include comprehensive error handling:
- `psutil.NoSuchProcess`: Process no longer exists
- `psutil.AccessDenied`: Insufficient permissions
- `psutil.TimeoutExpired`: Operation timeout
- Generic exceptions: Logged with details

## Logging

Structured logging at INFO level by default:
```python
import logging
logging.basicConfig(level=logging.DEBUG)  # Enable debug logging
```

## Production Deployment

### Security Considerations
- Run with least privileges required
- Validate PIDs before operations
- Implement access controls for multi-user environments
- Monitor the monitor (meta-monitoring)

### Best Practices
- Set appropriate alert thresholds
- Use alert cooldowns to prevent spam
- Implement custom alert handlers for notifications
- Regular cleanup of old alerts
- Monitor monitoring thread health

### Example Systemd Service

```ini
[Unit]
Description=Process Monitor Service
After=network.target

[Service]
Type=simple
User=monitor
ExecStart=/usr/bin/python3 /opt/process_monitor/monitor_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### Common Issues

**Permission Denied**
- Solution: Run with sudo for privileged operations or adjust permissions

**Process Not Found**
- Solution: Process may have terminated; check PID validity

**High CPU Usage**
- Solution: Increase check_interval or reduce number of monitors

**No Alerts Generated**
- Solution: Verify thresholds are appropriate; check monitoring is active

## License

Copyright (c) 2025 BrillConsulting. All rights reserved.

## Support

For issues, questions, or contributions, please contact BrillConsulting.

## Changelog

### Version 2.0.0 (2025-11-06)
- Complete production-ready rewrite
- Added real-time monitoring with automated alerts
- Implemented resource tracking over time
- Added comprehensive error handling and logging
- Thread-safe monitoring system
- Resource limit management
- Enhanced process tree visualization
- System statistics dashboard
- Alert severity levels and cooldown periods
- Flexible filtering and sorting
- Full psutil integration
