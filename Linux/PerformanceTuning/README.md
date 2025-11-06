# Linux Performance Tuning System

**Version:** 2.0.0
**Author:** BrillConsulting
**Status:** Production-Ready

## Overview

Comprehensive production-ready Linux performance tuning system for optimizing CPU, memory, disk I/O, and network performance. Includes advanced kernel parameter management, process priority control, resource limits, performance profiling, benchmarking suite, and tuned profile management.

## Features

### Core Optimization

- **CPU Tuning**
  - CPU frequency governors (performance, powersave, ondemand, schedutil)
  - Intel Turbo Boost / AMD Turbo Core control
  - CPU affinity and isolation for dedicated workloads
  - IRQ balancing and distribution
  - NUMA balancing configuration
  - CPU C-state management

- **Memory Optimization**
  - Swap behavior tuning (swappiness, cache pressure)
  - Transparent Huge Pages (THP) configuration
  - Dirty memory thresholds and writeback tuning
  - Memory overcommit settings
  - NUMA memory allocation policies
  - OOM killer tunables

- **Disk I/O Optimization**
  - Multiple I/O scheduler support (mq-deadline, BFQ, kyber, none)
  - Read-ahead buffer optimization
  - Request queue depth tuning
  - SSD-specific optimizations
  - Scheduler-specific parameter tuning
  - Rotational vs non-rotational device handling

- **Network Performance**
  - TCP buffer size optimization
  - BBR congestion control
  - TCP Fast Open
  - Connection tracking tunables
  - Network backlog optimization
  - Advanced TCP features (SACK, timestamps, window scaling)

### Advanced Features

- **Kernel Parameter Management (sysctl)**
  - Apply custom kernel parameters dynamically
  - Generate persistent sysctl configurations
  - Parameter validation and error handling

- **Process Priority Management**
  - CPU nice values (-20 to 19)
  - I/O priority classes (realtime, best-effort, idle)
  - OOM score adjustment
  - CPU affinity binding

- **Resource Limits**
  - Maximum open files configuration
  - Process limits
  - Memory locking limits
  - Stack size limits
  - CPU time limits
  - ulimit and systemd service limits

- **Performance Profiling**
  - perf integration for CPU profiling
  - System call tracing with strace
  - Cache profiling
  - Event-based profiling
  - Call graph generation

- **Comprehensive Benchmarking**
  - CPU benchmarks (sysbench, stress tests)
  - Memory bandwidth and latency tests
  - Disk I/O benchmarks (fio, sequential/random)
  - Network throughput testing (iperf3)
  - IOPS measurements

- **Tuned Profile Management**
  - Create custom tuned profiles
  - Apply pre-configured profiles
  - Profile inheritance and customization
  - System-specific optimization presets

- **Performance Analysis**
  - CPU bottleneck detection
  - Memory pressure analysis
  - Disk I/O saturation detection
  - Automated recommendations

## Architecture

```
PerformanceTuner (Main Class)
├── CPU Tuning
│   ├── Frequency governors
│   ├── Turbo boost control
│   ├── CPU affinity
│   └── IRQ balancing
├── Memory Tuning
│   ├── Swap configuration
│   ├── THP management
│   ├── Dirty memory tuning
│   └── NUMA policies
├── Disk I/O Tuning
│   ├── Scheduler selection
│   ├── Queue optimization
│   └── SSD tuning
├── Network Tuning
│   ├── TCP optimization
│   ├── Buffer sizing
│   └── Congestion control
├── Kernel Parameters
│   └── sysctl management
├── Process Management
│   ├── Priority setting
│   └── Affinity control
├── Resource Limits
│   └── ulimit configuration
├── Profiling
│   ├── perf integration
│   └── Event tracing
├── Benchmarking
│   └── Multi-component tests
└── Tuned Profiles
    ├── Profile creation
    └── Profile application

PerformanceAnalyzer (Analysis Class)
├── CPU bottleneck analysis
├── Memory pressure detection
└── Disk I/O saturation analysis
```

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd Linux/PerformanceTuning

# No external dependencies required (Python 3.7+)
python3 performance_tuning.py
```

## Usage

### Basic Usage

```python
from performance_tuning import (
    PerformanceTuner,
    CPUTuningConfig,
    MemoryTuningConfig,
    DiskIOConfig
)

# Initialize tuner
tuner = PerformanceTuner(hostname='prod-server-01', dry_run=False)

# CPU optimization
cpu_config = CPUTuningConfig(
    governor="performance",
    turbo_boost=True,
    irq_balance=True
)
tuner.tune_cpu(cpu_config)

# Memory optimization
memory_config = MemoryTuningConfig(
    swappiness=10,
    transparent_hugepages="madvise"
)
tuner.tune_memory(memory_config)

# Disk I/O optimization
disk_config = DiskIOConfig(
    device="sda",
    scheduler="mq-deadline",
    read_ahead_kb=512
)
tuner.tune_disk_io(disk_config)

# Network optimization
tuner.tune_network({
    'congestion_control': 'bbr',
    'tcp_fastopen': 3
})
```

### Process Priority Management

```python
from performance_tuning import ProcessPriority

# Set high priority for critical process
priority = ProcessPriority(
    nice=-10,
    ionice_class="realtime",
    ionice_priority=0
)
tuner.set_process_priority(pid=1234, priority=priority)
```

### Resource Limits

```python
from performance_tuning import ResourceLimits

# Configure system resource limits
limits = ResourceLimits(
    max_open_files=65536,
    max_processes=4096,
    max_locked_memory=64
)
tuner.configure_resource_limits(limits)
```

### Performance Profiling

```python
# Profile system performance
tuner.profile_performance(
    duration=60,
    events=['cycles', 'cache-misses', 'branch-misses']
)
```

### Benchmarking

```python
# Run comprehensive benchmarks
benchmark_results = tuner.run_benchmarks()
```

### Tuned Profiles

```python
# Create custom tuned profile
tuner.create_tuned_profile(
    profile_name='custom-db-server',
    profile_type='throughput-performance'
)

# Apply existing profile
tuner.apply_tuned_profile('latency-performance')
```

### Generate Tuning Script

```python
# Generate shell script with all optimizations
script_path = tuner.generate_tuning_script('/tmp/apply_tuning.sh')

# Execute the script
# bash /tmp/apply_tuning.sh
```

### Export Configuration

```python
# Export all configurations to JSON
config_path = tuner.export_configuration('/tmp/perf_config.json')
```

## Configuration Examples

### High-Performance Computing

```python
# Optimize for compute-intensive workloads
tuner = PerformanceTuner(hostname='hpc-node')

cpu_config = CPUTuningConfig(
    governor="performance",
    turbo_boost=True,
    cpu_isolation=[8, 9, 10, 11]  # Isolate cores for compute
)
tuner.tune_cpu(cpu_config)

memory_config = MemoryTuningConfig(
    swappiness=1,  # Minimize swapping
    transparent_hugepages="always",  # Enable THP
    dirty_ratio=40
)
tuner.tune_memory(memory_config)
```

### Database Server

```python
# Optimize for database workloads
tuner = PerformanceTuner(hostname='db-server')

memory_config = MemoryTuningConfig(
    swappiness=10,
    cache_pressure=50,
    transparent_hugepages="never"  # Often better for databases
)
tuner.tune_memory(memory_config)

disk_config = DiskIOConfig(
    scheduler="mq-deadline",
    read_ahead_kb=256,
    nr_requests=512
)
tuner.tune_disk_io(disk_config)
```

### Web Server

```python
# Optimize for web serving
tuner = PerformanceTuner(hostname='web-server')

tuner.tune_network({
    'congestion_control': 'bbr',
    'tcp_fastopen': 3,
    'tcp_window_scaling': True
})

limits = ResourceLimits(
    max_open_files=100000,  # Handle many connections
    max_processes=8192
)
tuner.configure_resource_limits(limits)
```

## Monitoring Commands

The system generates comprehensive monitoring commands:

```bash
# CPU monitoring
mpstat -P ALL 1 10
top -b -n 1 | head -20

# Memory monitoring
free -h
vmstat 1 10

# Disk I/O monitoring
iostat -x 1 10
iotop -b -n 1

# Network monitoring
sar -n DEV 1 10
ss -s
```

## Performance Analysis

```python
from performance_tuning import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Analyze bottlenecks
cpu_analysis = analyzer.analyze_cpu_bottlenecks()
memory_analysis = analyzer.analyze_memory_bottlenecks()
disk_analysis = analyzer.analyze_disk_bottlenecks()

# Get recommendations
for rec in cpu_analysis['recommendations']:
    print(rec)
```

## System Requirements

- **OS:** Linux (kernel 4.x or higher recommended)
- **Python:** 3.7+
- **Privileges:** Root/sudo access required for most operations
- **Tools:** cpupower, sysctl, tuned (optional), perf (optional)

## Best Practices

1. **Always test in dry-run mode first**
   ```python
   tuner = PerformanceTuner(dry_run=True)
   ```

2. **Backup current settings before applying changes**
   ```bash
   sysctl -a > /tmp/sysctl_backup.conf
   ```

3. **Apply changes incrementally and monitor impact**

4. **Use appropriate profiles for workload type**
   - Latency-sensitive: Use latency-performance profile
   - Throughput-focused: Use throughput-performance profile
   - Power-efficient: Use balanced or powersave profile

5. **Profile before and after optimization**
   ```python
   # Before optimization
   tuner.profile_performance(duration=60)

   # Apply optimizations
   # ...

   # After optimization
   tuner.profile_performance(duration=60)
   ```

6. **Document all changes**
   ```python
   tuner.export_configuration('/etc/performance/config.json')
   ```

## Technologies

- **Linux Kernel:** sysctl, /proc, /sys interfaces
- **CPU Tools:** cpupower, turbostat
- **I/O Schedulers:** mq-deadline, BFQ, kyber
- **Profiling:** perf, strace, ltrace, bpftrace
- **Benchmarking:** sysbench, fio, iperf3
- **Tuning Framework:** tuned daemon
- **Network:** TCP BBR, TCP Fast Open

## Use Cases

- **High-Performance Computing (HPC)**
  - Scientific simulations
  - Parallel computing workloads
  - CPU-intensive applications

- **Database Servers**
  - PostgreSQL, MySQL, MongoDB optimization
  - Memory-intensive workloads
  - I/O optimization for storage engines

- **Web Servers**
  - Nginx, Apache optimization
  - High-concurrency scenarios
  - Network throughput optimization

- **Real-Time Applications**
  - Low-latency requirements
  - Deterministic performance
  - CPU isolation and RT kernels

- **Virtualization Hosts**
  - KVM/QEMU optimization
  - Resource allocation
  - Guest performance tuning

- **Storage Servers**
  - NFS, Ceph optimization
  - I/O scheduler tuning
  - Cache management

## Troubleshooting

### Permission Denied

```bash
# Run with sudo
sudo python3 performance_tuning.py
```

### Tuned Service Not Found

```bash
# Install tuned
sudo apt-get install tuned  # Debian/Ubuntu
sudo yum install tuned      # RHEL/CentOS
```

### perf Command Not Found

```bash
# Install perf
sudo apt-get install linux-tools-generic
```

## References

- [Linux Kernel Documentation](https://www.kernel.org/doc/Documentation/)
- [Red Hat Performance Tuning Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/monitoring_and_managing_system_status_and_performance/)
- [TCP BBR Congestion Control](https://queue.acm.org/detail.cfm?id=3022184)
- [Linux Performance Analysis](http://www.brendangregg.com/linuxperf.html)
- [I/O Schedulers Documentation](https://www.kernel.org/doc/html/latest/block/index.html)

## License

Copyright (c) 2024 BrillConsulting. All rights reserved.

## Support

For issues, questions, or contributions, please contact BrillConsulting.
