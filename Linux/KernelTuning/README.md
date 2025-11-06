# KernelTuning v2.0.0

## Production-Ready Linux Kernel Optimization System

**Author:** BrillConsulting
**License:** MIT
**Python Version:** 3.8+

A comprehensive, enterprise-grade system for optimizing Linux kernel parameters, managing kernel modules, configuring real-time systems, and automating kernel compilation with production-ready features.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Tuning Profiles](#tuning-profiles)
- [System Components](#system-components)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

KernelTuning provides a unified interface for managing all aspects of Linux kernel optimization, from runtime sysctl parameters to boot-time configuration and kernel compilation. The system is designed for system administrators, DevOps engineers, and performance engineers who need reliable, repeatable kernel tuning.

### Key Capabilities

- **Intelligent Tuning Profiles**: Pre-configured optimization profiles for common workloads (web servers, databases, HPC, real-time systems)
- **Automated Backup & Rollback**: All changes are backed up automatically with timestamp tracking
- **NUMA Optimization**: Automatic NUMA topology detection and optimization
- **Real-time Kernel Support**: Configuration and validation for PREEMPT_RT kernels
- **Module Management**: Load, configure, and persist kernel module settings
- **Boot Configuration**: Manage GRUB parameters and boot-time kernel configuration
- **Compilation Automation**: Prepare and configure custom kernel builds
- **System Analysis**: Comprehensive system profiling and tuning recommendations

---

## Features

### 1. Sysctl Parameter Management

- Get, set, and persist kernel parameters
- Automatic backup before changes
- Parameter validation and error handling
- Category-based organization (network, memory, filesystem, scheduler)
- Batch parameter application with rollback capability

### 2. Kernel Module Management

- Load modules with custom parameters
- Configure persistent module loading
- Module dependency tracking
- Runtime and boot-time configuration
- Module information retrieval

### 3. Kernel Compilation Automation

- Environment preparation and validation
- Dependency checking
- Profile-based kernel configuration
- Custom config generation for different workloads
- Support for real-time patches

### 4. Performance Tuning Profiles

**Pre-configured profiles:**
- **Web Server**: Optimized for high-concurrency HTTP/HTTPS traffic
- **Database**: Memory and I/O optimization for RDBMS workloads
- **Real-time**: Low-latency configuration for time-critical applications
- **HPC**: High-Performance Computing optimization with NUMA awareness
- **Storage**: I/O subsystem tuning for file servers and storage systems
- **Network**: High-throughput network optimization
- **Desktop**: Balanced performance for workstation use

### 5. Boot Parameter Configuration

- GRUB configuration management
- Automatic backup and validation
- Distribution-agnostic update commands
- Boot parameter persistence
- Cmdline parameter inspection

### 6. Real-time Kernel Setup

- RT kernel detection and validation
- Real-time scheduling parameter configuration
- Latency optimization
- CPU isolation configuration
- IRQ affinity management

### 7. NUMA Tuning

- Automatic topology detection
- NUMA node information extraction
- Memory policy optimization
- CPU affinity recommendations
- Zone reclaim configuration

---

## Architecture

### Component Structure

```
KernelTuningManager (Main Controller)
├── SysctlManager (Parameter Management)
│   ├── Get/Set parameters
│   ├── Backup/Restore
│   └── Persistence management
├── KernelModuleManager (Module Operations)
│   ├── Load/Unload modules
│   ├── Boot configuration
│   └── Parameter management
├── GRUBManager (Boot Configuration)
│   ├── Cmdline management
│   ├── GRUB updates
│   └── Parameter persistence
├── NUMAManager (NUMA Optimization)
│   ├── Topology detection
│   ├── Node information
│   └── Balancing configuration
├── RealtimeKernelManager (RT Configuration)
│   ├── RT kernel detection
│   ├── Scheduling parameters
│   └── Latency optimization
├── KernelCompilationManager (Build Automation)
│   ├── Environment setup
│   ├── Config generation
│   └── Dependency management
└── TuningProfileManager (Profile Management)
    ├── Profile application
    ├── Parameter sets
    └── Workload optimization
```

### Data Models

- **SysctlParameter**: Kernel parameter with metadata
- **KernelModule**: Module configuration with parameters
- **TuningResult**: Operation result with change tracking
- **TuningProfile**: Enum of available profiles

---

## Installation

### Prerequisites

```bash
# System requirements
- Linux kernel 4.0+ (5.0+ recommended)
- Python 3.8 or higher
- Root or sudo access for system modifications

# Required system packages
sudo apt-get install -y sysctl lsmod modprobe numactl
# or
sudo yum install -y procps-ng kmod numactl
```

### Python Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check system compatibility
python kernel_tuning.py --operation info

# Verify all managers are functional
python kernel_tuning.py --operation analyze
```

---

## Quick Start

### 1. System Information

```bash
# Get comprehensive system information
python kernel_tuning.py --operation info --json

# Analyze current kernel tuning
python kernel_tuning.py --operation analyze
```

### 2. Apply a Tuning Profile

```bash
# Apply web server optimizations
python kernel_tuning.py --operation apply_profile --profile web_server

# Apply database optimizations
python kernel_tuning.py --operation apply_profile --profile database

# Apply HPC optimizations
python kernel_tuning.py --operation apply_profile --profile hpc
```

### 3. Set Individual Parameters

```bash
# Reduce swappiness
python kernel_tuning.py --operation set_parameter --param vm.swappiness --value 10

# Increase file handles
python kernel_tuning.py --operation set_parameter --param fs.file-max --value 2097152
```

### 4. NUMA Optimization

```bash
# Optimize NUMA settings
python kernel_tuning.py --operation numa_optimize
```

### 5. Real-time Configuration

```bash
# Configure real-time parameters
python kernel_tuning.py --operation rt_configure
```

---

## Usage Guide

### Command-Line Interface

```bash
python kernel_tuning.py [options]

Options:
  --operation {info,analyze,apply_profile,set_parameter,load_module,numa_optimize,rt_configure,add_boot_param}
                        Operation to perform
  --profile {web_server,database,realtime,hpc,storage,network,desktop}
                        Tuning profile to apply
  --param PARAM         Parameter name
  --value VALUE         Parameter value
  --module MODULE       Module name
  --json                Output in JSON format
```

### Python API

```python
from kernel_tuning import KernelTuningManager, TuningProfile

# Initialize manager
manager = KernelTuningManager()

# Get system information
info = manager.get_system_info()
print(info)

# Apply a profile
result = manager.execute(
    operation="apply_profile",
    profile=TuningProfile.WEB_SERVER
)
print(result)

# Set individual parameter
result = manager.execute(
    operation="set_parameter",
    parameter="vm.swappiness",
    value="10"
)

# Load kernel module
result = manager.execute(
    operation="load_module",
    module="tcp_bbr",
    parameters={"param1": "value1"}
)
```

---

## Tuning Profiles

### Web Server Profile

Optimizes for high-concurrency network connections and low latency.

**Key Parameters:**
- `net.core.somaxconn=65535` - Maximum socket connections
- `net.ipv4.tcp_max_syn_backlog=8192` - SYN backlog queue
- `net.ipv4.tcp_tw_reuse=1` - Reuse TIME-WAIT sockets
- `vm.swappiness=10` - Reduce swap usage

**Use Cases:** Nginx, Apache, HAProxy, load balancers

### Database Profile

Optimizes memory management and I/O for database workloads.

**Key Parameters:**
- `vm.swappiness=1` - Minimize swapping
- `vm.dirty_ratio=15` - Control dirty page cache
- `kernel.shmmax=68719476736` - Large shared memory (64GB)
- `fs.file-max=2097152` - Maximum file handles

**Use Cases:** PostgreSQL, MySQL, MongoDB, Redis, Oracle

### Real-time Profile

Minimizes latency and jitter for time-critical applications.

**Key Parameters:**
- `kernel.sched_rt_runtime_us=950000` - RT runtime allocation
- `kernel.sched_latency_ns=10000000` - Scheduler latency
- `vm.swappiness=0` - Disable swapping
- `kernel.sched_migration_cost_ns=5000000` - CPU migration cost

**Use Cases:** Industrial control, audio processing, trading systems

### HPC Profile

Optimizes for computational throughput and NUMA systems.

**Key Parameters:**
- `kernel.numa_balancing=1` - Enable NUMA balancing
- `vm.zone_reclaim_mode=0` - NUMA zone reclaim
- `kernel.sched_autogroup_enabled=0` - Disable task grouping
- `kernel.sched_latency_ns=24000000` - Throughput-focused scheduling

**Use Cases:** Scientific computing, data analytics, ML training

### Storage Profile

Optimizes I/O subsystem for file servers and storage systems.

**Key Parameters:**
- `vm.dirty_ratio=40` - Large dirty page cache
- `vm.dirty_background_ratio=10` - Background writeback
- `fs.file-max=2097152` - Maximum file handles
- `fs.aio-max-nr=1048576` - Async I/O requests

**Use Cases:** NFS, Samba, object storage, backup systems

### Network Profile

Maximizes network throughput for high-bandwidth applications.

**Key Parameters:**
- `net.core.rmem_max=134217728` - Receive buffer (128MB)
- `net.core.wmem_max=134217728` - Send buffer (128MB)
- `net.ipv4.tcp_congestion_control=bbr` - BBR congestion control
- `net.core.netdev_max_backlog=250000` - Device backlog

**Use Cases:** CDN, media streaming, data transfer, VPN

---

## System Components

### SysctlManager

Manages kernel runtime parameters via sysctl interface.

**Methods:**
- `get_parameter(param)` - Retrieve current value
- `set_parameter(param, value, persistent=True)` - Set parameter
- `backup_current_config()` - Create configuration backup
- `apply_profile_parameters(parameters)` - Apply parameter set

**Example:**
```python
sysctl_mgr = SysctlManager()
current_value = sysctl_mgr.get_parameter("vm.swappiness")
sysctl_mgr.set_parameter("vm.swappiness", "10")
backup_file = sysctl_mgr.backup_current_config()
```

### KernelModuleManager

Manages kernel module loading and configuration.

**Methods:**
- `is_module_loaded(module_name)` - Check if module is loaded
- `load_module(module)` - Load module with parameters
- `unload_module(module_name)` - Unload module
- `get_module_info(module_name)` - Get module information

**Example:**
```python
module_mgr = KernelModuleManager()
module = KernelModule(
    name="tcp_bbr",
    parameters={"bbr_cwnd_gain": "2"},
    load_on_boot=True
)
module_mgr.load_module(module)
```

### GRUBManager

Manages GRUB bootloader and kernel boot parameters.

**Methods:**
- `get_cmdline_parameters()` - Get current boot parameters
- `add_boot_parameter(parameter)` - Add parameter to GRUB config

**Example:**
```python
grub_mgr = GRUBManager()
cmdline = grub_mgr.get_cmdline_parameters()
grub_mgr.add_boot_parameter("isolcpus=1,2,3")
```

### NUMAManager

Manages NUMA topology and optimization.

**Methods:**
- `get_numa_topology()` - Get NUMA node information
- `optimize_numa_settings()` - Apply NUMA optimizations

**Example:**
```python
numa_mgr = NUMAManager()
if numa_mgr.numa_available:
    topology = numa_mgr.get_numa_topology()
    changes = numa_mgr.optimize_numa_settings()
```

### RealtimeKernelManager

Manages real-time kernel configuration.

**Methods:**
- `get_rt_info()` - Get RT kernel information
- `configure_rt_parameters()` - Apply RT optimizations

**Example:**
```python
rt_mgr = RealtimeKernelManager()
if rt_mgr.is_rt_kernel:
    info = rt_mgr.get_rt_info()
    changes = rt_mgr.configure_rt_parameters()
```

### KernelCompilationManager

Automates kernel compilation preparation.

**Methods:**
- `get_current_config()` - Get current kernel config
- `prepare_compilation_env()` - Prepare build environment
- `generate_custom_config(profile)` - Generate profile-based config

**Example:**
```python
compile_mgr = KernelCompilationManager()
config = compile_mgr.get_current_config()
env = compile_mgr.prepare_compilation_env()
custom_config = compile_mgr.generate_custom_config(TuningProfile.REALTIME)
```

---

## API Reference

### Main Manager Class

#### `KernelTuningManager`

Primary interface for all kernel tuning operations.

**Methods:**

##### `get_system_info() -> Dict[str, Any]`
Returns comprehensive system information including kernel version, architecture, resources, NUMA topology, and RT status.

##### `analyze_current_tuning() -> Dict[str, Any]`
Analyzes current kernel configuration and provides recommendations based on system characteristics.

##### `execute(operation, profile=None, **kwargs) -> Dict[str, Any]`
Execute kernel tuning operations.

**Parameters:**
- `operation`: Operation to perform (info, analyze, apply_profile, etc.)
- `profile`: TuningProfile enum (optional)
- `**kwargs`: Additional operation-specific parameters

**Returns:**
```python
{
    "operation": str,
    "timestamp": str,
    "success": bool,
    "data": dict,
    "error": str (if failed)
}
```

---

## Best Practices

### 1. Always Backup Before Changes

```python
# Automatic backup is built-in, but you can manually backup too
sysctl_mgr = SysctlManager()
backup_file = sysctl_mgr.backup_current_config()
```

### 2. Test Changes in Development First

Apply tuning profiles in non-production environments first to understand their impact.

### 3. Monitor After Changes

Use monitoring tools to validate that changes improve performance:
```bash
# Monitor system performance
vmstat 1
iostat -x 1
netstat -s
```

### 4. Document Custom Configurations

Keep records of why specific parameters were changed and their expected impact.

### 5. Gradual Optimization

Apply one profile or parameter at a time, measure impact, then proceed to the next optimization.

### 6. Review Logs Regularly

```python
import logging
logging.basicConfig(level=logging.INFO)
# All operations are logged with INFO level
```

### 7. Consider Kernel Version Compatibility

Some parameters may not be available on older kernels. The system handles this gracefully but always check compatibility.

---

## Troubleshooting

### Common Issues

#### 1. Permission Denied

**Problem:** Cannot write to /etc/sysctl.d or modify kernel parameters

**Solution:**
```bash
# Run with sudo
sudo python kernel_tuning.py --operation apply_profile --profile web_server
```

#### 2. Parameter Not Found

**Problem:** Sysctl parameter doesn't exist

**Solution:**
```bash
# Check available parameters
sysctl -a | grep parameter_name

# Verify kernel version supports the parameter
uname -r
```

#### 3. GRUB Update Fails

**Problem:** Cannot update GRUB configuration

**Solution:**
```bash
# Manually update GRUB
sudo update-grub  # Debian/Ubuntu
sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # RHEL/CentOS
```

#### 4. Module Load Failure

**Problem:** Kernel module cannot be loaded

**Solution:**
```bash
# Check module availability
modinfo module_name

# Check kernel logs
dmesg | tail -20

# Verify module dependencies
modprobe --show-depends module_name
```

#### 5. NUMA Not Detected

**Problem:** NUMA features not available

**Solution:**
```bash
# Verify NUMA hardware support
numactl --hardware

# Check BIOS settings for NUMA enablement
# Some systems require BIOS configuration
```

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

manager = KernelTuningManager()
result = manager.execute(operation="info")
```

### Rollback Procedure

If changes cause issues:
```bash
# Restore from backup
sudo cp /var/backups/kernel-tuning/sysctl_backup_TIMESTAMP.conf /etc/sysctl.d/99-kernel-tuning.conf
sudo sysctl -p /etc/sysctl.d/99-kernel-tuning.conf

# Or reboot to default settings
sudo rm /etc/sysctl.d/99-kernel-tuning.conf
sudo reboot
```

---

## Performance Validation

### Benchmarking

After applying tuning profiles, validate performance improvements:

```bash
# Network performance
iperf3 -s  # server
iperf3 -c server_ip  # client

# Disk I/O
fio --name=randwrite --ioengine=libaio --iodepth=32 --rw=randwrite --bs=4k --size=1G

# Latency testing (RT systems)
cyclictest -p 99 -m -n -i 200 -l 1000000

# Database benchmarks
sysbench oltp_read_write --mysql-host=localhost --mysql-user=test --mysql-password=test prepare
sysbench oltp_read_write --mysql-host=localhost --mysql-user=test --mysql-password=test run
```

---

## Advanced Usage

### Custom Profile Creation

```python
from kernel_tuning import SysctlParameter, TuningProfile

custom_params = [
    SysctlParameter(
        name="net.ipv4.tcp_congestion_control",
        value="bbr",
        description="Use BBR congestion control",
        category="network"
    ),
    SysctlParameter(
        name="net.core.default_qdisc",
        value="fq",
        description="Fair queueing",
        category="network"
    )
]

# Apply custom parameters
sysctl_mgr = SysctlManager()
changes = sysctl_mgr.apply_profile_parameters(custom_params)
```

### Automated Deployment

```python
#!/usr/bin/env python3
"""
Automated kernel tuning deployment script
"""
from kernel_tuning import KernelTuningManager, TuningProfile

def deploy_tuning(server_type):
    manager = KernelTuningManager()

    # Backup current config
    print("Creating backup...")
    manager.sysctl_mgr.backup_current_config()

    # Apply appropriate profile
    profile_map = {
        'web': TuningProfile.WEB_SERVER,
        'db': TuningProfile.DATABASE,
        'hpc': TuningProfile.HPC
    }

    profile = profile_map.get(server_type)
    if profile:
        print(f"Applying {profile.value} profile...")
        result = manager.execute(
            operation="apply_profile",
            profile=profile
        )

        if result['success']:
            print(f"Applied {len(result['data']['changes'])} changes")
            for change in result['data']['changes']:
                print(f"  - {change}")
        else:
            print(f"Error: {result.get('error')}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        deploy_tuning(sys.argv[1])
```

---

## Contributing

Contributions are welcome! Please ensure:

1. Code follows PEP 8 style guidelines
2. All new features include documentation
3. Changes are tested on multiple Linux distributions
4. Backup and rollback mechanisms are maintained

---

## Security Considerations

- **Root Access Required**: Most operations require root/sudo privileges
- **Backup Everything**: Automatic backups are created but verify they exist
- **Test First**: Always test in non-production environments
- **Audit Logs**: All operations are logged for audit trails
- **Parameter Validation**: Input validation prevents malicious parameter injection

---

## Support

For issues, questions, or contributions:

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: This README and inline code documentation
- **Logs**: Check `/var/log/syslog` or `journalctl` for system-level issues

---

## License

MIT License - See LICENSE file for details

---

## Changelog

### v2.0.0 (2025-11-06)
- Complete production-ready rewrite
- Added 7 optimized tuning profiles
- Implemented NUMA topology detection and optimization
- Added real-time kernel support and validation
- Implemented GRUB boot parameter management
- Added kernel module management system
- Automated backup and rollback capability
- Comprehensive error handling and logging
- Full API documentation
- CLI interface with multiple operations

### v1.0.0
- Initial basic implementation

---

**Built with expertise by BrillConsulting**

