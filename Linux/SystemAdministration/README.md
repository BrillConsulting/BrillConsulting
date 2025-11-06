# Linux System Administration Toolkit

Production-ready comprehensive Linux system administration and management toolkit with advanced monitoring, security hardening, backup/restore, and automated management capabilities.

## Version

**v2.0.0** - Production-Ready Release

## Overview

This toolkit provides a complete Python-based solution for Linux system administration tasks, including user/group management, package installation, service control, firewall configuration, SSH security, backup/restore operations, system monitoring, and security hardening.

## Features

### Core System Administration

#### User & Group Management
- **Create Users**: Add system users with custom UID, GID, home directory, shell, and group memberships
- **Modify Users**: Update user properties including shell, home directory, groups, and passwords
- **Delete Users**: Remove users with optional home directory cleanup
- **List Users**: Query all system users with detailed information
- **Create Groups**: Add system groups with custom GID and members
- **List Groups**: Query all system groups with membership details

#### Package Management
- **Multi-Distribution Support**: Works with apt (Debian/Ubuntu), yum (RHEL/CentOS), dnf (Fedora), pacman (Arch)
- **Package Installation**: Install packages with dependency resolution
- **Auto-Detection**: Automatically detects system package manager

#### Service Management (systemd)
- **Service Control**: Start, stop, restart, enable, disable services
- **Unit File Generation**: Create custom systemd service unit files
- **Service Status**: Query service state and configuration
- **Multi-Service Management**: Batch service operations

### Security & Hardening

#### Firewall Configuration
- **Multiple Backends**: Support for UFW, iptables, and firewalld
- **Rule Management**: Add/remove firewall rules for ports and protocols
- **Source/Destination Filtering**: Configure network access controls
- **Port Scanning**: Identify open ports and listening services

#### SSH Security
- **Key Generation**: Create SSH key pairs (RSA, Ed25519, ECDSA)
- **Authorized Keys**: Manage authorized_keys files for users
- **SSH Hardening**: Apply security best practices to SSH configuration
- **Configuration Templates**: Generate secure sshd_config files

#### Security Hardening
- **Comprehensive Hardening**: 8-point security hardening checklist
- **Best Practices**: Disable root login, enforce key-based auth, configure fail2ban
- **Audit Logging**: Enable system auditing and intrusion detection
- **Automatic Updates**: Configure unattended security updates

### System Monitoring

#### Resource Monitoring
- **CPU Metrics**: Monitor CPU usage, load average, per-core utilization
- **Memory Monitoring**: Track RAM usage, swap, available memory
- **Disk Monitoring**: Monitor disk usage across all partitions
- **Network Statistics**: Track network I/O, packets sent/received
- **Real-Time Metrics**: Live system resource monitoring with psutil

#### Process Management
- **Process Listing**: View running processes sorted by CPU or memory
- **Resource Usage**: Detailed process resource consumption
- **Top Processes**: Identify resource-intensive applications

#### Health Checks
- **Automated Health Checks**: Comprehensive system health assessment
- **Threshold Monitoring**: Alert on high CPU, memory, or disk usage
- **Status Reporting**: Health status with warnings and errors

### Backup & Restore

- **Full Backups**: Create compressed tar archives of directories
- **Incremental Support**: Exclude patterns for optimized backups
- **Backup Listing**: View available backups with size and date
- **Restore Operations**: Extract backups to specified locations
- **Compression**: gzip compression for space efficiency

### Automation & Scheduling

#### Cron Job Management
- **Create Cron Jobs**: Schedule automated tasks with cron syntax
- **User-Specific Crons**: Set up cron jobs for specific users
- **Flexible Scheduling**: Support for minute, hourly, daily, weekly, monthly schedules

#### Log Management
- **Log Rotation**: Configure logrotate for log file management
- **Retention Policies**: Set rotation intervals and retention periods
- **Compression**: Automatic log compression
- **Size-Based Rotation**: Rotate logs based on file size

### Network Configuration

- **Interface Configuration**: Configure network interfaces with IP, netmask, gateway
- **DNS Configuration**: Set up DNS servers
- **Connectivity Testing**: Test network connectivity to multiple targets
- **Ping Tests**: Automated reachability checks

### File Management

- **Permission Management**: Set file and directory permissions (chmod)
- **Ownership Management**: Change file owner and group (chown)
- **Recursive Operations**: Apply changes recursively to directories
- **Disk Cleanup**: Automated cleanup of temp files, old logs, package caches

## Installation

### Prerequisites

- Python 3.7+
- Linux operating system
- Root or sudo access (for most operations)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Python Packages

- `psutil>=5.9.0` - System monitoring and process management
- `paramiko>=3.3.0` - SSH operations (optional)
- `pyyaml>=6.0.1` - Configuration file support (optional)

## Usage

### Basic Usage

```python
from linux_admin import LinuxSystemAdmin

# Initialize admin (dry_run=True for testing without executing)
admin = LinuxSystemAdmin(hostname='prod-server-01', dry_run=False)

# Create a user
admin.create_user({
    'username': 'appuser',
    'shell': '/bin/bash',
    'groups': ['docker', 'sudo'],
    'sudo': True
})

# Install a package
admin.install_package({
    'package': 'nginx',
    'package_manager': 'apt'
})

# Manage a service
admin.manage_service({
    'name': 'nginx',
    'action': 'start',
    'enabled': True
})

# Configure firewall
admin.configure_firewall({
    'action': 'allow',
    'port': 80,
    'protocol': 'tcp',
    'tool': 'ufw'
})
```

### Advanced Examples

#### System Monitoring

```python
# Monitor system resources
metrics = admin.monitor_system_resources()
print(f"CPU Usage: {metrics['cpu']['percent']}%")
print(f"Memory Usage: {metrics['memory']['percent']}%")

# Check system health
health = admin.check_system_health()
if health['status'] == 'warning':
    print(f"Warnings: {health['warnings']}")

# List top processes by memory
processes = admin.list_processes(sort_by='memory', limit=10)
```

#### SSH Security

```python
# Generate SSH key
admin.generate_ssh_key({
    'name': 'deploy_key',
    'key_type': 'ed25519',
    'bits': 4096,
    'email': 'deploy@example.com'
})

# Add SSH authorized key
admin.add_ssh_authorized_key({
    'user': 'appuser',
    'key_file': '/root/.ssh/deploy_key.pub'
})

# Apply SSH hardening
admin.configure_ssh_hardening()
```

#### Backup & Restore

```python
# Create backup
backup = admin.create_backup({
    'source': '/var/www',
    'destination': '/backup',
    'type': 'full',
    'exclude': ['*.log', '*.cache', 'tmp/*']
})

# List backups
backups = admin.list_backups(backup_dir='/backup')

# Restore backup
admin.restore_backup({
    'archive': '/backup/backup_20250106_120000.tar.gz',
    'destination': '/var/www'
})
```

#### User Management

```python
# List all users
users = admin.list_users()

# Modify user
admin.modify_user('appuser', {
    'shell': '/bin/zsh',
    'groups': ['docker', 'developers']
})

# Delete user
admin.delete_user('olduser', remove_home=True)

# List all groups
groups = admin.list_groups()
```

#### Security Hardening

```python
# Apply comprehensive security hardening
hardening = admin.apply_security_hardening()

# Scan for open ports
ports = admin.scan_open_ports()

# Cleanup disk space
cleanup = admin.manage_disk_space({})
```

#### Cron Jobs

```python
# Schedule database backup
admin.create_cron_job({
    'name': 'database_backup',
    'schedule': '0 2 * * *',  # Daily at 2 AM
    'command': '/usr/local/bin/backup-db.sh',
    'user': 'postgres'
})

# Weekly log cleanup
admin.create_cron_job({
    'name': 'log_cleanup',
    'schedule': '0 3 * * 0',  # Sundays at 3 AM
    'command': '/usr/local/bin/cleanup-logs.sh',
    'user': 'root'
})
```

#### Network Operations

```python
# Test connectivity
results = admin.test_network_connectivity(['8.8.8.8', 'google.com'])

# Configure network interface
admin.configure_network_interface({
    'interface': 'eth0',
    'ip_address': '192.168.1.100',
    'netmask': '255.255.255.0',
    'gateway': '192.168.1.1',
    'dns': ['8.8.8.8', '1.1.1.1']
})
```

## Demo

Run the comprehensive demo to see all features in action:

```bash
python linux_admin.py
```

The demo runs in **dry-run mode** by default, showing commands without executing them. To execute real commands, modify the code to set `dry_run=False`.

## Configuration

### Dry-Run Mode

For testing and validation without executing commands:

```python
admin = LinuxSystemAdmin(hostname='server-01', dry_run=True)
```

### Logging

The toolkit uses Python's logging module. Configure logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Considerations

- **Root Access**: Most operations require root or sudo privileges
- **Dry-Run Testing**: Always test with `dry_run=True` first
- **Backup First**: Create backups before making system changes
- **SSH Keys**: Use strong key types (Ed25519, RSA 4096-bit)
- **Firewall**: Test firewall rules carefully to avoid lockouts
- **Password Policies**: Enforce strong password requirements

## Architecture

### Class: LinuxSystemAdmin

Main class providing all system administration functionality.

**Constructor Parameters:**
- `hostname` (str): Server hostname for identification
- `dry_run` (bool): If True, only show commands without executing

**Key Methods:**
- User Management: `create_user()`, `modify_user()`, `delete_user()`, `list_users()`
- Group Management: `create_group()`, `list_groups()`
- Package Management: `install_package()`
- Service Management: `manage_service()`, `create_systemd_service()`
- Firewall: `configure_firewall()`, `scan_open_ports()`
- SSH: `generate_ssh_key()`, `add_ssh_authorized_key()`, `configure_ssh_hardening()`
- Backup: `create_backup()`, `list_backups()`, `restore_backup()`
- Monitoring: `monitor_system_resources()`, `check_system_health()`, `list_processes()`
- Security: `apply_security_hardening()`, `set_file_permissions()`
- Automation: `create_cron_job()`, `setup_log_rotation()`
- Network: `configure_network_interface()`, `test_network_connectivity()`

## System Requirements

- **OS**: Linux (Debian, Ubuntu, RHEL, CentOS, Fedora, Arch)
- **Python**: 3.7 or higher
- **Privileges**: Root or sudo access
- **Tools**: systemd, cron, ssh, tar (standard on most Linux systems)

## Supported Distributions

- Debian/Ubuntu (apt)
- RHEL/CentOS (yum)
- Fedora (dnf)
- Arch Linux (pacman)

## Best Practices

1. **Test First**: Always use `dry_run=True` for testing
2. **Backup**: Create backups before major changes
3. **Monitoring**: Regularly check system health and resources
4. **Security**: Apply hardening recommendations
5. **Automation**: Use cron jobs for regular maintenance tasks
6. **Documentation**: Document all system changes
7. **Logging**: Monitor logs for issues and errors

## Troubleshooting

### psutil Not Available

If psutil is not installed, monitoring features return mock data. Install with:

```bash
pip install psutil
```

### Permission Denied

Most operations require root access. Run with sudo:

```bash
sudo python linux_admin.py
```

### Command Not Found

Ensure required system tools are installed:

```bash
# Debian/Ubuntu
apt-get install systemd cron openssh-server

# RHEL/CentOS
yum install systemd cronie openssh-server
```

## Contributing

Contributions are welcome! This toolkit is designed to be extended with additional system administration capabilities.

## License

Copyright (c) 2025 BrillConsulting. All rights reserved.

## Author

**BrillConsulting**
Linux System Administration Toolkit v2.0.0

## Version History

- **v2.0.0** (2025-01-06): Production-ready release with comprehensive features
  - Advanced system monitoring with psutil
  - SSH key generation and management
  - Backup and restore functionality
  - Security hardening framework
  - Process management
  - Network configuration and testing
  - Disk space management
  - Enhanced error handling and logging

- **v1.0.0**: Initial release with basic features

## Support

For issues, questions, or contributions, please contact BrillConsulting.
