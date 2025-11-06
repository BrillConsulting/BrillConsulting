# Linux Shell Scripting Toolkit

Production-ready Bash scripting and automation toolkit with comprehensive features for system administration, DevOps, and infrastructure management. Generate battle-tested shell scripts with built-in error handling, logging, monitoring, and best practices.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Script Types](#script-types)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [API Reference](#api-reference)

## Features

### Core Capabilities

- **Backup Automation**: Intelligent backup scripts with compression, encryption, retention policies, and integrity verification
- **System Monitoring**: Comprehensive monitoring with CPU, memory, disk thresholds, and alert notifications
- **Deployment Automation**: Application deployment with Git integration, testing, rollback capabilities, and service management
- **Log Analysis**: Advanced log parsing and analysis with statistical reporting
- **Database Backups**: Multi-database support (PostgreSQL, MySQL, MongoDB) with automated scheduling
- **System Health Checks**: Complete health monitoring including services, ports, URLs, and system metrics
- **Process Management**: Process monitoring with auto-restart and notification capabilities
- **Cron Job Management**: Automated cron job creation with logging and lock file management
- **Security Hardening**: Security best practices automation for Linux systems
- **Performance Monitoring**: Detailed system performance metrics and reporting

### Advanced Features

- **Error Handling**: Robust error handling with `set -euo pipefail` and custom error traps
- **Retry Mechanisms**: Automatic retry logic for transient failures
- **Lock Files**: Prevent concurrent execution with process lock management
- **Colored Output**: User-friendly colored terminal output for better readability
- **Structured Logging**: Comprehensive logging with timestamps and severity levels
- **Notification Support**: Email and Slack webhook integration for alerts
- **Disk Space Checks**: Automatic verification of available disk space
- **Template System**: Reusable script templates and common functions
- **Script Generation**: Programmatic script generation via Python API

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/shell-scripting-toolkit.git
cd shell-scripting-toolkit

# No external dependencies required
# Scripts are generated as pure Bash
```

## Quick Start

### 1. Generate a Backup Script

```python
from shell_scripts import ShellScriptGenerator

generator = ShellScriptGenerator()

# Generate enhanced backup script
backup_script = generator.generate_backup_script({
    'source': '/var/www/html',
    'destination': '/backup/www',
    'retention_days': 14,
    'compress': True,
    'encrypt': False,
    'exclude_patterns': ['*.tmp', '*.log', 'cache/*'],
    'notification_email': 'admin@example.com',
    'min_disk_space_gb': 20
})

# Save to file
generator.save_script(backup_script, 'backup.sh', '/usr/local/bin')
```

### 2. Generate a Monitoring Script

```python
# Generate system monitoring script
monitoring_script = generator.generate_monitoring_script({
    'cpu_threshold': 80,
    'memory_threshold': 85,
    'disk_threshold': 90,
    'email': 'admin@example.com'
})

generator.save_script(monitoring_script, 'monitor.sh', '/usr/local/bin')
```

### 3. Run Demo

```bash
python shell_scripts.py
```

## Script Types

### 1. Backup Scripts

Automated backup with compression, encryption, and retention:

- Configurable source and destination paths
- Optional gzip compression
- GPG encryption support
- Retention policy management
- Backup integrity verification
- Disk space validation
- Email/Slack notifications
- Detailed logging

**Configuration Options:**
- `source`: Source directory to backup
- `destination`: Backup destination directory
- `retention_days`: Days to keep backups (default: 7)
- `compress`: Enable gzip compression (default: True)
- `encrypt`: Enable GPG encryption (default: False)
- `exclude_patterns`: Patterns to exclude from backup
- `notification_email`: Email for notifications
- `min_disk_space_gb`: Minimum required disk space

### 2. System Monitoring Scripts

Real-time system monitoring with threshold-based alerts:

- CPU usage monitoring
- Memory utilization tracking
- Disk space monitoring
- Service status checks
- Network connectivity tests
- Load average monitoring
- Alert notifications

**Configuration Options:**
- `cpu_threshold`: CPU usage threshold % (default: 80)
- `memory_threshold`: Memory usage threshold % (default: 80)
- `disk_threshold`: Disk usage threshold % (default: 90)
- `email`: Email address for alerts

### 3. Deployment Scripts

Application deployment automation:

- Git repository integration
- Branch-based deployment
- Automatic dependency installation
- Test execution
- Service restart management
- Rollback capabilities
- Backup before deployment

**Configuration Options:**
- `app_name`: Application name
- `git_repo`: Git repository URL
- `branch`: Branch to deploy (default: main)
- `deploy_path`: Deployment directory

### 4. Log Analysis Scripts

Parse and analyze log files:

- Request counting and statistics
- Top IP addresses analysis
- Most requested URLs
- HTTP status code distribution
- Error request tracking
- User agent analysis

**Configuration Options:**
- `log_file`: Path to log file
- `output_file`: Report output path

### 5. Database Backup Scripts

Automated database backups:

- PostgreSQL support (pg_dump)
- MySQL support (mysqldump)
- MongoDB support
- Compression and retention
- Scheduled execution
- Backup verification

**Configuration Options:**
- `db_type`: Database type (postgresql, mysql, mongodb)
- `db_name`: Database name
- `backup_path`: Backup directory
- `retention_days`: Retention period

### 6. System Health Check Scripts

Comprehensive system health monitoring:

- Service status verification
- Port availability checks
- URL accessibility tests
- CPU, memory, disk monitoring
- Security audit (failed logins)
- System uptime tracking
- Load average analysis

**Configuration Options:**
- `services`: List of services to check
- `ports`: List of ports to verify
- `urls`: List of URLs to test
- `disk_threshold`: Disk usage threshold %
- `memory_threshold`: Memory usage threshold %

### 7. Process Monitoring Scripts

Monitor and manage critical processes:

- Process existence checking
- Auto-restart on failure
- Resource usage tracking (CPU, memory)
- Process uptime monitoring
- Alert notifications

**Configuration Options:**
- `processes`: List of process names
- `restart_on_failure`: Enable auto-restart (default: True)
- `notification_email`: Email for alerts

### 8. Cron Job Management Scripts

Automated cron job management:

- Install/remove cron jobs
- Wrapper scripts with logging
- Lock file management
- Status checking
- Log rotation

**Configuration Options:**
- `job_name`: Cron job name
- `command`: Command to execute
- `schedule`: Cron schedule expression
- `user`: User to run cron as (default: root)

### 9. Security Hardening Scripts

Apply security best practices:

- System package updates
- Firewall configuration (UFW/firewalld)
- SSH hardening
- fail2ban installation and configuration
- File permission securing
- Unnecessary service disabling
- Audit logging setup

### 10. Performance Monitoring Scripts

System performance analysis:

- CPU information and usage trends
- Memory usage statistics
- Disk I/O analysis
- Top processes by CPU/memory
- Network statistics
- Connection tracking
- Load average monitoring

## Advanced Features

### Error Handling

All generated scripts include comprehensive error handling:

```bash
# Automatic error detection and handling
set -euo pipefail

# Custom error handler
trap 'handle_error "An error occurred" $LINENO' ERR

# Cleanup on exit
trap 'release_lock "$LOCK_FILE"' EXIT
```

### Retry Mechanism

Built-in retry logic for transient failures:

```bash
# Retry a command up to 3 times
retry 3 curl -f http://example.com
```

### Lock File Management

Prevent concurrent execution:

```bash
# Acquire lock
acquire_lock "$LOCK_FILE"

# Automatic release on exit
trap 'release_lock "$LOCK_FILE"' EXIT
```

### Colored Logging

Structured logging with colors:

```bash
log_info "Informational message"
log_success "Success message"
log_warning "Warning message"
log_error "Error message"
log_step "Step indicator"
```

### Notifications

Email and Slack notifications:

```bash
send_notification "Subject" "Message body"
```

## Best Practices

### 1. Always Use Error Handling

```bash
set -euo pipefail  # Exit on error, undefined variables, pipe failures
```

### 2. Implement Logging

```bash
# Log to both console and file
exec > >(tee -a "$LOG_FILE")
exec 2>&1
```

### 3. Use Lock Files

```bash
# Prevent concurrent execution
acquire_lock "/tmp/script.lock"
trap 'release_lock "/tmp/script.lock"' EXIT
```

### 4. Validate Prerequisites

```bash
check_command tar
check_disk_space "/backup" 10
```

### 5. Implement Notifications

```bash
send_notification "Backup Failed" "Details..."
```

## Examples

### Complete Backup Solution

```python
generator = ShellScriptGenerator()

# Generate backup script
backup = generator.generate_backup_script({
    'source': '/var/www',
    'destination': '/backup',
    'retention_days': 30,
    'compress': True,
    'encrypt': True,
    'notification_email': 'admin@example.com'
})

# Generate cron job to run daily at 2 AM
cron = generator.generate_cron_job_script({
    'job_name': 'daily_backup',
    'command': '/usr/local/bin/backup.sh',
    'schedule': '0 2 * * *',
    'user': 'root'
})

# Save scripts
generator.save_script(backup, 'backup.sh', '/usr/local/bin')
generator.save_script(cron, 'setup_backup_cron.sh', '/tmp')
```

### Monitoring and Alerting

```python
# System health check
health = generator.generate_system_health_check({
    'services': ['nginx', 'postgresql', 'redis'],
    'ports': [80, 443, 5432, 6379],
    'urls': ['http://localhost/health'],
    'disk_threshold': 90,
    'memory_threshold': 85
})

# Process monitoring with auto-restart
process = generator.generate_process_monitor({
    'processes': ['nginx', 'postgresql'],
    'restart_on_failure': True,
    'notification_email': 'ops@example.com'
})

generator.save_script(health, 'health_check.sh', '/usr/local/bin')
generator.save_script(process, 'process_monitor.sh', '/usr/local/bin')
```

## API Reference

### ShellScriptGenerator

Main class for generating shell scripts.

#### Methods

- `generate_backup_script(config)`: Generate backup automation script
- `generate_monitoring_script(config)`: Generate system monitoring script
- `generate_deployment_script(config)`: Generate deployment automation script
- `generate_log_analyzer_script(config)`: Generate log analysis script
- `generate_database_backup_script(config)`: Generate database backup script
- `generate_system_health_check(config)`: Generate health check script
- `generate_process_monitor(config)`: Generate process monitoring script
- `generate_cron_job_script(config)`: Generate cron job management script
- `generate_security_hardening_script()`: Generate security hardening script
- `generate_performance_monitor()`: Generate performance monitoring script
- `save_script(content, filename, output_dir)`: Save script to file
- `get_generator_info()`: Get generator statistics
- `get_common_functions()`: Get common shell functions template
- `get_error_handling()`: Get error handling template

## Technologies

- **Bash 4.0+**: Modern Bash scripting
- **GNU Core Utilities**: grep, awk, sed, find, etc.
- **systemd**: Service management
- **Git**: Version control integration
- **Python 3.7+**: Script generation framework

## Contributing

Contributions are welcome! Please ensure:

- Scripts follow Bash best practices
- Error handling is comprehensive
- Logging is structured and informative
- Scripts are tested on major Linux distributions

## License

Copyright (c) 2025 BrillConsulting. All rights reserved.

## Support

For issues, questions, or contributions, please contact BrillConsulting.
