# Linux Shell Scripting

Advanced Bash scripting and automation toolkit with production-ready script generation.

## Features

- **Backup Scripts**: Generate automated backup scripts with compression and retention
- **Monitoring Scripts**: System monitoring with CPU, memory, disk thresholds
- **Deployment Scripts**: Application deployment with Git, testing, service restart
- **Log Analyzers**: Parse and analyze log files (Apache, Nginx)
- **Database Backups**: PostgreSQL, MySQL backup automation
- **Error Handling**: Robust error handling with set -euo pipefail
- **Colored Output**: User-friendly output with color coding
- **Logging Functions**: Structured logging (info, error, warning)

## Technologies

- Bash scripting
- Shell utilities (awk, sed, grep)
- systemd
- Git

## Usage

```python
from shell_scripts import ShellScriptGenerator

# Initialize generator
generator = ShellScriptGenerator()

# Generate backup script
backup_script = generator.generate_backup_script({
    'source': '/var/www/html',
    'destination': '/backup/www',
    'retention_days': 14,
    'compress': True
})

# Generate monitoring script
monitoring_script = generator.generate_monitoring_script({
    'cpu_threshold': 80,
    'memory_threshold': 85,
    'disk_threshold': 90
})

# Save to file
with open('backup.sh', 'w') as f:
    f.write(backup_script)
```

## Demo

```bash
python shell_scripts.py
```
