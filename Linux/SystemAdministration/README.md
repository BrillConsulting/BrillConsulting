# Linux System Administration

Complete Linux system administration and management toolkit.

## Features

- **User Management**: Create users and groups with custom configurations
- **Package Management**: Install packages with apt, yum, dnf, pacman
- **Service Management**: Control systemd services (start, stop, enable)
- **Systemd Units**: Generate service unit files
- **Firewall Configuration**: UFW, iptables, firewalld rules
- **Cron Jobs**: Schedule automated tasks
- **File Permissions**: Set chmod/chown permissions
- **SSH Configuration**: Secure SSH server setup
- **Log Rotation**: Configure logrotate for log management

## Technologies

- Linux System Administration
- systemd
- UFW/iptables/firewalld
- cron
- SSH

## Usage

```python
from linux_admin import LinuxSystemAdmin

# Initialize admin
admin = LinuxSystemAdmin(hostname='prod-server-01')

# Create user
admin.create_user({
    'username': 'appuser',
    'shell': '/bin/bash',
    'groups': ['docker', 'sudo']
})

# Manage service
admin.manage_service({
    'name': 'nginx',
    'action': 'start',
    'enabled': True
})

# Configure firewall
admin.configure_firewall({
    'action': 'allow',
    'port': 80,
    'protocol': 'tcp'
})
```

## Demo

```bash
python linux_admin.py
```
