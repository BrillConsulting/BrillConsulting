"""
Linux System Administration
Author: BrillConsulting
Description: Complete Linux system administration and management toolkit
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class LinuxSystemAdmin:
    """Comprehensive Linux system administration"""

    def __init__(self, hostname: str = 'localhost'):
        """
        Initialize Linux system admin

        Args:
            hostname: Server hostname
        """
        self.hostname = hostname
        self.users = []
        self.groups = []
        self.services = []
        self.packages = []
        self.firewall_rules = []
        self.cron_jobs = []

    def create_user(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create system user

        Args:
            user_config: User configuration

        Returns:
            User details
        """
        user = {
            'username': user_config.get('username', 'newuser'),
            'uid': user_config.get('uid', 1000 + len(self.users)),
            'gid': user_config.get('gid', 1000 + len(self.users)),
            'home': user_config.get('home', f"/home/{user_config.get('username', 'newuser')}"),
            'shell': user_config.get('shell', '/bin/bash'),
            'groups': user_config.get('groups', []),
            'sudo': user_config.get('sudo', False),
            'created_at': datetime.now().isoformat()
        }

        self.users.append(user)
        command = f"useradd -m -s {user['shell']} -u {user['uid']} {user['username']}"

        print(f"✓ User created: {user['username']}")
        print(f"  UID: {user['uid']}, Home: {user['home']}, Shell: {user['shell']}")
        print(f"  Command: {command}")
        return user

    def create_group(self, group_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create system group

        Args:
            group_config: Group configuration

        Returns:
            Group details
        """
        group = {
            'groupname': group_config.get('groupname', 'newgroup'),
            'gid': group_config.get('gid', 2000 + len(self.groups)),
            'members': group_config.get('members', []),
            'created_at': datetime.now().isoformat()
        }

        self.groups.append(group)
        command = f"groupadd -g {group['gid']} {group['groupname']}"

        print(f"✓ Group created: {group['groupname']}")
        print(f"  GID: {group['gid']}, Members: {len(group['members'])}")
        print(f"  Command: {command}")
        return group

    def manage_service(self, service_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage systemd service

        Args:
            service_config: Service configuration

        Returns:
            Service details
        """
        service = {
            'name': service_config.get('name', 'myapp'),
            'action': service_config.get('action', 'start'),
            'enabled': service_config.get('enabled', True),
            'state': 'running' if service_config.get('action') == 'start' else 'stopped',
            'unit_file': f"/etc/systemd/system/{service_config.get('name', 'myapp')}.service",
            'managed_at': datetime.now().isoformat()
        }

        command = f"systemctl {service['action']} {service['name']}"
        if service['enabled']:
            command += f" && systemctl enable {service['name']}"

        print(f"✓ Service managed: {service['name']}")
        print(f"  Action: {service['action']}, State: {service['state']}, Enabled: {service['enabled']}")
        print(f"  Command: {command}")
        return service

    def create_systemd_service(self, service_config: Dict[str, Any]) -> str:
        """
        Generate systemd service unit file

        Args:
            service_config: Service configuration

        Returns:
            Service unit file content
        """
        name = service_config.get('name', 'myapp')
        description = service_config.get('description', 'My Application')
        exec_start = service_config.get('exec_start', '/usr/bin/myapp')
        user = service_config.get('user', 'root')
        working_dir = service_config.get('working_directory', '/')
        restart = service_config.get('restart', 'always')

        unit_file = f"""[Unit]
Description={description}
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={working_dir}
ExecStart={exec_start}
Restart={restart}
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

        print(f"✓ Systemd service unit file generated: {name}.service")
        return unit_file

    def install_package(self, package_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Install system package

        Args:
            package_config: Package configuration

        Returns:
            Installation details
        """
        result = {
            'package': package_config.get('package', 'vim'),
            'package_manager': package_config.get('package_manager', 'apt'),
            'version': package_config.get('version', 'latest'),
            'state': 'installed',
            'dependencies': package_config.get('dependencies', []),
            'installed_at': datetime.now().isoformat()
        }

        commands = {
            'apt': f"apt-get update && apt-get install -y {result['package']}",
            'yum': f"yum install -y {result['package']}",
            'dnf': f"dnf install -y {result['package']}",
            'pacman': f"pacman -S --noconfirm {result['package']}"
        }

        command = commands.get(result['package_manager'], commands['apt'])

        self.packages.append(result)
        print(f"✓ Package installed: {result['package']}")
        print(f"  Package manager: {result['package_manager']}, Version: {result['version']}")
        print(f"  Command: {command}")
        return result

    def configure_firewall(self, firewall_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure firewall rules

        Args:
            firewall_config: Firewall configuration

        Returns:
            Firewall rule details
        """
        rule = {
            'rule_id': len(self.firewall_rules) + 1,
            'action': firewall_config.get('action', 'allow'),
            'port': firewall_config.get('port', 80),
            'protocol': firewall_config.get('protocol', 'tcp'),
            'source': firewall_config.get('source', 'any'),
            'destination': firewall_config.get('destination', 'any'),
            'tool': firewall_config.get('tool', 'ufw'),
            'created_at': datetime.now().isoformat()
        }

        commands = {
            'ufw': f"ufw {rule['action']} {rule['port']}/{rule['protocol']}",
            'iptables': f"iptables -A INPUT -p {rule['protocol']} --dport {rule['port']} -j {rule['action'].upper()}",
            'firewalld': f"firewall-cmd --permanent --add-port={rule['port']}/{rule['protocol']}"
        }

        command = commands.get(rule['tool'], commands['ufw'])

        self.firewall_rules.append(rule)
        print(f"✓ Firewall rule added: {rule['action']} {rule['protocol']}/{rule['port']}")
        print(f"  Source: {rule['source']}, Destination: {rule['destination']}")
        print(f"  Command: {command}")
        return rule

    def create_cron_job(self, cron_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create cron job

        Args:
            cron_config: Cron configuration

        Returns:
            Cron job details
        """
        cron = {
            'name': cron_config.get('name', 'backup_job'),
            'schedule': cron_config.get('schedule', '0 2 * * *'),
            'command': cron_config.get('command', '/usr/bin/backup.sh'),
            'user': cron_config.get('user', 'root'),
            'enabled': cron_config.get('enabled', True),
            'created_at': datetime.now().isoformat()
        }

        cron_entry = f"{cron['schedule']} {cron['command']}"

        self.cron_jobs.append(cron)
        print(f"✓ Cron job created: {cron['name']}")
        print(f"  Schedule: {cron['schedule']}, User: {cron['user']}")
        print(f"  Command: {cron['command']}")
        print(f"  Cron entry: {cron_entry}")
        return cron

    def set_file_permissions(self, permission_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set file permissions

        Args:
            permission_config: Permission configuration

        Returns:
            Permission details
        """
        result = {
            'path': permission_config.get('path', '/var/www/html'),
            'mode': permission_config.get('mode', '755'),
            'owner': permission_config.get('owner', 'www-data'),
            'group': permission_config.get('group', 'www-data'),
            'recursive': permission_config.get('recursive', True),
            'set_at': datetime.now().isoformat()
        }

        chmod_cmd = f"chmod {'-R ' if result['recursive'] else ''}{result['mode']} {result['path']}"
        chown_cmd = f"chown {'-R ' if result['recursive'] else ''}{result['owner']}:{result['group']} {result['path']}"

        print(f"✓ Permissions set: {result['path']}")
        print(f"  Mode: {result['mode']}, Owner: {result['owner']}:{result['group']}")
        print(f"  Commands:")
        print(f"    {chmod_cmd}")
        print(f"    {chown_cmd}")
        return result

    def configure_ssh(self, ssh_config: Dict[str, Any]) -> str:
        """
        Generate SSH server configuration

        Args:
            ssh_config: SSH configuration

        Returns:
            SSH config content
        """
        port = ssh_config.get('port', 22)
        permit_root = ssh_config.get('permit_root_login', 'no')
        password_auth = ssh_config.get('password_authentication', 'no')
        pubkey_auth = ssh_config.get('pubkey_authentication', 'yes')
        max_sessions = ssh_config.get('max_sessions', 10)

        config = f"""# SSH Server Configuration
Port {port}
Protocol 2

# Authentication
PermitRootLogin {permit_root}
PubkeyAuthentication {pubkey_auth}
PasswordAuthentication {password_auth}
PermitEmptyPasswords no

# Session settings
MaxSessions {max_sessions}
MaxAuthTries 3
LoginGraceTime 60

# Security
X11Forwarding no
AllowTcpForwarding yes
ClientAliveInterval 300
ClientAliveCountMax 2

# Subsystem
Subsystem sftp /usr/lib/openssh/sftp-server
"""

        print(f"✓ SSH configuration generated")
        print(f"  Port: {port}, Root login: {permit_root}, Password auth: {password_auth}")
        return config

    def setup_log_rotation(self, logrotate_config: Dict[str, Any]) -> str:
        """
        Generate logrotate configuration

        Args:
            logrotate_config: Logrotate configuration

        Returns:
            Logrotate config content
        """
        log_path = logrotate_config.get('log_path', '/var/log/myapp/*.log')
        rotate = logrotate_config.get('rotate', 7)
        size = logrotate_config.get('size', '100M')
        compress = logrotate_config.get('compress', True)

        config = f"""{log_path} {{
    daily
    rotate {rotate}
    size {size}
    {'compress' if compress else 'nocompress'}
    delaycompress
    notifempty
    create 0640 root adm
    sharedscripts
    postrotate
        systemctl reload myapp > /dev/null 2>&1 || true
    endscript
}}
"""

        print(f"✓ Logrotate configuration generated")
        print(f"  Path: {log_path}, Rotate: {rotate} days, Size: {size}")
        return config

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'hostname': self.hostname,
            'users': len(self.users),
            'groups': len(self.groups),
            'services': len(self.services),
            'packages': len(self.packages),
            'firewall_rules': len(self.firewall_rules),
            'cron_jobs': len(self.cron_jobs),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Linux system administration"""

    print("=" * 60)
    print("Linux System Administration Demo")
    print("=" * 60)

    # Initialize admin
    admin = LinuxSystemAdmin(hostname='prod-server-01')

    print("\n1. Creating system users...")
    admin.create_user({
        'username': 'appuser',
        'shell': '/bin/bash',
        'groups': ['docker', 'sudo'],
        'sudo': True
    })

    admin.create_user({
        'username': 'deploy',
        'shell': '/bin/bash',
        'groups': ['deployers'],
        'sudo': False
    })

    print("\n2. Creating system groups...")
    admin.create_group({
        'groupname': 'developers',
        'members': ['appuser', 'deploy']
    })

    print("\n3. Installing packages...")
    admin.install_package({
        'package': 'nginx',
        'package_manager': 'apt'
    })

    admin.install_package({
        'package': 'postgresql-14',
        'package_manager': 'apt'
    })

    print("\n4. Creating systemd service...")
    service_unit = admin.create_systemd_service({
        'name': 'webapp',
        'description': 'Web Application Service',
        'exec_start': '/usr/bin/python3 /opt/webapp/app.py',
        'user': 'appuser',
        'working_directory': '/opt/webapp',
        'restart': 'always'
    })
    print(service_unit[:150] + "...")

    print("\n5. Managing services...")
    admin.manage_service({
        'name': 'nginx',
        'action': 'start',
        'enabled': True
    })

    admin.manage_service({
        'name': 'postgresql',
        'action': 'start',
        'enabled': True
    })

    print("\n6. Configuring firewall...")
    admin.configure_firewall({
        'action': 'allow',
        'port': 80,
        'protocol': 'tcp',
        'tool': 'ufw'
    })

    admin.configure_firewall({
        'action': 'allow',
        'port': 443,
        'protocol': 'tcp',
        'tool': 'ufw'
    })

    admin.configure_firewall({
        'action': 'allow',
        'port': 22,
        'protocol': 'tcp',
        'source': '10.0.0.0/8',
        'tool': 'ufw'
    })

    print("\n7. Creating cron jobs...")
    admin.create_cron_job({
        'name': 'database_backup',
        'schedule': '0 2 * * *',
        'command': '/usr/local/bin/backup-db.sh',
        'user': 'postgres'
    })

    admin.create_cron_job({
        'name': 'log_cleanup',
        'schedule': '0 3 * * 0',
        'command': '/usr/local/bin/cleanup-logs.sh',
        'user': 'root'
    })

    print("\n8. Setting file permissions...")
    admin.set_file_permissions({
        'path': '/var/www/html',
        'mode': '755',
        'owner': 'www-data',
        'group': 'www-data',
        'recursive': True
    })

    print("\n9. Configuring SSH server...")
    ssh_config = admin.configure_ssh({
        'port': 22,
        'permit_root_login': 'no',
        'password_authentication': 'no',
        'pubkey_authentication': 'yes'
    })
    print(ssh_config[:200] + "...")

    print("\n10. Setting up log rotation...")
    logrotate_config = admin.setup_log_rotation({
        'log_path': '/var/log/webapp/*.log',
        'rotate': 14,
        'size': '100M',
        'compress': True
    })
    print(logrotate_config[:150] + "...")

    print("\n11. System summary:")
    info = admin.get_system_info()
    print(f"  Hostname: {info['hostname']}")
    print(f"  Users: {info['users']}")
    print(f"  Groups: {info['groups']}")
    print(f"  Packages: {info['packages']}")
    print(f"  Firewall rules: {info['firewall_rules']}")
    print(f"  Cron jobs: {info['cron_jobs']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
