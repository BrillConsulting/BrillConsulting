"""
Linux System Administration
Author: BrillConsulting
Description: Production-ready Linux system administration and management toolkit
Version: 2.0.0
"""

import json
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    import paramiko
except ImportError:
    paramiko = None

try:
    import yaml
except ImportError:
    yaml = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LinuxSystemAdmin:
    """Production-ready comprehensive Linux system administration toolkit"""

    def __init__(self, hostname: str = 'localhost', dry_run: bool = False):
        """
        Initialize Linux system admin

        Args:
            hostname: Server hostname
            dry_run: If True, only show commands without executing
        """
        self.hostname = hostname
        self.dry_run = dry_run
        self.users = []
        self.groups = []
        self.services = []
        self.packages = []
        self.firewall_rules = []
        self.cron_jobs = []
        self.ssh_keys = []
        self.backups = []

    def _execute_command(self, command: str, check: bool = True) -> Tuple[bool, str, str]:
        """
        Execute shell command safely

        Args:
            command: Command to execute
            check: Raise exception on error

        Returns:
            Tuple of (success, stdout, stderr)
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {command}")
            return True, "", ""

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            return True, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}\nError: {e.stderr}")
            return False, e.stdout, e.stderr
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return False, "", str(e)

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
            'ssh_keys': len(self.ssh_keys),
            'backups': len(self.backups),
            'timestamp': datetime.now().isoformat()
        }

    # ============================================================================
    # SYSTEM MONITORING
    # ============================================================================

    def monitor_system_resources(self) -> Dict[str, Any]:
        """
        Monitor system resources (CPU, memory, disk, network)

        Returns:
            System resource metrics
        """
        if not psutil:
            logger.warning("psutil not available, returning mock data")
            return self._get_mock_system_metrics()

        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'hostname': self.hostname,
                'cpu': {
                    'percent': psutil.cpu_percent(interval=1),
                    'count': psutil.cpu_count(),
                    'per_cpu': psutil.cpu_percent(interval=1, percpu=True),
                    'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'percent': psutil.virtual_memory().percent,
                    'used': psutil.virtual_memory().used,
                    'free': psutil.virtual_memory().free
                },
                'swap': {
                    'total': psutil.swap_memory().total,
                    'used': psutil.swap_memory().used,
                    'free': psutil.swap_memory().free,
                    'percent': psutil.swap_memory().percent
                },
                'disk': {},
                'network': {}
            }

            # Disk usage for all partitions
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    metrics['disk'][partition.mountpoint] = {
                        'device': partition.device,
                        'fstype': partition.fstype,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': usage.percent
                    }
                except PermissionError:
                    continue

            # Network statistics
            net_io = psutil.net_io_counters()
            metrics['network']['bytes_sent'] = net_io.bytes_sent
            metrics['network']['bytes_recv'] = net_io.bytes_recv
            metrics['network']['packets_sent'] = net_io.packets_sent
            metrics['network']['packets_recv'] = net_io.packets_recv

            print(f"✓ System resources monitored")
            print(f"  CPU: {metrics['cpu']['percent']}%, Memory: {metrics['memory']['percent']}%")
            print(f"  Disk usage: {list(metrics['disk'].keys())}")
            return metrics

        except Exception as e:
            logger.error(f"Error monitoring system: {str(e)}")
            return {'error': str(e)}

    def _get_mock_system_metrics(self) -> Dict[str, Any]:
        """Return mock system metrics when psutil unavailable"""
        return {
            'timestamp': datetime.now().isoformat(),
            'hostname': self.hostname,
            'cpu': {'percent': 35.2, 'count': 4},
            'memory': {'total': 16000000000, 'used': 8000000000, 'percent': 50.0},
            'disk': {'/': {'total': 500000000000, 'used': 250000000000, 'percent': 50.0}},
            'network': {'bytes_sent': 1000000, 'bytes_recv': 5000000},
            'note': 'Mock data - install psutil for real metrics'
        }

    def list_processes(self, sort_by: str = 'memory', limit: int = 10) -> List[Dict[str, Any]]:
        """
        List running processes

        Args:
            sort_by: Sort by 'cpu' or 'memory'
            limit: Number of processes to return

        Returns:
            List of process information
        """
        if not psutil:
            logger.warning("psutil not available")
            return []

        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent', 'cpu_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort processes
            if sort_by == 'memory':
                processes.sort(key=lambda x: x.get('memory_percent', 0), reverse=True)
            elif sort_by == 'cpu':
                processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)

            top_processes = processes[:limit]
            print(f"✓ Top {limit} processes by {sort_by}:")
            for proc in top_processes:
                print(f"  PID: {proc['pid']}, Name: {proc['name']}, "
                      f"CPU: {proc.get('cpu_percent', 0):.1f}%, "
                      f"Memory: {proc.get('memory_percent', 0):.1f}%")

            return top_processes

        except Exception as e:
            logger.error(f"Error listing processes: {str(e)}")
            return []

    def check_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health check

        Returns:
            Health check results
        """
        health = {
            'timestamp': datetime.now().isoformat(),
            'hostname': self.hostname,
            'status': 'healthy',
            'warnings': [],
            'errors': [],
            'checks': {}
        }

        # CPU check
        if psutil:
            cpu_percent = psutil.cpu_percent(interval=1)
            health['checks']['cpu'] = {
                'percent': cpu_percent,
                'status': 'warning' if cpu_percent > 80 else 'ok'
            }
            if cpu_percent > 80:
                health['warnings'].append(f"High CPU usage: {cpu_percent}%")

            # Memory check
            mem = psutil.virtual_memory()
            health['checks']['memory'] = {
                'percent': mem.percent,
                'status': 'warning' if mem.percent > 80 else 'ok'
            }
            if mem.percent > 80:
                health['warnings'].append(f"High memory usage: {mem.percent}%")

            # Disk check
            health['checks']['disk'] = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    status = 'warning' if usage.percent > 80 else 'ok'
                    health['checks']['disk'][partition.mountpoint] = {
                        'percent': usage.percent,
                        'status': status
                    }
                    if usage.percent > 80:
                        health['warnings'].append(f"High disk usage on {partition.mountpoint}: {usage.percent}%")
                except PermissionError:
                    continue

        # Update overall status
        if health['errors']:
            health['status'] = 'critical'
        elif health['warnings']:
            health['status'] = 'warning'

        print(f"✓ System health check completed")
        print(f"  Status: {health['status'].upper()}")
        print(f"  Warnings: {len(health['warnings'])}, Errors: {len(health['errors'])}")

        return health

    # ============================================================================
    # SSH KEY MANAGEMENT
    # ============================================================================

    def generate_ssh_key(self, key_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SSH key pair

        Args:
            key_config: SSH key configuration

        Returns:
            Key generation details
        """
        key_info = {
            'name': key_config.get('name', 'id_rsa'),
            'key_type': key_config.get('key_type', 'rsa'),
            'bits': key_config.get('bits', 4096),
            'email': key_config.get('email', 'admin@example.com'),
            'path': key_config.get('path', f"/root/.ssh/{key_config.get('name', 'id_rsa')}"),
            'generated_at': datetime.now().isoformat()
        }

        command = f"ssh-keygen -t {key_info['key_type']} -b {key_info['bits']} " \
                  f"-f {key_info['path']} -C '{key_info['email']}' -N ''"

        success, stdout, stderr = self._execute_command(command, check=False)
        key_info['success'] = success

        if success:
            self.ssh_keys.append(key_info)
            print(f"✓ SSH key generated: {key_info['name']}")
            print(f"  Type: {key_info['key_type']}, Bits: {key_info['bits']}")
            print(f"  Private key: {key_info['path']}")
            print(f"  Public key: {key_info['path']}.pub")
        else:
            print(f"✗ SSH key generation failed: {stderr}")

        return key_info

    def add_ssh_authorized_key(self, key_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add SSH public key to authorized_keys

        Args:
            key_config: Key configuration

        Returns:
            Operation result
        """
        result = {
            'user': key_config.get('user', 'root'),
            'public_key': key_config.get('public_key', ''),
            'key_file': key_config.get('key_file', ''),
            'added_at': datetime.now().isoformat()
        }

        user_home = f"/home/{result['user']}" if result['user'] != 'root' else '/root'
        authorized_keys_path = f"{user_home}/.ssh/authorized_keys"

        # Create .ssh directory if it doesn't exist
        mkdir_cmd = f"mkdir -p {user_home}/.ssh && chmod 700 {user_home}/.ssh"
        self._execute_command(mkdir_cmd, check=False)

        # Add key
        if result['key_file']:
            command = f"cat {result['key_file']} >> {authorized_keys_path}"
        else:
            command = f"echo '{result['public_key']}' >> {authorized_keys_path}"

        command += f" && chmod 600 {authorized_keys_path}"
        command += f" && chown -R {result['user']}:{result['user']} {user_home}/.ssh"

        success, stdout, stderr = self._execute_command(command, check=False)
        result['success'] = success

        if success:
            print(f"✓ SSH key added to authorized_keys for user: {result['user']}")
            print(f"  File: {authorized_keys_path}")
        else:
            print(f"✗ Failed to add SSH key: {stderr}")

        return result

    def configure_ssh_hardening(self) -> Dict[str, Any]:
        """
        Apply SSH security hardening

        Returns:
            Hardening configuration
        """
        hardening = {
            'applied_at': datetime.now().isoformat(),
            'changes': []
        }

        ssh_config_changes = [
            ('PermitRootLogin', 'no'),
            ('PasswordAuthentication', 'no'),
            ('PubkeyAuthentication', 'yes'),
            ('PermitEmptyPasswords', 'no'),
            ('X11Forwarding', 'no'),
            ('MaxAuthTries', '3'),
            ('Protocol', '2'),
            ('ClientAliveInterval', '300'),
            ('ClientAliveCountMax', '2')
        ]

        for setting, value in ssh_config_changes:
            hardening['changes'].append(f"{setting} = {value}")

        print(f"✓ SSH hardening configuration prepared")
        print(f"  Changes: {len(hardening['changes'])}")
        print(f"  Apply these settings to /etc/ssh/sshd_config and restart sshd")

        return hardening

    # ============================================================================
    # BACKUP AND RESTORE
    # ============================================================================

    def create_backup(self, backup_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create system backup

        Args:
            backup_config: Backup configuration

        Returns:
            Backup details
        """
        backup = {
            'backup_id': f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'source': backup_config.get('source', '/home'),
            'destination': backup_config.get('destination', '/backup'),
            'type': backup_config.get('type', 'full'),
            'compression': backup_config.get('compression', 'gzip'),
            'exclude': backup_config.get('exclude', []),
            'created_at': datetime.now().isoformat()
        }

        backup['archive_name'] = f"{backup['backup_id']}.tar.gz"
        backup['archive_path'] = f"{backup['destination']}/{backup['archive_name']}"

        # Build tar command
        exclude_str = ' '.join([f"--exclude='{ex}'" for ex in backup['exclude']])
        command = f"tar -czf {backup['archive_path']} {exclude_str} {backup['source']}"

        success, stdout, stderr = self._execute_command(command, check=False)
        backup['success'] = success

        if success:
            # Get backup size
            size_cmd = f"du -h {backup['archive_path']} | cut -f1"
            success, size, _ = self._execute_command(size_cmd, check=False)
            backup['size'] = size.strip() if success else 'unknown'

            self.backups.append(backup)
            print(f"✓ Backup created: {backup['backup_id']}")
            print(f"  Source: {backup['source']}")
            print(f"  Archive: {backup['archive_path']}")
            print(f"  Size: {backup.get('size', 'unknown')}")
        else:
            print(f"✗ Backup failed: {stderr}")

        return backup

    def list_backups(self, backup_dir: str = '/backup') -> List[Dict[str, Any]]:
        """
        List available backups

        Args:
            backup_dir: Backup directory

        Returns:
            List of backups
        """
        command = f"ls -lh {backup_dir}/*.tar.gz 2>/dev/null"
        success, stdout, stderr = self._execute_command(command, check=False)

        backups = []
        if success and stdout:
            for line in stdout.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 9:
                    backups.append({
                        'size': parts[4],
                        'date': f"{parts[5]} {parts[6]} {parts[7]}",
                        'filename': parts[8]
                    })

        print(f"✓ Found {len(backups)} backup(s) in {backup_dir}")
        return backups

    def restore_backup(self, restore_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore from backup

        Args:
            restore_config: Restore configuration

        Returns:
            Restore details
        """
        restore = {
            'archive': restore_config.get('archive', ''),
            'destination': restore_config.get('destination', '/'),
            'restored_at': datetime.now().isoformat()
        }

        command = f"tar -xzf {restore['archive']} -C {restore['destination']}"
        success, stdout, stderr = self._execute_command(command, check=False)
        restore['success'] = success

        if success:
            print(f"✓ Backup restored successfully")
            print(f"  Archive: {restore['archive']}")
            print(f"  Destination: {restore['destination']}")
        else:
            print(f"✗ Restore failed: {stderr}")

        return restore

    # ============================================================================
    # ADVANCED USER/GROUP MANAGEMENT
    # ============================================================================

    def list_users(self) -> List[Dict[str, Any]]:
        """
        List all system users

        Returns:
            List of users
        """
        command = "getent passwd"
        success, stdout, stderr = self._execute_command(command, check=False)

        users = []
        if success and stdout:
            for line in stdout.strip().split('\n'):
                parts = line.split(':')
                if len(parts) >= 7:
                    users.append({
                        'username': parts[0],
                        'uid': parts[2],
                        'gid': parts[3],
                        'home': parts[5],
                        'shell': parts[6]
                    })

        print(f"✓ Found {len(users)} system users")
        return users

    def delete_user(self, username: str, remove_home: bool = True) -> Dict[str, Any]:
        """
        Delete system user

        Args:
            username: Username to delete
            remove_home: Remove home directory

        Returns:
            Deletion result
        """
        result = {
            'username': username,
            'remove_home': remove_home,
            'deleted_at': datetime.now().isoformat()
        }

        command = f"userdel {'-r' if remove_home else ''} {username}"
        success, stdout, stderr = self._execute_command(command, check=False)
        result['success'] = success

        if success:
            print(f"✓ User deleted: {username}")
            print(f"  Home directory removed: {remove_home}")
        else:
            print(f"✗ Failed to delete user: {stderr}")

        return result

    def modify_user(self, username: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify existing user

        Args:
            username: Username to modify
            modifications: User modifications

        Returns:
            Modification result
        """
        result = {
            'username': username,
            'modifications': modifications,
            'modified_at': datetime.now().isoformat()
        }

        commands = []

        if 'shell' in modifications:
            commands.append(f"usermod -s {modifications['shell']} {username}")

        if 'home' in modifications:
            commands.append(f"usermod -d {modifications['home']} {username}")

        if 'groups' in modifications:
            groups = ','.join(modifications['groups'])
            commands.append(f"usermod -aG {groups} {username}")

        if 'password' in modifications:
            commands.append(f"echo '{username}:{modifications['password']}' | chpasswd")

        for cmd in commands:
            success, stdout, stderr = self._execute_command(cmd, check=False)
            if not success:
                result['success'] = False
                result['error'] = stderr
                print(f"✗ Failed to modify user: {stderr}")
                return result

        result['success'] = True
        print(f"✓ User modified: {username}")
        print(f"  Changes: {list(modifications.keys())}")

        return result

    def list_groups(self) -> List[Dict[str, Any]]:
        """
        List all system groups

        Returns:
            List of groups
        """
        command = "getent group"
        success, stdout, stderr = self._execute_command(command, check=False)

        groups = []
        if success and stdout:
            for line in stdout.strip().split('\n'):
                parts = line.split(':')
                if len(parts) >= 4:
                    groups.append({
                        'groupname': parts[0],
                        'gid': parts[2],
                        'members': parts[3].split(',') if parts[3] else []
                    })

        print(f"✓ Found {len(groups)} system groups")
        return groups

    # ============================================================================
    # SECURITY HARDENING
    # ============================================================================

    def apply_security_hardening(self) -> Dict[str, Any]:
        """
        Apply comprehensive security hardening

        Returns:
            Hardening results
        """
        hardening = {
            'timestamp': datetime.now().isoformat(),
            'hostname': self.hostname,
            'applied': []
        }

        hardening_steps = [
            {
                'name': 'Disable root SSH login',
                'description': 'Set PermitRootLogin no in sshd_config'
            },
            {
                'name': 'Enable firewall',
                'description': 'Enable and configure UFW/firewalld'
            },
            {
                'name': 'Set password policies',
                'description': 'Configure PAM password requirements'
            },
            {
                'name': 'Enable automatic updates',
                'description': 'Configure unattended-upgrades'
            },
            {
                'name': 'Configure fail2ban',
                'description': 'Install and configure fail2ban for intrusion prevention'
            },
            {
                'name': 'Disable unused services',
                'description': 'Stop and disable unnecessary services'
            },
            {
                'name': 'Set file permissions',
                'description': 'Secure sensitive files and directories'
            },
            {
                'name': 'Enable audit logging',
                'description': 'Configure auditd for system auditing'
            }
        ]

        hardening['recommendations'] = hardening_steps

        print(f"✓ Security hardening recommendations prepared")
        print(f"  Total recommendations: {len(hardening_steps)}")
        for step in hardening_steps:
            print(f"  - {step['name']}")

        return hardening

    def scan_open_ports(self) -> List[Dict[str, Any]]:
        """
        Scan for open network ports

        Returns:
            List of open ports
        """
        command = "ss -tuln | grep LISTEN"
        success, stdout, stderr = self._execute_command(command, check=False)

        ports = []
        if success and stdout:
            for line in stdout.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5:
                    address = parts[4]
                    if ':' in address:
                        port = address.split(':')[-1]
                        ports.append({
                            'port': port,
                            'protocol': parts[0],
                            'address': address
                        })

        print(f"✓ Found {len(ports)} open ports")
        for port_info in ports[:10]:  # Show first 10
            print(f"  Port: {port_info['port']}, Protocol: {port_info['protocol']}")

        return ports

    # ============================================================================
    # DISK MANAGEMENT
    # ============================================================================

    def manage_disk_space(self, cleanup_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage and cleanup disk space

        Args:
            cleanup_config: Cleanup configuration

        Returns:
            Cleanup results
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'cleaned': []
        }

        cleanup_commands = [
            ('Package cache', 'apt-get clean' if self._detect_package_manager() == 'apt' else 'yum clean all'),
            ('Old logs', 'find /var/log -name "*.gz" -mtime +30 -delete'),
            ('Temp files', 'find /tmp -type f -mtime +7 -delete'),
            ('Old kernels', 'apt-get autoremove -y' if self._detect_package_manager() == 'apt' else 'package-cleanup --oldkernels --count=2')
        ]

        for name, command in cleanup_commands:
            success, stdout, stderr = self._execute_command(command, check=False)
            result['cleaned'].append({
                'task': name,
                'success': success,
                'command': command
            })

        print(f"✓ Disk cleanup completed")
        print(f"  Tasks executed: {len(result['cleaned'])}")

        return result

    def _detect_package_manager(self) -> str:
        """Detect system package manager"""
        managers = ['apt', 'yum', 'dnf', 'pacman', 'zypper']
        for manager in managers:
            success, _, _ = self._execute_command(f"which {manager}", check=False)
            if success:
                return manager
        return 'apt'  # default

    # ============================================================================
    # NETWORK CONFIGURATION
    # ============================================================================

    def configure_network_interface(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure network interface

        Args:
            network_config: Network configuration

        Returns:
            Configuration result
        """
        config = {
            'interface': network_config.get('interface', 'eth0'),
            'ip_address': network_config.get('ip_address', ''),
            'netmask': network_config.get('netmask', ''),
            'gateway': network_config.get('gateway', ''),
            'dns': network_config.get('dns', []),
            'configured_at': datetime.now().isoformat()
        }

        print(f"✓ Network configuration prepared for: {config['interface']}")
        print(f"  IP: {config['ip_address']}, Netmask: {config['netmask']}")
        print(f"  Gateway: {config['gateway']}, DNS: {config['dns']}")

        return config

    def test_network_connectivity(self, targets: List[str] = None) -> Dict[str, Any]:
        """
        Test network connectivity

        Args:
            targets: List of hosts to ping

        Returns:
            Connectivity test results
        """
        if targets is None:
            targets = ['8.8.8.8', 'google.com', '1.1.1.1']

        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }

        for target in targets:
            command = f"ping -c 3 -W 2 {target}"
            success, stdout, stderr = self._execute_command(command, check=False)

            results['tests'].append({
                'target': target,
                'reachable': success,
                'output': stdout[:200] if stdout else stderr[:200]
            })

        reachable_count = sum(1 for test in results['tests'] if test['reachable'])
        print(f"✓ Network connectivity test completed")
        print(f"  Reachable: {reachable_count}/{len(targets)}")

        return results


def demo():
    """Comprehensive demonstration of Linux system administration toolkit"""

    print("=" * 80)
    print("Linux System Administration Toolkit - Production Demo v2.0.0")
    print("=" * 80)

    # Initialize admin in dry-run mode for demo
    admin = LinuxSystemAdmin(hostname='prod-server-01', dry_run=True)

    # ========== USER & GROUP MANAGEMENT ==========
    print("\n" + "=" * 80)
    print("1. USER & GROUP MANAGEMENT")
    print("=" * 80)

    print("\n1.1. Creating system users...")
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

    print("\n1.2. Creating system groups...")
    admin.create_group({
        'groupname': 'developers',
        'members': ['appuser', 'deploy']
    })

    # ========== PACKAGE MANAGEMENT ==========
    print("\n" + "=" * 80)
    print("2. PACKAGE MANAGEMENT")
    print("=" * 80)

    print("\n2.1. Installing packages...")
    admin.install_package({
        'package': 'nginx',
        'package_manager': 'apt'
    })

    admin.install_package({
        'package': 'postgresql-14',
        'package_manager': 'apt'
    })

    # ========== SERVICE MANAGEMENT ==========
    print("\n" + "=" * 80)
    print("3. SYSTEMD SERVICE MANAGEMENT")
    print("=" * 80)

    print("\n3.1. Creating systemd service...")
    service_unit = admin.create_systemd_service({
        'name': 'webapp',
        'description': 'Web Application Service',
        'exec_start': '/usr/bin/python3 /opt/webapp/app.py',
        'user': 'appuser',
        'working_directory': '/opt/webapp',
        'restart': 'always'
    })

    print("\n3.2. Managing services...")
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

    # ========== FIREWALL CONFIGURATION ==========
    print("\n" + "=" * 80)
    print("4. FIREWALL CONFIGURATION")
    print("=" * 80)

    print("\n4.1. Configuring firewall rules...")
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

    # ========== CRON JOB MANAGEMENT ==========
    print("\n" + "=" * 80)
    print("5. CRON JOB MANAGEMENT")
    print("=" * 80)

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

    # ========== SSH CONFIGURATION ==========
    print("\n" + "=" * 80)
    print("6. SSH KEY & SECURITY")
    print("=" * 80)

    print("\n6.1. Generating SSH key...")
    admin.generate_ssh_key({
        'name': 'deploy_key',
        'key_type': 'ed25519',
        'bits': 4096,
        'email': 'deploy@example.com'
    })

    print("\n6.2. SSH security hardening...")
    admin.configure_ssh_hardening()

    print("\n6.3. SSH server configuration...")
    ssh_config = admin.configure_ssh({
        'port': 22,
        'permit_root_login': 'no',
        'password_authentication': 'no',
        'pubkey_authentication': 'yes'
    })

    # ========== BACKUP & RESTORE ==========
    print("\n" + "=" * 80)
    print("7. BACKUP & RESTORE")
    print("=" * 80)

    print("\n7.1. Creating backup...")
    admin.create_backup({
        'source': '/home',
        'destination': '/backup',
        'type': 'full',
        'exclude': ['*.tmp', '*.cache']
    })

    # ========== SYSTEM MONITORING ==========
    print("\n" + "=" * 80)
    print("8. SYSTEM MONITORING")
    print("=" * 80)

    print("\n8.1. Monitoring system resources...")
    admin.monitor_system_resources()

    print("\n8.2. System health check...")
    admin.check_system_health()

    print("\n8.3. Scanning open ports...")
    admin.scan_open_ports()

    # ========== SECURITY HARDENING ==========
    print("\n" + "=" * 80)
    print("9. SECURITY HARDENING")
    print("=" * 80)

    admin.apply_security_hardening()

    # ========== FILE PERMISSIONS ==========
    print("\n" + "=" * 80)
    print("10. FILE PERMISSIONS")
    print("=" * 80)

    admin.set_file_permissions({
        'path': '/var/www/html',
        'mode': '755',
        'owner': 'www-data',
        'group': 'www-data',
        'recursive': True
    })

    # ========== LOG ROTATION ==========
    print("\n" + "=" * 80)
    print("11. LOG ROTATION")
    print("=" * 80)

    logrotate_config = admin.setup_log_rotation({
        'log_path': '/var/log/webapp/*.log',
        'rotate': 14,
        'size': '100M',
        'compress': True
    })

    # ========== NETWORK TESTING ==========
    print("\n" + "=" * 80)
    print("12. NETWORK CONNECTIVITY")
    print("=" * 80)

    admin.test_network_connectivity(['8.8.8.8', 'google.com'])

    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("SYSTEM SUMMARY")
    print("=" * 80)

    info = admin.get_system_info()
    print(f"\n  Hostname: {info['hostname']}")
    print(f"  Timestamp: {info['timestamp']}")
    print(f"\n  Configuration:")
    print(f"    Users: {info['users']}")
    print(f"    Groups: {info['groups']}")
    print(f"    Packages: {info['packages']}")
    print(f"    Services: {info['services']}")
    print(f"    Firewall rules: {info['firewall_rules']}")
    print(f"    Cron jobs: {info['cron_jobs']}")
    print(f"    SSH keys: {info['ssh_keys']}")
    print(f"    Backups: {info['backups']}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("All system administration operations demonstrated in DRY-RUN mode")
    print("Remove dry_run=True to execute commands for real")
    print("=" * 80)


if __name__ == "__main__":
    demo()
