"""
UserAuditing - Production-Ready User Activity Monitoring System
Author: BrillConsulting
Description: Comprehensive user auditing with login monitoring, command history analysis,
             sudo log analysis, privilege escalation detection, and SIEM integration
Version: 2.0.0
"""

import os
import re
import pwd
import grp
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LoginEvent:
    """Login event data structure"""
    username: str
    login_time: str
    logout_time: Optional[str]
    duration: Optional[str]
    terminal: str
    ip_address: Optional[str]
    login_type: str


@dataclass
class SudoEvent:
    """Sudo command event data structure"""
    timestamp: str
    username: str
    command: str
    target_user: str
    terminal: str
    working_directory: str
    success: bool


@dataclass
class AuditEvent:
    """Generic audit event structure"""
    timestamp: str
    event_type: str
    username: str
    action: str
    details: Dict[str, Any]
    severity: str


@dataclass
class ComplianceReport:
    """Compliance report structure"""
    report_id: str
    generated_at: str
    period_start: str
    period_end: str
    total_events: int
    critical_events: int
    high_risk_users: List[str]
    findings: List[Dict[str, Any]]
    compliance_score: float


class LoginMonitor:
    """Monitor and analyze user login activities"""

    def __init__(self):
        self.last_log_path = "/var/log/wtmp"
        self.auth_log_path = "/var/log/auth.log"

    def get_current_logins(self) -> List[Dict[str, Any]]:
        """Get currently logged in users"""
        try:
            result = subprocess.run(
                ['who', '-u'],
                capture_output=True,
                text=True,
                timeout=10
            )

            logins = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        logins.append({
                            'username': parts[0],
                            'terminal': parts[1],
                            'login_time': ' '.join(parts[2:4]),
                            'idle': parts[4] if len(parts) > 4 else 'unknown',
                            'pid': parts[5] if len(parts) > 5 else 'unknown',
                            'ip': parts[6] if len(parts) > 6 else 'local'
                        })

            return logins
        except Exception as e:
            logger.error(f"Error getting current logins: {e}")
            return []

    def get_login_history(self, days: int = 7) -> List[LoginEvent]:
        """Get login history for specified days"""
        try:
            result = subprocess.run(
                ['last', '-F', f'-{days * 50}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            events = []
            for line in result.stdout.strip().split('\n'):
                if line and not line.startswith('wtmp') and not line.startswith('reboot'):
                    match = re.match(
                        r'^(\S+)\s+(\S+)\s+([\d\.:]+|\S+)?\s+(.+?)(?:\s+-\s+(.+?))?(?:\s+\((.+?)\))?$',
                        line
                    )
                    if match:
                        username, terminal, ip, login_time = match.group(1), match.group(2), match.group(3), match.group(4)
                        logout_time = match.group(5)
                        duration = match.group(6)

                        events.append(LoginEvent(
                            username=username,
                            login_time=login_time.strip(),
                            logout_time=logout_time.strip() if logout_time else None,
                            duration=duration,
                            terminal=terminal,
                            ip_address=ip if ip and ip != 'system' else None,
                            login_type='remote' if ip and ':' in ip or (ip and '.' in ip) else 'local'
                        ))

            return events
        except Exception as e:
            logger.error(f"Error getting login history: {e}")
            return []

    def detect_suspicious_logins(self, events: List[LoginEvent]) -> List[Dict[str, Any]]:
        """Detect suspicious login patterns"""
        suspicious = []

        # Group by username
        user_logins = defaultdict(list)
        for event in events:
            user_logins[event.username].append(event)

        for username, logins in user_logins.items():
            # Multiple failed login attempts
            if len(logins) > 10:
                suspicious.append({
                    'type': 'high_frequency_logins',
                    'username': username,
                    'count': len(logins),
                    'severity': 'medium',
                    'description': f'User {username} has {len(logins)} login events'
                })

            # Different IP addresses
            ips = set(login.ip_address for login in logins if login.ip_address)
            if len(ips) > 5:
                suspicious.append({
                    'type': 'multiple_ips',
                    'username': username,
                    'ip_count': len(ips),
                    'ips': list(ips),
                    'severity': 'high',
                    'description': f'User {username} logged in from {len(ips)} different IPs'
                })

            # Off-hours logins (example: 10 PM - 6 AM)
            for login in logins:
                try:
                    # Simple time check (would need proper parsing in production)
                    if any(hour in login.login_time for hour in ['22:', '23:', '00:', '01:', '02:', '03:', '04:', '05:']):
                        suspicious.append({
                            'type': 'off_hours_login',
                            'username': username,
                            'time': login.login_time,
                            'ip': login.ip_address,
                            'severity': 'low',
                            'description': f'User {username} logged in during off-hours'
                        })
                except Exception:
                    pass

        return suspicious


class CommandHistoryAnalyzer:
    """Analyze user command history for security concerns"""

    def __init__(self):
        self.dangerous_commands = [
            'rm -rf', 'dd if=', 'mkfs', 'fdisk', ':(){:|:&};:',  # Dangerous commands
            'wget', 'curl', 'nc -l', 'ncat',  # Network commands
            'chmod 777', 'chmod -R 777',  # Insecure permissions
            'iptables -F', 'ufw disable',  # Firewall manipulation
        ]

    def get_user_history(self, username: str) -> List[str]:
        """Get command history for a user"""
        try:
            user_info = pwd.getpwnam(username)
            home_dir = user_info.pw_dir

            commands = []
            history_files = [
                f'{home_dir}/.bash_history',
                f'{home_dir}/.zsh_history',
                f'{home_dir}/.history'
            ]

            for history_file in history_files:
                if os.path.exists(history_file):
                    try:
                        with open(history_file, 'r', errors='ignore') as f:
                            commands.extend(f.readlines())
                    except PermissionError:
                        logger.warning(f"Permission denied reading {history_file}")

            return [cmd.strip() for cmd in commands if cmd.strip()]
        except Exception as e:
            logger.error(f"Error getting history for {username}: {e}")
            return []

    def analyze_commands(self, commands: List[str]) -> Dict[str, Any]:
        """Analyze commands for security concerns"""
        analysis = {
            'total_commands': len(commands),
            'dangerous_commands': [],
            'network_activity': [],
            'privilege_escalation': [],
            'data_exfiltration': [],
            'system_modification': []
        }

        for cmd in commands:
            cmd_lower = cmd.lower()

            # Check for dangerous commands
            for dangerous in self.dangerous_commands:
                if dangerous in cmd_lower:
                    analysis['dangerous_commands'].append(cmd)
                    break

            # Network activity
            if any(net_cmd in cmd_lower for net_cmd in ['wget', 'curl', 'scp', 'rsync', 'nc', 'netcat']):
                analysis['network_activity'].append(cmd)

            # Privilege escalation attempts
            if any(priv_cmd in cmd_lower for priv_cmd in ['sudo', 'su -', 'pkexec']):
                analysis['privilege_escalation'].append(cmd)

            # Data exfiltration patterns
            if any(exfil in cmd_lower for exfil in ['base64', 'gzip', 'tar -c', 'zip']):
                if any(net in cmd_lower for net in ['>', '|', 'curl', 'wget', 'nc']):
                    analysis['data_exfiltration'].append(cmd)

            # System modifications
            if any(sys_cmd in cmd_lower for sys_cmd in ['systemctl', 'service', 'crontab', 'at ']):
                analysis['system_modification'].append(cmd)

        return analysis

    def get_all_users_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Analyze command history for all users"""
        results = {}

        try:
            for user_entry in pwd.getpwall():
                username = user_entry.pw_name
                # Skip system users
                if user_entry.pw_uid < 1000 and username != 'root':
                    continue

                commands = self.get_user_history(username)
                if commands:
                    results[username] = self.analyze_commands(commands)
        except Exception as e:
            logger.error(f"Error analyzing all users: {e}")

        return results


class SudoLogAnalyzer:
    """Analyze sudo logs for privilege escalation and security issues"""

    def __init__(self):
        self.auth_log_paths = [
            '/var/log/auth.log',
            '/var/log/secure',
            '/var/log/sudo.log'
        ]

    def parse_sudo_logs(self, days: int = 7) -> List[SudoEvent]:
        """Parse sudo logs from auth logs"""
        events = []
        cutoff_date = datetime.now() - timedelta(days=days)

        for log_path in self.auth_log_paths:
            if not os.path.exists(log_path):
                continue

            try:
                with open(log_path, 'r', errors='ignore') as f:
                    for line in f:
                        if 'sudo' in line.lower() and 'COMMAND' in line:
                            event = self._parse_sudo_line(line)
                            if event:
                                events.append(event)
            except PermissionError:
                logger.warning(f"Permission denied reading {log_path}")
            except Exception as e:
                logger.error(f"Error reading {log_path}: {e}")

        return events

    def _parse_sudo_line(self, line: str) -> Optional[SudoEvent]:
        """Parse a single sudo log line"""
        try:
            # Example: Dec  1 10:30:15 hostname sudo:    user : TTY=pts/0 ; PWD=/home/user ; USER=root ; COMMAND=/usr/bin/apt-get update
            match = re.search(
                r'(\w+\s+\d+\s+\d+:\d+:\d+).*?sudo:.*?(\w+)\s*:.*?TTY=(\S+).*?PWD=(\S+).*?USER=(\S+).*?COMMAND=(.+)',
                line
            )

            if match:
                timestamp, username, terminal, pwd, target_user, command = match.groups()
                return SudoEvent(
                    timestamp=timestamp,
                    username=username,
                    command=command.strip(),
                    target_user=target_user,
                    terminal=terminal,
                    working_directory=pwd,
                    success='incorrect password' not in line.lower()
                )
        except Exception as e:
            logger.debug(f"Error parsing sudo line: {e}")

        return None

    def analyze_sudo_usage(self, events: List[SudoEvent]) -> Dict[str, Any]:
        """Analyze sudo usage patterns"""
        analysis = {
            'total_commands': len(events),
            'users': defaultdict(int),
            'failed_attempts': 0,
            'privileged_commands': [],
            'suspicious_commands': [],
            'user_switching': []
        }

        suspicious_patterns = ['bash', 'sh', '/bin/', 'passwd', 'usermod', 'userdel', 'chmod', 'chown']

        for event in events:
            analysis['users'][event.username] += 1

            if not event.success:
                analysis['failed_attempts'] += 1

            # Check for shell access
            cmd_lower = event.command.lower()
            if any(pattern in cmd_lower for pattern in suspicious_patterns):
                analysis['suspicious_commands'].append({
                    'user': event.username,
                    'command': event.command,
                    'timestamp': event.timestamp,
                    'target_user': event.target_user
                })

            # User switching
            if event.target_user != 'root':
                analysis['user_switching'].append({
                    'user': event.username,
                    'target': event.target_user,
                    'command': event.command,
                    'timestamp': event.timestamp
                })

        analysis['users'] = dict(analysis['users'])
        return analysis


class FileAccessAuditor:
    """Monitor and audit file access patterns"""

    def __init__(self):
        self.sensitive_paths = [
            '/etc/passwd',
            '/etc/shadow',
            '/etc/sudoers',
            '/etc/ssh',
            '/root',
            '/var/log',
            '/home/*/.ssh'
        ]

    def check_file_permissions(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Check permissions on sensitive files"""
        issues = []

        for path in paths:
            if not os.path.exists(path):
                continue

            try:
                stat_info = os.stat(path)
                mode = oct(stat_info.st_mode)[-3:]

                # Check for world-writable
                if mode[-1] in ['2', '3', '6', '7']:
                    issues.append({
                        'path': path,
                        'issue': 'world_writable',
                        'permissions': mode,
                        'severity': 'critical'
                    })

                # Check for world-readable on sensitive files
                if path in ['/etc/shadow', '/etc/sudoers'] and mode[-1] in ['4', '5', '6', '7']:
                    issues.append({
                        'path': path,
                        'issue': 'world_readable',
                        'permissions': mode,
                        'severity': 'critical'
                    })
            except Exception as e:
                logger.debug(f"Error checking {path}: {e}")

        return issues

    def audit_sensitive_files(self) -> Dict[str, Any]:
        """Audit sensitive file access and permissions"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'permission_issues': [],
            'ownership_issues': [],
            'modifications': []
        }

        # Check permissions
        results['permission_issues'] = self.check_file_permissions([
            '/etc/passwd',
            '/etc/shadow',
            '/etc/sudoers',
            '/etc/ssh/sshd_config'
        ])

        # Check ownership
        for path in ['/etc/passwd', '/etc/shadow', '/etc/sudoers']:
            if os.path.exists(path):
                try:
                    stat_info = os.stat(path)
                    if stat_info.st_uid != 0:
                        results['ownership_issues'].append({
                            'path': path,
                            'owner_uid': stat_info.st_uid,
                            'expected_uid': 0,
                            'severity': 'critical'
                        })
                except Exception as e:
                    logger.debug(f"Error checking ownership of {path}: {e}")

        return results


class PrivilegeEscalationDetector:
    """Detect potential privilege escalation attempts"""

    def __init__(self):
        self.detection_rules = {
            'suid_files': self._check_suid_files,
            'sudoers_modifications': self._check_sudoers,
            'user_additions': self._check_new_users,
            'group_changes': self._check_group_changes
        }

    def _check_suid_files(self) -> List[Dict[str, Any]]:
        """Check for suspicious SUID files"""
        suspicious = []
        try:
            result = subprocess.run(
                ['find', '/usr', '/bin', '/sbin', '-perm', '-4000', '-type', 'f', '2>/dev/null'],
                capture_output=True,
                text=True,
                timeout=60,
                shell=True
            )

            # Known safe SUID binaries
            safe_suid = {'/usr/bin/sudo', '/usr/bin/su', '/usr/bin/passwd', '/bin/mount', '/bin/umount'}

            for line in result.stdout.strip().split('\n'):
                if line and line not in safe_suid:
                    suspicious.append({
                        'type': 'suspicious_suid',
                        'path': line,
                        'severity': 'high'
                    })
        except Exception as e:
            logger.error(f"Error checking SUID files: {e}")

        return suspicious

    def _check_sudoers(self) -> List[Dict[str, Any]]:
        """Check sudoers file for recent modifications"""
        issues = []
        sudoers_path = '/etc/sudoers'

        if os.path.exists(sudoers_path):
            try:
                stat_info = os.stat(sudoers_path)
                mtime = datetime.fromtimestamp(stat_info.st_mtime)

                # If modified in last 24 hours
                if datetime.now() - mtime < timedelta(days=1):
                    issues.append({
                        'type': 'sudoers_modified',
                        'path': sudoers_path,
                        'modified': mtime.isoformat(),
                        'severity': 'medium'
                    })
            except Exception as e:
                logger.debug(f"Error checking sudoers: {e}")

        return issues

    def _check_new_users(self) -> List[Dict[str, Any]]:
        """Check for recently added users"""
        new_users = []

        try:
            # Check passwd file modification time
            passwd_stat = os.stat('/etc/passwd')
            passwd_mtime = datetime.fromtimestamp(passwd_stat.st_mtime)

            if datetime.now() - passwd_mtime < timedelta(days=7):
                # List all users with UID >= 1000 (excluding system users)
                for user_entry in pwd.getpwall():
                    if user_entry.pw_uid >= 1000 or user_entry.pw_name == 'root':
                        new_users.append({
                            'type': 'user_account',
                            'username': user_entry.pw_name,
                            'uid': user_entry.pw_uid,
                            'home': user_entry.pw_dir,
                            'shell': user_entry.pw_shell,
                            'severity': 'medium' if user_entry.pw_uid >= 1000 else 'high'
                        })
        except Exception as e:
            logger.error(f"Error checking new users: {e}")

        return new_users

    def _check_group_changes(self) -> List[Dict[str, Any]]:
        """Check for suspicious group memberships"""
        suspicious = []

        privileged_groups = ['sudo', 'wheel', 'root', 'admin', 'adm']

        try:
            for group_name in privileged_groups:
                try:
                    group_info = grp.getgrnam(group_name)
                    for member in group_info.gr_mem:
                        suspicious.append({
                            'type': 'privileged_group_member',
                            'username': member,
                            'group': group_name,
                            'gid': group_info.gr_gid,
                            'severity': 'medium'
                        })
                except KeyError:
                    pass
        except Exception as e:
            logger.error(f"Error checking groups: {e}")

        return suspicious

    def detect_all(self) -> Dict[str, Any]:
        """Run all privilege escalation detection checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'findings': []
        }

        for rule_name, rule_func in self.detection_rules.items():
            try:
                findings = rule_func()
                results['findings'].extend(findings)
            except Exception as e:
                logger.error(f"Error running {rule_name}: {e}")

        return results


class AuditdIntegration:
    """Integration with Linux auditd subsystem"""

    def __init__(self):
        self.auditd_log = '/var/log/audit/audit.log'
        self.aureport_cmd = 'aureport'
        self.ausearch_cmd = 'ausearch'

    def check_auditd_status(self) -> Dict[str, Any]:
        """Check if auditd is running and configured"""
        status = {
            'installed': False,
            'running': False,
            'log_exists': False,
            'rules_count': 0
        }

        try:
            # Check if auditctl is available
            result = subprocess.run(
                ['which', 'auditctl'],
                capture_output=True,
                timeout=5
            )
            status['installed'] = result.returncode == 0

            # Check if service is running
            result = subprocess.run(
                ['systemctl', 'is-active', 'auditd'],
                capture_output=True,
                text=True,
                timeout=5
            )
            status['running'] = result.stdout.strip() == 'active'

            # Check if log exists
            status['log_exists'] = os.path.exists(self.auditd_log)

            # Count audit rules
            if status['installed']:
                result = subprocess.run(
                    ['auditctl', '-l'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                status['rules_count'] = len([line for line in result.stdout.split('\n') if line.strip()])

        except Exception as e:
            logger.error(f"Error checking auditd status: {e}")

        return status

    def get_audit_events(self, event_type: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get audit events from auditd"""
        events = []

        if not os.path.exists(self.auditd_log):
            logger.warning("Audit log not found")
            return events

        try:
            cmd = ['ausearch', '-ts', f'-{hours}h']
            if event_type:
                cmd.extend(['-m', event_type])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse ausearch output
            for line in result.stdout.split('\n'):
                if line.startswith('type='):
                    events.append({'raw': line.strip()})

        except FileNotFoundError:
            logger.warning("ausearch command not found")
        except Exception as e:
            logger.error(f"Error getting audit events: {e}")

        return events

    def setup_user_audit_rules(self) -> Dict[str, Any]:
        """Return recommended audit rules for user monitoring"""
        rules = {
            'description': 'Recommended auditd rules for user monitoring',
            'rules': [
                # Monitor user authentication
                '-w /var/log/auth.log -p wa -k auth_log',
                '-w /var/log/secure -p wa -k secure_log',

                # Monitor user/group modifications
                '-w /etc/passwd -p wa -k passwd_changes',
                '-w /etc/shadow -p wa -k shadow_changes',
                '-w /etc/group -p wa -k group_changes',
                '-w /etc/sudoers -p wa -k sudoers_changes',

                # Monitor privileged commands
                '-a always,exit -F arch=b64 -S execve -F euid=0 -k root_commands',
                '-a always,exit -F arch=b32 -S execve -F euid=0 -k root_commands',

                # Monitor sudo
                '-w /usr/bin/sudo -p x -k sudo_execution',

                # Monitor SSH
                '-w /etc/ssh/sshd_config -p wa -k sshd_config',

                # Monitor file deletions
                '-a always,exit -F arch=b64 -S unlink -S unlinkat -S rename -S renameat -k delete'
            ]
        }

        return rules


class SIEMExporter:
    """Export audit data in various SIEM-compatible formats"""

    def __init__(self, output_dir: str = '/var/log/user-audit'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_json(self, data: Dict[str, Any], filename: str) -> str:
        """Export data as JSON"""
        filepath = os.path.join(self.output_dir, f"{filename}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Exported JSON to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            return ""

    def export_cef(self, events: List[AuditEvent]) -> str:
        """Export in Common Event Format (CEF) for SIEM ingestion"""
        filepath = os.path.join(self.output_dir, f"audit_cef_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        try:
            with open(filepath, 'w') as f:
                for event in events:
                    cef_line = self._format_cef(event)
                    f.write(cef_line + '\n')

            logger.info(f"Exported CEF to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error exporting CEF: {e}")
            return ""

    def _format_cef(self, event: AuditEvent) -> str:
        """Format event in CEF format"""
        # CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
        cef_header = "CEF:0|BrillConsulting|UserAuditing|2.0|{event_type}|{action}|{severity}".format(
            event_type=event.event_type,
            action=event.action,
            severity=self._severity_to_number(event.severity)
        )

        extensions = f"rt={event.timestamp} suser={event.username} act={event.action}"

        return f"{cef_header}|{extensions}"

    def _severity_to_number(self, severity: str) -> int:
        """Convert severity string to CEF number"""
        mapping = {
            'low': 3,
            'medium': 5,
            'high': 8,
            'critical': 10
        }
        return mapping.get(severity.lower(), 5)

    def export_syslog(self, events: List[AuditEvent], host: str = 'localhost', port: int = 514) -> bool:
        """Export events via syslog protocol"""
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            for event in events:
                message = f"<{self._severity_to_syslog(event.severity)}>UserAudit[{os.getpid()}]: {event.action} by {event.username} - {json.dumps(event.details)}"
                sock.sendto(message.encode(), (host, port))

            sock.close()
            logger.info(f"Exported {len(events)} events via syslog")
            return True
        except Exception as e:
            logger.error(f"Error exporting via syslog: {e}")
            return False

    def _severity_to_syslog(self, severity: str) -> int:
        """Convert severity to syslog priority"""
        mapping = {
            'low': 6,      # Informational
            'medium': 5,   # Notice
            'high': 4,     # Warning
            'critical': 2  # Critical
        }
        return mapping.get(severity.lower(), 5)


class ComplianceReporter:
    """Generate compliance reports for various standards"""

    def __init__(self):
        self.standards = ['PCI-DSS', 'HIPAA', 'SOX', 'GDPR', 'ISO27001']

    def generate_report(
        self,
        login_events: List[LoginEvent],
        sudo_events: List[SudoEvent],
        privilege_findings: Dict[str, Any],
        file_audit: Dict[str, Any],
        period_days: int = 30
    ) -> ComplianceReport:
        """Generate comprehensive compliance report"""

        report_id = hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:16]
        period_end = datetime.now()
        period_start = period_end - timedelta(days=period_days)

        findings = []
        critical_count = 0
        high_risk_users = set()

        # Analyze login events
        suspicious_logins = self._analyze_logins(login_events)
        findings.extend(suspicious_logins)

        # Analyze sudo usage
        sudo_findings = self._analyze_sudo(sudo_events)
        findings.extend(sudo_findings)

        # Analyze privilege escalation
        priv_findings = self._analyze_privilege_escalation(privilege_findings)
        findings.extend(priv_findings)

        # Analyze file access
        file_findings = self._analyze_file_access(file_audit)
        findings.extend(file_findings)

        # Count critical events and identify high-risk users
        for finding in findings:
            if finding.get('severity') == 'critical':
                critical_count += 1
            if finding.get('severity') in ['critical', 'high']:
                if 'username' in finding:
                    high_risk_users.add(finding['username'])

        # Calculate compliance score (0-100)
        compliance_score = self._calculate_compliance_score(findings, len(login_events) + len(sudo_events))

        report = ComplianceReport(
            report_id=report_id,
            generated_at=datetime.now().isoformat(),
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
            total_events=len(login_events) + len(sudo_events),
            critical_events=critical_count,
            high_risk_users=list(high_risk_users),
            findings=findings,
            compliance_score=compliance_score
        )

        return report

    def _analyze_logins(self, events: List[LoginEvent]) -> List[Dict[str, Any]]:
        """Analyze login events for compliance"""
        findings = []

        # Check for root logins
        root_logins = [e for e in events if e.username == 'root']
        if root_logins:
            findings.append({
                'category': 'authentication',
                'finding': 'Direct root login detected',
                'count': len(root_logins),
                'severity': 'high',
                'recommendation': 'Disable direct root login and use sudo instead',
                'compliance_ref': ['PCI-DSS 8.1', 'SOX AC-2']
            })

        return findings

    def _analyze_sudo(self, events: List[SudoEvent]) -> List[Dict[str, Any]]:
        """Analyze sudo events for compliance"""
        findings = []

        # Check for failed sudo attempts
        failed = [e for e in events if not e.success]
        if len(failed) > 10:
            findings.append({
                'category': 'privilege_escalation',
                'finding': 'Multiple failed sudo attempts',
                'count': len(failed),
                'severity': 'medium',
                'recommendation': 'Review user access and investigate failed attempts',
                'compliance_ref': ['PCI-DSS 10.2.4', 'ISO27001 A.9.4.2']
            })

        return findings

    def _analyze_privilege_escalation(self, priv_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze privilege escalation findings"""
        findings = []

        if priv_data.get('findings'):
            for finding in priv_data['findings']:
                if finding.get('type') == 'suspicious_suid':
                    findings.append({
                        'category': 'privilege_escalation',
                        'finding': 'Suspicious SUID binary detected',
                        'details': finding,
                        'severity': 'high',
                        'recommendation': 'Review and remove unnecessary SUID binaries',
                        'compliance_ref': ['PCI-DSS 2.2', 'CIS Benchmark 1.6.1']
                    })

        return findings

    def _analyze_file_access(self, file_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze file access patterns"""
        findings = []

        if file_data.get('permission_issues'):
            for issue in file_data['permission_issues']:
                findings.append({
                    'category': 'file_integrity',
                    'finding': f"Permission issue: {issue['issue']}",
                    'details': issue,
                    'severity': issue.get('severity', 'medium'),
                    'recommendation': 'Fix file permissions to follow security best practices',
                    'compliance_ref': ['PCI-DSS 2.2.4', 'HIPAA 164.312(a)(1)']
                })

        return findings

    def _calculate_compliance_score(self, findings: List[Dict[str, Any]], total_events: int) -> float:
        """Calculate overall compliance score"""
        if not findings:
            return 100.0

        # Deduct points based on severity
        deductions = {
            'critical': 10,
            'high': 5,
            'medium': 2,
            'low': 0.5
        }

        total_deduction = 0
        for finding in findings:
            severity = finding.get('severity', 'low')
            total_deduction += deductions.get(severity, 0)

        # Calculate score (minimum 0, maximum 100)
        score = max(0, 100 - total_deduction)

        return round(score, 2)

    def export_report(self, report: ComplianceReport, format: str = 'json') -> str:
        """Export compliance report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format == 'json':
            filename = f"compliance_report_{timestamp}.json"
            filepath = f"/var/log/user-audit/{filename}"

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)

            return filepath

        elif format == 'html':
            filename = f"compliance_report_{timestamp}.html"
            filepath = f"/var/log/user-audit/{filename}"

            html_content = self._generate_html_report(report)

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                f.write(html_content)

            return filepath

        return ""

    def _generate_html_report(self, report: ComplianceReport) -> str:
        """Generate HTML compliance report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Compliance Report {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .critical {{ color: #e74c3c; font-weight: bold; }}
        .high {{ color: #e67e22; font-weight: bold; }}
        .medium {{ color: #f39c12; }}
        .low {{ color: #27ae60; }}
        .score {{ font-size: 48px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border: 1px solid #ddd; }}
        th {{ background-color: #34495e; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>User Audit Compliance Report</h1>
        <p>Report ID: {report.report_id}</p>
        <p>Generated: {report.generated_at}</p>
        <p>Period: {report.period_start} to {report.period_end}</p>
    </div>

    <div class="section">
        <h2>Compliance Score</h2>
        <div class="score">{report.compliance_score}/100</div>
    </div>

    <div class="section">
        <h2>Summary</h2>
        <p>Total Events: {report.total_events}</p>
        <p>Critical Events: <span class="critical">{report.critical_events}</span></p>
        <p>High-Risk Users: {len(report.high_risk_users)}</p>
        <p>Total Findings: {len(report.findings)}</p>
    </div>

    <div class="section">
        <h2>High-Risk Users</h2>
        <ul>
            {''.join(f'<li>{user}</li>' for user in report.high_risk_users)}
        </ul>
    </div>

    <div class="section">
        <h2>Findings</h2>
        <table>
            <tr>
                <th>Severity</th>
                <th>Category</th>
                <th>Finding</th>
                <th>Recommendation</th>
            </tr>
            {''.join(f'''<tr>
                <td class="{finding.get('severity', 'low')}">{finding.get('severity', 'N/A').upper()}</td>
                <td>{finding.get('category', 'N/A')}</td>
                <td>{finding.get('finding', 'N/A')}</td>
                <td>{finding.get('recommendation', 'N/A')}</td>
            </tr>''' for finding in report.findings)}
        </table>
    </div>
</body>
</html>
        """

        return html


class UserAuditingManager:
    """Main manager class coordinating all auditing components"""

    def __init__(self, output_dir: str = '/var/log/user-audit'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.login_monitor = LoginMonitor()
        self.command_analyzer = CommandHistoryAnalyzer()
        self.sudo_analyzer = SudoLogAnalyzer()
        self.file_auditor = FileAccessAuditor()
        self.privilege_detector = PrivilegeEscalationDetector()
        self.auditd_integration = AuditdIntegration()
        self.siem_exporter = SIEMExporter(output_dir)
        self.compliance_reporter = ComplianceReporter()

        logger.info("UserAuditingManager initialized")

    def run_full_audit(self, days: int = 7) -> Dict[str, Any]:
        """Run complete audit of all user activities"""
        logger.info(f"Starting full audit for last {days} days")

        results = {
            'audit_id': hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:16],
            'timestamp': datetime.now().isoformat(),
            'period_days': days,
            'components': {}
        }

        try:
            # Login monitoring
            logger.info("Collecting login data...")
            current_logins = self.login_monitor.get_current_logins()
            login_history = self.login_monitor.get_login_history(days)
            suspicious_logins = self.login_monitor.detect_suspicious_logins(login_history)

            results['components']['login_monitoring'] = {
                'current_logins': current_logins,
                'login_history_count': len(login_history),
                'suspicious_logins': suspicious_logins
            }

            # Command history analysis
            logger.info("Analyzing command history...")
            command_analysis = self.command_analyzer.get_all_users_analysis()
            results['components']['command_history'] = command_analysis

            # Sudo log analysis
            logger.info("Analyzing sudo logs...")
            sudo_events = self.sudo_analyzer.parse_sudo_logs(days)
            sudo_analysis = self.sudo_analyzer.analyze_sudo_usage(sudo_events)
            results['components']['sudo_analysis'] = sudo_analysis

            # File access auditing
            logger.info("Auditing file access...")
            file_audit = self.file_auditor.audit_sensitive_files()
            results['components']['file_audit'] = file_audit

            # Privilege escalation detection
            logger.info("Detecting privilege escalation...")
            privilege_findings = self.privilege_detector.detect_all()
            results['components']['privilege_escalation'] = privilege_findings

            # Auditd integration
            logger.info("Checking auditd status...")
            auditd_status = self.auditd_integration.check_auditd_status()
            results['components']['auditd'] = auditd_status

            # Generate compliance report
            logger.info("Generating compliance report...")
            compliance_report = self.compliance_reporter.generate_report(
                login_history,
                sudo_events,
                privilege_findings,
                file_audit,
                days
            )
            results['compliance_report'] = asdict(compliance_report)

            # Export results
            logger.info("Exporting results...")
            json_path = self.siem_exporter.export_json(results, f"audit_{results['audit_id']}")
            results['exports'] = {'json': json_path}

            # Export compliance report
            report_path = self.compliance_reporter.export_report(compliance_report, 'html')
            results['exports']['compliance_html'] = report_path

            logger.info("Full audit completed successfully")

        except Exception as e:
            logger.error(f"Error during full audit: {e}")
            results['error'] = str(e)

        return results

    def monitor_realtime(self, interval: int = 60):
        """Monitor user activities in real-time"""
        logger.info(f"Starting real-time monitoring (interval: {interval}s)")

        import time

        try:
            while True:
                # Get current logins
                current_logins = self.login_monitor.get_current_logins()

                # Create audit events
                events = []
                for login in current_logins:
                    event = AuditEvent(
                        timestamp=datetime.now().isoformat(),
                        event_type='active_session',
                        username=login['username'],
                        action='active_login',
                        details=login,
                        severity='low'
                    )
                    events.append(event)

                # Export to SIEM
                if events:
                    self.siem_exporter.export_json(
                        {'events': [asdict(e) for e in events]},
                        f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )

                logger.info(f"Monitored {len(current_logins)} active sessions")
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Real-time monitoring stopped")

    def generate_summary_report(self) -> str:
        """Generate human-readable summary report"""
        results = self.run_full_audit(days=7)

        summary = []
        summary.append("=" * 80)
        summary.append("USER AUDITING SUMMARY REPORT")
        summary.append("=" * 80)
        summary.append(f"Report ID: {results['audit_id']}")
        summary.append(f"Generated: {results['timestamp']}")
        summary.append(f"Period: Last {results['period_days']} days")
        summary.append("")

        # Login monitoring
        if 'login_monitoring' in results['components']:
            login_data = results['components']['login_monitoring']
            summary.append("LOGIN MONITORING")
            summary.append("-" * 80)
            summary.append(f"Current Active Sessions: {len(login_data['current_logins'])}")
            summary.append(f"Total Login Events: {login_data['login_history_count']}")
            summary.append(f"Suspicious Login Patterns: {len(login_data['suspicious_logins'])}")
            summary.append("")

        # Compliance
        if 'compliance_report' in results:
            comp = results['compliance_report']
            summary.append("COMPLIANCE STATUS")
            summary.append("-" * 80)
            summary.append(f"Compliance Score: {comp['compliance_score']}/100")
            summary.append(f"Total Events: {comp['total_events']}")
            summary.append(f"Critical Events: {comp['critical_events']}")
            summary.append(f"High-Risk Users: {len(comp['high_risk_users'])}")
            summary.append("")

        summary.append("=" * 80)

        return '\n'.join(summary)

    def execute(self) -> Dict[str, Any]:
        """Execute default audit operation"""
        return self.run_full_audit(days=7)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='User Auditing System')
    parser.add_argument('--mode', choices=['audit', 'monitor', 'report'], default='audit',
                       help='Operation mode')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to audit')
    parser.add_argument('--output', default='/var/log/user-audit',
                       help='Output directory')
    parser.add_argument('--interval', type=int, default=60,
                       help='Monitor interval in seconds')

    args = parser.parse_args()

    manager = UserAuditingManager(output_dir=args.output)

    if args.mode == 'audit':
        results = manager.run_full_audit(days=args.days)
        print(json.dumps(results, indent=2, default=str))

    elif args.mode == 'monitor':
        manager.monitor_realtime(interval=args.interval)

    elif args.mode == 'report':
        report = manager.generate_summary_report()
        print(report)


if __name__ == "__main__":
    main()
