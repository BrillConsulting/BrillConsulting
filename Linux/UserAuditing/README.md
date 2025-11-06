# User Auditing System

**Version:** 2.0.0
**Author:** BrillConsulting
**License:** MIT

A comprehensive, production-ready Linux user auditing system for monitoring, analyzing, and reporting on user activities across your infrastructure. This enterprise-grade solution provides complete visibility into user behavior, privilege escalation attempts, and security compliance.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Components](#components)
- [Compliance & Security](#compliance--security)
- [SIEM Integration](#siem-integration)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

The User Auditing System is a comprehensive security monitoring solution designed for Linux environments. It collects, analyzes, and reports on all user activities, providing security teams with actionable intelligence to detect threats, ensure compliance, and maintain system integrity.

### Key Capabilities

- **Real-time Monitoring**: Track active user sessions and activities as they occur
- **Historical Analysis**: Deep-dive into past activities with configurable time ranges
- **Threat Detection**: Identify suspicious patterns and potential security breaches
- **Compliance Reporting**: Generate reports for PCI-DSS, HIPAA, SOX, GDPR, ISO27001
- **SIEM Integration**: Export data in multiple formats (JSON, CEF, Syslog)
- **Zero Dependencies**: Works with standard Linux utilities (last, who, auditd)

## Features

### 1. Login Monitoring

Track all user login activities across your systems:

- **Current Sessions**: View all currently logged-in users
- **Login History**: Comprehensive history of all login events
- **Remote Access Tracking**: Monitor SSH and remote desktop connections
- **Suspicious Login Detection**:
  - Multiple failed login attempts
  - Unusual IP addresses
  - Off-hours access (configurable)
  - Brute force attack detection
  - Geographic anomaly detection

### 2. Command History Analysis

Analyze user command execution for security risks:

- **Multi-shell Support**: bash, zsh, and other shell histories
- **Dangerous Command Detection**: Identify potentially harmful commands
- **Network Activity Monitoring**: Track data transfers and remote connections
- **Privilege Escalation Tracking**: Monitor sudo usage and su commands
- **Data Exfiltration Detection**: Identify suspicious file operations
- **System Modification Alerts**: Track configuration changes

### 3. Sudo Log Analysis

Comprehensive sudo activity monitoring:

- **Command Execution Tracking**: Every sudo command with full context
- **Failed Attempt Detection**: Security alert on failed privilege escalations
- **User-to-User Switching**: Monitor lateral movement
- **Shell Access Detection**: Alert on sudo bash/sh access
- **Time-series Analysis**: Pattern detection over time
- **Target User Tracking**: See which accounts users switch to

### 4. File Access Auditing

Monitor access to sensitive system files:

- **Permission Monitoring**: Detect insecure file permissions
- **Ownership Verification**: Ensure critical files have correct ownership
- **Modification Tracking**: Alert on changes to sensitive files
- **Access Pattern Analysis**: Identify unusual access patterns
- **Sensitive Path Protection**: Monitor /etc, /var/log, SSH keys, etc.

### 5. Privilege Escalation Detection

Advanced threat detection mechanisms:

- **SUID Binary Analysis**: Detect suspicious SUID/SGID files
- **Sudoers File Monitoring**: Alert on sudoers modifications
- **User Account Changes**: Track new user creation and modifications
- **Group Membership Tracking**: Monitor privileged group additions
- **Real-time Alerts**: Immediate notification of escalation attempts

### 6. Compliance Reporting

Generate comprehensive compliance reports:

- **Multi-Standard Support**: PCI-DSS, HIPAA, SOX, GDPR, ISO27001
- **Automated Scoring**: Calculate compliance scores (0-100)
- **Risk Assessment**: Identify high-risk users and activities
- **Detailed Findings**: Category-based security findings
- **Actionable Recommendations**: Clear remediation steps
- **Multiple Export Formats**: JSON, HTML, PDF-ready

### 7. Auditd Integration

Native integration with Linux Audit Framework:

- **Status Monitoring**: Check auditd service health
- **Rule Management**: Comprehensive audit rule templates
- **Event Collection**: Parse and analyze audit logs
- **Custom Rules**: User activity monitoring rules
- **Performance Optimization**: Efficient log parsing

### 8. SIEM Export

Export data to your existing SIEM platform:

- **JSON Export**: Structured data for custom integrations
- **CEF Format**: Common Event Format for enterprise SIEMs
- **Syslog Integration**: Real-time streaming to syslog servers
- **Batch Processing**: Efficient bulk data export
- **Custom Formatters**: Extensible export architecture

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  User Auditing Manager                      │
│                    (Main Orchestrator)                      │
└────────┬────────────────────────────────────────────────────┘
         │
         ├──────┬──────┬──────┬──────┬──────┬──────┬──────┬───┐
         │      │      │      │      │      │      │      │   │
         ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼   ▼
      ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
      │Login│Command│Sudo │File │Priv │Audit│SIEM│Comp│
      │Mon │Analyze│Anal │Audit│Esc  │d Int│Exp │Rep │
      └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘
         │      │      │      │      │      │      │      │
         └──────┴──────┴──────┴──────┴──────┴──────┴──────┴───┐
                                                                │
                                                                ▼
                                                    ┌───────────────────┐
                                                    │  Data Outputs     │
                                                    │  - JSON Files     │
                                                    │  - HTML Reports   │
                                                    │  - CEF Logs       │
                                                    │  - Syslog Stream  │
                                                    └───────────────────┘
```

## Installation

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+, CentOS 7+, RHEL 7+, Debian 9+)
- **Python Version**: Python 3.6 or higher
- **Privileges**: Root or sudo access required for full functionality
- **Disk Space**: Minimum 100MB for logs and reports
- **Memory**: Minimum 512MB RAM

### Quick Install

```bash
# Clone repository
cd /opt
git clone https://github.com/BrillConsulting/UserAuditing.git
cd UserAuditing

# Install dependencies
pip3 install -r requirements.txt

# Create log directory
mkdir -p /var/log/user-audit
chmod 750 /var/log/user-audit

# Test installation
python3 user_auditing.py --mode report
```

### Manual Installation

```bash
# Install Python dependencies
pip3 install psutil>=5.9.0 python-dateutil>=2.8.2

# Copy script to system location
cp user_auditing.py /usr/local/bin/user-auditing
chmod +x /usr/local/bin/user-auditing

# Create configuration directory
mkdir -p /etc/user-audit
```

### Optional: Install Auditd

For enhanced monitoring capabilities:

```bash
# Ubuntu/Debian
apt-get install auditd audispd-plugins

# CentOS/RHEL
yum install audit audit-libs

# Enable and start service
systemctl enable auditd
systemctl start auditd
```

## Configuration

### Basic Configuration

The system works out-of-the-box with sensible defaults. For customization:

```python
# Initialize with custom output directory
from user_auditing import UserAuditingManager

manager = UserAuditingManager(output_dir='/custom/path/logs')
```

### Auditd Rules

To enable auditd integration, apply recommended rules:

```bash
# Get recommended rules
python3 user_auditing.py --mode audit | jq '.components.auditd.rules[]'

# Apply rules manually
auditctl -w /etc/passwd -p wa -k passwd_changes
auditctl -w /etc/shadow -p wa -k shadow_changes
auditctl -w /etc/sudoers -p wa -k sudoers_changes
```

### Scheduled Audits

Set up automated audits with cron:

```bash
# Add to crontab
# Daily audit at 2 AM
0 2 * * * /usr/local/bin/user-auditing --mode audit --days 1 > /var/log/user-audit/daily.log 2>&1

# Weekly compliance report
0 3 * * 0 /usr/local/bin/user-auditing --mode report > /var/log/user-audit/weekly_report.txt 2>&1
```

### Syslog Integration

Configure syslog forwarding:

```python
# Example: Export to remote syslog
from user_auditing import UserAuditingManager, AuditEvent

manager = UserAuditingManager()
events = []  # Your audit events

manager.siem_exporter.export_syslog(
    events,
    host='siem.example.com',
    port=514
)
```

## Usage

### Command Line Interface

```bash
# Run full audit (default: 7 days)
python3 user_auditing.py --mode audit

# Custom time range
python3 user_auditing.py --mode audit --days 30

# Generate compliance report
python3 user_auditing.py --mode report

# Real-time monitoring (60 second intervals)
python3 user_auditing.py --mode monitor --interval 60

# Custom output directory
python3 user_auditing.py --mode audit --output /custom/path
```

### Python API

```python
from user_auditing import UserAuditingManager

# Initialize manager
manager = UserAuditingManager(output_dir='/var/log/user-audit')

# Run full audit
results = manager.run_full_audit(days=7)
print(f"Audit ID: {results['audit_id']}")
print(f"Compliance Score: {results['compliance_report']['compliance_score']}")

# Generate summary report
report = manager.generate_summary_report()
print(report)

# Real-time monitoring
manager.monitor_realtime(interval=60)
```

### Individual Components

```python
from user_auditing import (
    LoginMonitor,
    CommandHistoryAnalyzer,
    SudoLogAnalyzer,
    PrivilegeEscalationDetector
)

# Login monitoring
login_mon = LoginMonitor()
current_logins = login_mon.get_current_logins()
history = login_mon.get_login_history(days=7)
suspicious = login_mon.detect_suspicious_logins(history)

# Command history analysis
cmd_analyzer = CommandHistoryAnalyzer()
analysis = cmd_analyzer.get_all_users_analysis()

# Sudo log analysis
sudo_analyzer = SudoLogAnalyzer()
sudo_events = sudo_analyzer.parse_sudo_logs(days=7)
sudo_analysis = sudo_analyzer.analyze_sudo_usage(sudo_events)

# Privilege escalation detection
priv_detector = PrivilegeEscalationDetector()
findings = priv_detector.detect_all()
```

## Components

### LoginMonitor

Tracks user login activities:

- `get_current_logins()` - Currently logged-in users
- `get_login_history(days)` - Historical login data
- `detect_suspicious_logins(events)` - Anomaly detection

### CommandHistoryAnalyzer

Analyzes user command execution:

- `get_user_history(username)` - Get commands for specific user
- `analyze_commands(commands)` - Security analysis of commands
- `get_all_users_analysis()` - System-wide command analysis

### SudoLogAnalyzer

Monitors privilege escalation:

- `parse_sudo_logs(days)` - Extract sudo events from logs
- `analyze_sudo_usage(events)` - Pattern analysis and threat detection

### FileAccessAuditor

Monitors file system security:

- `check_file_permissions(paths)` - Permission verification
- `audit_sensitive_files()` - Comprehensive file security audit

### PrivilegeEscalationDetector

Detects privilege escalation attempts:

- `detect_all()` - Run all detection rules
- SUID file detection
- Sudoers modification detection
- User account change detection
- Group membership tracking

### AuditdIntegration

Integrates with Linux Audit Framework:

- `check_auditd_status()` - Service health check
- `get_audit_events(event_type, hours)` - Retrieve audit events
- `setup_user_audit_rules()` - Get recommended audit rules

### SIEMExporter

Export data to SIEM platforms:

- `export_json(data, filename)` - JSON export
- `export_cef(events)` - Common Event Format
- `export_syslog(events, host, port)` - Syslog streaming

### ComplianceReporter

Generate compliance reports:

- `generate_report(...)` - Create comprehensive compliance report
- `export_report(report, format)` - Export as JSON or HTML

## Compliance & Security

### Supported Standards

- **PCI-DSS**: Payment Card Industry Data Security Standard
- **HIPAA**: Health Insurance Portability and Accountability Act
- **SOX**: Sarbanes-Oxley Act
- **GDPR**: General Data Protection Regulation
- **ISO27001**: Information Security Management

### Compliance Scoring

The system calculates a compliance score (0-100) based on:

- Critical security findings (10 points each)
- High-severity issues (5 points each)
- Medium-severity issues (2 points each)
- Low-severity issues (0.5 points each)

### Security Findings Categories

1. **Authentication**: Login security and access control
2. **Privilege Escalation**: Unauthorized privilege attempts
3. **File Integrity**: File permission and ownership issues
4. **User Management**: Account creation and modification
5. **Command Execution**: Dangerous or suspicious commands

## SIEM Integration

### Supported Formats

#### JSON Export
```json
{
  "audit_id": "a1b2c3d4e5f6",
  "timestamp": "2025-11-06T10:30:00",
  "components": {
    "login_monitoring": {...},
    "sudo_analysis": {...},
    "privilege_escalation": {...}
  }
}
```

#### CEF (Common Event Format)
```
CEF:0|BrillConsulting|UserAuditing|2.0|login|suspicious_login|8|rt=2025-11-06T10:30:00 suser=jdoe act=multiple_ips
```

#### Syslog
```
<4>UserAudit[12345]: suspicious_login by jdoe - {"ips": ["10.0.0.1", "192.168.1.1"]}
```

### Popular SIEM Integrations

- **Splunk**: JSON import via HTTP Event Collector
- **ELK Stack**: JSON or syslog ingestion
- **QRadar**: CEF format via syslog
- **ArcSight**: CEF format via syslog
- **Azure Sentinel**: JSON via Log Analytics API

## API Reference

### UserAuditingManager

Main orchestration class for all auditing operations.

```python
class UserAuditingManager:
    def __init__(self, output_dir: str = '/var/log/user-audit')
    def run_full_audit(self, days: int = 7) -> Dict[str, Any]
    def monitor_realtime(self, interval: int = 60) -> None
    def generate_summary_report(self) -> str
    def execute(self) -> Dict[str, Any]
```

### Data Structures

#### LoginEvent
```python
@dataclass
class LoginEvent:
    username: str
    login_time: str
    logout_time: Optional[str]
    duration: Optional[str]
    terminal: str
    ip_address: Optional[str]
    login_type: str
```

#### SudoEvent
```python
@dataclass
class SudoEvent:
    timestamp: str
    username: str
    command: str
    target_user: str
    terminal: str
    working_directory: str
    success: bool
```

#### ComplianceReport
```python
@dataclass
class ComplianceReport:
    report_id: str
    generated_at: str
    period_start: str
    period_end: str
    total_events: int
    critical_events: int
    high_risk_users: List[str]
    findings: List[Dict[str, Any]]
    compliance_score: float
```

## Best Practices

### Security

1. **Run with Appropriate Privileges**: Use sudo only when necessary
2. **Secure Log Files**: Set permissions to 640 or 600 on audit logs
3. **Rotate Logs**: Implement log rotation to manage disk space
4. **Monitor the Monitor**: Set up alerts if auditing fails
5. **Review Regularly**: Schedule weekly report reviews

### Performance

1. **Adjust Time Windows**: Use shorter time ranges for frequent audits
2. **Filter System Users**: Focus on UIDs >= 1000 for user accounts
3. **Archive Old Data**: Move old audit data to cold storage
4. **Optimize Queries**: Use appropriate time ranges for queries
5. **Resource Monitoring**: Monitor CPU and disk I/O during audits

### Deployment

1. **Test in Non-Production**: Validate configuration before production use
2. **Gradual Rollout**: Deploy to small groups before full deployment
3. **Baseline Normal Activity**: Establish normal patterns before alerting
4. **Document Findings**: Keep records of all security incidents
5. **Update Regularly**: Keep system and dependencies up to date

## Troubleshooting

### Common Issues

#### Permission Denied Errors

```bash
# Run with sudo
sudo python3 user_auditing.py --mode audit

# Or fix log directory permissions
sudo chown $USER:$USER /var/log/user-audit
```

#### Missing Logs

```bash
# Check if log files exist
ls -la /var/log/auth.log /var/log/wtmp

# On CentOS/RHEL, logs may be in different locations
ls -la /var/log/secure
```

#### Auditd Not Available

```bash
# Install auditd
sudo apt-get install auditd  # Ubuntu/Debian
sudo yum install audit       # CentOS/RHEL

# Start service
sudo systemctl start auditd
```

#### High Resource Usage

```bash
# Reduce audit window
python3 user_auditing.py --mode audit --days 1

# Schedule during off-peak hours
# Add to crontab for 2 AM execution
0 2 * * * /path/to/user_auditing.py
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from user_auditing import UserAuditingManager
manager = UserAuditingManager()
manager.run_full_audit(days=7)
```

## Output Files

### JSON Audit Reports

Location: `/var/log/user-audit/audit_*.json`

Contains complete audit results with all components:
- Login monitoring data
- Command history analysis
- Sudo log analysis
- File access audit
- Privilege escalation findings
- Compliance report

### HTML Compliance Reports

Location: `/var/log/user-audit/compliance_report_*.html`

Human-readable compliance reports with:
- Executive summary
- Compliance score
- High-risk users
- Detailed findings
- Recommendations

### CEF Log Files

Location: `/var/log/user-audit/audit_cef_*.log`

SIEM-ready event logs in Common Event Format.

## Performance Metrics

Typical performance on a standard Linux server:

- **Full Audit (7 days)**: 10-30 seconds
- **Real-time Monitoring**: < 1% CPU
- **Memory Usage**: 50-150 MB
- **Disk I/O**: Minimal (sequential reads)
- **Log Size**: ~1-5 MB per day per 100 users

## Security Considerations

### Data Protection

- Audit logs contain sensitive information
- Implement encryption at rest for audit data
- Use secure channels (TLS/SSL) for syslog forwarding
- Restrict access to audit reports
- Comply with data retention policies

### Integrity

- Verify audit log integrity regularly
- Implement tamper detection mechanisms
- Use separate systems for audit storage
- Enable write-once-read-many (WORM) storage

## Roadmap

### Version 2.1 (Planned)

- Machine learning anomaly detection
- Real-time alerting via email/Slack/PagerDuty
- Web-based dashboard
- Multi-host aggregation
- Advanced threat intelligence integration

### Version 2.2 (Future)

- Container environment support
- Cloud platform integration (AWS, Azure, GCP)
- Advanced behavioral analytics
- Automated remediation actions
- Custom rule engine

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

- **Documentation**: https://docs.brillconsulting.com/user-auditing
- **Issues**: https://github.com/BrillConsulting/UserAuditing/issues
- **Email**: support@brillconsulting.com
- **Community**: https://community.brillconsulting.com

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Linux Audit Project
- MITRE ATT&CK Framework
- CIS Benchmarks
- NIST Cybersecurity Framework

---

**BrillConsulting** - Enterprise Security Solutions
Version 2.0.0 | Last Updated: November 2025
