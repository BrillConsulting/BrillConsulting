# Linux Security Hardening System

**Version:** 2.0.0
**Author:** BrillConsulting
**Category:** Enterprise Security & Compliance

A production-ready, comprehensive Linux security hardening framework implementing industry best practices, CIS benchmarks, and automated security controls.

## Overview

This system provides enterprise-grade security hardening capabilities for Linux servers, covering mandatory access controls, file integrity monitoring, antivirus protection, intrusion prevention, vulnerability scanning, and compliance auditing. Designed for DevSecOps teams managing production infrastructure requiring PCI-DSS, HIPAA, NIST, or ISO 27001 compliance.

## Features

### Core Security Components

#### 1. Mandatory Access Control (MAC)
- **SELinux Configuration**
  - Enforcing/permissive mode management
  - Policy type configuration (targeted, strict, MLS)
  - Boolean settings and custom policies
  - Port labeling and context management
  - Automated policy generation

- **AppArmor Profiles**
  - Application confinement policies
  - Profile enforcement/complain modes
  - Custom profile generation for web servers, databases, applications
  - Automated profile learning and tuning
  - Integration with systemd

#### 2. File Integrity Monitoring
- **AIDE (Advanced Intrusion Detection Environment)**
  - Comprehensive file system monitoring
  - Cryptographic hash verification (MD5, SHA256, SHA512)
  - Permission and attribute tracking
  - Automated baseline generation
  - Daily integrity checks via systemd timers
  - Detailed change reporting and alerts
  - Configurable monitoring rules per directory

#### 3. Password Security Policies
- **PAM Configuration**
  - Minimum password length and complexity enforcement
  - Password history tracking (prevent reuse)
  - Account lockout after failed attempts
  - Automatic unlock timers
  - Password aging controls
  - SHA512 password hashing with 65536 rounds
  - Dictionary and palindrome checking
  - Username inclusion prevention

#### 4. Intrusion Prevention
- **fail2ban Configuration**
  - Multi-service jail management (SSH, HTTP, FTP, etc.)
  - Configurable ban times and retry limits
  - IP whitelist/blacklist management
  - Email notifications on bans
  - Log monitoring and pattern matching
  - Integration with firewall (iptables/nftables)
  - Distributed ban synchronization

#### 5. Antivirus Protection
- **ClamAV Integration**
  - Real-time virus scanning
  - Automated signature updates (24 checks/day)
  - Scheduled full system scans
  - Quarantine management for infected files
  - Multiple scan path configuration
  - Email alerts on detection
  - Integration with file system monitors
  - Performance-optimized scanning

#### 6. System Auditing
- **auditd Configuration**
  - Comprehensive system call monitoring
  - User action tracking
  - File access auditing
  - Privilege escalation detection
  - MAC policy change monitoring
  - Network connection logging
  - Tamper-proof audit logs
  - CIS-compliant audit rules

#### 7. SSH Hardening
- **OpenSSH Security**
  - Protocol 2 enforcement
  - Root login prevention
  - Password authentication disabling (key-only)
  - Strong cipher suites (ChaCha20, AES-256-GCM)
  - Modern key exchange algorithms
  - Failed authentication limits
  - Idle timeout configuration
  - User access restrictions

#### 8. Firewall Hardening
- **iptables/nftables Rules**
  - Default-deny policy
  - Connection rate limiting
  - SYN flood protection
  - Port scan detection and blocking
  - ICMP flood protection
  - Stateful packet inspection
  - GeoIP-based blocking (optional)
  - Automated rule persistence

#### 9. Kernel Hardening
- **sysctl Security Parameters**
  - IP forwarding controls
  - ICMP redirect disabling
  - SYN cookie protection
  - Source routing prevention
  - Kernel pointer restriction
  - dmesg access restriction
  - Reverse path filtering
  - ASLR and DEP enforcement

#### 10. Compliance Checking
- **CIS Benchmark Automation**
  - Ubuntu/Debian/RHEL/CentOS support
  - 100+ automated security checks
  - Filesystem configuration auditing
  - Service configuration validation
  - Network security verification
  - Logging and auditing compliance
  - SSH configuration assessment
  - File permission verification
  - Compliance scoring and reporting
  - Automated remediation suggestions

#### 11. Vulnerability Scanning
- **Comprehensive Security Assessment**
  - Package vulnerability detection
  - Configuration weakness identification
  - Permission anomaly scanning
  - SUID/SGID binary auditing
  - World-writable file detection
  - Open port enumeration
  - Kernel vulnerability checking
  - CVSS scoring integration
  - Risk-based prioritization

#### 12. Automated Hardening
- **One-Click Security Baseline**
  - System package updates
  - Security tool installation
  - Unnecessary service disabling
  - File permission corrections
  - Kernel parameter tuning
  - Service enablement and configuration
  - AIDE database initialization
  - ClamAV signature updates
  - Rollback capability

## Architecture

```
SecurityHardening (Main Class)
├── configure_selinux()          # SELinux MAC configuration
├── configure_apparmor()         # AppArmor profile management
├── configure_aide()             # File integrity monitoring
├── configure_password_policies() # PAM security configuration
├── configure_fail2ban()         # Intrusion prevention
├── configure_clamav()           # Antivirus protection
├── configure_auditd()           # System auditing
├── harden_ssh()                 # SSH security hardening
├── configure_firewall_hardening() # Advanced firewall rules
├── configure_kernel_hardening() # Kernel parameter tuning
├── run_cis_benchmark()          # Compliance checking
├── scan_vulnerabilities()       # Security assessment
├── automated_hardening()        # One-click hardening
└── generate_security_report()   # Comprehensive reporting
```

## Installation

```bash
# Clone repository
git clone https://github.com/BrillConsulting/Linux-SecurityHardening.git
cd Linux-SecurityHardening/Linux/SecurityHardening

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    fail2ban aide clamav clamav-daemon \
    auditd audispd-plugins \
    libpam-pwquality libpam-tmpdir \
    apparmor apparmor-utils \
    iptables-persistent

# For RHEL/CentOS
sudo yum install -y \
    fail2ban aide clamav clamav-update \
    audit audispd-plugins \
    pam_pwquality \
    policycoreutils-python-utils \
    iptables-services

# Install Python dependencies (if any)
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from security_hardening import SecurityHardening

# Initialize security system
security = SecurityHardening(hostname='prod-server-01', auto_remediate=False)

# Configure file integrity monitoring
aide_config = security.configure_aide({
    'monitored_paths': ['/etc', '/bin', '/sbin', '/usr/bin', '/usr/sbin'],
    'excluded_paths': ['/var/log', '/tmp']
})

# Set password policies
password_policy = security.configure_password_policies({
    'min_length': 14,
    'min_complexity': 3,
    'max_age': 90,
    'lock_after_failed': 5
})

# Configure antivirus
clamav = security.configure_clamav({
    'scan_paths': ['/home', '/var/www'],
    'scan_frequency': 'daily',
    'quarantine_infected': True
})

# Run CIS benchmark
cis_results = security.run_cis_benchmark(standard='ubuntu20.04')
print(f"Compliance Score: {cis_results['compliance_score']}%")

# Scan for vulnerabilities
vuln_scan = security.scan_vulnerabilities()
print(f"Critical: {vuln_scan['critical_count']}, High: {vuln_scan['high_count']}")

# Automated hardening
hardening = security.automated_hardening({
    'enable_apparmor': True,
    'harden_kernel': True,
    'install_aide': True,
    'install_clamav': True
})
```

### Running the Demo

```bash
# Run comprehensive security demo
python security_hardening.py

# Output shows configuration of all security components
```

### Configuration Files Generated

The system generates production-ready configuration files:

- `/etc/selinux/config` - SELinux configuration
- `/etc/apparmor.d/*` - AppArmor profiles
- `/etc/aide/aide.conf` - File integrity monitoring rules
- `/etc/security/pwquality.conf` - Password quality requirements
- `/etc/pam.d/common-password` - PAM password configuration
- `/etc/pam.d/common-auth` - Account lockout policies
- `/etc/fail2ban/jail.local` - Intrusion prevention rules
- `/etc/clamav/clamd.conf` - Antivirus daemon configuration
- `/etc/clamav/freshclam.conf` - Virus signature updates
- `/etc/audit/rules.d/audit.rules` - System audit rules
- `/etc/ssh/sshd_config` - Hardened SSH configuration
- `/etc/sysctl.d/99-security-hardening.conf` - Kernel parameters
- `/etc/iptables/rules.v4` - Firewall rules

## Compliance Standards

This system implements security controls mapped to:

- **CIS Benchmarks** (Level 1 & 2)
  - Ubuntu 18.04/20.04/22.04
  - Debian 9/10/11
  - RHEL 7/8/9
  - CentOS 7/8

- **PCI-DSS v4.0**
  - Requirement 2: Secure configurations
  - Requirement 8: Strong authentication
  - Requirement 10: Logging and monitoring

- **HIPAA Security Rule**
  - Access controls (§164.312(a))
  - Audit controls (§164.312(b))
  - Integrity controls (§164.312(c))
  - Authentication (§164.312(d))

- **NIST 800-53**
  - AC (Access Control)
  - AU (Audit and Accountability)
  - CM (Configuration Management)
  - IA (Identification and Authentication)
  - SC (System and Communications Protection)

- **ISO 27001:2013**
  - A.9: Access control
  - A.12: Operations security
  - A.14: System acquisition, development and maintenance

## Security Best Practices

### Recommended Deployment Workflow

1. **Assessment Phase**
   ```python
   # Run baseline assessment
   vuln_scan = security.scan_vulnerabilities()
   cis_baseline = security.run_cis_benchmark()
   ```

2. **Planning Phase**
   - Review scan results and compliance gaps
   - Identify critical vulnerabilities
   - Plan remediation priorities

3. **Implementation Phase**
   ```python
   # Apply automated hardening
   security.automated_hardening({
       'enable_apparmor': True,
       'harden_kernel': True,
       'install_aide': True
   })
   ```

4. **Verification Phase**
   ```python
   # Verify hardening
   post_scan = security.scan_vulnerabilities()
   post_cis = security.run_cis_benchmark()
   ```

5. **Monitoring Phase**
   - Enable AIDE daily checks
   - Configure fail2ban email alerts
   - Review audit logs regularly
   - Monitor ClamAV scan results

### Production Considerations

- **Backup Configuration**: Always backup original configs before hardening
- **Testing**: Test hardening in staging environment first
- **Application Compatibility**: Verify applications work with MAC policies
- **Performance Impact**: AIDE and ClamAV scans can be CPU-intensive
- **Maintenance Windows**: Some hardening requires service restarts
- **Documentation**: Document all custom configurations
- **Change Management**: Use version control for security configs

## Monitoring and Maintenance

### Daily Tasks (Automated)
- AIDE file integrity checks
- ClamAV virus scans
- ClamAV signature updates
- Fail2ban ban reports

### Weekly Tasks
- Review audit logs
- Check failed login attempts
- Review firewall logs
- Update security packages

### Monthly Tasks
- Run full vulnerability scan
- CIS benchmark compliance check
- Review and tune MAC policies
- Security configuration review

## Troubleshooting

### Common Issues

**AIDE False Positives**
```bash
# Update AIDE database after legitimate changes
aide --update
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db
```

**SELinux/AppArmor Blocking Applications**
```bash
# Check audit logs
ausearch -m avc -ts recent

# Generate policy exceptions
audit2allow -a -M myapp
semodule -i myapp.pp

# Or switch to permissive temporarily
setenforce 0
```

**Fail2ban Not Banning**
```bash
# Check jail status
fail2ban-client status sshd

# Test regex patterns
fail2ban-regex /var/log/auth.log /etc/fail2ban/filter.d/sshd.conf
```

## Performance Impact

Typical resource usage on production systems:

| Component | CPU Impact | Memory Usage | Disk I/O |
|-----------|-----------|--------------|----------|
| SELinux/AppArmor | <2% | 50-100 MB | Low |
| AIDE (checking) | 5-15% | 100-200 MB | High |
| ClamAV (scanning) | 20-40% | 500 MB - 1 GB | High |
| fail2ban | <1% | 20-50 MB | Low |
| auditd | 1-3% | 50-100 MB | Medium |
| Firewall | <1% | 10-20 MB | Low |

**Optimization Tips:**
- Schedule AIDE and ClamAV scans during off-peak hours
- Exclude cache directories from AIDE monitoring
- Use ionice for ClamAV scans to reduce I/O impact
- Limit ClamAV scan concurrency on busy servers

## Security Considerations

- **Root Access Required**: Most hardening operations require root privileges
- **Service Disruption**: Some hardening may require service restarts
- **SSH Lockout Risk**: Test SSH hardening carefully to avoid lockouts
- **MAC Policy Impact**: SELinux/AppArmor can block legitimate applications
- **Audit Log Size**: auditd can generate large log volumes
- **Backup Critical**: Always maintain backups before hardening

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test thoroughly in lab environment
4. Submit pull request with detailed description
5. Include security impact assessment

## License

Copyright (c) 2024 BrillConsulting. All rights reserved.

This software is provided for educational and professional use.
Always test in non-production environments first.

## Support

- **Documentation**: See inline code documentation
- **Issues**: GitHub issue tracker
- **Security Bugs**: Report privately to security@brillconsulting.com
- **Professional Services**: Available for enterprise deployments

## References

- CIS Benchmarks: https://www.cisecurity.org/cis-benchmarks/
- NIST 800-53: https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final
- AIDE Documentation: https://aide.github.io/
- SELinux Project: https://github.com/SELinuxProject
- AppArmor Wiki: https://gitlab.com/apparmor/apparmor/-/wikis/home
- fail2ban Manual: https://www.fail2ban.org/
- ClamAV Documentation: https://docs.clamav.net/

## Version History

### v2.0.0 (Current)
- Added AIDE file integrity monitoring
- Implemented comprehensive password policies
- Integrated ClamAV antivirus protection
- Added CIS benchmark automation
- Enhanced vulnerability scanning
- Implemented automated hardening workflows
- Added compliance reporting

### v1.0.0
- Initial release
- Basic SELinux/AppArmor support
- fail2ban configuration
- SSH hardening
- Kernel hardening
- Basic vulnerability scanning
