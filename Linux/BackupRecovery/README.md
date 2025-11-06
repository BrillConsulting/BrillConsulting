# Linux Backup and Recovery System

Enterprise-grade backup strategies and disaster recovery solutions for Linux systems. A production-ready Python framework for comprehensive backup management with automated scheduling, verification, and recovery testing.

## Version 2.0.0

## Overview

This system provides a complete, production-ready backup and disaster recovery solution designed for enterprise Linux environments. It supports multiple backup strategies, automated scheduling, verification, encryption, compression, and remote synchronization.

## Key Features

### Backup Strategies
- **Full Backups**: Complete system or directory backups using rsync or tar
- **Incremental Backups**: Backup only changes since last backup (any type)
- **Differential Backups**: Backup only changes since last full backup
- **LVM Snapshots**: Point-in-time snapshots for instant recovery points
- **Database Backups**: Native support for PostgreSQL, MySQL, MongoDB

### Advanced Capabilities
- **Multiple Methods**: rsync (fast, space-efficient) or tar (portable, compressed)
- **Compression**: gzip, bzip2, xz, zstd, or none
- **Encryption**: AES-256-CBC, AES-128-CBC, GPG, or none
- **Verification**: SHA-256 checksum validation
- **Recovery Testing**: Automated test restores in isolated environments
- **Remote Sync**: Automatic synchronization to remote backup servers
- **Retention Policies**: Configurable daily, weekly, and monthly retention
- **Automated Scheduling**: Cron-based scheduling with generated scripts
- **Metadata Tracking**: Complete backup history and statistics
- **Bandwidth Limiting**: Control backup bandwidth usage

## Architecture

### Core Components

1. **BackupRecovery Class**: Main orchestration engine
2. **Backup Methods**: Pluggable backup execution strategies
3. **Verification Engine**: Integrity checking and validation
4. **Retention Manager**: Automated cleanup based on policies
5. **Scheduler**: Cron job generation and management
6. **Metadata Store**: JSON-based backup catalog

### Backup Types

```python
class BackupType(Enum):
    FULL = "full"           # Complete backup
    INCREMENTAL = "incremental"  # Changes since last backup
    DIFFERENTIAL = "differential"  # Changes since last full
    SNAPSHOT = "snapshot"   # LVM/ZFS/Btrfs snapshot
```

### Backup Status

```python
class BackupStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
```

## Installation

### Prerequisites

```bash
# System packages required
sudo apt-get install -y rsync tar gzip bzip2 xz-utils zstd openssl

# For LVM snapshots
sudo apt-get install -y lvm2

# For database backups
sudo apt-get install -y postgresql-client mysql-client mongodb-clients

# Python 3.8+
python3 --version
```

### Setup

```bash
# Clone or copy the project
cd /home/user/BrillConsulting/Linux/BackupRecovery

# Install (no external dependencies required)
pip install -r requirements.txt

# Create necessary directories
sudo mkdir -p /backup
sudo mkdir -p /var/lib/backup
sudo mkdir -p /var/log

# Set permissions
sudo chown $USER:$USER /backup /var/lib/backup
```

## Configuration

### Configuration File

Create `/etc/backup_recovery.json`:

```json
{
  "hostname": "prod-server-01",
  "backup_root": "/backup",
  "retention_days": 30,
  "retention_weeks": 4,
  "retention_months": 12,
  "compression": "zstd",
  "encryption": "aes-256-cbc",
  "verify_backups": true,
  "remote_enabled": true,
  "remote_host": "backup.example.com",
  "remote_path": "/backup/servers",
  "remote_user": "backup",
  "notification_email": "admin@example.com",
  "max_parallel_backups": 2,
  "bandwidth_limit_mbps": 100,
  "metadata_file": "/var/lib/backup/metadata.json"
}
```

### Environment Variables

```bash
# Encryption password (recommended over hardcoding)
export BACKUP_ENCRYPTION_KEY="your-secure-password"

# Remote backup SSH key
export BACKUP_SSH_KEY="/home/user/.ssh/backup_key"
```

## Usage

### Basic Usage

```python
from backup_recovery import BackupRecovery

# Initialize with config file
br = BackupRecovery(config_file='/etc/backup_recovery.json')

# Or use defaults
br = BackupRecovery()
```

### Creating Backups

#### Full Backup

```python
# Full backup with rsync (recommended for large filesystems)
full_backup = br.create_full_backup({
    'source': '/',
    'destination': '/backup',
    'method': 'rsync',
    'compression': 'zstd',
    'encryption': 'aes-256-cbc'
})

# Full backup with tar (portable archives)
full_backup = br.create_full_backup({
    'source': '/data',
    'destination': '/backup',
    'method': 'tar',
    'compression': 'gzip'
})
```

#### Incremental Backup

```python
# Incremental backup (changes since last backup)
incremental = br.create_incremental_backup({
    'source': '/data',
    'destination': '/backup',
    'method': 'rsync'
})
```

#### Differential Backup

```python
# Differential backup (changes since last full backup)
differential = br.create_differential_backup({
    'source': '/data',
    'destination': '/backup',
    'method': 'tar',
    'compression': 'xz'
})
```

#### LVM Snapshot

```python
# Create LVM snapshot
snapshot = br.create_lvm_snapshot({
    'volume': '/dev/vg0/lv_data',
    'snapshot_name': 'data_snapshot',
    'size_gb': 10,
    'mount_point': '/mnt/snapshots/data'
})
```

#### Database Backup

```python
# PostgreSQL backup
db_backup = br.backup_database({
    'db_type': 'postgresql',
    'db_name': 'production',
    'db_host': 'localhost',
    'db_port': 5432,
    'compression': 'gzip',
    'encryption': 'aes-256-cbc'
})

# MySQL backup
db_backup = br.backup_database({
    'db_type': 'mysql',
    'db_name': 'app_db',
    'db_host': '10.0.1.5',
    'db_port': 3306
})

# MongoDB backup
db_backup = br.backup_database({
    'db_type': 'mongodb',
    'db_name': 'analytics',
    'db_host': 'mongo.local',
    'db_port': 27017
})
```

### Verification and Testing

#### Verify Backup

```python
# Verify backup integrity
verified = br.verify_backup(backup)
print(f"Status: {verified['status']}")
print(f"Checksum: {verified['checksum']}")
```

#### Test Recovery

```python
# Test backup recovery in isolated environment
test_result = br.test_recovery('full_20250106_020000', {
    'test_path': '/tmp/backup_test',
    'cleanup': True
})
print(f"Test status: {test_result['status']}")
```

### Restore Operations

```python
# Restore from backup
restore = br.restore_backup({
    'backup_id': 'full_20250106_020000',
    'restore_path': '/restore',
    'verify': True
})

print(f"Restore completed: {restore['status']}")
```

### Remote Synchronization

```python
# Sync backup to remote server
synced = br.sync_to_remote(backup)
print(f"Remote sync: {synced['remote_synced']}")
```

### Retention Policy

```python
# Apply retention policy (remove old backups)
br.apply_retention_policy()

# Policy:
# - Keep all backups for retention_days (30 days)
# - Keep weekly backups for retention_weeks (4 weeks)
# - Keep monthly backups for retention_months (12 months)
```

### Automated Scheduling

```python
# Setup automated backup schedule
schedule = br.setup_backup_schedule({
    'full_backup_schedule': '0 2 * * 0',      # Sunday 2 AM
    'incremental_backup_schedule': '0 3 * * *',  # Daily 3 AM
    'differential_backup_schedule': '0 1 * * 6',  # Saturday 1 AM
    'database_backup_schedule': '0 */6 * * *',   # Every 6 hours
    'retention_schedule': '0 4 * * *',        # Daily 4 AM
    'verification_schedule': '0 5 * * 1'      # Monday 5 AM
})

# Generate backup scripts
scripts = br.generate_backup_scripts('/usr/local/bin')
```

### Statistics and Reporting

```python
# Get backup statistics
stats = br.get_backup_statistics()
print(f"Total backups: {stats['total_backups']}")
print(f"Total size: {stats['total_size_gb']} GB")
print(f"Backup types: {stats['backup_types']}")

# Export comprehensive report
report = br.export_backup_report('/var/log/backup_report.json')
```

## Backup Strategies

### Strategy 1: Full + Incremental (Recommended)

Best for: Daily operations with frequent changes

```
Sunday:    Full backup
Monday:    Incremental (changes since Sunday)
Tuesday:   Incremental (changes since Monday)
Wednesday: Incremental (changes since Tuesday)
...
```

**Pros**: Efficient storage, fast daily backups
**Cons**: Restore requires full + all incrementals

### Strategy 2: Full + Differential

Best for: Weekly cycles with moderate changes

```
Sunday:    Full backup
Monday:    Differential (changes since Sunday)
Tuesday:   Differential (changes since Sunday)
Wednesday: Differential (changes since Sunday)
...
```

**Pros**: Faster restore (only full + last differential)
**Cons**: Larger differential backups over time

### Strategy 3: Full Weekly + Incremental Daily

Best for: Production environments

```
Week 1:
  Sunday:   Full backup
  Mon-Sat:  Incremental backups

Week 2:
  Sunday:   Full backup
  Mon-Sat:  Incremental backups
```

### Strategy 4: LVM Snapshots + Scheduled Backups

Best for: High-availability systems

```
Continuous: LVM snapshots every hour (fast, space-efficient)
Daily:      Incremental backup from snapshot
Weekly:     Full backup to external storage
```

## Command-Line Tools

### Manual Backup Commands

```bash
# Full backup with rsync
rsync -aAXv --delete \
  --exclude=/dev/* --exclude=/proc/* --exclude=/sys/* \
  / /backup/full_$(date +%Y%m%d)/

# Incremental backup with tar
tar -czpf /backup/incr_$(date +%Y%m%d).tar.gz \
  --listed-incremental=/backup/backup.snar /data

# Database backup
pg_dump production | gzip > /backup/db_$(date +%Y%m%d).sql.gz

# Encrypted backup
tar -czpf - /data | openssl enc -aes-256-cbc -salt \
  -out /backup/encrypted_$(date +%Y%m%d).tar.gz.enc
```

### LVM Snapshot Commands

```bash
# Create snapshot
lvcreate -L 10G -s -n data_snap /dev/vg0/lv_data

# Mount snapshot
mkdir -p /mnt/snapshots/data
mount /dev/vg0/data_snap /mnt/snapshots/data

# Backup from snapshot
rsync -aAXv /mnt/snapshots/data/ /backup/snap_$(date +%Y%m%d)/

# Remove snapshot
umount /mnt/snapshots/data
lvremove -f /dev/vg0/data_snap
```

### Restore Commands

```bash
# Restore from rsync backup
rsync -aAXv /backup/full_20250106/ /restore/

# Restore from tar backup
tar -xzpf /backup/full_20250106.tar.gz -C /restore/

# Restore encrypted backup
openssl enc -d -aes-256-cbc \
  -in /backup/encrypted_20250106.tar.gz.enc | \
  tar -xzpf - -C /restore/

# Restore database
gunzip < /backup/db_20250106.sql.gz | psql production
```

## Monitoring and Alerting

### Log Files

```bash
# Main backup log
tail -f /var/log/backup_recovery.log

# Individual backup logs
tail -f /var/log/backup_full.log
tail -f /var/log/backup_incremental.log
tail -f /var/log/backup_database.log
```

### Email Notifications

Add to cron scripts:

```bash
# Send email on failure
if [ $? -ne 0 ]; then
  echo "Backup failed on $(hostname)" | \
    mail -s "BACKUP FAILED" admin@example.com
fi
```

### Integration with Monitoring Systems

```python
# Export metrics for Prometheus/Grafana
stats = br.get_backup_statistics()
with open('/var/lib/node_exporter/backup_metrics.prom', 'w') as f:
    f.write(f'backup_total_count {stats["total_backups"]}\n')
    f.write(f'backup_completed_count {stats["completed_backups"]}\n')
    f.write(f'backup_failed_count {stats["failed_backups"]}\n')
    f.write(f'backup_size_bytes {stats["total_size_bytes"]}\n')
```

## Disaster Recovery Procedures

### Complete System Recovery

1. **Boot from Live CD/USB**
2. **Partition and format disks**
   ```bash
   fdisk /dev/sda
   mkfs.ext4 /dev/sda1
   mount /dev/sda1 /mnt
   ```

3. **Restore backup**
   ```bash
   rsync -aAXv /backup/latest/ /mnt/
   ```

4. **Reinstall bootloader**
   ```bash
   grub-install --root-directory=/mnt /dev/sda
   update-grub
   ```

5. **Reboot**

### Database Recovery

```bash
# PostgreSQL point-in-time recovery
pg_restore -d production /backup/db_latest.dump

# MySQL recovery
mysql production < /backup/db_latest.sql

# MongoDB recovery
mongorestore --db production /backup/db_latest/
```

## Best Practices

1. **Follow 3-2-1 Rule**: 3 copies, 2 different media, 1 offsite
2. **Test Restores Regularly**: Monthly recovery tests
3. **Encrypt Sensitive Data**: Always encrypt production backups
4. **Monitor Backup Jobs**: Set up alerts for failures
5. **Document Procedures**: Keep recovery procedures updated
6. **Verify Checksums**: Always verify backup integrity
7. **Rotate Encryption Keys**: Change keys quarterly
8. **Use LVM Snapshots**: For consistent backups of live systems
9. **Bandwidth Limiting**: Prevent backup jobs from saturating network
10. **Keep Metadata**: Maintain backup catalog for quick recovery

## Troubleshooting

### Backup Fails with "No space left on device"

```bash
# Check disk space
df -h /backup

# Apply retention policy
python3 -c "from backup_recovery import BackupRecovery; BackupRecovery().apply_retention_policy()"
```

### Restore Fails with Permission Errors

```bash
# Restore with root permissions
sudo rsync -aAXv /backup/latest/ /restore/

# Fix ownership after restore
sudo chown -R user:user /restore/
```

### Incremental Backup Not Working

```bash
# Check base backup exists
ls -la /backup/

# Verify metadata file
cat /var/lib/backup/metadata.json

# Recreate full backup if needed
```

### Remote Sync Fails

```bash
# Test SSH connection
ssh backup@backup.example.com

# Check SSH key permissions
chmod 600 ~/.ssh/backup_key

# Test rsync manually
rsync -avz /backup/ backup@backup.example.com:/backup/servers/
```

## Performance Optimization

### Large Filesystems

- Use rsync instead of tar (faster, incremental)
- Enable compression at source (zstd recommended)
- Use bandwidth limiting during business hours
- Split backups into multiple volumes

### Database Backups

- Use native dump tools (pg_dump, mysqldump)
- Backup from replicas, not masters
- Use streaming backups for large databases
- Schedule during low-traffic periods

### Network Optimization

- Use compression for remote transfers
- Enable SSH multiplexing
- Use rsync's --bwlimit option
- Consider dedicated backup network

## Security Considerations

1. **Encryption**: Always encrypt backups containing sensitive data
2. **Key Management**: Store encryption keys securely (HashiCorp Vault, AWS KMS)
3. **Access Control**: Restrict backup directory permissions (700)
4. **SSH Keys**: Use dedicated SSH keys for backup operations
5. **Audit Logs**: Enable logging for all backup operations
6. **Network Security**: Use VPN or private networks for remote sync
7. **Data Classification**: Implement different policies for different data types

## API Reference

### Class: BackupRecovery

#### Methods

- `create_full_backup(config)`: Create full backup
- `create_incremental_backup(config)`: Create incremental backup
- `create_differential_backup(config)`: Create differential backup
- `create_lvm_snapshot(config)`: Create LVM snapshot
- `backup_database(config)`: Backup database
- `verify_backup(backup)`: Verify backup integrity
- `test_recovery(backup_id, config)`: Test backup recovery
- `restore_backup(config)`: Restore from backup
- `sync_to_remote(backup)`: Sync to remote server
- `apply_retention_policy()`: Apply retention rules
- `setup_backup_schedule(config)`: Setup cron schedule
- `generate_backup_scripts(dir)`: Generate backup scripts
- `get_backup_statistics()`: Get statistics
- `export_backup_report(file)`: Export JSON report

## Technologies Used

- **Python 3.8+**: Core implementation
- **rsync**: Fast incremental file transfer
- **tar**: Archive creation and extraction
- **gzip/bzip2/xz/zstd**: Compression algorithms
- **OpenSSL**: Encryption (AES-256)
- **LVM**: Logical Volume Management snapshots
- **PostgreSQL/MySQL/MongoDB**: Database dump tools
- **Cron**: Job scheduling
- **SHA-256**: Checksum verification
- **JSON**: Metadata storage

## Use Cases

- **Production Server Backups**: Automated daily backups with retention
- **Database Backup Automation**: Continuous database protection
- **Disaster Recovery Planning**: Complete system recovery procedures
- **Point-in-Time Recovery**: LVM snapshots for instant rollback
- **Data Migration**: Safe data transfer between systems
- **Compliance**: Automated backups for regulatory requirements
- **Development Environments**: Quick backup/restore for testing
- **Multi-Server Deployments**: Centralized backup management

## License

Copyright (c) 2025 BrillConsulting. All rights reserved.

## Support

For issues, questions, or contributions:
- Email: support@brillconsulting.com
- Documentation: /home/user/BrillConsulting/Linux/BackupRecovery/README.md

## Version History

- **2.0.0** (2025-01-06): Production-ready release with comprehensive features
- **1.0.0** (2025-01-05): Initial implementation
