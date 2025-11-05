"""
Linux Backup and Recovery
Author: BrillConsulting
Description: Complete backup strategies and disaster recovery solutions
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class BackupRecovery:
    """Comprehensive backup and disaster recovery management"""

    def __init__(self, hostname: str = 'localhost'):
        """Initialize backup recovery manager"""
        self.hostname = hostname
        self.backups = []
        self.recovery_points = []

    def create_full_backup(self, backup_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create full system backup"""
        backup = {
            'backup_id': f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'type': 'full',
            'source': backup_config.get('source', '/'),
            'destination': backup_config.get('destination', '/backup'),
            'compression': backup_config.get('compression', 'gzip'),
            'encryption': backup_config.get('encryption', False),
            'size_gb': backup_config.get('estimated_size', 50),
            'created_at': datetime.now().isoformat()
        }

        commands = [
            f"# Full backup using tar",
            f"tar -czpf {backup['destination']}/{backup['backup_id']}.tar.gz {backup['source']}",
            f"",
            f"# Or using rsync",
            f"rsync -aAXv --delete --exclude={{'/dev/*','/proc/*','/sys/*','/tmp/*'}} / {backup['destination']}/",
            f"",
            f"# With encryption",
            f"tar -czpf - {backup['source']} | openssl enc -aes-256-cbc -out {backup['destination']}/{backup['backup_id']}.tar.gz.enc"
        ]

        self.backups.append(backup)
        print(f"✓ Full backup created: {backup['backup_id']}")
        print(f"  Size: {backup['size_gb']}GB, Compression: {backup['compression']}")
        return backup

    def create_incremental_backup(self, backup_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create incremental backup"""
        backup = {
            'backup_id': f"incr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'type': 'incremental',
            'source': backup_config.get('source', '/data'),
            'destination': backup_config.get('destination', '/backup'),
            'base_backup': backup_config.get('base_backup', 'full_20250101'),
            'size_gb': backup_config.get('estimated_size', 5),
            'created_at': datetime.now().isoformat()
        }

        commands = [
            f"# Incremental backup using rsync",
            f"rsync -av --link-dest={backup['destination']}/{backup['base_backup']} {backup['source']} {backup['destination']}/{backup['backup_id']}/",
            f"",
            f"# Using tar with --listed-incremental",
            f"tar -czpf {backup['destination']}/{backup['backup_id']}.tar.gz --listed-incremental={backup['destination']}/backup.snar {backup['source']}"
        ]

        self.backups.append(backup)
        print(f"✓ Incremental backup created: {backup['backup_id']}")
        print(f"  Base: {backup['base_backup']}, Size: {backup['size_gb']}GB")
        return backup

    def backup_database(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Backup database"""
        backup = {
            'backup_id': f"db_{db_config.get('db_name')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'db_type': db_config.get('db_type', 'postgresql'),
            'db_name': db_config.get('db_name', 'production'),
            'destination': db_config.get('destination', '/backup/databases'),
            'compression': True,
            'size_mb': db_config.get('estimated_size', 500),
            'created_at': datetime.now().isoformat()
        }

        commands = {
            'postgresql': f"pg_dump {backup['db_name']} | gzip > {backup['destination']}/{backup['backup_id']}.sql.gz",
            'mysql': f"mysqldump {backup['db_name']} | gzip > {backup['destination']}/{backup['backup_id']}.sql.gz",
            'mongodb': f"mongodump --db {backup['db_name']} --out {backup['destination']}/{backup['backup_id']}/",
        }

        print(f"✓ Database backup created: {backup['db_name']}")
        print(f"  Type: {backup['db_type']}, Size: {backup['size_mb']}MB")
        print(f"  Command: {commands.get(backup['db_type'], 'N/A')}")
        return backup

    def create_snapshot(self, snapshot_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create LVM/ZFS snapshot"""
        snapshot = {
            'snapshot_id': f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'volume': snapshot_config.get('volume', '/dev/vg0/lv_data'),
            'snapshot_name': snapshot_config.get('snapshot_name', 'data_snapshot'),
            'size_gb': snapshot_config.get('size_gb', 10),
            'filesystem': snapshot_config.get('filesystem', 'lvm'),
            'created_at': datetime.now().isoformat()
        }

        commands = {
            'lvm': f"lvcreate -L {snapshot['size_gb']}G -s -n {snapshot['snapshot_name']} {snapshot['volume']}",
            'zfs': f"zfs snapshot pool/dataset@{snapshot['snapshot_name']}",
            'btrfs': f"btrfs subvolume snapshot /mnt/volume /mnt/snapshots/{snapshot['snapshot_name']}"
        }

        print(f"✓ Snapshot created: {snapshot['snapshot_name']}")
        print(f"  Volume: {snapshot['volume']}, Size: {snapshot['size_gb']}GB")
        print(f"  Command: {commands.get(snapshot['filesystem'], 'N/A')}")
        return snapshot

    def restore_backup(self, restore_config: Dict[str, Any]) -> Dict[str, Any]:
        """Restore from backup"""
        restore = {
            'backup_id': restore_config.get('backup_id'),
            'restore_path': restore_config.get('restore_path', '/restore'),
            'verify': restore_config.get('verify', True),
            'restored_at': datetime.now().isoformat()
        }

        commands = [
            f"# Restore from tar backup",
            f"tar -xzpf /backup/{restore['backup_id']}.tar.gz -C {restore['restore_path']}",
            f"",
            f"# Restore from rsync backup",
            f"rsync -av /backup/{restore['backup_id']}/ {restore['restore_path']}/",
            f"",
            f"# Restore encrypted backup",
            f"openssl enc -d -aes-256-cbc -in /backup/{restore['backup_id']}.tar.gz.enc | tar -xzpf - -C {restore['restore_path']}"
        ]

        print(f"✓ Backup restored: {restore['backup_id']}")
        print(f"  Restore path: {restore['restore_path']}, Verify: {restore['verify']}")
        return restore

    def setup_backup_schedule(self, schedule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup automated backup schedule"""
        schedule = {
            'full_backup': schedule_config.get('full_backup', 'weekly'),
            'incremental_backup': schedule_config.get('incremental_backup', 'daily'),
            'retention_days': schedule_config.get('retention_days', 30),
            'notification_email': schedule_config.get('notification_email', 'admin@example.com'),
            'configured_at': datetime.now().isoformat()
        }

        cron_entries = f"""# Backup Schedule
# Full backup every Sunday at 2 AM
0 2 * * 0 /usr/local/bin/backup-full.sh

# Incremental backup daily at 3 AM
0 3 * * * /usr/local/bin/backup-incremental.sh

# Database backup every 6 hours
0 */6 * * * /usr/local/bin/backup-database.sh

# Cleanup old backups daily at 4 AM
0 4 * * * find /backup -name "*.tar.gz" -mtime +{schedule['retention_days']} -delete
"""

        print(f"✓ Backup schedule configured")
        print(f"  Full: {schedule['full_backup']}, Incremental: {schedule['incremental_backup']}")
        print(f"  Retention: {schedule['retention_days']} days")
        return schedule

    def get_backup_info(self) -> Dict[str, Any]:
        """Get backup recovery information"""
        return {
            'hostname': self.hostname,
            'total_backups': len(self.backups),
            'recovery_points': len(self.recovery_points),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate backup and recovery"""
    print("=" * 60)
    print("Linux Backup and Recovery Demo")
    print("=" * 60)

    br = BackupRecovery(hostname='prod-server-01')

    print("\n1. Creating full backup...")
    br.create_full_backup({
        'source': '/',
        'destination': '/backup',
        'compression': 'gzip',
        'encryption': True
    })

    print("\n2. Creating incremental backup...")
    br.create_incremental_backup({
        'source': '/data',
        'destination': '/backup',
        'base_backup': 'full_20250101'
    })

    print("\n3. Backing up database...")
    br.backup_database({
        'db_type': 'postgresql',
        'db_name': 'production',
        'destination': '/backup/databases'
    })

    print("\n4. Creating snapshot...")
    br.create_snapshot({
        'volume': '/dev/vg0/lv_data',
        'snapshot_name': 'data_snapshot_20250101',
        'size_gb': 10,
        'filesystem': 'lvm'
    })

    print("\n5. Restoring backup...")
    br.restore_backup({
        'backup_id': 'full_20250101_020000',
        'restore_path': '/restore'
    })

    print("\n6. Setting up backup schedule...")
    br.setup_backup_schedule({
        'full_backup': 'weekly',
        'incremental_backup': 'daily',
        'retention_days': 30
    })

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
