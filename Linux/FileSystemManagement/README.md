# FileSystemManagement v2.0.0

Production-grade filesystem management system for Linux environments with comprehensive support for filesystem operations, storage management, and monitoring.

**Author:** BrillConsulting
**Version:** 2.0.0
**Python Requirements:** Python 3.8+

## Overview

FileSystemManagement is an enterprise-ready system providing complete control over Linux filesystem operations including mount/unmount operations, LVM management, RAID configuration, quota management, filesystem checking/repair, usage monitoring, inode tracking, and NFS/CIFS integration.

## Features

### Core Filesystem Operations
- **Create Filesystems**: Support for ext3/ext4, XFS, Btrfs, F2FS, NTFS, VFAT, ExFAT
- **Mount/Unmount**: Safe mounting and unmounting with options support
- **Filesystem Checking**: Integrity checking and automatic repair (e2fsck, xfs_repair)
- **Mount Point Management**: Comprehensive mount point information and statistics

### Disk Management
- **Disk Discovery**: Automatic detection of all block devices
- **SMART Monitoring**: Disk health monitoring via smartctl
- **Disk Wiping**: Secure disk wiping with zero/random methods
- **Partition Information**: Detailed partition layout and metadata

### LVM (Logical Volume Manager)
- **Physical Volumes**: Create and manage PVs
- **Volume Groups**: Create, extend, and manage VGs
- **Logical Volumes**: Create, extend, and manage LVs
- **Snapshots**: Create and manage LVM snapshots
- **JSON Reporting**: Complete LVM state reporting

### RAID Management
- **RAID Creation**: Support for RAID 0, 1, 5, 6, 10
- **Status Monitoring**: Real-time RAID array status
- **Device Management**: Add/remove devices and spares
- **mdstat Parsing**: Complete RAID status from /proc/mdstat

### Quota Management
- **User Quotas**: Set and manage user disk quotas
- **Group Quotas**: Set and manage group disk quotas
- **Quota Reports**: Comprehensive quota usage reporting
- **Enable/Disable**: Dynamic quota activation

### Monitoring & Analytics
- **Disk Usage**: Real-time disk space monitoring
- **Inode Tracking**: Inode usage statistics and alerts
- **I/O Statistics**: Read/write performance metrics
- **Large File Detection**: Find space-consuming files
- **Directory Sizing**: Calculate directory sizes

### Network Filesystems
- **NFS Support**: Export and mount NFS shares
- **CIFS/SMB**: Mount Windows/Samba shares
- **Mount Listing**: Track all network mounts
- **Authentication**: Username/password/domain support

## Architecture

```
FileSystemManagementOrchestrator
├── FilesystemManager (mount/unmount, create, check)
├── DiskManager (disk discovery, SMART, wipe)
├── LVMManager (PV/VG/LV operations, snapshots)
├── RAIDManager (RAID creation, status, device management)
├── QuotaManager (user/group quotas, reports)
├── MonitoringManager (usage, inodes, I/O stats)
├── NFSManager (NFS export, mount, list)
└── CIFSManager (CIFS/SMB mount, list)
```

## Installation

### Prerequisites

```bash
# Install system dependencies
sudo apt-get install -y lvm2 mdadm smartmontools nfs-common cifs-utils quota

# Install Python dependencies
pip install -r requirements.txt
```

### Package Installation

```bash
cd /home/user/BrillConsulting/Linux/FileSystemManagement
pip install -r requirements.txt
```

## Usage Examples

### Basic Usage

```python
from file_system_management import FileSystemManagementOrchestrator

# Initialize orchestrator
orchestrator = FileSystemManagementOrchestrator()

# Get system overview
overview = orchestrator.get_system_overview()
print(f"Mounted filesystems: {overview['overview']['mounts']['count']}")

# Perform health check
health = orchestrator.health_check()
if health['success']:
    print("System health check passed")
```

### Filesystem Operations

```python
from file_system_management import FilesystemManager, FilesystemType

fs_mgr = FilesystemManager()

# Create ext4 filesystem
result = fs_mgr.create_filesystem(
    device='/dev/sdb1',
    fstype=FilesystemType.EXT4,
    label='data',
    force=True
)

# Mount filesystem
result = fs_mgr.mount_filesystem(
    device='/dev/sdb1',
    mountpoint='/mnt/data',
    options=['rw', 'noatime']
)

# Check filesystem
result = fs_mgr.check_filesystem(
    device='/dev/sdb1',
    fstype=FilesystemType.EXT4,
    auto_repair=True
)

# List all mounts
mounts = fs_mgr.get_mounts()
for mount in mounts['mounts']:
    print(f"{mount['device']} on {mount['mountpoint']} ({mount['percent_used']}% used)")
```

### LVM Management

```python
from file_system_management import LVMManager

lvm_mgr = LVMManager()

# Create physical volume
result = lvm_mgr.create_physical_volume('/dev/sdb')

# Create volume group
result = lvm_mgr.create_volume_group('vg_data', ['/dev/sdb', '/dev/sdc'])

# Create logical volume
result = lvm_mgr.create_logical_volume('vg_data', 'lv_data', '50G')

# Extend logical volume
result = lvm_mgr.extend_logical_volume('vg_data', 'lv_data', '20G')

# Create snapshot
result = lvm_mgr.create_snapshot('vg_data', 'lv_data', 'lv_snap', '10G')

# List all volume groups
vgs = lvm_mgr.list_volume_groups()
print(f"Volume groups: {len(vgs['volume_groups'])}")
```

### RAID Configuration

```python
from file_system_management import RAIDManager, RAIDLevel

raid_mgr = RAIDManager()

# Create RAID 5 array
result = raid_mgr.create_raid(
    raid_device='/dev/md0',
    level=RAIDLevel.RAID5,
    devices=['/dev/sdb', '/dev/sdc', '/dev/sdd'],
    spare_devices=['/dev/sde']
)

# Get RAID status
status = raid_mgr.get_raid_status()
print(status['mdstat'])

# Add spare device
result = raid_mgr.add_spare_device('/dev/md0', '/dev/sdf')

# Remove device
result = raid_mgr.remove_device('/dev/md0', '/dev/sdd', fail_first=True)
```

### Quota Management

```python
from file_system_management import QuotaManager

quota_mgr = QuotaManager()

# Enable quotas
result = quota_mgr.enable_quota('/dev/sda1', user_quota=True, group_quota=True)

# Set user quota
result = quota_mgr.set_user_quota(
    user='john',
    filesystem='/dev/sda1',
    block_soft=1000000,  # 1GB soft limit
    block_hard=2000000,  # 2GB hard limit
    inode_soft=10000,
    inode_hard=20000
)

# Get user quota
quota = quota_mgr.get_user_quota('john')
print(quota['quota_info'])

# Get quota report
report = quota_mgr.get_quota_report('/dev/sda1')
print(report['report'])
```

### Monitoring & Analytics

```python
from file_system_management import MonitoringManager

monitor_mgr = MonitoringManager()

# Get disk usage
usage = monitor_mgr.get_disk_usage('/')
print(f"Disk usage: {usage['percent']}% ({usage['used_gb']}GB / {usage['total_gb']}GB)")

# Get inode usage
inodes = monitor_mgr.get_inode_usage('/')
print(f"Inode usage: {inodes['inodes_percent']}")

# Get I/O statistics
io_stats = monitor_mgr.get_io_stats()
for disk, stats in io_stats['io_stats'].items():
    print(f"{disk}: Read {stats['read_mb']}MB, Write {stats['write_mb']}MB")

# Find large files
large_files = monitor_mgr.find_large_files('/home', size_mb=100, limit=10)
print(f"Found {large_files['files_found']} files larger than 100MB")

# Get directory size
size = monitor_mgr.get_directory_size('/var/log')
print(f"Directory size: {size['size']}")
```

### NFS Operations

```python
from file_system_management import NFSManager

nfs_mgr = NFSManager()

# Export directory
result = nfs_mgr.export_directory(
    directory='/data/shared',
    client='192.168.1.0/24',
    options=['rw', 'sync', 'no_subtree_check']
)

# Mount NFS share
result = nfs_mgr.mount_nfs(
    server='192.168.1.100',
    remote_path='/exports/data',
    local_path='/mnt/nfs',
    options=['rw', 'hard', 'intr']
)

# List NFS mounts
nfs_mounts = nfs_mgr.list_nfs_mounts()
print(f"NFS mounts: {nfs_mounts['count']}")
```

### CIFS/SMB Operations

```python
from file_system_management import CIFSManager

cifs_mgr = CIFSManager()

# Mount CIFS share
result = cifs_mgr.mount_cifs(
    server='fileserver.local',
    share='public',
    local_path='/mnt/smb',
    username='user',
    password='password',
    domain='DOMAIN'
)

# List CIFS mounts
cifs_mounts = cifs_mgr.list_cifs_mounts()
print(f"CIFS mounts: {cifs_mounts['count']}")
```

## Data Models

### MountPoint
```python
@dataclass
class MountPoint:
    device: str           # Device path
    mountpoint: str       # Mount point path
    fstype: str          # Filesystem type
    options: str         # Mount options
    total_space: int     # Total bytes
    used_space: int      # Used bytes
    free_space: int      # Free bytes
    percent_used: float  # Usage percentage
```

### DiskInfo
```python
@dataclass
class DiskInfo:
    device: str                    # Device path
    size: int                      # Size in bytes
    model: str                     # Disk model
    serial: str                    # Serial number
    type: str                      # Device type
    partitions: List[Dict[str, Any]]  # Partition list
```

### QuotaInfo
```python
@dataclass
class QuotaInfo:
    user: str           # Username
    filesystem: str     # Filesystem path
    blocks_used: int    # Used blocks
    blocks_soft: int    # Soft limit (blocks)
    blocks_hard: int    # Hard limit (blocks)
    inodes_used: int    # Used inodes
    inodes_soft: int    # Soft limit (inodes)
    inodes_hard: int    # Hard limit (inodes)
```

## Configuration

### Environment Variables
```bash
# Optional logging configuration
export FS_LOG_LEVEL=INFO
export FS_LOG_FILE=/var/log/filesystem_mgmt.log
```

### Security Considerations

1. **Permissions**: Most operations require root privileges
2. **Data Safety**: Always backup before destructive operations
3. **Network Security**: Use secure authentication for NFS/CIFS
4. **Command Injection**: All inputs are sanitized
5. **Error Handling**: Comprehensive error handling and logging

## Command-Line Interface

Run the system directly:

```bash
# Run demonstration
python file_system_management.py

# With specific operations (can be extended)
python -c "
from file_system_management import FileSystemManagementOrchestrator
orch = FileSystemManagementOrchestrator()
print(orch.health_check())
"
```

## Health Checks

The system provides automated health checking:

- **Disk Space**: Warns at 80%, critical at 90%
- **RAID Health**: Monitors array status
- **Mount Status**: Verifies all mounts
- **I/O Performance**: Tracks read/write metrics

## Error Handling

All operations return standardized response dictionaries:

```python
{
    'success': bool,        # Operation success status
    'error': str,           # Error message (if failed)
    'timestamp': str,       # ISO 8601 timestamp
    # ... operation-specific fields
}
```

## Logging

Comprehensive logging to track all operations:

```python
2025-11-06 10:30:45 - FileSystemManagement - INFO - Mounting /dev/sdb1 to /mnt/data
2025-11-06 10:30:46 - FileSystemManagement - INFO - Creating RAID5 array /dev/md0
2025-11-06 10:30:47 - FileSystemManagement - WARNING - Disk space at 85% on /home
```

## Best Practices

1. **Always verify device paths** before operations
2. **Unmount before filesystem operations** (check/repair)
3. **Test RAID configurations** in non-production first
4. **Monitor quotas regularly** to prevent issues
5. **Keep LVM snapshots temporary** - they impact performance
6. **Use labels** for filesystems to avoid device name changes
7. **Regular health checks** to catch issues early
8. **Backup critical data** before structural changes

## Troubleshooting

### Common Issues

**Mount fails with "device busy"**
```python
# Use lazy unmount
fs_mgr.unmount_filesystem('/mnt/data', lazy=True)
```

**LVM commands not found**
```bash
sudo apt-get install lvm2
```

**RAID array degraded**
```python
# Add spare device
raid_mgr.add_spare_device('/dev/md0', '/dev/sdx')
```

**Quota not working**
```bash
# Enable quota in /etc/fstab
# Add: defaults,usrquota,grpquota
sudo quotacheck -cug /dev/sda1
sudo quotaon /dev/sda1
```

## Performance Considerations

- **LVM snapshots**: Use COW (copy-on-write) space efficiently
- **RAID levels**: RAID5/6 have write penalties, RAID10 better for performance
- **Filesystem choice**: XFS for large files, ext4 for general use
- **Mount options**: noatime reduces I/O overhead
- **Monitoring frequency**: Balance between visibility and overhead

## System Requirements

- Linux kernel 3.10+
- Python 3.8 or higher
- Root/sudo access for most operations
- 100MB disk space for installation
- System utilities: lsblk, mount, umount, mdadm, lvm tools

## Dependencies

- **psutil**: System and process utilities
- **subprocess**: Command execution
- **Standard library**: os, re, json, logging, pathlib, dataclasses, enum

## License

Proprietary - BrillConsulting

## Support

For issues, questions, or contributions:
- Internal Documentation: See company wiki
- Issue Tracking: Internal issue tracker
- Contact: BrillConsulting IT Operations Team

## Version History

### v2.0.0 (2025-11-06)
- Complete production implementation
- Added comprehensive filesystem operations
- Implemented LVM management (PV/VG/LV/snapshots)
- Added RAID configuration and monitoring
- Implemented quota management
- Added usage monitoring and inode tracking
- Integrated NFS and CIFS support
- Added health check system
- Comprehensive error handling and logging

### v1.0.0
- Initial skeleton implementation

## Roadmap

Future enhancements planned:
- Web-based dashboard
- Alert notification system (email/Slack)
- Automated backup integration
- ZFS support
- Filesystem deduplication support
- Performance profiling tools
- Multi-node cluster support
