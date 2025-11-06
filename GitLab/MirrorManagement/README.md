# Mirror Management - Repository Mirroring & Sync

Comprehensive GitLab repository mirroring system providing pull/push mirrors, authentication, scheduling, conflict resolution, bandwidth control, and monitoring.

## Features

### Pull Mirrors (Import)
- **Sync from External**: Import from GitHub, Bitbucket, etc.
- **Auto Updates**: Scheduled synchronization
- **Branch Filtering**: Mirror specific branches with regex
- **Protected Only**: Option to mirror only protected branches
- **Authentication**: Password, SSH key, token support

### Push Mirrors (Export)
- **Sync to External**: Export to external repositories
- **Backup**: Automatic repository backups
- **Multi-Destination**: Push to multiple remotes
- **Protected Only**: Export only protected branches
- **Divergent Refs**: Option to keep or force push

### Authentication
- **Password**: Username/password authentication
- **SSH Keys**: SSH private key with optional passphrase
- **Tokens**: Personal access tokens, deploy tokens
- **Credential Storage**: Secure credential management

### Scheduling
- **Interval-Based**: Update every N seconds
- **Cron Expressions**: Advanced scheduling
- **Manual Triggers**: On-demand updates
- **Enable/Disable**: Pause/resume schedules

### Conflict Resolution
- **Detection**: Automatic conflict detection
- **Strategies**: ours, theirs, manual, abort
- **History**: Track all conflicts and resolutions

### Monitoring
- **Health Metrics**: Success rates, update times
- **Error Tracking**: Last error, failed updates
- **Performance**: Average update duration
- **Alerts**: Identify unhealthy mirrors

### Bandwidth Control
- **Rate Limiting**: Max bytes per second
- **Concurrent Limits**: Max simultaneous updates
- **Throttling**: Slow down after failures

## Usage Example

```python
from mirror_manager import MirrorManagementManager, AuthMethod

# Initialize
mgr = MirrorManagementManager()

# 1. Create pull mirror (import from GitHub)
pull = mgr.pull_mirrors.create_pull_mirror({
    'project_id': 'myorg/myproject',
    'url': 'https://github.com/upstream/repo.git',
    'auth_method': AuthMethod.TOKEN.value,
    'update_interval': 300,  # 5 minutes
    'only_protected_branches': False
})

mgr.auth.store_token_auth(pull['mirror_id'], 'github_token')

# 2. Create push mirror (backup to Bitbucket)
push = mgr.push_mirrors.create_push_mirror({
    'project_id': 'myorg/myproject',
    'url': 'https://bitbucket.org/backup/repo.git',
    'auth_method': AuthMethod.PASSWORD.value,
    'only_protected_branches': True,
    'keep_divergent_refs': False
})

mgr.auth.store_password_auth(push['mirror_id'], 'username', 'password')

# 3. Set update schedule
mgr.scheduler.set_update_schedule(pull['mirror_id'], {
    'interval_seconds': 300,
    'enabled': True
})

# 4. Manual update
mgr.pull_mirrors.update_pull_mirror(pull['mirror_id'])
mgr.push_mirrors.update_push_mirror(push['mirror_id'])

# 5. Monitor health
mgr.monitoring.record_update(pull['mirror_id'], True, 1200)
health = mgr.monitoring.get_mirror_health(pull['mirror_id'])
print(f"Success rate: {health['success_rate']}%")

# 6. Set bandwidth limits
mgr.bandwidth.set_bandwidth_limit(pull['mirror_id'], {
    'max_bytes_per_second': 10 * 1024 * 1024,  # 10 MB/s
    'max_concurrent_updates': 3
})

# 7. Handle conflicts
conflict = mgr.conflicts.detect_conflicts(pull['mirror_id'])
mgr.conflicts.resolve_conflict(pull['mirror_id'], conflict['conflict_id'], 'theirs')

# 8. List all mirrors
mirrors = mgr.get_all_mirrors('myorg/myproject')
```

## Authentication Methods

| Method | Use Case | Configuration |
|--------|----------|---------------|
| **Password** | HTTPS with username/password | username, password |
| **SSH Key** | SSH authentication | private_key, passphrase (optional) |
| **Token** | Personal access token, deploy token | token |
| **None** | Public repositories | - |

### Password Authentication
```python
mgr.auth.store_password_auth('mirror-1', 'username', 'password')
```

### SSH Key Authentication
```python
mgr.auth.store_ssh_key('mirror-1', ssh_private_key, passphrase='optional')
```

### Token Authentication
```python
mgr.auth.store_token_auth('mirror-1', 'ghp_xxxxxxxxxxxx')
```

## Mirror Directions

### Pull Mirror (Import)
- Brings changes from external repo into GitLab
- Runs on schedule or manual trigger
- Updates GitLab repository
- Use for: upstream synchronization, importing from GitHub

### Push Mirror (Export)
- Sends changes from GitLab to external repo
- Triggered on push to GitLab
- Updates external repository
- Use for: backups, multi-platform hosting, distribution

## Scheduling

```python
# Every 5 minutes
mgr.scheduler.set_update_schedule('mirror-1', {
    'interval_seconds': 300,
    'enabled': True
})

# Hourly
mgr.scheduler.set_update_schedule('mirror-2', {
    'interval_seconds': 3600
})

# Disable schedule
mgr.scheduler.disable_schedule('mirror-1')

# Get schedule
schedule = mgr.scheduler.get_schedule('mirror-1')
```

## Conflict Resolution

### Strategies

| Strategy | Behavior | Use When |
|----------|----------|----------|
| **ours** | Keep local changes | Local changes are correct |
| **theirs** | Accept remote changes | Remote changes are correct |
| **manual** | Manual resolution required | Need to review conflicts |
| **abort** | Cancel update | Cannot resolve automatically |

```python
# Detect conflicts
conflict = mgr.conflicts.detect_conflicts('mirror-1')

# Resolve with strategy
mgr.conflicts.resolve_conflict('mirror-1', conflict['conflict_id'], 'theirs')

# View conflict history
conflicts = mgr.conflicts.get_conflicts('mirror-1')
```

## Monitoring & Health

```python
# Record update result
mgr.monitoring.record_update('mirror-1', success=True, duration_ms=1500)

# Get health metrics
health = mgr.monitoring.get_mirror_health('mirror-1')
# Returns: success_rate, total_updates, average_duration_ms, last_error

# Find unhealthy mirrors (< 90% success rate)
unhealthy = mgr.monitoring.get_unhealthy_mirrors(min_success_rate=90.0)
```

## Bandwidth Control

```python
# Set bandwidth limit
mgr.bandwidth.set_bandwidth_limit('mirror-1', {
    'max_bytes_per_second': 10 * 1024 * 1024,  # 10 MB/s
    'max_concurrent_updates': 5,
    'throttle_on_failure': True
})

# Get current limits
limits = mgr.bandwidth.get_bandwidth_limit('mirror-1')
```

## Best Practices

### Pull Mirrors
1. **Update Frequency**: Balance freshness with load (5-15 minutes)
2. **Protected Branches**: Mirror only protected branches for stability
3. **Authentication**: Use tokens over passwords
4. **Monitoring**: Track success rates, alert on failures
5. **Bandwidth**: Limit to prevent network saturation

### Push Mirrors
1. **Protected Only**: Push only protected branches to prevent clutter
2. **Force Push**: Use `keep_divergent_refs: false` for clean mirrors
3. **Multiple Destinations**: Push to multiple backup locations
4. **Authentication**: Use deploy tokens or SSH keys
5. **Monitoring**: Ensure push mirrors stay in sync

### Authentication
1. **Tokens**: Prefer tokens over passwords
2. **SSH Keys**: Use for high-security environments
3. **Rotation**: Rotate credentials regularly
4. **Permissions**: Minimum required permissions
5. **Storage**: Store credentials securely

### Scheduling
1. **Reasonable Intervals**: Don't over-sync (5-15 minutes minimum)
2. **Off-Peak Hours**: Schedule heavy syncs during low traffic
3. **Enable/Disable**: Disable during maintenance
4. **Manual Triggers**: Use for immediate syncs
5. **Cron Expressions**: Use for precise scheduling

## Common Use Cases

### GitHub to GitLab Sync
```python
# Mirror from GitHub (pull)
mgr.pull_mirrors.create_pull_mirror({
    'project_id': 'myorg/project',
    'url': 'https://github.com/upstream/repo.git',
    'auth_method': AuthMethod.TOKEN.value,
    'update_interval': 600
})
```

### GitLab to Bitbucket Backup
```python
# Backup to Bitbucket (push)
mgr.push_mirrors.create_push_mirror({
    'project_id': 'myorg/project',
    'url': 'https://bitbucket.org/myorg/backup.git',
    'auth_method': AuthMethod.PASSWORD.value,
    'only_protected_branches': True
})
```

### Multi-Platform Distribution
```python
# Push to multiple platforms
for platform in [
    {'name': 'github', 'url': 'https://github.com/myorg/repo.git'},
    {'name': 'bitbucket', 'url': 'https://bitbucket.org/myorg/repo.git'},
    {'name': 'gitlab-backup', 'url': 'https://gitlab-backup.com/myorg/repo.git'}
]:
    mgr.push_mirrors.create_push_mirror({
        'project_id': 'myorg/project',
        'url': platform['url'],
        'only_protected_branches': True
    })
```

## Troubleshooting

**Issue**: Mirror update fails with authentication error
- Verify credentials are correct
- Check token hasn't expired
- Ensure SSH key has proper permissions
- Test authentication manually with git

**Issue**: Pull mirror not updating
- Check schedule is enabled
- Verify update interval configuration
- Check for conflict resolution blocks
- Review error logs in monitoring

**Issue**: Push mirror falling behind
- Check bandwidth limits
- Verify push mirror is enabled
- Review conflict resolution strategy
- Check remote repository permissions

**Issue**: High bandwidth usage
- Reduce update frequency
- Enable bandwidth limits
- Mirror only protected branches
- Use delta synchronization

**Issue**: Conflicts blocking updates
- Review conflict resolution strategy
- Use appropriate resolution (ours/theirs)
- Consider manual resolution for complex conflicts
- Check branch divergence

## Requirements

```
datetime (standard library)
typing (standard library)
enum (standard library)
```

No external dependencies required.

## Configuration

```python
from mirror_manager import MirrorManagementManager

mgr = MirrorManagementManager(gitlab_url='https://gitlab.com')
```

## Author

BrillConsulting - Enterprise Cloud Solutions
