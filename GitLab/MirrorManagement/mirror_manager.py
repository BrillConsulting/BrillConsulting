"""
GitLab Mirror Management - Repository Mirroring & Synchronization

This module provides comprehensive repository mirroring including pull/push mirrors,
authentication, scheduling, conflict resolution, and monitoring.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class MirrorDirection(Enum):
    """Mirror direction types."""
    PULL = "pull"  # Import from external repo
    PUSH = "push"  # Export to external repo


class MirrorStatus(Enum):
    """Mirror synchronization status."""
    IDLE = "idle"
    SCHEDULED = "scheduled"
    UPDATING = "updating"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


class AuthMethod(Enum):
    """Authentication methods for mirrors."""
    PASSWORD = "password"
    SSH_KEY = "ssh_key"
    TOKEN = "token"
    NONE = "none"


class PullMirrorManager:
    """
    Manages pull mirrors (importing from external repositories).

    Pull mirrors automatically sync changes from external sources.
    """

    def __init__(self):
        self.pull_mirrors: Dict[str, Dict[str, Any]] = {}

    def create_pull_mirror(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create pull mirror configuration.

        Config:
            project_id: Target project ID
            url: Source repository URL
            auth_method: Authentication method
            username: Optional username
            password: Optional password/token
            ssh_key: Optional SSH private key
            update_interval: Update frequency in seconds
            only_protected_branches: Mirror only protected branches
            mirror_branch_regex: Regex pattern for branches to mirror
        """
        mirror_id = f"pull-{len(self.pull_mirrors) + 1}"

        mirror = {
            'mirror_id': mirror_id,
            'project_id': config.get('project_id'),
            'direction': MirrorDirection.PULL.value,
            'url': config.get('url'),
            'auth_method': config.get('auth_method', AuthMethod.NONE.value),
            'username': config.get('username', ''),
            'enabled': config.get('enabled', True),
            'update_interval': config.get('update_interval', 300),  # 5 minutes
            'only_protected_branches': config.get('only_protected_branches', False),
            'mirror_branch_regex': config.get('mirror_branch_regex', '.*'),
            'last_update': None,
            'last_successful_update': None,
            'last_error': None,
            'status': MirrorStatus.IDLE.value,
            'created_at': datetime.now().isoformat()
        }

        self.pull_mirrors[mirror_id] = mirror
        print(f"✅ Created pull mirror: {mirror['url']}")
        print(f"   Update interval: {mirror['update_interval']}s")
        return mirror

    def update_pull_mirror(self, mirror_id: str) -> Dict[str, Any]:
        """Trigger manual update of pull mirror."""
        if mirror_id not in self.pull_mirrors:
            return {'status': 'not_found'}

        mirror = self.pull_mirrors[mirror_id]
        mirror['status'] = MirrorStatus.UPDATING.value
        mirror['last_update'] = datetime.now().isoformat()

        # Simulate successful update
        mirror['status'] = MirrorStatus.SUCCESS.value
        mirror['last_successful_update'] = datetime.now().isoformat()

        print(f"🔄 Pull mirror updated: {mirror_id}")
        return mirror

    def list_pull_mirrors(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List pull mirrors, optionally filtered by project."""
        mirrors = list(self.pull_mirrors.values())
        if project_id:
            mirrors = [m for m in mirrors if m['project_id'] == project_id]
        return mirrors


class PushMirrorManager:
    """
    Manages push mirrors (exporting to external repositories).

    Push mirrors automatically sync changes to external destinations.
    """

    def __init__(self):
        self.push_mirrors: Dict[str, Dict[str, Any]] = {}

    def create_push_mirror(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create push mirror configuration.

        Config:
            project_id: Source project ID
            url: Destination repository URL
            auth_method: Authentication method
            username: Optional username
            password: Optional password/token
            ssh_key: Optional SSH private key
            only_protected_branches: Push only protected branches
            keep_divergent_refs: Keep refs that diverge from source
            mirror_branch_regex: Regex pattern for branches to mirror
        """
        mirror_id = f"push-{len(self.push_mirrors) + 1}"

        mirror = {
            'mirror_id': mirror_id,
            'project_id': config.get('project_id'),
            'direction': MirrorDirection.PUSH.value,
            'url': config.get('url'),
            'auth_method': config.get('auth_method', AuthMethod.PASSWORD.value),
            'username': config.get('username', ''),
            'enabled': config.get('enabled', True),
            'only_protected_branches': config.get('only_protected_branches', True),
            'keep_divergent_refs': config.get('keep_divergent_refs', False),
            'mirror_branch_regex': config.get('mirror_branch_regex', '.*'),
            'last_update': None,
            'last_successful_update': None,
            'last_error': None,
            'status': MirrorStatus.IDLE.value,
            'created_at': datetime.now().isoformat()
        }

        self.push_mirrors[mirror_id] = mirror
        print(f"✅ Created push mirror: {mirror['url']}")
        print(f"   Only protected branches: {mirror['only_protected_branches']}")
        return mirror

    def update_push_mirror(self, mirror_id: str) -> Dict[str, Any]:
        """Trigger manual update of push mirror."""
        if mirror_id not in self.push_mirrors:
            return {'status': 'not_found'}

        mirror = self.push_mirrors[mirror_id]
        mirror['status'] = MirrorStatus.UPDATING.value
        mirror['last_update'] = datetime.now().isoformat()

        # Simulate successful push
        mirror['status'] = MirrorStatus.SUCCESS.value
        mirror['last_successful_update'] = datetime.now().isoformat()

        print(f"🔄 Push mirror updated: {mirror_id}")
        return mirror

    def list_push_mirrors(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List push mirrors, optionally filtered by project."""
        mirrors = list(self.push_mirrors.values())
        if project_id:
            mirrors = [m for m in mirrors if m['project_id'] == project_id]
        return mirrors


class MirrorAuthManager:
    """
    Manages authentication credentials for mirrors.

    Supports password, SSH keys, and tokens.
    """

    def __init__(self):
        self.credentials: Dict[str, Dict[str, Any]] = {}

    def store_password_auth(self, mirror_id: str, username: str, password: str) -> Dict[str, Any]:
        """Store username/password authentication."""
        cred = {
            'mirror_id': mirror_id,
            'auth_method': AuthMethod.PASSWORD.value,
            'username': username,
            'password': '***',  # Masked
            'created_at': datetime.now().isoformat()
        }
        self.credentials[mirror_id] = cred
        print(f"🔐 Stored password authentication for {mirror_id}")
        return cred

    def store_ssh_key(self, mirror_id: str, ssh_private_key: str, passphrase: Optional[str] = None) -> Dict[str, Any]:
        """Store SSH key authentication."""
        cred = {
            'mirror_id': mirror_id,
            'auth_method': AuthMethod.SSH_KEY.value,
            'ssh_key_fingerprint': 'SHA256:...',  # Would be actual fingerprint
            'has_passphrase': passphrase is not None,
            'created_at': datetime.now().isoformat()
        }
        self.credentials[mirror_id] = cred
        print(f"🔐 Stored SSH key for {mirror_id}")
        return cred

    def store_token_auth(self, mirror_id: str, token: str) -> Dict[str, Any]:
        """Store token authentication (PAT, deploy token, etc.)."""
        cred = {
            'mirror_id': mirror_id,
            'auth_method': AuthMethod.TOKEN.value,
            'token': '***',  # Masked
            'created_at': datetime.now().isoformat()
        }
        self.credentials[mirror_id] = cred
        print(f"🔐 Stored token authentication for {mirror_id}")
        return cred

    def get_credentials(self, mirror_id: str) -> Optional[Dict[str, Any]]:
        """Get credentials for mirror (password masked)."""
        return self.credentials.get(mirror_id)


class MirrorSchedulerManager:
    """
    Manages mirror update scheduling.

    Handles automatic updates based on intervals or cron schedules.
    """

    def __init__(self):
        self.schedules: Dict[str, Dict[str, Any]] = {}

    def set_update_schedule(self, mirror_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set update schedule for mirror.

        Config:
            interval_seconds: Update interval in seconds
            cron_expression: Optional cron expression
            enabled: Enable/disable scheduled updates
        """
        schedule = {
            'mirror_id': mirror_id,
            'interval_seconds': config.get('interval_seconds', 300),
            'cron_expression': config.get('cron_expression'),
            'enabled': config.get('enabled', True),
            'next_run': self._calculate_next_run(config.get('interval_seconds', 300)),
            'last_run': None,
            'updated_at': datetime.now().isoformat()
        }

        self.schedules[mirror_id] = schedule
        print(f"⏰ Set schedule for {mirror_id}: every {schedule['interval_seconds']}s")
        return schedule

    def _calculate_next_run(self, interval_seconds: int) -> str:
        """Calculate next scheduled run time."""
        next_run = datetime.now() + timedelta(seconds=interval_seconds)
        return next_run.isoformat()

    def get_schedule(self, mirror_id: str) -> Optional[Dict[str, Any]]:
        """Get schedule for mirror."""
        return self.schedules.get(mirror_id)

    def disable_schedule(self, mirror_id: str) -> Dict[str, str]:
        """Disable scheduled updates."""
        if mirror_id in self.schedules:
            self.schedules[mirror_id]['enabled'] = False
            print(f"⏸️  Disabled schedule for {mirror_id}")
            return {'status': 'disabled'}
        return {'status': 'not_found'}


class MirrorConflictManager:
    """
    Manages merge conflicts in mirroring.

    Handles conflict detection and resolution strategies.
    """

    def __init__(self):
        self.conflicts: Dict[str, List[Dict[str, Any]]] = {}

    def detect_conflicts(self, mirror_id: str) -> Dict[str, Any]:
        """Detect conflicts in mirror update."""
        conflict = {
            'conflict_id': f"conflict-{len(self.conflicts.get(mirror_id, [])) + 1}",
            'mirror_id': mirror_id,
            'detected_at': datetime.now().isoformat(),
            'conflicting_branches': [],
            'conflicting_files': [],
            'resolution_strategy': 'manual'
        }

        if mirror_id not in self.conflicts:
            self.conflicts[mirror_id] = []

        self.conflicts[mirror_id].append(conflict)
        print(f"⚠️  Conflict detected in {mirror_id}")
        return conflict

    def resolve_conflict(self, mirror_id: str, conflict_id: str, strategy: str) -> Dict[str, Any]:
        """
        Resolve conflict with specified strategy.

        Strategies: ours, theirs, manual, abort
        """
        resolution = {
            'conflict_id': conflict_id,
            'mirror_id': mirror_id,
            'strategy': strategy,
            'resolved_at': datetime.now().isoformat(),
            'status': 'resolved'
        }

        print(f"✅ Resolved conflict {conflict_id} using '{strategy}' strategy")
        return resolution

    def get_conflicts(self, mirror_id: str) -> List[Dict[str, Any]]:
        """Get all conflicts for mirror."""
        return self.conflicts.get(mirror_id, [])


class MirrorMonitoringManager:
    """
    Monitors mirror health and performance.

    Tracks success rates, update times, and errors.
    """

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def record_update(self, mirror_id: str, success: bool, duration_ms: int, error: Optional[str] = None) -> None:
        """Record mirror update metrics."""
        if mirror_id not in self.metrics:
            self.metrics[mirror_id] = {
                'total_updates': 0,
                'successful_updates': 0,
                'failed_updates': 0,
                'total_duration_ms': 0,
                'last_error': None
            }

        metrics = self.metrics[mirror_id]
        metrics['total_updates'] += 1
        metrics['total_duration_ms'] += duration_ms

        if success:
            metrics['successful_updates'] += 1
        else:
            metrics['failed_updates'] += 1
            metrics['last_error'] = error

    def get_mirror_health(self, mirror_id: str) -> Dict[str, Any]:
        """Get health metrics for mirror."""
        if mirror_id not in self.metrics:
            return {'status': 'no_data'}

        metrics = self.metrics[mirror_id]
        success_rate = (
            metrics['successful_updates'] / metrics['total_updates'] * 100
            if metrics['total_updates'] > 0 else 0
        )
        avg_duration = (
            metrics['total_duration_ms'] / metrics['total_updates']
            if metrics['total_updates'] > 0 else 0
        )

        return {
            'mirror_id': mirror_id,
            'success_rate': success_rate,
            'total_updates': metrics['total_updates'],
            'successful_updates': metrics['successful_updates'],
            'failed_updates': metrics['failed_updates'],
            'average_duration_ms': avg_duration,
            'last_error': metrics['last_error']
        }

    def get_unhealthy_mirrors(self, min_success_rate: float = 90.0) -> List[str]:
        """Get mirrors with low success rates."""
        unhealthy = []
        for mirror_id, metrics in self.metrics.items():
            if metrics['total_updates'] > 0:
                success_rate = metrics['successful_updates'] / metrics['total_updates'] * 100
                if success_rate < min_success_rate:
                    unhealthy.append(mirror_id)
        return unhealthy


class BandwidthControlManager:
    """
    Manages bandwidth and rate limiting for mirrors.

    Controls network usage and prevents overload.
    """

    def __init__(self):
        self.limits: Dict[str, Dict[str, Any]] = {}

    def set_bandwidth_limit(self, mirror_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set bandwidth limits for mirror.

        Config:
            max_bytes_per_second: Maximum transfer rate
            max_concurrent_updates: Maximum concurrent mirror updates
            throttle_on_failure: Slow down after failures
        """
        limit = {
            'mirror_id': mirror_id,
            'max_bytes_per_second': config.get('max_bytes_per_second', 10 * 1024 * 1024),  # 10 MB/s
            'max_concurrent_updates': config.get('max_concurrent_updates', 5),
            'throttle_on_failure': config.get('throttle_on_failure', True),
            'configured_at': datetime.now().isoformat()
        }

        self.limits[mirror_id] = limit
        print(f"⚡ Set bandwidth limit: {limit['max_bytes_per_second'] / 1024 / 1024:.1f} MB/s")
        return limit

    def get_bandwidth_limit(self, mirror_id: str) -> Optional[Dict[str, Any]]:
        """Get bandwidth limits for mirror."""
        return self.limits.get(mirror_id)


class MirrorManagementManager:
    """
    Main mirror management orchestration class.

    Coordinates all mirroring operations including pull/push mirrors,
    authentication, scheduling, and monitoring.
    """

    def __init__(self, gitlab_url: str = 'https://gitlab.com'):
        self.gitlab_url = gitlab_url

        # Initialize all managers
        self.pull_mirrors = PullMirrorManager()
        self.push_mirrors = PushMirrorManager()
        self.auth = MirrorAuthManager()
        self.scheduler = MirrorSchedulerManager()
        self.conflicts = MirrorConflictManager()
        self.monitoring = MirrorMonitoringManager()
        self.bandwidth = BandwidthControlManager()

    def get_all_mirrors(self, project_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get all mirrors (pull and push) for project."""
        return {
            'pull_mirrors': self.pull_mirrors.list_pull_mirrors(project_id),
            'push_mirrors': self.push_mirrors.list_push_mirrors(project_id)
        }


def demo_mirror_management():
    """Demonstrate comprehensive mirror management."""
    print("\n" + "="*60)
    print("🔄 GitLab Mirror Management Demo")
    print("="*60)

    mgr = MirrorManagementManager()

    # 1. Create pull mirror
    print("\n1️⃣  Creating Pull Mirror")
    pull_mirror = mgr.pull_mirrors.create_pull_mirror({
        'project_id': 'myorg/myproject',
        'url': 'https://github.com/upstream/repo.git',
        'auth_method': AuthMethod.TOKEN.value,
        'update_interval': 300,
        'only_protected_branches': False
    })

    # Store authentication
    mgr.auth.store_token_auth(pull_mirror['mirror_id'], 'github_token_here')

    # 2. Create push mirror
    print("\n2️⃣  Creating Push Mirror")
    push_mirror = mgr.push_mirrors.create_push_mirror({
        'project_id': 'myorg/myproject',
        'url': 'https://bitbucket.org/backup/repo.git',
        'auth_method': AuthMethod.PASSWORD.value,
        'only_protected_branches': True,
        'keep_divergent_refs': False
    })

    mgr.auth.store_password_auth(push_mirror['mirror_id'], 'username', 'password')

    # 3. Set schedules
    print("\n3️⃣  Setting Update Schedules")
    mgr.scheduler.set_update_schedule(pull_mirror['mirror_id'], {
        'interval_seconds': 300,  # 5 minutes
        'enabled': True
    })

    # 4. Update mirrors
    print("\n4️⃣  Triggering Updates")
    mgr.pull_mirrors.update_pull_mirror(pull_mirror['mirror_id'])
    mgr.push_mirrors.update_push_mirror(push_mirror['mirror_id'])

    # 5. Monitor health
    print("\n5️⃣  Monitoring Mirror Health")
    mgr.monitoring.record_update(pull_mirror['mirror_id'], True, 1200, None)
    health = mgr.monitoring.get_mirror_health(pull_mirror['mirror_id'])
    print(f"   Success rate: {health['success_rate']:.1f}%")
    print(f"   Avg duration: {health['average_duration_ms']}ms")

    # 6. Set bandwidth limits
    print("\n6️⃣  Setting Bandwidth Limits")
    mgr.bandwidth.set_bandwidth_limit(pull_mirror['mirror_id'], {
        'max_bytes_per_second': 10 * 1024 * 1024,  # 10 MB/s
        'max_concurrent_updates': 3
    })

    # 7. List all mirrors
    print("\n7️⃣  Listing All Mirrors")
    all_mirrors = mgr.get_all_mirrors('myorg/myproject')
    print(f"   Pull mirrors: {len(all_mirrors['pull_mirrors'])}")
    print(f"   Push mirrors: {len(all_mirrors['push_mirrors'])}")

    print("\n" + "="*60)
    print("✅ Mirror Management Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_mirror_management()
