"""
GitLab Group Management - Comprehensive Group Organization System

This module provides complete group management including hierarchical groups,
member management, permissions, LDAP sync, shared projects, and group settings.
"""

from enum import Enum, IntEnum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class AccessLevel(IntEnum):
    """GitLab group access levels."""
    NO_ACCESS = 0
    MINIMAL_ACCESS = 5
    GUEST = 10
    REPORTER = 20
    DEVELOPER = 30
    MAINTAINER = 40
    OWNER = 50


class GroupVisibility(Enum):
    """Group visibility levels."""
    PRIVATE = "private"
    INTERNAL = "internal"
    PUBLIC = "public"


class ProjectCreationLevel(Enum):
    """Who can create projects in group."""
    NO_ONE = "noone"
    MAINTAINER = "maintainer"
    DEVELOPER = "developer"


class SubgroupCreationLevel(Enum):
    """Who can create subgroups."""
    OWNER = "owner"
    MAINTAINER = "maintainer"


class GroupManager:
    """
    Manages GitLab groups and hierarchies.

    Handles group creation, subgroups, visibility, and settings.
    """

    def __init__(self):
        self.groups: Dict[str, Dict[str, Any]] = {}

    def create_group(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create group or subgroup.

        Config:
            name: Group name
            path: URL path
            description: Group description
            visibility: PRIVATE, INTERNAL, or PUBLIC
            parent_id: Parent group ID for subgroups
        """
        group_id = f"group-{len(self.groups) + 1}"

        group = {
            'group_id': group_id,
            'name': config.get('name'),
            'path': config.get('path'),
            'description': config.get('description', ''),
            'visibility': config.get('visibility', GroupVisibility.PRIVATE.value),
            'parent_id': config.get('parent_id'),
            'full_path': self._build_full_path(config.get('parent_id'), config.get('path')),
            'created_at': datetime.now().isoformat(),
            'members_count': 0,
            'projects_count': 0,
            'subgroups_count': 0
        }

        self.groups[group_id] = group

        is_subgroup = " (subgroup)" if group['parent_id'] else ""
        print(f"✅ Created group{is_subgroup}: {group['name']}")
        print(f"   Path: {group['full_path']}")
        return group

    def _build_full_path(self, parent_id: Optional[str], path: str) -> str:
        """Build full hierarchical path."""
        if not parent_id:
            return path

        if parent_id in self.groups:
            parent_path = self.groups[parent_id]['full_path']
            return f"{parent_path}/{path}"

        return path

    def list_groups(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List groups with optional filtering."""
        groups = list(self.groups.values())

        if filters:
            if 'parent_id' in filters:
                groups = [g for g in groups if g['parent_id'] == filters['parent_id']]
            if 'visibility' in filters:
                groups = [g for g in groups if g['visibility'] == filters['visibility']]
            if 'search' in filters:
                search = filters['search'].lower()
                groups = [g for g in groups if search in g['name'].lower()]

        return groups

    def get_group(self, group_id: str) -> Optional[Dict[str, Any]]:
        """Get group details."""
        return self.groups.get(group_id)

    def update_group(self, group_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update group settings."""
        if group_id in self.groups:
            self.groups[group_id].update(updates)
            self.groups[group_id]['updated_at'] = datetime.now().isoformat()
            print(f"✅ Updated group: {self.groups[group_id]['name']}")
            return self.groups[group_id]
        return {'status': 'not_found'}

    def delete_group(self, group_id: str) -> Dict[str, str]:
        """Delete group and all subgroups."""
        if group_id in self.groups:
            group = self.groups[group_id]
            # In reality, would delete subgroups recursively
            group['status'] = 'deleted'
            group['deleted_at'] = datetime.now().isoformat()
            print(f"🗑️  Deleted group: {group['name']}")
            return {'status': 'deleted', 'group_id': group_id}
        return {'status': 'not_found'}

    def get_subgroups(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get all subgroups of a parent group."""
        return [g for g in self.groups.values() if g['parent_id'] == parent_id]


class GroupMemberManager:
    """
    Manages group members and their access levels.

    Handles adding, updating, and removing members with expiration support.
    """

    def __init__(self):
        self.members: Dict[str, List[Dict[str, Any]]] = {}

    def add_member(self, group_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add member to group.

        Config:
            user_id: User ID
            username: Username
            access_level: AccessLevel enum value
            expires_at: Optional expiration date
        """
        if group_id not in self.members:
            self.members[group_id] = []

        member = {
            'member_id': f"member-{len(self.members[group_id]) + 1}",
            'user_id': config.get('user_id'),
            'username': config.get('username'),
            'access_level': config.get('access_level', AccessLevel.DEVELOPER),
            'expires_at': config.get('expires_at'),
            'added_at': datetime.now().isoformat()
        }

        self.members[group_id].append(member)
        print(f"✅ Added member: {member['username']} as {AccessLevel(member['access_level']).name}")
        return member

    def update_member(self, group_id: str, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update member access level or expiration."""
        if group_id in self.members:
            for member in self.members[group_id]:
                if member['user_id'] == user_id:
                    member.update(updates)
                    member['updated_at'] = datetime.now().isoformat()
                    print(f"✅ Updated member: {member['username']}")
                    return member
        return {'status': 'not_found'}

    def remove_member(self, group_id: str, user_id: str) -> Dict[str, str]:
        """Remove member from group."""
        if group_id in self.members:
            self.members[group_id] = [m for m in self.members[group_id] if m['user_id'] != user_id]
            print(f"🗑️  Removed member from group")
            return {'status': 'removed'}
        return {'status': 'not_found'}

    def list_members(self, group_id: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List group members with optional filtering."""
        members = self.members.get(group_id, [])

        if filters:
            if 'access_level' in filters:
                members = [m for m in members if m['access_level'] == filters['access_level']]
            if 'include_inherited' in filters and filters['include_inherited']:
                # In reality, would include members from parent groups
                pass

        return members

    def get_member_count(self, group_id: str) -> int:
        """Get total member count."""
        return len(self.members.get(group_id, []))


class GroupPermissionsManager:
    """
    Manages group-level permissions and settings.

    Controls project creation, sharing, branch protection, and security.
    """

    def __init__(self):
        self.permissions: Dict[str, Dict[str, Any]] = {}

    def set_permissions(self, group_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set group permissions.

        Config:
            project_creation_level: Who can create projects
            subgroup_creation_level: Who can create subgroups
            require_two_factor_auth: Require 2FA for members
            share_with_group_lock: Prevent sharing with other groups
            mention_disabled: Disable @group mentions
            allow_merge_on_skipped_pipeline: Allow merges on skipped pipelines
        """
        permissions = {
            'group_id': group_id,
            'project_creation_level': config.get('project_creation_level', ProjectCreationLevel.MAINTAINER.value),
            'subgroup_creation_level': config.get('subgroup_creation_level', SubgroupCreationLevel.OWNER.value),
            'require_two_factor_auth': config.get('require_two_factor_auth', False),
            'two_factor_grace_period': config.get('two_factor_grace_period', 48),  # hours
            'share_with_group_lock': config.get('share_with_group_lock', False),
            'mention_disabled': config.get('mention_disabled', False),
            'allow_merge_on_skipped_pipeline': config.get('allow_merge_on_skipped_pipeline', False),
            'configured_at': datetime.now().isoformat()
        }

        self.permissions[group_id] = permissions
        print(f"⚙️  Set permissions for group {group_id}")
        print(f"   2FA required: {permissions['require_two_factor_auth']}")
        return permissions

    def get_permissions(self, group_id: str) -> Optional[Dict[str, Any]]:
        """Get group permissions."""
        return self.permissions.get(group_id)

    def require_two_factor_auth(self, group_id: str, enabled: bool, grace_period: int = 48) -> Dict[str, Any]:
        """Require two-factor authentication for all members."""
        if group_id not in self.permissions:
            self.permissions[group_id] = {}

        self.permissions[group_id]['require_two_factor_auth'] = enabled
        self.permissions[group_id]['two_factor_grace_period'] = grace_period
        print(f"🔐 2FA requirement: {enabled} (grace period: {grace_period}h)")
        return self.permissions[group_id]


class SharedProjectManager:
    """
    Manages projects shared with groups.

    Handles sharing projects between groups with access control.
    """

    def __init__(self):
        self.shared_projects: Dict[str, List[Dict[str, Any]]] = {}

    def share_project(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Share project with group.

        Config:
            project_id: Project to share
            group_id: Group to share with
            group_access: Access level for group members
            expires_at: Optional expiration date
        """
        group_id = config.get('group_id')

        if group_id not in self.shared_projects:
            self.shared_projects[group_id] = []

        share = {
            'share_id': f"share-{len(self.shared_projects[group_id]) + 1}",
            'project_id': config.get('project_id'),
            'group_id': group_id,
            'group_access': config.get('group_access', AccessLevel.DEVELOPER),
            'expires_at': config.get('expires_at'),
            'shared_at': datetime.now().isoformat()
        }

        self.shared_projects[group_id].append(share)
        print(f"🔗 Shared project {share['project_id']} with group {group_id}")
        print(f"   Access level: {AccessLevel(share['group_access']).name}")
        return share

    def unshare_project(self, group_id: str, project_id: str) -> Dict[str, str]:
        """Remove project sharing with group."""
        if group_id in self.shared_projects:
            self.shared_projects[group_id] = [
                s for s in self.shared_projects[group_id]
                if s['project_id'] != project_id
            ]
            print(f"🔓 Unshared project {project_id} from group {group_id}")
            return {'status': 'unshared'}
        return {'status': 'not_found'}

    def list_shared_projects(self, group_id: str) -> List[Dict[str, Any]]:
        """List all projects shared with group."""
        return self.shared_projects.get(group_id, [])


class GroupLDAPManager:
    """
    Manages LDAP/SAML group synchronization.

    Syncs group membership from LDAP/SAML providers.
    """

    def __init__(self):
        self.ldap_links: Dict[str, Dict[str, Any]] = {}
        self.sync_history: List[Dict[str, Any]] = []

    def link_ldap_group(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Link GitLab group to LDAP group.

        Config:
            group_id: GitLab group ID
            ldap_provider: LDAP provider name
            ldap_cn: LDAP common name
            ldap_filter: LDAP filter
            group_access: Default access level for synced members
        """
        group_id = config.get('group_id')

        link = {
            'link_id': f"link-{len(self.ldap_links) + 1}",
            'group_id': group_id,
            'ldap_provider': config.get('ldap_provider'),
            'ldap_cn': config.get('ldap_cn'),
            'ldap_filter': config.get('ldap_filter', ''),
            'group_access': config.get('group_access', AccessLevel.DEVELOPER),
            'linked_at': datetime.now().isoformat()
        }

        self.ldap_links[group_id] = link
        print(f"🔗 Linked group to LDAP: {link['ldap_cn']}")
        return link

    def unlink_ldap_group(self, group_id: str) -> Dict[str, str]:
        """Remove LDAP link from group."""
        if group_id in self.ldap_links:
            del self.ldap_links[group_id]
            print(f"🔓 Unlinked group from LDAP")
            return {'status': 'unlinked'}
        return {'status': 'not_found'}

    def sync_ldap_group(self, group_id: str) -> Dict[str, Any]:
        """Sync group members from LDAP."""
        sync = {
            'sync_id': f"sync-{len(self.sync_history) + 1}",
            'group_id': group_id,
            'started_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat(),
            'members_added': 0,
            'members_removed': 0,
            'members_updated': 0,
            'status': 'success'
        }

        self.sync_history.append(sync)
        print(f"🔄 LDAP sync completed:")
        print(f"   Added: {sync['members_added']}, Removed: {sync['members_removed']}")
        return sync

    def get_sync_history(self, group_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get LDAP sync history for group."""
        history = [s for s in self.sync_history if s['group_id'] == group_id]
        return history[-limit:]


class GroupVariablesManager:
    """
    Manages group-level CI/CD variables.

    Inheritable variables available to all projects in group.
    """

    def __init__(self):
        self.variables: Dict[str, List[Dict[str, Any]]] = {}

    def add_variable(self, group_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add group variable.

        Config:
            key: Variable name
            value: Variable value
            protected: Only available in protected branches
            masked: Hide value in job logs
            environment_scope: Environment pattern (* for all)
        """
        if group_id not in self.variables:
            self.variables[group_id] = []

        variable = {
            'variable_id': f"var-{len(self.variables[group_id]) + 1}",
            'key': config.get('key'),
            'value': config.get('value'),
            'protected': config.get('protected', False),
            'masked': config.get('masked', False),
            'environment_scope': config.get('environment_scope', '*'),
            'created_at': datetime.now().isoformat()
        }

        self.variables[group_id].append(variable)
        masked_value = "***" if variable['masked'] else variable['value']
        print(f"✅ Added variable: {variable['key']} = {masked_value}")
        return variable

    def remove_variable(self, group_id: str, key: str) -> Dict[str, str]:
        """Remove group variable."""
        if group_id in self.variables:
            self.variables[group_id] = [v for v in self.variables[group_id] if v['key'] != key]
            print(f"🗑️  Removed variable: {key}")
            return {'status': 'removed'}
        return {'status': 'not_found'}

    def list_variables(self, group_id: str) -> List[Dict[str, Any]]:
        """List all group variables."""
        return self.variables.get(group_id, [])


class GroupStatisticsManager:
    """
    Tracks group statistics and analytics.

    Members, projects, storage, and activity metrics.
    """

    def __init__(self):
        self.stats: Dict[str, Dict[str, Any]] = {}

    def get_group_statistics(self, group_id: str) -> Dict[str, Any]:
        """Get comprehensive group statistics."""
        return {
            'group_id': group_id,
            'members_count': 0,
            'projects_count': 0,
            'subgroups_count': 0,
            'storage_size_bytes': 0,
            'repository_size_bytes': 0,
            'wiki_size_bytes': 0,
            'lfs_size_bytes': 0,
            'packages_size_bytes': 0,
            'last_activity_at': datetime.now().isoformat()
        }

    def get_member_activity(self, group_id: str, days: int = 30) -> Dict[str, Any]:
        """Get member activity statistics."""
        return {
            'group_id': group_id,
            'period_days': days,
            'active_members': 0,
            'inactive_members': 0,
            'total_commits': 0,
            'total_merge_requests': 0,
            'total_issues': 0
        }


class GroupManagementManager:
    """
    Main group management orchestration class.

    Coordinates all group operations including creation, members,
    permissions, LDAP sync, and shared projects.
    """

    def __init__(self, gitlab_url: str = 'https://gitlab.com'):
        self.gitlab_url = gitlab_url

        # Initialize all managers
        self.groups = GroupManager()
        self.members = GroupMemberManager()
        self.permissions = GroupPermissionsManager()
        self.shared_projects = SharedProjectManager()
        self.ldap = GroupLDAPManager()
        self.variables = GroupVariablesManager()
        self.statistics = GroupStatisticsManager()


def demo_group_management():
    """Demonstrate comprehensive group management."""
    print("\n" + "="*60)
    print("👥 GitLab Group Management Demo")
    print("="*60)

    mgr = GroupManagementManager()

    # 1. Create parent group
    print("\n1️⃣  Creating Parent Group")
    engineering = mgr.groups.create_group({
        'name': 'Engineering',
        'path': 'engineering',
        'description': 'Engineering department',
        'visibility': GroupVisibility.PRIVATE.value
    })

    # 2. Create subgroups
    print("\n2️⃣  Creating Subgroups")
    frontend = mgr.groups.create_group({
        'name': 'Frontend Team',
        'path': 'frontend',
        'parent_id': engineering['group_id'],
        'visibility': GroupVisibility.PRIVATE.value
    })

    backend = mgr.groups.create_group({
        'name': 'Backend Team',
        'path': 'backend',
        'parent_id': engineering['group_id'],
        'visibility': GroupVisibility.PRIVATE.value
    })

    # 3. Add members
    print("\n3️⃣  Adding Members")
    mgr.members.add_member(engineering['group_id'], {
        'user_id': 'user-1',
        'username': 'alice',
        'access_level': AccessLevel.OWNER
    })

    mgr.members.add_member(frontend['group_id'], {
        'user_id': 'user-2',
        'username': 'bob',
        'access_level': AccessLevel.MAINTAINER
    })

    # 4. Set permissions
    print("\n4️⃣  Setting Permissions")
    mgr.permissions.set_permissions(engineering['group_id'], {
        'project_creation_level': ProjectCreationLevel.MAINTAINER.value,
        'require_two_factor_auth': True,
        'two_factor_grace_period': 48
    })

    # 5. Share project
    print("\n5️⃣  Sharing Project")
    mgr.shared_projects.share_project({
        'project_id': 'project-123',
        'group_id': engineering['group_id'],
        'group_access': AccessLevel.DEVELOPER
    })

    # 6. LDAP sync
    print("\n6️⃣  LDAP Integration")
    mgr.ldap.link_ldap_group({
        'group_id': engineering['group_id'],
        'ldap_provider': 'main',
        'ldap_cn': 'cn=engineering,ou=groups,dc=company,dc=com',
        'group_access': AccessLevel.DEVELOPER
    })

    # 7. Add CI/CD variables
    print("\n7️⃣  Adding CI/CD Variables")
    mgr.variables.add_variable(engineering['group_id'], {
        'key': 'DEPLOY_KEY',
        'value': 'secret-key-value',
        'protected': True,
        'masked': True
    })

    # 8. Get statistics
    print("\n8️⃣  Group Statistics")
    stats = mgr.statistics.get_group_statistics(engineering['group_id'])
    print(f"   Members: {stats['members_count']}")
    print(f"   Projects: {stats['projects_count']}")
    print(f"   Subgroups: {stats['subgroups_count']}")

    print("\n" + "="*60)
    print("✅ Group Management Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_group_management()
