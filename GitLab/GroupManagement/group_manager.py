"""
GitLab Group Management
Author: BrillConsulting
Description: Groups, projects, and permissions management
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class GroupManager:
    """GitLab Group and Permissions management"""

    def __init__(self, gitlab_url: str, token: str):
        self.gitlab_url = gitlab_url
        self.token = token
        self.groups = []
        self.members = []

    def create_group(self, group_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create GitLab group"""
        group = {
            'id': len(self.groups) + 1,
            'name': group_config.get('name', 'mygroup'),
            'path': group_config.get('path', 'mygroup'),
            'visibility': group_config.get('visibility', 'private'),
            'description': group_config.get('description', ''),
            'parent_id': group_config.get('parent_id', None),
            'created_at': datetime.now().isoformat()
        }
        self.groups.append(group)
        print(f"✓ Group created: {group['name']}")
        print(f"  ID: {group['id']}, Visibility: {group['visibility']}")
        return group

    def add_member(self, member_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add member to group"""
        member = {
            'user_id': member_config.get('user_id', 1),
            'username': member_config.get('username', 'user1'),
            'group_id': member_config.get('group_id', 1),
            'access_level': member_config.get('access_level', 30),
            'expires_at': member_config.get('expires_at', None),
            'added_at': datetime.now().isoformat()
        }

        access_levels = {
            10: 'Guest',
            20: 'Reporter',
            30: 'Developer',
            40: 'Maintainer',
            50: 'Owner'
        }

        self.members.append(member)
        print(f"✓ Member added: {member['username']}")
        print(f"  Access: {access_levels.get(member['access_level'], 'Unknown')}")
        return member

    def manage_permissions(self, perm_config: Dict[str, Any]) -> Dict[str, Any]:
        """Manage group permissions"""
        permissions = {
            'group_id': perm_config.get('group_id', 1),
            'default_branch_protection': perm_config.get('default_branch_protection', 2),
            'share_with_group_lock': perm_config.get('share_lock', False),
            'require_two_factor_auth': perm_config.get('2fa', False),
            'project_creation_level': perm_config.get('project_creation', 'maintainer'),
            'configured_at': datetime.now().isoformat()
        }
        print(f"✓ Permissions configured for group {permissions['group_id']}")
        print(f"  2FA required: {permissions['require_two_factor_auth']}")
        return permissions


def demo():
    """Demonstrate group management"""
    print("=" * 60)
    print("GitLab Group Management Demo")
    print("=" * 60)

    mgr = GroupManager('https://gitlab.example.com', 'token')

    print("\n1. Creating group...")
    mgr.create_group({'name': 'engineering', 'path': 'engineering', 'visibility': 'private'})

    print("\n2. Adding member...")
    mgr.add_member({'username': 'developer1', 'group_id': 1, 'access_level': 30})

    print("\n3. Managing permissions...")
    mgr.manage_permissions({'group_id': 1, '2fa': True, 'project_creation': 'maintainer'})

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
