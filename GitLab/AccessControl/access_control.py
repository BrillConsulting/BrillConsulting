"""
GitLab Access Control - Advanced Permission Management
=======================================================

Comprehensive GitLab access control with:
- Role-based access control (RBAC)
- Project and group member management
- Protected branches and tags
- Deploy keys and access tokens
- LDAP/SAML group sync
- Access level inheritance
- Permission auditing

Author: Brill Consulting
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import IntEnum


class AccessLevel(IntEnum):
    """GitLab access levels."""
    NO_ACCESS = 0
    MINIMAL_ACCESS = 5
    GUEST = 10
    REPORTER = 20
    DEVELOPER = 30
    MAINTAINER = 40
    OWNER = 50


class ProjectMemberManager:
    """Manages project-level access control."""

    def __init__(self, project_id: int):
        self.project_id = project_id
        self.members = {}

    def add_member(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add member to project.

        Config:
        - user_id: User ID
        - access_level: AccessLevel (GUEST, REPORTER, DEVELOPER, MAINTAINER, OWNER)
        - expires_at: Optional expiration date (ISO format)
        """
        user_id = config.get('user_id')
        access_level = config.get('access_level', AccessLevel.DEVELOPER)
        expires_at = config.get('expires_at')

        print(f"\n👤 Adding member to project {self.project_id}")
        print(f"   User ID: {user_id}")
        print(f"   Access Level: {AccessLevel(access_level).name} ({access_level})")
        if expires_at:
            print(f"   Expires: {expires_at}")

        member = {
            "user_id": user_id,
            "project_id": self.project_id,
            "access_level": access_level,
            "access_level_name": AccessLevel(access_level).name,
            "expires_at": expires_at,
            "created_at": datetime.now().isoformat()
        }

        self.members[user_id] = member

        print(f"✓ Member added successfully")

        return member

    def update_member(self, user_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update member access level or expiration."""
        if user_id not in self.members:
            return {"error": f"Member {user_id} not found"}

        print(f"\n🔄 Updating member {user_id} in project {self.project_id}")

        member = self.members[user_id]

        if 'access_level' in config:
            old_level = member['access_level']
            new_level = config['access_level']
            member['access_level'] = new_level
            member['access_level_name'] = AccessLevel(new_level).name
            print(f"   Access Level: {AccessLevel(old_level).name} → {AccessLevel(new_level).name}")

        if 'expires_at' in config:
            member['expires_at'] = config['expires_at']
            print(f"   New Expiration: {config['expires_at']}")

        member['updated_at'] = datetime.now().isoformat()

        print(f"✓ Member updated successfully")

        return member

    def remove_member(self, user_id: int) -> Dict[str, str]:
        """Remove member from project."""
        if user_id not in self.members:
            return {"error": f"Member {user_id} not found"}

        print(f"\n🗑️  Removing member {user_id} from project {self.project_id}")

        del self.members[user_id]

        print(f"✓ Member removed successfully")

        return {"status": "removed", "user_id": str(user_id)}

    def list_members(self, min_access_level: Optional[int] = None) -> List[Dict[str, Any]]:
        """List project members with optional access level filter."""
        members = list(self.members.values())

        if min_access_level:
            members = [m for m in members if m['access_level'] >= min_access_level]

        return members

    def get_member(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get member details."""
        return self.members.get(user_id)


class GroupMemberManager:
    """Manages group-level access control."""

    def __init__(self, group_id: int):
        self.group_id = group_id
        self.members = {}

    def add_member(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add member to group.

        Config:
        - user_id: User ID
        - access_level: AccessLevel
        - expires_at: Optional expiration date
        """
        user_id = config.get('user_id')
        access_level = config.get('access_level', AccessLevel.DEVELOPER)
        expires_at = config.get('expires_at')

        print(f"\n👥 Adding member to group {self.group_id}")
        print(f"   User ID: {user_id}")
        print(f"   Access Level: {AccessLevel(access_level).name} ({access_level})")
        if expires_at:
            print(f"   Expires: {expires_at}")

        member = {
            "user_id": user_id,
            "group_id": self.group_id,
            "access_level": access_level,
            "access_level_name": AccessLevel(access_level).name,
            "expires_at": expires_at,
            "created_at": datetime.now().isoformat()
        }

        self.members[user_id] = member

        print(f"✓ Member added to group")

        return member

    def share_with_group(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Share group with another group.

        Config:
        - shared_with_group_id: Target group ID
        - group_access_level: Access level for shared group
        - expires_at: Optional expiration
        """
        shared_with_group_id = config.get('shared_with_group_id')
        group_access_level = config.get('group_access_level', AccessLevel.DEVELOPER)

        print(f"\n🔗 Sharing group {self.group_id} with group {shared_with_group_id}")
        print(f"   Access Level: {AccessLevel(group_access_level).name}")

        share = {
            "group_id": self.group_id,
            "shared_with_group_id": shared_with_group_id,
            "group_access_level": group_access_level,
            "expires_at": config.get('expires_at'),
            "created_at": datetime.now().isoformat()
        }

        print(f"✓ Group shared successfully")

        return share


class ProtectedBranchManager:
    """Manages protected branch access control."""

    def __init__(self, project_id: int):
        self.project_id = project_id
        self.protected_branches = {}

    def protect_branch(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Protect a branch.

        Config:
        - name: Branch name (supports wildcards like 'release-*')
        - push_access_level: AccessLevel for push (default: MAINTAINER)
        - merge_access_level: AccessLevel for merge (default: MAINTAINER)
        - unprotect_access_level: AccessLevel for unprotect (default: MAINTAINER)
        - allow_force_push: Allow force push (default: False)
        - code_owner_approval_required: Require code owner approval (default: False)
        """
        name = config.get('name')
        push_access = config.get('push_access_level', AccessLevel.MAINTAINER)
        merge_access = config.get('merge_access_level', AccessLevel.MAINTAINER)
        allow_force_push = config.get('allow_force_push', False)
        code_owner_approval = config.get('code_owner_approval_required', False)

        print(f"\n🔒 Protecting branch: {name}")
        print(f"   Project: {self.project_id}")
        print(f"   Push Access: {AccessLevel(push_access).name}")
        print(f"   Merge Access: {AccessLevel(merge_access).name}")
        print(f"   Force Push: {allow_force_push}")
        print(f"   Code Owner Approval: {code_owner_approval}")

        protection = {
            "name": name,
            "project_id": self.project_id,
            "push_access_level": push_access,
            "merge_access_level": merge_access,
            "unprotect_access_level": config.get('unprotect_access_level', AccessLevel.MAINTAINER),
            "allow_force_push": allow_force_push,
            "code_owner_approval_required": code_owner_approval,
            "created_at": datetime.now().isoformat()
        }

        self.protected_branches[name] = protection

        print(f"✓ Branch protected successfully")

        return protection

    def unprotect_branch(self, name: str) -> Dict[str, str]:
        """Unprotect a branch."""
        if name not in self.protected_branches:
            return {"error": f"Branch {name} is not protected"}

        print(f"\n🔓 Unprotecting branch: {name}")

        del self.protected_branches[name]

        print(f"✓ Branch unprotected successfully")

        return {"status": "unprotected", "branch": name}

    def list_protected_branches(self) -> List[Dict[str, Any]]:
        """List all protected branches."""
        return list(self.protected_branches.values())


class ProtectedTagManager:
    """Manages protected tag access control."""

    def __init__(self, project_id: int):
        self.project_id = project_id
        self.protected_tags = {}

    def protect_tag(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Protect tags.

        Config:
        - name: Tag pattern (e.g., 'v*', 'release-*')
        - create_access_level: AccessLevel for creating tags (default: MAINTAINER)
        """
        name = config.get('name')
        create_access = config.get('create_access_level', AccessLevel.MAINTAINER)

        print(f"\n🏷️  Protecting tags: {name}")
        print(f"   Project: {self.project_id}")
        print(f"   Create Access: {AccessLevel(create_access).name}")

        protection = {
            "name": name,
            "project_id": self.project_id,
            "create_access_level": create_access,
            "created_at": datetime.now().isoformat()
        }

        self.protected_tags[name] = protection

        print(f"✓ Tags protected successfully")

        return protection

    def unprotect_tag(self, name: str) -> Dict[str, str]:
        """Unprotect tags."""
        if name not in self.protected_tags:
            return {"error": f"Tag pattern {name} is not protected"}

        print(f"\n🔓 Unprotecting tags: {name}")

        del self.protected_tags[name]

        print(f"✓ Tags unprotected successfully")

        return {"status": "unprotected", "tag_pattern": name}


class DeployKeyManager:
    """Manages deploy keys for repository access."""

    def __init__(self, project_id: int):
        self.project_id = project_id
        self.deploy_keys = {}

    def create_deploy_key(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create deploy key.

        Config:
        - title: Key title
        - key: SSH public key
        - can_push: Allow push access (default: False)
        """
        title = config.get('title')
        key = config.get('key')
        can_push = config.get('can_push', False)

        print(f"\n🔑 Creating deploy key: {title}")
        print(f"   Project: {self.project_id}")
        print(f"   Can Push: {can_push}")
        print(f"   Key: {key[:50]}...")

        key_id = len(self.deploy_keys) + 1

        deploy_key = {
            "id": key_id,
            "title": title,
            "key": key,
            "can_push": can_push,
            "project_id": self.project_id,
            "created_at": datetime.now().isoformat()
        }

        self.deploy_keys[key_id] = deploy_key

        print(f"✓ Deploy key created (ID: {key_id})")

        return deploy_key

    def enable_deploy_key(self, key_id: int) -> Dict[str, Any]:
        """Enable existing deploy key."""
        print(f"\n✅ Enabling deploy key {key_id} for project {self.project_id}")

        enabled_key = {
            "key_id": key_id,
            "project_id": self.project_id,
            "enabled_at": datetime.now().isoformat()
        }

        print(f"✓ Deploy key enabled")

        return enabled_key

    def delete_deploy_key(self, key_id: int) -> Dict[str, str]:
        """Delete deploy key."""
        if key_id not in self.deploy_keys:
            return {"error": f"Deploy key {key_id} not found"}

        print(f"\n🗑️  Deleting deploy key {key_id}")

        del self.deploy_keys[key_id]

        print(f"✓ Deploy key deleted")

        return {"status": "deleted", "key_id": str(key_id)}


class AccessTokenManager:
    """Manages project and personal access tokens."""

    def __init__(self, project_id: Optional[int] = None):
        self.project_id = project_id
        self.tokens = {}

    def create_project_access_token(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create project access token.

        Config:
        - name: Token name
        - scopes: List of scopes (api, read_api, read_repository, write_repository)
        - access_level: AccessLevel (default: MAINTAINER)
        - expires_at: Expiration date
        """
        name = config.get('name')
        scopes = config.get('scopes', ['api'])
        access_level = config.get('access_level', AccessLevel.MAINTAINER)
        expires_at = config.get('expires_at')

        print(f"\n🎫 Creating project access token: {name}")
        print(f"   Project: {self.project_id}")
        print(f"   Scopes: {', '.join(scopes)}")
        print(f"   Access Level: {AccessLevel(access_level).name}")
        print(f"   Expires: {expires_at}")

        token_id = len(self.tokens) + 1
        token_value = f"glpat-{datetime.now().timestamp()}"

        token = {
            "id": token_id,
            "name": name,
            "token": token_value,
            "scopes": scopes,
            "access_level": access_level,
            "project_id": self.project_id,
            "expires_at": expires_at,
            "created_at": datetime.now().isoformat(),
            "active": True
        }

        self.tokens[token_id] = token

        print(f"✓ Token created (ID: {token_id})")
        print(f"   Token: {token_value}")

        return token

    def revoke_token(self, token_id: int) -> Dict[str, str]:
        """Revoke access token."""
        if token_id not in self.tokens:
            return {"error": f"Token {token_id} not found"}

        print(f"\n🚫 Revoking token {token_id}")

        self.tokens[token_id]['active'] = False
        self.tokens[token_id]['revoked_at'] = datetime.now().isoformat()

        print(f"✓ Token revoked")

        return {"status": "revoked", "token_id": str(token_id)}

    def rotate_token(self, token_id: int) -> Dict[str, Any]:
        """Rotate access token (revoke old, create new)."""
        if token_id not in self.tokens:
            return {"error": f"Token {token_id} not found"}

        old_token = self.tokens[token_id]

        print(f"\n🔄 Rotating token {token_id}: {old_token['name']}")

        # Revoke old token
        self.revoke_token(token_id)

        # Create new token with same settings
        new_token = self.create_project_access_token({
            "name": old_token['name'],
            "scopes": old_token['scopes'],
            "access_level": old_token['access_level'],
            "expires_at": old_token['expires_at']
        })

        print(f"✓ Token rotated successfully")

        return new_token


class LDAPGroupSyncManager:
    """Manages LDAP/SAML group synchronization."""

    def __init__(self, group_id: int):
        self.group_id = group_id
        self.syncs = {}

    def configure_ldap_sync(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure LDAP group sync.

        Config:
        - cn: LDAP Common Name
        - group_access: AccessLevel for synced users
        - provider: LDAP provider name
        """
        cn = config.get('cn')
        group_access = config.get('group_access', AccessLevel.DEVELOPER)
        provider = config.get('provider', 'ldapmain')

        print(f"\n🔗 Configuring LDAP sync for group {self.group_id}")
        print(f"   CN: {cn}")
        print(f"   Access Level: {AccessLevel(group_access).name}")
        print(f"   Provider: {provider}")

        sync_config = {
            "group_id": self.group_id,
            "cn": cn,
            "group_access": group_access,
            "provider": provider,
            "created_at": datetime.now().isoformat()
        }

        self.syncs[cn] = sync_config

        print(f"✓ LDAP sync configured")

        return sync_config

    def trigger_sync(self) -> Dict[str, Any]:
        """Trigger manual LDAP group sync."""
        print(f"\n🔄 Triggering LDAP sync for group {self.group_id}")

        sync_result = {
            "group_id": self.group_id,
            "syncs": len(self.syncs),
            "users_synced": len(self.syncs) * 5,  # Simulate
            "synced_at": datetime.now().isoformat()
        }

        print(f"✓ Sync completed")
        print(f"   Users synced: {sync_result['users_synced']}")

        return sync_result


class AccessAuditManager:
    """Manages access control auditing."""

    def __init__(self):
        self.audit_log = []

    def log_access_change(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Log access control change."""
        entry = {
            "event_type": event.get('event_type'),
            "resource_type": event.get('resource_type'),
            "resource_id": event.get('resource_id'),
            "user_id": event.get('user_id'),
            "details": event.get('details'),
            "timestamp": datetime.now().isoformat()
        }

        self.audit_log.append(entry)

        return entry

    def query_audit_log(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query audit log with filters."""
        results = self.audit_log

        if filters:
            if 'event_type' in filters:
                results = [e for e in results if e['event_type'] == filters['event_type']]
            if 'resource_type' in filters:
                results = [e for e in results if e['resource_type'] == filters['resource_type']]
            if 'user_id' in filters:
                results = [e for e in results if e['user_id'] == filters['user_id']]

        return results

    def get_access_report(self) -> Dict[str, Any]:
        """Generate access control report."""
        print(f"\n📊 Generating access control report")

        report = {
            "total_events": len(self.audit_log),
            "events_by_type": {},
            "generated_at": datetime.now().isoformat()
        }

        # Count events by type
        for entry in self.audit_log:
            event_type = entry['event_type']
            report['events_by_type'][event_type] = report['events_by_type'].get(event_type, 0) + 1

        print(f"✓ Report generated")
        print(f"   Total Events: {report['total_events']}")

        return report


class AccessControlManager:
    """Main access control manager integrating all components."""

    def __init__(self, project_id: Optional[int] = None, group_id: Optional[int] = None):
        self.project_id = project_id
        self.group_id = group_id

        if project_id:
            self.project_members = ProjectMemberManager(project_id)
            self.protected_branches = ProtectedBranchManager(project_id)
            self.protected_tags = ProtectedTagManager(project_id)
            self.deploy_keys = DeployKeyManager(project_id)
            self.access_tokens = AccessTokenManager(project_id)

        if group_id:
            self.group_members = GroupMemberManager(group_id)
            self.ldap_sync = LDAPGroupSyncManager(group_id)

        self.audit = AccessAuditManager()

    def info(self) -> Dict[str, Any]:
        """Get access control information."""
        return {
            "project_id": self.project_id,
            "group_id": self.group_id,
            "access_levels": {
                "NO_ACCESS": AccessLevel.NO_ACCESS,
                "MINIMAL_ACCESS": AccessLevel.MINIMAL_ACCESS,
                "GUEST": AccessLevel.GUEST,
                "REPORTER": AccessLevel.REPORTER,
                "DEVELOPER": AccessLevel.DEVELOPER,
                "MAINTAINER": AccessLevel.MAINTAINER,
                "OWNER": AccessLevel.OWNER
            }
        }


def demo():
    """Demo GitLab access control features."""
    print("=" * 70)
    print("GitLab Access Control - Advanced Demo")
    print("=" * 70)

    # 1. Project member management
    print("\n1. Project Member Management")
    print("-" * 70)

    mgr = AccessControlManager(project_id=123)

    # Add members with different access levels
    mgr.project_members.add_member({
        "user_id": 101,
        "access_level": AccessLevel.DEVELOPER
    })

    mgr.project_members.add_member({
        "user_id": 102,
        "access_level": AccessLevel.MAINTAINER,
        "expires_at": (datetime.now() + timedelta(days=90)).isoformat()
    })

    mgr.project_members.add_member({
        "user_id": 103,
        "access_level": AccessLevel.REPORTER
    })

    # Update member
    mgr.project_members.update_member(101, {
        "access_level": AccessLevel.MAINTAINER
    })

    # 2. Protected branches
    print("\n2. Protected Branch Configuration")
    print("-" * 70)

    # Protect main branch
    mgr.protected_branches.protect_branch({
        "name": "main",
        "push_access_level": AccessLevel.MAINTAINER,
        "merge_access_level": AccessLevel.DEVELOPER,
        "allow_force_push": False,
        "code_owner_approval_required": True
    })

    # Protect release branches with wildcard
    mgr.protected_branches.protect_branch({
        "name": "release-*",
        "push_access_level": AccessLevel.MAINTAINER,
        "merge_access_level": AccessLevel.MAINTAINER,
        "allow_force_push": False
    })

    # 3. Protected tags
    print("\n3. Protected Tag Configuration")
    print("-" * 70)

    mgr.protected_tags.protect_tag({
        "name": "v*",
        "create_access_level": AccessLevel.MAINTAINER
    })

    mgr.protected_tags.protect_tag({
        "name": "release-*",
        "create_access_level": AccessLevel.OWNER
    })

    # 4. Deploy keys
    print("\n4. Deploy Key Management")
    print("-" * 70)

    deploy_key = mgr.deploy_keys.create_deploy_key({
        "title": "Production Deploy Key",
        "key": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDZx...",
        "can_push": False
    })

    read_write_key = mgr.deploy_keys.create_deploy_key({
        "title": "CI/CD Pipeline Key",
        "key": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCxy...",
        "can_push": True
    })

    # 5. Access tokens
    print("\n5. Project Access Tokens")
    print("-" * 70)

    api_token = mgr.access_tokens.create_project_access_token({
        "name": "api-integration",
        "scopes": ["api", "read_repository"],
        "access_level": AccessLevel.DEVELOPER,
        "expires_at": (datetime.now() + timedelta(days=365)).isoformat()
    })

    ci_token = mgr.access_tokens.create_project_access_token({
        "name": "ci-cd-pipeline",
        "scopes": ["api", "write_repository"],
        "access_level": AccessLevel.MAINTAINER,
        "expires_at": (datetime.now() + timedelta(days=90)).isoformat()
    })

    # Rotate token
    mgr.access_tokens.rotate_token(api_token['id'])

    # 6. Group management
    print("\n6. Group Member Management")
    print("-" * 70)

    group_mgr = AccessControlManager(group_id=456)

    group_mgr.group_members.add_member({
        "user_id": 201,
        "access_level": AccessLevel.OWNER
    })

    group_mgr.group_members.add_member({
        "user_id": 202,
        "access_level": AccessLevel.MAINTAINER
    })

    # Share group with another group
    group_mgr.group_members.share_with_group({
        "shared_with_group_id": 789,
        "group_access_level": AccessLevel.DEVELOPER,
        "expires_at": (datetime.now() + timedelta(days=180)).isoformat()
    })

    # 7. LDAP group sync
    print("\n7. LDAP Group Synchronization")
    print("-" * 70)

    group_mgr.ldap_sync.configure_ldap_sync({
        "cn": "developers",
        "group_access": AccessLevel.DEVELOPER,
        "provider": "ldapmain"
    })

    group_mgr.ldap_sync.configure_ldap_sync({
        "cn": "maintainers",
        "group_access": AccessLevel.MAINTAINER,
        "provider": "ldapmain"
    })

    sync_result = group_mgr.ldap_sync.trigger_sync()

    # 8. Access audit
    print("\n8. Access Control Auditing")
    print("-" * 70)

    # Log some events
    mgr.audit.log_access_change({
        "event_type": "member_added",
        "resource_type": "project",
        "resource_id": 123,
        "user_id": 101,
        "details": {"access_level": "DEVELOPER"}
    })

    mgr.audit.log_access_change({
        "event_type": "branch_protected",
        "resource_type": "project",
        "resource_id": 123,
        "user_id": 102,
        "details": {"branch": "main"}
    })

    report = mgr.audit.get_access_report()

    # Summary
    print("\n9. Access Control Summary")
    print("-" * 70)

    info = mgr.info()
    print(f"\n  Project: {info['project_id']}")
    print(f"  Members: {len(mgr.project_members.members)}")
    print(f"  Protected Branches: {len(mgr.protected_branches.protected_branches)}")
    print(f"  Protected Tags: {len(mgr.protected_tags.protected_tags)}")
    print(f"  Deploy Keys: {len(mgr.deploy_keys.deploy_keys)}")
    print(f"  Access Tokens: {len(mgr.access_tokens.tokens)}")

    print("\n" + "=" * 70)
    print("✓ Access Control Advanced Demo Complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
