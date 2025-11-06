# Group Management - Organization & Hierarchy

Comprehensive GitLab group management system providing hierarchical organization, member management, permissions, LDAP sync, shared projects, and CI/CD variables.

## Features

### Group Hierarchy
- **Create Groups**: Top-level groups for departments/teams
- **Subgroups**: Unlimited nesting for complex structures
- **Full Paths**: Hierarchical paths (engineering/frontend/ui-team)
- **Visibility Levels**: PRIVATE, INTERNAL, PUBLIC
- **List & Filter**: Search by name, visibility, parent

### Member Management
- **Add Members**: User access with 7 levels (Guest → Owner)
- **Update Members**: Change access levels, set expiration
- **Remove Members**: Remove user access
- **Access Levels**: NO_ACCESS, MINIMAL, GUEST, REPORTER, DEVELOPER, MAINTAINER, OWNER
- **Member Expiration**: Temporary access with expiration dates
- **Member Statistics**: Active/inactive member tracking

### Permissions & Settings
- **Project Creation**: Control who can create projects
- **Subgroup Creation**: Control who can create subgroups
- **Two-Factor Auth**: Require 2FA for all members
- **2FA Grace Period**: Configurable grace period (hours)
- **Sharing Controls**: Lock group sharing
- **Mention Controls**: Disable @group mentions
- **Merge Settings**: Configure merge requirements

### Shared Projects
- **Share Projects**: Share projects with other groups
- **Access Control**: Set access level for shared projects
- **Expiration**: Temporary sharing with expiration
- **Unshare**: Remove project sharing
- **List Shared**: View all shared projects

### LDAP/SAML Integration
- **Link LDAP Groups**: Sync members from LDAP/AD
- **LDAP Providers**: Support multiple LDAP servers
- **LDAP Filters**: Custom LDAP queries
- **Auto Sync**: Automatic member synchronization
- **Sync History**: Track all sync operations
- **Default Access**: Set default level for synced users

### CI/CD Variables
- **Group Variables**: Variables inherited by all projects
- **Protected Variables**: Only available in protected branches
- **Masked Variables**: Hide values in logs
- **Environment Scopes**: Limit to specific environments
- **Variable Management**: Add, update, remove variables

### Statistics & Analytics
- **Member Counts**: Total and active members
- **Project Counts**: Projects per group
- **Subgroup Counts**: Nested group statistics
- **Storage Usage**: Repository, wiki, LFS, packages
- **Activity Tracking**: Last activity timestamps
- **Member Activity**: Commits, MRs, issues per member

## Usage Example

```python
from group_manager import GroupManagementManager, AccessLevel, GroupVisibility

# Initialize
mgr = GroupManagementManager()

# Create parent group
engineering = mgr.groups.create_group({
    'name': 'Engineering',
    'path': 'engineering',
    'description': 'Engineering department',
    'visibility': GroupVisibility.PRIVATE.value
})

# Create subgroups
frontend = mgr.groups.create_group({
    'name': 'Frontend Team',
    'path': 'frontend',
    'parent_id': engineering['group_id']
})

backend = mgr.groups.create_group({
    'name': 'Backend Team',
    'path': 'backend',
    'parent_id': engineering['group_id']
})

# Add members
mgr.members.add_member(engineering['group_id'], {
    'user_id': 'user-1',
    'username': 'alice',
    'access_level': AccessLevel.OWNER
})

mgr.members.add_member(frontend['group_id'], {
    'user_id': 'user-2',
    'username': 'bob',
    'access_level': AccessLevel.MAINTAINER,
    'expires_at': '2025-12-31'
})

# Set permissions
mgr.permissions.set_permissions(engineering['group_id'], {
    'project_creation_level': 'maintainer',
    'require_two_factor_auth': True,
    'two_factor_grace_period': 48
})

# Share project with group
mgr.shared_projects.share_project({
    'project_id': 'project-123',
    'group_id': engineering['group_id'],
    'group_access': AccessLevel.DEVELOPER
})

# Link LDAP group
mgr.ldap.link_ldap_group({
    'group_id': engineering['group_id'],
    'ldap_provider': 'main',
    'ldap_cn': 'cn=engineering,ou=groups,dc=company,dc=com',
    'group_access': AccessLevel.DEVELOPER
})

# Sync LDAP
sync_result = mgr.ldap.sync_ldap_group(engineering['group_id'])

# Add CI/CD variable
mgr.variables.add_variable(engineering['group_id'], {
    'key': 'DEPLOY_KEY',
    'value': 'secret-value',
    'protected': True,
    'masked': True,
    'environment_scope': 'production'
})

# Get statistics
stats = mgr.statistics.get_group_statistics(engineering['group_id'])
activity = mgr.statistics.get_member_activity(engineering['group_id'], days=30)
```

## Access Levels

| Level | Value | Permissions |
|-------|-------|-------------|
| NO_ACCESS | 0 | No access |
| MINIMAL_ACCESS | 5 | Minimal view access |
| GUEST | 10 | View, comment |
| REPORTER | 20 | Pull code, view CI/CD |
| DEVELOPER | 30 | Push code, manage issues |
| MAINTAINER | 40 | Manage settings, members |
| OWNER | 50 | Full control, delete group |

## Group Hierarchies

```python
# Create nested structure
company = mgr.groups.create_group({'name': 'MyCompany', 'path': 'mycompany'})

engineering = mgr.groups.create_group({
    'name': 'Engineering',
    'path': 'engineering',
    'parent_id': company['group_id']
})

frontend = mgr.groups.create_group({
    'name': 'Frontend',
    'path': 'frontend',
    'parent_id': engineering['group_id']
})
# Results in: mycompany/engineering/frontend

# List subgroups
subgroups = mgr.groups.get_subgroups(engineering['group_id'])

# List root groups only
root_groups = mgr.groups.list_groups({'parent_id': None})
```

## Visibility Levels

| Level | Access | Use Case |
|-------|--------|----------|
| **PRIVATE** | Group members only | Internal teams |
| **INTERNAL** | All authenticated users | Company-wide groups |
| **PUBLIC** | Everyone (including anonymous) | Open source communities |

## LDAP Synchronization

```python
# Link LDAP group
mgr.ldap.link_ldap_group({
    'group_id': 'group-1',
    'ldap_provider': 'main',
    'ldap_cn': 'cn=developers,ou=groups,dc=company,dc=com',
    'ldap_filter': '(objectClass=person)',
    'group_access': AccessLevel.DEVELOPER
})

# Manual sync
sync = mgr.ldap.sync_ldap_group('group-1')
print(f"Added: {sync['members_added']}, Removed: {sync['members_removed']}")

# View sync history
history = mgr.ldap.get_sync_history('group-1', limit=10)

# Unlink LDAP
mgr.ldap.unlink_ldap_group('group-1')
```

## CI/CD Variables

```python
# Add protected variable
mgr.variables.add_variable('group-1', {
    'key': 'DATABASE_URL',
    'value': 'postgresql://...',
    'protected': True,  # Only in protected branches
    'masked': True,  # Hidden in logs
    'environment_scope': 'production'
})

# Add environment-specific variable
mgr.variables.add_variable('group-1', {
    'key': 'API_KEY',
    'value': 'staging-key',
    'environment_scope': 'staging'
})

# List all variables
variables = mgr.variables.list_variables('group-1')

# Remove variable
mgr.variables.remove_variable('group-1', 'OLD_KEY')
```

## Shared Projects

```python
# Share project with group
mgr.shared_projects.share_project({
    'project_id': 'myorg/shared-lib',
    'group_id': 'group-1',
    'group_access': AccessLevel.REPORTER,
    'expires_at': '2025-12-31'
})

# List shared projects
shared = mgr.shared_projects.list_shared_projects('group-1')

# Unshare project
mgr.shared_projects.unshare_project('group-1', 'myorg/shared-lib')
```

## Best Practices

### Group Structure
1. **Logical Hierarchy**: Mirror organizational structure
2. **Consistent Naming**: Use clear, descriptive names
3. **Limited Depth**: Keep nesting to 3-4 levels max
4. **Group Purpose**: One clear purpose per group
5. **Documentation**: Describe group purpose in description

### Member Management
1. **Least Privilege**: Minimum required access level
2. **Regular Audits**: Review member access quarterly
3. **Expiration Dates**: Use for temporary access
4. **Group Owners**: Multiple owners for redundancy
5. **Remove Inactive**: Remove users who left organization

### Permissions
1. **Enable 2FA**: Require for all members
2. **Lock Sharing**: Prevent unauthorized sharing
3. **Control Creation**: Limit who can create projects/subgroups
4. **Review Settings**: Regular permission audits
5. **Document Policies**: Clear access policies

### LDAP Integration
1. **Test First**: Test sync with small group
2. **Monitor Syncs**: Check sync logs regularly
3. **Backup Strategy**: Manual backup before first sync
4. **Access Levels**: Appropriate defaults for synced users
5. **Sync Schedule**: Regular automated syncs

### CI/CD Variables
1. **Secret Management**: Use masked variables for secrets
2. **Protected Variables**: Limit to protected branches
3. **Environment Scopes**: Separate dev/staging/prod
4. **Naming Convention**: Consistent variable naming
5. **Documentation**: Document all variable usage

## Common Use Cases

### Department Structure
```python
# Create department hierarchy
company = mgr.groups.create_group({'name': 'MyCompany', 'path': 'mycompany'})
engineering = mgr.groups.create_group({'name': 'Engineering', 'path': 'engineering', 'parent_id': company['group_id']})
marketing = mgr.groups.create_group({'name': 'Marketing', 'path': 'marketing', 'parent_id': company['group_id']})
```

### Team Access Control
```python
# Add team members with different roles
mgr.members.add_member('group-1', {'username': 'tech-lead', 'access_level': AccessLevel.MAINTAINER})
mgr.members.add_member('group-1', {'username': 'developer', 'access_level': AccessLevel.DEVELOPER})
mgr.members.add_member('group-1', {'username': 'intern', 'access_level': AccessLevel.REPORTER, 'expires_at': '2025-09-01'})
```

### Cross-Team Collaboration
```python
# Share library project with multiple teams
for team in ['frontend-team', 'backend-team', 'mobile-team']:
    mgr.shared_projects.share_project({
        'project_id': 'shared-components',
        'group_id': team,
        'group_access': AccessLevel.DEVELOPER
    })
```

## Troubleshooting

**Issue**: Member can't access group projects
- Check member access level (must be Guest or higher)
- Verify group visibility settings
- Check if member's access has expired

**Issue**: LDAP sync not working
- Verify LDAP connection and credentials
- Check LDAP CN and filter syntax
- Review sync history for error messages

**Issue**: CI/CD variables not available
- Check environment scope matches
- Verify protected status matches branch
- Ensure variable key is correct

**Issue**: Can't create subgroup
- Check subgroup creation permissions
- Verify parent group access (need Maintainer or Owner)
- Check group nesting depth limits

## Requirements

```
datetime (standard library)
typing (standard library)
enum (standard library)
```

No external dependencies required.

## Configuration

```python
from group_manager import GroupManagementManager

mgr = GroupManagementManager(gitlab_url='https://gitlab.com')
```

## Author

BrillConsulting - Enterprise Cloud Solutions
