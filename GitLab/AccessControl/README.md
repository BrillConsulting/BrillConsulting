# Access Control - Fine-Grained Permission Management

Comprehensive GitLab access control implementation for managing project and group permissions, protected branches and tags, deploy keys, access tokens, and LDAP synchronization with full auditing capabilities.

## Features

### Project Member Management
- **Add Members**: Add users to projects with specific access levels
- **Update Members**: Change access levels or expiration dates
- **Remove Members**: Remove users from projects
- **List Members**: Query members with optional access level filtering
- **Member Details**: Get detailed information about specific members
- **Expiration Control**: Set expiration dates for temporary access

### Group Member Management
- **Add Group Members**: Add users to groups with access levels
- **Group Sharing**: Share groups with other groups
- **Inherited Permissions**: Group members inherit project access
- **Cascade Updates**: Changes propagate to all group projects

### Protected Branch Management
- **Branch Protection**: Protect branches with push/merge access controls
- **Wildcard Support**: Protect patterns (e.g., 'release-*', 'hotfix-*')
- **Force Push Control**: Allow or deny force push operations
- **Code Owner Approval**: Require code owner approval for merges
- **Flexible Access Levels**: Configure different levels for push vs merge
- **Unprotect Branches**: Remove protection when needed

### Protected Tag Management
- **Tag Protection**: Protect tags with create access controls
- **Pattern Matching**: Support wildcards (e.g., 'v*', 'release-*')
- **Access Level Control**: Configure who can create protected tags
- **Unprotect Tags**: Remove tag protection

### Deploy Key Management
- **Create Deploy Keys**: Generate SSH keys for repository access
- **Read-Only Keys**: Keys with read-only repository access
- **Read-Write Keys**: Keys with write access for CI/CD
- **Enable Existing Keys**: Reuse keys across projects
- **Delete Keys**: Remove deploy keys when no longer needed
- **Key Metadata**: Store titles and metadata

### Access Token Management
- **Project Access Tokens**: Create tokens for API access
- **Token Scopes**: Configure permissions (api, read_api, read_repository, write_repository)
- **Access Levels**: Set token access level (GUEST to OWNER)
- **Expiration Control**: Set token expiration dates
- **Token Revocation**: Revoke tokens immediately
- **Token Rotation**: Automatic rotation with old token revocation
- **Audit Trail**: Track all token operations

### LDAP/SAML Group Synchronization
- **LDAP Integration**: Sync groups with LDAP providers
- **SAML Support**: SAML-based group synchronization
- **Automatic Sync**: Scheduled synchronization
- **Manual Trigger**: On-demand sync operations
- **Group Mapping**: Map LDAP groups to GitLab access levels
- **Sync Filtering**: Configure which groups to sync

### Access Control Auditing
- **Change Logging**: Log all access control changes
- **Audit Queries**: Query audit log with filters
- **Access Reports**: Generate comprehensive access reports
- **Event Types**: Track member changes, protection changes, token operations
- **Timestamp Tracking**: ISO format timestamps for all events
- **User Attribution**: Track who made each change

## Usage Example

```python
from access_control import AccessControlManager, AccessLevel

# Initialize manager
mgr = AccessControlManager(
    project_id='myorg/myproject',
    gitlab_url='https://gitlab.com'
)

# 1. Add project members with different access levels
developer = mgr.project_members.add_member({
    'user_id': 101,
    'access_level': AccessLevel.DEVELOPER
})

maintainer = mgr.project_members.add_member({
    'user_id': 102,
    'access_level': AccessLevel.MAINTAINER,
    'expires_at': '2025-12-31'  # Temporary access
})

# Update member access level
mgr.project_members.update_member(101, {
    'access_level': AccessLevel.MAINTAINER
})

# List members with specific access level
developers = mgr.project_members.list_members(
    access_level=AccessLevel.DEVELOPER
)

# 2. Protected branch configuration
# Protect main branch
main_protection = mgr.protected_branches.protect_branch({
    'name': 'main',
    'push_access_level': AccessLevel.MAINTAINER,
    'merge_access_level': AccessLevel.DEVELOPER,
    'allow_force_push': False,
    'code_owner_approval_required': True
})

# Protect all release branches with wildcard
release_protection = mgr.protected_branches.protect_branch({
    'name': 'release-*',
    'push_access_level': AccessLevel.MAINTAINER,
    'merge_access_level': AccessLevel.MAINTAINER
})

# Protect development branch
dev_protection = mgr.protected_branches.protect_branch({
    'name': 'develop',
    'push_access_level': AccessLevel.DEVELOPER,
    'merge_access_level': AccessLevel.DEVELOPER
})

# 3. Protected tag configuration
# Protect version tags
version_tags = mgr.protected_tags.protect_tag({
    'name': 'v*',
    'create_access_level': AccessLevel.MAINTAINER
})

# Protect release tags
release_tags = mgr.protected_tags.protect_tag({
    'name': 'release-*',
    'create_access_level': AccessLevel.MAINTAINER
})

# 4. Deploy key management
# Read-only deploy key for CI
readonly_key = mgr.deploy_keys.create_deploy_key({
    'title': 'CI Read-Only Key',
    'key': 'ssh-rsa AAAAB3NzaC1yc2E...',
    'can_push': False
})

# Read-write deploy key for deployment
readwrite_key = mgr.deploy_keys.create_deploy_key({
    'title': 'Deployment Key',
    'key': 'ssh-rsa AAAAB3NzaC1yc2E...',
    'can_push': True
})

# 5. Access token management
# API integration token
api_token = mgr.access_tokens.create_project_access_token({
    'name': 'api-integration',
    'scopes': ['api', 'read_repository'],
    'access_level': AccessLevel.REPORTER,
    'expires_at': '2025-12-31'
})

# CI/CD pipeline token
cicd_token = mgr.access_tokens.create_project_access_token({
    'name': 'cicd-pipeline',
    'scopes': ['api', 'read_repository', 'write_repository'],
    'access_level': AccessLevel.DEVELOPER,
    'expires_at': '2025-06-30'
})

# Rotate token (security best practice)
new_token = mgr.access_tokens.rotate_token(api_token['id'])

# Revoke token
mgr.access_tokens.revoke_token(cicd_token['id'])

# 6. Group member management
# Add user to group
group_member = mgr.group_members.add_member({
    'group_id': 'myorg/engineering',
    'user_id': 105,
    'access_level': AccessLevel.DEVELOPER
})

# Share group with another group
group_sharing = mgr.group_members.share_with_group({
    'source_group_id': 'myorg/engineering',
    'target_group_id': 'myorg/devops',
    'access_level': AccessLevel.DEVELOPER
})

# 7. LDAP group synchronization
# Configure LDAP sync
ldap_config = mgr.ldap_sync.configure_ldap_sync({
    'provider': 'ldap',
    'group_dn': 'cn=developers,ou=groups,dc=example,dc=com',
    'access_level': AccessLevel.DEVELOPER,
    'sync_schedule': '0 0 * * *'  # Daily at midnight
})

# Trigger manual sync
sync_result = mgr.ldap_sync.trigger_sync('ldap-sync-1')

# 8. Access control auditing
# Log access change
mgr.access_audit.log_access_change({
    'event_type': 'member_added',
    'user_id': 101,
    'access_level': AccessLevel.DEVELOPER,
    'actor_id': 1
})

# Query audit log
recent_changes = mgr.access_audit.query_audit_log({
    'event_type': 'member_added',
    'start_date': '2025-01-01',
    'end_date': '2025-12-31'
})

# Generate access report
report = mgr.access_audit.get_access_report()
# Returns: total_members, access_level_distribution, protected_branches_count, etc.
```

## Access Levels

### GitLab Access Level Hierarchy

| Level | Value | Permissions |
|-------|-------|-------------|
| **NO_ACCESS** | 0 | No access to the project |
| **MINIMAL_ACCESS** | 5 | View project, issues, and merge requests |
| **GUEST** | 10 | Create issues, leave comments |
| **REPORTER** | 20 | Pull code, download artifacts, view CI/CD |
| **DEVELOPER** | 30 | Push to branches, create merge requests, manage issues |
| **MAINTAINER** | 40 | Push to protected branches, manage team, edit project |
| **OWNER** | 50 | Full access, delete project, transfer project |

### Access Level Usage

#### GUEST (10)
- View project and issues
- Leave comments
- Cannot access code

**Use Cases**: External stakeholders, product managers, QA viewers

#### REPORTER (20)
- Pull code
- Download artifacts
- View CI/CD pipelines
- Cannot push

**Use Cases**: QA engineers, auditors, documentation writers

#### DEVELOPER (30)
- Push to unprotected branches
- Create merge requests
- Manage issues and labels
- Run CI/CD pipelines

**Use Cases**: Software developers, feature contributors

#### MAINTAINER (40)
- Push to protected branches
- Merge to main/master
- Manage team members
- Configure project settings

**Use Cases**: Team leads, senior developers, DevOps engineers

#### OWNER (50)
- Full project control
- Delete project
- Transfer ownership
- Manage billing

**Use Cases**: Project owners, administrators

## Protected Branch Patterns

### Exact Match
- `main` - Only main branch
- `master` - Only master branch
- `production` - Only production branch

### Wildcard Patterns
- `release-*` - All release branches (release-1.0, release-2.0)
- `hotfix-*` - All hotfix branches
- `feature/*` - All feature branches
- `*-stable` - All stable branches

### Multiple Branches
Configure protection for multiple branch patterns to enforce consistent policies across your repository.

## Protected Tag Patterns

### Version Tags
- `v*` - All version tags (v1.0.0, v2.1.3)
- `v[0-9]*` - Numeric version tags only

### Release Tags
- `release-*` - All release tags
- `stable-*` - All stable release tags

## Token Scopes

### Available Scopes
- **api**: Full API access (read and write)
- **read_api**: Read-only API access
- **read_repository**: Read repository code
- **write_repository**: Write to repository (push commits)
- **read_registry**: Read container registry
- **write_registry**: Write to container registry

### Scope Combinations

#### Read-Only Integration
```python
scopes: ['read_api', 'read_repository']
access_level: AccessLevel.REPORTER
```

#### CI/CD Pipeline
```python
scopes: ['api', 'read_repository', 'write_repository']
access_level: AccessLevel.DEVELOPER
```

#### Registry Management
```python
scopes: ['read_registry', 'write_registry']
access_level: AccessLevel.MAINTAINER
```

## Best Practices

### Member Management
1. **Use expiration dates** for temporary access (contractors, temporary team members)
2. **Apply least privilege** - Start with lower access levels, increase as needed
3. **Regular access reviews** - Quarterly review of all members and their access levels
4. **Remove inactive members** - Audit and remove members who no longer need access
5. **Group-based access** - Use groups for consistent permission management

### Branch Protection
1. **Always protect main/master** - Require MAINTAINER level for direct pushes
2. **Use wildcards** - Protect release branches with patterns (release-*)
3. **Require code reviews** - Set merge_access_level lower than push_access_level
4. **Enable code owner approval** - For critical branches
5. **Disable force push** - Prevent history rewriting on important branches

### Tag Protection
1. **Protect version tags** - Use v* pattern for semantic versions
2. **MAINTAINER only** - Restrict tag creation to maintainers
3. **Release automation** - Use CI/CD with appropriate tokens for automated releases

### Deploy Keys
1. **Use read-only keys** - Whenever possible, limit to read-only
2. **Rotate keys regularly** - Every 90 days minimum
3. **One key per purpose** - Separate keys for different CI/CD systems
4. **Document key usage** - Use descriptive titles and maintain documentation

### Access Tokens
1. **Set expiration dates** - Never create tokens without expiration
2. **Rotate tokens regularly** - Every 90 days for production systems
3. **Minimum scopes** - Only grant necessary permissions
4. **Audit token usage** - Monitor and log all token operations
5. **Revoke immediately** - When compromised or no longer needed
6. **Secure storage** - Store tokens in secret managers (HashiCorp Vault, AWS Secrets Manager)

### LDAP/SAML Sync
1. **Test sync configuration** - Verify group mappings before enabling
2. **Monitor sync errors** - Alert on failed synchronizations
3. **Document group mappings** - Maintain clear documentation of LDAP → GitLab mappings
4. **Scheduled sync** - Run during off-peak hours

### Auditing
1. **Enable comprehensive logging** - Log all access control changes
2. **Regular audit reviews** - Weekly review of access changes
3. **Automated alerts** - Alert on suspicious access changes (OWNER added, protection removed)
4. **Retention policy** - Retain audit logs for compliance requirements (90 days minimum)

## Common Use Cases

### Onboarding New Developer
```python
# Add as GUEST initially
mgr.project_members.add_member({
    'user_id': new_dev_id,
    'access_level': AccessLevel.GUEST
})

# After onboarding, promote to DEVELOPER
mgr.project_members.update_member(new_dev_id, {
    'access_level': AccessLevel.DEVELOPER
})
```

### Contractor with Temporary Access
```python
# 3-month contract
mgr.project_members.add_member({
    'user_id': contractor_id,
    'access_level': AccessLevel.DEVELOPER,
    'expires_at': (datetime.now() + timedelta(days=90)).isoformat()
})
```

### Setting Up New Repository
```python
# Protect main branch
mgr.protected_branches.protect_branch({
    'name': 'main',
    'push_access_level': AccessLevel.MAINTAINER,
    'merge_access_level': AccessLevel.DEVELOPER,
    'code_owner_approval_required': True
})

# Protect release branches
mgr.protected_branches.protect_branch({
    'name': 'release-*',
    'push_access_level': AccessLevel.MAINTAINER,
    'merge_access_level': AccessLevel.MAINTAINER
})

# Protect version tags
mgr.protected_tags.protect_tag({
    'name': 'v*',
    'create_access_level': AccessLevel.MAINTAINER
})
```

### CI/CD Pipeline Setup
```python
# Create pipeline token
pipeline_token = mgr.access_tokens.create_project_access_token({
    'name': 'github-actions-pipeline',
    'scopes': ['api', 'read_repository', 'write_repository'],
    'access_level': AccessLevel.DEVELOPER,
    'expires_at': (datetime.now() + timedelta(days=365)).isoformat()
})

# Create deploy key for deployment
deploy_key = mgr.deploy_keys.create_deploy_key({
    'title': 'Production Deployment',
    'key': production_ssh_key,
    'can_push': True
})
```

### Security Incident Response
```python
# Immediately revoke compromised token
mgr.access_tokens.revoke_token(compromised_token_id)

# Remove compromised user
mgr.project_members.remove_member(compromised_user_id)

# Audit recent access changes
recent_changes = mgr.access_audit.query_audit_log({
    'start_date': (datetime.now() - timedelta(days=7)).isoformat()
})

# Rotate all tokens
for token in mgr.access_tokens.tokens.values():
    if token['status'] == 'active':
        mgr.access_tokens.rotate_token(token['id'])
```

## Requirements

```
python-gitlab
requests
```

Install dependencies:
```bash
pip install python-gitlab requests
```

## Configuration

### Environment Variables
```bash
export GITLAB_URL="https://gitlab.com"
export GITLAB_PRIVATE_TOKEN="your-personal-access-token"
```

### Python Configuration
```python
from access_control import AccessControlManager

mgr = AccessControlManager(
    project_id='myorg/myproject',
    gitlab_url='https://gitlab.com',
    private_token='your-token'
)
```

### GitLab API Token Requirements

Your personal access token needs these scopes:
- **api**: Full API access
- **read_repository**: Read repository access
- **write_repository**: Write repository access (if managing deploy keys)

## Security Considerations

1. **Token Security**
   - Store tokens in environment variables or secret managers
   - Never commit tokens to repositories
   - Rotate tokens regularly (90-day maximum)

2. **Access Level Principles**
   - Start with minimum required access (GUEST or REPORTER)
   - Increase access levels based on proven need
   - Regular access reviews to ensure appropriate levels

3. **Branch Protection**
   - Always protect main/master branches
   - Require code reviews for protected branches
   - Enable code owner approval for critical paths

4. **Audit Compliance**
   - Enable comprehensive audit logging
   - Retain logs per compliance requirements
   - Regular review of access changes
   - Automated alerting for sensitive changes

5. **LDAP/SAML Security**
   - Secure LDAP connection (LDAPS)
   - Validate group mappings
   - Monitor sync errors
   - Test before production deployment

## Integration Examples

### With CI/CD
```python
# Create token for GitHub Actions
github_token = mgr.access_tokens.create_project_access_token({
    'name': 'github-actions',
    'scopes': ['read_repository', 'write_repository'],
    'access_level': AccessLevel.DEVELOPER,
    'expires_at': '2025-12-31'
})

# Use in GitHub Actions
# GITLAB_TOKEN: ${{ secrets.GITLAB_TOKEN }}
```

### With Terraform
```python
# Create service account token for Terraform
terraform_token = mgr.access_tokens.create_project_access_token({
    'name': 'terraform-automation',
    'scopes': ['api'],
    'access_level': AccessLevel.MAINTAINER,
    'expires_at': '2026-12-31'
})
```

### With Monitoring Systems
```python
# Configure audit log forwarding
def forward_audit_logs(webhook_url):
    recent_changes = mgr.access_audit.query_audit_log({
        'start_date': (datetime.now() - timedelta(hours=1)).isoformat()
    })

    for change in recent_changes:
        requests.post(webhook_url, json=change)
```

## Troubleshooting

### Common Issues

**Issue**: Member not showing in project
- **Solution**: Check if member was added to group instead of project
- **Solution**: Verify member has accepted invitation

**Issue**: Cannot push to protected branch
- **Solution**: Verify user has required access level (MAINTAINER for most protected branches)
- **Solution**: Check branch protection rules

**Issue**: Token authentication failing
- **Solution**: Verify token has not expired
- **Solution**: Check token has required scopes
- **Solution**: Rotate token if compromised

**Issue**: LDAP sync not working
- **Solution**: Verify LDAP connection and credentials
- **Solution**: Check group DN syntax
- **Solution**: Review sync logs for errors

## Author

BrillConsulting - Enterprise Cloud Solutions
