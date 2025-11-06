# API Management - GitLab REST API Integration

Comprehensive GitLab REST API integration providing programmatic access to projects, issues, merge requests, pipelines, users, and webhooks with full CRUD operations and advanced filtering.

## Features

### Project Management
- **Create Projects**: New repositories with visibility, README, default branch
- **List Projects**: Filter by visibility, archived status, search
- **Update Projects**: Modify settings, description, visibility
- **Archive Projects**: Archive/unarchive projects
- **Project Details**: Full metadata including URLs, statistics

### Issue Tracking
- **Create Issues**: Tickets with labels, assignees, milestones, weights
- **Update Issues**: Change state, labels, assignees
- **List Issues**: Filter by state, labels, assignee, author
- **Close Issues**: Mark issues as resolved
- **Confidential Issues**: Private issue tracking
- **Issue IDs**: Per-project IID tracking (Issue #1, #2, etc.)

### Merge Requests
- **Create MRs**: Source→target branch with reviewers, labels
- **Approve MRs**: Multi-approver support
- **Merge MRs**: Squash, remove source branch options
- **List MRs**: Filter by state, labels, author
- **MR Validation**: Conflict detection, approval requirements
- **Draft MRs**: Work-in-progress merge requests

### CI/CD Pipelines
- **Create Pipelines**: Trigger pipelines on branches/tags
- **Run Pipelines**: Execute pipeline jobs
- **Monitor Pipelines**: Track status, duration
- **Pipeline Variables**: Custom variables per pipeline
- **List Pipelines**: Filter by status, ref

### Webhooks
- **Create Webhooks**: Project event notifications
- **Event Types**: Push, MR, issues, pipelines, jobs, deployments
- **SSL Verification**: Optional SSL verification
- **Trigger Webhooks**: Simulate webhook events
- **Token Auth**: Secret token for webhook security

### User Management
- **Create Users**: Username, email, permissions
- **List Users**: Search by name/username
- **User Profiles**: Bio, location, social links
- **Admin Users**: Administrative permissions
- **2FA Support**: Two-factor authentication tracking

## Usage Example

```python
from gitlab_api import GitLabAPIManager, ProjectVisibility, IssueState, MergeRequestState

# Initialize
mgr = GitLabAPIManager(
    gitlab_url='https://gitlab.com',
    token='glpat-xxxxxxxxxxxx'
)

# 1. Create users
dev = mgr.users.create_user({
    'username': 'alice',
    'email': 'alice@example.com',
    'name': 'Alice Developer',
    'can_create_group': True
})

reviewer = mgr.users.create_user({
    'username': 'bob',
    'email': 'bob@example.com',
    'name': 'Bob Reviewer'
})

# 2. Create project
project = mgr.projects.create_project({
    'name': 'My Web App',
    'namespace_id': 'mycompany',
    'description': 'Production web application',
    'visibility': ProjectVisibility.PRIVATE.value,
    'initialize_with_readme': True,
    'default_branch': 'main'
})

# 3. Create issue
issue = mgr.issues.create_issue({
    'project_id': project['id'],
    'title': 'Add user authentication',
    'description': 'Implement JWT-based auth',
    'labels': ['enhancement', 'security'],
    'assignee_ids': [dev['id']],
    'weight': 5,
    'due_date': '2025-12-31'
})

# 4. Create merge request
mr = mgr.merge_requests.create_merge_request({
    'project_id': project['id'],
    'title': 'Implement user authentication',
    'description': f"Adds JWT auth.\n\nCloses #{issue['iid']}",
    'source_branch': 'feature/auth',
    'target_branch': 'main',
    'author_id': dev['id'],
    'reviewer_ids': [reviewer['id']],
    'labels': ['enhancement'],
    'squash': True,
    'remove_source_branch': True,
    'approvals_required': 1
})

# 5. Create and run pipeline
pipeline = mgr.pipelines.create_pipeline({
    'project_id': project['id'],
    'ref': mr['source_branch'],
    'variables': {
        'ENVIRONMENT': 'staging',
        'RUN_TESTS': 'true'
    }
})

# Run pipeline
mgr.pipelines.run_pipeline(pipeline['id'])

# Complete pipeline
mgr.pipelines.complete_pipeline(pipeline['id'], 'success')

# 6. Approve merge request
mgr.merge_requests.approve_merge_request(mr['id'], reviewer['id'])

# 7. Merge merge request
merged_mr = mgr.merge_requests.merge_merge_request(mr['id'], {
    'merged_by': reviewer['id']
})

# 8. Close issue
mgr.issues.update_issue(issue['id'], {
    'state': IssueState.CLOSED.value
})

# 9. Create webhook
webhook = mgr.webhooks.create_webhook({
    'project_id': project['id'],
    'url': 'https://ci.example.com/webhook',
    'token': 'secret-token',
    'push_events': True,
    'merge_requests_events': True,
    'pipeline_events': True,
    'enable_ssl_verification': True
})

# 10. List resources with filters
# List open issues
open_issues = mgr.issues.list_issues({
    'project_id': project['id'],
    'state': IssueState.OPENED.value
})

# List issues by label
bug_issues = mgr.issues.list_issues({
    'project_id': project['id'],
    'labels': ['bug']
})

# List merged MRs
merged_mrs = mgr.merge_requests.list_merge_requests({
    'project_id': project['id'],
    'state': MergeRequestState.MERGED.value
})

# List successful pipelines
success_pipelines = mgr.pipelines.list_pipelines(project['id'], {
    'status': 'success'
})

# 11. Search projects
private_projects = mgr.projects.list_projects({
    'visibility': ProjectVisibility.PRIVATE.value
})

search_results = mgr.projects.list_projects({
    'search': 'web app'
})

# 12. Get statistics
stats = mgr.get_statistics()
print(f"Projects: {stats['projects_count']}")
print(f"Issues: {stats['issues_count']}")
print(f"Merge Requests: {stats['merge_requests_count']}")
```

## API Entities

### Project
| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique project ID |
| name | str | Project name |
| path | str | URL path |
| visibility | str | private, internal, public |
| default_branch | str | Default branch name |
| ssh_url_to_repo | str | SSH clone URL |
| http_url_to_repo | str | HTTPS clone URL |
| web_url | str | Web interface URL |
| star_count | int | Number of stars |
| forks_count | int | Number of forks |
| archived | bool | Archive status |

### Issue
| Field | Type | Description |
|-------|------|-------------|
| id | str | Global issue ID |
| iid | int | Project-specific issue number |
| project_id | str | Parent project |
| title | str | Issue title |
| description | str | Issue description |
| state | str | opened, closed |
| labels | List[str] | Issue labels |
| assignees | List[str] | Assigned user IDs |
| weight | int | Issue weight (complexity) |
| due_date | str | Due date (YYYY-MM-DD) |
| confidential | bool | Confidential flag |

### Merge Request
| Field | Type | Description |
|-------|------|-------------|
| id | str | Global MR ID |
| iid | int | Project-specific MR number |
| project_id | str | Parent project |
| title | str | MR title |
| source_branch | str | Source branch |
| target_branch | str | Target branch |
| state | str | opened, merged, closed |
| author | str | Author user ID |
| reviewers | List[str] | Reviewer user IDs |
| approvals | List[Dict] | Approval records |
| has_conflicts | bool | Merge conflict status |
| squash | bool | Squash commits on merge |
| remove_source_branch | bool | Delete source after merge |

### Pipeline
| Field | Type | Description |
|-------|------|-------------|
| id | str | Pipeline ID |
| project_id | str | Parent project |
| ref | str | Branch/tag name |
| status | str | Pipeline status |
| duration | float | Duration in seconds |
| variables | Dict | Pipeline variables |

## Visibility Levels

| Level | Access | Use Case |
|-------|--------|----------|
| **PRIVATE** | Project members only | Internal projects |
| **INTERNAL** | All authenticated users | Company-wide projects |
| **PUBLIC** | Everyone (including anonymous) | Open source projects |

## Pipeline Statuses

| Status | Description |
|--------|-------------|
| **created** | Pipeline created, not yet scheduled |
| **pending** | Waiting to run |
| **running** | Pipeline executing |
| **success** | All jobs passed |
| **failed** | One or more jobs failed |
| **canceled** | Manually canceled |
| **skipped** | Pipeline skipped |

## Filtering Examples

### Filter Issues
```python
# Open issues assigned to user
issues = mgr.issues.list_issues({
    'project_id': 'project-1',
    'state': IssueState.OPENED.value,
    'assignee_id': 'user-1'
})

# Issues with specific labels
bug_issues = mgr.issues.list_issues({
    'labels': ['bug', 'critical']
})

# Issues by author
my_issues = mgr.issues.list_issues({
    'author_id': 'user-1'
})
```

### Filter Merge Requests
```python
# Open MRs
open_mrs = mgr.merge_requests.list_merge_requests({
    'project_id': 'project-1',
    'state': MergeRequestState.OPENED.value
})

# MRs by author
my_mrs = mgr.merge_requests.list_merge_requests({
    'author_id': 'user-1'
})

# MRs with label
feature_mrs = mgr.merge_requests.list_merge_requests({
    'labels': ['feature']
})
```

### Filter Pipelines
```python
# Successful pipelines
success = mgr.pipelines.list_pipelines('project-1', {
    'status': PipelineStatus.SUCCESS.value
})

# Pipelines on main branch
main_pipelines = mgr.pipelines.list_pipelines('project-1', {
    'ref': 'main'
})
```

### Filter Projects
```python
# Private projects
private = mgr.projects.list_projects({
    'visibility': ProjectVisibility.PRIVATE.value
})

# Search by name
web_projects = mgr.projects.list_projects({
    'search': 'web'
})

# Non-archived projects
active = mgr.projects.list_projects({
    'archived': False
})
```

## Webhook Events

| Event | Trigger | Payload |
|-------|---------|---------|
| **push_events** | Git push to repository | Commits, branch, user |
| **merge_requests_events** | MR created/updated/merged | MR details, changes |
| **issues_events** | Issue created/updated/closed | Issue details, changes |
| **pipeline_events** | Pipeline status change | Pipeline status, jobs |
| **job_events** | Job status change | Job details, artifacts |
| **deployment_events** | Deployment created | Environment, status |
| **tag_push_events** | Tag created/deleted | Tag name, user |
| **wiki_page_events** | Wiki page created/updated | Page title, content |

## Best Practices

### Projects
1. **Naming**: Use clear, descriptive project names
2. **Visibility**: Default to PRIVATE, use PUBLIC only for open source
3. **README**: Always initialize with README
4. **Default Branch**: Use 'main' as default branch
5. **Archiving**: Archive instead of delete for historical projects

### Issues
1. **Labels**: Use consistent label taxonomy (bug, enhancement, documentation)
2. **Assignees**: Assign to responsible person, not teams
3. **Weights**: Use consistent weight scale (1-10)
4. **Due Dates**: Set realistic due dates
5. **Descriptions**: Provide clear problem description and acceptance criteria

### Merge Requests
1. **Small MRs**: Keep MRs focused and small (<400 lines)
2. **Descriptions**: Reference issues with "Closes #123"
3. **Reviewers**: Assign at least one reviewer
4. **Approvals**: Require approvals before merge
5. **Squash**: Enable squash for clean commit history
6. **Remove Branch**: Auto-delete source branch after merge

### Pipelines
1. **Variables**: Use variables for configuration, not hardcoded values
2. **Artifacts**: Save test results and build outputs
3. **Caching**: Cache dependencies to speed up builds
4. **Parallel Jobs**: Run independent jobs in parallel
5. **Manual Gates**: Use manual jobs for production deployments

### Webhooks
1. **SSL Verification**: Always enable SSL verification
2. **Secret Tokens**: Use secret tokens to verify webhook authenticity
3. **Event Selection**: Only subscribe to needed events
4. **Idempotency**: Handle duplicate webhook deliveries
5. **Retries**: Implement exponential backoff for webhook failures

## Common Workflows

### Feature Development
```python
# 1. Create issue
issue = mgr.issues.create_issue({
    'project_id': project['id'],
    'title': 'Add dark mode',
    'labels': ['enhancement']
})

# 2. Create merge request
mr = mgr.merge_requests.create_merge_request({
    'project_id': project['id'],
    'title': 'Add dark mode',
    'description': f"Closes #{issue['iid']}",
    'source_branch': 'feature/dark-mode',
    'target_branch': 'main'
})

# 3. Run pipeline
pipeline = mgr.pipelines.create_pipeline({
    'project_id': project['id'],
    'ref': mr['source_branch']
})
mgr.pipelines.run_pipeline(pipeline['id'])

# 4. Approve and merge
mgr.merge_requests.approve_merge_request(mr['id'], 'reviewer-id')
mgr.merge_requests.merge_merge_request(mr['id'])

# 5. Close issue
mgr.issues.update_issue(issue['id'], {
    'state': IssueState.CLOSED.value
})
```

### Release Process
```python
# 1. List merged MRs since last release
merged_mrs = mgr.merge_requests.list_merge_requests({
    'project_id': project['id'],
    'state': MergeRequestState.MERGED.value
})

# 2. Create release issue
release_issue = mgr.issues.create_issue({
    'project_id': project['id'],
    'title': 'Release v2.0.0',
    'labels': ['release'],
    'description': f"Includes {len(merged_mrs)} changes"
})

# 3. Run release pipeline
release_pipeline = mgr.pipelines.create_pipeline({
    'project_id': project['id'],
    'ref': 'main',
    'variables': {'RELEASE_VERSION': 'v2.0.0'}
})
```

### Bug Tracking
```python
# 1. Create bug issue
bug = mgr.issues.create_issue({
    'project_id': project['id'],
    'title': 'Login fails with special characters',
    'labels': ['bug', 'critical'],
    'assignee_ids': ['dev-id'],
    'weight': 3
})

# 2. Create hotfix MR
hotfix_mr = mgr.merge_requests.create_merge_request({
    'project_id': project['id'],
    'title': 'Fix login bug',
    'source_branch': 'hotfix/login-bug',
    'target_branch': 'main',
    'labels': ['bug', 'critical']
})

# 3. Fast-track merge
mgr.merge_requests.approve_merge_request(hotfix_mr['id'], 'reviewer-id')
mgr.merge_requests.merge_merge_request(hotfix_mr['id'])
mgr.issues.update_issue(bug['id'], {'state': IssueState.CLOSED.value})
```

## Troubleshooting

**Issue**: Cannot merge MR due to conflicts
- Check `has_conflicts` field in MR
- Rebase source branch on target branch
- Resolve conflicts locally and push

**Issue**: Pipeline not starting
- Verify `.gitlab-ci.yml` exists in repository
- Check pipeline variables are correct
- Ensure runners are available

**Issue**: Webhook not triggering
- Verify webhook URL is accessible
- Check event types are enabled
- Review webhook logs for errors
- Ensure SSL verification matches server config

**Issue**: Cannot approve MR
- Check user has appropriate permissions
- Verify approval requirements are configured
- Ensure pipeline has passed (if required)

**Issue**: User cannot create project
- Check `can_create_project` permission
- Verify namespace permissions
- Check project limits for namespace

## Requirements

```
hashlib (standard library)
datetime (standard library)
typing (standard library)
enum (standard library)
```

No external dependencies required.

## Configuration

```python
from gitlab_api import GitLabAPIManager

mgr = GitLabAPIManager(
    gitlab_url='https://gitlab.com',
    token='glpat-xxxxxxxxxxxx'
)
```

## Author

BrillConsulting - Enterprise Cloud Solutions
