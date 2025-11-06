"""
GitLab API Management
Author: BrillConsulting
Description: Comprehensive GitLab REST API integration with projects, issues, merge requests, pipelines, users, groups, and webhooks
"""

import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum


class ProjectVisibility(Enum):
    """Project visibility levels."""
    PRIVATE = "private"
    INTERNAL = "internal"
    PUBLIC = "public"


class MergeRequestState(Enum):
    """Merge request states."""
    OPENED = "opened"
    CLOSED = "closed"
    MERGED = "merged"
    LOCKED = "locked"


class IssueState(Enum):
    """Issue states."""
    OPENED = "opened"
    CLOSED = "closed"


class PipelineStatus(Enum):
    """Pipeline statuses."""
    CREATED = "created"
    WAITING_FOR_RESOURCE = "waiting_for_resource"
    PREPARING = "preparing"
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELED = "canceled"
    SKIPPED = "skipped"
    MANUAL = "manual"


class AccessLevel(Enum):
    """Project/group access levels."""
    NO_ACCESS = 0
    MINIMAL_ACCESS = 5
    GUEST = 10
    REPORTER = 20
    DEVELOPER = 30
    MAINTAINER = 40
    OWNER = 50


class ProjectManager:
    """Project management operations."""

    def __init__(self):
        self.projects = {}

    def create_project(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create GitLab project.

        Config:
            name: str - Project name
            path: str - URL path
            namespace_id: str - Parent group/namespace
            description: str - Project description
            visibility: str - private, internal, public
            initialize_with_readme: bool - Create README
            default_branch: str - Default branch name
        """
        project_id = f"project-{len(self.projects) + 1}"
        namespace = config.get('namespace_id', 'root')
        path = config.get('path', config['name'].lower().replace(' ', '-'))

        project = {
            'id': project_id,
            'name': config['name'],
            'path': path,
            'path_with_namespace': f"{namespace}/{path}",
            'namespace_id': namespace,
            'description': config.get('description', ''),
            'visibility': config.get('visibility', ProjectVisibility.PRIVATE.value),
            'default_branch': config.get('default_branch', 'main'),
            'ssh_url_to_repo': f"git@gitlab.com:{namespace}/{path}.git",
            'http_url_to_repo': f"https://gitlab.com/{namespace}/{path}.git",
            'web_url': f"https://gitlab.com/{namespace}/{path}",
            'readme_url': f"https://gitlab.com/{namespace}/{path}/-/blob/main/README.md" if config.get('initialize_with_readme') else None,
            'star_count': 0,
            'forks_count': 0,
            'open_issues_count': 0,
            'created_at': datetime.now().isoformat(),
            'last_activity_at': datetime.now().isoformat(),
            'archived': False,
            'shared_with_groups': []
        }

        self.projects[project_id] = project
        return project

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project by ID."""
        return self.projects.get(project_id)

    def list_projects(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List projects with optional filters.

        Filters:
            visibility: str - Filter by visibility level
            archived: bool - Include/exclude archived projects
            search: str - Search by name/path
            owned: bool - Only owned projects
        """
        projects = list(self.projects.values())

        if filters:
            if 'visibility' in filters:
                projects = [p for p in projects if p['visibility'] == filters['visibility']]
            if 'archived' in filters:
                projects = [p for p in projects if p['archived'] == filters['archived']]
            if 'search' in filters:
                search = filters['search'].lower()
                projects = [p for p in projects if search in p['name'].lower() or search in p['path'].lower()]

        return projects

    def archive_project(self, project_id: str) -> Dict[str, Any]:
        """Archive project."""
        if project_id in self.projects:
            self.projects[project_id]['archived'] = True
            self.projects[project_id]['archived_at'] = datetime.now().isoformat()
            return self.projects[project_id]
        raise ValueError(f"Project {project_id} not found")

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update project settings."""
        if project_id in self.projects:
            self.projects[project_id].update(updates)
            self.projects[project_id]['updated_at'] = datetime.now().isoformat()
            return self.projects[project_id]
        raise ValueError(f"Project {project_id} not found")


class IssueManager:
    """Issue tracking operations."""

    def __init__(self):
        self.issues = {}
        self.issue_counter = {}

    def create_issue(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create issue.

        Config:
            project_id: str - Project identifier
            title: str - Issue title
            description: str - Issue description
            assignee_ids: List[str] - User IDs to assign
            labels: List[str] - Issue labels
            milestone_id: str - Milestone ID
            due_date: str - Due date (YYYY-MM-DD)
            weight: int - Issue weight
            confidential: bool - Confidential flag
        """
        project_id = config['project_id']
        if project_id not in self.issue_counter:
            self.issue_counter[project_id] = 0

        self.issue_counter[project_id] += 1
        iid = self.issue_counter[project_id]
        issue_id = f"issue-{len(self.issues) + 1}"

        issue = {
            'id': issue_id,
            'iid': iid,
            'project_id': project_id,
            'title': config['title'],
            'description': config.get('description', ''),
            'state': IssueState.OPENED.value,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'closed_at': None,
            'labels': config.get('labels', []),
            'assignees': config.get('assignee_ids', []),
            'author': config.get('author_id', 'user-1'),
            'milestone': config.get('milestone_id'),
            'due_date': config.get('due_date'),
            'weight': config.get('weight'),
            'confidential': config.get('confidential', False),
            'upvotes': 0,
            'downvotes': 0,
            'user_notes_count': 0,
            'web_url': f"https://gitlab.com/project/-/issues/{iid}"
        }

        self.issues[issue_id] = issue
        return issue

    def update_issue(self, issue_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update issue."""
        if issue_id in self.issues:
            self.issues[issue_id].update(updates)
            self.issues[issue_id]['updated_at'] = datetime.now().isoformat()

            if updates.get('state') == IssueState.CLOSED.value:
                self.issues[issue_id]['closed_at'] = datetime.now().isoformat()

            return self.issues[issue_id]
        raise ValueError(f"Issue {issue_id} not found")

    def list_issues(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List issues with filters.

        Filters:
            project_id: str - Filter by project
            state: str - opened, closed
            labels: List[str] - Filter by labels
            assignee_id: str - Filter by assignee
            author_id: str - Filter by author
            milestone: str - Filter by milestone
        """
        issues = list(self.issues.values())

        if filters:
            if 'project_id' in filters:
                issues = [i for i in issues if i['project_id'] == filters['project_id']]
            if 'state' in filters:
                issues = [i for i in issues if i['state'] == filters['state']]
            if 'labels' in filters:
                labels = filters['labels']
                issues = [i for i in issues if any(label in i['labels'] for label in labels)]
            if 'assignee_id' in filters:
                issues = [i for i in issues if filters['assignee_id'] in i['assignees']]
            if 'author_id' in filters:
                issues = [i for i in issues if i['author'] == filters['author_id']]

        return issues


class MergeRequestManager:
    """Merge request operations."""

    def __init__(self):
        self.merge_requests = {}
        self.mr_counter = {}

    def create_merge_request(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create merge request.

        Config:
            project_id: str - Project identifier
            title: str - MR title
            description: str - MR description
            source_branch: str - Source branch
            target_branch: str - Target branch
            assignee_ids: List[str] - Assignee user IDs
            reviewer_ids: List[str] - Reviewer user IDs
            labels: List[str] - MR labels
            remove_source_branch: bool - Delete source after merge
            squash: bool - Squash commits on merge
        """
        project_id = config['project_id']
        if project_id not in self.mr_counter:
            self.mr_counter[project_id] = 0

        self.mr_counter[project_id] += 1
        iid = self.mr_counter[project_id]
        mr_id = f"mr-{len(self.merge_requests) + 1}"

        merge_request = {
            'id': mr_id,
            'iid': iid,
            'project_id': project_id,
            'title': config['title'],
            'description': config.get('description', ''),
            'state': MergeRequestState.OPENED.value,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'merged_at': None,
            'closed_at': None,
            'source_branch': config['source_branch'],
            'target_branch': config['target_branch'],
            'author': config.get('author_id', 'user-1'),
            'assignees': config.get('assignee_ids', []),
            'reviewers': config.get('reviewer_ids', []),
            'labels': config.get('labels', []),
            'draft': config.get('draft', False),
            'work_in_progress': config.get('work_in_progress', False),
            'merge_status': 'can_be_merged',
            'has_conflicts': False,
            'blocking_discussions_resolved': True,
            'approvals_required': config.get('approvals_required', 1),
            'approvals': [],
            'pipeline_id': None,
            'pipeline_status': None,
            'sha': hashlib.sha1(f"{mr_id}-{datetime.now()}".encode()).hexdigest(),
            'squash': config.get('squash', False),
            'remove_source_branch': config.get('remove_source_branch', False),
            'web_url': f"https://gitlab.com/project/-/merge_requests/{iid}"
        }

        self.merge_requests[mr_id] = merge_request
        return merge_request

    def approve_merge_request(self, mr_id: str, user_id: str) -> Dict[str, Any]:
        """Approve merge request."""
        if mr_id in self.merge_requests:
            approval = {
                'user_id': user_id,
                'approved_at': datetime.now().isoformat()
            }
            self.merge_requests[mr_id]['approvals'].append(approval)
            self.merge_requests[mr_id]['updated_at'] = datetime.now().isoformat()
            return self.merge_requests[mr_id]
        raise ValueError(f"Merge request {mr_id} not found")

    def merge_merge_request(self, mr_id: str, merge_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge merge request.

        Options:
            merge_commit_message: str - Custom merge commit message
            squash_commit_message: str - Custom squash commit message
            should_remove_source_branch: bool - Delete source branch
        """
        if mr_id not in self.merge_requests:
            raise ValueError(f"Merge request {mr_id} not found")

        mr = self.merge_requests[mr_id]
        merge_options = merge_options or {}

        if mr['has_conflicts']:
            raise ValueError("Cannot merge: MR has conflicts")

        if len(mr['approvals']) < mr['approvals_required']:
            raise ValueError(f"Cannot merge: Requires {mr['approvals_required']} approvals, has {len(mr['approvals'])}")

        mr['state'] = MergeRequestState.MERGED.value
        mr['merged_at'] = datetime.now().isoformat()
        mr['merged_by'] = merge_options.get('merged_by', 'user-1')
        mr['merge_commit_sha'] = hashlib.sha1(f"merge-{mr_id}".encode()).hexdigest()

        return mr

    def list_merge_requests(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List merge requests with filters."""
        mrs = list(self.merge_requests.values())

        if filters:
            if 'project_id' in filters:
                mrs = [m for m in mrs if m['project_id'] == filters['project_id']]
            if 'state' in filters:
                mrs = [m for m in mrs if m['state'] == filters['state']]
            if 'labels' in filters:
                labels = filters['labels']
                mrs = [m for m in mrs if any(label in m['labels'] for label in labels)]
            if 'author_id' in filters:
                mrs = [m for m in mrs if m['author'] == filters['author_id']]

        return mrs


class PipelineManager:
    """CI/CD pipeline operations."""

    def __init__(self):
        self.pipelines = {}

    def create_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create CI/CD pipeline.

        Config:
            project_id: str - Project identifier
            ref: str - Branch/tag name
            variables: Dict - Pipeline variables
        """
        pipeline_id = f"pipeline-{len(self.pipelines) + 1}"

        pipeline = {
            'id': pipeline_id,
            'project_id': config['project_id'],
            'ref': config.get('ref', 'main'),
            'sha': hashlib.sha1(f"{pipeline_id}-commit".encode()).hexdigest(),
            'status': PipelineStatus.PENDING.value,
            'source': config.get('source', 'push'),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'started_at': None,
            'finished_at': None,
            'duration': None,
            'variables': config.get('variables', {}),
            'jobs': [],
            'web_url': f"https://gitlab.com/project/-/pipelines/{pipeline_id}"
        }

        self.pipelines[pipeline_id] = pipeline
        return pipeline

    def run_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """Run pipeline."""
        if pipeline_id in self.pipelines:
            self.pipelines[pipeline_id]['status'] = PipelineStatus.RUNNING.value
            self.pipelines[pipeline_id]['started_at'] = datetime.now().isoformat()
            return self.pipelines[pipeline_id]
        raise ValueError(f"Pipeline {pipeline_id} not found")

    def complete_pipeline(self, pipeline_id: str, status: str) -> Dict[str, Any]:
        """Complete pipeline with status."""
        if pipeline_id in self.pipelines:
            finished_at = datetime.now()
            started_at = datetime.fromisoformat(self.pipelines[pipeline_id]['started_at'] or datetime.now().isoformat())

            self.pipelines[pipeline_id]['status'] = status
            self.pipelines[pipeline_id]['finished_at'] = finished_at.isoformat()
            self.pipelines[pipeline_id]['duration'] = (finished_at - started_at).total_seconds()

            return self.pipelines[pipeline_id]
        raise ValueError(f"Pipeline {pipeline_id} not found")

    def list_pipelines(self, project_id: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List project pipelines."""
        pipelines = [p for p in self.pipelines.values() if p['project_id'] == project_id]

        if filters:
            if 'status' in filters:
                pipelines = [p for p in pipelines if p['status'] == filters['status']]
            if 'ref' in filters:
                pipelines = [p for p in pipelines if p['ref'] == filters['ref']]

        return pipelines


class WebhookManager:
    """Project webhook operations."""

    def __init__(self):
        self.webhooks = {}

    def create_webhook(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create project webhook.

        Config:
            project_id: str - Project identifier
            url: str - Webhook URL
            token: str - Secret token
            push_events: bool - Trigger on push
            merge_requests_events: bool - Trigger on MR events
            issues_events: bool - Trigger on issue events
            pipeline_events: bool - Trigger on pipeline events
            job_events: bool - Trigger on job events
            deployment_events: bool - Trigger on deployments
            enable_ssl_verification: bool - Verify SSL
        """
        webhook_id = f"webhook-{len(self.webhooks) + 1}"

        webhook = {
            'id': webhook_id,
            'project_id': config['project_id'],
            'url': config['url'],
            'token': config.get('token'),
            'push_events': config.get('push_events', True),
            'merge_requests_events': config.get('merge_requests_events', True),
            'issues_events': config.get('issues_events', True),
            'pipeline_events': config.get('pipeline_events', True),
            'job_events': config.get('job_events', False),
            'deployment_events': config.get('deployment_events', False),
            'tag_push_events': config.get('tag_push_events', False),
            'wiki_page_events': config.get('wiki_page_events', False),
            'enable_ssl_verification': config.get('enable_ssl_verification', True),
            'created_at': datetime.now().isoformat(),
            'last_triggered_at': None
        }

        self.webhooks[webhook_id] = webhook
        return webhook

    def trigger_webhook(self, webhook_id: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger webhook with event."""
        if webhook_id in self.webhooks:
            self.webhooks[webhook_id]['last_triggered_at'] = datetime.now().isoformat()
            return {
                'webhook_id': webhook_id,
                'event_type': event_type,
                'status': 'triggered',
                'triggered_at': datetime.now().isoformat()
            }
        raise ValueError(f"Webhook {webhook_id} not found")

    def list_webhooks(self, project_id: str) -> List[Dict[str, Any]]:
        """List project webhooks."""
        return [w for w in self.webhooks.values() if w['project_id'] == project_id]


class UserManager:
    """User management operations."""

    def __init__(self):
        self.users = {}

    def create_user(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create user.

        Config:
            username: str - Username
            email: str - Email address
            name: str - Full name
            password: str - Password
            admin: bool - Admin flag
            can_create_group: bool - Can create groups
            skip_confirmation: bool - Skip email confirmation
        """
        user_id = f"user-{len(self.users) + 1}"

        user = {
            'id': user_id,
            'username': config['username'],
            'email': config['email'],
            'name': config.get('name', config['username']),
            'state': 'active',
            'avatar_url': f"https://www.gravatar.com/avatar/{hashlib.md5(config['email'].encode()).hexdigest()}",
            'web_url': f"https://gitlab.com/{config['username']}",
            'created_at': datetime.now().isoformat(),
            'is_admin': config.get('admin', False),
            'bio': config.get('bio', ''),
            'location': config.get('location', ''),
            'public_email': config.get('public_email', ''),
            'skype': config.get('skype', ''),
            'linkedin': config.get('linkedin', ''),
            'twitter': config.get('twitter', ''),
            'website_url': config.get('website_url', ''),
            'organization': config.get('organization', ''),
            'can_create_group': config.get('can_create_group', True),
            'can_create_project': config.get('can_create_project', True),
            'two_factor_enabled': False,
            'confirmed_at': datetime.now().isoformat() if config.get('skip_confirmation') else None
        }

        self.users[user_id] = user
        return user

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        return self.users.get(user_id)

    def list_users(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List users with filters."""
        users = list(self.users.values())

        if filters:
            if 'active' in filters:
                users = [u for u in users if (u['state'] == 'active') == filters['active']]
            if 'search' in filters:
                search = filters['search'].lower()
                users = [u for u in users if search in u['username'].lower() or search in u['name'].lower()]

        return users


class GitLabAPIManager:
    """Main GitLab API orchestration."""

    def __init__(self, gitlab_url: str = 'https://gitlab.com', token: str = None):
        self.gitlab_url = gitlab_url
        self.token = token
        self.projects = ProjectManager()
        self.issues = IssueManager()
        self.merge_requests = MergeRequestManager()
        self.pipelines = PipelineManager()
        self.webhooks = WebhookManager()
        self.users = UserManager()

    def get_api_version(self) -> Dict[str, Any]:
        """Get API version info."""
        return {
            'version': '15.0',
            'revision': 'abc123',
            'kas': {
                'enabled': True,
                'version': '15.0.0'
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get instance statistics."""
        return {
            'gitlab_url': self.gitlab_url,
            'projects_count': len(self.projects.projects),
            'issues_count': len(self.issues.issues),
            'merge_requests_count': len(self.merge_requests.merge_requests),
            'pipelines_count': len(self.pipelines.pipelines),
            'users_count': len(self.users.users),
            'webhooks_count': len(self.webhooks.webhooks),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate comprehensive GitLab API usage."""
    print("=" * 70)
    print("GitLab API Management - Comprehensive Demo")
    print("=" * 70)

    mgr = GitLabAPIManager(
        gitlab_url='https://gitlab.com',
        token='glpat-xxxxxxxxxxxx'
    )

    # 1. Create users
    print("\n1. Creating users...")
    dev1 = mgr.users.create_user({
        'username': 'developer1',
        'email': 'dev1@example.com',
        'name': 'Alice Developer',
        'can_create_group': True
    })
    print(f"   ✓ User created: {dev1['username']} ({dev1['email']})")

    reviewer = mgr.users.create_user({
        'username': 'reviewer1',
        'email': 'reviewer@example.com',
        'name': 'Bob Reviewer'
    })
    print(f"   ✓ User created: {reviewer['username']} ({reviewer['email']})")

    # 2. Create project
    print("\n2. Creating project...")
    project = mgr.projects.create_project({
        'name': 'My Awesome App',
        'namespace_id': 'mycompany',
        'description': 'A production-ready web application',
        'visibility': ProjectVisibility.PRIVATE.value,
        'initialize_with_readme': True
    })
    print(f"   ✓ Project created: {project['name']}")
    print(f"   URL: {project['web_url']}")
    print(f"   Clone: {project['http_url_to_repo']}")

    # 3. Create issues
    print("\n3. Creating issues...")
    issue1 = mgr.issues.create_issue({
        'project_id': project['id'],
        'title': 'Implement user authentication',
        'description': 'Add JWT-based authentication system',
        'labels': ['enhancement', 'backend'],
        'assignee_ids': [dev1['id']],
        'weight': 5
    })
    print(f"   ✓ Issue created: #{issue1['iid']} - {issue1['title']}")

    issue2 = mgr.issues.create_issue({
        'project_id': project['id'],
        'title': 'Fix login bug',
        'description': 'Users cannot log in with special characters',
        'labels': ['bug', 'critical'],
        'assignee_ids': [dev1['id']]
    })
    print(f"   ✓ Issue created: #{issue2['iid']} - {issue2['title']}")

    # 4. Create merge request
    print("\n4. Creating merge request...")
    mr = mgr.merge_requests.create_merge_request({
        'project_id': project['id'],
        'title': 'Add user authentication system',
        'description': f"Implements JWT authentication.\n\nCloses #{issue1['iid']}",
        'source_branch': 'feature/user-auth',
        'target_branch': 'main',
        'author_id': dev1['id'],
        'reviewer_ids': [reviewer['id']],
        'labels': ['enhancement'],
        'squash': True,
        'remove_source_branch': True
    })
    print(f"   ✓ Merge request created: !{mr['iid']} - {mr['title']}")
    print(f"   {mr['source_branch']} → {mr['target_branch']}")

    # 5. Create pipeline
    print("\n5. Creating and running pipeline...")
    pipeline = mgr.pipelines.create_pipeline({
        'project_id': project['id'],
        'ref': mr['source_branch'],
        'variables': {'ENVIRONMENT': 'staging'}
    })
    print(f"   ✓ Pipeline created: {pipeline['id']}")

    mgr.pipelines.run_pipeline(pipeline['id'])
    print(f"   ✓ Pipeline running: {pipeline['id']}")

    mgr.pipelines.complete_pipeline(pipeline['id'], PipelineStatus.SUCCESS.value)
    print(f"   ✓ Pipeline completed: SUCCESS")

    # 6. Approve and merge MR
    print("\n6. Approving and merging merge request...")
    mgr.merge_requests.approve_merge_request(mr['id'], reviewer['id'])
    print(f"   ✓ MR approved by {reviewer['username']}")

    merged_mr = mgr.merge_requests.merge_merge_request(mr['id'], {
        'merged_by': reviewer['id']
    })
    print(f"   ✓ MR merged: !{merged_mr['iid']}")

    # 7. Close issue
    print("\n7. Closing issue...")
    closed_issue = mgr.issues.update_issue(issue1['id'], {
        'state': IssueState.CLOSED.value
    })
    print(f"   ✓ Issue closed: #{closed_issue['iid']}")

    # 8. Create webhook
    print("\n8. Creating webhook...")
    webhook = mgr.webhooks.create_webhook({
        'project_id': project['id'],
        'url': 'https://ci.example.com/gitlab-webhook',
        'push_events': True,
        'merge_requests_events': True,
        'pipeline_events': True
    })
    print(f"   ✓ Webhook created: {webhook['url']}")
    print(f"   Events: push={webhook['push_events']}, MR={webhook['merge_requests_events']}")

    # 9. List resources
    print("\n9. Listing resources...")
    all_issues = mgr.issues.list_issues({'project_id': project['id']})
    print(f"   ✓ Total issues: {len(all_issues)}")

    open_issues = mgr.issues.list_issues({
        'project_id': project['id'],
        'state': IssueState.OPENED.value
    })
    print(f"   ✓ Open issues: {len(open_issues)}")

    pipelines = mgr.pipelines.list_pipelines(project['id'])
    print(f"   ✓ Pipelines: {len(pipelines)}")

    # 10. Statistics
    print("\n10. API statistics...")
    stats = mgr.get_statistics()
    print(f"   ✓ GitLab URL: {stats['gitlab_url']}")
    print(f"   ✓ Projects: {stats['projects_count']}")
    print(f"   ✓ Issues: {stats['issues_count']}")
    print(f"   ✓ Merge Requests: {stats['merge_requests_count']}")
    print(f"   ✓ Pipelines: {stats['pipelines_count']}")
    print(f"   ✓ Users: {stats['users_count']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
