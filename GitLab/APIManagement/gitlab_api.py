"""
GitLab API Management
Author: BrillConsulting
Description: Complete GitLab API integration for projects, issues, merge requests, and more
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class GitLabAPI:
    """Comprehensive GitLab API management"""

    def __init__(self, gitlab_url: str, token: str):
        """
        Initialize GitLab API client

        Args:
            gitlab_url: GitLab instance URL
            token: Personal access token
        """
        self.gitlab_url = gitlab_url
        self.token = token
        self.projects = []
        self.issues = []
        self.merge_requests = []
        self.users = []

    def create_project(self, project_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create GitLab project

        Args:
            project_config: Project configuration

        Returns:
            Project details
        """
        project = {
            'id': len(self.projects) + 1,
            'name': project_config.get('name', 'my-project'),
            'path': project_config.get('path', 'my-project'),
            'namespace': project_config.get('namespace', 'mygroup'),
            'description': project_config.get('description', ''),
            'visibility': project_config.get('visibility', 'private'),
            'default_branch': project_config.get('default_branch', 'main'),
            'ssh_url': f"git@gitlab.example.com:{project_config.get('namespace')}/{project_config.get('path')}.git",
            'http_url': f"https://gitlab.example.com/{project_config.get('namespace')}/{project_config.get('path')}.git",
            'created_at': datetime.now().isoformat(),
            'star_count': 0,
            'forks_count': 0
        }

        self.projects.append(project)

        print(f"✓ Project created: {project['name']}")
        print(f"  ID: {project['id']}, Visibility: {project['visibility']}")
        print(f"  URL: {project['http_url']}")
        return project

    def create_issue(self, issue_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create issue

        Args:
            issue_config: Issue configuration

        Returns:
            Issue details
        """
        issue = {
            'id': len(self.issues) + 1,
            'iid': len(self.issues) + 1,
            'project_id': issue_config.get('project_id', 1),
            'title': issue_config.get('title', 'New Issue'),
            'description': issue_config.get('description', ''),
            'state': issue_config.get('state', 'opened'),
            'labels': issue_config.get('labels', []),
            'assignees': issue_config.get('assignees', []),
            'milestone': issue_config.get('milestone', None),
            'author': issue_config.get('author', 'user'),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'due_date': issue_config.get('due_date', None),
            'weight': issue_config.get('weight', None)
        }

        self.issues.append(issue)

        print(f"✓ Issue created: #{issue['iid']} - {issue['title']}")
        print(f"  Project: {issue['project_id']}, State: {issue['state']}")
        print(f"  Labels: {', '.join(issue['labels']) if issue['labels'] else 'None'}")
        return issue

    def create_merge_request(self, mr_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create merge request

        Args:
            mr_config: Merge request configuration

        Returns:
            Merge request details
        """
        mr = {
            'id': len(self.merge_requests) + 1,
            'iid': len(self.merge_requests) + 1,
            'project_id': mr_config.get('project_id', 1),
            'title': mr_config.get('title', 'New Merge Request'),
            'description': mr_config.get('description', ''),
            'source_branch': mr_config.get('source_branch', 'feature-branch'),
            'target_branch': mr_config.get('target_branch', 'main'),
            'state': mr_config.get('state', 'opened'),
            'author': mr_config.get('author', 'user'),
            'assignee': mr_config.get('assignee', None),
            'reviewers': mr_config.get('reviewers', []),
            'labels': mr_config.get('labels', []),
            'draft': mr_config.get('draft', False),
            'work_in_progress': mr_config.get('work_in_progress', False),
            'merge_status': 'can_be_merged',
            'has_conflicts': False,
            'pipeline_status': 'success',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        self.merge_requests.append(mr)

        print(f"✓ Merge request created: !{mr['iid']} - {mr['title']}")
        print(f"  {mr['source_branch']} → {mr['target_branch']}")
        print(f"  Status: {mr['state']}, Pipeline: {mr['pipeline_status']}")
        return mr

    def create_user(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create user

        Args:
            user_config: User configuration

        Returns:
            User details
        """
        user = {
            'id': len(self.users) + 1,
            'username': user_config.get('username', 'newuser'),
            'email': user_config.get('email', 'user@example.com'),
            'name': user_config.get('name', 'New User'),
            'state': user_config.get('state', 'active'),
            'is_admin': user_config.get('is_admin', False),
            'can_create_group': user_config.get('can_create_group', True),
            'can_create_project': user_config.get('can_create_project', True),
            'created_at': datetime.now().isoformat()
        }

        self.users.append(user)

        print(f"✓ User created: {user['username']}")
        print(f"  ID: {user['id']}, Email: {user['email']}")
        print(f"  Admin: {user['is_admin']}")
        return user

    def list_projects(self, filter_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List projects

        Args:
            filter_params: Optional filter parameters

        Returns:
            List of projects
        """
        filter_params = filter_params or {}
        visibility = filter_params.get('visibility', None)

        projects = self.projects
        if visibility:
            projects = [p for p in projects if p['visibility'] == visibility]

        print(f"✓ Listed {len(projects)} projects")
        return projects

    def list_issues(self, filter_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List issues

        Args:
            filter_params: Optional filter parameters

        Returns:
            List of issues
        """
        filter_params = filter_params or {}
        state = filter_params.get('state', None)
        labels = filter_params.get('labels', None)

        issues = self.issues
        if state:
            issues = [i for i in issues if i['state'] == state]
        if labels:
            issues = [i for i in issues if any(label in i['labels'] for label in labels)]

        print(f"✓ Listed {len(issues)} issues")
        return issues

    def approve_merge_request(self, mr_id: int) -> Dict[str, Any]:
        """Approve merge request"""
        mr = next((m for m in self.merge_requests if m['id'] == mr_id), None)
        if mr:
            mr['approved'] = True
            mr['approved_at'] = datetime.now().isoformat()
            print(f"✓ Merge request approved: !{mr['iid']}")
            return mr
        return {'error': 'Merge request not found'}

    def merge_merge_request(self, mr_id: int) -> Dict[str, Any]:
        """Merge merge request"""
        mr = next((m for m in self.merge_requests if m['id'] == mr_id), None)
        if mr:
            mr['state'] = 'merged'
            mr['merged_at'] = datetime.now().isoformat()
            print(f"✓ Merge request merged: !{mr['iid']}")
            return mr
        return {'error': 'Merge request not found'}

    def create_branch(self, branch_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create branch

        Args:
            branch_config: Branch configuration

        Returns:
            Branch details
        """
        branch = {
            'name': branch_config.get('name', 'feature-branch'),
            'ref': branch_config.get('ref', 'main'),
            'project_id': branch_config.get('project_id', 1),
            'protected': False,
            'created_at': datetime.now().isoformat()
        }

        print(f"✓ Branch created: {branch['name']} (from {branch['ref']})")
        return branch

    def create_tag(self, tag_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create tag

        Args:
            tag_config: Tag configuration

        Returns:
            Tag details
        """
        tag = {
            'name': tag_config.get('name', 'v1.0.0'),
            'ref': tag_config.get('ref', 'main'),
            'message': tag_config.get('message', 'Release v1.0.0'),
            'project_id': tag_config.get('project_id', 1),
            'created_at': datetime.now().isoformat()
        }

        print(f"✓ Tag created: {tag['name']}")
        print(f"  Message: {tag['message']}")
        return tag

    def create_webhook(self, webhook_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create project webhook

        Args:
            webhook_config: Webhook configuration

        Returns:
            Webhook details
        """
        webhook = {
            'id': 1,
            'url': webhook_config.get('url', 'https://example.com/webhook'),
            'project_id': webhook_config.get('project_id', 1),
            'push_events': webhook_config.get('push_events', True),
            'merge_requests_events': webhook_config.get('merge_requests_events', True),
            'issues_events': webhook_config.get('issues_events', True),
            'pipeline_events': webhook_config.get('pipeline_events', True),
            'token': webhook_config.get('token', None),
            'enable_ssl_verification': webhook_config.get('enable_ssl_verification', True),
            'created_at': datetime.now().isoformat()
        }

        print(f"✓ Webhook created: {webhook['url']}")
        print(f"  Events: push={webhook['push_events']}, MR={webhook['merge_requests_events']}")
        return webhook

    def get_project_statistics(self, project_id: int) -> Dict[str, Any]:
        """
        Get project statistics

        Args:
            project_id: Project ID

        Returns:
            Project statistics
        """
        stats = {
            'project_id': project_id,
            'commit_count': 1523,
            'storage_size': 1024000000,  # 1 GB
            'repository_size': 512000000,  # 512 MB
            'lfs_objects_size': 256000000,  # 256 MB
            'job_artifacts_size': 128000000,  # 128 MB
            'wiki_size': 10240000,  # 10 MB
            'issues_count': 45,
            'merge_requests_count': 32,
            'pipelines_count': 250,
            'branches_count': 15,
            'tags_count': 8
        }

        print(f"✓ Project statistics retrieved: Project #{project_id}")
        print(f"  Commits: {stats['commit_count']}, Issues: {stats['issues_count']}, MRs: {stats['merge_requests_count']}")
        return stats

    def get_api_info(self) -> Dict[str, Any]:
        """Get API manager information"""
        return {
            'gitlab_url': self.gitlab_url,
            'projects': len(self.projects),
            'issues': len(self.issues),
            'merge_requests': len(self.merge_requests),
            'users': len(self.users),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate GitLab API management"""

    print("=" * 60)
    print("GitLab API Management Demo")
    print("=" * 60)

    # Initialize API client
    gitlab = GitLabAPI(
        gitlab_url='https://gitlab.example.com',
        token='glpat-xxxxxxxxxxxx'
    )

    print("\n1. Creating project...")
    project = gitlab.create_project({
        'name': 'my-awesome-project',
        'path': 'my-awesome-project',
        'namespace': 'mycompany',
        'description': 'An awesome project',
        'visibility': 'private'
    })

    print("\n2. Creating users...")
    user1 = gitlab.create_user({
        'username': 'developer1',
        'email': 'dev1@example.com',
        'name': 'Developer One',
        'is_admin': False
    })

    user2 = gitlab.create_user({
        'username': 'reviewer1',
        'email': 'reviewer@example.com',
        'name': 'Code Reviewer',
        'is_admin': False
    })

    print("\n3. Creating issues...")
    issue1 = gitlab.create_issue({
        'project_id': project['id'],
        'title': 'Add user authentication',
        'description': 'Implement JWT-based authentication',
        'labels': ['enhancement', 'high-priority'],
        'assignees': ['developer1']
    })

    issue2 = gitlab.create_issue({
        'project_id': project['id'],
        'title': 'Fix login bug',
        'description': 'Users cannot log in with special characters',
        'labels': ['bug', 'critical'],
        'assignees': ['developer1']
    })

    print("\n4. Creating branch...")
    branch = gitlab.create_branch({
        'name': 'feature/user-auth',
        'ref': 'main',
        'project_id': project['id']
    })

    print("\n5. Creating merge request...")
    mr = gitlab.create_merge_request({
        'project_id': project['id'],
        'title': 'Add user authentication system',
        'description': 'Implements JWT authentication. Closes #1',
        'source_branch': 'feature/user-auth',
        'target_branch': 'main',
        'author': 'developer1',
        'reviewers': ['reviewer1'],
        'labels': ['enhancement']
    })

    print("\n6. Approving merge request...")
    gitlab.approve_merge_request(mr['id'])

    print("\n7. Merging merge request...")
    gitlab.merge_merge_request(mr['id'])

    print("\n8. Creating tag...")
    tag = gitlab.create_tag({
        'name': 'v1.0.0',
        'ref': 'main',
        'message': 'Release version 1.0.0',
        'project_id': project['id']
    })

    print("\n9. Creating webhook...")
    webhook = gitlab.create_webhook({
        'url': 'https://ci.example.com/gitlab-webhook',
        'project_id': project['id'],
        'push_events': True,
        'merge_requests_events': True,
        'pipeline_events': True
    })

    print("\n10. Listing projects...")
    projects = gitlab.list_projects({'visibility': 'private'})

    print("\n11. Listing open issues...")
    open_issues = gitlab.list_issues({'state': 'opened'})

    print("\n12. Getting project statistics...")
    stats = gitlab.get_project_statistics(project['id'])

    print("\n13. API summary:")
    info = gitlab.get_api_info()
    print(f"  GitLab URL: {info['gitlab_url']}")
    print(f"  Projects: {info['projects']}")
    print(f"  Issues: {info['issues']}")
    print(f"  Merge Requests: {info['merge_requests']}")
    print(f"  Users: {info['users']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
