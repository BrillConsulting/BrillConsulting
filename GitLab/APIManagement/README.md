# GitLab API Management

Complete GitLab API integration for projects, issues, merge requests, users, and more.

## Features

- **Project Management**: Create, list, and manage projects
- **Issue Tracking**: Create and manage issues with labels, assignees
- **Merge Requests**: Create, approve, and merge MRs
- **User Management**: Create and manage users
- **Branch & Tag Management**: Create branches and tags
- **Webhook Configuration**: Set up project webhooks
- **Project Statistics**: Get commit counts, storage size, metrics
- **Filters**: Filter projects, issues, MRs by various criteria

## Technologies

- GitLab REST API
- python-gitlab
- requests

## Usage

```python
from gitlab_api import GitLabAPI

# Initialize API client
gitlab = GitLabAPI(
    gitlab_url='https://gitlab.example.com',
    token='glpat-xxxxxxxxxxxx'
)

# Create project
project = gitlab.create_project({
    'name': 'my-project',
    'visibility': 'private'
})

# Create issue
issue = gitlab.create_issue({
    'project_id': project['id'],
    'title': 'Add feature',
    'labels': ['enhancement']
})

# Create merge request
mr = gitlab.create_merge_request({
    'project_id': project['id'],
    'title': 'Add new feature',
    'source_branch': 'feature-branch',
    'target_branch': 'main'
})

# Approve and merge
gitlab.approve_merge_request(mr['id'])
gitlab.merge_merge_request(mr['id'])
```

## Demo

```bash
python gitlab_api.py
```
