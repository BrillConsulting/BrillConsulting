# GitLab Runner Management

Complete GitLab Runner registration, configuration, and monitoring.

## Features

- **Runner Registration**: Register Docker, Shell, Kubernetes runners
- **Configuration Generation**: Generate config.toml files
- **Docker Deployment**: docker-compose.yml for runners
- **Kubernetes Deployment**: K8s manifests for runners
- **Runner Control**: Pause, resume, unregister runners
- **Status Monitoring**: Check runner status and health
- **Job Tracking**: Monitor jobs executed by runners
- **Multi-Executor**: Support for Docker, Shell, Kubernetes executors

## Technologies

- GitLab Runner
- Docker
- Kubernetes
- Shell executor

## Usage

```python
from gitlab_runner import GitLabRunnerManager

# Initialize runner manager
runner_mgr = GitLabRunnerManager(
    gitlab_url='https://gitlab.example.com',
    registration_token='GR1348941...'
)

# Register Docker runner
docker_runner = runner_mgr.register_runner({
    'description': 'Docker Runner 1',
    'executor': 'docker',
    'tags': ['docker', 'linux']
})

# Generate config.toml
config_toml = runner_mgr.generate_runner_config({
    'concurrent': 4,
    'runners': [
        {'name': 'docker-runner-1', 'executor': 'docker'}
    ]
})

# Check runner status
status = runner_mgr.get_runner_status(docker_runner['id'])

# Pause/resume runner
runner_mgr.pause_runner(runner_id)
runner_mgr.resume_runner(runner_id)
```

## Demo

```bash
python gitlab_runner.py
```
