"""
GitLab Runner Management
Author: BrillConsulting
Description: Complete GitLab Runner registration, configuration, and monitoring
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class GitLabRunnerManager:
    """Comprehensive GitLab Runner management"""

    def __init__(self, gitlab_url: str, registration_token: str):
        """
        Initialize GitLab Runner manager

        Args:
            gitlab_url: GitLab instance URL
            registration_token: Registration token
        """
        self.gitlab_url = gitlab_url
        self.registration_token = registration_token
        self.runners = []

    def register_runner(self, runner_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register GitLab Runner

        Args:
            runner_config: Runner configuration

        Returns:
            Runner details
        """
        runner = {
            'id': len(self.runners) + 1,
            'token': f"runner-token-{len(self.runners) + 1}",
            'description': runner_config.get('description', 'Docker Runner'),
            'executor': runner_config.get('executor', 'docker'),
            'tags': runner_config.get('tags', ['docker', 'linux']),
            'run_untagged': runner_config.get('run_untagged', False),
            'locked': runner_config.get('locked', False),
            'access_level': runner_config.get('access_level', 'not_protected'),
            'maximum_timeout': runner_config.get('maximum_timeout', 3600),
            'status': 'online',
            'version': '16.5.0',
            'platform': 'linux/amd64',
            'architecture': 'amd64',
            'registered_at': datetime.now().isoformat(),
            'contacted_at': datetime.now().isoformat()
        }

        self.runners.append(runner)

        command = f"gitlab-runner register --non-interactive \\\n"
        command += f"  --url {self.gitlab_url} \\\n"
        command += f"  --registration-token {self.registration_token} \\\n"
        command += f"  --description \"{runner['description']}\" \\\n"
        command += f"  --executor {runner['executor']} \\\n"
        command += f"  --tag-list \"{','.join(runner['tags'])}\" \\\n"
        command += f"  {'--run-untagged' if runner['run_untagged'] else ''}"

        print(f"✓ Runner registered: {runner['description']}")
        print(f"  ID: {runner['id']}, Executor: {runner['executor']}")
        print(f"  Tags: {', '.join(runner['tags'])}")
        print(f"\n  Registration command:")
        print(f"  {command}")
        return runner

    def generate_runner_config(self, config: Dict[str, Any]) -> str:
        """
        Generate GitLab Runner config.toml

        Args:
            config: Runner configuration

        Returns:
            config.toml content
        """
        concurrent = config.get('concurrent', 4)
        check_interval = config.get('check_interval', 0)
        log_level = config.get('log_level', 'info')
        runners = config.get('runners', [])

        toml = f"""# GitLab Runner Configuration
# Generated: {datetime.now().isoformat()}

concurrent = {concurrent}
check_interval = {check_interval}
log_level = "{log_level}"

[session_server]
  session_timeout = 1800

"""

        for runner in runners:
            toml += f"""[[runners]]
  name = "{runner.get('name', 'docker-runner')}"
  url = "{self.gitlab_url}"
  token = "{runner.get('token', 'runner-token-xxx')}"
  executor = "{runner.get('executor', 'docker')}"
  [runners.docker]
    tls_verify = false
    image = "{runner.get('docker_image', 'alpine:latest')}"
    privileged = {str(runner.get('privileged', False)).lower()}
    disable_entrypoint_overwrite = false
    oom_kill_disable = false
    disable_cache = false
    volumes = ["/cache"]
    shm_size = 0
  [runners.cache]
    [runners.cache.s3]
    [runners.cache.gcs]
    [runners.cache.azure]

"""

        print(f"✓ Runner configuration generated")
        print(f"  Concurrent jobs: {concurrent}")
        print(f"  Runners configured: {len(runners)}")
        return toml

    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml for GitLab Runner"""

        compose = """# GitLab Runner Docker Compose
version: '3.8'

services:
  gitlab-runner:
    image: gitlab/gitlab-runner:latest
    container_name: gitlab-runner
    restart: always
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./config:/etc/gitlab-runner
      - ./cache:/cache
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
    networks:
      - gitlab-runner-network

  # Example: Multiple runners with different executors
  gitlab-runner-shell:
    image: gitlab/gitlab-runner:latest
    container_name: gitlab-runner-shell
    restart: always
    volumes:
      - ./config-shell:/etc/gitlab-runner
    networks:
      - gitlab-runner-network

networks:
  gitlab-runner-network:
    driver: bridge
"""

        print("✓ Docker Compose configuration generated for GitLab Runner")
        return compose

    def create_kubernetes_runner(self) -> str:
        """Generate Kubernetes deployment for GitLab Runner"""

        yaml = """# GitLab Runner Kubernetes Deployment
apiVersion: v1
kind: ConfigMap
metadata:
  name: gitlab-runner-config
  namespace: gitlab-runner
data:
  config.toml: |
    concurrent = 10
    check_interval = 3

    [[runners]]
      name = "kubernetes-runner"
      url = "https://gitlab.example.com"
      token = "RUNNER_TOKEN"
      executor = "kubernetes"
      [runners.kubernetes]
        namespace = "gitlab-runner"
        image = "alpine:latest"
        privileged = true

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gitlab-runner
  namespace: gitlab-runner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gitlab-runner
  template:
    metadata:
      labels:
        app: gitlab-runner
    spec:
      serviceAccountName: gitlab-runner
      containers:
      - name: gitlab-runner
        image: gitlab/gitlab-runner:latest
        volumeMounts:
        - name: config
          mountPath: /etc/gitlab-runner
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
      volumes:
      - name: config
        configMap:
          name: gitlab-runner-config

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: gitlab-runner
  namespace: gitlab-runner

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: gitlab-runner
  namespace: gitlab-runner
rules:
- apiGroups: [""]
  resources: ["pods", "pods/exec", "pods/log"]
  verbs: ["get", "list", "watch", "create", "delete", "update"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: gitlab-runner
  namespace: gitlab-runner
subjects:
- kind: ServiceAccount
  name: gitlab-runner
  namespace: gitlab-runner
roleRef:
  kind: Role
  name: gitlab-runner
  apiGroup: rbac.authorization.k8s.io
"""

        print("✓ Kubernetes deployment YAML generated for GitLab Runner")
        return yaml

    def get_runner_status(self, runner_id: int) -> Dict[str, Any]:
        """
        Get runner status

        Args:
            runner_id: Runner ID

        Returns:
            Runner status details
        """
        runner = next((r for r in self.runners if r['id'] == runner_id), None)
        if runner:
            status = {
                'id': runner['id'],
                'description': runner['description'],
                'status': 'online',
                'is_shared': False,
                'ip_address': '192.168.1.100',
                'contacted_at': datetime.now().isoformat(),
                'jobs_count': 125,
                'online': True,
                'projects': []
            }
            print(f"✓ Runner status: {status['description']}")
            print(f"  Status: {status['status']}, Jobs run: {status['jobs_count']}")
            return status
        return {'error': 'Runner not found'}

    def pause_runner(self, runner_id: int) -> Dict[str, Any]:
        """Pause runner"""
        runner = next((r for r in self.runners if r['id'] == runner_id), None)
        if runner:
            runner['status'] = 'paused'
            print(f"✓ Runner paused: #{runner_id}")
            return runner
        return {'error': 'Runner not found'}

    def resume_runner(self, runner_id: int) -> Dict[str, Any]:
        """Resume runner"""
        runner = next((r for r in self.runners if r['id'] == runner_id), None)
        if runner:
            runner['status'] = 'online'
            print(f"✓ Runner resumed: #{runner_id}")
            return runner
        return {'error': 'Runner not found'}

    def unregister_runner(self, runner_id: int) -> Dict[str, Any]:
        """Unregister runner"""
        runner = next((r for r in self.runners if r['id'] == runner_id), None)
        if runner:
            self.runners.remove(runner)
            print(f"✓ Runner unregistered: #{runner_id}")
            return {'status': 'unregistered'}
        return {'error': 'Runner not found'}

    def list_runners(self, filter_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List runners

        Args:
            filter_params: Optional filter parameters

        Returns:
            List of runners
        """
        filter_params = filter_params or {}
        status = filter_params.get('status', None)

        runners = self.runners
        if status:
            runners = [r for r in runners if r['status'] == status]

        print(f"✓ Listed {len(runners)} runners")
        return runners

    def get_runner_jobs(self, runner_id: int) -> List[Dict[str, Any]]:
        """
        Get jobs run by runner

        Args:
            runner_id: Runner ID

        Returns:
            List of jobs
        """
        jobs = [
            {
                'id': 1,
                'status': 'success',
                'stage': 'build',
                'name': 'build-job',
                'duration': 45,
                'finished_at': datetime.now().isoformat()
            },
            {
                'id': 2,
                'status': 'success',
                'stage': 'test',
                'name': 'test-job',
                'duration': 120,
                'finished_at': datetime.now().isoformat()
            }
        ]

        print(f"✓ Retrieved {len(jobs)} jobs for runner #{runner_id}")
        return jobs

    def get_manager_info(self) -> Dict[str, Any]:
        """Get runner manager information"""
        return {
            'gitlab_url': self.gitlab_url,
            'runners': len(self.runners),
            'online_runners': len([r for r in self.runners if r['status'] == 'online']),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate GitLab Runner management"""

    print("=" * 60)
    print("GitLab Runner Management Demo")
    print("=" * 60)

    # Initialize runner manager
    runner_mgr = GitLabRunnerManager(
        gitlab_url='https://gitlab.example.com',
        registration_token='GR1348941...'
    )

    print("\n1. Registering Docker runner...")
    docker_runner = runner_mgr.register_runner({
        'description': 'Docker Runner 1',
        'executor': 'docker',
        'tags': ['docker', 'linux', 'amd64'],
        'run_untagged': False,
        'maximum_timeout': 3600
    })

    print("\n2. Registering Shell runner...")
    shell_runner = runner_mgr.register_runner({
        'description': 'Shell Runner 1',
        'executor': 'shell',
        'tags': ['shell', 'linux'],
        'run_untagged': True
    })

    print("\n3. Registering Kubernetes runner...")
    k8s_runner = runner_mgr.register_runner({
        'description': 'Kubernetes Runner 1',
        'executor': 'kubernetes',
        'tags': ['kubernetes', 'k8s', 'cloud'],
        'run_untagged': False
    })

    print("\n4. Generating runner config.toml...")
    config_toml = runner_mgr.generate_runner_config({
        'concurrent': 4,
        'check_interval': 0,
        'log_level': 'info',
        'runners': [
            {
                'name': 'docker-runner-1',
                'token': 'runner-token-123',
                'executor': 'docker',
                'docker_image': 'alpine:latest',
                'privileged': False
            }
        ]
    })
    print(config_toml[:300] + "...\n")

    print("\n5. Generating Docker Compose for runners...")
    docker_compose = runner_mgr.generate_docker_compose()
    print(docker_compose[:300] + "...\n")

    print("\n6. Generating Kubernetes deployment...")
    k8s_yaml = runner_mgr.create_kubernetes_runner()
    print(k8s_yaml[:300] + "...\n")

    print("\n7. Checking runner status...")
    status = runner_mgr.get_runner_status(docker_runner['id'])

    print("\n8. Pausing runner...")
    runner_mgr.pause_runner(shell_runner['id'])

    print("\n9. Resuming runner...")
    runner_mgr.resume_runner(shell_runner['id'])

    print("\n10. Listing all runners...")
    all_runners = runner_mgr.list_runners()

    print("\n11. Listing online runners...")
    online_runners = runner_mgr.list_runners({'status': 'online'})

    print("\n12. Getting runner jobs...")
    jobs = runner_mgr.get_runner_jobs(docker_runner['id'])

    print("\n13. Runner manager summary:")
    info = runner_mgr.get_manager_info()
    print(f"  GitLab URL: {info['gitlab_url']}")
    print(f"  Total runners: {info['runners']}")
    print(f"  Online runners: {info['online_runners']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
