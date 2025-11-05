"""
GitLab CI/CD Pipeline Management
Complete GitLab CI/CD pipeline creation and management
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class GitLabCICD:
    """Comprehensive GitLab CI/CD pipeline management"""

    def __init__(self, gitlab_url: str, token: str):
        """
        Initialize GitLab CI/CD manager

        Args:
            gitlab_url: GitLab instance URL
            token: Personal access token
        """
        self.gitlab_url = gitlab_url
        self.token = token
        self.pipelines = []
        self.jobs = []

    def generate_gitlab_ci(self, pipeline_config: Dict[str, Any]) -> str:
        """
        Generate .gitlab-ci.yml file

        Args:
            pipeline_config: Pipeline configuration

        Returns:
            GitLab CI YAML content
        """
        stages = pipeline_config.get('stages', ['build', 'test', 'deploy'])
        image = pipeline_config.get('image', 'python:3.11')
        variables = pipeline_config.get('variables', {})

        yaml_content = f"""# GitLab CI/CD Pipeline
# Generated: {datetime.now().isoformat()}

image: {image}

stages:
"""
        for stage in stages:
            yaml_content += f"  - {stage}\n"

        if variables:
            yaml_content += "\nvariables:\n"
            for key, value in variables.items():
                yaml_content += f"  {key}: \"{value}\"\n"

        yaml_content += """
before_script:
  - echo "Starting CI/CD pipeline..."
  - pip install --upgrade pip

after_script:
  - echo "Pipeline completed"

"""

        # Build stage
        yaml_content += """build:
  stage: build
  script:
    - echo "Building application..."
    - pip install -r requirements.txt
    - python setup.py build
  artifacts:
    paths:
      - build/
    expire_in: 1 hour
  only:
    - main
    - develop

"""

        # Test stage
        yaml_content += """test:unit:
  stage: test
  script:
    - echo "Running unit tests..."
    - pip install -r requirements-test.txt
    - pytest tests/unit --cov --cov-report=xml
  coverage: '/TOTAL.*\\s+(\\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
  only:
    - main
    - develop

test:integration:
  stage: test
  script:
    - echo "Running integration tests..."
    - pytest tests/integration
  only:
    - main
    - develop

"""

        # Deploy stage
        yaml_content += """deploy:staging:
  stage: deploy
  script:
    - echo "Deploying to staging..."
    - ssh deploy@staging-server "cd /opt/app && git pull && systemctl restart app"
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

deploy:production:
  stage: deploy
  script:
    - echo "Deploying to production..."
    - ssh deploy@prod-server "cd /opt/app && git pull && systemctl restart app"
  environment:
    name: production
    url: https://example.com
  only:
    - main
  when: manual

"""

        print(f"✓ GitLab CI configuration generated")
        print(f"  Stages: {', '.join(stages)}")
        print(f"  Image: {image}")
        return yaml_content

    def create_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create CI/CD pipeline

        Args:
            pipeline_config: Pipeline configuration

        Returns:
            Pipeline details
        """
        pipeline = {
            'pipeline_id': len(self.pipelines) + 1,
            'ref': pipeline_config.get('ref', 'main'),
            'status': 'running',
            'stages': pipeline_config.get('stages', ['build', 'test', 'deploy']),
            'created_at': datetime.now().isoformat(),
            'duration': 0,
            'jobs': []
        }

        self.pipelines.append(pipeline)

        print(f"✓ Pipeline created: #{pipeline['pipeline_id']}")
        print(f"  Ref: {pipeline['ref']}, Stages: {len(pipeline['stages'])}")
        return pipeline

    def create_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create CI/CD job

        Args:
            job_config: Job configuration

        Returns:
            Job details
        """
        job = {
            'job_id': len(self.jobs) + 1,
            'name': job_config.get('name', 'build'),
            'stage': job_config.get('stage', 'build'),
            'status': job_config.get('status', 'success'),
            'duration': job_config.get('duration', 45),
            'runner': job_config.get('runner', 'docker-runner-1'),
            'created_at': datetime.now().isoformat(),
            'artifacts': job_config.get('artifacts', []),
            'coverage': job_config.get('coverage', None)
        }

        self.jobs.append(job)

        print(f"✓ Job created: {job['name']} (Stage: {job['stage']})")
        print(f"  Status: {job['status']}, Duration: {job['duration']}s")
        return job

    def generate_docker_ci(self) -> str:
        """Generate Docker-based CI/CD pipeline"""

        yaml_content = """# Docker CI/CD Pipeline

image: docker:latest

services:
  - docker:dind

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA

stages:
  - build
  - test
  - push
  - deploy

before_script:
  - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY

build:
  stage: build
  script:
    - docker build -t $IMAGE_TAG .
    - docker tag $IMAGE_TAG $CI_REGISTRY_IMAGE:latest
  only:
    - main
    - develop

test:
  stage: test
  script:
    - docker run --rm $IMAGE_TAG pytest tests/
  only:
    - main
    - develop

push:
  stage: push
  script:
    - docker push $IMAGE_TAG
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main

deploy:kubernetes:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context production
    - kubectl set image deployment/app app=$IMAGE_TAG
    - kubectl rollout status deployment/app
  only:
    - main
  when: manual
"""

        print("✓ Docker CI/CD configuration generated")
        return yaml_content

    def generate_nodejs_ci(self) -> str:
        """Generate Node.js CI/CD pipeline"""

        yaml_content = """# Node.js CI/CD Pipeline

image: node:18

stages:
  - install
  - test
  - build
  - deploy

cache:
  paths:
    - node_modules/

install:
  stage: install
  script:
    - npm ci
  artifacts:
    paths:
      - node_modules/
    expire_in: 1 hour

test:
  stage: test
  script:
    - npm run test:coverage
  coverage: '/Statements\\s*:\\s*(\\d+\\.\\d+)%/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

lint:
  stage: test
  script:
    - npm run lint

build:
  stage: build
  script:
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 week

deploy:staging:
  stage: deploy
  script:
    - npm run deploy:staging
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

deploy:production:
  stage: deploy
  script:
    - npm run deploy:production
  environment:
    name: production
    url: https://example.com
  only:
    - main
  when: manual
"""

        print("✓ Node.js CI/CD configuration generated")
        return yaml_content

    def generate_kubernetes_deployment(self) -> str:
        """Generate Kubernetes deployment CI/CD"""

        yaml_content = """# Kubernetes Deployment Pipeline

image: bitnami/kubectl:latest

stages:
  - build
  - deploy

variables:
  KUBE_NAMESPACE: production
  APP_NAME: myapp

build:image:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA

deploy:k8s:
  stage: deploy
  script:
    - kubectl config use-context $KUBE_CONTEXT
    - kubectl config set-context --current --namespace=$KUBE_NAMESPACE
    - |
      cat <<EOF | kubectl apply -f -
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: $APP_NAME
      spec:
        replicas: 3
        selector:
          matchLabels:
            app: $APP_NAME
        template:
          metadata:
            labels:
              app: $APP_NAME
          spec:
            containers:
            - name: $APP_NAME
              image: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
              ports:
              - containerPort: 8080
      EOF
    - kubectl rollout status deployment/$APP_NAME
  environment:
    name: production
    kubernetes:
      namespace: $KUBE_NAMESPACE
  only:
    - main
"""

        print("✓ Kubernetes deployment CI/CD configuration generated")
        return yaml_content

    def get_pipeline_status(self, pipeline_id: int) -> Dict[str, Any]:
        """Get pipeline status"""
        pipeline = next((p for p in self.pipelines if p['pipeline_id'] == pipeline_id), None)
        if pipeline:
            pipeline['status'] = 'success'
            pipeline['duration'] = 180
            return pipeline
        return {'error': 'Pipeline not found'}

    def retry_job(self, job_id: int) -> Dict[str, Any]:
        """Retry failed job"""
        job = next((j for j in self.jobs if j['job_id'] == job_id), None)
        if job:
            job['status'] = 'pending'
            print(f"✓ Job retry initiated: #{job_id}")
            return job
        return {'error': 'Job not found'}

    def get_cicd_info(self) -> Dict[str, Any]:
        """Get CI/CD manager information"""
        return {
            'gitlab_url': self.gitlab_url,
            'pipelines': len(self.pipelines),
            'jobs': len(self.jobs),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate GitLab CI/CD management"""

    print("=" * 60)
    print("GitLab CI/CD Pipeline Management Demo")
    print("=" * 60)

    # Initialize CI/CD manager
    cicd = GitLabCICD(
        gitlab_url='https://gitlab.example.com',
        token='glpat-xxxxxxxxxxxx'
    )

    print("\n1. Generating basic GitLab CI configuration...")
    gitlab_ci = cicd.generate_gitlab_ci({
        'stages': ['build', 'test', 'deploy'],
        'image': 'python:3.11',
        'variables': {
            'PIP_CACHE_DIR': '$CI_PROJECT_DIR/.cache/pip',
            'POSTGRES_DB': 'test_db'
        }
    })
    print(gitlab_ci[:400] + "...\n")

    print("\n2. Generating Docker-based CI/CD...")
    docker_ci = cicd.generate_docker_ci()
    print(docker_ci[:300] + "...\n")

    print("\n3. Generating Node.js CI/CD...")
    nodejs_ci = cicd.generate_nodejs_ci()
    print(nodejs_ci[:300] + "...\n")

    print("\n4. Generating Kubernetes deployment CI/CD...")
    k8s_ci = cicd.generate_kubernetes_deployment()
    print(k8s_ci[:300] + "...\n")

    print("\n5. Creating pipeline...")
    pipeline = cicd.create_pipeline({
        'ref': 'main',
        'stages': ['build', 'test', 'deploy']
    })

    print("\n6. Creating jobs...")
    build_job = cicd.create_job({
        'name': 'build',
        'stage': 'build',
        'status': 'success',
        'duration': 45
    })

    test_job = cicd.create_job({
        'name': 'test:unit',
        'stage': 'test',
        'status': 'success',
        'duration': 120,
        'coverage': 85.5
    })

    deploy_job = cicd.create_job({
        'name': 'deploy:production',
        'stage': 'deploy',
        'status': 'success',
        'duration': 90
    })

    print("\n7. Checking pipeline status...")
    status = cicd.get_pipeline_status(pipeline['pipeline_id'])
    print(f"  Status: {status['status']}, Duration: {status['duration']}s")

    print("\n8. CI/CD summary:")
    info = cicd.get_cicd_info()
    print(f"  GitLab URL: {info['gitlab_url']}")
    print(f"  Pipelines: {info['pipelines']}")
    print(f"  Jobs: {info['jobs']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
