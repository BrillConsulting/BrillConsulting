"""
Cloud Run Serverless Container Platform
Author: BrillConsulting
Description: Advanced Cloud Run with deployment, scaling, and traffic management
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import time


class CloudRunService:
    """Cloud Run service deployment and management"""

    def __init__(self, project_id: str, region: str = 'us-central1'):
        """
        Initialize Cloud Run service manager

        Args:
            project_id: GCP project ID
            region: Cloud Run region
        """
        self.project_id = project_id
        self.region = region
        self.services = []

    def deploy_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy Cloud Run service

        Args:
            config: Service configuration

        Returns:
            Deployment result
        """
        print(f"\n{'='*60}")
        print("Deploying Cloud Run Service")
        print(f"{'='*60}")

        service_name = config.get('service_name', 'my-service')
        image = config.get('image', 'gcr.io/my-project/my-image:latest')
        port = config.get('port', 8080)
        memory = config.get('memory', '512Mi')
        cpu = config.get('cpu', '1')

        code = f"""
from google.cloud import run_v2

client = run_v2.ServicesClient()

# Define service
service = run_v2.Service()
service.name = f"projects/{self.project_id}/locations/{self.region}/services/{service_name}"

# Container configuration
container = run_v2.Container()
container.image = "{image}"
container.ports = [run_v2.ContainerPort(container_port={port})]

# Resource limits
resources = run_v2.ResourceRequirements()
resources.limits = {{
    "memory": "{memory}",
    "cpu": "{cpu}"
}}
container.resources = resources

# Template configuration
template = run_v2.RevisionTemplate()
template.containers = [container]

service.template = template

# Deploy service
request = run_v2.CreateServiceRequest(
    parent=f"projects/{self.project_id}/locations/{self.region}",
    service=service,
    service_id="{service_name}"
)

operation = client.create_service(request=request)
result = operation.result()

print(f"Service deployed: {{result.uri}}")
"""

        result = {
            'service_name': service_name,
            'image': image,
            'region': self.region,
            'port': port,
            'memory': memory,
            'cpu': cpu,
            'url': f"https://{service_name}-{self.region}.run.app",
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.services.append(result)

        print(f"✓ Service deployed: {service_name}")
        print(f"  Image: {image}")
        print(f"  URL: {result['url']}")
        print(f"  Resources: {memory} memory, {cpu} CPU")
        print(f"{'='*60}")

        return result

    def deploy_with_env_vars(self, service_name: str, env_vars: Dict[str, str]) -> str:
        """
        Deploy with environment variables

        Args:
            service_name: Service name
            env_vars: Environment variables

        Returns:
            Deployment code
        """
        env_list = [f"{{'name': '{k}', 'value': '{v}'}}" for k, v in env_vars.items()]

        code = f"""
from google.cloud import run_v2

client = run_v2.ServicesClient()

# Configure environment variables
container = run_v2.Container()
container.image = "gcr.io/{self.project_id}/{service_name}:latest"
container.env = [
    {',\n    '.join(env_list)}
]

# Deploy service
# ... (rest of deployment code)
"""

        print(f"\n✓ Environment variables configured: {list(env_vars.keys())}")
        return code


class CloudRunScaling:
    """Cloud Run autoscaling configuration"""

    def __init__(self, project_id: str):
        """Initialize scaling manager"""
        self.project_id = project_id

    def configure_autoscaling(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure autoscaling parameters

        Args:
            config: Scaling configuration

        Returns:
            Scaling details
        """
        print(f"\n{'='*60}")
        print("Configuring Autoscaling")
        print(f"{'='*60}")

        service_name = config.get('service_name')
        min_instances = config.get('min_instances', 0)
        max_instances = config.get('max_instances', 100)
        concurrency = config.get('concurrency', 80)

        code = f"""
from google.cloud import run_v2

client = run_v2.ServicesClient()

# Configure scaling
template = run_v2.RevisionTemplate()

# Scaling settings
template.scaling = run_v2.RevisionScaling(
    min_instance_count={min_instances},
    max_instance_count={max_instances}
)

# Concurrency (requests per container)
template.max_instance_request_concurrency = {concurrency}

# Update service
service = run_v2.Service()
service.name = f"projects/{self.project_id}/locations/us-central1/services/{service_name}"
service.template = template

request = run_v2.UpdateServiceRequest(service=service)
operation = client.update_service(request=request)
result = operation.result()

print(f"Autoscaling configured: {{result.name}}")
"""

        result = {
            'service_name': service_name,
            'min_instances': min_instances,
            'max_instances': max_instances,
            'concurrency': concurrency,
            'code': code
        }

        print(f"✓ Autoscaling configured: {service_name}")
        print(f"  Min instances: {min_instances}")
        print(f"  Max instances: {max_instances}")
        print(f"  Concurrency: {concurrency} req/container")
        print(f"{'='*60}")

        return result

    def configure_cpu_throttling(self, service_name: str, always_allocated: bool = False) -> str:
        """
        Configure CPU allocation

        Args:
            service_name: Service name
            always_allocated: CPU always allocated vs only during request

        Returns:
            CPU configuration code
        """
        cpu_mode = "cpu-always" if always_allocated else "cpu-throttling"

        code = f"""
from google.cloud import run_v2

client = run_v2.ServicesClient()

# Configure CPU allocation
template = run_v2.RevisionTemplate()

# CPU always allocated (for background tasks)
template.execution_environment = run_v2.ExecutionEnvironment.EXECUTION_ENVIRONMENT_GEN2

# Annotation for CPU allocation
template.annotations = {{
    "run.googleapis.com/cpu-throttling": "{'false' if always_allocated else 'true'}"
}}

# Update service
# ... (rest of update code)
"""

        print(f"\n✓ CPU allocation: {'Always allocated' if always_allocated else 'Throttled'}")
        return code


class CloudRunTraffic:
    """Cloud Run traffic splitting and management"""

    def __init__(self, project_id: str, region: str = 'us-central1'):
        """Initialize traffic manager"""
        self.project_id = project_id
        self.region = region

    def split_traffic(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure traffic splitting between revisions

        Args:
            config: Traffic configuration

        Returns:
            Traffic split details
        """
        print(f"\n{'='*60}")
        print("Configuring Traffic Split")
        print(f"{'='*60}")

        service_name = config.get('service_name')
        splits = config.get('splits', [])

        code = f"""
from google.cloud import run_v2

client = run_v2.ServicesClient()

# Configure traffic split
service = run_v2.Service()
service.name = f"projects/{self.project_id}/locations/{self.region}/services/{service_name}"

# Traffic targets
traffic = []
"""

        for split in splits:
            revision = split.get('revision', 'latest')
            percent = split.get('percent', 100)
            code += f"""
traffic.append(run_v2.TrafficTarget(
    revision="{revision}",
    percent={percent}
))
"""

        code += """
service.traffic = traffic

# Update service
request = run_v2.UpdateServiceRequest(service=service)
operation = client.update_service(request=request)
result = operation.result()

print(f"Traffic split configured: {result.name}")
"""

        result = {
            'service_name': service_name,
            'splits': splits,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Traffic split configured: {service_name}")
        for split in splits:
            print(f"  {split['revision']}: {split['percent']}%")
        print(f"{'='*60}")

        return result

    def canary_deployment(self, service_name: str, new_revision: str, canary_percent: int = 10) -> str:
        """
        Configure canary deployment

        Args:
            service_name: Service name
            new_revision: New revision name
            canary_percent: Percent of traffic to new revision

        Returns:
            Canary deployment code
        """
        code = f"""
from google.cloud import run_v2

client = run_v2.ServicesClient()

# Canary deployment: {canary_percent}% to new revision
service = run_v2.Service()
service.name = f"projects/{self.project_id}/locations/{self.region}/services/{service_name}"

service.traffic = [
    run_v2.TrafficTarget(revision="latest", percent={100 - canary_percent}),
    run_v2.TrafficTarget(revision="{new_revision}", percent={canary_percent})
]

request = run_v2.UpdateServiceRequest(service=service)
operation = client.update_service(request=request)
result = operation.result()

print(f"Canary deployment: {canary_percent}% to {new_revision}")
"""

        print(f"\n✓ Canary deployment: {canary_percent}% to {new_revision}")
        return code

    def rollback_traffic(self, service_name: str, target_revision: str) -> str:
        """
        Rollback traffic to specific revision

        Args:
            service_name: Service name
            target_revision: Revision to rollback to

        Returns:
            Rollback code
        """
        code = f"""
from google.cloud import run_v2

client = run_v2.ServicesClient()

# Rollback to previous revision
service = run_v2.Service()
service.name = f"projects/{self.project_id}/locations/{self.region}/services/{service_name}"

service.traffic = [
    run_v2.TrafficTarget(revision="{target_revision}", percent=100)
]

request = run_v2.UpdateServiceRequest(service=service)
operation = client.update_service(request=request)
result = operation.result()

print(f"Traffic rolled back to: {target_revision}")
"""

        print(f"\n✓ Rollback configured to: {target_revision}")
        return code


class CloudRunIAM:
    """Cloud Run IAM and permissions management"""

    def __init__(self, project_id: str):
        """Initialize IAM manager"""
        self.project_id = project_id

    def make_service_public(self, service_name: str, region: str) -> str:
        """
        Make service publicly accessible

        Args:
            service_name: Service name
            region: Service region

        Returns:
            IAM policy code
        """
        code = f"""
from google.cloud import run_v2
from google.iam.v1 import iam_policy_pb2

client = run_v2.ServicesClient()

# Make service public
service_name = f"projects/{self.project_id}/locations/{region}/services/{service_name}"

policy = iam_policy_pb2.Policy()
binding = iam_policy_pb2.Binding(
    role="roles/run.invoker",
    members=["allUsers"]
)
policy.bindings.append(binding)

request = iam_policy_pb2.SetIamPolicyRequest(
    resource=service_name,
    policy=policy
)

client.set_iam_policy(request=request)
print("Service is now publicly accessible")
"""

        print(f"\n✓ Public access configured for: {service_name}")
        return code

    def grant_service_account_access(self, service_name: str, service_account: str) -> str:
        """
        Grant service account access

        Args:
            service_name: Service name
            service_account: Service account email

        Returns:
            IAM policy code
        """
        code = f"""
from google.cloud import run_v2
from google.iam.v1 import iam_policy_pb2

client = run_v2.ServicesClient()

# Grant service account access
service_name = f"projects/{self.project_id}/locations/us-central1/services/{service_name}"

policy = iam_policy_pb2.Policy()
binding = iam_policy_pb2.Binding(
    role="roles/run.invoker",
    members=[f"serviceAccount:{service_account}"]
)
policy.bindings.append(binding)

request = iam_policy_pb2.SetIamPolicyRequest(
    resource=service_name,
    policy=policy
)

client.set_iam_policy(request=request)
print(f"Access granted to: {service_account}")
"""

        print(f"\n✓ Access granted to: {service_account}")
        return code


class CloudRunSecrets:
    """Cloud Run secrets management"""

    def __init__(self, project_id: str):
        """Initialize secrets manager"""
        self.project_id = project_id

    def mount_secret(self, service_name: str, secret_name: str, mount_path: str) -> str:
        """
        Mount secret as volume

        Args:
            service_name: Service name
            secret_name: Secret Manager secret name
            mount_path: Container mount path

        Returns:
            Secret mounting code
        """
        code = f"""
from google.cloud import run_v2

client = run_v2.ServicesClient()

# Mount secret as volume
container = run_v2.Container()
container.image = "gcr.io/{self.project_id}/{service_name}:latest"

# Secret volume
volume = run_v2.Volume()
volume.name = "secret-volume"
volume.secret = run_v2.SecretVolumeSource(
    secret=f"projects/{self.project_id}/secrets/{secret_name}",
    items=[run_v2.VersionToPath(version="latest", path="secret.txt")]
)

# Volume mount
volume_mount = run_v2.VolumeMount()
volume_mount.name = "secret-volume"
volume_mount.mount_path = "{mount_path}"

container.volume_mounts = [volume_mount]

# Update service with secret
# ... (rest of deployment code)
"""

        print(f"\n✓ Secret mounted: {secret_name} at {mount_path}")
        return code

    def use_secret_as_env(self, service_name: str, secret_name: str, env_var: str) -> str:
        """
        Use secret as environment variable

        Args:
            service_name: Service name
            secret_name: Secret name
            env_var: Environment variable name

        Returns:
            Secret env code
        """
        code = f"""
from google.cloud import run_v2

client = run_v2.ServicesClient()

# Use secret as environment variable
container = run_v2.Container()
container.image = "gcr.io/{self.project_id}/{service_name}:latest"

# Secret as env var
env_var = run_v2.EnvVar()
env_var.name = "{env_var}"
env_var.value_source = run_v2.EnvVarSource(
    secret_key_ref=run_v2.SecretKeySelector(
        secret=f"projects/{self.project_id}/secrets/{secret_name}",
        version="latest"
    )
)

container.env = [env_var]

# Update service
# ... (rest of deployment code)
"""

        print(f"\n✓ Secret as env var: {secret_name} → {env_var}")
        return code


class CloudRunManager:
    """Comprehensive Cloud Run management"""

    def __init__(self, project_id: str = 'my-project', region: str = 'us-central1'):
        """Initialize Cloud Run manager"""
        self.project_id = project_id
        self.region = region
        self.services = []

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'region': self.region,
            'services': len(self.services),
            'features': [
                'serverless_containers',
                'autoscaling',
                'traffic_splitting',
                'canary_deployments',
                'iam_permissions',
                'secrets_management'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Cloud Run capabilities"""
    print("=" * 60)
    print("Cloud Run Serverless Platform Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'
    region = 'us-central1'

    # Deploy service
    service_mgr = CloudRunService(project_id, region)

    deployment = service_mgr.deploy_service({
        'service_name': 'my-api',
        'image': 'gcr.io/my-project/my-api:v1.0.0',
        'port': 8080,
        'memory': '1Gi',
        'cpu': '2'
    })

    env_code = service_mgr.deploy_with_env_vars('my-api', {
        'DATABASE_URL': 'postgresql://...',
        'API_KEY': 'secret-key',
        'ENVIRONMENT': 'production'
    })

    # Configure scaling
    scaling_mgr = CloudRunScaling(project_id)

    scaling = scaling_mgr.configure_autoscaling({
        'service_name': 'my-api',
        'min_instances': 1,
        'max_instances': 100,
        'concurrency': 80
    })

    cpu_code = scaling_mgr.configure_cpu_throttling('my-api', always_allocated=True)

    # Traffic management
    traffic_mgr = CloudRunTraffic(project_id, region)

    traffic_split = traffic_mgr.split_traffic({
        'service_name': 'my-api',
        'splits': [
            {'revision': 'my-api-v1', 'percent': 90},
            {'revision': 'my-api-v2', 'percent': 10}
        ]
    })

    canary_code = traffic_mgr.canary_deployment('my-api', 'my-api-v2', 10)
    rollback_code = traffic_mgr.rollback_traffic('my-api', 'my-api-v1')

    # IAM permissions
    iam_mgr = CloudRunIAM(project_id)
    public_code = iam_mgr.make_service_public('my-api', region)
    sa_code = iam_mgr.grant_service_account_access('my-api', 'my-sa@project.iam.gserviceaccount.com')

    # Secrets
    secrets_mgr = CloudRunSecrets(project_id)
    mount_code = secrets_mgr.mount_secret('my-api', 'db-password', '/secrets')
    env_secret_code = secrets_mgr.use_secret_as_env('my-api', 'api-key', 'API_KEY')

    # Manager info
    mgr = CloudRunManager(project_id, region)
    mgr.services.append(deployment)

    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Cloud Run Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Region: {info['region']}")
    print(f"Services: {info['services']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    demo()
