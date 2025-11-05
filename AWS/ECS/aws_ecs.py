"""
AWS ECS (Elastic Container Service)
=====================================

Container orchestration with Fargate and EC2 launch types.

Author: Brill Consulting
"""

import boto3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ECSManager:
    """
    Advanced AWS ECS Management System

    Provides comprehensive ECS operations including:
    - Cluster management
    - Task definition registration
    - Service deployment and scaling
    - Task execution and monitoring
    - Fargate and EC2 launch types
    - Load balancer integration
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize ECS Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.ecs_client = session.client('ecs', region_name=region)
            self.region = region
            logger.info(f"ECS Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing ECS Manager: {e}")
            raise

    # ==================== Cluster Management ====================

    def create_cluster(
        self,
        cluster_name: str,
        capacity_providers: Optional[List[str]] = None,
        tags: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create ECS cluster.

        Args:
            cluster_name: Cluster name
            capacity_providers: ['FARGATE', 'FARGATE_SPOT', 'EC2']
            tags: Resource tags
        """
        try:
            logger.info(f"Creating cluster: {cluster_name}")

            params = {'clusterName': cluster_name}

            if capacity_providers:
                params['capacityProviders'] = capacity_providers

            if tags:
                params['tags'] = tags

            response = self.ecs_client.create_cluster(**params)

            cluster = response['cluster']
            logger.info(f"✓ Cluster created: {cluster_name}")

            return {
                'cluster_name': cluster['clusterName'],
                'cluster_arn': cluster['clusterArn'],
                'status': cluster['status']
            }

        except ClientError as e:
            logger.error(f"Error creating cluster: {e}")
            raise

    def describe_clusters(self, cluster_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Describe clusters."""
        try:
            params = {}
            if cluster_names:
                params['clusters'] = cluster_names

            response = self.ecs_client.describe_clusters(**params)

            clusters = [
                {
                    'cluster_name': c['clusterName'],
                    'cluster_arn': c['clusterArn'],
                    'status': c['status'],
                    'running_tasks': c['runningTasksCount'],
                    'pending_tasks': c['pendingTasksCount'],
                    'active_services': c['activeServicesCount']
                }
                for c in response.get('clusters', [])
            ]

            logger.info(f"Found {len(clusters)} cluster(s)")
            return clusters

        except ClientError as e:
            logger.error(f"Error describing clusters: {e}")
            raise

    def delete_cluster(self, cluster_name: str) -> None:
        """Delete cluster."""
        try:
            self.ecs_client.delete_cluster(cluster=cluster_name)
            logger.info(f"✓ Cluster deleted: {cluster_name}")

        except ClientError as e:
            logger.error(f"Error deleting cluster: {e}")
            raise

    # ==================== Task Definitions ====================

    def register_task_definition(
        self,
        family: str,
        container_definitions: List[Dict[str, Any]],
        task_role_arn: Optional[str] = None,
        execution_role_arn: Optional[str] = None,
        network_mode: str = "awsvpc",
        requires_compatibilities: Optional[List[str]] = None,
        cpu: Optional[str] = None,
        memory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register task definition.

        Args:
            family: Task definition family name
            container_definitions: Container configurations
            task_role_arn: IAM role for task
            execution_role_arn: IAM role for ECS agent
            network_mode: 'awsvpc', 'bridge', 'host', 'none'
            requires_compatibilities: ['FARGATE', 'EC2']
            cpu: Task CPU (256, 512, 1024, etc.)
            memory: Task memory (512, 1024, 2048, etc.)
        """
        try:
            logger.info(f"Registering task definition: {family}")

            params = {
                'family': family,
                'containerDefinitions': container_definitions,
                'networkMode': network_mode
            }

            if task_role_arn:
                params['taskRoleArn'] = task_role_arn
            if execution_role_arn:
                params['executionRoleArn'] = execution_role_arn
            if requires_compatibilities:
                params['requiresCompatibilities'] = requires_compatibilities
            if cpu:
                params['cpu'] = cpu
            if memory:
                params['memory'] = memory

            response = self.ecs_client.register_task_definition(**params)

            task_def = response['taskDefinition']
            logger.info(f"✓ Task definition registered: {family}:{task_def['revision']}")

            return {
                'family': task_def['family'],
                'revision': task_def['revision'],
                'task_definition_arn': task_def['taskDefinitionArn'],
                'status': task_def['status']
            }

        except ClientError as e:
            logger.error(f"Error registering task definition: {e}")
            raise

    def describe_task_definition(self, task_definition: str) -> Dict[str, Any]:
        """Describe task definition."""
        try:
            response = self.ecs_client.describe_task_definition(
                taskDefinition=task_definition
            )

            task_def = response['taskDefinition']
            return {
                'family': task_def['family'],
                'revision': task_def['revision'],
                'task_definition_arn': task_def['taskDefinitionArn'],
                'status': task_def['status'],
                'network_mode': task_def['networkMode'],
                'cpu': task_def.get('cpu'),
                'memory': task_def.get('memory'),
                'container_count': len(task_def['containerDefinitions'])
            }

        except ClientError as e:
            logger.error(f"Error describing task definition: {e}")
            raise

    def deregister_task_definition(self, task_definition: str) -> None:
        """Deregister task definition."""
        try:
            self.ecs_client.deregister_task_definition(
                taskDefinition=task_definition
            )
            logger.info(f"✓ Task definition deregistered: {task_definition}")

        except ClientError as e:
            logger.error(f"Error deregistering task definition: {e}")
            raise

    # ==================== Services ====================

    def create_service(
        self,
        cluster: str,
        service_name: str,
        task_definition: str,
        desired_count: int = 1,
        launch_type: str = "FARGATE",
        network_configuration: Optional[Dict[str, Any]] = None,
        load_balancers: Optional[List[Dict[str, Any]]] = None,
        health_check_grace_period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create ECS service.

        Args:
            cluster: Cluster name or ARN
            service_name: Service name
            task_definition: Task definition family:revision
            desired_count: Number of tasks to run
            launch_type: 'FARGATE' or 'EC2'
            network_configuration: VPC configuration for awsvpc mode
            load_balancers: Load balancer configurations
            health_check_grace_period: Seconds before health checks start
        """
        try:
            logger.info(f"Creating service: {service_name}")

            params = {
                'cluster': cluster,
                'serviceName': service_name,
                'taskDefinition': task_definition,
                'desiredCount': desired_count,
                'launchType': launch_type
            }

            if network_configuration:
                params['networkConfiguration'] = network_configuration

            if load_balancers:
                params['loadBalancers'] = load_balancers

            if health_check_grace_period:
                params['healthCheckGracePeriodSeconds'] = health_check_grace_period

            response = self.ecs_client.create_service(**params)

            service = response['service']
            logger.info(f"✓ Service created: {service_name}")

            return {
                'service_name': service['serviceName'],
                'service_arn': service['serviceArn'],
                'status': service['status'],
                'desired_count': service['desiredCount']
            }

        except ClientError as e:
            logger.error(f"Error creating service: {e}")
            raise

    def update_service(
        self,
        cluster: str,
        service_name: str,
        desired_count: Optional[int] = None,
        task_definition: Optional[str] = None,
        force_new_deployment: bool = False
    ) -> Dict[str, Any]:
        """Update service."""
        try:
            logger.info(f"Updating service: {service_name}")

            params = {
                'cluster': cluster,
                'service': service_name,
                'forceNewDeployment': force_new_deployment
            }

            if desired_count is not None:
                params['desiredCount'] = desired_count

            if task_definition:
                params['taskDefinition'] = task_definition

            response = self.ecs_client.update_service(**params)

            service = response['service']
            logger.info(f"✓ Service updated: {service_name}")

            return {
                'service_name': service['serviceName'],
                'status': service['status'],
                'desired_count': service['desiredCount'],
                'running_count': service['runningCount']
            }

        except ClientError as e:
            logger.error(f"Error updating service: {e}")
            raise

    def describe_services(
        self,
        cluster: str,
        services: List[str]
    ) -> List[Dict[str, Any]]:
        """Describe services."""
        try:
            response = self.ecs_client.describe_services(
                cluster=cluster,
                services=services
            )

            services_info = [
                {
                    'service_name': s['serviceName'],
                    'service_arn': s['serviceArn'],
                    'status': s['status'],
                    'desired_count': s['desiredCount'],
                    'running_count': s['runningCount'],
                    'pending_count': s['pendingCount'],
                    'launch_type': s['launchType'],
                    'created_at': s['createdAt'].isoformat()
                }
                for s in response.get('services', [])
            ]

            logger.info(f"Described {len(services_info)} service(s)")
            return services_info

        except ClientError as e:
            logger.error(f"Error describing services: {e}")
            raise

    def delete_service(
        self,
        cluster: str,
        service_name: str,
        force: bool = False
    ) -> None:
        """Delete service."""
        try:
            self.ecs_client.delete_service(
                cluster=cluster,
                service=service_name,
                force=force
            )
            logger.info(f"✓ Service deleted: {service_name}")

        except ClientError as e:
            logger.error(f"Error deleting service: {e}")
            raise

    # ==================== Tasks ====================

    def run_task(
        self,
        cluster: str,
        task_definition: str,
        launch_type: str = "FARGATE",
        network_configuration: Optional[Dict[str, Any]] = None,
        count: int = 1
    ) -> List[Dict[str, Any]]:
        """Run standalone task."""
        try:
            logger.info(f"Running task: {task_definition}")

            params = {
                'cluster': cluster,
                'taskDefinition': task_definition,
                'launchType': launch_type,
                'count': count
            }

            if network_configuration:
                params['networkConfiguration'] = network_configuration

            response = self.ecs_client.run_task(**params)

            tasks = [
                {
                    'task_arn': t['taskArn'],
                    'last_status': t['lastStatus'],
                    'desired_status': t['desiredStatus']
                }
                for t in response.get('tasks', [])
            ]

            logger.info(f"✓ Started {len(tasks)} task(s)")
            return tasks

        except ClientError as e:
            logger.error(f"Error running task: {e}")
            raise

    def stop_task(self, cluster: str, task_arn: str, reason: str = "") -> None:
        """Stop task."""
        try:
            self.ecs_client.stop_task(
                cluster=cluster,
                task=task_arn,
                reason=reason
            )
            logger.info(f"✓ Task stopped: {task_arn}")

        except ClientError as e:
            logger.error(f"Error stopping task: {e}")
            raise

    def list_tasks(
        self,
        cluster: str,
        service_name: Optional[str] = None,
        desired_status: str = "RUNNING"
    ) -> List[str]:
        """List tasks."""
        try:
            params = {
                'cluster': cluster,
                'desiredStatus': desired_status
            }

            if service_name:
                params['serviceName'] = service_name

            response = self.ecs_client.list_tasks(**params)

            task_arns = response.get('taskArns', [])
            logger.info(f"Found {len(task_arns)} task(s)")

            return task_arns

        except ClientError as e:
            logger.error(f"Error listing tasks: {e}")
            raise

    # ==================== Monitoring ====================

    def get_summary(self, cluster: Optional[str] = None) -> Dict[str, Any]:
        """Get ECS summary."""
        try:
            if cluster:
                clusters = self.describe_clusters([cluster])
            else:
                response = self.ecs_client.list_clusters()
                cluster_arns = response.get('clusterArns', [])
                clusters = self.describe_clusters(cluster_arns) if cluster_arns else []

            return {
                'region': self.region,
                'cluster_count': len(clusters),
                'clusters': clusters,
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of ECS Manager capabilities."""
    print("AWS ECS Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    print("\n1️⃣  Cluster Management:")
    print("""
    ecs = ECSManager(region='us-east-1')

    # Create cluster
    cluster = ecs.create_cluster(
        cluster_name='prod-cluster',
        capacity_providers=['FARGATE', 'FARGATE_SPOT']
    )

    # Describe clusters
    clusters = ecs.describe_clusters(['prod-cluster'])
    """)

    print("\n2️⃣  Task Definitions:")
    print("""
    # Register task definition
    task_def = ecs.register_task_definition(
        family='web-app',
        container_definitions=[{
            'name': 'nginx',
            'image': 'nginx:latest',
            'memory': 512,
            'portMappings': [{
                'containerPort': 80,
                'protocol': 'tcp'
            }]
        }],
        requires_compatibilities=['FARGATE'],
        network_mode='awsvpc',
        cpu='256',
        memory='512'
    )
    """)

    print("\n3️⃣  Services:")
    print("""
    # Create service
    service = ecs.create_service(
        cluster='prod-cluster',
        service_name='web-service',
        task_definition='web-app:1',
        desired_count=3,
        launch_type='FARGATE',
        network_configuration={
            'awsvpcConfiguration': {
                'subnets': ['subnet-12345'],
                'securityGroups': ['sg-12345'],
                'assignPublicIp': 'ENABLED'
            }
        }
    )

    # Update service (scale to 5 tasks)
    ecs.update_service(
        cluster='prod-cluster',
        service_name='web-service',
        desired_count=5
    )
    """)

    print("\n4️⃣  Task Management:")
    print("""
    # Run standalone task
    tasks = ecs.run_task(
        cluster='prod-cluster',
        task_definition='web-app:1',
        launch_type='FARGATE',
        count=2
    )

    # List running tasks
    task_arns = ecs.list_tasks('prod-cluster', desired_status='RUNNING')

    # Stop task
    ecs.stop_task('prod-cluster', task_arns[0], reason='Manual stop')
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
