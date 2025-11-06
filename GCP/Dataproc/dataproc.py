"""
Dataproc - Managed Spark and Hadoop Clusters
Author: BrillConsulting
Description: Comprehensive cluster management and job execution for big data processing
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class DataprocCluster:
    """Create and manage Dataproc clusters"""

    def __init__(self, project_id: str, region: str = 'us-central1'):
        """
        Initialize cluster manager

        Args:
            project_id: GCP project ID
            region: Cluster region
        """
        self.project_id = project_id
        self.region = region
        self.clusters = []

    def create_cluster(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a Dataproc cluster

        Args:
            config: Cluster configuration

        Returns:
            Cluster creation result
        """
        print(f"\n{'='*60}")
        print("Creating Dataproc Cluster")
        print(f"{'='*60}")

        cluster_name = config.get('cluster_name', 'my-cluster')
        master_machine_type = config.get('master_machine_type', 'n1-standard-4')
        worker_machine_type = config.get('worker_machine_type', 'n1-standard-4')
        num_workers = config.get('num_workers', 2)
        image_version = config.get('image_version', '2.1-debian11')

        code = f"""
from google.cloud import dataproc_v1

# Create cluster client
client = dataproc_v1.ClusterControllerClient(
    client_options={{
        'api_endpoint': f'{self.region}-dataproc.googleapis.com:443'
    }}
)

# Define cluster configuration
cluster = {{
    'project_id': '{self.project_id}',
    'cluster_name': '{cluster_name}',
    'config': {{
        'master_config': {{
            'num_instances': 1,
            'machine_type_uri': '{master_machine_type}',
            'disk_config': {{
                'boot_disk_size_gb': 500,
                'boot_disk_type': 'pd-standard',
            }}
        }},
        'worker_config': {{
            'num_instances': {num_workers},
            'machine_type_uri': '{worker_machine_type}',
            'disk_config': {{
                'boot_disk_size_gb': 500,
                'num_local_ssds': 0,
            }}
        }},
        'software_config': {{
            'image_version': '{image_version}',
            'optional_components': [
                'JUPYTER',
                'ZEPPELIN',
            ],
            'properties': {{
                'spark:spark.executor.memory': '4g',
                'spark:spark.driver.memory': '2g',
            }}
        }},
        'initialization_actions': [
            {{
                'executable_file': 'gs://bucket/init-scripts/install-dependencies.sh',
                'execution_timeout': {{'seconds': 300}}
            }}
        ],
        'lifecycle_config': {{
            'idle_delete_ttl': {{'seconds': 3600}},  # Auto-delete after 1 hour idle
        }},
    }}
}}

# Create the cluster
operation = client.create_cluster(
    request={{
        'project_id': '{self.project_id}',
        'region': '{self.region}',
        'cluster': cluster
    }}
)

# Wait for cluster to be ready
result = operation.result()
print(f"Cluster created: {{result.cluster_name}}")
print(f"Status: {{result.status.state.name}}")
"""

        result = {
            'cluster_name': cluster_name,
            'master_machine_type': master_machine_type,
            'worker_machine_type': worker_machine_type,
            'num_workers': num_workers,
            'image_version': image_version,
            'region': self.region,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.clusters.append(result)

        print(f"✓ Cluster created: {cluster_name}")
        print(f"  Master: {master_machine_type}")
        print(f"  Workers: {num_workers}x {worker_machine_type}")
        print(f"  Image: {image_version}")
        print(f"{'='*60}")

        return result

    def create_autoscaling_cluster(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create cluster with autoscaling

        Args:
            config: Cluster configuration with autoscaling

        Returns:
            Cluster creation result
        """
        print(f"\n{'='*60}")
        print("Creating Autoscaling Cluster")
        print(f"{'='*60}")

        cluster_name = config.get('cluster_name', 'autoscaling-cluster')
        min_workers = config.get('min_workers', 2)
        max_workers = config.get('max_workers', 10)

        code = f"""
from google.cloud import dataproc_v1

client = dataproc_v1.ClusterControllerClient(
    client_options={{'api_endpoint': f'{self.region}-dataproc.googleapis.com:443'}}
)

# Autoscaling policy
autoscaling_policy = dataproc_v1.AutoscalingPolicy(
    id='{cluster_name}-policy',
    worker_config=dataproc_v1.InstanceGroupAutoscalingPolicyConfig(
        min_instances={min_workers},
        max_instances={max_workers},
        weight=1
    ),
    basic_algorithm=dataproc_v1.BasicAutoscalingAlgorithm(
        yarn_config=dataproc_v1.BasicYarnAutoscalingConfig(
            graceful_decommission_timeout={{'seconds': 300}},
            scale_up_factor=0.5,
            scale_down_factor=0.5,
            scale_up_min_worker_fraction=0.0,
            scale_down_min_worker_fraction=0.0,
        )
    )
)

# Create autoscaling policy first
policy_client = dataproc_v1.AutoscalingPolicyServiceClient()
policy = policy_client.create_autoscaling_policy(
    parent=f'projects/{self.project_id}/regions/{self.region}',
    policy=autoscaling_policy
)

# Cluster with autoscaling
cluster = {{
    'cluster_name': '{cluster_name}',
    'config': {{
        'master_config': {{'num_instances': 1, 'machine_type_uri': 'n1-standard-4'}},
        'worker_config': {{
            'num_instances': {min_workers},
            'machine_type_uri': 'n1-standard-4',
        }},
        'autoscaling_config': {{
            'policy_uri': policy.name
        }},
    }}
}}

operation = client.create_cluster(
    request={{
        'project_id': '{self.project_id}',
        'region': '{self.region}',
        'cluster': cluster
    }}
)

result = operation.result()
print(f"Autoscaling cluster created: {{result.cluster_name}}")
print(f"Workers: {min_workers}-{max_workers} (autoscaling)")
"""

        result = {
            'cluster_name': cluster_name,
            'min_workers': min_workers,
            'max_workers': max_workers,
            'autoscaling': True,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Autoscaling cluster created: {cluster_name}")
        print(f"  Workers: {min_workers}-{max_workers}")
        print(f"  Autoscaling: Enabled")
        print(f"{'='*60}")

        return result


class DataprocJobs:
    """Submit and manage Dataproc jobs"""

    def __init__(self, project_id: str, region: str = 'us-central1'):
        """Initialize job manager"""
        self.project_id = project_id
        self.region = region
        self.jobs = []

    def submit_spark_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit Spark job

        Args:
            config: Job configuration

        Returns:
            Job submission result
        """
        print(f"\n{'='*60}")
        print("Submitting Spark Job")
        print(f"{'='*60}")

        cluster_name = config.get('cluster_name', 'my-cluster')
        main_class = config.get('main_class', 'com.example.Main')
        jar_file = config.get('jar_file', 'gs://bucket/jars/my-spark-job.jar')
        args = config.get('args', [])

        code = f"""
from google.cloud import dataproc_v1

client = dataproc_v1.JobControllerClient(
    client_options={{'api_endpoint': f'{self.region}-dataproc.googleapis.com:443'}}
)

# Define Spark job
job = {{
    'placement': {{
        'cluster_name': '{cluster_name}'
    }},
    'spark_job': {{
        'main_class': '{main_class}',
        'jar_file_uris': ['{jar_file}'],
        'args': {args},
        'properties': {{
            'spark.executor.instances': '4',
            'spark.executor.memory': '4g',
            'spark.driver.memory': '2g',
        }}
    }}
}}

# Submit job
operation = client.submit_job_as_operation(
    request={{
        'project_id': '{self.project_id}',
        'region': '{self.region}',
        'job': job
    }}
)

# Wait for job completion
result = operation.result()
print(f"Spark job completed: {{result.reference.job_id}}")
print(f"Status: {{result.status.state.name}}")
"""

        result = {
            'job_type': 'SPARK',
            'cluster_name': cluster_name,
            'main_class': main_class,
            'jar_file': jar_file,
            'args': args,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.jobs.append(result)

        print(f"✓ Spark job submitted")
        print(f"  Cluster: {cluster_name}")
        print(f"  Main class: {main_class}")
        print(f"  JAR: {jar_file}")
        print(f"{'='*60}")

        return result

    def submit_pyspark_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit PySpark job

        Args:
            config: Job configuration

        Returns:
            Job submission result
        """
        print(f"\n{'='*60}")
        print("Submitting PySpark Job")
        print(f"{'='*60}")

        cluster_name = config.get('cluster_name', 'my-cluster')
        main_python_file = config.get('main_python_file', 'gs://bucket/scripts/job.py')
        python_files = config.get('python_files', [])

        code = f"""
from google.cloud import dataproc_v1

client = dataproc_v1.JobControllerClient(
    client_options={{'api_endpoint': f'{self.region}-dataproc.googleapis.com:443'}}
)

# Define PySpark job
job = {{
    'placement': {{
        'cluster_name': '{cluster_name}'
    }},
    'pyspark_job': {{
        'main_python_file_uri': '{main_python_file}',
        'python_file_uris': {python_files},
        'properties': {{
            'spark.executor.instances': '4',
            'spark.executor.cores': '2',
        }}
    }}
}}

# Submit job
operation = client.submit_job_as_operation(
    request={{
        'project_id': '{self.project_id}',
        'region': '{self.region}',
        'job': job
    }}
)

result = operation.result()
print(f"PySpark job completed: {{result.reference.job_id}}")
"""

        result = {
            'job_type': 'PYSPARK',
            'cluster_name': cluster_name,
            'main_python_file': main_python_file,
            'python_files': python_files,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.jobs.append(result)

        print(f"✓ PySpark job submitted")
        print(f"  Cluster: {cluster_name}")
        print(f"  Script: {main_python_file}")
        print(f"{'='*60}")

        return result

    def submit_hive_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit Hive job

        Args:
            config: Job configuration

        Returns:
            Job submission result
        """
        print(f"\n{'='*60}")
        print("Submitting Hive Job")
        print(f"{'='*60}")

        cluster_name = config.get('cluster_name', 'my-cluster')
        query_file = config.get('query_file', 'gs://bucket/hive/query.sql')

        code = f"""
from google.cloud import dataproc_v1

client = dataproc_v1.JobControllerClient(
    client_options={{'api_endpoint': f'{self.region}-dataproc.googleapis.com:443'}}
)

# Define Hive job
job = {{
    'placement': {{
        'cluster_name': '{cluster_name}'
    }},
    'hive_job': {{
        'query_file_uri': '{query_file}',
        'script_variables': {{
            'INPUT_PATH': 'gs://bucket/data/input',
            'OUTPUT_PATH': 'gs://bucket/data/output',
        }}
    }}
}}

# Submit job
operation = client.submit_job_as_operation(
    request={{
        'project_id': '{self.project_id}',
        'region': '{self.region}',
        'job': job
    }}
)

result = operation.result()
print(f"Hive job completed: {{result.reference.job_id}}")
"""

        result = {
            'job_type': 'HIVE',
            'cluster_name': cluster_name,
            'query_file': query_file,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Hive job submitted")
        print(f"  Cluster: {cluster_name}")
        print(f"  Query: {query_file}")
        print(f"{'='*60}")

        return result


class WorkflowTemplates:
    """Manage workflow templates for orchestrating jobs"""

    def __init__(self, project_id: str, region: str = 'us-central1'):
        """Initialize workflow manager"""
        self.project_id = project_id
        self.region = region

    def create_workflow_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create workflow template

        Args:
            config: Template configuration

        Returns:
            Template creation result
        """
        print(f"\n{'='*60}")
        print("Creating Workflow Template")
        print(f"{'='*60}")

        template_id = config.get('template_id', 'etl-workflow')
        cluster_name = config.get('cluster_name', 'workflow-cluster')

        code = f"""
from google.cloud import dataproc_v1

client = dataproc_v1.WorkflowTemplateServiceClient(
    client_options={{'api_endpoint': f'{self.region}-dataproc.googleapis.com:443'}}
)

# Define workflow template
template = dataproc_v1.WorkflowTemplate(
    id='{template_id}',
    placement=dataproc_v1.WorkflowTemplatePlacement(
        managed_cluster=dataproc_v1.ManagedCluster(
            cluster_name='{cluster_name}',
            config=dataproc_v1.ClusterConfig(
                master_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=1,
                    machine_type_uri='n1-standard-4'
                ),
                worker_config=dataproc_v1.InstanceGroupConfig(
                    num_instances=2,
                    machine_type_uri='n1-standard-4'
                ),
            )
        )
    ),
    jobs=[
        # Job 1: Data ingestion
        dataproc_v1.OrderedJob(
            step_id='ingestion',
            spark_job=dataproc_v1.SparkJob(
                main_class='com.example.Ingestion',
                jar_file_uris=['gs://bucket/jars/ingestion.jar']
            )
        ),
        # Job 2: Data transformation
        dataproc_v1.OrderedJob(
            step_id='transformation',
            pyspark_job=dataproc_v1.PySparkJob(
                main_python_file_uri='gs://bucket/scripts/transform.py'
            ),
            prerequisite_step_ids=['ingestion']  # Wait for ingestion
        ),
        # Job 3: Data loading
        dataproc_v1.OrderedJob(
            step_id='loading',
            spark_job=dataproc_v1.SparkJob(
                main_class='com.example.Loading',
                jar_file_uris=['gs://bucket/jars/loading.jar']
            ),
            prerequisite_step_ids=['transformation']  # Wait for transformation
        ),
    ]
)

# Create the template
parent = f'projects/{self.project_id}/regions/{self.region}'
result = client.create_workflow_template(
    parent=parent,
    template=template
)

print(f"Workflow template created: {{result.id}}")
"""

        result = {
            'template_id': template_id,
            'cluster_name': cluster_name,
            'jobs_count': 3,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Workflow template created: {template_id}")
        print(f"  Jobs: 3 (ingestion → transformation → loading)")
        print(f"{'='*60}")

        return result

    def instantiate_workflow(self, template_id: str) -> str:
        """
        Run a workflow template

        Args:
            template_id: Template ID

        Returns:
            Workflow execution code
        """
        code = f"""
from google.cloud import dataproc_v1

client = dataproc_v1.WorkflowTemplateServiceClient(
    client_options={{'api_endpoint': f'{self.region}-dataproc.googleapis.com:443'}}
)

# Instantiate workflow
name = f'projects/{self.project_id}/regions/{self.region}/workflowTemplates/{template_id}'

operation = client.instantiate_workflow_template(name=name)
result = operation.result()

print(f"Workflow completed: {template_id}")
"""

        print(f"\n✓ Workflow instantiation code generated for: {template_id}")
        return code


class ClusterMonitoring:
    """Monitor and manage clusters"""

    def __init__(self, project_id: str, region: str = 'us-central1'):
        """Initialize monitoring"""
        self.project_id = project_id
        self.region = region

    def list_clusters(self) -> str:
        """
        List all clusters

        Returns:
            Code to list clusters
        """
        code = f"""
from google.cloud import dataproc_v1

client = dataproc_v1.ClusterControllerClient(
    client_options={{'api_endpoint': f'{self.region}-dataproc.googleapis.com:443'}}
)

# List clusters
print("Dataproc Clusters:")
print("=" * 60)

for cluster in client.list_clusters(
    request={{'project_id': '{self.project_id}', 'region': '{self.region}'}}
):
    print(f"Cluster: {{cluster.cluster_name}}")
    print(f"  State: {{cluster.status.state.name}}")
    print(f"  Workers: {{cluster.config.worker_config.num_instances}}")
    print(f"  Image: {{cluster.config.software_config.image_version}}")
    print(f"  Created: {{cluster.status.state_start_time}}")
    print("-" * 60)
"""

        print("\n✓ Cluster listing code generated")
        return code

    def delete_cluster(self, cluster_name: str) -> Dict[str, Any]:
        """
        Delete a cluster

        Args:
            cluster_name: Cluster name

        Returns:
            Delete operation result
        """
        print(f"\n{'='*60}")
        print("Deleting Cluster")
        print(f"{'='*60}")

        code = f"""
from google.cloud import dataproc_v1

client = dataproc_v1.ClusterControllerClient(
    client_options={{'api_endpoint': f'{self.region}-dataproc.googleapis.com:443'}}
)

# Delete cluster
operation = client.delete_cluster(
    request={{
        'project_id': '{self.project_id}',
        'region': '{self.region}',
        'cluster_name': '{cluster_name}'
    }}
)

operation.result()
print(f"Cluster deleted: {cluster_name}")
"""

        result = {
            'cluster_name': cluster_name,
            'action': 'DELETED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Cluster deleted: {cluster_name}")
        print(f"{'='*60}")

        return result


class DataprocManager:
    """Comprehensive Dataproc management"""

    def __init__(self, project_id: str = 'my-project', region: str = 'us-central1'):
        """
        Initialize Dataproc manager

        Args:
            project_id: GCP project ID
            region: Dataproc region
        """
        self.project_id = project_id
        self.region = region
        self.cluster_manager = DataprocCluster(project_id, region)
        self.job_manager = DataprocJobs(project_id, region)
        self.workflow_manager = WorkflowTemplates(project_id, region)
        self.monitoring = ClusterMonitoring(project_id, region)

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'region': self.region,
            'clusters': len(self.cluster_manager.clusters),
            'jobs': len(self.job_manager.jobs),
            'features': [
                'spark_jobs',
                'pyspark_jobs',
                'hive_jobs',
                'autoscaling',
                'workflow_templates',
                'cluster_management'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Dataproc capabilities"""
    print("=" * 60)
    print("Dataproc Comprehensive Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'
    region = 'us-central1'

    # Initialize manager
    mgr = DataprocManager(project_id, region)

    # Create cluster
    cluster_result = mgr.cluster_manager.create_cluster({
        'cluster_name': 'analytics-cluster',
        'master_machine_type': 'n1-standard-4',
        'worker_machine_type': 'n1-standard-4',
        'num_workers': 2,
        'image_version': '2.1-debian11'
    })

    # Create autoscaling cluster
    autoscaling_result = mgr.cluster_manager.create_autoscaling_cluster({
        'cluster_name': 'autoscaling-cluster',
        'min_workers': 2,
        'max_workers': 10
    })

    # Submit Spark job
    spark_result = mgr.job_manager.submit_spark_job({
        'cluster_name': 'analytics-cluster',
        'main_class': 'com.example.ETLJob',
        'jar_file': 'gs://my-bucket/jars/etl-job.jar',
        'args': ['--input', 'gs://my-bucket/input', '--output', 'gs://my-bucket/output']
    })

    # Submit PySpark job
    pyspark_result = mgr.job_manager.submit_pyspark_job({
        'cluster_name': 'analytics-cluster',
        'main_python_file': 'gs://my-bucket/scripts/analysis.py',
        'python_files': ['gs://my-bucket/libs/utils.py']
    })

    # Submit Hive job
    hive_result = mgr.job_manager.submit_hive_job({
        'cluster_name': 'analytics-cluster',
        'query_file': 'gs://my-bucket/hive/aggregation.sql'
    })

    # Create workflow template
    workflow_result = mgr.workflow_manager.create_workflow_template({
        'template_id': 'daily-etl-workflow',
        'cluster_name': 'workflow-cluster'
    })

    # Run workflow
    workflow_run_code = mgr.workflow_manager.instantiate_workflow('daily-etl-workflow')

    # Monitoring
    list_code = mgr.monitoring.list_clusters()

    # Manager info
    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Dataproc Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Region: {info['region']}")
    print(f"Clusters: {info['clusters']}")
    print(f"Jobs: {info['jobs']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")
    print("\nDataproc Best Practices:")
    print("  1. Use autoscaling for variable workloads")
    print("  2. Enable preemptible workers for cost savings")
    print("  3. Use workflow templates for complex pipelines")
    print("  4. Set idle deletion TTL to reduce costs")
    print("  5. Monitor job metrics and cluster utilization")
    print("  6. Use initialization actions for custom setup")


if __name__ == "__main__":
    demo()
