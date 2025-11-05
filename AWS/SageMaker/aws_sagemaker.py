"""
AWS SageMaker Management
========================

Comprehensive machine learning operations with training, deployment, and inference.

Author: Brill Consulting
"""

import boto3
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SageMakerManager:
    """
    Advanced AWS SageMaker Management System

    Provides comprehensive ML operations including:
    - Training job management
    - Model creation and deployment
    - Endpoint configuration and hosting
    - Batch transform jobs
    - Hyperparameter tuning
    - Model monitoring
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize SageMaker Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.sagemaker_client = session.client('sagemaker', region_name=region)
            self.sagemaker_runtime = session.client('sagemaker-runtime', region_name=region)
            self.region = region
            logger.info(f"SageMaker Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing SageMaker Manager: {e}")
            raise

    # ==================== Training Jobs ====================

    def create_training_job(
        self,
        job_name: str,
        role_arn: str,
        algorithm_specification: Dict[str, str],
        input_data_config: List[Dict[str, Any]],
        output_data_config: Dict[str, str],
        resource_config: Dict[str, Any],
        stopping_condition: Dict[str, int],
        hyperparameters: Optional[Dict[str, str]] = None,
        vpc_config: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Create SageMaker training job.

        Args:
            job_name: Unique training job name
            role_arn: IAM role ARN with SageMaker permissions
            algorithm_specification: Algorithm details (TrainingImage, TrainingInputMode)
            input_data_config: Input data channels
            output_data_config: Output location (S3OutputPath)
            resource_config: Instance config (InstanceType, InstanceCount, VolumeSizeInGB)
            stopping_condition: MaxRuntimeInSeconds
            hyperparameters: Algorithm hyperparameters
            vpc_config: VPC configuration
        """
        try:
            logger.info(f"Creating training job: {job_name}")

            params = {
                'TrainingJobName': job_name,
                'RoleArn': role_arn,
                'AlgorithmSpecification': algorithm_specification,
                'InputDataConfig': input_data_config,
                'OutputDataConfig': output_data_config,
                'ResourceConfig': resource_config,
                'StoppingCondition': stopping_condition
            }

            if hyperparameters:
                params['HyperParameters'] = hyperparameters
            if vpc_config:
                params['VpcConfig'] = vpc_config

            response = self.sagemaker_client.create_training_job(**params)

            logger.info(f"✓ Training job created: {response['TrainingJobArn']}")

            return {
                'job_name': job_name,
                'job_arn': response['TrainingJobArn'],
                'status': 'InProgress'
            }

        except ClientError as e:
            logger.error(f"Error creating training job: {e}")
            raise

    def describe_training_job(self, job_name: str) -> Dict[str, Any]:
        """Get training job details and status."""
        try:
            response = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)

            job_info = {
                'job_name': response['TrainingJobName'],
                'job_arn': response['TrainingJobArn'],
                'status': response['TrainingJobStatus'],
                'creation_time': response['CreationTime'].isoformat(),
                'algorithm': response['AlgorithmSpecification']['TrainingImage'],
                'instance_type': response['ResourceConfig']['InstanceType'],
                'instance_count': response['ResourceConfig']['InstanceCount']
            }

            if response.get('TrainingEndTime'):
                job_info['end_time'] = response['TrainingEndTime'].isoformat()
                job_info['training_time_seconds'] = response.get('TrainingTimeInSeconds', 0)

            if response.get('FinalMetricDataList'):
                job_info['metrics'] = {
                    m['MetricName']: m['Value']
                    for m in response['FinalMetricDataList']
                }

            logger.info(f"Training job status: {job_info['status']}")
            return job_info

        except ClientError as e:
            logger.error(f"Error describing training job: {e}")
            raise

    def list_training_jobs(
        self,
        status_equals: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """List training jobs."""
        try:
            params = {'MaxResults': max_results}
            if status_equals:
                params['StatusEquals'] = status_equals

            response = self.sagemaker_client.list_training_jobs(**params)

            jobs = []
            for job in response.get('TrainingJobSummaries', []):
                jobs.append({
                    'job_name': job['TrainingJobName'],
                    'status': job['TrainingJobStatus'],
                    'creation_time': job['CreationTime'].isoformat()
                })

            logger.info(f"Found {len(jobs)} training job(s)")
            return jobs

        except ClientError as e:
            logger.error(f"Error listing training jobs: {e}")
            raise

    def stop_training_job(self, job_name: str) -> None:
        """Stop a running training job."""
        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
            logger.info(f"✓ Training job stopped: {job_name}")
        except ClientError as e:
            logger.error(f"Error stopping training job: {e}")
            raise

    # ==================== Models ====================

    def create_model(
        self,
        model_name: str,
        role_arn: str,
        primary_container: Dict[str, str],
        vpc_config: Optional[Dict[str, List[str]]] = None,
        tags: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create SageMaker model.

        Args:
            model_name: Model name
            role_arn: IAM role ARN
            primary_container: Container definition (Image, ModelDataUrl)
            vpc_config: VPC configuration
            tags: Resource tags
        """
        try:
            logger.info(f"Creating model: {model_name}")

            params = {
                'ModelName': model_name,
                'ExecutionRoleArn': role_arn,
                'PrimaryContainer': primary_container
            }

            if vpc_config:
                params['VpcConfig'] = vpc_config
            if tags:
                params['Tags'] = tags

            response = self.sagemaker_client.create_model(**params)

            logger.info(f"✓ Model created: {response['ModelArn']}")

            return {
                'model_name': model_name,
                'model_arn': response['ModelArn']
            }

        except ClientError as e:
            logger.error(f"Error creating model: {e}")
            raise

    def describe_model(self, model_name: str) -> Dict[str, Any]:
        """Get model details."""
        try:
            response = self.sagemaker_client.describe_model(ModelName=model_name)

            return {
                'model_name': response['ModelName'],
                'model_arn': response['ModelArn'],
                'creation_time': response['CreationTime'].isoformat(),
                'execution_role': response['ExecutionRoleArn'],
                'container_image': response['PrimaryContainer']['Image'],
                'model_data': response['PrimaryContainer'].get('ModelDataUrl', 'N/A')
            }

        except ClientError as e:
            logger.error(f"Error describing model: {e}")
            raise

    def delete_model(self, model_name: str) -> None:
        """Delete model."""
        try:
            self.sagemaker_client.delete_model(ModelName=model_name)
            logger.info(f"✓ Model deleted: {model_name}")
        except ClientError as e:
            logger.error(f"Error deleting model: {e}")
            raise

    # ==================== Endpoints ====================

    def create_endpoint_config(
        self,
        config_name: str,
        production_variants: List[Dict[str, Any]],
        data_capture_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create endpoint configuration."""
        try:
            logger.info(f"Creating endpoint config: {config_name}")

            params = {
                'EndpointConfigName': config_name,
                'ProductionVariants': production_variants
            }

            if data_capture_config:
                params['DataCaptureConfig'] = data_capture_config

            response = self.sagemaker_client.create_endpoint_config(**params)

            logger.info(f"✓ Endpoint config created: {response['EndpointConfigArn']}")

            return {
                'config_name': config_name,
                'config_arn': response['EndpointConfigArn']
            }

        except ClientError as e:
            logger.error(f"Error creating endpoint config: {e}")
            raise

    def create_endpoint(self, endpoint_name: str, config_name: str) -> Dict[str, Any]:
        """Create and deploy endpoint."""
        try:
            logger.info(f"Creating endpoint: {endpoint_name}")

            response = self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )

            logger.info(f"✓ Endpoint created: {response['EndpointArn']}")

            return {
                'endpoint_name': endpoint_name,
                'endpoint_arn': response['EndpointArn'],
                'status': 'Creating'
            }

        except ClientError as e:
            logger.error(f"Error creating endpoint: {e}")
            raise

    def describe_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Get endpoint details and status."""
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)

            return {
                'endpoint_name': response['EndpointName'],
                'endpoint_arn': response['EndpointArn'],
                'status': response['EndpointStatus'],
                'creation_time': response['CreationTime'].isoformat(),
                'last_modified': response['LastModifiedTime'].isoformat()
            }

        except ClientError as e:
            logger.error(f"Error describing endpoint: {e}")
            raise

    def update_endpoint(self, endpoint_name: str, new_config_name: str) -> None:
        """Update endpoint with new configuration."""
        try:
            self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=new_config_name
            )
            logger.info(f"✓ Endpoint update initiated: {endpoint_name}")
        except ClientError as e:
            logger.error(f"Error updating endpoint: {e}")
            raise

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete endpoint."""
        try:
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"✓ Endpoint deleted: {endpoint_name}")
        except ClientError as e:
            logger.error(f"Error deleting endpoint: {e}")
            raise

    # ==================== Inference ====================

    def invoke_endpoint(
        self,
        endpoint_name: str,
        payload: bytes,
        content_type: str = 'application/json',
        accept: str = 'application/json'
    ) -> Dict[str, Any]:
        """Invoke endpoint for real-time inference."""
        try:
            logger.info(f"Invoking endpoint: {endpoint_name}")

            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=payload,
                ContentType=content_type,
                Accept=accept
            )

            result = response['Body'].read()

            logger.info(f"✓ Inference completed")

            return {
                'predictions': json.loads(result) if accept == 'application/json' else result,
                'content_type': response['ContentType']
            }

        except ClientError as e:
            logger.error(f"Error invoking endpoint: {e}")
            raise

    # ==================== Batch Transform ====================

    def create_transform_job(
        self,
        job_name: str,
        model_name: str,
        transform_input: Dict[str, Any],
        transform_output: Dict[str, str],
        transform_resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create batch transform job for batch inference."""
        try:
            logger.info(f"Creating transform job: {job_name}")

            response = self.sagemaker_client.create_transform_job(
                TransformJobName=job_name,
                ModelName=model_name,
                TransformInput=transform_input,
                TransformOutput=transform_output,
                TransformResources=transform_resources
            )

            logger.info(f"✓ Transform job created: {response['TransformJobArn']}")

            return {
                'job_name': job_name,
                'job_arn': response['TransformJobArn']
            }

        except ClientError as e:
            logger.error(f"Error creating transform job: {e}")
            raise

    # ==================== Monitoring ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get SageMaker resource summary."""
        try:
            training_jobs = self.list_training_jobs(max_results=100)

            return {
                'region': self.region,
                'training_jobs': len(training_jobs),
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of SageMaker Manager capabilities."""
    print("AWS SageMaker Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    print("\n1️⃣  Create Training Job:")
    print("""
    sagemaker = SageMakerManager(region='us-east-1')

    job = sagemaker.create_training_job(
        job_name='xgboost-training-2024',
        role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
        algorithm_specification={
            'TrainingImage': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1',
            'TrainingInputMode': 'File'
        },
        input_data_config=[{
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://my-bucket/train/',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        }],
        output_data_config={'S3OutputPath': 's3://my-bucket/output/'},
        resource_config={
            'InstanceType': 'ml.m5.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        stopping_condition={'MaxRuntimeInSeconds': 3600},
        hyperparameters={
            'max_depth': '5',
            'eta': '0.2',
            'objective': 'binary:logistic',
            'num_round': '100'
        }
    )
    """)

    print("\n2️⃣  Deploy Model:")
    print("""
    # Create model from training output
    model = sagemaker.create_model(
        model_name='xgboost-model',
        role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
        primary_container={
            'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1',
            'ModelDataUrl': 's3://my-bucket/output/model.tar.gz'
        }
    )

    # Create endpoint configuration
    config = sagemaker.create_endpoint_config(
        config_name='xgboost-config',
        production_variants=[{
            'VariantName': 'AllTraffic',
            'ModelName': 'xgboost-model',
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large'
        }]
    )

    # Deploy endpoint
    endpoint = sagemaker.create_endpoint(
        endpoint_name='xgboost-endpoint',
        config_name='xgboost-config'
    )
    """)

    print("\n3️⃣  Real-time Inference:")
    print("""
    import json

    # Prepare input data
    data = {'instances': [[5.1, 3.5, 1.4, 0.2]]}
    payload = json.dumps(data).encode('utf-8')

    # Invoke endpoint
    response = sagemaker.invoke_endpoint(
        endpoint_name='xgboost-endpoint',
        payload=payload,
        content_type='application/json'
    )

    print(f"Predictions: {response['predictions']}")
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")
    print("\n⚠️  Setup: Configure AWS credentials and IAM roles")


if __name__ == '__main__':
    demo()
