"""
AWS CodePipeline Management
============================

Comprehensive CI/CD pipeline management with multi-stage workflows and integrations.

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


class CodePipelineManager:
    """
    Advanced AWS CodePipeline Management System

    Provides comprehensive CI/CD pipeline operations including:
    - Pipeline creation with multi-stage workflows
    - Source integrations (GitHub, CodeCommit, S3)
    - Build integrations (CodeBuild)
    - Deploy integrations (ECS, Lambda, CloudFormation)
    - Execution management and monitoring
    - Approval actions
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize CodePipeline Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.codepipeline_client = session.client('codepipeline', region_name=region)
            self.region = region
            logger.info(f"CodePipeline Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing CodePipeline Manager: {e}")
            raise

    # ==================== Pipeline Management ====================

    def create_pipeline(
        self,
        pipeline_name: str,
        role_arn: str,
        artifact_store: Dict[str, str],
        stages: List[Dict[str, Any]],
        tags: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create CodePipeline.

        Args:
            pipeline_name: Pipeline name
            role_arn: IAM role ARN with pipeline permissions
            artifact_store: S3 bucket for artifacts {'type': 'S3', 'location': 'bucket-name'}
            stages: Pipeline stages configuration
            tags: Resource tags

        Returns:
            Pipeline details
        """
        try:
            logger.info(f"Creating pipeline: {pipeline_name}")

            pipeline_config = {
                'name': pipeline_name,
                'roleArn': role_arn,
                'artifactStore': artifact_store,
                'stages': stages
            }

            params = {'pipeline': pipeline_config}
            if tags:
                params['tags'] = tags

            response = self.codepipeline_client.create_pipeline(**params)

            logger.info(f"✓ Pipeline created: {pipeline_name}")

            return {
                'pipeline_name': response['pipeline']['name'],
                'version': response['pipeline']['version'],
                'stages': len(response['pipeline']['stages'])
            }

        except ClientError as e:
            logger.error(f"Error creating pipeline: {e}")
            raise

    def get_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        """Get pipeline configuration."""
        try:
            response = self.codepipeline_client.get_pipeline(name=pipeline_name)

            pipeline = response['pipeline']
            return {
                'name': pipeline['name'],
                'version': pipeline['version'],
                'role_arn': pipeline['roleArn'],
                'stages': [
                    {
                        'name': stage['name'],
                        'actions': [a['name'] for a in stage['actions']]
                    }
                    for stage in pipeline['stages']
                ],
                'artifact_store': pipeline['artifactStore']
            }

        except ClientError as e:
            logger.error(f"Error getting pipeline: {e}")
            raise

    def update_pipeline(
        self,
        pipeline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update pipeline configuration."""
        try:
            logger.info(f"Updating pipeline: {pipeline['name']}")

            response = self.codepipeline_client.update_pipeline(pipeline=pipeline)

            logger.info(f"✓ Pipeline updated")
            return response['pipeline']

        except ClientError as e:
            logger.error(f"Error updating pipeline: {e}")
            raise

    def delete_pipeline(self, pipeline_name: str) -> None:
        """Delete pipeline."""
        try:
            self.codepipeline_client.delete_pipeline(name=pipeline_name)
            logger.info(f"✓ Pipeline deleted: {pipeline_name}")
        except ClientError as e:
            logger.error(f"Error deleting pipeline: {e}")
            raise

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines."""
        try:
            response = self.codepipeline_client.list_pipelines()

            pipelines = []
            for pipeline in response.get('pipelines', []):
                pipelines.append({
                    'name': pipeline['name'],
                    'version': pipeline['version'],
                    'created': pipeline['created'].isoformat() if 'created' in pipeline else 'N/A',
                    'updated': pipeline['updated'].isoformat() if 'updated' in pipeline else 'N/A'
                })

            logger.info(f"Found {len(pipelines)} pipeline(s)")
            return pipelines

        except ClientError as e:
            logger.error(f"Error listing pipelines: {e}")
            raise

    # ==================== Execution Management ====================

    def start_pipeline_execution(
        self,
        pipeline_name: str,
        client_request_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start pipeline execution."""
        try:
            logger.info(f"Starting pipeline execution: {pipeline_name}")

            params = {'name': pipeline_name}
            if client_request_token:
                params['clientRequestToken'] = client_request_token

            response = self.codepipeline_client.start_pipeline_execution(**params)

            logger.info(f"✓ Execution started: {response['pipelineExecutionId']}")

            return {
                'pipeline_name': pipeline_name,
                'execution_id': response['pipelineExecutionId']
            }

        except ClientError as e:
            logger.error(f"Error starting pipeline execution: {e}")
            raise

    def get_pipeline_execution(
        self,
        pipeline_name: str,
        execution_id: str
    ) -> Dict[str, Any]:
        """Get pipeline execution details."""
        try:
            response = self.codepipeline_client.get_pipeline_execution(
                pipelineName=pipeline_name,
                pipelineExecutionId=execution_id
            )

            execution = response['pipelineExecution']
            return {
                'pipeline_name': pipeline_name,
                'execution_id': execution['pipelineExecutionId'],
                'status': execution['status'],
                'version': execution.get('pipelineVersion', 'N/A')
            }

        except ClientError as e:
            logger.error(f"Error getting pipeline execution: {e}")
            raise

    def list_pipeline_executions(
        self,
        pipeline_name: str,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """List pipeline executions."""
        try:
            response = self.codepipeline_client.list_pipeline_executions(
                pipelineName=pipeline_name,
                maxResults=max_results
            )

            executions = []
            for execution in response.get('pipelineExecutionSummaries', []):
                executions.append({
                    'execution_id': execution['pipelineExecutionId'],
                    'status': execution['status'],
                    'start_time': execution.get('startTime', datetime.now()).isoformat(),
                    'last_update': execution.get('lastUpdateTime', datetime.now()).isoformat()
                })

            logger.info(f"Found {len(executions)} execution(s)")
            return executions

        except ClientError as e:
            logger.error(f"Error listing pipeline executions: {e}")
            raise

    def stop_pipeline_execution(
        self,
        pipeline_name: str,
        execution_id: str,
        abandon: bool = False,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stop pipeline execution."""
        try:
            logger.info(f"Stopping execution: {execution_id}")

            params = {
                'pipelineName': pipeline_name,
                'pipelineExecutionId': execution_id,
                'abandon': abandon
            }
            if reason:
                params['reason'] = reason

            response = self.codepipeline_client.stop_pipeline_execution(**params)

            logger.info(f"✓ Execution stopped")
            return response

        except ClientError as e:
            logger.error(f"Error stopping execution: {e}")
            raise

    # ==================== Pipeline State ====================

    def get_pipeline_state(self, pipeline_name: str) -> Dict[str, Any]:
        """Get current pipeline state."""
        try:
            response = self.codepipeline_client.get_pipeline_state(name=pipeline_name)

            stages = []
            for stage in response['stageStates']:
                stage_info = {
                    'name': stage['stageName'],
                    'status': stage.get('latestExecution', {}).get('status', 'N/A')
                }
                stages.append(stage_info)

            return {
                'pipeline_name': response['pipelineName'],
                'version': response.get('pipelineVersion', 'N/A'),
                'stages': stages,
                'created': response.get('created', datetime.now()).isoformat(),
                'updated': response.get('updated', datetime.now()).isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting pipeline state: {e}")
            raise

    # ==================== Approval Actions ====================

    def put_approval_result(
        self,
        pipeline_name: str,
        stage_name: str,
        action_name: str,
        token: str,
        result: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Approve or reject manual approval action.

        Args:
            pipeline_name: Pipeline name
            stage_name: Stage name
            action_name: Action name
            token: Approval token
            result: {'summary': 'message', 'status': 'Approved' or 'Rejected'}
        """
        try:
            logger.info(f"Putting approval result for {pipeline_name}")

            response = self.codepipeline_client.put_approval_result(
                pipelineName=pipeline_name,
                stageName=stage_name,
                actionName=action_name,
                token=token,
                result=result
            )

            logger.info(f"✓ Approval result submitted: {result['status']}")
            return response

        except ClientError as e:
            logger.error(f"Error putting approval result: {e}")
            raise

    # ==================== Monitoring ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get CodePipeline summary."""
        try:
            pipelines = self.list_pipelines()

            return {
                'region': self.region,
                'total_pipelines': len(pipelines),
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of CodePipeline Manager capabilities."""
    print("AWS CodePipeline Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    print("\n1️⃣  Create Pipeline:")
    print("""
    cp = CodePipelineManager(region='us-east-1')

    # Define pipeline stages
    stages = [
        {
            'name': 'Source',
            'actions': [{
                'name': 'SourceAction',
                'actionTypeId': {
                    'category': 'Source',
                    'owner': 'AWS',
                    'provider': 'CodeCommit',
                    'version': '1'
                },
                'configuration': {
                    'RepositoryName': 'my-repo',
                    'BranchName': 'main'
                },
                'outputArtifacts': [{'name': 'SourceOutput'}]
            }]
        },
        {
            'name': 'Build',
            'actions': [{
                'name': 'BuildAction',
                'actionTypeId': {
                    'category': 'Build',
                    'owner': 'AWS',
                    'provider': 'CodeBuild',
                    'version': '1'
                },
                'configuration': {'ProjectName': 'my-build-project'},
                'inputArtifacts': [{'name': 'SourceOutput'}],
                'outputArtifacts': [{'name': 'BuildOutput'}]
            }]
        },
        {
            'name': 'Deploy',
            'actions': [{
                'name': 'DeployAction',
                'actionTypeId': {
                    'category': 'Deploy',
                    'owner': 'AWS',
                    'provider': 'ECS',
                    'version': '1'
                },
                'configuration': {
                    'ClusterName': 'my-cluster',
                    'ServiceName': 'my-service'
                },
                'inputArtifacts': [{'name': 'BuildOutput'}]
            }]
        }
    ]

    pipeline = cp.create_pipeline(
        pipeline_name='my-app-pipeline',
        role_arn='arn:aws:iam::123456789012:role/CodePipelineRole',
        artifact_store={'type': 'S3', 'location': 'my-artifacts-bucket'},
        stages=stages
    )
    """)

    print("\n2️⃣  Execute Pipeline:")
    print("""
    # Start execution
    execution = cp.start_pipeline_execution('my-app-pipeline')
    print(f"Execution ID: {execution['execution_id']}")

    # Monitor execution
    status = cp.get_pipeline_execution(
        'my-app-pipeline',
        execution['execution_id']
    )
    print(f"Status: {status['status']}")

    # Get pipeline state
    state = cp.get_pipeline_state('my-app-pipeline')
    for stage in state['stages']:
        print(f"Stage {stage['name']}: {stage['status']}")
    """)

    print("\n3️⃣  Manual Approval:")
    print("""
    # Approve deployment
    cp.put_approval_result(
        pipeline_name='my-app-pipeline',
        stage_name='Approval',
        action_name='ManualApproval',
        token='approval-token-12345',
        result={
            'summary': 'Approved by DevOps team',
            'status': 'Approved'
        }
    )
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")
    print("\n⚠️  Setup: Configure AWS credentials and IAM roles")


if __name__ == '__main__':
    demo()
