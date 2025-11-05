"""
AWS Lambda Management
=====================

Comprehensive serverless function management with event triggers,
layer deployment, VPC configuration, and monitoring.

Author: Brill Consulting
"""

import boto3
import logging
import json
import zipfile
import io
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LambdaManager:
    """
    Advanced AWS Lambda Management System

    Provides comprehensive Lambda function management including:
    - Function creation and deployment with code packaging
    - Event source mappings (S3, DynamoDB, SQS, SNS, API Gateway)
    - Layer management for shared dependencies
    - Environment variable configuration
    - VPC and security group integration
    - Invocation with different payload types
    - Alias and version management
    - CloudWatch Logs integration
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """
        Initialize Lambda Manager.

        Args:
            region: AWS region (default: us-east-1)
            profile: AWS CLI profile name (optional)
        """
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.lambda_client = session.client('lambda', region_name=region)
            self.iam_client = session.client('iam', region_name=region)
            self.logs_client = session.client('logs', region_name=region)
            self.region = region
            logger.info(f"Lambda Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing Lambda Manager: {e}")
            raise

    # ==================== Function Management ====================

    def create_function(
        self,
        function_name: str,
        runtime: str,
        handler: str,
        role_arn: str,
        code_content: Optional[str] = None,
        zip_file_path: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_key: Optional[str] = None,
        description: str = "",
        timeout: int = 60,
        memory_size: int = 256,
        environment_variables: Optional[Dict[str, str]] = None,
        layers: Optional[List[str]] = None,
        vpc_config: Optional[Dict[str, List[str]]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create Lambda function.

        Args:
            function_name: Function name
            runtime: Runtime (python3.11, nodejs18.x, etc.)
            handler: Handler (e.g., index.handler)
            role_arn: IAM role ARN for execution
            code_content: Inline code string (for simple functions)
            zip_file_path: Path to ZIP file containing function code
            s3_bucket: S3 bucket containing code
            s3_key: S3 key for code object
            description: Function description
            timeout: Timeout in seconds (max 900)
            memory_size: Memory in MB (128-10240)
            environment_variables: Environment variables dict
            layers: List of layer ARNs
            vpc_config: VPC configuration with SubnetIds and SecurityGroupIds
            tags: Resource tags

        Returns:
            Function configuration details
        """
        try:
            logger.info(f"Creating Lambda function: {function_name}")

            # Prepare code
            code = {}
            if code_content:
                # Create ZIP from inline code
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.writestr('index.py', code_content)
                code['ZipFile'] = zip_buffer.getvalue()
            elif zip_file_path:
                with open(zip_file_path, 'rb') as f:
                    code['ZipFile'] = f.read()
            elif s3_bucket and s3_key:
                code['S3Bucket'] = s3_bucket
                code['S3Key'] = s3_key
            else:
                raise ValueError("Must provide code_content, zip_file_path, or S3 location")

            # Build parameters
            params = {
                'FunctionName': function_name,
                'Runtime': runtime,
                'Role': role_arn,
                'Handler': handler,
                'Code': code,
                'Description': description,
                'Timeout': timeout,
                'MemorySize': memory_size,
                'Publish': True  # Publish as version 1
            }

            if environment_variables:
                params['Environment'] = {'Variables': environment_variables}

            if layers:
                params['Layers'] = layers

            if vpc_config:
                params['VpcConfig'] = vpc_config

            if tags:
                params['Tags'] = tags

            response = self.lambda_client.create_function(**params)

            logger.info(f"✓ Function created: {response['FunctionArn']}")

            return {
                'function_name': response['FunctionName'],
                'function_arn': response['FunctionArn'],
                'runtime': response['Runtime'],
                'handler': response['Handler'],
                'role': response['Role'],
                'state': response['State'],
                'version': response['Version']
            }

        except ClientError as e:
            logger.error(f"Error creating function: {e}")
            raise

    def update_function_code(
        self,
        function_name: str,
        zip_file_path: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_key: Optional[str] = None,
        publish: bool = True
    ) -> Dict[str, Any]:
        """Update function code."""
        try:
            logger.info(f"Updating function code: {function_name}")

            params = {'FunctionName': function_name, 'Publish': publish}

            if zip_file_path:
                with open(zip_file_path, 'rb') as f:
                    params['ZipFile'] = f.read()
            elif s3_bucket and s3_key:
                params['S3Bucket'] = s3_bucket
                params['S3Key'] = s3_key
            else:
                raise ValueError("Must provide zip_file_path or S3 location")

            response = self.lambda_client.update_function_code(**params)

            logger.info(f"✓ Function code updated, Version: {response.get('Version')}")
            return response

        except ClientError as e:
            logger.error(f"Error updating function code: {e}")
            raise

    def update_function_configuration(
        self,
        function_name: str,
        timeout: Optional[int] = None,
        memory_size: Optional[int] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        layers: Optional[List[str]] = None,
        vpc_config: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """Update function configuration."""
        try:
            logger.info(f"Updating function configuration: {function_name}")

            params = {'FunctionName': function_name}

            if timeout is not None:
                params['Timeout'] = timeout
            if memory_size is not None:
                params['MemorySize'] = memory_size
            if environment_variables is not None:
                params['Environment'] = {'Variables': environment_variables}
            if layers is not None:
                params['Layers'] = layers
            if vpc_config is not None:
                params['VpcConfig'] = vpc_config

            response = self.lambda_client.update_function_configuration(**params)

            logger.info("✓ Function configuration updated")
            return response

        except ClientError as e:
            logger.error(f"Error updating function configuration: {e}")
            raise

    def get_function(self, function_name: str) -> Dict[str, Any]:
        """Get function details."""
        try:
            response = self.lambda_client.get_function(FunctionName=function_name)

            function = {
                'function_name': response['Configuration']['FunctionName'],
                'function_arn': response['Configuration']['FunctionArn'],
                'runtime': response['Configuration']['Runtime'],
                'handler': response['Configuration']['Handler'],
                'role': response['Configuration']['Role'],
                'timeout': response['Configuration']['Timeout'],
                'memory_size': response['Configuration']['MemorySize'],
                'last_modified': response['Configuration']['LastModified'],
                'state': response['Configuration']['State'],
                'code_location': response['Code'].get('Location', 'N/A')
            }

            logger.info(f"Retrieved function: {function_name}")
            return function

        except ClientError as e:
            logger.error(f"Error getting function: {e}")
            raise

    def list_functions(self) -> List[Dict[str, Any]]:
        """List all Lambda functions."""
        try:
            response = self.lambda_client.list_functions()

            functions = []
            for func in response.get('Functions', []):
                functions.append({
                    'function_name': func['FunctionName'],
                    'function_arn': func['FunctionArn'],
                    'runtime': func['Runtime'],
                    'memory_size': func['MemorySize'],
                    'timeout': func['Timeout'],
                    'last_modified': func['LastModified']
                })

            logger.info(f"Found {len(functions)} function(s)")
            return functions

        except ClientError as e:
            logger.error(f"Error listing functions: {e}")
            raise

    def delete_function(self, function_name: str) -> None:
        """Delete Lambda function."""
        try:
            logger.info(f"Deleting function: {function_name}")
            self.lambda_client.delete_function(FunctionName=function_name)
            logger.info(f"✓ Function deleted: {function_name}")
        except ClientError as e:
            logger.error(f"Error deleting function: {e}")
            raise

    # ==================== Function Invocation ====================

    def invoke_function(
        self,
        function_name: str,
        payload: Dict[str, Any],
        invocation_type: str = "RequestResponse",
        log_type: str = "None"
    ) -> Dict[str, Any]:
        """
        Invoke Lambda function.

        Args:
            function_name: Function to invoke
            payload: Event payload
            invocation_type: RequestResponse, Event, or DryRun
            log_type: None or Tail (include last 4KB of logs)

        Returns:
            Invocation response with payload and logs
        """
        try:
            logger.info(f"Invoking function: {function_name}")

            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType=invocation_type,
                LogType=log_type,
                Payload=json.dumps(payload)
            )

            result = {
                'status_code': response['StatusCode'],
                'executed_version': response.get('ExecutedVersion', 'N/A'),
                'payload': json.loads(response['Payload'].read())
            }

            if log_type == 'Tail' and 'LogResult' in response:
                result['logs'] = base64.b64decode(response['LogResult']).decode('utf-8')

            logger.info(f"✓ Function invoked successfully (Status: {result['status_code']})")
            return result

        except ClientError as e:
            logger.error(f"Error invoking function: {e}")
            raise

    def invoke_async(self, function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke function asynchronously (Event invocation)."""
        return self.invoke_function(
            function_name=function_name,
            payload=payload,
            invocation_type="Event"
        )

    # ==================== Event Source Mappings ====================

    def create_event_source_mapping(
        self,
        function_name: str,
        event_source_arn: str,
        enabled: bool = True,
        batch_size: int = 10,
        starting_position: str = "LATEST"
    ) -> Dict[str, Any]:
        """
        Create event source mapping (for DynamoDB Streams, Kinesis, SQS).

        Args:
            function_name: Lambda function name
            event_source_arn: ARN of event source (DynamoDB, Kinesis, SQS)
            enabled: Enable mapping
            batch_size: Batch size for processing
            starting_position: LATEST or TRIM_HORIZON (for streams)

        Returns:
            Mapping details
        """
        try:
            logger.info(f"Creating event source mapping for: {function_name}")

            params = {
                'FunctionName': function_name,
                'EventSourceArn': event_source_arn,
                'Enabled': enabled,
                'BatchSize': batch_size
            }

            # Starting position only for streams (not SQS)
            if 'dynamodb' in event_source_arn.lower() or 'kinesis' in event_source_arn.lower():
                params['StartingPosition'] = starting_position

            response = self.lambda_client.create_event_source_mapping(**params)

            logger.info(f"✓ Event source mapping created: {response['UUID']}")

            return {
                'uuid': response['UUID'],
                'function_arn': response['FunctionArn'],
                'event_source_arn': response['EventSourceArn'],
                'state': response['State'],
                'batch_size': response['BatchSize']
            }

        except ClientError as e:
            logger.error(f"Error creating event source mapping: {e}")
            raise

    def list_event_source_mappings(
        self,
        function_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List event source mappings."""
        try:
            params = {}
            if function_name:
                params['FunctionName'] = function_name

            response = self.lambda_client.list_event_source_mappings(**params)

            mappings = []
            for mapping in response.get('EventSourceMappings', []):
                mappings.append({
                    'uuid': mapping['UUID'],
                    'function_arn': mapping['FunctionArn'],
                    'event_source_arn': mapping['EventSourceArn'],
                    'state': mapping['State'],
                    'batch_size': mapping.get('BatchSize', 'N/A')
                })

            logger.info(f"Found {len(mappings)} event source mapping(s)")
            return mappings

        except ClientError as e:
            logger.error(f"Error listing event source mappings: {e}")
            raise

    # ==================== Layers ====================

    def publish_layer_version(
        self,
        layer_name: str,
        description: str,
        zip_file_path: str,
        compatible_runtimes: List[str]
    ) -> Dict[str, Any]:
        """
        Publish Lambda layer version.

        Args:
            layer_name: Layer name
            description: Layer description
            zip_file_path: Path to ZIP file containing layer content
            compatible_runtimes: List of compatible runtimes

        Returns:
            Layer version details
        """
        try:
            logger.info(f"Publishing layer: {layer_name}")

            with open(zip_file_path, 'rb') as f:
                zip_content = f.read()

            response = self.lambda_client.publish_layer_version(
                LayerName=layer_name,
                Description=description,
                Content={'ZipFile': zip_content},
                CompatibleRuntimes=compatible_runtimes
            )

            logger.info(f"✓ Layer published: {response['LayerArn']} (Version {response['Version']})")

            return {
                'layer_arn': response['LayerArn'],
                'layer_version_arn': response['LayerVersionArn'],
                'version': response['Version'],
                'compatible_runtimes': response['CompatibleRuntimes']
            }

        except ClientError as e:
            logger.error(f"Error publishing layer: {e}")
            raise

    # ==================== Aliases and Versions ====================

    def create_alias(
        self,
        function_name: str,
        alias_name: str,
        function_version: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """Create function alias."""
        try:
            logger.info(f"Creating alias {alias_name} for {function_name}")

            response = self.lambda_client.create_alias(
                FunctionName=function_name,
                Name=alias_name,
                FunctionVersion=function_version,
                Description=description
            )

            logger.info(f"✓ Alias created: {response['AliasArn']}")

            return {
                'alias_arn': response['AliasArn'],
                'name': response['Name'],
                'function_version': response['FunctionVersion']
            }

        except ClientError as e:
            logger.error(f"Error creating alias: {e}")
            raise

    # ==================== Monitoring ====================

    def get_function_logs(
        self,
        function_name: str,
        limit: int = 50
    ) -> List[str]:
        """Get recent CloudWatch logs for function."""
        try:
            log_group_name = f"/aws/lambda/{function_name}"

            # Get log streams
            streams_response = self.logs_client.describe_log_streams(
                logGroupName=log_group_name,
                orderBy='LastEventTime',
                descending=True,
                limit=1
            )

            if not streams_response.get('logStreams'):
                return []

            log_stream_name = streams_response['logStreams'][0]['logStreamName']

            # Get log events
            events_response = self.logs_client.get_log_events(
                logGroupName=log_group_name,
                logStreamName=log_stream_name,
                limit=limit
            )

            logs = [event['message'] for event in events_response['events']]
            return logs

        except ClientError as e:
            logger.warning(f"Could not retrieve logs: {e}")
            return []

    def get_summary(self) -> Dict[str, Any]:
        """Get Lambda summary."""
        try:
            functions = self.list_functions()

            return {
                'region': self.region,
                'total_functions': len(functions),
                'runtimes': list(set(f['runtime'] for f in functions)),
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """
    Demonstration of Lambda Manager capabilities.

    Note: This demo shows API usage examples. For actual deployment,
    you need valid IAM roles, code packages, and AWS credentials.
    """
    print("AWS Lambda Manager - Advanced Demo")
    print("=" * 70)

    # Initialize manager
    # Uncomment to use: lambda_manager = LambdaManager(region="us-east-1")

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    # Example 1: Create function
    print("\n1️⃣  Create Lambda Function:")
    print("""
    # Simple inline function
    lambda_manager.create_function(
        function_name='hello-world',
        runtime='python3.11',
        handler='index.handler',
        role_arn='arn:aws:iam::123456789012:role/lambda-exec-role',
        code_content='''
def handler(event, context):
    return {
        'statusCode': 200,
        'body': 'Hello from Lambda!'
    }
        ''',
        environment_variables={'ENV': 'production'},
        tags={'Project': 'Demo', 'Team': 'Backend'}
    )
    """)

    # Example 2: Invoke function
    print("\n2️⃣  Invoke Lambda Function:")
    print("""
    # Synchronous invocation
    result = lambda_manager.invoke_function(
        function_name='hello-world',
        payload={'name': 'John', 'action': 'greet'},
        log_type='Tail'  # Include logs in response
    )

    print(f"Status: {result['status_code']}")
    print(f"Response: {result['payload']}")
    print(f"Logs: {result['logs']}")

    # Asynchronous invocation
    lambda_manager.invoke_async(
        function_name='process-data',
        payload={'records': [1, 2, 3]}
    )
    """)

    # Example 3: Event source mapping
    print("\n3️⃣  Create Event Source Mapping:")
    print("""
    # Connect DynamoDB Stream to Lambda
    lambda_manager.create_event_source_mapping(
        function_name='process-dynamodb-records',
        event_source_arn='arn:aws:dynamodb:us-east-1:123456789012:table/MyTable/stream/...',
        batch_size=100,
        starting_position='LATEST'
    )

    # Connect SQS queue to Lambda
    lambda_manager.create_event_source_mapping(
        function_name='process-sqs-messages',
        event_source_arn='arn:aws:sqs:us-east-1:123456789012:my-queue',
        batch_size=10
    )
    """)

    # Example 4: Layers
    print("\n4️⃣  Publish Lambda Layer:")
    print("""
    # Publish shared dependencies as layer
    layer = lambda_manager.publish_layer_version(
        layer_name='common-dependencies',
        description='Shared Python libraries',
        zip_file_path='./layers/dependencies.zip',
        compatible_runtimes=['python3.11', 'python3.10']
    )

    # Use layer in function
    lambda_manager.create_function(
        function_name='my-function',
        runtime='python3.11',
        handler='index.handler',
        role_arn='arn:aws:iam::123456789012:role/lambda-role',
        zip_file_path='./function.zip',
        layers=[layer['layer_version_arn']]
    )
    """)

    # Example 5: Aliases and versions
    print("\n5️⃣  Create Alias for Blue/Green Deployment:")
    print("""
    # Create production alias pointing to version 1
    lambda_manager.create_alias(
        function_name='my-function',
        alias_name='prod',
        function_version='1',
        description='Production environment'
    )

    # Invoke using alias
    result = lambda_manager.invoke_function(
        function_name='my-function:prod',
        payload={'test': True}
    )
    """)

    # Example 6: Monitoring
    print("\n6️⃣  Monitor Function:")
    print("""
    # Get function details
    function = lambda_manager.get_function('my-function')
    print(f"Memory: {function['memory_size']} MB")
    print(f"Timeout: {function['timeout']} seconds")

    # Get recent logs
    logs = lambda_manager.get_function_logs('my-function', limit=20)
    for log_line in logs:
        print(log_line)

    # Get summary
    summary = lambda_manager.get_summary()
    print(f"Total functions: {summary['total_functions']}")
    print(f"Runtimes: {summary['runtimes']}")
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")
    print("\n⚠️  Setup Instructions:")
    print("   1. Configure AWS credentials: aws configure")
    print("   2. Create IAM role with Lambda execution permissions")
    print("   3. Package function code as ZIP file")
    print("   4. Ensure appropriate permissions for event sources")


if __name__ == '__main__':
    demo()
