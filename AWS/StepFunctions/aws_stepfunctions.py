"""
AWS Step Functions
==================

Serverless workflow orchestration using Amazon States Language.

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


class StepFunctionsManager:
    """
    Advanced AWS Step Functions Management System

    Provides comprehensive Step Functions operations including:
    - State machine creation (Standard and Express)
    - Execution management (start, stop, describe)
    - State machine definitions with Amazon States Language
    - Activity workers
    - Execution history and logging
    - Error handling and retry strategies
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize Step Functions Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.sfn_client = session.client('stepfunctions', region_name=region)
            self.region = region
            logger.info(f"Step Functions Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing Step Functions Manager: {e}")
            raise

    # ==================== State Machine Management ====================

    def create_state_machine(
        self,
        name: str,
        definition: Dict[str, Any],
        role_arn: str,
        state_machine_type: str = "STANDARD",
        logging_configuration: Optional[Dict[str, Any]] = None,
        tags: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create state machine.

        Args:
            name: State machine name
            definition: Amazon States Language definition (dict)
            role_arn: IAM role ARN for execution
            state_machine_type: 'STANDARD' or 'EXPRESS'
            logging_configuration: CloudWatch logging config
            tags: Resource tags
        """
        try:
            logger.info(f"Creating state machine: {name}")

            params = {
                'name': name,
                'definition': json.dumps(definition),
                'roleArn': role_arn,
                'type': state_machine_type
            }

            if logging_configuration:
                params['loggingConfiguration'] = logging_configuration

            if tags:
                params['tags'] = tags

            response = self.sfn_client.create_state_machine(**params)

            logger.info(f"✓ State machine created: {name}")
            return {
                'state_machine_arn': response['stateMachineArn'],
                'creation_date': response['creationDate'].isoformat()
            }

        except ClientError as e:
            logger.error(f"Error creating state machine: {e}")
            raise

    def describe_state_machine(self, state_machine_arn: str) -> Dict[str, Any]:
        """Describe state machine."""
        try:
            response = self.sfn_client.describe_state_machine(
                stateMachineArn=state_machine_arn
            )

            return {
                'name': response['name'],
                'state_machine_arn': response['stateMachineArn'],
                'role_arn': response['roleArn'],
                'type': response['type'],
                'status': response['status'],
                'creation_date': response['creationDate'].isoformat(),
                'definition': json.loads(response['definition'])
            }

        except ClientError as e:
            logger.error(f"Error describing state machine: {e}")
            raise

    def update_state_machine(
        self,
        state_machine_arn: str,
        definition: Optional[Dict[str, Any]] = None,
        role_arn: Optional[str] = None,
        logging_configuration: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update state machine."""
        try:
            logger.info(f"Updating state machine: {state_machine_arn}")

            params = {'stateMachineArn': state_machine_arn}

            if definition:
                params['definition'] = json.dumps(definition)
            if role_arn:
                params['roleArn'] = role_arn
            if logging_configuration:
                params['loggingConfiguration'] = logging_configuration

            response = self.sfn_client.update_state_machine(**params)

            logger.info(f"✓ State machine updated")
            return {
                'update_date': response['updateDate'].isoformat()
            }

        except ClientError as e:
            logger.error(f"Error updating state machine: {e}")
            raise

    def list_state_machines(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """List state machines."""
        try:
            response = self.sfn_client.list_state_machines(maxResults=max_results)

            machines = [
                {
                    'name': sm['name'],
                    'state_machine_arn': sm['stateMachineArn'],
                    'type': sm['type'],
                    'creation_date': sm['creationDate'].isoformat()
                }
                for sm in response.get('stateMachines', [])
            ]

            logger.info(f"Found {len(machines)} state machine(s)")
            return machines

        except ClientError as e:
            logger.error(f"Error listing state machines: {e}")
            raise

    def delete_state_machine(self, state_machine_arn: str) -> None:
        """Delete state machine."""
        try:
            self.sfn_client.delete_state_machine(stateMachineArn=state_machine_arn)
            logger.info(f"✓ State machine deleted: {state_machine_arn}")

        except ClientError as e:
            logger.error(f"Error deleting state machine: {e}")
            raise

    # ==================== Execution Management ====================

    def start_execution(
        self,
        state_machine_arn: str,
        input_data: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start execution.

        Args:
            state_machine_arn: State machine ARN
            input_data: Input JSON for execution
            name: Execution name (optional, auto-generated if not provided)
        """
        try:
            logger.info(f"Starting execution: {state_machine_arn}")

            params = {'stateMachineArn': state_machine_arn}

            if input_data:
                params['input'] = json.dumps(input_data)

            if name:
                params['name'] = name

            response = self.sfn_client.start_execution(**params)

            logger.info(f"✓ Execution started: {response['executionArn']}")
            return {
                'execution_arn': response['executionArn'],
                'start_date': response['startDate'].isoformat()
            }

        except ClientError as e:
            logger.error(f"Error starting execution: {e}")
            raise

    def describe_execution(self, execution_arn: str) -> Dict[str, Any]:
        """Describe execution."""
        try:
            response = self.sfn_client.describe_execution(executionArn=execution_arn)

            result = {
                'execution_arn': response['executionArn'],
                'state_machine_arn': response['stateMachineArn'],
                'name': response['name'],
                'status': response['status'],
                'start_date': response['startDate'].isoformat()
            }

            if 'stopDate' in response:
                result['stop_date'] = response['stopDate'].isoformat()

            if 'input' in response:
                result['input'] = json.loads(response['input'])

            if 'output' in response:
                result['output'] = json.loads(response['output'])

            return result

        except ClientError as e:
            logger.error(f"Error describing execution: {e}")
            raise

    def stop_execution(
        self,
        execution_arn: str,
        error: Optional[str] = None,
        cause: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stop execution."""
        try:
            logger.info(f"Stopping execution: {execution_arn}")

            params = {'executionArn': execution_arn}
            if error:
                params['error'] = error
            if cause:
                params['cause'] = cause

            response = self.sfn_client.stop_execution(**params)

            logger.info(f"✓ Execution stopped")
            return {
                'stop_date': response['stopDate'].isoformat()
            }

        except ClientError as e:
            logger.error(f"Error stopping execution: {e}")
            raise

    def list_executions(
        self,
        state_machine_arn: str,
        status_filter: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List executions.

        Args:
            state_machine_arn: State machine ARN
            status_filter: 'RUNNING', 'SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED'
            max_results: Max results to return
        """
        try:
            params = {
                'stateMachineArn': state_machine_arn,
                'maxResults': max_results
            }

            if status_filter:
                params['statusFilter'] = status_filter

            response = self.sfn_client.list_executions(**params)

            executions = [
                {
                    'execution_arn': ex['executionArn'],
                    'name': ex['name'],
                    'status': ex['status'],
                    'start_date': ex['startDate'].isoformat(),
                    'stop_date': ex.get('stopDate', datetime.now()).isoformat() if 'stopDate' in ex else None
                }
                for ex in response.get('executions', [])
            ]

            logger.info(f"Found {len(executions)} execution(s)")
            return executions

        except ClientError as e:
            logger.error(f"Error listing executions: {e}")
            raise

    def get_execution_history(
        self,
        execution_arn: str,
        max_results: int = 100,
        reverse_order: bool = False
    ) -> List[Dict[str, Any]]:
        """Get execution history."""
        try:
            response = self.sfn_client.get_execution_history(
                executionArn=execution_arn,
                maxResults=max_results,
                reverseOrder=reverse_order
            )

            events = [
                {
                    'event_id': event['id'],
                    'timestamp': event['timestamp'].isoformat(),
                    'type': event['type']
                }
                for event in response.get('events', [])
            ]

            logger.info(f"Retrieved {len(events)} history event(s)")
            return events

        except ClientError as e:
            logger.error(f"Error getting execution history: {e}")
            raise

    # ==================== Activities ====================

    def create_activity(self, name: str, tags: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Create activity for task workers."""
        try:
            logger.info(f"Creating activity: {name}")

            params = {'name': name}
            if tags:
                params['tags'] = tags

            response = self.sfn_client.create_activity(**params)

            logger.info(f"✓ Activity created: {name}")
            return {
                'activity_arn': response['activityArn'],
                'creation_date': response['creationDate'].isoformat()
            }

        except ClientError as e:
            logger.error(f"Error creating activity: {e}")
            raise

    def list_activities(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """List activities."""
        try:
            response = self.sfn_client.list_activities(maxResults=max_results)

            activities = [
                {
                    'activity_arn': act['activityArn'],
                    'name': act['name'],
                    'creation_date': act['creationDate'].isoformat()
                }
                for act in response.get('activities', [])
            ]

            logger.info(f"Found {len(activities)} activit(ies)")
            return activities

        except ClientError as e:
            logger.error(f"Error listing activities: {e}")
            raise

    def delete_activity(self, activity_arn: str) -> None:
        """Delete activity."""
        try:
            self.sfn_client.delete_activity(activityArn=activity_arn)
            logger.info(f"✓ Activity deleted: {activity_arn}")

        except ClientError as e:
            logger.error(f"Error deleting activity: {e}")
            raise

    # ==================== Monitoring ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get Step Functions summary."""
        try:
            state_machines = self.list_state_machines()

            return {
                'region': self.region,
                'state_machines': len(state_machines),
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of Step Functions Manager capabilities."""
    print("AWS Step Functions Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    print("\n1️⃣  Create State Machine:")
    print("""
    sfn = StepFunctionsManager(region='us-east-1')

    # Define workflow
    definition = {
        'Comment': 'Hello World workflow',
        'StartAt': 'HelloWorld',
        'States': {
            'HelloWorld': {
                'Type': 'Task',
                'Resource': 'arn:aws:lambda:us-east-1:123456789012:function:HelloWorld',
                'End': True
            }
        }
    }

    # Create state machine
    state_machine = sfn.create_state_machine(
        name='HelloWorldStateMachine',
        definition=definition,
        role_arn='arn:aws:iam::123456789012:role/StepFunctionsRole'
    )
    """)

    print("\n2️⃣  Execute Workflow:")
    print("""
    # Start execution
    execution = sfn.start_execution(
        state_machine_arn='arn:aws:states:us-east-1:123456789012:stateMachine:HelloWorld',
        input_data={'message': 'Hello from Step Functions!'},
        name='execution-001'
    )

    # Check status
    status = sfn.describe_execution(execution['execution_arn'])
    print(f"Status: {status['status']}")
    """)

    print("\n3️⃣  Complex Workflow with Error Handling:")
    print("""
    definition = {
        'StartAt': 'ProcessData',
        'States': {
            'ProcessData': {
                'Type': 'Task',
                'Resource': 'arn:aws:lambda:...:function:process',
                'Retry': [{
                    'ErrorEquals': ['States.ALL'],
                    'IntervalSeconds': 2,
                    'MaxAttempts': 3
                }],
                'Catch': [{
                    'ErrorEquals': ['States.ALL'],
                    'Next': 'HandleError'
                }],
                'Next': 'Success'
            },
            'HandleError': {
                'Type': 'Task',
                'Resource': 'arn:aws:lambda:...:function:error-handler',
                'End': True
            },
            'Success': {
                'Type': 'Succeed'
            }
        }
    }
    """)

    print("\n4️⃣  Parallel Execution:")
    print("""
    definition = {
        'StartAt': 'ParallelProcessing',
        'States': {
            'ParallelProcessing': {
                'Type': 'Parallel',
                'Branches': [
                    {
                        'StartAt': 'Task1',
                        'States': {
                            'Task1': {'Type': 'Task', 'Resource': '...', 'End': True}
                        }
                    },
                    {
                        'StartAt': 'Task2',
                        'States': {
                            'Task2': {'Type': 'Task', 'Resource': '...', 'End': True}
                        }
                    }
                ],
                'End': True
            }
        }
    }
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
