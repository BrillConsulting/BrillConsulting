"""
Azure Logic Apps
Author: BrillConsulting
Description: Serverless workflow automation service for integrating apps, data, services, and systems
             with visual workflow designer, triggers, actions, and enterprise connectors
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import uuid


class TriggerType(Enum):
    """Workflow trigger types"""
    HTTP = "Http"
    SCHEDULE = "Recurrence"
    EVENT_GRID = "EventGrid"
    SERVICE_BUS = "ServiceBus"
    BLOB_STORAGE = "BlobTrigger"
    MANUAL = "Manual"


class ActionType(Enum):
    """Workflow action types"""
    HTTP = "Http"
    CONDITION = "If"
    SWITCH = "Switch"
    FOR_EACH = "Foreach"
    UNTIL = "Until"
    COMPOSE = "Compose"
    PARSE_JSON = "ParseJson"
    INITIALIZE_VARIABLE = "InitializeVariable"
    SET_VARIABLE = "SetVariable"
    RESPONSE = "Response"
    AZURE_FUNCTION = "AzureFunction"
    SEND_EMAIL = "SendEmail"


class WorkflowStatus(Enum):
    """Workflow status"""
    ENABLED = "Enabled"
    DISABLED = "Disabled"


class RunStatus(Enum):
    """Workflow run status"""
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    WAITING = "Waiting"
    SKIPPED = "Skipped"


class RetryPolicy(Enum):
    """Retry policy types"""
    NONE = "None"
    FIXED = "Fixed"
    EXPONENTIAL = "Exponential"


@dataclass
class Trigger:
    """Workflow trigger configuration"""
    name: str
    trigger_type: TriggerType
    recurrence: Optional[Dict[str, Any]] = None  # For schedule triggers
    conditions: Optional[Dict[str, Any]] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    split_on: Optional[str] = None  # For batch processing


@dataclass
class Action:
    """Workflow action configuration"""
    name: str
    action_type: ActionType
    inputs: Dict[str, Any] = field(default_factory=dict)
    run_after: Dict[str, List[str]] = field(default_factory=dict)  # Dependencies
    retry_policy: Optional[Dict[str, Any]] = None
    timeout: Optional[str] = None


@dataclass
class Condition:
    """Conditional logic"""
    expression: str
    actions_if_true: List[Action] = field(default_factory=list)
    actions_if_false: List[Action] = field(default_factory=list)


@dataclass
class ForEachLoop:
    """For-each loop configuration"""
    items: str  # Expression for items to iterate
    actions: List[Action] = field(default_factory=list)
    is_sequential: bool = False
    degree_of_parallelism: int = 20


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    name: str
    triggers: Dict[str, Trigger] = field(default_factory=dict)
    actions: Dict[str, Action] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowRun:
    """Workflow run instance"""
    run_id: str
    workflow_name: str
    status: RunStatus
    trigger_time: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    trigger_outputs: Dict[str, Any] = field(default_factory=dict)
    action_results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Connection:
    """Managed API connection"""
    name: str
    connection_type: str  # office365, azureblob, servicebus, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    authentication: Dict[str, str] = field(default_factory=dict)


class LogicAppsManager:
    """
    Comprehensive Azure Logic Apps manager

    Features:
    - Workflow definition and creation
    - HTTP, schedule, and event-based triggers
    - Action definitions with Azure service integrations
    - Conditional logic and loops
    - Error handling and retry policies
    - Parallel processing
    - Workflow run management and history
    - Monitoring and diagnostics
    - Managed API connections
    """

    def __init__(
        self,
        resource_group: str,
        subscription_id: str,
        location: str = "eastus"
    ):
        """
        Initialize Logic Apps manager

        Args:
            resource_group: Azure resource group
            subscription_id: Azure subscription ID
            location: Azure region
        """
        self.resource_group = resource_group
        self.subscription_id = subscription_id
        self.location = location
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_status: Dict[str, WorkflowStatus] = {}
        self.workflow_runs: Dict[str, List[WorkflowRun]] = {}
        self.connections: Dict[str, Connection] = {}

    # ===========================================
    # Workflow Management
    # ===========================================

    def create_workflow(
        self,
        workflow_name: str,
        trigger: Trigger,
        actions: List[Action],
        parameters: Optional[Dict[str, Any]] = None
    ) -> WorkflowDefinition:
        """
        Create a Logic App workflow

        Args:
            workflow_name: Workflow name
            trigger: Workflow trigger
            actions: List of actions
            parameters: Workflow parameters

        Returns:
            WorkflowDefinition object
        """
        if workflow_name in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' already exists")

        workflow = WorkflowDefinition(
            name=workflow_name,
            triggers={trigger.name: trigger},
            actions={action.name: action for action in actions},
            parameters=parameters or {}
        )

        self.workflows[workflow_name] = workflow
        self.workflow_status[workflow_name] = WorkflowStatus.ENABLED
        self.workflow_runs[workflow_name] = []

        return workflow

    def update_workflow(
        self,
        workflow_name: str,
        trigger: Optional[Trigger] = None,
        actions: Optional[List[Action]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> WorkflowDefinition:
        """Update an existing workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        workflow = self.workflows[workflow_name]

        if trigger:
            workflow.triggers = {trigger.name: trigger}
        if actions:
            workflow.actions = {action.name: action for action in actions}
        if parameters:
            workflow.parameters.update(parameters)

        return workflow

    def delete_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Delete a workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        del self.workflows[workflow_name]
        del self.workflow_status[workflow_name]
        del self.workflow_runs[workflow_name]

        return {
            "status": "deleted",
            "workflow_name": workflow_name,
            "deleted_at": datetime.now().isoformat()
        }

    def list_workflows(self) -> List[WorkflowDefinition]:
        """List all workflows"""
        return list(self.workflows.values())

    def enable_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Enable a workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        self.workflow_status[workflow_name] = WorkflowStatus.ENABLED

        return {
            "workflow_name": workflow_name,
            "status": "enabled",
            "updated_at": datetime.now().isoformat()
        }

    def disable_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Disable a workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        self.workflow_status[workflow_name] = WorkflowStatus.DISABLED

        return {
            "workflow_name": workflow_name,
            "status": "disabled",
            "updated_at": datetime.now().isoformat()
        }

    # ===========================================
    # Trigger Definitions
    # ===========================================

    def create_http_trigger(
        self,
        name: str = "manual",
        method: str = "POST",
        relative_path: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Trigger:
        """
        Create an HTTP trigger

        Args:
            name: Trigger name
            method: HTTP method (GET, POST, etc.)
            relative_path: Relative URL path
            schema: JSON schema for request body

        Returns:
            Trigger object
        """
        trigger = Trigger(
            name=name,
            trigger_type=TriggerType.HTTP,
            inputs={
                "method": method,
                "relativePath": relative_path or "",
                "schema": schema or {}
            }
        )

        return trigger

    def create_schedule_trigger(
        self,
        name: str = "schedule",
        frequency: str = "Day",
        interval: int = 1,
        start_time: Optional[datetime] = None,
        time_zone: str = "UTC"
    ) -> Trigger:
        """
        Create a schedule (recurrence) trigger

        Args:
            name: Trigger name
            frequency: Frequency (Second, Minute, Hour, Day, Week, Month)
            interval: Interval value
            start_time: Start time
            time_zone: Time zone

        Returns:
            Trigger object
        """
        recurrence = {
            "frequency": frequency,
            "interval": interval,
            "timeZone": time_zone
        }

        if start_time:
            recurrence["startTime"] = start_time.isoformat()

        trigger = Trigger(
            name=name,
            trigger_type=TriggerType.SCHEDULE,
            recurrence=recurrence,
            inputs={"recurrence": recurrence}
        )

        return trigger

    def create_event_grid_trigger(
        self,
        name: str = "event_grid_trigger",
        event_type: str = "Microsoft.Storage.BlobCreated",
        subject_filter: Optional[str] = None
    ) -> Trigger:
        """
        Create an Event Grid trigger

        Args:
            name: Trigger name
            event_type: Event type to subscribe to
            subject_filter: Subject filter pattern

        Returns:
            Trigger object
        """
        trigger = Trigger(
            name=name,
            trigger_type=TriggerType.EVENT_GRID,
            inputs={
                "eventType": event_type,
                "subjectFilter": subject_filter or ""
            }
        )

        return trigger

    # ===========================================
    # Action Definitions
    # ===========================================

    def create_http_action(
        self,
        name: str,
        method: str,
        uri: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        authentication: Optional[Dict[str, str]] = None,
        retry_policy: Optional[str] = None
    ) -> Action:
        """
        Create an HTTP action

        Args:
            name: Action name
            method: HTTP method
            uri: Target URI
            headers: HTTP headers
            body: Request body
            authentication: Authentication config
            retry_policy: Retry policy type

        Returns:
            Action object
        """
        inputs = {
            "method": method,
            "uri": uri
        }

        if headers:
            inputs["headers"] = headers
        if body:
            inputs["body"] = body
        if authentication:
            inputs["authentication"] = authentication

        action = Action(
            name=name,
            action_type=ActionType.HTTP,
            inputs=inputs
        )

        if retry_policy:
            action.retry_policy = {
                "type": retry_policy,
                "count": 4,
                "interval": "PT30S"
            }

        return action

    def create_condition_action(
        self,
        name: str,
        expression: str,
        actions_if_true: List[Action],
        actions_if_false: Optional[List[Action]] = None
    ) -> Action:
        """
        Create a condition (if) action

        Args:
            name: Action name
            expression: Condition expression
            actions_if_true: Actions to run if true
            actions_if_false: Actions to run if false

        Returns:
            Action object
        """
        action = Action(
            name=name,
            action_type=ActionType.CONDITION,
            inputs={
                "expression": expression,
                "actions_if_true": [asdict(a) for a in actions_if_true],
                "actions_if_false": [asdict(a) for a in (actions_if_false or [])]
            }
        )

        return action

    def create_for_each_action(
        self,
        name: str,
        items_expression: str,
        actions: List[Action],
        is_sequential: bool = False,
        degree_of_parallelism: int = 20
    ) -> Action:
        """
        Create a for-each loop action

        Args:
            name: Action name
            items_expression: Expression for items to iterate
            actions: Actions to run for each item
            is_sequential: Run sequentially or in parallel
            degree_of_parallelism: Max parallel executions

        Returns:
            Action object
        """
        action = Action(
            name=name,
            action_type=ActionType.FOR_EACH,
            inputs={
                "foreach": items_expression,
                "actions": [asdict(a) for a in actions],
                "runtimeConfiguration": {
                    "concurrency": {
                        "repetitions": 1 if is_sequential else degree_of_parallelism
                    }
                }
            }
        )

        return action

    def create_parse_json_action(
        self,
        name: str,
        content: str,
        schema: Dict[str, Any]
    ) -> Action:
        """
        Create a parse JSON action

        Args:
            name: Action name
            content: Content to parse (expression)
            schema: JSON schema

        Returns:
            Action object
        """
        action = Action(
            name=name,
            action_type=ActionType.PARSE_JSON,
            inputs={
                "content": content,
                "schema": schema
            }
        )

        return action

    def create_response_action(
        self,
        name: str = "Response",
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None
    ) -> Action:
        """
        Create a response action (for HTTP triggers)

        Args:
            name: Action name
            status_code: HTTP status code
            headers: Response headers
            body: Response body

        Returns:
            Action object
        """
        action = Action(
            name=name,
            action_type=ActionType.RESPONSE,
            inputs={
                "statusCode": status_code,
                "headers": headers or {},
                "body": body
            }
        )

        return action

    def create_compose_action(
        self,
        name: str,
        inputs: Any
    ) -> Action:
        """
        Create a compose action (transform data)

        Args:
            name: Action name
            inputs: Data to compose

        Returns:
            Action object
        """
        action = Action(
            name=name,
            action_type=ActionType.COMPOSE,
            inputs={"compose": inputs}
        )

        return action

    # ===========================================
    # Workflow Execution
    # ===========================================

    def run_workflow(
        self,
        workflow_name: str,
        trigger_data: Optional[Dict[str, Any]] = None
    ) -> WorkflowRun:
        """
        Trigger a workflow run

        Args:
            workflow_name: Workflow name
            trigger_data: Data to pass to trigger

        Returns:
            WorkflowRun object
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        if self.workflow_status[workflow_name] != WorkflowStatus.ENABLED:
            raise ValueError(f"Workflow '{workflow_name}' is disabled")

        run = WorkflowRun(
            run_id=str(uuid.uuid4()),
            workflow_name=workflow_name,
            status=RunStatus.RUNNING,
            trigger_time=datetime.now(),
            start_time=datetime.now(),
            trigger_outputs=trigger_data or {}
        )

        # Simulate workflow execution
        workflow = self.workflows[workflow_name]

        # Execute actions
        for action_name, action in workflow.actions.items():
            run.action_results[action_name] = {
                "status": "Succeeded",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "outputs": {"result": "Action executed successfully"}
            }

        run.status = RunStatus.SUCCEEDED
        run.end_time = datetime.now()

        self.workflow_runs[workflow_name].append(run)

        return run

    def get_run_status(
        self,
        workflow_name: str,
        run_id: str
    ) -> Optional[WorkflowRun]:
        """Get status of a specific workflow run"""
        if workflow_name not in self.workflow_runs:
            return None

        for run in self.workflow_runs[workflow_name]:
            if run.run_id == run_id:
                return run

        return None

    def list_workflow_runs(
        self,
        workflow_name: str,
        top: int = 50,
        status_filter: Optional[RunStatus] = None
    ) -> List[WorkflowRun]:
        """
        List workflow runs

        Args:
            workflow_name: Workflow name
            top: Maximum number of runs to return
            status_filter: Filter by status

        Returns:
            List of WorkflowRun objects
        """
        if workflow_name not in self.workflow_runs:
            return []

        runs = self.workflow_runs[workflow_name]

        if status_filter:
            runs = [run for run in runs if run.status == status_filter]

        return runs[:top]

    def cancel_run(
        self,
        workflow_name: str,
        run_id: str
    ) -> Dict[str, Any]:
        """Cancel a running workflow"""
        run = self.get_run_status(workflow_name, run_id)

        if not run:
            raise ValueError(f"Run '{run_id}' not found")

        if run.status != RunStatus.RUNNING:
            raise ValueError("Can only cancel running workflows")

        run.status = RunStatus.CANCELLED
        run.end_time = datetime.now()

        return {
            "run_id": run_id,
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat()
        }

    def resubmit_run(
        self,
        workflow_name: str,
        run_id: str
    ) -> WorkflowRun:
        """Resubmit a failed workflow run"""
        original_run = self.get_run_status(workflow_name, run_id)

        if not original_run:
            raise ValueError(f"Run '{run_id}' not found")

        # Create new run with same trigger data
        return self.run_workflow(workflow_name, original_run.trigger_outputs)

    # ===========================================
    # Connections Management
    # ===========================================

    def create_connection(
        self,
        connection_name: str,
        connection_type: str,
        parameters: Dict[str, Any],
        authentication: Optional[Dict[str, str]] = None
    ) -> Connection:
        """
        Create a managed API connection

        Args:
            connection_name: Connection name
            connection_type: Connection type (office365, azureblob, etc.)
            parameters: Connection parameters
            authentication: Authentication configuration

        Returns:
            Connection object
        """
        if connection_name in self.connections:
            raise ValueError(f"Connection '{connection_name}' already exists")

        connection = Connection(
            name=connection_name,
            connection_type=connection_type,
            parameters=parameters,
            authentication=authentication or {}
        )

        self.connections[connection_name] = connection

        return connection

    def delete_connection(self, connection_name: str) -> Dict[str, Any]:
        """Delete a connection"""
        if connection_name not in self.connections:
            raise ValueError(f"Connection '{connection_name}' not found")

        del self.connections[connection_name]

        return {
            "status": "deleted",
            "connection_name": connection_name,
            "deleted_at": datetime.now().isoformat()
        }

    def list_connections(self) -> List[Connection]:
        """List all connections"""
        return list(self.connections.values())

    # ===========================================
    # Monitoring and Diagnostics
    # ===========================================

    def get_workflow_metrics(
        self,
        workflow_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Get workflow metrics

        Args:
            workflow_name: Workflow name
            start_time: Start time for metrics
            end_time: End time for metrics

        Returns:
            Metrics dictionary
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        runs = self.workflow_runs[workflow_name]

        # Filter runs by time range
        filtered_runs = [
            run for run in runs
            if start_time <= run.trigger_time <= end_time
        ]

        succeeded = sum(1 for run in filtered_runs if run.status == RunStatus.SUCCEEDED)
        failed = sum(1 for run in filtered_runs if run.status == RunStatus.FAILED)
        cancelled = sum(1 for run in filtered_runs if run.status == RunStatus.CANCELLED)

        return {
            "workflow_name": workflow_name,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_runs": len(filtered_runs),
            "succeeded_runs": succeeded,
            "failed_runs": failed,
            "cancelled_runs": cancelled,
            "success_rate": (succeeded / len(filtered_runs) * 100) if filtered_runs else 0,
            "avg_duration_seconds": 5.2,  # Simulated
            "timestamp": datetime.now().isoformat()
        }

    def get_action_metrics(
        self,
        workflow_name: str,
        action_name: str
    ) -> Dict[str, Any]:
        """Get metrics for a specific action"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        runs = self.workflow_runs[workflow_name]

        succeeded = sum(
            1 for run in runs
            if action_name in run.action_results
            and run.action_results[action_name].get("status") == "Succeeded"
        )

        failed = sum(
            1 for run in runs
            if action_name in run.action_results
            and run.action_results[action_name].get("status") == "Failed"
        )

        return {
            "workflow_name": workflow_name,
            "action_name": action_name,
            "total_executions": succeeded + failed,
            "succeeded": succeeded,
            "failed": failed,
            "success_rate": (succeeded / (succeeded + failed) * 100) if (succeeded + failed) > 0 else 0,
            "avg_duration_ms": 250.5,  # Simulated
            "timestamp": datetime.now().isoformat()
        }

    def get_run_history(
        self,
        workflow_name: str,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Get detailed run history

        Args:
            workflow_name: Workflow name
            run_id: Run ID

        Returns:
            Detailed run history
        """
        run = self.get_run_status(workflow_name, run_id)

        if not run:
            raise ValueError(f"Run '{run_id}' not found")

        return {
            "run_id": run.run_id,
            "workflow_name": run.workflow_name,
            "status": run.status.value,
            "trigger_time": run.trigger_time.isoformat(),
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "duration_seconds": (run.end_time - run.start_time).total_seconds() if run.end_time and run.start_time else 0,
            "trigger_outputs": run.trigger_outputs,
            "action_results": run.action_results,
            "error": run.error
        }

    def export_workflow_definition(
        self,
        workflow_name: str
    ) -> Dict[str, Any]:
        """Export workflow definition as JSON"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        workflow = self.workflows[workflow_name]

        return {
            "definition": {
                "$schema": "https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#",
                "contentVersion": "1.0.0.0",
                "parameters": workflow.parameters,
                "triggers": {
                    name: asdict(trigger)
                    for name, trigger in workflow.triggers.items()
                },
                "actions": {
                    name: asdict(action)
                    for name, action in workflow.actions.items()
                },
                "outputs": workflow.outputs
            }
        }


# ===========================================
# Demo Functions
# ===========================================

def demo_http_workflow():
    """Demonstrate HTTP-triggered workflow"""
    print("=== HTTP Workflow Demo ===\n")

    manager = LogicAppsManager(
        resource_group="my-resource-group",
        subscription_id="subscription-id"
    )

    # Create HTTP trigger
    trigger = manager.create_http_trigger(
        name="manual",
        method="POST",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            }
        }
    )

    # Create actions
    parse_action = manager.create_parse_json_action(
        name="Parse_Request",
        content="@triggerBody()",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            }
        }
    )

    http_action = manager.create_http_action(
        name="Call_API",
        method="POST",
        uri="https://api.example.com/users",
        body="@body('Parse_Request')",
        retry_policy="Exponential"
    )

    response_action = manager.create_response_action(
        name="Response",
        status_code=200,
        body={"status": "success", "message": "User created"}
    )

    # Create workflow
    workflow = manager.create_workflow(
        "UserRegistration",
        trigger,
        [parse_action, http_action, response_action]
    )

    print(f"Created workflow: {workflow.name}")
    print(f"Trigger type: {trigger.trigger_type.value}")
    print(f"Actions: {len(workflow.actions)}\n")

    # Run workflow
    run = manager.run_workflow(
        "UserRegistration",
        {"name": "John Doe", "email": "john@example.com"}
    )

    print(f"Workflow run: {run.run_id}")
    print(f"Status: {run.status.value}")
    print(f"Executed actions: {len(run.action_results)}\n")


def demo_scheduled_workflow():
    """Demonstrate scheduled workflow"""
    print("=== Scheduled Workflow Demo ===\n")

    manager = LogicAppsManager(
        resource_group="my-resource-group",
        subscription_id="subscription-id"
    )

    # Create schedule trigger (daily at midnight)
    trigger = manager.create_schedule_trigger(
        name="daily_schedule",
        frequency="Day",
        interval=1,
        start_time=datetime.now() + timedelta(days=1)
    )

    # Create actions
    http_action = manager.create_http_action(
        name="Get_Daily_Report",
        method="GET",
        uri="https://api.example.com/reports/daily"
    )

    compose_action = manager.create_compose_action(
        name="Transform_Data",
        inputs="@body('Get_Daily_Report')"
    )

    # Create workflow
    workflow = manager.create_workflow(
        "DailyReportProcessing",
        trigger,
        [http_action, compose_action]
    )

    print(f"Created scheduled workflow: {workflow.name}")
    print(f"Frequency: {trigger.recurrence['frequency']}")
    print(f"Interval: {trigger.recurrence['interval']}\n")

    # Simulate run
    run = manager.run_workflow("DailyReportProcessing")
    print(f"Run ID: {run.run_id}")
    print(f"Status: {run.status.value}\n")


def demo_conditional_logic():
    """Demonstrate conditional logic and branching"""
    print("=== Conditional Logic Demo ===\n")

    manager = LogicAppsManager(
        resource_group="my-resource-group",
        subscription_id="subscription-id"
    )

    # Create trigger
    trigger = manager.create_http_trigger(
        name="manual",
        method="POST"
    )

    # Create actions for different paths
    success_action = manager.create_http_action(
        name="Send_Success_Email",
        method="POST",
        uri="https://api.example.com/email/success"
    )

    error_action = manager.create_http_action(
        name="Send_Error_Email",
        method="POST",
        uri="https://api.example.com/email/error"
    )

    # Create condition
    condition = manager.create_condition_action(
        name="Check_Status",
        expression="@equals(triggerBody()['status'], 'success')",
        actions_if_true=[success_action],
        actions_if_false=[error_action]
    )

    # Create workflow
    workflow = manager.create_workflow(
        "ConditionalWorkflow",
        trigger,
        [condition]
    )

    print(f"Created workflow: {workflow.name}")
    print(f"Condition: Check status and branch\n")

    # Run with success
    run1 = manager.run_workflow(
        "ConditionalWorkflow",
        {"status": "success"}
    )
    print(f"Run 1 (success): {run1.status.value}")

    # Run with error
    run2 = manager.run_workflow(
        "ConditionalWorkflow",
        {"status": "error"}
    )
    print(f"Run 2 (error): {run2.status.value}\n")


def demo_foreach_loop():
    """Demonstrate for-each loop processing"""
    print("=== For-Each Loop Demo ===\n")

    manager = LogicAppsManager(
        resource_group="my-resource-group",
        subscription_id="subscription-id"
    )

    # Create trigger
    trigger = manager.create_http_trigger(
        name="manual",
        method="POST"
    )

    # Create action to execute for each item
    process_action = manager.create_http_action(
        name="Process_Item",
        method="POST",
        uri="https://api.example.com/process",
        body="@item()"
    )

    # Create for-each loop
    foreach_action = manager.create_for_each_action(
        name="Process_All_Items",
        items_expression="@triggerBody()['items']",
        actions=[process_action],
        is_sequential=False,
        degree_of_parallelism=10
    )

    # Create workflow
    workflow = manager.create_workflow(
        "BatchProcessing",
        trigger,
        [foreach_action]
    )

    print(f"Created workflow: {workflow.name}")
    print(f"For-each action: {foreach_action.name}")
    print(f"Parallel processing: Yes\n")

    # Run with multiple items
    items = [{"id": i, "value": i * 10} for i in range(1, 11)]
    run = manager.run_workflow(
        "BatchProcessing",
        {"items": items}
    )

    print(f"Run ID: {run.run_id}")
    print(f"Status: {run.status.value}")
    print(f"Items processed: {len(items)}\n")


def demo_workflow_management():
    """Demonstrate workflow management operations"""
    print("=== Workflow Management Demo ===\n")

    manager = LogicAppsManager(
        resource_group="my-resource-group",
        subscription_id="subscription-id"
    )

    # Create multiple workflows
    for i in range(1, 4):
        trigger = manager.create_http_trigger(name=f"trigger{i}")
        action = manager.create_http_action(
            name=f"action{i}",
            method="GET",
            uri=f"https://api.example.com/endpoint{i}"
        )
        manager.create_workflow(f"Workflow{i}", trigger, [action])

    # List workflows
    workflows = manager.list_workflows()
    print(f"Total workflows: {len(workflows)}")
    for wf in workflows:
        print(f"  - {wf.name}")
    print()

    # Disable a workflow
    result = manager.disable_workflow("Workflow2")
    print(f"Disabled workflow: {result['workflow_name']}")
    print(f"Status: {result['status']}\n")

    # Enable it again
    result = manager.enable_workflow("Workflow2")
    print(f"Enabled workflow: {result['workflow_name']}")
    print(f"Status: {result['status']}\n")

    # Delete a workflow
    result = manager.delete_workflow("Workflow3")
    print(f"Deleted workflow: {result['workflow_name']}\n")

    # List workflows again
    workflows = manager.list_workflows()
    print(f"Remaining workflows: {len(workflows)}\n")


def demo_run_history():
    """Demonstrate run history and monitoring"""
    print("=== Run History and Monitoring Demo ===\n")

    manager = LogicAppsManager(
        resource_group="my-resource-group",
        subscription_id="subscription-id"
    )

    # Create workflow
    trigger = manager.create_http_trigger()
    action = manager.create_http_action(
        name="Call_API",
        method="GET",
        uri="https://api.example.com/data"
    )
    manager.create_workflow("MonitoredWorkflow", trigger, [action])

    # Execute multiple runs
    run_ids = []
    for i in range(5):
        run = manager.run_workflow("MonitoredWorkflow", {"iteration": i + 1})
        run_ids.append(run.run_id)

    print(f"Executed {len(run_ids)} workflow runs\n")

    # List runs
    runs = manager.list_workflow_runs("MonitoredWorkflow")
    print(f"Total runs: {len(runs)}")
    for run in runs:
        print(f"  - {run.run_id}: {run.status.value}")
    print()

    # Get detailed history for first run
    history = manager.get_run_history("MonitoredWorkflow", run_ids[0])
    print(f"Run details:")
    print(f"  Run ID: {history['run_id']}")
    print(f"  Status: {history['status']}")
    print(f"  Duration: {history['duration_seconds']}s")
    print(f"  Actions executed: {len(history['action_results'])}\n")

    # Get workflow metrics
    metrics = manager.get_workflow_metrics(
        "MonitoredWorkflow",
        datetime.now() - timedelta(hours=1),
        datetime.now()
    )
    print("Workflow metrics:")
    print(f"  Total runs: {metrics['total_runs']}")
    print(f"  Succeeded: {metrics['succeeded_runs']}")
    print(f"  Failed: {metrics['failed_runs']}")
    print(f"  Success rate: {metrics['success_rate']:.1f}%\n")


def demo_connections():
    """Demonstrate managed API connections"""
    print("=== Managed Connections Demo ===\n")

    manager = LogicAppsManager(
        resource_group="my-resource-group",
        subscription_id="subscription-id"
    )

    # Create connections
    office365 = manager.create_connection(
        "office365_connection",
        "office365",
        {"userPrincipalName": "user@company.com"},
        {"type": "ManagedServiceIdentity"}
    )
    print(f"Created connection: {office365.name}")
    print(f"Type: {office365.connection_type}\n")

    blob_storage = manager.create_connection(
        "blob_connection",
        "azureblob",
        {
            "accountName": "mystorageaccount",
            "accessKey": "access-key-here"
        }
    )
    print(f"Created connection: {blob_storage.name}")
    print(f"Type: {blob_storage.connection_type}\n")

    # List connections
    connections = manager.list_connections()
    print(f"Total connections: {len(connections)}")
    for conn in connections:
        print(f"  - {conn.name} ({conn.connection_type})")
    print()


if __name__ == "__main__":
    print("Azure Logic Apps - Advanced Implementation")
    print("=" * 60)
    print()

    # Run all demos
    demo_http_workflow()
    demo_scheduled_workflow()
    demo_conditional_logic()
    demo_foreach_loop()
    demo_workflow_management()
    demo_run_history()
    demo_connections()

    print("=" * 60)
    print("All demos completed successfully!")
