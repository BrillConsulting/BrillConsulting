"""
Agent Workflow Framework
=========================

Business process automation with intelligent agents:
- Workflow definition and execution
- Task routing and assignment
- State management
- Exception handling
- Human-in-the-loop

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random


class StepType(Enum):
    """Workflow step types."""
    AUTOMATED = "automated"
    HUMAN_APPROVAL = "human_approval"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"


class StepStatus(Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"


@dataclass
class WorkflowStep:
    """Represents a workflow step."""
    name: str
    type: StepType
    status: StepStatus = StepStatus.PENDING
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class WorkflowDefinition:
    """Defines a workflow."""
    name: str
    steps: List[Dict[str, str]]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class WorkflowInstance:
    """Running instance of workflow."""
    instance_id: str
    workflow_name: str
    steps: List[WorkflowStep]
    current_step: int = 0
    status: str = "running"
    input_data: Dict = field(default_factory=dict)
    output_data: Dict = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


class WorkflowEngine:
    """Workflow execution engine."""

    def __init__(self, name: str = "WorkflowEngine"):
        """Initialize workflow engine."""
        self.name = name
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.instances: Dict[str, WorkflowInstance] = {}

        print(f"⚙️  Workflow Engine '{name}' initialized")

    def create_workflow(
        self,
        name: str,
        steps: List[Dict[str, str]]
    ) -> WorkflowDefinition:
        """Define a new workflow."""
        print(f"\n📋 Creating workflow: {name}")

        workflow = WorkflowDefinition(name=name, steps=steps)
        self.workflows[name] = workflow

        print(f"   ✓ Workflow created with {len(steps)} steps")
        return workflow

    def instantiate_workflow(
        self,
        workflow_name: str,
        input_data: Dict[str, Any]
    ) -> WorkflowInstance:
        """Create workflow instance."""
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        instance_id = f"wf_{len(self.instances) + 1}"

        # Create steps from definition
        steps = [
            WorkflowStep(
                name=step["name"],
                type=StepType(step["type"])
            )
            for step in workflow.steps
        ]

        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_name=workflow_name,
            steps=steps,
            input_data=input_data
        )

        self.instances[instance_id] = instance

        print(f"   ✓ Instance created: {instance_id}")
        return instance

    def execute_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> bool:
        """Execute a single workflow step."""
        print(f"      Executing: {step.name}")
        step.status = StepStatus.RUNNING

        if step.type == StepType.AUTOMATED:
            # Simulate automated execution
            success = random.random() > 0.1  # 90% success rate
            if success:
                step.status = StepStatus.COMPLETED
                step.output_data = {"status": "success", "result": f"Completed {step.name}"}
                print(f"         ✓ Completed")
                return True
            else:
                step.status = StepStatus.FAILED
                step.error = "Execution failed"
                print(f"         ✗ Failed")
                return False

        elif step.type == StepType.HUMAN_APPROVAL:
            # Simulate human approval (auto-approve for demo)
            step.status = StepStatus.WAITING_APPROVAL
            print(f"         ⏸  Waiting for approval...")

            # Auto-approve after simulation
            approved = random.random() > 0.2  # 80% approval rate
            if approved:
                step.status = StepStatus.COMPLETED
                step.output_data = {"approved": True}
                print(f"         ✓ Approved")
                return True
            else:
                step.status = StepStatus.FAILED
                step.error = "Not approved"
                print(f"         ✗ Rejected")
                return False

        return True


class WorkflowAgent:
    """Agent that executes workflows."""

    def __init__(self, name: str, engine: WorkflowEngine):
        """Initialize workflow agent."""
        self.name = name
        self.engine = engine

        print(f"🤖 Workflow Agent '{name}' created")

    def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute complete workflow."""
        print(f"\n{'='*60}")
        print(f"Executing Workflow: {workflow_id}")
        print(f"{'='*60}")

        # Create instance
        instance = self.engine.instantiate_workflow(workflow_id, input_data)

        # Execute steps
        for i, step in enumerate(instance.steps, 1):
            print(f"\n   Step {i}/{len(instance.steps)}: {step.name}")

            success = self.engine.execute_step(instance, step)

            if not success:
                instance.status = "failed"
                print(f"\n❌ Workflow failed at step: {step.name}")
                return {
                    "instance_id": instance.instance_id,
                    "status": "failed",
                    "failed_step": step.name,
                    "error": step.error
                }

            instance.current_step = i

        # Complete workflow
        instance.status = "completed"
        instance.completed_at = datetime.now().isoformat()

        print(f"\n✅ Workflow completed successfully")

        return {
            "instance_id": instance.instance_id,
            "status": "completed",
            "output": instance.output_data
        }

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status."""
        instance = self.engine.instances.get(workflow_id)

        if not instance:
            return {"error": "Workflow not found"}

        completed_steps = sum(
            1 for step in instance.steps
            if step.status == StepStatus.COMPLETED
        )

        return {
            "instance_id": instance.instance_id,
            "workflow_name": instance.workflow_name,
            "status": instance.status,
            "progress": f"{completed_steps}/{len(instance.steps)}",
            "current_step": instance.steps[instance.current_step].name
            if instance.current_step < len(instance.steps) else "Complete"
        }


def demo():
    """Demonstrate workflow agents."""
    print("=" * 60)
    print("Agent Workflow Framework Demo")
    print("=" * 60)

    # Create engine
    engine = WorkflowEngine(name="ProcessEngine")

    # Define workflow
    workflow = engine.create_workflow(
        name="OrderProcessing",
        steps=[
            {"name": "validate_order", "type": "automated"},
            {"name": "check_inventory", "type": "automated"},
            {"name": "approve_payment", "type": "human_approval"},
            {"name": "ship_order", "type": "automated"}
        ]
    )

    # Create agent
    agent = WorkflowAgent(name="ProcessBot", engine=engine)

    # Execute workflow
    result = agent.execute_workflow(
        workflow_id="OrderProcessing",
        input_data={"order_id": "ORD-123", "amount": 99.99}
    )

    # Check status
    print("\n" + "=" * 60)
    import json
    status = agent.get_workflow_status(result['instance_id'])
    print("Workflow Status:")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    demo()
