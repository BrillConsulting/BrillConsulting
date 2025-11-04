"""
Agent Orchestration Framework
==============================

Orchestrate complex agent workflows and execution patterns:
- Workflow definition (sequential, parallel, conditional)
- Agent execution management
- State management across steps
- Error handling and retries
- Workflow monitoring

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import json


class ExecutionMode(Enum):
    """Workflow execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


class WorkflowStep:
    """Individual step in a workflow."""

    def __init__(self, step_id: str, agent_id: str, action: Callable,
                 params: Optional[Dict] = None, condition: Optional[Callable] = None):
        self.id = step_id
        self.agent_id = agent_id
        self.action = action
        self.params = params or {}
        self.condition = condition
        self.status = "pending"
        self.result = None
        self.error = None
        self.retries = 0
        self.max_retries = 3

    def should_execute(self, context: Dict) -> bool:
        """Check if step should execute based on condition."""
        if self.condition is None:
            return True
        return self.condition(context)

    def execute(self, context: Dict) -> Dict:
        """Execute the step."""
        if not self.should_execute(context):
            return {
                "step_id": self.id,
                "status": "skipped",
                "reason": "condition_not_met"
            }

        self.status = "running"

        try:
            # Execute action with context
            result = self.action(**self.params, context=context)
            self.status = "completed"
            self.result = result

            return {
                "step_id": self.id,
                "status": "completed",
                "result": result
            }

        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            self.retries += 1

            return {
                "step_id": self.id,
                "status": "failed",
                "error": str(e),
                "retries": self.retries
            }

    def to_dict(self) -> Dict:
        """Convert step to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "retries": self.retries
        }


class Workflow:
    """Workflow definition."""

    def __init__(self, workflow_id: str, name: str, mode: ExecutionMode = ExecutionMode.SEQUENTIAL):
        self.id = workflow_id
        self.name = name
        self.mode = mode
        self.steps: List[WorkflowStep] = []
        self.status = "created"
        self.context = {}

    def add_step(self, step: WorkflowStep):
        """Add step to workflow."""
        self.steps.append(step)

    def to_dict(self) -> Dict:
        """Convert workflow to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode.value,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status,
            "context": self.context
        }


class AgentOrchestrator:
    """Orchestrate agent workflows."""

    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.agents: Dict[str, Any] = {}
        self.execution_history = []

    def register_agent(self, agent_id: str, agent: Any):
        """Register an agent."""
        self.agents[agent_id] = agent
        print(f"✓ Agent registered: {agent_id}")

    def create_workflow(self, workflow_id: str, name: str,
                       mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> Workflow:
        """Create a new workflow."""
        workflow = Workflow(workflow_id, name, mode)
        self.workflows[workflow_id] = workflow
        print(f"✓ Workflow created: {name} ({mode.value})")
        return workflow

    def execute_workflow(self, workflow_id: str) -> Dict:
        """Execute a workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {"error": f"Workflow not found: {workflow_id}"}

        print(f"\n{'='*60}")
        print(f"🚀 Executing Workflow: {workflow.name}")
        print(f"   Mode: {workflow.mode.value}")
        print(f"   Steps: {len(workflow.steps)}")
        print(f"{'='*60}")

        workflow.status = "running"
        start_time = datetime.now()

        if workflow.mode == ExecutionMode.SEQUENTIAL:
            results = self._execute_sequential(workflow)
        elif workflow.mode == ExecutionMode.PARALLEL:
            results = self._execute_parallel(workflow)
        elif workflow.mode == ExecutionMode.CONDITIONAL:
            results = self._execute_conditional(workflow)
        else:
            results = {"error": "Unknown execution mode"}

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Determine final status
        all_completed = all(r.get("status") in ["completed", "skipped"] for r in results)
        workflow.status = "completed" if all_completed else "failed"

        execution_result = {
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "status": workflow.status,
            "duration_seconds": duration,
            "steps_executed": len(results),
            "steps_completed": sum(1 for r in results if r.get("status") == "completed"),
            "steps_failed": sum(1 for r in results if r.get("status") == "failed"),
            "steps_skipped": sum(1 for r in results if r.get("status") == "skipped"),
            "results": results,
            "timestamp": end_time.isoformat()
        }

        self.execution_history.append(execution_result)

        print(f"\n{'='*60}")
        print(f"✓ Workflow {workflow.status.upper()}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Completed: {execution_result['steps_completed']}/{len(results)}")
        print(f"{'='*60}\n")

        return execution_result

    def _execute_sequential(self, workflow: Workflow) -> List[Dict]:
        """Execute steps sequentially."""
        results = []

        for i, step in enumerate(workflow.steps, 1):
            print(f"\nStep {i}/{len(workflow.steps)}: {step.id}")
            print("-" * 60)

            result = step.execute(workflow.context)
            results.append(result)

            # Update context with result
            if result["status"] == "completed":
                workflow.context[step.id] = result["result"]
                print(f"✓ Step completed")
            elif result["status"] == "skipped":
                print(f"⊘ Step skipped")
            else:
                print(f"✗ Step failed: {result.get('error')}")

                # Retry if possible
                if step.retries < step.max_retries:
                    print(f"🔄 Retrying... (attempt {step.retries + 1}/{step.max_retries})")
                    result = step.execute(workflow.context)
                    results[-1] = result

                    if result["status"] == "completed":
                        workflow.context[step.id] = result["result"]
                        continue

                # Stop on critical failure
                if step.params.get("critical", False):
                    print(f"⚠️  Critical step failed, stopping workflow")
                    break

        return results

    def _execute_parallel(self, workflow: Workflow) -> List[Dict]:
        """Execute steps in parallel (simulated)."""
        print("\n⚡ Executing steps in parallel...")
        print("-" * 60)

        results = []
        for step in workflow.steps:
            result = step.execute(workflow.context)
            results.append(result)

            # Update context with result
            if result["status"] == "completed":
                workflow.context[step.id] = result["result"]

        return results

    def _execute_conditional(self, workflow: Workflow) -> List[Dict]:
        """Execute steps based on conditions."""
        return self._execute_sequential(workflow)  # Uses conditions in steps

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get workflow status."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None

        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status,
            "total_steps": len(workflow.steps),
            "completed_steps": sum(1 for s in workflow.steps if s.status == "completed"),
            "failed_steps": sum(1 for s in workflow.steps if s.status == "failed")
        }

    def get_execution_history(self) -> List[Dict]:
        """Get execution history."""
        return self.execution_history


def demo():
    """Demo agent orchestration."""
    print("Agent Orchestration Demo")
    print("=" * 60)

    orchestrator = AgentOrchestrator()

    # Define sample agent actions
    def analyze_data(**kwargs) -> Dict:
        context = kwargs.get("context", {})
        return {"records_analyzed": 1000, "insights": ["trend_up", "anomaly_detected"]}

    def generate_report(**kwargs) -> Dict:
        context = kwargs.get("context", {})
        analysis = context.get("analyze_data", {})
        return {"report_id": "RPT_001", "insights_included": len(analysis.get("insights", []))}

    def send_email(**kwargs) -> Dict:
        context = kwargs.get("context", {})
        return {"email_sent": True, "recipients": 3}

    def validate_results(**kwargs) -> Dict:
        return {"validation_passed": True, "score": 0.95}

    # 1. Sequential Workflow
    print("\n1. Sequential Workflow")
    print("-" * 60)

    seq_workflow = orchestrator.create_workflow(
        "workflow_1",
        "Data Analysis Pipeline",
        ExecutionMode.SEQUENTIAL
    )

    seq_workflow.add_step(WorkflowStep("analyze_data", "agent_1", analyze_data))
    seq_workflow.add_step(WorkflowStep("generate_report", "agent_2", generate_report))
    seq_workflow.add_step(WorkflowStep("send_email", "agent_3", send_email))

    result1 = orchestrator.execute_workflow("workflow_1")

    # 2. Parallel Workflow
    print("\n2. Parallel Workflow")
    print("-" * 60)

    parallel_workflow = orchestrator.create_workflow(
        "workflow_2",
        "Multi-Source Data Collection",
        ExecutionMode.PARALLEL
    )

    parallel_workflow.add_step(WorkflowStep("fetch_source1", "agent_1", analyze_data))
    parallel_workflow.add_step(WorkflowStep("fetch_source2", "agent_2", analyze_data))
    parallel_workflow.add_step(WorkflowStep("fetch_source3", "agent_3", analyze_data))

    result2 = orchestrator.execute_workflow("workflow_2")

    # 3. Conditional Workflow
    print("\n3. Conditional Workflow")
    print("-" * 60)

    cond_workflow = orchestrator.create_workflow(
        "workflow_3",
        "Conditional Processing",
        ExecutionMode.CONDITIONAL
    )

    def check_quality(context: Dict) -> bool:
        # Only send email if validation passed
        validation = context.get("validate_results", {})
        return validation.get("validation_passed", False)

    cond_workflow.add_step(WorkflowStep("analyze_data", "agent_1", analyze_data))
    cond_workflow.add_step(WorkflowStep("validate_results", "agent_2", validate_results))
    cond_workflow.add_step(WorkflowStep(
        "send_email",
        "agent_3",
        send_email,
        condition=check_quality
    ))

    result3 = orchestrator.execute_workflow("workflow_3")

    # 4. Execution Summary
    print("\n4. Execution History")
    print("-" * 60)

    history = orchestrator.get_execution_history()
    for i, execution in enumerate(history, 1):
        print(f"\n  Execution {i}:")
        print(f"    Workflow: {execution['workflow_name']}")
        print(f"    Status: {execution['status']}")
        print(f"    Duration: {execution['duration_seconds']:.2f}s")
        print(f"    Steps: {execution['steps_completed']}/{execution['steps_executed']} completed")

    print("\n✓ Agent Orchestration Demo Complete!")


if __name__ == '__main__':
    demo()
