"""
Hierarchical Agents Framework
==============================

Multi-level agent hierarchies with delegation and supervision:
- Manager-worker relationships
- Task delegation and escalation
- Performance monitoring
- Organizational structures
- Team coordination

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class AgentLevel(Enum):
    """Hierarchical levels in organization."""
    EXECUTIVE = 1
    MANAGER = 2
    SPECIALIST = 3
    WORKER = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a task in the hierarchy."""
    id: str
    description: str
    complexity: float
    required_skills: List[str]
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    parent_task: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    result: Optional[Any] = None


class HierarchicalAgent:
    """Base agent in hierarchical organization."""

    def __init__(
        self,
        name: str,
        level: AgentLevel,
        skills: Optional[List[str]] = None,
        capacity: int = 5
    ):
        """Initialize hierarchical agent."""
        self.name = name
        self.level = level
        self.skills = skills or []
        self.capacity = capacity
        self.current_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.parent: Optional['ManagerAgent'] = None
        self.performance_score = 1.0

    def can_handle(self, task: Task) -> bool:
        """Check if agent can handle task."""
        # Check capacity
        if len(self.current_tasks) >= self.capacity:
            return False

        # Check skills
        required_skills = set(task.required_skills)
        agent_skills = set(self.skills)

        return required_skills.issubset(agent_skills)

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute assigned task."""
        print(f"   {self.name} executing: {task.description}")

        task.status = TaskStatus.IN_PROGRESS
        self.current_tasks.append(task)

        # Simulate task execution
        import random
        success_prob = max(0.7, self.performance_score * 0.9)
        success = random.random() < success_prob

        if success:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.result = {"status": "success", "output": f"Completed by {self.name}"}
        else:
            task.status = TaskStatus.FAILED
            task.result = {"status": "failed", "reason": "execution_error"}

        # Move to completed
        self.current_tasks.remove(task)
        self.completed_tasks.append(task)

        return {
            "task_id": task.id,
            "agent": self.name,
            "status": task.status.value,
            "result": task.result
        }

    def report_to_parent(self, task: Task) -> None:
        """Report task completion to parent."""
        if self.parent:
            self.parent.receive_report(task, self.name)

    def get_workload(self) -> float:
        """Calculate current workload."""
        return len(self.current_tasks) / self.capacity


class WorkerAgent(HierarchicalAgent):
    """Worker-level agent that executes tasks."""

    def __init__(self, name: str, skills: Optional[List[str]] = None):
        """Initialize worker agent."""
        super().__init__(
            name=name,
            level=AgentLevel.WORKER,
            skills=skills or ["general"],
            capacity=3
        )
        print(f"👷 Worker Agent '{name}' created (skills: {', '.join(self.skills)})")


class ManagerAgent(HierarchicalAgent):
    """Manager-level agent that delegates tasks."""

    def __init__(
        self,
        name: str,
        level: AgentLevel = AgentLevel.MANAGER,
        specialization: Optional[str] = None
    ):
        """Initialize manager agent."""
        super().__init__(
            name=name,
            level=level,
            skills=[specialization] if specialization else ["management"],
            capacity=10
        )
        self.subordinates: List[HierarchicalAgent] = []
        self.specialization = specialization
        self.delegation_count = 0

        print(f"👔 Manager Agent '{name}' created (level: {level.value})")

    def add_subordinate(self, agent: HierarchicalAgent) -> None:
        """Add subordinate to team."""
        agent.parent = self
        self.subordinates.append(agent)
        print(f"   ✓ {agent.name} added to {self.name}'s team")

    def delegate_task(
        self,
        task: Task,
        strategy: str = "optimal"
    ) -> Dict[str, Any]:
        """Delegate task to subordinates."""
        print(f"\n📋 {self.name} delegating: {task.description}")

        # Check if should decompose task
        if task.complexity > 0.7 and self.subordinates:
            print(f"   Task complexity high - decomposing...")
            return self._decompose_and_delegate(task)

        # Find best subordinate
        best_agent = self._select_agent(task, strategy)

        if best_agent:
            print(f"   Assigned to: {best_agent.name}")
            task.assigned_to = best_agent.name
            task.status = TaskStatus.ASSIGNED
            self.delegation_count += 1

            result = best_agent.execute_task(task)
            return result
        else:
            # Escalate if no suitable subordinate
            print(f"   ⬆️  Escalating to parent...")
            return self._escalate(task)

    def _select_agent(
        self,
        task: Task,
        strategy: str
    ) -> Optional[HierarchicalAgent]:
        """Select best agent for task."""
        candidates = [
            agent for agent in self.subordinates
            if agent.can_handle(task)
        ]

        if not candidates:
            return None

        if strategy == "optimal":
            # Select least loaded agent with best performance
            return min(
                candidates,
                key=lambda a: (a.get_workload(), -a.performance_score)
            )
        elif strategy == "round_robin":
            return candidates[self.delegation_count % len(candidates)]
        else:
            return candidates[0]

    def _decompose_and_delegate(self, task: Task) -> Dict[str, Any]:
        """Decompose complex task into subtasks."""
        # Create subtasks
        num_subtasks = min(3, len(self.subordinates))
        subtasks = []

        for i in range(num_subtasks):
            subtask = Task(
                id=f"{task.id}_sub{i+1}",
                description=f"Subtask {i+1}: {task.description}",
                complexity=task.complexity / num_subtasks,
                required_skills=task.required_skills,
                parent_task=task.id
            )
            subtasks.append(subtask)
            task.subtasks.append(subtask.id)

        # Delegate subtasks
        results = []
        for subtask in subtasks:
            result = self.delegate_task(subtask, strategy="optimal")
            results.append(result)

        # Aggregate results
        all_completed = all(
            r.get("status") == "completed" for r in results
        )

        task.status = TaskStatus.COMPLETED if all_completed else TaskStatus.FAILED
        task.result = {
            "status": "completed" if all_completed else "partial",
            "subtask_results": results
        }

        return {
            "task_id": task.id,
            "decomposed": True,
            "subtasks": len(subtasks),
            "status": task.status.value,
            "results": results
        }

    def _escalate(self, task: Task) -> Dict[str, Any]:
        """Escalate task to parent."""
        task.status = TaskStatus.ESCALATED

        if self.parent:
            print(f"   Escalating to {self.parent.name}")
            return self.parent.delegate_task(task)
        else:
            print(f"   ⚠️  No parent available for escalation")
            task.status = TaskStatus.FAILED
            return {
                "task_id": task.id,
                "status": "failed",
                "reason": "no_escalation_path"
            }

    def receive_report(self, task: Task, from_agent: str) -> None:
        """Receive task completion report from subordinate."""
        print(f"   📊 {self.name} received report from {from_agent}")

        # Update performance tracking
        if task.status == TaskStatus.COMPLETED:
            agent = next(
                (a for a in self.subordinates if a.name == from_agent),
                None
            )
            if agent:
                agent.performance_score = min(1.0, agent.performance_score + 0.05)

    def get_team_metrics(self) -> Dict[str, Any]:
        """Get metrics for managed team."""
        if not self.subordinates:
            return {"message": "No subordinates"}

        total_tasks = sum(
            len(agent.completed_tasks) for agent in self.subordinates
        )
        avg_workload = sum(
            agent.get_workload() for agent in self.subordinates
        ) / len(self.subordinates)

        return {
            "team_size": len(self.subordinates),
            "total_completed_tasks": total_tasks,
            "average_workload": round(avg_workload, 2),
            "delegations": self.delegation_count
        }


class HierarchyBuilder:
    """Builder for creating hierarchical agent structures."""

    def __init__(self):
        """Initialize hierarchy builder."""
        self.agents: Dict[str, HierarchicalAgent] = {}
        self.root_agents: List[HierarchicalAgent] = []

        print("🏢 Hierarchy Builder initialized")

    def create_manager(
        self,
        name: str,
        level: int = 2,
        specialization: Optional[str] = None,
        parent: Optional[ManagerAgent] = None
    ) -> ManagerAgent:
        """Create manager agent in hierarchy."""
        agent_level = AgentLevel(level)
        manager = ManagerAgent(
            name=name,
            level=agent_level,
            specialization=specialization
        )

        self.agents[name] = manager

        if parent:
            parent.add_subordinate(manager)
        else:
            self.root_agents.append(manager)

        return manager

    def create_worker(
        self,
        name: str,
        skills: Optional[List[str]] = None,
        parent: Optional[ManagerAgent] = None
    ) -> WorkerAgent:
        """Create worker agent in hierarchy."""
        worker = WorkerAgent(name=name, skills=skills)

        self.agents[name] = worker

        if parent:
            parent.add_subordinate(worker)

        return worker

    def get_hierarchy_metrics(self) -> Dict[str, Any]:
        """Get metrics for entire hierarchy."""
        print("\n📊 Calculating hierarchy metrics...")

        total_agents = len(self.agents)
        managers = sum(
            1 for a in self.agents.values()
            if isinstance(a, ManagerAgent)
        )
        workers = total_agents - managers

        total_completed = sum(
            len(agent.completed_tasks)
            for agent in self.agents.values()
        )

        metrics = {
            "total_agents": total_agents,
            "managers": managers,
            "workers": workers,
            "total_completed_tasks": total_completed,
            "hierarchy_depth": self._calculate_depth()
        }

        print(f"✓ Metrics calculated")
        return metrics

    def _calculate_depth(self) -> int:
        """Calculate maximum hierarchy depth."""
        def get_depth(agent: HierarchicalAgent, current_depth: int = 1) -> int:
            if isinstance(agent, ManagerAgent) and agent.subordinates:
                return max(
                    get_depth(sub, current_depth + 1)
                    for sub in agent.subordinates
                )
            return current_depth

        if not self.root_agents:
            return 0

        return max(get_depth(root) for root in self.root_agents)

    def visualize_hierarchy(self) -> None:
        """Visualize organizational hierarchy."""
        print("\n🌳 Organizational Hierarchy:")
        print("=" * 60)

        for root in self.root_agents:
            self._print_tree(root, level=0)

    def _print_tree(self, agent: HierarchicalAgent, level: int) -> None:
        """Recursively print hierarchy tree."""
        indent = "  " * level
        icon = "👔" if isinstance(agent, ManagerAgent) else "👷"
        workload = agent.get_workload()

        print(f"{indent}{icon} {agent.name} (load: {workload:.0%})")

        if isinstance(agent, ManagerAgent):
            for subordinate in agent.subordinates:
                self._print_tree(subordinate, level + 1)


def demo():
    """Demonstrate hierarchical agents framework."""
    print("=" * 60)
    print("Hierarchical Agents Framework Demo")
    print("=" * 60)

    # Build hierarchy
    builder = HierarchyBuilder()

    # Create CEO
    ceo = builder.create_manager(
        name="CEO",
        level=1,
        specialization="strategic_planning"
    )

    # Create department managers
    sales_mgr = builder.create_manager(
        name="SalesManager",
        level=2,
        specialization="sales",
        parent=ceo
    )

    tech_mgr = builder.create_manager(
        name="TechManager",
        level=2,
        specialization="technology",
        parent=ceo
    )

    # Create workers
    for i in range(2):
        builder.create_worker(
            name=f"SalesRep{i+1}",
            skills=["sales", "communication"],
            parent=sales_mgr
        )

    for i in range(3):
        builder.create_worker(
            name=f"Developer{i+1}",
            skills=["technology", "coding"],
            parent=tech_mgr
        )

    # Visualize hierarchy
    builder.visualize_hierarchy()

    # Create and delegate tasks
    print("\n" + "=" * 60)
    print("Task Delegation")
    print("=" * 60)

    tasks = [
        Task(
            id="task1",
            description="Close 5 new deals",
            complexity=0.6,
            required_skills=["sales"]
        ),
        Task(
            id="task2",
            description="Build customer dashboard",
            complexity=0.8,
            required_skills=["technology"]
        ),
        Task(
            id="task3",
            description="Strategic planning for Q4",
            complexity=0.9,
            required_skills=["strategic_planning"]
        )
    ]

    for task in tasks:
        result = ceo.delegate_task(task, strategy="optimal")
        print(f"   Result: {result['status']}")

    # Get metrics
    print("\n" + "=" * 60)
    print("Hierarchy Metrics")
    print("=" * 60)
    metrics = builder.get_hierarchy_metrics()
    print(json.dumps(metrics, indent=2))

    # Team metrics
    print("\n" + "=" * 60)
    print("Team Performance")
    print("=" * 60)
    print(f"\nSales Team:")
    print(json.dumps(sales_mgr.get_team_metrics(), indent=2))
    print(f"\nTech Team:")
    print(json.dumps(tech_mgr.get_team_metrics(), indent=2))


if __name__ == "__main__":
    demo()
