"""
Task Planning - Hierarchical Task Networks (HTN)
=================================================

Hierarchical task decomposition, method selection, and automated planning
using HTN paradigm for complex task execution.

Author: Brill Consulting
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy


class TaskType(Enum):
    """Task types in HTN."""
    PRIMITIVE = "primitive"  # Directly executable action
    COMPOUND = "compound"    # Requires decomposition


@dataclass
class Task:
    """HTN task (primitive or compound)."""
    name: str
    task_type: TaskType
    parameters: Dict[str, str] = field(default_factory=dict)
    preconditions: Set[str] = field(default_factory=set)
    effects: Set[str] = field(default_factory=set)

    def __str__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.name}({params})" if params else self.name


@dataclass
class Method:
    """Decomposition method for compound tasks."""
    name: str
    task_name: str  # Which compound task this decomposes
    preconditions: Set[str] = field(default_factory=set)
    subtasks: List[Task] = field(default_factory=list)
    constraints: List[Tuple[str, str]] = field(default_factory=list)  # Ordering constraints

    def is_applicable(self, state: Set[str]) -> bool:
        """Check if method can be applied in current state."""
        return self.preconditions.issubset(state)


@dataclass
class Plan:
    """Sequence of primitive tasks."""
    tasks: List[Task] = field(default_factory=list)
    cost: float = 0.0

    def add_task(self, task: Task):
        """Add task to plan."""
        self.tasks.append(task)
        self.cost += 1.0

    def __len__(self):
        return len(self.tasks)

    def __str__(self):
        return " → ".join([str(task) for task in self.tasks])


class HTNPlanner:
    """Hierarchical Task Network planner."""

    def __init__(self):
        self.methods: Dict[str, List[Method]] = {}
        self.primitive_tasks: Dict[str, Task] = {}
        self.nodes_explored = 0

    def add_method(self, method: Method):
        """Add decomposition method."""
        if method.task_name not in self.methods:
            self.methods[method.task_name] = []
        self.methods[method.task_name].append(method)

    def add_primitive(self, task: Task):
        """Add primitive task."""
        if task.task_type != TaskType.PRIMITIVE:
            raise ValueError(f"Task {task.name} is not primitive")
        self.primitive_tasks[task.name] = task

    def plan(self, initial_tasks: List[Task], initial_state: Set[str],
             goal: Set[str]) -> Optional[Plan]:
        """
        Generate plan using HTN planning.

        Args:
            initial_tasks: Initial task list
            initial_state: Initial world state
            goal: Goal conditions

        Returns:
            Plan if found, None otherwise
        """
        self.nodes_explored = 0
        plan = Plan()
        state = initial_state.copy()
        tasks = initial_tasks.copy()

        return self._decompose(tasks, state, plan, goal)

    def _decompose(self, tasks: List[Task], state: Set[str],
                   plan: Plan, goal: Set[str]) -> Optional[Plan]:
        """Recursively decompose tasks."""
        self.nodes_explored += 1

        # Base case: no more tasks
        if not tasks:
            # Check if goal satisfied
            if goal.issubset(state):
                return plan
            return None

        # Get first task
        current_task = tasks[0]
        remaining_tasks = tasks[1:]

        # If primitive, execute it
        if current_task.task_type == TaskType.PRIMITIVE:
            # Check preconditions
            if not current_task.preconditions.issubset(state):
                return None

            # Execute task
            new_state = (state - current_task.effects) | current_task.effects
            new_plan = deepcopy(plan)
            new_plan.add_task(current_task)

            # Continue with remaining tasks
            return self._decompose(remaining_tasks, new_state, new_plan, goal)

        # If compound, try decomposition methods
        if current_task.name not in self.methods:
            return None

        for method in self.methods[current_task.name]:
            # Check if method applicable
            if not method.is_applicable(state):
                continue

            # Decompose: add subtasks before remaining tasks
            new_tasks = method.subtasks + remaining_tasks

            # Try to plan with decomposition
            result = self._decompose(new_tasks, state, plan, goal)
            if result:
                return result

        return None


class TaskDecomposer:
    """Utilities for task decomposition and analysis."""

    @staticmethod
    def decompose_sequential(compound_task: str, subtasks: List[Task]) -> Method:
        """Create sequential decomposition method."""
        return Method(
            name=f"sequential_{compound_task}",
            task_name=compound_task,
            subtasks=subtasks
        )

    @staticmethod
    def decompose_parallel(compound_task: str, subtasks: List[Task]) -> Method:
        """Create parallel decomposition (no ordering constraints)."""
        return Method(
            name=f"parallel_{compound_task}",
            task_name=compound_task,
            subtasks=subtasks,
            constraints=[]  # No ordering
        )

    @staticmethod
    def decompose_conditional(compound_task: str, condition: str,
                            if_tasks: List[Task],
                            else_tasks: List[Task]) -> List[Method]:
        """Create conditional decomposition methods."""
        return [
            Method(
                name=f"if_{compound_task}",
                task_name=compound_task,
                preconditions={condition},
                subtasks=if_tasks
            ),
            Method(
                name=f"else_{compound_task}",
                task_name=compound_task,
                preconditions=set(),  # Default case
                subtasks=else_tasks
            )
        ]


def demo():
    """Demonstration of Task Planning."""
    print("Task Planning - Hierarchical Task Networks Demo")
    print("=" * 70)

    planner = HTNPlanner()

    # Define primitive tasks for travel domain
    print("\n1️⃣  Defining Primitive Tasks")
    print("-" * 70)

    # Transportation
    drive = Task(
        name="drive",
        task_type=TaskType.PRIMITIVE,
        parameters={"from": "home", "to": "airport"},
        preconditions={"at_home", "have_car"},
        effects={"at_airport"}
    )

    take_taxi = Task(
        name="take_taxi",
        task_type=TaskType.PRIMITIVE,
        preconditions={"at_home"},
        effects={"at_airport"}
    )

    # Airplane
    buy_ticket = Task(
        name="buy_ticket",
        task_type=TaskType.PRIMITIVE,
        preconditions={"at_airport"},
        effects={"have_ticket"}
    )

    check_in = Task(
        name="check_in",
        task_type=TaskType.PRIMITIVE,
        preconditions={"at_airport", "have_ticket"},
        effects={"checked_in"}
    )

    board_plane = Task(
        name="board_plane",
        task_type=TaskType.PRIMITIVE,
        preconditions={"checked_in"},
        effects={"on_plane"}
    )

    fly = Task(
        name="fly",
        task_type=TaskType.PRIMITIVE,
        preconditions={"on_plane"},
        effects={"at_destination"}
    )

    planner.add_primitive(drive)
    planner.add_primitive(take_taxi)
    planner.add_primitive(buy_ticket)
    planner.add_primitive(check_in)
    planner.add_primitive(board_plane)
    planner.add_primitive(fly)

    print(f"✓ Defined {len(planner.primitive_tasks)} primitive tasks")
    for task_name in list(planner.primitive_tasks.keys())[:3]:
        print(f"  • {task_name}")

    # Define compound tasks
    print("\n2️⃣  Defining Compound Tasks and Methods")
    print("-" * 70)

    travel_by_air = Task(
        name="travel_by_air",
        task_type=TaskType.COMPOUND
    )

    get_to_airport = Task(
        name="get_to_airport",
        task_type=TaskType.COMPOUND
    )

    # Methods for getting to airport
    method_drive = Method(
        name="drive_to_airport",
        task_name="get_to_airport",
        preconditions={"have_car"},
        subtasks=[drive]
    )

    method_taxi = Method(
        name="taxi_to_airport",
        task_name="get_to_airport",
        preconditions=set(),  # Always available
        subtasks=[take_taxi]
    )

    planner.add_method(method_drive)
    planner.add_method(method_taxi)

    # Method for air travel
    method_air_travel = Method(
        name="standard_air_travel",
        task_name="travel_by_air",
        subtasks=[
            get_to_airport,
            buy_ticket,
            check_in,
            board_plane,
            fly
        ]
    )

    planner.add_method(method_air_travel)

    print(f"✓ Defined {len(planner.methods)} compound task methods")
    for task_name, methods in planner.methods.items():
        print(f"  • {task_name}: {len(methods)} method(s)")

    # Planning scenario 1: With car
    print("\n3️⃣  Planning Scenario 1: Travel with Car")
    print("-" * 70)

    initial_state_1 = {"at_home", "have_car"}
    goal_1 = {"at_destination"}
    initial_tasks_1 = [travel_by_air]

    print(f"Initial state: {initial_state_1}")
    print(f"Goal: {goal_1}")
    print(f"Initial tasks: {[str(t) for t in initial_tasks_1]}")

    plan_1 = planner.plan(initial_tasks_1, initial_state_1, goal_1)

    if plan_1:
        print(f"\n✓ Plan found!")
        print(f"Plan length: {len(plan_1)} steps")
        print(f"Plan cost: {plan_1.cost}")
        print("\nPlan steps:")
        for i, task in enumerate(plan_1.tasks, 1):
            print(f"  {i}. {task}")
    else:
        print("✗ No plan found")

    # Planning scenario 2: Without car
    print("\n4️⃣  Planning Scenario 2: Travel without Car")
    print("-" * 70)

    initial_state_2 = {"at_home"}  # No car
    goal_2 = {"at_destination"}
    initial_tasks_2 = [travel_by_air]

    print(f"Initial state: {initial_state_2}")
    print(f"Goal: {goal_2}")

    planner.nodes_explored = 0
    plan_2 = planner.plan(initial_tasks_2, initial_state_2, goal_2)

    if plan_2:
        print(f"\n✓ Plan found!")
        print(f"Plan length: {len(plan_2)} steps")
        print(f"Nodes explored: {planner.nodes_explored}")
        print("\nPlan steps:")
        for i, task in enumerate(plan_2.tasks, 1):
            print(f"  {i}. {task}")
    else:
        print("✗ No plan found")

    # Task decomposition utilities
    print("\n5️⃣  Task Decomposition Utilities")
    print("-" * 70)

    decomposer = TaskDecomposer()

    # Sequential decomposition
    prepare_meal = Task("prepare_meal", TaskType.COMPOUND)
    cook = Task("cook", TaskType.PRIMITIVE)
    serve = Task("serve", TaskType.PRIMITIVE)

    seq_method = decomposer.decompose_sequential(
        "prepare_meal",
        [cook, serve]
    )

    print("Sequential decomposition:")
    print(f"  Task: {seq_method.task_name}")
    print(f"  Subtasks: {[str(t) for t in seq_method.subtasks]}")

    # Conditional decomposition
    go_to_work = Task("go_to_work", TaskType.COMPOUND)
    drive_to_work = Task("drive_to_work", TaskType.PRIMITIVE)
    take_bus = Task("take_bus", TaskType.PRIMITIVE)

    cond_methods = decomposer.decompose_conditional(
        "go_to_work",
        "have_car",
        [drive_to_work],
        [take_bus]
    )

    print("\nConditional decomposition:")
    for method in cond_methods:
        print(f"  Method: {method.name}")
        print(f"    Preconditions: {method.preconditions}")
        print(f"    Subtasks: {[str(t) for t in method.subtasks]}")

    # Statistics
    print("\n6️⃣  Planning Statistics")
    print("-" * 70)
    print(f"Total primitive tasks: {len(planner.primitive_tasks)}")
    print(f"Total decomposition methods: {sum(len(m) for m in planner.methods.values())}")
    print(f"Compound tasks: {len(planner.methods)}")
    print(f"Final plan length: {len(plan_2)} steps")
    print(f"Nodes explored: {planner.nodes_explored}")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
