"""
Planning Agents - STRIPS and Goal-Oriented Planning
====================================================

Classical AI planning with STRIPS algorithm, goal decomposition, and
constraint satisfaction for automated planning and reasoning.

Author: Brill Consulting
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy
from collections import deque
import time


@dataclass
class State:
    """World state represented as a set of predicates."""
    predicates: Set[str] = field(default_factory=set)

    def satisfies(self, conditions: Set[str]) -> bool:
        """Check if state satisfies all conditions."""
        return conditions.issubset(self.predicates)

    def apply_effects(self, add_effects: Set[str], del_effects: Set[str]) -> 'State':
        """Apply action effects to create new state."""
        new_predicates = (self.predicates - del_effects) | add_effects
        return State(new_predicates)

    def __hash__(self):
        return hash(frozenset(self.predicates))

    def __eq__(self, other):
        return self.predicates == other.predicates

    def __str__(self):
        return f"State({', '.join(sorted(self.predicates))})"


@dataclass
class Action:
    """STRIPS action with preconditions and effects."""
    name: str
    preconditions: Set[str] = field(default_factory=set)
    add_effects: Set[str] = field(default_factory=set)
    del_effects: Set[str] = field(default_factory=set)
    cost: float = 1.0
    parameters: Dict[str, str] = field(default_factory=dict)

    def is_applicable(self, state: State) -> bool:
        """Check if action can be applied in given state."""
        return state.satisfies(self.preconditions)

    def apply(self, state: State) -> State:
        """Apply action to state, return new state."""
        if not self.is_applicable(state):
            raise ValueError(f"Action {self.name} not applicable in current state")
        return state.apply_effects(self.add_effects, self.del_effects)

    def __str__(self):
        params = f"({', '.join(f'{k}={v}' for k, v in self.parameters.items())})" if self.parameters else ""
        return f"{self.name}{params}"


@dataclass
class Plan:
    """Sequence of actions forming a plan."""
    actions: List[Action] = field(default_factory=list)
    cost: float = 0.0

    def add_action(self, action: Action):
        """Add action to plan."""
        self.actions.append(action)
        self.cost += action.cost

    def validate(self, initial_state: State, goal: Set[str]) -> bool:
        """Validate plan achieves goal from initial state."""
        current_state = initial_state

        for action in self.actions:
            if not action.is_applicable(current_state):
                return False
            current_state = action.apply(current_state)

        return current_state.satisfies(goal)

    def __len__(self):
        return len(self.actions)

    def __str__(self):
        return " -> ".join([str(action) for action in self.actions])


class STRIPSPlanner:
    """STRIPS planning algorithm with forward search."""

    def __init__(self, actions: List[Action]):
        """
        Initialize STRIPS planner.

        Args:
            actions: Available actions in the domain
        """
        self.actions = actions
        self.nodes_explored = 0
        self.max_depth = 50

    def plan(self, initial_state: State, goal: Set[str]) -> Optional[Plan]:
        """
        Generate plan using forward state-space search.

        Args:
            initial_state: Starting state
            goal: Goal predicates to achieve

        Returns:
            Plan if found, None otherwise
        """
        self.nodes_explored = 0

        # BFS with state tracking
        queue = deque([(initial_state, Plan())])
        visited = {initial_state}

        while queue:
            current_state, current_plan = queue.popleft()
            self.nodes_explored += 1

            # Check goal
            if current_state.satisfies(goal):
                return current_plan

            # Depth limit
            if len(current_plan) >= self.max_depth:
                continue

            # Try all applicable actions
            for action in self.actions:
                if action.is_applicable(current_state):
                    new_state = action.apply(current_state)

                    if new_state not in visited:
                        visited.add(new_state)
                        new_plan = deepcopy(current_plan)
                        new_plan.add_action(action)
                        queue.append((new_state, new_plan))

        return None  # No plan found


class GoalDecomposer:
    """Hierarchical goal decomposition for complex goals."""

    def __init__(self):
        self.decomposition_rules: Dict[str, List[Set[str]]] = {}

    def add_rule(self, goal: str, subgoals: List[Set[str]]):
        """
        Add decomposition rule.

        Args:
            goal: High-level goal
            subgoals: Ordered list of subgoal sets
        """
        self.decomposition_rules[goal] = subgoals

    def decompose(self, goal: Set[str]) -> List[Set[str]]:
        """
        Decompose complex goal into subgoals.

        Returns:
            Ordered list of subgoal sets
        """
        # Check if any predicate has decomposition rule
        for predicate in goal:
            if predicate in self.decomposition_rules:
                subgoals = self.decomposition_rules[predicate]
                remaining = goal - {predicate}

                if remaining:
                    return subgoals + [remaining]
                return subgoals

        # No decomposition needed
        return [goal]


class ConstraintChecker:
    """Check plan constraints and temporal conditions."""

    def __init__(self):
        self.constraints: List[Tuple[str, str]] = []  # (type, description)

    def add_mutex_constraint(self, pred1: str, pred2: str):
        """Add mutual exclusion constraint (can't both be true)."""
        self.constraints.append(('mutex', f"{pred1} and {pred2} are mutually exclusive"))

    def add_ordering_constraint(self, action1: str, action2: str):
        """Add ordering constraint (action1 must come before action2)."""
        self.constraints.append(('order', f"{action1} must come before {action2}"))

    def validate_state(self, state: State) -> Tuple[bool, List[str]]:
        """
        Validate state against constraints.

        Returns:
            (is_valid, list of violations)
        """
        violations = []

        for constraint_type, description in self.constraints:
            if constraint_type == 'mutex':
                # Parse predicates from description
                parts = description.split(' and ')
                if len(parts) == 2:
                    pred1 = parts[0].strip()
                    pred2 = parts[1].replace(' are mutually exclusive', '').strip()

                    if pred1 in state.predicates and pred2 in state.predicates:
                        violations.append(description)

        return len(violations) == 0, violations

    def validate_plan(self, plan: Plan) -> Tuple[bool, List[str]]:
        """
        Validate plan against ordering constraints.

        Returns:
            (is_valid, list of violations)
        """
        violations = []
        action_names = [action.name for action in plan.actions]

        for constraint_type, description in self.constraints:
            if constraint_type == 'order':
                # Parse actions from description
                parts = description.split(' must come before ')
                if len(parts) == 2:
                    action1 = parts[0].strip()
                    action2 = parts[1].strip()

                    if action1 in action_names and action2 in action_names:
                        idx1 = action_names.index(action1)
                        idx2 = action_names.index(action2)

                        if idx1 > idx2:
                            violations.append(description)

        return len(violations) == 0, violations


class HierarchicalPlanner:
    """Hierarchical Task Network (HTN) style planning."""

    def __init__(self, planner: STRIPSPlanner, decomposer: GoalDecomposer):
        self.planner = planner
        self.decomposer = decomposer

    def plan(self, initial_state: State, goal: Set[str]) -> Optional[Plan]:
        """
        Generate hierarchical plan by decomposing goal.

        Args:
            initial_state: Starting state
            goal: High-level goal

        Returns:
            Combined plan if found, None otherwise
        """
        # Decompose goal into subgoals
        subgoals = self.decomposer.decompose(goal)

        # Plan for each subgoal sequentially
        combined_plan = Plan()
        current_state = initial_state

        for subgoal in subgoals:
            subplan = self.planner.plan(current_state, subgoal)

            if subplan is None:
                return None  # Failed to achieve subgoal

            # Add subplan actions to combined plan
            for action in subplan.actions:
                combined_plan.add_action(action)
                current_state = action.apply(current_state)

        return combined_plan


def create_blocks_world_domain() -> List[Action]:
    """Create classic blocks world domain actions."""
    actions = []

    blocks = ['A', 'B', 'C']

    for block in blocks:
        # Pick up block from table
        actions.append(Action(
            name=f'pickup_{block}',
            preconditions={f'on_table_{block}', f'clear_{block}', 'hand_empty'},
            add_effects={f'holding_{block}'},
            del_effects={f'on_table_{block}', f'clear_{block}', 'hand_empty'},
            parameters={'block': block}
        ))

        # Put block on table
        actions.append(Action(
            name=f'putdown_{block}',
            preconditions={f'holding_{block}'},
            add_effects={f'on_table_{block}', f'clear_{block}', 'hand_empty'},
            del_effects={f'holding_{block}'},
            parameters={'block': block}
        ))

        # Stack blocks
        for other in blocks:
            if block != other:
                actions.append(Action(
                    name=f'stack_{block}_on_{other}',
                    preconditions={f'holding_{block}', f'clear_{other}'},
                    add_effects={f'on_{block}_{other}', f'clear_{block}', 'hand_empty'},
                    del_effects={f'holding_{block}', f'clear_{other}'},
                    parameters={'block': block, 'on': other}
                ))

                actions.append(Action(
                    name=f'unstack_{block}_from_{other}',
                    preconditions={f'on_{block}_{other}', f'clear_{block}', 'hand_empty'},
                    add_effects={f'holding_{block}', f'clear_{other}'},
                    del_effects={f'on_{block}_{other}', f'clear_{block}', 'hand_empty'},
                    parameters={'block': block, 'from': other}
                ))

    return actions


def demo():
    """Demonstration of Planning Agents."""
    print("Planning Agents - STRIPS and Hierarchical Planning Demo")
    print("=" * 70)

    # Create blocks world domain
    actions = create_blocks_world_domain()

    print("\n1️⃣  Blocks World Domain")
    print("-" * 70)
    print(f"Actions defined: {len(actions)}")
    print("Sample actions:")
    for action in actions[:5]:
        print(f"  • {action.name}")
        print(f"    Pre: {action.preconditions}")
        print(f"    Add: {action.add_effects}")
        print(f"    Del: {action.del_effects}")

    # Define initial state: A on B on table, C on table
    initial_state = State({
        'on_A_B', 'on_table_B', 'on_table_C',
        'clear_A', 'clear_C', 'hand_empty'
    })

    # Define goal: C on A on B
    goal = {'on_C_A', 'on_A_B'}

    print("\n2️⃣  Planning Problem")
    print("-" * 70)
    print(f"Initial State: {initial_state}")
    print(f"Goal: {goal}")

    # Create STRIPS planner
    planner = STRIPSPlanner(actions)

    print("\n3️⃣  STRIPS Planning")
    print("-" * 70)
    start_time = time.time()
    plan = planner.plan(initial_state, goal)
    planning_time = time.time() - start_time

    if plan:
        print(f"✓ Plan found in {planning_time:.4f} seconds!")
        print(f"Plan length: {len(plan)} actions")
        print(f"Plan cost: {plan.cost}")
        print(f"Nodes explored: {planner.nodes_explored}")
        print("\nPlan steps:")
        for i, action in enumerate(plan.actions, 1):
            print(f"  {i}. {action}")
    else:
        print("✗ No plan found")

    # Validate plan
    print("\n4️⃣  Plan Validation")
    print("-" * 70)
    if plan:
        is_valid = plan.validate(initial_state, goal)
        print(f"Plan validation: {'✓ VALID' if is_valid else '✗ INVALID'}")

        # Simulate execution
        print("\nPlan execution simulation:")
        current_state = initial_state
        for i, action in enumerate(plan.actions, 1):
            print(f"\nStep {i}: {action}")
            current_state = action.apply(current_state)
            print(f"State: {len(current_state.predicates)} predicates")

        print(f"\nFinal state satisfies goal: {current_state.satisfies(goal)}")

    # Goal decomposition
    print("\n5️⃣  Hierarchical Planning")
    print("-" * 70)

    decomposer = GoalDecomposer()
    decomposer.add_rule(
        'on_C_A',
        [{'clear_C', 'clear_A'}, {'on_C_A'}]
    )

    hierarchical_planner = HierarchicalPlanner(planner, decomposer)

    complex_goal = {'on_C_A'}
    print(f"Complex goal: {complex_goal}")

    subgoals = decomposer.decompose(complex_goal)
    print(f"Decomposed into {len(subgoals)} subgoals:")
    for i, subgoal in enumerate(subgoals, 1):
        print(f"  {i}. {subgoal}")

    # Constraint checking
    print("\n6️⃣  Constraint Validation")
    print("-" * 70)

    checker = ConstraintChecker()
    checker.add_mutex_constraint('holding_A', 'holding_B')
    checker.add_ordering_constraint('pickup_A', 'putdown_A')

    print(f"Constraints defined: {len(checker.constraints)}")

    if plan:
        is_valid, violations = checker.validate_plan(plan)
        print(f"Constraint check: {'✓ SATISFIED' if is_valid else '✗ VIOLATED'}")
        if violations:
            print("Violations:")
            for violation in violations:
                print(f"  • {violation}")

    # Planning statistics
    print("\n7️⃣  Planning Statistics")
    print("-" * 70)
    print(f"Domain size: {len(actions)} actions")
    print(f"Problem complexity: {len(initial_state.predicates)} initial predicates")
    print(f"Goal size: {len(goal)} predicates")
    print(f"Search efficiency: {len(plan) / planner.nodes_explored:.4f} (plan/explored)")
    print(f"Planning time: {planning_time * 1000:.2f} ms")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
