"""
Reactive Agents - Behavior Trees and Subsumption Architecture
==============================================================

Fast, real-time reactive agents using behavior trees, condition-action rules,
and subsumption architecture for immediate response systems.

Author: Brill Consulting
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time


class NodeStatus(Enum):
    """Behavior tree node execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BehaviorNode(ABC):
    """Base class for behavior tree nodes."""

    def __init__(self, name: str):
        self.name = name
        self.status = NodeStatus.FAILURE

    @abstractmethod
    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        """Execute node logic and return status."""
        pass

    def reset(self):
        """Reset node state."""
        self.status = NodeStatus.FAILURE


class ActionNode(BehaviorNode):
    """Leaf node that executes an action."""

    def __init__(self, name: str, action: Callable[[Dict[str, Any]], bool]):
        super().__init__(name)
        self.action = action

    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        """Execute action and return status."""
        try:
            success = self.action(blackboard)
            self.status = NodeStatus.SUCCESS if success else NodeStatus.FAILURE
        except Exception as e:
            blackboard['last_error'] = str(e)
            self.status = NodeStatus.FAILURE

        return self.status


class ConditionNode(BehaviorNode):
    """Leaf node that checks a condition."""

    def __init__(self, name: str, condition: Callable[[Dict[str, Any]], bool]):
        super().__init__(name)
        self.condition = condition

    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        """Check condition and return status."""
        result = self.condition(blackboard)
        self.status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        return self.status


class SequenceNode(BehaviorNode):
    """Executes children in order until one fails."""

    def __init__(self, name: str, children: List[BehaviorNode] = None):
        super().__init__(name)
        self.children = children or []

    def add_child(self, child: BehaviorNode):
        """Add child node."""
        self.children.append(child)

    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        """Execute children sequentially."""
        for child in self.children:
            status = child.tick(blackboard)

            if status == NodeStatus.FAILURE:
                self.status = NodeStatus.FAILURE
                return self.status

            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status

        self.status = NodeStatus.SUCCESS
        return self.status


class SelectorNode(BehaviorNode):
    """Executes children in order until one succeeds."""

    def __init__(self, name: str, children: List[BehaviorNode] = None):
        super().__init__(name)
        self.children = children or []

    def add_child(self, child: BehaviorNode):
        """Add child node."""
        self.children.append(child)

    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        """Execute children until success."""
        for child in self.children:
            status = child.tick(blackboard)

            if status == NodeStatus.SUCCESS:
                self.status = NodeStatus.SUCCESS
                return self.status

            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status

        self.status = NodeStatus.FAILURE
        return self.status


class ParallelNode(BehaviorNode):
    """Executes all children simultaneously."""

    def __init__(self, name: str, success_threshold: int = 1,
                 children: List[BehaviorNode] = None):
        super().__init__(name)
        self.children = children or []
        self.success_threshold = success_threshold

    def add_child(self, child: BehaviorNode):
        """Add child node."""
        self.children.append(child)

    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        """Execute all children in parallel."""
        success_count = 0
        failure_count = 0

        for child in self.children:
            status = child.tick(blackboard)

            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1

        # Success if enough children succeeded
        if success_count >= self.success_threshold:
            self.status = NodeStatus.SUCCESS
            return self.status

        # Still running if not enough failures
        if failure_count < len(self.children) - self.success_threshold + 1:
            self.status = NodeStatus.RUNNING
            return self.status

        self.status = NodeStatus.FAILURE
        return self.status


class DecoratorNode(BehaviorNode):
    """Modifies behavior of child node."""

    def __init__(self, name: str, child: BehaviorNode):
        super().__init__(name)
        self.child = child

    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        """Execute child with modification."""
        return self.child.tick(blackboard)


class InverterNode(DecoratorNode):
    """Inverts child's success/failure."""

    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        """Invert child status."""
        status = self.child.tick(blackboard)

        if status == NodeStatus.SUCCESS:
            self.status = NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            self.status = NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.RUNNING

        return self.status


class RepeaterNode(DecoratorNode):
    """Repeats child N times."""

    def __init__(self, name: str, child: BehaviorNode, repeat_count: int = -1):
        super().__init__(name, child)
        self.repeat_count = repeat_count  # -1 = infinite
        self.current_count = 0

    def tick(self, blackboard: Dict[str, Any]) -> NodeStatus:
        """Repeat child execution."""
        if self.repeat_count > 0 and self.current_count >= self.repeat_count:
            self.status = NodeStatus.SUCCESS
            return self.status

        status = self.child.tick(blackboard)

        if status == NodeStatus.SUCCESS or status == NodeStatus.FAILURE:
            self.current_count += 1
            self.child.reset()

        self.status = NodeStatus.RUNNING
        return self.status


class BehaviorTree:
    """Behavior tree with root node and blackboard."""

    def __init__(self, root: BehaviorNode):
        self.root = root
        self.blackboard: Dict[str, Any] = {}
        self.tick_count = 0

    def tick(self) -> NodeStatus:
        """Execute one tree iteration."""
        self.tick_count += 1
        return self.root.tick(self.blackboard)

    def reset(self):
        """Reset tree state."""
        self.root.reset()
        self.tick_count = 0


# Subsumption Architecture


@dataclass
class Behavior:
    """Reactive behavior with condition and action."""
    name: str
    priority: int
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Any]
    enabled: bool = True

    def is_triggered(self, sensors: Dict[str, Any]) -> bool:
        """Check if behavior should activate."""
        return self.enabled and self.condition(sensors)

    def execute(self, sensors: Dict[str, Any]) -> Any:
        """Execute behavior action."""
        return self.action(sensors)


class SubsumptionAgent:
    """Reactive agent with subsumption architecture."""

    def __init__(self, name: str):
        self.name = name
        self.behaviors: List[Behavior] = []
        self.sensors: Dict[str, Any] = {}
        self.active_behavior: Optional[str] = None
        self.execution_count = 0

    def add_behavior(self, behavior: Behavior):
        """Add behavior to agent (sorted by priority)."""
        self.behaviors.append(behavior)
        self.behaviors.sort(key=lambda b: b.priority, reverse=True)

    def update_sensors(self, sensor_data: Dict[str, Any]):
        """Update sensor readings."""
        self.sensors.update(sensor_data)

    def step(self) -> Optional[Any]:
        """Execute one agent cycle."""
        self.execution_count += 1

        # Check behaviors in priority order
        for behavior in self.behaviors:
            if behavior.is_triggered(self.sensors):
                self.active_behavior = behavior.name
                return behavior.execute(self.sensors)

        self.active_behavior = None
        return None


# Condition-Action Rules


@dataclass
class Rule:
    """Simple condition-action rule."""
    name: str
    conditions: List[Callable[[Dict[str, Any]], bool]]
    action: Callable[[Dict[str, Any]], Any]

    def matches(self, state: Dict[str, Any]) -> bool:
        """Check if all conditions are satisfied."""
        return all(condition(state) for condition in self.conditions)

    def fire(self, state: Dict[str, Any]) -> Any:
        """Execute rule action."""
        return self.action(state)


class RuleBasedAgent:
    """Simple reactive agent using condition-action rules."""

    def __init__(self, name: str):
        self.name = name
        self.rules: List[Rule] = []
        self.state: Dict[str, Any] = {}
        self.fired_rules: List[str] = []

    def add_rule(self, rule: Rule):
        """Add rule to agent."""
        self.rules.append(rule)

    def update_state(self, updates: Dict[str, Any]):
        """Update agent state."""
        self.state.update(updates)

    def step(self):
        """Execute one reasoning cycle."""
        for rule in self.rules:
            if rule.matches(self.state):
                result = rule.fire(self.state)
                self.fired_rules.append(rule.name)
                return result

        return None


def demo():
    """Demonstration of Reactive Agents."""
    print("Reactive Agents - Behavior Trees and Subsumption Demo")
    print("=" * 70)

    # Demo 1: Behavior Tree
    print("\n1️⃣  Behavior Tree Example")
    print("-" * 70)

    # Create behavior tree for a patrol and attack robot
    def check_enemy_near(bb):
        return bb.get('enemy_distance', 100) < 10

    def check_health_low(bb):
        return bb.get('health', 100) < 30

    def attack_enemy(bb):
        print("    → Attacking enemy!")
        bb['enemy_distance'] = 100
        return True

    def retreat(bb):
        print("    → Retreating to safety!")
        bb['health'] = 100
        return True

    def patrol(bb):
        print("    → Patrolling area...")
        position = bb.get('position', 0)
        bb['position'] = position + 1
        return True

    # Build tree: If enemy near and health good -> attack, else if low health -> retreat, else patrol
    root = SelectorNode("root")

    # Attack sequence
    attack_seq = SequenceNode("attack_sequence")
    attack_seq.add_child(ConditionNode("enemy_near", check_enemy_near))
    attack_seq.add_child(InverterNode("health_ok", ConditionNode("low_health", check_health_low)))
    attack_seq.add_child(ActionNode("attack", attack_enemy))

    # Retreat sequence
    retreat_seq = SequenceNode("retreat_sequence")
    retreat_seq.add_child(ConditionNode("low_health", check_health_low))
    retreat_seq.add_child(ActionNode("retreat", retreat))

    # Patrol action
    patrol_action = ActionNode("patrol", patrol)

    root.add_child(attack_seq)
    root.add_child(retreat_seq)
    root.add_child(patrol_action)

    tree = BehaviorTree(root)
    tree.blackboard = {'health': 100, 'enemy_distance': 100, 'position': 0}

    print("Initial state:", tree.blackboard)

    # Scenario 1: Patrol
    print("\nScenario 1: Normal patrol")
    status = tree.tick()
    print(f"Status: {status.value}, State: {tree.blackboard}")

    # Scenario 2: Enemy approaches
    print("\nScenario 2: Enemy detected!")
    tree.blackboard['enemy_distance'] = 5
    status = tree.tick()
    print(f"Status: {status.value}, State: {tree.blackboard}")

    # Scenario 3: Low health
    print("\nScenario 3: Low health!")
    tree.blackboard['health'] = 20
    tree.blackboard['enemy_distance'] = 5
    status = tree.tick()
    print(f"Status: {status.value}, State: {tree.blackboard}")

    # Demo 2: Subsumption Architecture
    print("\n2️⃣  Subsumption Architecture Example")
    print("-" * 70)

    agent = SubsumptionAgent("Robot")

    # Define behaviors (higher priority = more important)
    avoid_obstacle = Behavior(
        name="avoid_obstacle",
        priority=100,
        condition=lambda s: s.get('obstacle_distance', 100) < 20,
        action=lambda s: "Avoiding obstacle - turning right"
    )

    seek_goal = Behavior(
        name="seek_goal",
        priority=50,
        condition=lambda s: s.get('goal_visible', False),
        action=lambda s: "Moving toward goal"
    )

    wander = Behavior(
        name="wander",
        priority=10,
        condition=lambda s: True,  # Always active
        action=lambda s: "Wandering randomly"
    )

    agent.add_behavior(avoid_obstacle)
    agent.add_behavior(seek_goal)
    agent.add_behavior(wander)

    print(f"Agent '{agent.name}' with {len(agent.behaviors)} behaviors")
    print("Priority order:")
    for b in agent.behaviors:
        print(f"  {b.priority}: {b.name}")

    # Test scenarios
    scenarios = [
        {"obstacle_distance": 100, "goal_visible": False},
        {"obstacle_distance": 100, "goal_visible": True},
        {"obstacle_distance": 10, "goal_visible": True}
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario}")
        agent.update_sensors(scenario)
        result = agent.step()
        print(f"  Active: {agent.active_behavior} - {result}")

    # Demo 3: Rule-Based Agent
    print("\n3️⃣  Rule-Based Agent Example")
    print("-" * 70)

    rule_agent = RuleBasedAgent("Assistant")

    # Define rules
    emergency_rule = Rule(
        name="emergency_response",
        conditions=[
            lambda s: s.get('alert_level', 0) > 8,
            lambda s: s.get('system_status') == 'critical'
        ],
        action=lambda s: "EMERGENCY: Shutting down systems!"
    )

    maintenance_rule = Rule(
        name="routine_maintenance",
        conditions=[
            lambda s: s.get('uptime', 0) > 100,
            lambda s: s.get('system_status') == 'normal'
        ],
        action=lambda s: "Performing routine maintenance"
    )

    idle_rule = Rule(
        name="idle",
        conditions=[lambda s: True],
        action=lambda s: "System idle"
    )

    rule_agent.add_rule(emergency_rule)
    rule_agent.add_rule(maintenance_rule)
    rule_agent.add_rule(idle_rule)

    print(f"Agent '{rule_agent.name}' with {len(rule_agent.rules)} rules")

    # Test scenarios
    test_states = [
        {"alert_level": 5, "system_status": "normal", "uptime": 50},
        {"alert_level": 5, "system_status": "normal", "uptime": 150},
        {"alert_level": 9, "system_status": "critical", "uptime": 200}
    ]

    for i, state in enumerate(test_states, 1):
        print(f"\nState {i}: {state}")
        rule_agent.update_state(state)
        result = rule_agent.step()
        print(f"  Action: {result}")

    # Performance statistics
    print("\n4️⃣  Performance Statistics")
    print("-" * 70)
    print(f"Behavior Tree ticks: {tree.tick_count}")
    print(f"Subsumption agent cycles: {agent.execution_count}")
    print(f"Rules fired: {len(rule_agent.fired_rules)}")
    print(f"Rule names: {rule_agent.fired_rules}")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
