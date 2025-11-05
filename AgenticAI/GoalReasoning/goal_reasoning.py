"""
Goal Reasoning - Dynamic Goal Management
=========================================

Goal lifecycle management, priority adjustment, conflict resolution, and
adaptive goal pursuit for intelligent agent systems.

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import heapq


class GoalStatus(Enum):
    """Goal lifecycle states."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ACHIEVED = "achieved"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """Goal with priority and lifecycle management."""
    name: str
    description: str
    priority: float = 1.0
    status: GoalStatus = GoalStatus.INACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    progress: float = 0.0
    parent_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    preconditions: Set[str] = field(default_factory=set)
    effects: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if goal passed deadline."""
        if self.deadline:
            return datetime.now() > self.deadline
        return False

    def time_remaining(self) -> Optional[timedelta]:
        """Get time until deadline."""
        if self.deadline:
            return self.deadline - datetime.now()
        return None

    def __lt__(self, other):
        """Compare goals by priority (for heap)."""
        return self.priority > other.priority  # Higher priority first


class GoalConflict:
    """Detected conflict between goals."""

    def __init__(self, goal1: str, goal2: str, reason: str):
        self.goal1 = goal1
        self.goal2 = goal2
        self.reason = reason
        self.detected_at = datetime.now()

    def __str__(self):
        return f"Conflict: {self.goal1} ↔ {self.goal2} ({self.reason})"


class GoalManager:
    """Manages goal lifecycle, priorities, and conflicts."""

    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.active_goals: List[Goal] = []  # Priority queue
        self.conflicts: List[GoalConflict] = []
        self.world_state: Set[str] = set()
        self.achievement_count = 0
        self.failure_count = 0

    def add_goal(self, goal: Goal):
        """Add new goal to system."""
        if goal.name in self.goals:
            raise ValueError(f"Goal {goal.name} already exists")

        self.goals[goal.name] = goal

        # Auto-activate if no preconditions
        if not goal.preconditions or goal.preconditions.issubset(self.world_state):
            self.activate_goal(goal.name)

    def activate_goal(self, goal_name: str):
        """Activate a goal for pursuit."""
        if goal_name not in self.goals:
            raise ValueError(f"Goal {goal_name} not found")

        goal = self.goals[goal_name]

        # Check preconditions
        if goal.preconditions and not goal.preconditions.issubset(self.world_state):
            return False

        # Check for conflicts
        conflicts = self._detect_conflicts(goal)
        if conflicts:
            self.conflicts.extend(conflicts)
            return False

        goal.status = GoalStatus.ACTIVE
        heapq.heappush(self.active_goals, goal)
        return True

    def suspend_goal(self, goal_name: str, reason: str = ""):
        """Temporarily suspend goal."""
        goal = self.goals.get(goal_name)
        if goal and goal.status == GoalStatus.ACTIVE:
            goal.status = GoalStatus.SUSPENDED
            goal.metadata['suspend_reason'] = reason
            self._remove_from_active(goal_name)

    def resume_goal(self, goal_name: str):
        """Resume suspended goal."""
        goal = self.goals.get(goal_name)
        if goal and goal.status == GoalStatus.SUSPENDED:
            self.activate_goal(goal_name)

    def abandon_goal(self, goal_name: str, reason: str = ""):
        """Abandon goal permanently."""
        goal = self.goals.get(goal_name)
        if goal:
            goal.status = GoalStatus.ABANDONED
            goal.metadata['abandon_reason'] = reason
            self._remove_from_active(goal_name)

    def achieve_goal(self, goal_name: str):
        """Mark goal as achieved."""
        goal = self.goals.get(goal_name)
        if goal:
            goal.status = GoalStatus.ACHIEVED
            goal.progress = 1.0
            self.achievement_count += 1
            self._remove_from_active(goal_name)

            # Apply effects to world state
            self.world_state.update(goal.effects)

            # Check if parent goal can be activated
            if goal.parent_goal:
                self._check_parent_goal(goal.parent_goal)

    def fail_goal(self, goal_name: str, reason: str = ""):
        """Mark goal as failed."""
        goal = self.goals.get(goal_name)
        if goal:
            goal.status = GoalStatus.FAILED
            goal.metadata['failure_reason'] = reason
            self.failure_count += 1
            self._remove_from_active(goal_name)

    def update_priority(self, goal_name: str, new_priority: float):
        """Adjust goal priority dynamically."""
        goal = self.goals.get(goal_name)
        if goal:
            goal.priority = new_priority

            # Rebuild priority queue if active
            if goal.status == GoalStatus.ACTIVE:
                self._rebuild_priority_queue()

    def update_progress(self, goal_name: str, progress: float):
        """Update goal progress (0.0 to 1.0)."""
        goal = self.goals.get(goal_name)
        if goal:
            goal.progress = max(0.0, min(1.0, progress))

            if goal.progress >= 1.0:
                self.achieve_goal(goal_name)

    def get_current_goal(self) -> Optional[Goal]:
        """Get highest priority active goal."""
        if self.active_goals:
            return self.active_goals[0]
        return None

    def get_goals_by_status(self, status: GoalStatus) -> List[Goal]:
        """Get all goals with given status."""
        return [g for g in self.goals.values() if g.status == status]

    def _detect_conflicts(self, new_goal: Goal) -> List[GoalConflict]:
        """Detect conflicts between new goal and active goals."""
        conflicts = []

        for active_goal in self.active_goals:
            # Check mutually exclusive effects
            if new_goal.effects & active_goal.effects:
                conflicts.append(GoalConflict(
                    new_goal.name,
                    active_goal.name,
                    "Conflicting effects"
                ))

            # Check if new goal's effects negate active goal's preconditions
            if new_goal.effects and active_goal.preconditions:
                # Simplified conflict detection
                pass

        return conflicts

    def _remove_from_active(self, goal_name: str):
        """Remove goal from active queue."""
        self.active_goals = [g for g in self.active_goals if g.name != goal_name]
        heapq.heapify(self.active_goals)

    def _rebuild_priority_queue(self):
        """Rebuild priority queue after priority changes."""
        heapq.heapify(self.active_goals)

    def _check_parent_goal(self, parent_name: str):
        """Check if parent goal's subgoals are all achieved."""
        parent = self.goals.get(parent_name)
        if not parent:
            return

        # Check if all subgoals achieved
        all_achieved = all(
            self.goals[sg].status == GoalStatus.ACHIEVED
            for sg in parent.subgoals
            if sg in self.goals
        )

        if all_achieved:
            self.achieve_goal(parent_name)

    def monitor_deadlines(self):
        """Check for expired goals and adjust priorities."""
        for goal in self.active_goals:
            if goal.is_expired():
                self.fail_goal(goal.name, "Deadline exceeded")
            elif goal.deadline:
                # Increase priority as deadline approaches
                time_remaining = goal.time_remaining()
                if time_remaining and time_remaining < timedelta(hours=1):
                    goal.priority *= 1.5

        self._rebuild_priority_queue()

    def resolve_conflicts(self, strategy: str = "priority"):
        """Resolve detected conflicts."""
        for conflict in self.conflicts:
            goal1 = self.goals.get(conflict.goal1)
            goal2 = self.goals.get(conflict.goal2)

            if not goal1 or not goal2:
                continue

            if strategy == "priority":
                # Suspend lower priority goal
                if goal1.priority < goal2.priority:
                    self.suspend_goal(goal1.name, f"Conflict with {goal2.name}")
                else:
                    self.suspend_goal(goal2.name, f"Conflict with {goal1.name}")

        self.conflicts.clear()

    def get_statistics(self) -> Dict:
        """Get goal system statistics."""
        return {
            'total_goals': len(self.goals),
            'active_goals': len(self.active_goals),
            'achieved': self.achievement_count,
            'failed': self.failure_count,
            'conflicts': len(self.conflicts),
            'world_state_size': len(self.world_state)
        }


def demo():
    """Demonstration of Goal Reasoning."""
    print("Goal Reasoning - Dynamic Goal Management Demo")
    print("=" * 70)

    manager = GoalManager()

    # Define goals
    print("\n1️⃣  Creating Goals")
    print("-" * 70)

    # High-level goal
    complete_project = Goal(
        name="complete_project",
        description="Complete the software project",
        priority=10.0,
        deadline=datetime.now() + timedelta(days=7),
        subgoals=["write_code", "write_tests", "write_docs"]
    )
    manager.add_goal(complete_project)
    print(f"✓ Added: {complete_project.name} (priority: {complete_project.priority})")

    # Subgoals
    write_code = Goal(
        name="write_code",
        description="Implement main functionality",
        priority=8.0,
        parent_goal="complete_project",
        effects={"code_written"}
    )
    manager.add_goal(write_code)
    print(f"✓ Added: {write_code.name} (priority: {write_code.priority})")

    write_tests = Goal(
        name="write_tests",
        description="Write unit tests",
        priority=7.0,
        parent_goal="complete_project",
        preconditions={"code_written"},
        effects={"tests_written"}
    )
    manager.add_goal(write_tests)
    print(f"✓ Added: {write_tests.name} (priority: {write_tests.priority})")

    write_docs = Goal(
        name="write_docs",
        description="Write documentation",
        priority=5.0,
        parent_goal="complete_project",
        effects={"docs_written"}
    )
    manager.add_goal(write_docs)
    print(f"✓ Added: {write_docs.name} (priority: {write_docs.priority})")

    # Personal goal (lower priority)
    learn_skill = Goal(
        name="learn_skill",
        description="Learn new programming language",
        priority=3.0
    )
    manager.add_goal(learn_skill)
    print(f"✓ Added: {learn_skill.name} (priority: {learn_skill.priority})")

    # Goal selection
    print("\n2️⃣  Goal Selection")
    print("-" * 70)

    current = manager.get_current_goal()
    if current:
        print(f"Current goal: {current.name} (priority: {current.priority})")
        print(f"Description: {current.description}")

    print(f"\nActive goals: {len(manager.active_goals)}")
    for goal in sorted(manager.active_goals, key=lambda g: g.priority, reverse=True):
        print(f"  • {goal.name} (priority: {goal.priority})")

    # Progress updates
    print("\n3️⃣  Progress Updates")
    print("-" * 70)

    manager.update_progress("write_code", 0.5)
    print(f"Updated write_code progress: {manager.goals['write_code'].progress:.0%}")

    manager.update_progress("write_code", 1.0)
    print(f"✓ write_code achieved! (progress: {manager.goals['write_code'].progress:.0%})")

    # Activate dependent goal
    manager.activate_goal("write_tests")
    print(f"✓ write_tests activated (preconditions met)")

    # Priority adjustments
    print("\n4️⃣  Dynamic Priority Adjustment")
    print("-" * 70)

    print(f"Original priority - write_docs: {manager.goals['write_docs'].priority}")
    manager.update_priority("write_docs", 9.0)
    print(f"Updated priority - write_docs: {manager.goals['write_docs'].priority}")

    current = manager.get_current_goal()
    print(f"\nNew current goal: {current.name if current else 'None'}")

    # Conflict detection
    print("\n5️⃣  Conflict Resolution")
    print("-" * 70)

    conflicting_goal = Goal(
        name="rewrite_code",
        description="Rewrite with different approach",
        priority=6.0,
        effects={"code_written"}  # Conflicts with write_code
    )
    manager.add_goal(conflicting_goal)

    if manager.conflicts:
        print(f"Conflicts detected: {len(manager.conflicts)}")
        for conflict in manager.conflicts:
            print(f"  • {conflict}")

        manager.resolve_conflicts(strategy="priority")
        print(f"✓ Conflicts resolved using priority strategy")

    # Goal lifecycle
    print("\n6️⃣  Goal Lifecycle Management")
    print("-" * 70)

    manager.suspend_goal("learn_skill", "Low priority during deadline")
    print(f"Suspended: learn_skill ({manager.goals['learn_skill'].status.value})")

    manager.achieve_goal("write_tests")
    print(f"✓ Achieved: write_tests ({manager.goals['write_tests'].status.value})")

    manager.achieve_goal("write_docs")
    print(f"✓ Achieved: write_docs ({manager.goals['write_docs'].status.value})")

    # Check parent goal
    if manager.goals['complete_project'].status == GoalStatus.ACHIEVED:
        print(f"✓ Parent goal achieved: complete_project")

    # Statistics
    print("\n7️⃣  Goal System Statistics")
    print("-" * 70)

    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nGoals by status:")
    for status in GoalStatus:
        goals = manager.get_goals_by_status(status)
        if goals:
            print(f"  {status.value}: {[g.name for g in goals]}")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
