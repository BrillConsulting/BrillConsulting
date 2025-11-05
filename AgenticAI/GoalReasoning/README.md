# Goal Reasoning

Dynamic goal management system for AI agents with priority adjustment and conflict resolution.

## Features

- **Goal Lifecycle**: Create, activate, suspend, and complete goals
- **Priority Management**: Dynamic priority adjustment based on context
- **Conflict Resolution**: Identify and resolve conflicting goals
- **Goal Decomposition**: Break complex goals into subgoals
- **Achievement Tracking**: Monitor progress toward goals
- **Dynamic Adoption**: Adopt new goals based on situation
- **Abandonment Strategies**: Intelligently abandon unachievable goals

## Quick Start

```python
from goal_reasoning import GoalManager, Goal

# Initialize manager
gm = GoalManager()

# Create goals
main_goal = Goal(
    id="complete_project",
    description="Complete quarterly project",
    priority=0.9,
    deadline="2024-12-31"
)

subgoal1 = Goal(
    id="write_code",
    description="Implement features",
    priority=0.8,
    parent_goal="complete_project"
)

# Add goals
gm.add_goal(main_goal)
gm.add_goal(subgoal1)

# Adjust priorities based on context
gm.adjust_priority("write_code", increase_by=0.1, reason="urgent_deadline")

# Check for conflicts
conflicts = gm.detect_conflicts()

# Resolve conflicts
if conflicts:
    gm.resolve_conflict(conflicts[0], strategy="prioritize_urgent")
```

## Use Cases

- **Task Management**: Dynamic task prioritization
- **Robot Planning**: Adapt goals based on environment
- **Game AI**: Goal-driven NPC behavior
- **Personal Assistants**: Manage user objectives

## Best Practices

- Define clear goal success criteria
- Monitor goal progress regularly
- Handle goal dependencies explicitly
- Implement timeout mechanisms
- Log goal state changes

## Author

Brill Consulting
