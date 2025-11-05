# Task Planning

Hierarchical Task Network (HTN) planning for decomposing complex tasks into executable actions.

## Features

- **Hierarchical Task Networks**: Decompose abstract tasks into primitive actions
- **Method Selection**: Choose appropriate decomposition methods
- **Constraint Handling**: Enforce preconditions and postconditions
- **Plan Validation**: Verify plan correctness before execution
- **Backtracking**: Handle planning failures with intelligent backtracking
- **Replanning**: Dynamically adjust plans when conditions change
- **Task Ordering**: Manage dependencies and execution order
- **Resource Management**: Track and allocate resources during planning

## Quick Start

```python
from task_planning import TaskPlanner, Task, Method

# Initialize planner
planner = TaskPlanner()

# Define abstract task
travel_task = Task(
    name='travel_to_destination',
    task_type='abstract',
    parameters={'from': 'home', 'to': 'office'}
)

# Define decomposition methods
drive_method = Method(
    name='drive',
    preconditions=['has_car', 'has_license'],
    subtasks=[
        Task('get_keys', 'primitive'),
        Task('start_car', 'primitive'),
        Task('navigate', 'primitive', {'destination': 'office'}),
        Task('park', 'primitive')
    ]
)

transit_method = Method(
    name='public_transit',
    preconditions=['has_transit_card'],
    subtasks=[
        Task('walk_to_station', 'primitive'),
        Task('take_train', 'primitive', {'line': 'red'}),
        Task('walk_to_destination', 'primitive')
    ]
)

# Register methods
planner.add_method(travel_task.name, drive_method)
planner.add_method(travel_task.name, transit_method)

# Generate plan
plan = planner.plan(
    initial_state={'has_car': True, 'has_license': True},
    goal_task=travel_task
)

# Execute plan
for action in plan.actions:
    print(f"Execute: {action.name}")
```

## Use Cases

- **Robot Task Planning**: Decompose high-level robot tasks into low-level motions
- **Workflow Automation**: Plan multi-step business processes
- **Game AI**: Create complex NPC behaviors through task hierarchies
- **Manufacturing**: Plan assembly sequences with constraints
- **Logistics**: Route planning with multiple objectives
- **Personal Assistants**: Break down user requests into executable steps

## HTN Planning Concepts

### Tasks
- **Abstract Tasks**: High-level goals requiring decomposition
- **Primitive Tasks**: Directly executable actions
- **Compound Tasks**: Tasks with multiple subtasks

### Methods
Methods define how to decompose abstract tasks:
```python
Method(
    name='method_name',
    preconditions=['condition1', 'condition2'],
    subtasks=[task1, task2, task3],
    constraints=['ordering', 'resource_limits']
)
```

### Planning Process
1. Start with abstract goal task
2. Select applicable method based on current state
3. Decompose into subtasks
4. Recursively plan for abstract subtasks
5. Validate constraints and preconditions
6. Return sequence of primitive actions

## Advanced Features

### Conditional Planning
```python
# Plan with branching based on state
conditional_method = Method(
    name='conditional_travel',
    preconditions=[],
    subtasks=[
        Task('check_weather', 'primitive'),
        ConditionalTask(
            condition='is_raining',
            if_true=Task('take_umbrella', 'primitive'),
            if_false=Task('wear_sunglasses', 'primitive')
        )
    ]
)
```

### Resource Constraints
```python
# Track resource usage during planning
planner.add_resource_constraint(
    resource='battery',
    initial_amount=100,
    consumption={'drive': 20, 'walk': 0}
)
```

### Temporal Constraints
```python
# Add timing constraints
planner.add_temporal_constraint(
    task1='pickup',
    task2='delivery',
    constraint='before',  # task1 must occur before task2
    max_delay=3600  # Maximum 1 hour between tasks
)
```

## Best Practices

- Design methods with clear preconditions
- Keep task hierarchies shallow when possible
- Use primitive tasks for atomic actions
- Implement efficient method selection
- Cache plan results for repeated tasks
- Handle planning failures gracefully
- Test methods independently
- Document task hierarchies clearly

## Author

Brill Consulting
