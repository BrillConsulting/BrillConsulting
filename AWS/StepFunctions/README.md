# AWS Step Functions

Serverless workflow orchestration using Amazon States Language (ASL) for coordinating distributed applications.

## Features

- **State Machines**: Define workflows using Amazon States Language
- **Standard Workflows**: Long-running, durable workflows (up to 1 year)
- **Express Workflows**: High-volume, short-duration workflows (up to 5 minutes)
- **Error Handling**: Built-in retry and catch mechanisms
- **Parallel Execution**: Run multiple branches concurrently
- **Choice States**: Conditional branching based on input
- **Wait States**: Delays and scheduled execution
- **Activity Workers**: Integrate with external systems

## Quick Start

```python
from aws_stepfunctions import StepFunctionsManager

# Initialize
sfn = StepFunctionsManager(region='us-east-1')

# Define workflow
definition = {
    'Comment': 'Data processing workflow',
    'StartAt': 'ProcessData',
    'States': {
        'ProcessData': {
            'Type': 'Task',
            'Resource': 'arn:aws:lambda:us-east-1:123456789012:function:process',
            'Next': 'SaveResults'
        },
        'SaveResults': {
            'Type': 'Task',
            'Resource': 'arn:aws:lambda:us-east-1:123456789012:function:save',
            'End': True
        }
    }
}

# Create state machine
state_machine = sfn.create_state_machine(
    name='DataProcessingWorkflow',
    definition=definition,
    role_arn='arn:aws:iam::123456789012:role/StepFunctionsRole'
)

# Start execution
execution = sfn.start_execution(
    state_machine_arn=state_machine['state_machine_arn'],
    input_data={'bucket': 's3://my-data', 'key': 'file.csv'}
)

# Check status
status = sfn.describe_execution(execution['execution_arn'])
```

## Use Cases

- **Data Processing**: ETL pipelines and batch jobs
- **Microservices Orchestration**: Coordinate service interactions
- **IT Automation**: Infrastructure provisioning and maintenance
- **ML Workflows**: Training, validation, and deployment pipelines
- **Human Approval**: Workflows requiring manual intervention

## State Types

- **Task**: Execute work (Lambda, ECS, Batch, etc.)
- **Choice**: Conditional branching
- **Parallel**: Execute branches concurrently
- **Wait**: Delay for specified time
- **Pass**: Pass input to output
- **Succeed/Fail**: Terminal states
- **Map**: Iterate over array items

## Error Handling

Retry transient errors:
```python
'Retry': [{
    'ErrorEquals': ['States.Timeout'],
    'IntervalSeconds': 2,
    'MaxAttempts': 3,
    'BackoffRate': 2.0
}]
```

Catch and handle errors:
```python
'Catch': [{
    'ErrorEquals': ['States.ALL'],
    'Next': 'HandleError'
}]
```

## Author

Brill Consulting
