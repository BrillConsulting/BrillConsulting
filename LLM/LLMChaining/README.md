# LLMChaining - Production-Ready LLM Chain Orchestration System

A comprehensive, enterprise-grade framework for building, composing, and executing complex LLM chains with advanced features including parallel execution, conditional routing, error recovery, state management, retry logic, and async support.

## Features

### Core Capabilities
- **Sequential Chains**: Execute multiple chains in sequence with automatic state passing
- **Parallel Execution**: Run chains concurrently with configurable worker pools and result aggregation
- **Conditional Routing**: Route execution based on dynamic conditions
- **Chain Composition**: Build complex workflows from simple building blocks
- **Error Recovery**: Automatic error handling with configurable recovery strategies
- **State Management**: Persistent state tracking across chain executions with history
- **Retry Logic**: Exponential backoff with jitter for fault tolerance
- **Async Support**: Full asynchronous execution for high-performance workflows
- **Batch Processing**: Execute chains on multiple inputs in parallel
- **Validation**: Input validation with custom validators
- **Monitoring**: Comprehensive execution statistics and logging

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Chain

```python
from llmchaining import LLMChain, ChainComposer

# Create a simple LLM chain
chain = LLMChain(
    name="simple_llm",
    llm_function=your_llm_function,
    prompt_template="Question: {input}\nAnswer:"
)

# Execute the chain
result = chain.execute("What is AI?")
print(result.output)
```

### Sequential Chain

```python
from llmchaining import TransformChain, ChainComposer

# Create transformation steps
preprocess = TransformChain(
    name="preprocess",
    transform_function=lambda x, s: x.upper()
)

llm_process = LLMChain(
    name="llm",
    llm_function=your_llm_function
)

postprocess = TransformChain(
    name="postprocess",
    transform_function=lambda x, s: f"Result: {x}"
)

# Compose into a sequential pipeline
pipeline = ChainComposer.sequence(
    preprocess,
    llm_process,
    postprocess,
    name="full_pipeline"
)

# Execute the pipeline
result = pipeline.execute("input data")
```

### Parallel Chain

```python
from llmchaining import ParallelChain, ChainComposer

# Create parallel chains
sentiment_chain = LLMChain(
    name="sentiment",
    llm_function=sentiment_analyzer
)

summary_chain = LLMChain(
    name="summary",
    llm_function=summarizer
)

entities_chain = LLMChain(
    name="entities",
    llm_function=entity_extractor
)

# Execute in parallel
parallel = ChainComposer.parallel(
    sentiment_chain,
    summary_chain,
    entities_chain,
    name="parallel_analysis",
    aggregate=lambda results: {
        'sentiment': results[0],
        'summary': results[1],
        'entities': results[2]
    }
)

result = parallel.execute("Your text here")
```

### Conditional Chain

```python
from llmchaining import ConditionalChain, ChainComposer

# Create conditional routing
long_form = LLMChain(name="long", llm_function=detailed_response)
short_form = LLMChain(name="short", llm_function=brief_response)

conditional = ChainComposer.conditional(
    condition=lambda x, s: len(x) > 100,
    if_chain=long_form,
    else_chain=short_form,
    name="length_based_routing"
)

result = conditional.execute("Your input")
```

### Router Chain

```python
from llmchaining import RouterChain, ChainComposer

# Define routes
routes = {
    'technical': technical_chain,
    'creative': creative_chain,
    'analytical': analytical_chain
}

# Create router function
def route_by_type(input_data, state):
    if 'code' in input_data.lower():
        return 'technical'
    elif 'story' in input_data.lower():
        return 'creative'
    else:
        return 'analytical'

# Create router chain
router = ChainComposer.router(
    router_function=route_by_type,
    routes=routes,
    default=default_chain,
    name="task_router"
)

result = router.execute("Write a code function")
```

## Advanced Features

### Retry Strategy

```python
from llmchaining import RetryStrategy, LLMChain

# Configure retry behavior
retry_strategy = RetryStrategy(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True
)

chain = LLMChain(
    name="resilient_chain",
    llm_function=your_llm_function,
    retry_strategy=retry_strategy
)
```

### Error Recovery

```python
def error_handler(error, input_data, state):
    """Custom error handler"""
    logger.error(f"Error occurred: {error}")
    # Return fallback response
    return f"Fallback response for: {input_data}"

chain = LLMChain(
    name="safe_chain",
    llm_function=your_llm_function,
    error_handler=error_handler
)
```

### Input Validation

```python
def validate_length(input_data):
    """Validate input length"""
    return len(str(input_data)) <= 1000

def validate_format(input_data):
    """Validate input format"""
    return isinstance(input_data, str)

chain = LLMChain(
    name="validated_chain",
    llm_function=your_llm_function,
    validators=[validate_length, validate_format]
)
```

### State Management

```python
from llmchaining import ChainState

# Create and use state
state = ChainState()

# Set values
state.set('user_id', '12345')
state.set('context', 'conversation history')

# Get values
user_id = state.get('user_id')

# Update multiple values
state.update({
    'language': 'en',
    'temperature': 0.7
})

# Create snapshot
snapshot = state.snapshot()

# View history
for event in state.history:
    print(event)
```

### Async Execution

```python
import asyncio
from llmchaining import ParallelChain

async def main():
    # Execute chains asynchronously
    result = await parallel_chain.execute_async("input")

    # Batch execution
    results = await system.batch_execute_async(
        "chain_name",
        ["input1", "input2", "input3"]
    )

asyncio.run(main())
```

### System Orchestration

```python
from llmchaining import LLMChainingSystem

# Create system
system = LLMChainingSystem()

# Register chains
system.register_chain(pipeline1)
system.register_chain(pipeline2)

# Execute by name
result = system.execute_chain("pipeline1", input_data)

# Batch processing
results = system.batch_execute(
    "pipeline1",
    [input1, input2, input3],
    max_workers=4
)

# Get statistics
stats = system.get_statistics()
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Average execution time: {stats['average_time']:.3f}s")

# Export/Import state
system.export_state("state.json")
system.import_state("state.json")

# Reset
system.reset()
```

## Chain Types

### LLMChain
Execute LLM functions with prompt templating and state integration.

**Parameters:**
- `name`: Chain identifier
- `llm_function`: Callable that executes the LLM
- `prompt_template`: Optional template string with {input} and state variables
- `retry_strategy`: Optional retry configuration
- `error_handler`: Optional error recovery function
- `validators`: Optional list of validation functions

### TransformChain
Apply data transformations with access to chain state.

**Parameters:**
- `name`: Chain identifier
- `transform_function`: Callable(input_data, state) -> output
- `retry_strategy`: Optional retry configuration
- `error_handler`: Optional error recovery function
- `validators`: Optional list of validation functions

### SequentialChain
Execute chains in sequence, passing output from one to the next.

**Parameters:**
- `name`: Chain identifier
- `chains`: List of chains to execute sequentially
- `retry_strategy`: Optional retry configuration
- `error_handler`: Optional error recovery function

### ParallelChain
Execute chains concurrently and aggregate results.

**Parameters:**
- `name`: Chain identifier
- `chains`: List of chains to execute in parallel
- `max_workers`: Optional max thread pool size
- `aggregate_function`: Function to combine results
- `retry_strategy`: Optional retry configuration
- `error_handler`: Optional error recovery function

### ConditionalChain
Route execution based on runtime conditions.

**Parameters:**
- `name`: Chain identifier
- `condition`: Callable(input_data, state) -> bool
- `if_chain`: Chain to execute if condition is True
- `else_chain`: Optional chain to execute if condition is False
- `retry_strategy`: Optional retry configuration
- `error_handler`: Optional error recovery function

### RouterChain
Route to different chains based on input analysis.

**Parameters:**
- `name`: Chain identifier
- `routes`: Dict mapping route keys to chains
- `router_function`: Callable(input_data, state) -> route_key
- `default_chain`: Optional default chain if route not found
- `retry_strategy`: Optional retry configuration
- `error_handler`: Optional error recovery function

## Chain Results

All chain executions return a `ChainResult` object:

```python
@dataclass
class ChainResult:
    status: ChainStatus          # SUCCESS, FAILED, etc.
    output: Any                  # Chain output
    error: Optional[Exception]   # Error if failed
    execution_time: float        # Execution time in seconds
    metadata: Dict[str, Any]     # Additional metadata
    retries: int                 # Number of retries performed
```

## Logging

The system uses Python's built-in logging:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Get logger
logger = logging.getLogger('llmchaining')
```

## Best Practices

### 1. Chain Composition
Build complex workflows from simple, reusable chains:

```python
# Good: Modular, reusable chains
preprocess = TransformChain(name="prep", transform_function=clean_text)
llm = LLMChain(name="llm", llm_function=model)
postprocess = TransformChain(name="post", transform_function=format_output)
pipeline = ChainComposer.sequence(preprocess, llm, postprocess)

# Avoid: Monolithic chains
```

### 2. Error Handling
Always configure error handlers for production:

```python
def safe_handler(error, input_data, state):
    logging.error(f"Chain failed: {error}")
    return {"error": str(error), "input": input_data}

chain = LLMChain(
    name="production_chain",
    llm_function=llm,
    error_handler=safe_handler,
    retry_strategy=RetryStrategy(max_retries=3)
)
```

### 3. State Management
Use state for context sharing between chains:

```python
# Store intermediate results
chain1 = TransformChain(
    name="analyzer",
    transform_function=lambda x, s: s.set('analysis', analyze(x))
)

# Access in later chains
chain2 = LLMChain(
    name="responder",
    llm_function=llm,
    prompt_template="Analysis: {analysis}\nResponse:"
)
```

### 4. Parallel Execution
Use parallel chains for independent operations:

```python
# Good: Independent operations in parallel
parallel = ChainComposer.parallel(
    sentiment_chain,
    entity_chain,
    summary_chain
)

# Avoid: Dependent operations in parallel
```

### 5. Validation
Validate inputs early to fail fast:

```python
chain = LLMChain(
    name="validated",
    llm_function=llm,
    validators=[
        lambda x: len(x) > 0,
        lambda x: len(x) < 10000,
        lambda x: isinstance(x, str)
    ]
)
```

## Examples

### Example 1: Document Processing Pipeline

```python
# Define processing stages
clean = TransformChain(
    name="clean",
    transform_function=lambda x, s: clean_text(x)
)

extract = ParallelChain(
    name="extract",
    chains=[
        LLMChain(name="entities", llm_function=extract_entities),
        LLMChain(name="keywords", llm_function=extract_keywords),
        LLMChain(name="summary", llm_function=summarize)
    ],
    aggregate_function=lambda r: {
        'entities': r[0], 'keywords': r[1], 'summary': r[2]
    }
)

enrich = TransformChain(
    name="enrich",
    transform_function=lambda x, s: enrich_metadata(x)
)

# Build pipeline
pipeline = ChainComposer.sequence(clean, extract, enrich)

# Execute
result = pipeline.execute(document_text)
```

### Example 2: Multi-Model Ensemble

```python
# Define models
gpt4_chain = LLMChain(name="gpt4", llm_function=gpt4_api)
claude_chain = LLMChain(name="claude", llm_function=claude_api)
llama_chain = LLMChain(name="llama", llm_function=llama_api)

# Run in parallel
ensemble = ChainComposer.parallel(
    gpt4_chain,
    claude_chain,
    llama_chain,
    name="ensemble",
    aggregate=aggregate_responses
)

result = ensemble.execute(prompt)
```

### Example 3: Adaptive Question Answering

```python
# Route based on question type
def classify_question(question, state):
    if any(word in question.lower() for word in ['code', 'function', 'program']):
        return 'technical'
    elif any(word in question.lower() for word in ['calculate', 'compute', 'math']):
        return 'mathematical'
    else:
        return 'general'

routes = {
    'technical': technical_qa_chain,
    'mathematical': math_solver_chain,
    'general': general_qa_chain
}

router = ChainComposer.router(
    router_function=classify_question,
    routes=routes,
    name="qa_router"
)

answer = router.execute("How do I implement a binary search?")
```

## Performance Considerations

### Threading vs Async
- Use `ParallelChain` for CPU-bound operations
- Use `execute_async()` for I/O-bound operations
- Combine both for maximum performance

### Batch Processing
Process multiple inputs efficiently:

```python
# Instead of:
for input in inputs:
    result = chain.execute(input)

# Use:
results = system.batch_execute("chain_name", inputs, max_workers=10)
```

### State Management
- Use isolated states for parallel executions
- Share global state only when necessary
- Clear state periodically to prevent memory leaks

## Monitoring and Debugging

### Statistics

```python
stats = system.get_statistics()
print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Average time: {stats['average_time']:.3f}s")
print(f"Total retries: {stats['total_retries']}")
```

### Chain Result Inspection

```python
result = chain.execute(input_data)

if result.status == ChainStatus.SUCCESS:
    print(f"Output: {result.output}")
    print(f"Execution time: {result.execution_time:.3f}s")
else:
    print(f"Error: {result.error}")
    print(f"Retries: {result.retries}")
```

### State History

```python
state = ChainState()
# ... execute chains ...

for event in state.history:
    print(f"{event['timestamp']}: {event['action']} - {event.get('key', '')}")
```

## Testing

Run the included demonstrations:

```bash
python llmchaining.py
```

This will execute comprehensive examples of all features.

## Architecture

```
LLMChainingSystem
├── Chain Registry
├── Global State
└── Execution History

BaseChain (Abstract)
├── LLMChain
├── TransformChain
├── SequentialChain
├── ParallelChain
├── ConditionalChain
└── RouterChain

Supporting Components
├── ChainState (State Management)
├── ChainResult (Result Container)
├── RetryStrategy (Retry Logic)
├── ChainComposer (Builder Pattern)
└── Error Handlers
```

## Production Checklist

- [ ] Configure appropriate retry strategies
- [ ] Implement error handlers for all chains
- [ ] Add input validators
- [ ] Set up proper logging
- [ ] Monitor execution statistics
- [ ] Implement state persistence
- [ ] Configure thread pool sizes
- [ ] Add timeouts for long-running chains
- [ ] Set up alerting for failures
- [ ] Document chain dependencies

## Contributing

This is a production-ready framework designed by BrillConsulting. For enhancements or issues, please contact the development team.

## License

Copyright (c) BrillConsulting. All rights reserved.

## Support

For support and questions:
- Review the comprehensive examples above
- Check execution logs for debugging
- Consult the inline code documentation
- Contact BrillConsulting support team
