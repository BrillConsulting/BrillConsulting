# LLM Routing System

**Author:** BrillConsulting

A production-ready intelligent LLM routing system that optimally selects and routes queries to the best available language models based on multiple factors including cost, latency, quality, and load.

## Overview

The LLM Routing System provides intelligent, automated routing of queries to Large Language Models (LLMs) with advanced features including:

- **Intelligent Model Selection**: Multiple routing strategies (cost-optimized, latency-optimized, quality-optimized, balanced)
- **Cost Optimization**: Minimize costs while maintaining quality requirements
- **Latency-Based Routing**: Select fastest models based on real-time performance metrics
- **Load Balancing**: Distribute requests across models to prevent overload
- **Fallback Strategies**: Automatic failover to backup models on errors
- **Performance Tracking**: Real-time metrics collection and statistics
- **Router Ensembles**: Combine multiple strategies for optimal decisions
- **Request Caching**: Cache routing decisions for similar queries
- **Thread-Safe**: Safe for concurrent operations

## Features

### Core Components

1. **Model Selectors**
   - `CostOptimizedSelector`: Minimize cost per request
   - `LatencyOptimizedSelector`: Minimize response time
   - `QualityOptimizedSelector`: Maximize output quality
   - `BalancedSelector`: Balance cost, latency, and quality
   - `RoundRobinSelector`: Distribute evenly across models
   - `LeastLoadedSelector`: Route to least busy model

2. **Performance Monitoring**
   - Real-time latency tracking
   - Cost tracking per model
   - Success rate monitoring
   - Active request counting
   - Rolling window statistics (configurable window size)

3. **Load Management**
   - Concurrent request limiting
   - Automatic load balancing
   - Request queuing and throttling
   - Model availability tracking

4. **Fallback Handling**
   - Automatic retry with fallback models
   - Intelligent fallback selection
   - Error tracking and reporting

5. **Caching System**
   - Query result caching
   - Configurable TTL (Time To Live)
   - Size-limited cache with LRU eviction
   - Thread-safe implementation

## Installation

```bash
# Clone the repository
cd /path/to/BrillConsulting/LLM/LLMRouting

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from l_l_m_routing import LLMRoutingManager

# Initialize with default configuration
manager = LLMRoutingManager()

# Route a simple query
decision = manager.route_query(
    query="What is the capital of France?",
    capability="simple_qa",
    strategy="cost_optimized"
)

print(f"Selected Model: {decision['selected_model']['name']}")
print(f"Expected Cost: ${decision['expected_cost']:.4f}")
print(f"Expected Latency: {decision['expected_latency_ms']:.0f}ms")

# Execute query with automatic fallback
result = manager.execute_query(
    query="Write a Python function to sort a list",
    capability="code_generation",
    strategy="balanced",
    mock_execution=True  # Set False for real execution
)

print(f"Success: {result['success']}")
print(f"Model Used: {result['model_used']}")
```

## Architecture

### Class Hierarchy

```
LLMRoutingManager (High-level API)
    └── LLMRouter (Core routing engine)
        ├── ModelSelector (Abstract base)
        │   ├── CostOptimizedSelector
        │   ├── LatencyOptimizedSelector
        │   ├── QualityOptimizedSelector
        │   ├── BalancedSelector
        │   ├── RoundRobinSelector
        │   └── LeastLoadedSelector
        ├── PerformanceMetrics
        ├── LoadBalancer
        ├── FallbackStrategy
        ├── CacheManager
        └── RouterEnsemble
```

### Data Models

- **ModelConfig**: Configuration for LLM models
- **QueryRequest**: Query routing request with constraints
- **RoutingDecision**: Routing decision with fallbacks
- **ExecutionResult**: Query execution result with metrics

## Configuration

### Model Configuration

```python
config = {
    'enable_caching': True,
    'enable_ensemble': True,
    'cache_ttl': 300,  # seconds
    'max_workers': 10,
    'models': [
        {
            'model_id': 'gpt-4',
            'name': 'GPT-4',
            'provider': 'openai',
            'cost_per_1k_tokens': 0.03,
            'max_tokens': 8192,
            'avg_latency_ms': 2000,
            'capabilities': ['complex_reasoning', 'code_generation', 'analysis'],
            'quality_score': 0.95,
            'max_concurrent_requests': 50
        },
        # Add more models...
    ]
}

manager = LLMRoutingManager(config=config)
```

### Routing Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `cost_optimized` | Minimizes cost per request | Budget-conscious applications |
| `latency_optimized` | Minimizes response time | Real-time applications |
| `quality_optimized` | Maximizes output quality | Critical tasks requiring accuracy |
| `balanced` | Balances all factors | General purpose applications |
| `round_robin` | Even distribution | Testing and benchmarking |
| `least_loaded` | Routes to least busy model | High-traffic scenarios |

### Model Capabilities

- `simple_qa`: Simple question answering
- `complex_reasoning`: Complex analytical tasks
- `code_generation`: Programming and code tasks
- `creative_writing`: Content creation
- `summarization`: Text summarization
- `translation`: Language translation
- `analysis`: Data and text analysis
- `general`: General purpose tasks

## API Reference

### LLMRoutingManager

Main high-level interface for the routing system.

#### Methods

**`route_query(query, capability='general', strategy='balanced', **kwargs)`**

Route a query and return routing decision.

Parameters:
- `query` (str): Query text
- `capability` (str): Required capability
- `strategy` (str): Routing strategy
- `max_tokens` (int): Maximum tokens
- `max_cost` (float): Maximum acceptable cost
- `max_latency_ms` (float): Maximum acceptable latency
- `required_quality` (float): Minimum quality score (0-1)

Returns: Dictionary with routing decision

**`execute_query(query, capability='general', strategy='balanced', mock_execution=True, **kwargs)`**

Execute query with automatic routing and fallback.

Parameters: Same as `route_query` plus:
- `mock_execution` (bool): Use mock execution for testing

Returns: Dictionary with execution result

**`get_statistics()`**

Get comprehensive routing statistics.

Returns: Dictionary with statistics for all models

**`benchmark_strategies(test_queries, iterations=100)`**

Benchmark different routing strategies.

Parameters:
- `test_queries` (List[Dict]): List of test queries
- `iterations` (int): Number of iterations

Returns: Dictionary with benchmark results

**`export_config(filepath)`**

Export configuration to JSON file.

**`import_config(filepath)`**

Import configuration from JSON file.

### LLMRouter

Low-level routing engine.

#### Methods

**`route(request: QueryRequest) -> RoutingDecision`**

Route a query request to optimal model.

**`execute_with_fallback(request, execution_fn) -> ExecutionResult`**

Execute request with automatic fallback on failure.

**`get_statistics() -> Dict[str, Any]`**

Get routing statistics.

**`update_model_availability(model_id, is_available)`**

Update model availability status.

**`add_model(model: ModelConfig)`**

Add new model to router.

**`remove_model(model_id)`**

Remove model from router.

## Usage Examples

### Example 1: Basic Routing

```python
manager = LLMRoutingManager()

# Cost-optimized routing for simple query
decision = manager.route_query(
    query="What is 2+2?",
    capability="simple_qa",
    strategy="cost_optimized"
)
```

### Example 2: Constraint-Based Routing

```python
# Route with cost and latency constraints
decision = manager.route_query(
    query="Analyze this data...",
    capability="analysis",
    strategy="balanced",
    max_cost=0.01,
    max_latency_ms=1000,
    required_quality=0.85
)
```

### Example 3: Execute with Fallback

```python
# Execute with automatic fallback on failure
result = manager.execute_query(
    query="Generate code for...",
    capability="code_generation",
    strategy="quality_optimized"
)

if result['success']:
    print(f"Response: {result['response']}")
else:
    print(f"Error: {result['error']}")
```

### Example 4: Performance Monitoring

```python
# Get statistics after running queries
stats = manager.get_statistics()

for model_id, model_stats in stats['model_stats'].items():
    print(f"{model_id}:")
    print(f"  Avg Latency: {model_stats['avg_latency_ms']:.0f}ms")
    print(f"  Avg Cost: ${model_stats['avg_cost']:.4f}")
    print(f"  Success Rate: {model_stats['success_rate']:.1%}")
```

### Example 5: Strategy Benchmarking

```python
# Compare different strategies
test_queries = [
    {'query': 'Simple question', 'capability': 'simple_qa'},
    {'query': 'Complex analysis', 'capability': 'complex_reasoning'},
    {'query': 'Code task', 'capability': 'code_generation'}
]

benchmark = manager.benchmark_strategies(test_queries, iterations=100)

for strategy, metrics in benchmark.items():
    print(f"{strategy}:")
    print(f"  Avg Cost: ${metrics['avg_cost']:.4f}")
    print(f"  Avg Latency: {metrics['avg_latency']:.0f}ms")
    print(f"  Success Rate: {metrics['success_rate']:.1%}")
```

### Example 6: Custom Configuration

```python
# Load custom configuration
custom_config = {
    'enable_caching': True,
    'enable_ensemble': True,
    'cache_ttl': 600,
    'models': [
        # Your custom model configurations
    ]
}

manager = LLMRoutingManager(config=custom_config)

# Or import from file
manager.import_config('config.json')
```

## Performance Considerations

### Caching

- Default cache TTL: 300 seconds
- Max cache size: 1000 entries
- LRU eviction policy
- Thread-safe implementation

### Load Balancing

- Configurable max concurrent requests per model
- Automatic request queuing
- Load-aware routing

### Metrics Collection

- Rolling window (default: 1000 samples)
- Thread-safe statistics
- Minimal performance overhead

## Best Practices

1. **Choose the Right Strategy**
   - Use `cost_optimized` for high-volume, simple tasks
   - Use `latency_optimized` for real-time applications
   - Use `quality_optimized` for critical tasks
   - Use `balanced` as a general-purpose default

2. **Set Appropriate Constraints**
   - Define `max_cost` for budget control
   - Define `max_latency_ms` for time-sensitive operations
   - Define `required_quality` for quality assurance

3. **Monitor Performance**
   - Regularly check statistics
   - Adjust model configurations based on metrics
   - Update quality scores based on real performance

4. **Handle Failures Gracefully**
   - Always check `success` field in results
   - Implement appropriate error handling
   - Configure sufficient fallback models

5. **Optimize Configuration**
   - Tune cache TTL based on query patterns
   - Adjust max concurrent requests based on capacity
   - Enable ensemble mode for critical applications

## Testing

Run the demo:

```bash
python l_l_m_routing.py
```

The demo includes:
1. Cost-optimized routing example
2. Quality-optimized routing example
3. Execution with fallback
4. Performance metrics building
5. Statistics reporting
6. Strategy benchmarking

## Integration

### With OpenAI

```python
import openai

def openai_execution_fn(model: ModelConfig, req: QueryRequest):
    response = openai.ChatCompletion.create(
        model=model.model_id,
        messages=[{"role": "user", "content": req.query}],
        max_tokens=req.max_tokens
    )

    return ExecutionResult(
        success=True,
        model_id=model.model_id,
        actual_latency_ms=response['response_ms'],
        actual_cost=calculate_cost(response),
        tokens_used=response['usage']['total_tokens'],
        response=response['choices'][0]['message']['content']
    )

result = manager.router.execute_with_fallback(request, openai_execution_fn)
```

### With Anthropic

```python
import anthropic

def anthropic_execution_fn(model: ModelConfig, req: QueryRequest):
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model.model_id,
        max_tokens=req.max_tokens,
        messages=[{"role": "user", "content": req.query}]
    )

    return ExecutionResult(
        success=True,
        model_id=model.model_id,
        actual_latency_ms=measure_latency(),
        actual_cost=calculate_cost(message),
        tokens_used=message.usage.total_tokens,
        response=message.content[0].text
    )
```

## Troubleshooting

### Common Issues

**No available models**
- Check model availability status
- Verify models are properly configured

**High latency**
- Check model load
- Consider using `latency_optimized` strategy
- Adjust max concurrent requests

**High costs**
- Use `cost_optimized` strategy
- Set `max_cost` constraints
- Review quality requirements

**Low success rates**
- Check model availability
- Review fallback configurations
- Increase timeout settings

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- Documentation is updated
- Thread safety is maintained

## License

Copyright (c) BrillConsulting. All rights reserved.

## Support

For issues, questions, or contributions, please contact BrillConsulting.

## Changelog

### Version 1.0.0 (Production-Ready)

- Intelligent model selection with 6 routing strategies
- Cost optimization with success rate weighting
- Latency-based routing with load awareness
- Load balancing with concurrent request limits
- Automatic fallback strategies
- Real-time performance tracking and metrics
- Router ensembles with weighted voting
- Request caching with TTL and LRU eviction
- Thread-safe implementation
- Comprehensive benchmarking tools
- Configuration import/export
- Professional documentation
