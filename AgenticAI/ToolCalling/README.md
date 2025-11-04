# Tool Calling Framework

Function calling and tool use framework for AI agents.

## Features

- Tool registration and discovery
- Automatic parameter validation
- Type checking and schema validation
- Execution with error handling
- Tool chaining and parallel execution
- Usage tracking and analytics

## Usage

```python
from tool_calling import ToolRegistry, ToolCaller

# Create registry
registry = ToolRegistry()

# Register tools
registry.register(
    "get_weather",
    get_weather_function,
    "Get weather for a location",
    {"type": "object", "properties": {"location": {"type": "string"}}}
)

# Or auto-register from function
registry.register_from_function(calculate_function)

# Create caller
caller = ToolCaller(registry)

# Call single tool
result = caller.call_tool("get_weather", location="San Francisco")

# Execute tool chain
chain = [
    {"tool": "calculate", "params": {"operation": "add", "a": 10, "b": 5}},
    {"tool": "notify", "params": {"message": "Done"}}
]
results = caller.call_tool_chain(chain)
```

## Demo

```bash
python tool_calling.py
```
