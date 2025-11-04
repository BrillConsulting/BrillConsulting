"""
Tool Calling Framework
======================

Function calling and tool use framework for AI agents:
- Tool registration and discovery
- Parameter validation
- Execution with error handling
- Tool chaining
- Result formatting

Author: Brill Consulting
"""

from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
import json
import inspect


class Tool:
    """Tool definition with metadata."""

    def __init__(self, name: str, function: Callable, description: str, parameters: Dict):
        self.name = name
        self.function = function
        self.description = description
        self.parameters = parameters
        self.usage_count = 0
        self.last_used = None

    def validate_params(self, kwargs: Dict) -> tuple[bool, Optional[str]]:
        """Validate parameters against schema."""
        required = self.parameters.get("required", [])

        # Check required parameters
        for param in required:
            if param not in kwargs:
                return False, f"Missing required parameter: {param}"

        # Check parameter types (simplified)
        properties = self.parameters.get("properties", {})
        for param, value in kwargs.items():
            if param in properties:
                expected_type = properties[param].get("type")
                actual_type = type(value).__name__

                type_mapping = {
                    "string": "str",
                    "integer": "int",
                    "number": ["int", "float"],
                    "boolean": "bool",
                    "array": "list",
                    "object": "dict"
                }

                expected = type_mapping.get(expected_type, expected_type)
                if isinstance(expected, list):
                    if actual_type not in expected:
                        return False, f"Parameter {param}: expected {expected}, got {actual_type}"
                elif expected != actual_type:
                    return False, f"Parameter {param}: expected {expected}, got {actual_type}"

        return True, None

    def execute(self, **kwargs) -> Dict:
        """Execute tool with parameters."""
        # Validate parameters
        valid, error = self.validate_params(kwargs)
        if not valid:
            return {
                "success": False,
                "error": error,
                "tool": self.name
            }

        try:
            result = self.function(**kwargs)
            self.usage_count += 1
            self.last_used = datetime.now()

            return {
                "success": True,
                "result": result,
                "tool": self.name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }

    def to_dict(self) -> Dict:
        """Convert tool to dictionary format (for LLM function calling)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, name: str, function: Callable, description: str, parameters: Dict):
        """Register a new tool."""
        tool = Tool(name, function, description, parameters)
        self.tools[name] = tool
        print(f"✓ Tool registered: {name}")
        return tool

    def register_from_function(self, function: Callable, description: Optional[str] = None):
        """Auto-register tool from function with type hints."""
        name = function.__name__
        desc = description or function.__doc__ or f"Tool: {name}"

        # Extract parameters from function signature
        sig = inspect.signature(function)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"

            parameters["properties"][param_name] = {"type": param_type}

            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        return self.register(name, function, desc, parameters)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict]:
        """List all available tools."""
        return [tool.to_dict() for tool in self.tools.values()]

    def search_tools(self, query: str) -> List[Tool]:
        """Search tools by name or description."""
        results = []
        query_lower = query.lower()

        for tool in self.tools.values():
            if query_lower in tool.name.lower() or query_lower in tool.description.lower():
                results.append(tool)

        return results


class ToolCaller:
    """Tool calling orchestrator."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.execution_history = []

    def call_tool(self, tool_name: str, **kwargs) -> Dict:
        """Call a single tool."""
        print(f"\n🔧 Calling tool: {tool_name}")
        print(f"   Parameters: {kwargs}")

        tool = self.registry.get_tool(tool_name)
        if not tool:
            result = {
                "success": False,
                "error": f"Tool not found: {tool_name}"
            }
        else:
            result = tool.execute(**kwargs)

        self.execution_history.append({
            "tool": tool_name,
            "params": kwargs,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

        if result["success"]:
            print(f"✓ Tool execution successful")
        else:
            print(f"✗ Tool execution failed: {result['error']}")

        return result

    def call_tool_chain(self, chain: List[Dict]) -> List[Dict]:
        """Execute a chain of tool calls."""
        print(f"\n⛓️  Executing tool chain ({len(chain)} steps)...")

        results = []
        context = {}  # Store results for next tools

        for i, step in enumerate(chain, 1):
            tool_name = step["tool"]
            params = step.get("params", {})

            # Resolve parameter references from previous results
            resolved_params = {}
            for key, value in params.items():
                if isinstance(value, str) and value.startswith("$"):
                    # Reference to previous result
                    ref = value[1:]
                    resolved_params[key] = context.get(ref, value)
                else:
                    resolved_params[key] = value

            print(f"\nStep {i}/{len(chain)}: {tool_name}")
            result = self.call_tool(tool_name, **resolved_params)
            results.append(result)

            # Store result in context
            if result["success"]:
                context[f"step{i}_result"] = result["result"]

        print(f"\n✓ Tool chain completed: {len([r for r in results if r['success']])}/{len(results)} successful")
        return results

    def parallel_call(self, calls: List[Dict]) -> List[Dict]:
        """Execute multiple tool calls in parallel (simulated)."""
        print(f"\n⚡ Executing {len(calls)} tools in parallel...")

        results = []
        for call in calls:
            result = self.call_tool(call["tool"], **call.get("params", {}))
            results.append(result)

        print(f"\n✓ Parallel execution completed: {len([r for r in results if r['success']])}/{len(results)} successful")
        return results

    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get tool execution history."""
        if limit:
            return self.execution_history[-limit:]
        return self.execution_history


def demo():
    """Demo tool calling framework."""
    print("Tool Calling Framework Demo")
    print("=" * 60)

    # Create registry
    registry = ToolRegistry()

    # Define example tools
    def get_weather(location: str, units: str = "celsius") -> Dict:
        """Get weather for a location."""
        return {
            "location": location,
            "temperature": 22,
            "units": units,
            "condition": "Sunny"
        }

    def calculate(operation: str, a: float, b: float) -> float:
        """Perform mathematical calculation."""
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            return a / b if b != 0 else None
        return None

    def send_notification(message: str, priority: str = "normal") -> bool:
        """Send a notification."""
        print(f"   📧 Notification sent: {message} (priority: {priority})")
        return True

    # Register tools
    print("\n1. Tool Registration")
    print("-" * 60)

    registry.register(
        "get_weather",
        get_weather,
        "Get current weather for a location",
        {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string"}
            },
            "required": ["location"]
        }
    )

    registry.register_from_function(calculate, "Perform mathematical calculations")
    registry.register_from_function(send_notification)

    # List tools
    print("\n2. Available Tools")
    print("-" * 60)

    tools = registry.list_tools()
    for tool in tools:
        print(f"  • {tool['name']}: {tool['description']}")

    # Create caller
    caller = ToolCaller(registry)

    # Single tool call
    print("\n3. Single Tool Call")
    print("-" * 60)

    result = caller.call_tool("get_weather", location="San Francisco", units="fahrenheit")
    print(f"   Result: {result['result']}")

    # Tool chain
    print("\n4. Tool Chain Execution")
    print("-" * 60)

    chain = [
        {"tool": "calculate", "params": {"operation": "add", "a": 10, "b": 5}},
        {"tool": "calculate", "params": {"operation": "multiply", "a": "$step1_result", "b": 2}},
        {"tool": "send_notification", "params": {"message": "Calculation complete", "priority": "high"}}
    ]

    chain_results = caller.call_tool_chain(chain)

    # Parallel calls
    print("\n5. Parallel Tool Calls")
    print("-" * 60)

    parallel_calls = [
        {"tool": "get_weather", "params": {"location": "New York"}},
        {"tool": "get_weather", "params": {"location": "London"}},
        {"tool": "get_weather", "params": {"location": "Tokyo"}}
    ]

    parallel_results = caller.parallel_call(parallel_calls)

    # Execution history
    print("\n6. Execution History")
    print("-" * 60)

    history = caller.get_execution_history(limit=5)
    for i, entry in enumerate(history, 1):
        status = "✓" if entry["result"]["success"] else "✗"
        print(f"  {i}. {status} {entry['tool']} at {entry['timestamp']}")

    print("\n✓ Tool Calling Framework Demo Complete!")


if __name__ == '__main__':
    demo()
