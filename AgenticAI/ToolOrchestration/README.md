# Tool Orchestration System

Advanced tool orchestration for AI agents with code execution, browser automation, file system operations, and dynamic tool composition.

## Features

- **Code Execution** - Safe Python/JavaScript execution in sandboxed environments
- **Browser Automation** - Selenium/Playwright for web interaction
- **File System** - Read, write, search operations with permissions
- **API Integration** - REST API calls with authentication
- **Database Tools** - SQL/NoSQL query execution
- **Tool Composition** - Chain and compose tools dynamically
- **Security** - Sandboxing, permissions, resource limits
- **Monitoring** - Execution tracking, logging, performance metrics

## Architecture

```
[Agent] → [Tool Orchestrator] → [Tool Registry]
             ↓
    [Security Layer]
             ↓
    ┌─────────┬──────────┬──────────────┬─────────┐
    │  Code   │ Browser  │ File System  │   API   │
    │  Exec   │ Automation│  Operations  │  Calls  │
    └─────────┴──────────┴──────────────┴─────────┘
```

## Usage

### Code Execution Tool

```python
from tool_orchestration import CodeExecutor

executor = CodeExecutor(
    language="python",
    timeout=30,
    max_memory_mb=512
)

# Execute Python code
result = executor.execute("""
import numpy as np

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fibonacci(10))
""")

print(f"Output: {result.stdout}")
print(f"Execution time: {result.execution_time_ms}ms")
```

### Browser Automation

```python
from tool_orchestration import BrowserTool

browser = BrowserTool(
    driver="playwright",  # or "selenium"
    headless=True
)

# Navigate and extract data
result = browser.navigate_and_extract(
    url="https://example.com",
    selectors={
        "title": "h1",
        "description": ".description",
        "links": "a[href]"
    }
)

# Fill form
browser.fill_form(
    url="https://example.com/form",
    data={
        "email": "user@example.com",
        "name": "John Doe"
    },
    submit_button="#submit"
)

# Take screenshot
screenshot = browser.screenshot("https://example.com")
```

### File System Operations

```python
from tool_orchestration import FileSystemTool

fs = FileSystemTool(
    root_dir="/workspace",
    allowed_extensions=[".txt", ".py", ".json"],
    max_file_size_mb=10
)

# Read file
content = fs.read_file("data/input.txt")

# Write file
fs.write_file(
    path="output/results.json",
    content='{"status": "success"}',
    mode="w"
)

# Search files
matches = fs.search_files(
    pattern="*.py",
    content_pattern="def main"
)

# List directory
files = fs.list_directory("src/", recursive=True)
```

### API Integration

```python
from tool_orchestration import APITool

api = APITool(
    base_url="https://api.example.com",
    auth={"type": "bearer", "token": "your-token"}
)

# GET request
response = api.get(
    endpoint="/users/123",
    params={"include": "profile"}
)

# POST request
result = api.post(
    endpoint="/data",
    json={"name": "test", "value": 42}
)

# Batch requests
results = api.batch([
    {"method": "GET", "endpoint": "/users/1"},
    {"method": "GET", "endpoint": "/users/2"},
    {"method": "GET", "endpoint": "/users/3"}
])
```

## Tool Orchestrator

### Register and Execute Tools

```python
from tool_orchestration import ToolOrchestrator

orchestrator = ToolOrchestrator()

# Register tools
orchestrator.register_tool("code_exec", CodeExecutor())
orchestrator.register_tool("browser", BrowserTool())
orchestrator.register_tool("filesystem", FileSystemTool())
orchestrator.register_tool("api", APITool())

# Execute tool by name
result = orchestrator.execute_tool(
    tool_name="code_exec",
    code="print('Hello, World!')",
    language="python"
)

# Execute with security checks
result = orchestrator.execute_with_security(
    tool_name="filesystem",
    operation="read_file",
    path="/workspace/data.txt",
    permissions=["read"]
)
```

### Tool Composition

```python
# Chain tools
workflow = orchestrator.create_workflow([
    {
        "tool": "browser",
        "action": "navigate",
        "url": "https://example.com/data"
    },
    {
        "tool": "code_exec",
        "action": "execute",
        "code": "import json; data = json.loads(input())"
    },
    {
        "tool": "filesystem",
        "action": "write_file",
        "path": "output.json"
    }
])

result = workflow.execute()
```

## Security Features

### Sandboxed Code Execution

```python
executor = CodeExecutor(
    sandbox=True,
    allowed_imports=["numpy", "pandas", "math"],
    blocked_functions=["eval", "exec", "open"],
    network_access=False,
    filesystem_access=False
)

# Safe execution
result = executor.execute_safe("""
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr.mean())
""")
```

### Permission System

```python
# Define permissions
permissions = {
    "filesystem": ["read"],
    "network": ["https"],
    "code_exec": ["python"],
    "browser": ["navigate", "extract"]
}

orchestrator.set_permissions(permissions)

# Attempt restricted operation
result = orchestrator.execute_tool(
    tool_name="filesystem",
    operation="write_file",  # Will fail - no write permission
    path="/etc/passwd"
)

print(result.error)  # "Permission denied: write not allowed"
```

### Resource Limits

```python
executor = CodeExecutor(
    max_execution_time=30,  # 30 seconds
    max_memory_mb=512,      # 512 MB
    max_cpu_percent=50      # 50% CPU
)

# Execute with limits
result = executor.execute_with_limits(code)

if result.timeout:
    print("Execution timed out")
if result.memory_exceeded:
    print("Memory limit exceeded")
```

## Advanced Features

### Docker-based Execution

```python
from tool_orchestration import DockerExecutor

docker_exec = DockerExecutor(
    image="python:3.11-slim",
    network_mode="none",
    read_only=True
)

result = docker_exec.execute(
    code="print('Running in Docker!')",
    volumes={"/workspace": "/app"}
)
```

### WebAssembly Execution

```python
from tool_orchestration import WasmExecutor

wasm_exec = WasmExecutor()

# Execute in WASI runtime
result = wasm_exec.execute(
    module="fibonacci.wasm",
    function="fib",
    args=[10]
)
```

### Tool Marketplace

```python
from tool_orchestration import ToolMarketplace

marketplace = ToolMarketplace()

# Discover tools
tools = marketplace.search(
    category="data_processing",
    tags=["json", "transform"]
)

# Install tool
marketplace.install("json_transformer", version="1.2.0")

# Use installed tool
result = orchestrator.execute_tool(
    tool_name="json_transformer",
    input_data={"key": "value"}
)
```

## Use Cases

### Web Scraping Agent

```python
# Orchestrate browser + code + filesystem
orchestrator.create_workflow([
    {
        "tool": "browser",
        "action": "navigate_and_extract",
        "url": "https://news.example.com",
        "selectors": {"headlines": "h2.title"}
    },
    {
        "tool": "code_exec",
        "action": "process",
        "code": """
import json
headlines = json.loads(input())
filtered = [h for h in headlines if 'AI' in h]
print(json.dumps(filtered))
"""
    },
    {
        "tool": "filesystem",
        "action": "write_file",
        "path": "headlines.json"
    }
]).execute()
```

### Data Analysis Pipeline

```python
# Code exec + API + filesystem
orchestrator.create_workflow([
    {
        "tool": "api",
        "action": "get",
        "endpoint": "/data/sales"
    },
    {
        "tool": "code_exec",
        "action": "analyze",
        "code": """
import pandas as pd
import json

data = json.loads(input())
df = pd.DataFrame(data)

# Analysis
summary = {
    'total_sales': df['amount'].sum(),
    'avg_sale': df['amount'].mean(),
    'top_products': df.groupby('product')['amount'].sum().nlargest(5).to_dict()
}

print(json.dumps(summary))
"""
    },
    {
        "tool": "filesystem",
        "action": "write_file",
        "path": "analysis_results.json"
    }
]).execute()
```

### Automated Testing

```python
# Browser + code + API
test_suite = orchestrator.create_workflow([
    {
        "tool": "browser",
        "action": "navigate",
        "url": "http://localhost:3000"
    },
    {
        "tool": "browser",
        "action": "fill_form",
        "data": {"username": "test", "password": "test123"}
    },
    {
        "tool": "browser",
        "action": "click",
        "selector": "#submit"
    },
    {
        "tool": "code_exec",
        "action": "validate",
        "code": """
assert 'Dashboard' in page_title
print('Test passed')
"""
    }
]).execute()
```

## Technologies

- **Code Execution**: RestrictedPython, PyPy sandbox, Docker, WASM
- **Browser**: Selenium, Playwright, Puppeteer
- **File System**: pathlib, watchdog, aiofiles
- **Security**: AppArmor, seccomp, resource limits
- **Monitoring**: Prometheus, OpenTelemetry
- **Containers**: Docker, Podman

## Performance

### Execution Times

| Tool | Operation | Latency |
|------|-----------|---------|
| Code Exec (Python) | Simple script | 50-100ms |
| Code Exec (Docker) | Containerized | 200-500ms |
| Browser | Page load | 500-2000ms |
| File System | Read file | 1-10ms |
| API | REST call | 50-300ms |

### Resource Usage

- Code execution: 10-512 MB RAM per execution
- Browser automation: 200-500 MB RAM per instance
- File operations: <10 MB RAM

## Best Practices

✅ Use sandboxing for untrusted code
✅ Set resource limits (time, memory, CPU)
✅ Implement permission system
✅ Log all tool executions
✅ Handle errors gracefully
✅ Use Docker for isolated execution
✅ Validate inputs before execution
✅ Monitor resource usage

## Security Considerations

⚠️ Never execute untrusted code without sandboxing
⚠️ Restrict filesystem access to specific directories
⚠️ Disable network access for code execution
⚠️ Validate and sanitize all inputs
⚠️ Use least privilege principle
⚠️ Monitor for suspicious activity
⚠️ Implement rate limiting
⚠️ Audit tool usage regularly

## References

- RestrictedPython: https://github.com/zopefoundation/RestrictedPython
- Playwright: https://playwright.dev/
- Docker Security: https://docs.docker.com/engine/security/
- OWASP Sandbox Guide: https://cheatsheetseries.owasp.org/
