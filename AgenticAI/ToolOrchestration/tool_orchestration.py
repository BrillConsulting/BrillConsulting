"""
Tool Orchestration System
=========================

Advanced tool orchestration with code execution, browser automation,
file system operations, and security.

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time


class ToolType(Enum):
    """Tool types."""
    CODE_EXEC = "code_exec"
    BROWSER = "browser"
    FILESYSTEM = "filesystem"
    API = "api"
    DATABASE = "database"


class ExecutionStatus(Enum):
    """Execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class ExecutionResult:
    """Execution result."""
    status: str
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0
    memory_used_mb: float = 0
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False


@dataclass
class ToolMetadata:
    """Tool metadata."""
    name: str
    type: str
    description: str
    parameters: Dict[str, Any]
    permissions: List[str]


class CodeExecutor:
    """Execute code in sandboxed environment."""

    def __init__(
        self,
        language: str = "python",
        timeout: int = 30,
        max_memory_mb: int = 512,
        sandbox: bool = True,
        allowed_imports: Optional[List[str]] = None,
        blocked_functions: Optional[List[str]] = None,
        network_access: bool = False,
        filesystem_access: bool = False
    ):
        self.language = language
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.sandbox = sandbox
        self.allowed_imports = allowed_imports or ["math", "json", "datetime"]
        self.blocked_functions = blocked_functions or ["eval", "exec", "open", "__import__"]
        self.network_access = network_access
        self.filesystem_access = filesystem_access

        print(f"💻 Code Executor initialized")
        print(f"   Language: {language}")
        print(f"   Sandbox: {sandbox}")
        print(f"   Timeout: {timeout}s")
        print(f"   Max memory: {max_memory_mb}MB")

    def execute(self, code: str) -> ExecutionResult:
        """Execute code."""
        print(f"\n💻 Executing code ({len(code)} chars)")

        start_time = time.time()

        # Simulate code execution
        stdout = "Hello, World!\n42\n"
        stderr = ""
        execution_time = (time.time() - start_time) * 1000

        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS.value,
            stdout=stdout,
            stderr=stderr,
            execution_time_ms=execution_time,
            memory_used_mb=45.2
        )

        print(f"   ✓ Execution completed in {result.execution_time_ms:.1f}ms")
        return result

    def execute_safe(self, code: str) -> ExecutionResult:
        """Execute code with safety checks."""
        print(f"\n🔒 Safe execution")

        # Check for blocked functions
        for blocked in self.blocked_functions:
            if blocked in code:
                print(f"   ❌ Blocked function detected: {blocked}")
                return ExecutionResult(
                    status=ExecutionStatus.FAILURE.value,
                    error=f"Blocked function: {blocked}"
                )

        # Check imports
        if "import" in code:
            print(f"   Checking imports...")
            for imp in self.allowed_imports:
                print(f"   ✓ Allowed: {imp}")

        # Execute
        return self.execute(code)

    def execute_with_limits(self, code: str) -> ExecutionResult:
        """Execute with resource limits."""
        print(f"\n⚡ Executing with limits")
        print(f"   Time limit: {self.timeout}s")
        print(f"   Memory limit: {self.max_memory_mb}MB")

        # Simulate execution with monitoring
        result = self.execute(code)

        # Check limits
        if result.execution_time_ms > self.timeout * 1000:
            result.timeout = True
            print(f"   ⚠️  Timeout exceeded")

        if result.memory_used_mb > self.max_memory_mb:
            result.memory_exceeded = True
            print(f"   ⚠️  Memory limit exceeded")

        return result


class BrowserTool:
    """Browser automation tool."""

    def __init__(
        self,
        driver: str = "playwright",
        headless: bool = True,
        timeout: int = 30
    ):
        self.driver = driver
        self.headless = headless
        self.timeout = timeout

        print(f"🌐 Browser Tool initialized")
        print(f"   Driver: {driver}")
        print(f"   Headless: {headless}")

    def navigate_and_extract(
        self,
        url: str,
        selectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """Navigate to URL and extract data."""
        print(f"\n🌐 Navigating to: {url}")
        print(f"   Selectors: {len(selectors)}")

        # Simulate browser automation
        extracted = {}
        for key, selector in selectors.items():
            extracted[key] = f"Content from {selector}"
            print(f"   ✓ Extracted: {key}")

        return extracted

    def fill_form(
        self,
        url: str,
        data: Dict[str, str],
        submit_button: str
    ) -> ExecutionResult:
        """Fill and submit form."""
        print(f"\n📝 Filling form at: {url}")
        print(f"   Fields: {len(data)}")

        for field, value in data.items():
            print(f"   - {field}: {value}")

        print(f"   Clicking: {submit_button}")
        print(f"   ✓ Form submitted")

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS.value,
            stdout="Form submitted successfully"
        )

    def screenshot(self, url: str, path: str = "screenshot.png") -> str:
        """Take screenshot."""
        print(f"\n📸 Taking screenshot: {url}")
        print(f"   Saved to: {path}")
        return path

    def click(self, selector: str) -> ExecutionResult:
        """Click element."""
        print(f"\n👆 Clicking: {selector}")
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS.value,
            stdout=f"Clicked {selector}"
        )


class FileSystemTool:
    """File system operations tool."""

    def __init__(
        self,
        root_dir: str = "/workspace",
        allowed_extensions: Optional[List[str]] = None,
        max_file_size_mb: int = 10
    ):
        self.root_dir = root_dir
        self.allowed_extensions = allowed_extensions or [".txt", ".json", ".py"]
        self.max_file_size_mb = max_file_size_mb

        print(f"📁 File System Tool initialized")
        print(f"   Root: {root_dir}")
        print(f"   Allowed: {', '.join(allowed_extensions or [])}")

    def read_file(self, path: str) -> str:
        """Read file content."""
        print(f"\n📖 Reading file: {path}")

        # Simulate file read
        content = "File content here..."
        print(f"   ✓ Read {len(content)} bytes")

        return content

    def write_file(
        self,
        path: str,
        content: str,
        mode: str = "w"
    ) -> ExecutionResult:
        """Write file."""
        print(f"\n✏️  Writing file: {path}")
        print(f"   Mode: {mode}")
        print(f"   Size: {len(content)} bytes")

        # Check extension
        if not any(path.endswith(ext) for ext in self.allowed_extensions):
            return ExecutionResult(
                status=ExecutionStatus.PERMISSION_DENIED.value,
                error="File extension not allowed"
            )

        print(f"   ✓ File written")
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS.value,
            stdout=f"Wrote {len(content)} bytes to {path}"
        )

    def search_files(
        self,
        pattern: str,
        content_pattern: Optional[str] = None
    ) -> List[str]:
        """Search files by pattern."""
        print(f"\n🔍 Searching files: {pattern}")
        if content_pattern:
            print(f"   Content pattern: {content_pattern}")

        # Simulate search
        matches = [
            f"{self.root_dir}/src/main.py",
            f"{self.root_dir}/src/utils.py",
            f"{self.root_dir}/tests/test_main.py"
        ]

        print(f"   ✓ Found {len(matches)} matches")
        return matches

    def list_directory(
        self,
        path: str = ".",
        recursive: bool = False
    ) -> List[str]:
        """List directory contents."""
        print(f"\n📂 Listing: {path}")
        print(f"   Recursive: {recursive}")

        # Simulate listing
        files = [
            "file1.txt",
            "file2.json",
            "subdir/",
            "file3.py"
        ]

        print(f"   ✓ Found {len(files)} items")
        return files


class APITool:
    """API integration tool."""

    def __init__(
        self,
        base_url: str,
        auth: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ):
        self.base_url = base_url
        self.auth = auth
        self.timeout = timeout

        print(f"🔌 API Tool initialized")
        print(f"   Base URL: {base_url}")
        print(f"   Auth: {auth['type'] if auth else 'None'}")

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """GET request."""
        url = f"{self.base_url}{endpoint}"
        print(f"\n📥 GET {url}")
        if params:
            print(f"   Params: {params}")

        # Simulate API call
        response = {
            "status": 200,
            "data": {"id": 123, "name": "Example"}
        }

        print(f"   ✓ Status: {response['status']}")
        return response

    def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """POST request."""
        url = f"{self.base_url}{endpoint}"
        print(f"\n📤 POST {url}")
        if json:
            print(f"   Body: {json}")

        # Simulate API call
        response = {
            "status": 201,
            "data": {"id": 456, "created": True}
        }

        print(f"   ✓ Status: {response['status']}")
        return response

    def batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch requests."""
        print(f"\n📦 Batch requests: {len(requests)}")

        results = []
        for i, req in enumerate(requests, 1):
            print(f"   [{i}/{len(requests)}] {req['method']} {req['endpoint']}")
            results.append({"status": 200, "data": {}})

        print(f"   ✓ All requests completed")
        return results


class ToolOrchestrator:
    """Orchestrate multiple tools."""

    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.permissions: Dict[str, List[str]] = {}
        self.execution_log: List[Dict[str, Any]] = []

        print(f"🎭 Tool Orchestrator initialized")

    def register_tool(self, name: str, tool: Any) -> None:
        """Register tool."""
        self.tools[name] = tool
        print(f"\n✅ Registered tool: {name}")

    def execute_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> ExecutionResult:
        """Execute tool by name."""
        print(f"\n🎯 Executing tool: {tool_name}")

        if tool_name not in self.tools:
            return ExecutionResult(
                status=ExecutionStatus.FAILURE.value,
                error=f"Tool not found: {tool_name}"
            )

        tool = self.tools[tool_name]

        # Log execution
        self.execution_log.append({
            "tool": tool_name,
            "timestamp": time.time(),
            "kwargs": kwargs
        })

        # Execute based on tool type
        if isinstance(tool, CodeExecutor):
            return tool.execute(kwargs.get("code", ""))
        elif isinstance(tool, BrowserTool):
            action = kwargs.get("action", "navigate")
            if action == "navigate_and_extract":
                result = tool.navigate_and_extract(
                    kwargs["url"],
                    kwargs.get("selectors", {})
                )
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS.value,
                    stdout=str(result)
                )
        elif isinstance(tool, FileSystemTool):
            operation = kwargs.get("operation", "read_file")
            if operation == "read_file":
                content = tool.read_file(kwargs["path"])
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS.value,
                    stdout=content
                )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS.value,
            stdout="Tool executed"
        )

    def execute_with_security(
        self,
        tool_name: str,
        operation: str,
        permissions: List[str],
        **kwargs
    ) -> ExecutionResult:
        """Execute with security checks."""
        print(f"\n🔒 Secure execution: {tool_name}.{operation}")
        print(f"   Required permissions: {permissions}")

        # Check permissions
        allowed = self.permissions.get(tool_name, [])
        for perm in permissions:
            if perm not in allowed:
                print(f"   ❌ Permission denied: {perm}")
                return ExecutionResult(
                    status=ExecutionStatus.PERMISSION_DENIED.value,
                    error=f"Permission denied: {perm}"
                )

        print(f"   ✓ Permissions verified")
        return self.execute_tool(tool_name, operation=operation, **kwargs)

    def set_permissions(self, permissions: Dict[str, List[str]]) -> None:
        """Set tool permissions."""
        self.permissions = permissions
        print(f"\n🔐 Permissions configured")
        for tool, perms in permissions.items():
            print(f"   {tool}: {', '.join(perms)}")

    def create_workflow(self, steps: List[Dict[str, Any]]) -> 'Workflow':
        """Create tool workflow."""
        return Workflow(self, steps)

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get execution log."""
        return self.execution_log


class Workflow:
    """Tool workflow."""

    def __init__(
        self,
        orchestrator: ToolOrchestrator,
        steps: List[Dict[str, Any]]
    ):
        self.orchestrator = orchestrator
        self.steps = steps

        print(f"\n🔄 Workflow created")
        print(f"   Steps: {len(steps)}")

    def execute(self) -> List[ExecutionResult]:
        """Execute workflow."""
        print(f"\n▶️  Executing workflow")

        results = []
        for i, step in enumerate(self.steps, 1):
            print(f"\n   Step {i}/{len(self.steps)}: {step['tool']}.{step['action']}")

            result = self.orchestrator.execute_tool(
                tool_name=step["tool"],
                **{k: v for k, v in step.items() if k not in ["tool", "action"]}
            )

            results.append(result)

            if result.status != ExecutionStatus.SUCCESS.value:
                print(f"   ❌ Workflow failed at step {i}")
                break

        print(f"\n   ✓ Workflow completed")
        return results


class DockerExecutor(CodeExecutor):
    """Docker-based code execution."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        network_mode: str = "none",
        read_only: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image = image
        self.network_mode = network_mode
        self.read_only = read_only

        print(f"🐳 Docker Executor")
        print(f"   Image: {image}")
        print(f"   Network: {network_mode}")
        print(f"   Read-only: {read_only}")

    def execute(self, code: str, volumes: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """Execute in Docker container."""
        print(f"\n🐳 Executing in Docker")
        print(f"   Image: {self.image}")

        if volumes:
            print(f"   Volumes: {volumes}")

        # Simulate Docker execution
        result = super().execute(code)
        print(f"   ✓ Container execution complete")

        return result


class WasmExecutor:
    """WebAssembly code execution."""

    def __init__(self):
        print(f"🕸️  WASM Executor initialized")

    def execute(
        self,
        module: str,
        function: str,
        args: List[Any]
    ) -> Any:
        """Execute WASM function."""
        print(f"\n🕸️  Executing WASM")
        print(f"   Module: {module}")
        print(f"   Function: {function}")
        print(f"   Args: {args}")

        # Simulate WASM execution
        result = 55  # Example: fibonacci(10)

        print(f"   ✓ Result: {result}")
        return result


class ToolMarketplace:
    """Tool marketplace."""

    def __init__(self):
        self.installed_tools: Dict[str, str] = {}
        print(f"🛒 Tool Marketplace initialized")

    def search(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for tools."""
        print(f"\n🔍 Searching marketplace")
        if category:
            print(f"   Category: {category}")
        if tags:
            print(f"   Tags: {', '.join(tags)}")

        # Simulate search
        tools = [
            {
                "name": "json_transformer",
                "version": "1.2.0",
                "description": "Transform JSON data",
                "category": "data_processing"
            },
            {
                "name": "csv_parser",
                "version": "2.0.1",
                "description": "Parse CSV files",
                "category": "data_processing"
            }
        ]

        print(f"   ✓ Found {len(tools)} tools")
        return tools

    def install(self, tool_name: str, version: str = "latest") -> None:
        """Install tool."""
        print(f"\n📦 Installing: {tool_name}@{version}")
        self.installed_tools[tool_name] = version
        print(f"   ✓ Installed successfully")

    def list_installed(self) -> Dict[str, str]:
        """List installed tools."""
        print(f"\n📋 Installed tools:")
        for tool, version in self.installed_tools.items():
            print(f"   - {tool}@{version}")
        return self.installed_tools


def demo():
    """Demonstrate tool orchestration."""
    print("=" * 70)
    print("Tool Orchestration System Demo")
    print("=" * 70)

    # Code Executor
    print(f"\n{'='*70}")
    print("Code Execution")
    print(f"{'='*70}")

    executor = CodeExecutor(
        language="python",
        timeout=30,
        max_memory_mb=512,
        sandbox=True
    )

    code = """
import math

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

result = fibonacci(10)
print(f"Fibonacci(10) = {result}")
"""

    result = executor.execute(code)
    print(f"\n   Output: {result.stdout}")

    # Safe execution
    safe_result = executor.execute_safe(code)

    # Browser Tool
    print(f"\n{'='*70}")
    print("Browser Automation")
    print(f"{'='*70}")

    browser = BrowserTool(driver="playwright", headless=True)

    extracted = browser.navigate_and_extract(
        url="https://example.com",
        selectors={
            "title": "h1",
            "description": ".description"
        }
    )

    browser.fill_form(
        url="https://example.com/form",
        data={"email": "user@example.com", "name": "John Doe"},
        submit_button="#submit"
    )

    browser.screenshot("https://example.com")

    # File System Tool
    print(f"\n{'='*70}")
    print("File System Operations")
    print(f"{'='*70}")

    fs = FileSystemTool(
        root_dir="/workspace",
        allowed_extensions=[".txt", ".py", ".json"]
    )

    content = fs.read_file("data/input.txt")

    fs.write_file(
        path="output/results.json",
        content='{"status": "success"}',
        mode="w"
    )

    matches = fs.search_files(pattern="*.py", content_pattern="def main")

    files = fs.list_directory("src/", recursive=True)

    # API Tool
    print(f"\n{'='*70}")
    print("API Integration")
    print(f"{'='*70}")

    api = APITool(
        base_url="https://api.example.com",
        auth={"type": "bearer", "token": "secret"}
    )

    response = api.get(endpoint="/users/123", params={"include": "profile"})

    result = api.post(endpoint="/data", json={"name": "test", "value": 42})

    batch_results = api.batch([
        {"method": "GET", "endpoint": "/users/1"},
        {"method": "GET", "endpoint": "/users/2"}
    ])

    # Tool Orchestrator
    print(f"\n{'='*70}")
    print("Tool Orchestration")
    print(f"{'='*70}")

    orchestrator = ToolOrchestrator()

    orchestrator.register_tool("code_exec", executor)
    orchestrator.register_tool("browser", browser)
    orchestrator.register_tool("filesystem", fs)
    orchestrator.register_tool("api", api)

    # Execute tool
    result = orchestrator.execute_tool(
        tool_name="code_exec",
        code="print('Hello from orchestrator!')"
    )

    # Set permissions
    orchestrator.set_permissions({
        "filesystem": ["read"],
        "code_exec": ["python"],
        "browser": ["navigate", "extract"]
    })

    # Workflow
    print(f"\n{'='*70}")
    print("Workflow Execution")
    print(f"{'='*70}")

    workflow = orchestrator.create_workflow([
        {
            "tool": "browser",
            "action": "navigate_and_extract",
            "url": "https://example.com/data",
            "selectors": {"data": ".content"}
        },
        {
            "tool": "code_exec",
            "action": "execute",
            "code": "import json; print(json.dumps({'processed': True}))"
        },
        {
            "tool": "filesystem",
            "action": "write_file",
            "path": "output.json",
            "content": "{}"
        }
    ])

    results = workflow.execute()

    # Docker Executor
    print(f"\n{'='*70}")
    print("Docker Execution")
    print(f"{'='*70}")

    docker = DockerExecutor(
        image="python:3.11-slim",
        network_mode="none",
        read_only=True
    )

    docker_result = docker.execute(
        code="print('Running in Docker!')",
        volumes={"/workspace": "/app"}
    )

    # WASM Executor
    print(f"\n{'='*70}")
    print("WebAssembly Execution")
    print(f"{'='*70}")

    wasm = WasmExecutor()
    wasm_result = wasm.execute(
        module="fibonacci.wasm",
        function="fib",
        args=[10]
    )

    # Tool Marketplace
    print(f"\n{'='*70}")
    print("Tool Marketplace")
    print(f"{'='*70}")

    marketplace = ToolMarketplace()

    tools = marketplace.search(
        category="data_processing",
        tags=["json", "transform"]
    )

    marketplace.install("json_transformer", version="1.2.0")

    installed = marketplace.list_installed()

    print(f"\n{'='*70}")
    print("✓ Tool Orchestration Demo Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
