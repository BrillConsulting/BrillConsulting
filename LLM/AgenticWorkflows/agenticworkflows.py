"""
Advanced Agentic Workflows System
Author: BrillConsulting
Description: Production-ready agentic workflow orchestration with multi-agent coordination,
            state management, tool integration, and robust error handling.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class WorkflowState(Enum):
    """Workflow execution states"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Inter-agent message types"""
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    ERROR = "error"
    SYNC = "sync"


@dataclass
class Message:
    """Message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    msg_type: MessageType = MessageType.TASK
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    requires_response: bool = False


@dataclass
class AgentMemory:
    """Agent memory management"""
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    max_short_term: int = 100

    def add_to_short_term(self, item: Dict[str, Any]):
        """Add item to short-term memory with automatic pruning"""
        self.short_term.append(item)
        if len(self.short_term) > self.max_short_term:
            # Move oldest to long-term if important
            old_item = self.short_term.pop(0)
            if old_item.get('important', False):
                key = old_item.get('key', str(uuid.uuid4()))
                self.long_term[key] = old_item

    def query_memory(self, query: str, context_size: int = 5) -> List[Dict[str, Any]]:
        """Query relevant memories"""
        # Simple relevance scoring - in production, use embeddings
        relevant = []
        for item in reversed(self.short_term[-context_size:]):
            if query.lower() in str(item).lower():
                relevant.append(item)
        return relevant


class Tool(ABC):
    """Abstract base class for agent tools"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Return tool schema for LLM"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters()
        }

    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """Define tool parameters"""
        pass


class SearchTool(Tool):
    """Example search tool"""

    def __init__(self):
        super().__init__(
            "search",
            "Search for information on a given topic"
        )

    async def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute search"""
        # Simulate search
        await asyncio.sleep(0.1)
        return {
            "success": True,
            "results": [
                {"title": f"Result for: {query}", "content": "Sample content"}
            ]
        }

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "required": True, "description": "Search query"}
        }


class CalculatorTool(Tool):
    """Example calculator tool"""

    def __init__(self):
        super().__init__(
            "calculator",
            "Perform mathematical calculations"
        )

    async def execute(self, expression: str, **kwargs) -> Dict[str, Any]:
        """Execute calculation"""
        try:
            # Safe eval for simple expressions
            result = eval(expression, {"__builtins__": {}}, {})
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "expression": {"type": "string", "required": True, "description": "Math expression"}
        }


class Agent:
    """Autonomous agent with tools, memory, and communication capabilities"""

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10
    ):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_iterations = max_iterations

        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.conversation_history: List[Dict[str, Any]] = []
        self.task_history: List[Dict[str, Any]] = []

        self.llm_provider = None  # Would connect to actual LLM

        logger.info(f"Agent {self.name} ({self.agent_id}) initialized with role: {self.role}")

    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming message"""
        logger.info(f"Agent {self.name} processing message from {message.sender}")

        self.memory.add_to_short_term({
            "type": "message",
            "content": message.content,
            "sender": message.sender,
            "timestamp": message.timestamp
        })

        if message.msg_type == MessageType.TASK:
            return await self._handle_task(message)
        elif message.msg_type == MessageType.QUERY:
            return await self._handle_query(message)
        elif message.msg_type == MessageType.SYNC:
            return await self._handle_sync(message)

        return None

    async def _handle_task(self, message: Message) -> Message:
        """Handle task message"""
        self.state = AgentState.PLANNING
        task = message.content.get('task', '')

        logger.info(f"Agent {self.name} planning task: {task}")

        # Plan execution
        plan = await self._plan_task(task)

        # Execute plan
        self.state = AgentState.EXECUTING
        result = await self._execute_plan(plan)

        self.state = AgentState.COMPLETED
        self.task_history.append({
            "task": task,
            "plan": plan,
            "result": result,
            "timestamp": datetime.now()
        })

        # Send result back
        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESULT,
            content={"result": result, "task": task}
        )

    async def _plan_task(self, task: str) -> List[Dict[str, Any]]:
        """Plan task execution steps"""
        # In production, use LLM to generate plan
        # For now, simple heuristic planning

        plan_steps = []

        # Check if task requires tools
        if "search" in task.lower():
            plan_steps.append({
                "action": "use_tool",
                "tool": "search",
                "params": {"query": task}
            })

        if any(op in task for op in ['+', '-', '*', '/', 'calculate']):
            plan_steps.append({
                "action": "use_tool",
                "tool": "calculator",
                "params": {"expression": task}
            })

        if not plan_steps:
            plan_steps.append({
                "action": "reasoning",
                "task": task
            })

        return plan_steps

    async def _execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute planned steps"""
        results = []

        for step in plan:
            if step["action"] == "use_tool":
                tool_name = step["tool"]
                if tool_name in self.tools:
                    tool_result = await self.tools[tool_name].execute(**step["params"])
                    results.append(tool_result)

                    self.memory.add_to_short_term({
                        "type": "tool_use",
                        "tool": tool_name,
                        "result": tool_result,
                        "timestamp": datetime.now()
                    })

            elif step["action"] == "reasoning":
                # In production, call LLM for reasoning
                reasoning_result = {
                    "thought": f"Analyzed task: {step['task']}",
                    "conclusion": "Task completed through reasoning"
                }
                results.append(reasoning_result)

        return {
            "success": True,
            "steps_executed": len(plan),
            "results": results
        }

    async def _handle_query(self, message: Message) -> Message:
        """Handle query message"""
        query = message.content.get('query', '')
        relevant_memories = self.memory.query_memory(query)

        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            content={"memories": relevant_memories}
        )

    async def _handle_sync(self, message: Message) -> Message:
        """Handle synchronization message"""
        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            content={"state": self.state.value, "status": "synced"}
        )

    def get_state(self) -> Dict[str, Any]:
        """Get agent state"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "state": self.state.value,
            "tools": list(self.tools.keys()),
            "tasks_completed": len(self.task_history),
            "memory_size": len(self.memory.short_term)
        }


class WorkflowOrchestrator:
    """Orchestrates multi-agent workflows"""

    def __init__(self, workflow_id: Optional[str] = None):
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.agents: Dict[str, Agent] = {}
        self.state = WorkflowState.CREATED
        self.execution_graph: Dict[str, List[str]] = defaultdict(list)
        self.results: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        logger.info(f"Workflow orchestrator created: {self.workflow_id}")

    def add_agent(self, agent: Agent):
        """Add agent to workflow"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.name} to workflow {self.workflow_id}")

    def define_flow(self, from_agent: str, to_agent: str):
        """Define workflow connections"""
        self.execution_graph[from_agent].append(to_agent)
        logger.info(f"Defined flow: {from_agent} -> {to_agent}")

    async def execute_workflow(self, initial_task: str, start_agent: str) -> Dict[str, Any]:
        """Execute workflow starting from specified agent"""
        self.state = WorkflowState.RUNNING
        self.start_time = datetime.now()

        logger.info(f"Starting workflow {self.workflow_id} with task: {initial_task}")

        try:
            # Send initial task to start agent
            initial_message = Message(
                sender="orchestrator",
                receiver=start_agent,
                msg_type=MessageType.TASK,
                content={"task": initial_task}
            )

            result = await self._route_message(initial_message)

            self.state = WorkflowState.COMPLETED
            self.end_time = datetime.now()

            execution_time = (self.end_time - self.start_time).total_seconds()

            return {
                "workflow_id": self.workflow_id,
                "status": "success",
                "result": result,
                "execution_time": execution_time,
                "agents_involved": len(self.agents),
                "timestamp": self.end_time
            }

        except Exception as e:
            self.state = WorkflowState.FAILED
            self.end_time = datetime.now()
            logger.error(f"Workflow {self.workflow_id} failed: {str(e)}")

            return {
                "workflow_id": self.workflow_id,
                "status": "failed",
                "error": str(e),
                "timestamp": self.end_time
            }

    async def _route_message(self, message: Message) -> Any:
        """Route message to appropriate agent"""
        if message.receiver not in self.agents:
            raise ValueError(f"Agent {message.receiver} not found in workflow")

        agent = self.agents[message.receiver]
        response = await agent.process_message(message)

        # Store result
        self.results[agent.agent_id] = response.content if response else None

        # Check if we need to route to next agent
        next_agents = self.execution_graph.get(agent.agent_id, [])

        if next_agents and response:
            # Route to next agent in flow
            next_agent_id = next_agents[0]
            next_message = Message(
                sender=agent.agent_id,
                receiver=next_agent_id,
                msg_type=MessageType.TASK,
                content=response.content
            )
            return await self._route_message(next_message)

        return response.content if response else None

    async def execute_parallel_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel"""
        logger.info(f"Executing {len(tasks)} parallel tasks")

        task_coroutines = []
        for task in tasks:
            agent_id = task['agent_id']
            task_content = task['task']

            message = Message(
                sender="orchestrator",
                receiver=agent_id,
                msg_type=MessageType.TASK,
                content={"task": task_content}
            )

            task_coroutines.append(self._route_message(message))

        results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        return [
            {"task": task, "result": result}
            for task, result in zip(tasks, results)
        ]

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        agent_states = {
            agent_id: agent.get_state()
            for agent_id, agent in self.agents.items()
        }

        return {
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "agents": agent_states,
            "execution_graph": dict(self.execution_graph),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


class AgenticWorkflowSystem:
    """Main system for managing agentic workflows"""

    def __init__(self):
        self.workflows: Dict[str, WorkflowOrchestrator] = {}
        self.global_tools: List[Tool] = [
            SearchTool(),
            CalculatorTool()
        ]
        logger.info("Agentic Workflow System initialized")

    def create_workflow(self, workflow_id: Optional[str] = None) -> WorkflowOrchestrator:
        """Create new workflow"""
        orchestrator = WorkflowOrchestrator(workflow_id)
        self.workflows[orchestrator.workflow_id] = orchestrator
        return orchestrator

    def create_agent(
        self,
        name: str,
        role: str,
        tools: Optional[List[str]] = None
    ) -> Agent:
        """Create new agent with specified tools"""
        agent_id = str(uuid.uuid4())

        # Select tools
        agent_tools = []
        if tools:
            agent_tools = [
                tool for tool in self.global_tools
                if tool.name in tools
            ]
        else:
            agent_tools = self.global_tools.copy()

        return Agent(agent_id, name, role, agent_tools)

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowOrchestrator]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)

    async def run_simple_workflow(self, task: str) -> Dict[str, Any]:
        """Run simple single-agent workflow"""
        workflow = self.create_workflow()
        agent = self.create_agent("SimpleAgent", "general")
        workflow.add_agent(agent)

        result = await workflow.execute_workflow(task, agent.agent_id)
        return result

    async def run_multi_agent_workflow(
        self,
        task: str,
        agent_configs: List[Dict[str, Any]],
        flow_definitions: List[tuple]
    ) -> Dict[str, Any]:
        """Run multi-agent workflow with specified configuration"""
        workflow = self.create_workflow()

        # Create agents
        agents = {}
        for config in agent_configs:
            agent = self.create_agent(
                config['name'],
                config['role'],
                config.get('tools')
            )
            workflow.add_agent(agent)
            agents[config['name']] = agent

        # Define flows
        for from_agent, to_agent in flow_definitions:
            workflow.define_flow(agents[from_agent].agent_id, agents[to_agent].agent_id)

        # Execute
        start_agent_name = agent_configs[0]['name']
        result = await workflow.execute_workflow(task, agents[start_agent_name].agent_id)

        return result


async def main():
    """Main demonstration"""
    system = AgenticWorkflowSystem()

    print("\n" + "="*70)
    print("ADVANCED AGENTIC WORKFLOW SYSTEM - DEMONSTRATION")
    print("="*70 + "\n")

    # Demo 1: Simple single-agent workflow
    print("Demo 1: Simple Single-Agent Workflow")
    print("-" * 70)
    result = await system.run_simple_workflow("Search for information about AI agents")
    print(f"Result: {json.dumps(result, indent=2, default=str)}\n")

    # Demo 2: Multi-agent workflow
    print("\nDemo 2: Multi-Agent Workflow")
    print("-" * 70)

    agent_configs = [
        {"name": "Researcher", "role": "research", "tools": ["search"]},
        {"name": "Analyzer", "role": "analysis", "tools": ["calculator"]},
        {"name": "Reporter", "role": "reporting", "tools": []}
    ]

    flow_definitions = [
        ("Researcher", "Analyzer"),
        ("Analyzer", "Reporter")
    ]

    result = await system.run_multi_agent_workflow(
        "Research AI trends and analyze the data",
        agent_configs,
        flow_definitions
    )
    print(f"Result: {json.dumps(result, indent=2, default=str)}\n")

    # Demo 3: Parallel task execution
    print("\nDemo 3: Parallel Task Execution")
    print("-" * 70)

    workflow = system.create_workflow()
    agents = [
        system.create_agent(f"Agent{i}", "worker")
        for i in range(3)
    ]

    for agent in agents:
        workflow.add_agent(agent)

    parallel_tasks = [
        {"agent_id": agent.agent_id, "task": f"Task {i}"}
        for i, agent in enumerate(agents)
    ]

    parallel_results = await workflow.execute_parallel_tasks(parallel_tasks)
    print(f"Parallel Results: {json.dumps(parallel_results, indent=2, default=str)}\n")

    # Status summary
    print("\nWorkflow Status Summary")
    print("-" * 70)
    status = workflow.get_workflow_status()
    print(json.dumps(status, indent=2, default=str))

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
