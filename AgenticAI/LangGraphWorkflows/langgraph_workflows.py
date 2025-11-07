"""
LangGraph Workflows
===================

Advanced agentic workflows using graph-based orchestration.

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time


class NodeType(Enum):
    """Node types."""
    STANDARD = "standard"
    CONDITIONAL = "conditional"
    HUMAN = "human"
    TOOL = "tool"


END = "__end__"
START = "__start__"


@dataclass
class GraphState:
    """Base graph state."""
    messages: List[str] = field(default_factory=list)
    current_step: str = ""
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeResult:
    """Node execution result."""
    state: GraphState
    next_node: Optional[str] = None


class AgentGraph:
    """Graph-based agent workflow."""

    def __init__(self, state_class=GraphState):
        self.state_class = state_class
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, tuple] = {}
        self.entry_point: Optional[str] = None

        print(f"📊 Agent Graph initialized")

    def add_node(self, name: str, func: Callable) -> None:
        """Add node to graph."""
        self.nodes[name] = func
        print(f"   ➕ Added node: {name}")

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add edge between nodes."""
        self.edges[from_node] = to_node
        print(f"   → Edge: {from_node} → {to_node}")

    def add_conditional_edge(
        self,
        from_node: str,
        condition: Callable,
        mapping: Dict[str, str]
    ) -> None:
        """Add conditional edge."""
        self.conditional_edges[from_node] = (condition, mapping)
        print(f"   🔀 Conditional edge from: {from_node}")

    def set_entry_point(self, node: str) -> None:
        """Set entry point."""
        self.entry_point = node
        print(f"   🚀 Entry point: {node}")

    def compile(self) -> 'CompiledGraph':
        """Compile graph."""
        print(f"\n⚙️  Compiling graph...")
        print(f"   Nodes: {len(self.nodes)}")
        print(f"   Edges: {len(self.edges)}")
        print(f"   ✓ Compiled")

        return CompiledGraph(
            self.nodes,
            self.edges,
            self.conditional_edges,
            self.entry_point,
            self.state_class
        )


class CompiledGraph:
    """Compiled graph ready for execution."""

    def __init__(
        self,
        nodes: Dict[str, Callable],
        edges: Dict[str, str],
        conditional_edges: Dict[str, tuple],
        entry_point: str,
        state_class
    ):
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point
        self.state_class = state_class

    def invoke(self, initial_state: Dict[str, Any]) -> GraphState:
        """Execute graph."""
        print(f"\n▶️  Executing graph")

        state = self.state_class(**initial_state)
        current_node = self.entry_point

        step = 0
        while current_node and current_node != END:
            step += 1
            print(f"\n   Step {step}: {current_node}")

            # Execute node
            node_func = self.nodes[current_node]
            state = node_func(state)

            # Determine next node
            if current_node in self.conditional_edges:
                condition, mapping = self.conditional_edges[current_node]
                result = condition(state)
                current_node = mapping.get(result, END)
                print(f"   Condition result: {result} → {current_node}")
            elif current_node in self.edges:
                current_node = self.edges[current_node]
                print(f"   Next: {current_node}")
            else:
                current_node = END

        print(f"\n   ✓ Graph execution complete ({step} steps)")
        return state

    def stream(self, initial_state: Dict[str, Any]):
        """Stream execution."""
        print(f"\n📡 Streaming execution")

        state = self.state_class(**initial_state)
        current_node = self.entry_point

        while current_node and current_node != END:
            yield {"node": current_node, "state": state}

            node_func = self.nodes[current_node]
            state = node_func(state)

            if current_node in self.edges:
                current_node = self.edges[current_node]
            else:
                current_node = END

        yield {"node": END, "state": state}


class MultiAgentGraph:
    """Multi-agent collaboration graph."""

    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.workflow = []

        print(f"🤝 Multi-Agent Graph")
        print(f"   Agents: {', '.join(agents.keys())}")

    def add_sequential(self, agent_names: List[str]) -> None:
        """Add sequential execution."""
        print(f"\n   Sequential: {' → '.join(agent_names)}")
        self.workflow.append(("sequential", agent_names))

    def add_feedback_loop(
        self,
        critic: str,
        worker: str,
        max_iterations: int = 2
    ) -> None:
        """Add feedback loop."""
        print(f"\n   Feedback loop: {worker} ↔️ {critic} (max {max_iterations})")
        self.workflow.append(("feedback", (critic, worker, max_iterations)))

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-agent workflow."""
        print(f"\n▶️  Executing multi-agent workflow")

        state = input_data

        for workflow_type, workflow_data in self.workflow:
            if workflow_type == "sequential":
                for agent_name in workflow_data:
                    print(f"\n   Agent: {agent_name}")
                    agent = self.agents[agent_name]
                    # Simulate agent execution
                    state["last_agent"] = agent_name

            elif workflow_type == "feedback":
                critic, worker, max_iter = workflow_data
                for i in range(max_iter):
                    print(f"\n   Iteration {i+1}/{max_iter}")
                    print(f"   Worker: {worker}")
                    print(f"   Critic: {critic}")

        print(f"\n   ✓ Multi-agent workflow complete")
        return state


class HumanApprovalNode:
    """Human approval node."""

    def __init__(self):
        print(f"👤 Human Approval Node")

    def __call__(self, state: GraphState) -> GraphState:
        """Execute human approval."""
        print(f"\n   ⏸️  Pausing for human approval...")

        # Simulate human approval
        approved = True

        state.metadata["approved"] = approved
        print(f"   {'✅' if approved else '❌'} Approval: {approved}")

        return state


class ToolNode:
    """Tool execution node."""

    def __init__(self, tools: List[Dict[str, Any]]):
        self.tools = {t["name"]: t["function"] for t in tools}

        print(f"🔧 Tool Node")
        print(f"   Tools: {', '.join(self.tools.keys())}")

    def __call__(self, state: GraphState) -> GraphState:
        """Execute tools."""
        print(f"\n   🔧 Executing tools...")

        # Simulate tool selection and execution
        tool_name = "search"
        if tool_name in self.tools:
            print(f"   Calling: {tool_name}")
            result = "Tool result: found 5 items"
            state.messages.append(result)

        return state


class StreamingGraph(AgentGraph):
    """Graph with streaming support."""

    def stream(self, input_data: Dict[str, Any]):
        """Stream execution."""
        print(f"\n📡 Streaming execution")

        compiled = self.compile()

        for chunk in compiled.stream(input_data):
            yield chunk


class CheckpointGraph(AgentGraph):
    """Graph with checkpointing."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_interval: int = 1
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = []

        print(f"💾 Checkpoint Graph")
        print(f"   Directory: {checkpoint_dir}")
        print(f"   Interval: {checkpoint_interval} nodes")

    def save_checkpoint(self, checkpoint_id: str, state: GraphState) -> None:
        """Save checkpoint."""
        print(f"\n   💾 Saving checkpoint: {checkpoint_id}")
        self.checkpoints.append((checkpoint_id, state))

    def resume_from_checkpoint(self, checkpoint_id: Optional[str] = None):
        """Resume from checkpoint."""
        if not self.checkpoints:
            print(f"\n   ❌ No checkpoints found")
            return None

        checkpoint_id = checkpoint_id or self.checkpoints[-1][0]
        print(f"\n   ♻️  Resuming from checkpoint: {checkpoint_id}")

        for cid, state in self.checkpoints:
            if cid == checkpoint_id:
                return state

        return None


class SupervisorGraph:
    """Supervisor agent graph."""

    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents

        print(f"👔 Supervisor Graph")
        print(f"   Agents: {', '.join(agents.keys())}")

    def run(
        self,
        query: str,
        supervisor_model: str = "gpt-4"
    ) -> Dict[str, Any]:
        """Run with supervisor."""
        print(f"\n👔 Supervisor analyzing query...")
        print(f"   Query: {query}")

        # Supervisor decides which agent to call
        selected_agent = "researcher"  # Simulate decision

        print(f"   Selected agent: {selected_agent}")
        print(f"   Executing: {selected_agent}")

        result = {
            "agent": selected_agent,
            "output": f"Result from {selected_agent}"
        }

        return result


class HierarchicalGraph:
    """Hierarchical graph with planner and executor."""

    def __init__(self, planner: AgentGraph, executor: AgentGraph):
        self.planner = planner
        self.executor = executor

        print(f"🏢 Hierarchical Graph")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run hierarchical workflow."""
        print(f"\n▶️  Hierarchical execution")

        # High-level planning
        print(f"\n   📋 Planning phase...")
        plan = ["subtask_1", "subtask_2", "subtask_3"]

        # Low-level execution
        print(f"\n   ⚙️  Execution phase...")
        results = []
        for subtask in plan:
            print(f"   Executing: {subtask}")
            results.append(f"Result: {subtask}")

        return {"plan": plan, "results": results}


class ReflectionGraph:
    """Reflection pattern graph."""

    def __init__(
        self,
        actor_node: Callable,
        reflector_node: Callable,
        max_iterations: int = 3
    ):
        self.actor_node = actor_node
        self.reflector_node = reflector_node
        self.max_iterations = max_iterations

        print(f"🔄 Reflection Graph")
        print(f"   Max iterations: {max_iterations}")

    def run(
        self,
        prompt: str,
        quality_threshold: float = 0.9
    ) -> Dict[str, Any]:
        """Run reflection loop."""
        print(f"\n🔄 Reflection loop")

        state = GraphState(messages=[prompt])

        for i in range(self.max_iterations):
            print(f"\n   Iteration {i+1}/{self.max_iterations}")

            # Actor generates
            print(f"   🎭 Actor generating...")
            state = self.actor_node(state)

            # Reflector evaluates
            print(f"   🤔 Reflector evaluating...")
            state = self.reflector_node(state)

            # Check quality
            quality = 0.85 + (i * 0.05)  # Simulate improvement
            print(f"   Quality: {quality:.2f}")

            if quality >= quality_threshold:
                print(f"   ✓ Quality threshold reached")
                break

        return {"state": state, "iterations": i+1}


def create_research_workflow() -> AgentGraph:
    """Create research workflow."""
    print(f"\n📚 Creating research workflow")

    graph = AgentGraph()

    def research(state):
        print(f"   📖 Researching...")
        state.messages.append("Research findings")
        return state

    def plan(state):
        print(f"   📝 Planning...")
        state.messages.append("Article outline")
        return state

    def write(state):
        print(f"   ✍️  Writing...")
        state.messages.append("Draft content")
        return state

    def review(state):
        print(f"   👀 Reviewing...")
        state.iteration += 1
        return state

    graph.add_node("research", research)
    graph.add_node("plan", plan)
    graph.add_node("write", write)
    graph.add_node("review", review)

    graph.add_edge("research", "plan")
    graph.add_edge("plan", "write")
    graph.add_edge("write", "review")

    def should_continue(state):
        return "continue" if state.iteration < 2 else "end"

    graph.add_conditional_edge(
        "review",
        should_continue,
        {"continue": "write", "end": END}
    )

    graph.set_entry_point("research")

    return graph


def visualize_graph(graph: AgentGraph) -> str:
    """Generate Mermaid diagram."""
    print(f"\n📊 Generating graph visualization")

    mermaid = "graph TD\n"
    for from_node, to_node in graph.edges.items():
        mermaid += f"    {from_node} --> {to_node}\n"

    print(f"   ✓ Diagram generated")
    return mermaid


def demo():
    """Demonstrate LangGraph workflows."""
    print("=" * 70)
    print("LangGraph Workflows Demo")
    print("=" * 70)

    # Basic Graph
    print(f"\n{'='*70}")
    print("Basic Agent Graph")
    print(f"{'='*70}")

    graph = AgentGraph()

    def step1(state):
        state.messages.append("Step 1 complete")
        return state

    def step2(state):
        state.messages.append("Step 2 complete")
        return state

    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", END)
    graph.set_entry_point("step1")

    app = graph.compile()
    result = app.invoke({"messages": ["Starting"]})

    print(f"\n   Final messages: {result.messages}")

    # Research Workflow
    print(f"\n{'='*70}")
    print("Research Workflow")
    print(f"{'='*70}")

    research_flow = create_research_workflow()
    app = research_flow.compile()
    result = app.invoke({"messages": ["Write about AI"]})

    # Multi-Agent
    print(f"\n{'='*70}")
    print("Multi-Agent Collaboration")
    print(f"{'='*70}")

    agents = {
        "researcher": "ResearchAgent",
        "analyst": "AnalystAgent",
        "writer": "WriterAgent",
        "critic": "CriticAgent"
    }

    multi_graph = MultiAgentGraph(agents)
    multi_graph.add_sequential(["researcher", "analyst", "writer"])
    multi_graph.add_feedback_loop("critic", "writer", max_iterations=2)

    result = multi_graph.execute({"topic": "Climate change"})

    # Human-in-the-Loop
    print(f"\n{'='*70}")
    print("Human-in-the-Loop")
    print(f"{'='*70}")

    hitl_graph = AgentGraph()
    hitl_graph.add_node("generate", step1)
    hitl_graph.add_node("human_review", HumanApprovalNode())
    hitl_graph.add_node("publish", step2)

    hitl_graph.add_edge("generate", "human_review")
    hitl_graph.add_edge("human_review", "publish")
    hitl_graph.add_edge("publish", END)
    hitl_graph.set_entry_point("generate")

    app = hitl_graph.compile()
    result = app.invoke({"messages": []})

    # Streaming
    print(f"\n{'='*70}")
    print("Streaming Execution")
    print(f"{'='*70}")

    stream_graph = StreamingGraph()
    stream_graph.add_node("stream1", step1)
    stream_graph.add_node("stream2", step2)
    stream_graph.add_edge("stream1", "stream2")
    stream_graph.add_edge("stream2", END)
    stream_graph.set_entry_point("stream1")

    for chunk in stream_graph.stream({"messages": []}):
        print(f"   Chunk: {chunk['node']}")

    # Checkpointing
    print(f"\n{'='*70}")
    print("Checkpointing")
    print(f"{'='*70}")

    checkpoint_graph = CheckpointGraph(
        checkpoint_dir="./checkpoints",
        checkpoint_interval=1
    )

    checkpoint_graph.save_checkpoint("state_1", GraphState())
    resumed_state = checkpoint_graph.resume_from_checkpoint("state_1")

    # Supervisor
    print(f"\n{'='*70}")
    print("Supervisor Pattern")
    print(f"{'='*70}")

    supervisor = SupervisorGraph(agents)
    result = supervisor.run(
        query="Build a web scraper",
        supervisor_model="gpt-4"
    )

    # Hierarchical
    print(f"\n{'='*70}")
    print("Hierarchical Planning")
    print(f"{'='*70}")

    planner = AgentGraph()
    executor = AgentGraph()

    hierarchical = HierarchicalGraph(planner, executor)
    result = hierarchical.run({"goal": "Build a chatbot"})

    # Reflection
    print(f"\n{'='*70}")
    print("Reflection Pattern")
    print(f"{'='*70}")

    reflection = ReflectionGraph(
        actor_node=step1,
        reflector_node=step2,
        max_iterations=3
    )

    result = reflection.run(
        prompt="Write a technical article",
        quality_threshold=0.9
    )

    # Visualization
    print(f"\n{'='*70}")
    print("Graph Visualization")
    print(f"{'='*70}")

    diagram = visualize_graph(graph)
    print(f"\n{diagram}")

    print(f"\n{'='*70}")
    print("✓ LangGraph Workflows Demo Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
