# LangGraph Workflows

Advanced agentic workflows using LangGraph - stateful, cyclic graph-based agent orchestration with human-in-the-loop and streaming support.

## Features

- **Graph-Based Workflows** - Define agents as graphs with nodes and edges
- **Stateful Execution** - Maintain state across agent steps
- **Cyclic Flows** - Support loops and conditional routing
- **Human-in-the-Loop** - Pause for human approval
- **Streaming** - Real-time token streaming
- **Checkpointing** - Resume from failures
- **Parallel Execution** - Run multiple branches concurrently
- **Tool Integration** - Seamless tool calling

## Architecture

```
[State] → [Node 1] → [Conditional Edge] → [Node 2]
              ↓                              ↓
           [Node 3] ←───────────────────[Node 4]
              ↓
          [END]
```

## Installation

```bash
pip install langgraph langchain langchain-openai
```

## Usage

### Basic Agent Graph

```python
from langgraph_workflows import AgentGraph, GraphState

# Define state
class State(GraphState):
    messages: list
    current_step: str
    iteration: int

# Create graph
graph = AgentGraph(State)

# Add nodes
graph.add_node("researcher", research_node)
graph.add_node("writer", writer_node)
graph.add_node("reviewer", reviewer_node)

# Add edges
graph.add_edge("researcher", "writer")
graph.add_conditional_edge(
    "reviewer",
    should_continue,
    {
        "continue": "writer",
        "end": END
    }
)

# Set entry point
graph.set_entry_point("researcher")

# Compile and run
app = graph.compile()
result = app.invoke({"messages": ["Write about AI"]})
```

### Research Assistant Workflow

```python
from langgraph_workflows import create_research_workflow

workflow = create_research_workflow()

# Research → Plan → Write → Review → Revise (loop)
result = workflow.run(
    query="Explain quantum computing",
    max_iterations=3
)

print(result["final_output"])
```

### Multi-Agent Collaboration

```python
from langgraph_workflows import MultiAgentGraph

# Define agents
agents = {
    "researcher": ResearchAgent(),
    "analyst": AnalystAgent(),
    "writer": WriterAgent(),
    "critic": CriticAgent()
}

# Create collaborative workflow
graph = MultiAgentGraph(agents)

# Define collaboration flow
graph.add_sequential(["researcher", "analyst", "writer"])
graph.add_feedback_loop("critic", "writer", max_iterations=2)

# Execute
result = graph.execute({"topic": "Climate change"})
```

## Graph Patterns

### Sequential Flow

```python
graph = AgentGraph()

graph.add_node("step1", step1_fn)
graph.add_node("step2", step2_fn)
graph.add_node("step3", step3_fn)

graph.add_edge("step1", "step2")
graph.add_edge("step2", "step3")
graph.add_edge("step3", END)
```

### Conditional Routing

```python
def route_condition(state):
    if state["score"] > 0.8:
        return "good"
    else:
        return "needs_work"

graph.add_conditional_edge(
    "evaluator",
    route_condition,
    {
        "good": "publisher",
        "needs_work": "reviser"
    }
)
```

### Parallel Branches

```python
from langgraph.graph import Graph

graph = Graph()

# Parallel execution
graph.add_node("research_a", research_a_fn)
graph.add_node("research_b", research_b_fn)
graph.add_node("synthesize", synthesize_fn)

# Both research nodes run in parallel
graph.add_edge(START, "research_a")
graph.add_edge(START, "research_b")

# Synthesis waits for both
graph.add_edge("research_a", "synthesize")
graph.add_edge("research_b", "synthesize")
```

### Feedback Loops

```python
graph = AgentGraph()

graph.add_node("generator", generator_fn)
graph.add_node("validator", validator_fn)

# Loop until valid
graph.add_conditional_edge(
    "validator",
    lambda state: "generate" if not state["valid"] else "end",
    {
        "generate": "generator",
        "end": END
    }
)
```

## Human-in-the-Loop

### Approval Workflow

```python
from langgraph_workflows import HumanApprovalNode

graph = AgentGraph()

graph.add_node("generate", generate_content)
graph.add_node("human_review", HumanApprovalNode())
graph.add_node("publish", publish_content)

graph.add_edge("generate", "human_review")
graph.add_conditional_edge(
    "human_review",
    lambda state: "approved" if state["approved"] else "rejected",
    {
        "approved": "publish",
        "rejected": "generate"
    }
)
```

### Interactive Refinement

```python
from langgraph_workflows import InteractiveGraph

graph = InteractiveGraph()

# Pause for user input
graph.add_node("draft", create_draft)
graph.add_human_input_node("review", prompt="Review and edit:")
graph.add_node("finalize", finalize_document)

# User can edit at review step
result = graph.run_interactive({"topic": "AI Safety"})
```

## Streaming

### Real-time Output

```python
from langgraph_workflows import StreamingGraph

graph = StreamingGraph()

# Stream tokens as they're generated
for chunk in graph.stream({"query": "Explain neural networks"}):
    if "output" in chunk:
        print(chunk["output"], end="", flush=True)
```

### Event Streaming

```python
# Stream events
for event in graph.stream_events(input_data):
    if event["type"] == "node_start":
        print(f"Starting {event['node']}")
    elif event["type"] == "node_end":
        print(f"Finished {event['node']}")
    elif event["type"] == "token":
        print(event["token"], end="")
```

## Checkpointing

### Resume from Failures

```python
from langgraph_workflows import CheckpointGraph

graph = CheckpointGraph(
    checkpoint_dir="./checkpoints",
    checkpoint_interval=1  # Checkpoint after each node
)

try:
    result = graph.run(input_data)
except Exception as e:
    print(f"Error: {e}")
    # Resume from last checkpoint
    result = graph.resume_from_checkpoint()
```

### State Persistence

```python
# Save state
graph.save_checkpoint("workflow_state_123")

# Load and resume
graph = CheckpointGraph.load_from_checkpoint("workflow_state_123")
result = graph.continue_execution()
```

## Advanced Patterns

### Agent Supervisor

```python
from langgraph_workflows import SupervisorGraph

# Supervisor decides which agent to call
supervisor = SupervisorGraph(
    agents={
        "researcher": research_agent,
        "coder": coding_agent,
        "reviewer": review_agent
    }
)

# Supervisor routes to appropriate agent
result = supervisor.run(
    query="Build a web scraper",
    supervisor_model="gpt-4"
)
```

### Hierarchical Planning

```python
from langgraph_workflows import HierarchicalGraph

# High-level planner
planner_graph = AgentGraph()
planner_graph.add_node("decompose", decompose_task)

# Low-level executor
executor_graph = AgentGraph()
executor_graph.add_node("execute", execute_subtask)

# Hierarchical composition
hierarchical = HierarchicalGraph(
    planner=planner_graph,
    executor=executor_graph
)

result = hierarchical.run({"goal": "Build a chatbot"})
```

### Reflection Pattern

```python
from langgraph_workflows import ReflectionGraph

graph = ReflectionGraph(
    actor_node=generator,
    reflector_node=self_critic,
    max_iterations=3
)

# Generate → Reflect → Revise loop
result = graph.run(
    prompt="Write a technical article",
    quality_threshold=0.9
)
```

## Use Cases

### Content Creation Pipeline

```python
# Research → Outline → Draft → Edit → Publish
content_pipeline = create_content_pipeline()

result = content_pipeline.run(
    topic="Future of AI",
    target_length=2000,
    tone="technical"
)
```

### Code Generation Workflow

```python
# Plan → Code → Test → Debug → Review
code_workflow = create_code_workflow()

result = code_workflow.run(
    spec="Create a REST API for user management",
    language="Python",
    framework="FastAPI"
)
```

### Research Assistant

```python
# Query → Search → Extract → Synthesize → Cite
research_workflow = create_research_workflow()

report = research_workflow.run(
    query="Latest developments in quantum computing",
    sources=["arxiv", "google_scholar"],
    max_papers=10
)
```

### Customer Support Agent

```python
# Understand → Classify → Route → Resolve → Follow-up
support_workflow = create_support_workflow()

resolution = support_workflow.run(
    customer_message="My payment failed",
    context={"user_id": "123", "order_id": "456"}
)
```

## Integration with Tools

### Function Calling

```python
from langgraph_workflows import ToolNode

tools = [
    {"name": "search", "function": search_web},
    {"name": "calculator", "function": calculate},
    {"name": "database", "function": query_db}
]

graph.add_node("tools", ToolNode(tools))

# Agent decides which tools to call
graph.add_conditional_edge(
    "agent",
    should_use_tools,
    {
        "tools": "tools",
        "end": END
    }
)

# Tools output goes back to agent
graph.add_edge("tools", "agent")
```

### External APIs

```python
from langgraph_workflows import APINode

api_node = APINode(
    base_url="https://api.example.com",
    auth_token="secret"
)

graph.add_node("fetch_data", api_node)
```

## Monitoring & Debugging

### Visualization

```python
from langgraph_workflows import visualize_graph

# Generate Mermaid diagram
diagram = visualize_graph(graph)
print(diagram)

# Or save as image
graph.save_diagram("workflow.png")
```

### Execution Tracing

```python
from langgraph_workflows import TracingGraph

graph = TracingGraph()

result = graph.run(input_data)

# View execution trace
trace = graph.get_trace()
for step in trace:
    print(f"{step['node']}: {step['duration']}ms")
```

### Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Graph logs all state transitions
graph.run(input_data)
```

## Best Practices

✅ Keep state schema simple and well-defined
✅ Use type hints for state classes
✅ Implement proper error handling in nodes
✅ Set reasonable iteration limits for loops
✅ Use checkpointing for long-running workflows
✅ Add human-in-the-loop for critical decisions
✅ Monitor graph execution with tracing
✅ Test conditional edges thoroughly

## Performance

### Execution Time

| Pattern | Nodes | Avg Latency |
|---------|-------|-------------|
| Sequential | 5 | 10-15s |
| Parallel | 3 branches | 5-7s |
| Feedback loop | 3 iterations | 20-30s |
| Human-in-loop | 2 pauses | Variable |

### Optimization

```python
# Cache expensive operations
graph.add_node("search", search_node, cache=True)

# Parallel execution where possible
graph.set_parallel_execution(["research_a", "research_b"])

# Limit loop iterations
graph.set_max_iterations("feedback_loop", max_iter=5)
```

## Technologies

- **LangGraph**: Graph-based agent orchestration
- **LangChain**: Agent frameworks and tools
- **OpenAI**: LLM backend
- **Anthropic**: Claude models
- **Streaming**: Real-time output
- **Checkpointing**: SQLite, Redis

## References

- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- LangGraph Tutorials: https://langchain-ai.github.io/langgraph/tutorials/
- Agent Architectures: https://blog.langchain.dev/langgraph-multi-agent-workflows/
- Human-in-the-Loop: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/
