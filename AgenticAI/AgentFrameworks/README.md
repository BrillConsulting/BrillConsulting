# Agent Frameworks Integration

Comprehensive integration with leading multi-agent frameworks: AutoGen, CrewAI, and CAMEL for building collaborative AI systems.

## Features

- **AutoGen** - Microsoft's multi-agent conversation framework
- **CrewAI** - Role-based agent collaboration
- **CAMEL** - Communicative Agents for Mind Exploration
- **Framework Comparison** - Strengths and use cases
- **Unified Interface** - Common API across frameworks
- **Migration Tools** - Convert between frameworks

## Frameworks Overview

### AutoGen (Microsoft)

Multi-agent conversation framework with human-in-the-loop.

**Key Features:**
- Conversational agents
- Group chat
- Code execution
- Human proxy
- Nested chats

**Best For:**
- Complex multi-agent conversations
- Code generation workflows
- Human-AI collaboration

### CrewAI

Role-based multi-agent orchestration.

**Key Features:**
- Role definitions
- Task assignments
- Sequential/parallel execution
- Memory sharing
- Tool delegation

**Best For:**
- Business workflows
- Content creation teams
- Research projects

### CAMEL

Role-playing framework for autonomous cooperation.

**Key Features:**
- Role-playing scenarios
- Task decomposition
- Communication protocols
- Inception prompting
- Emergent collaboration

**Best For:**
- Research and exploration
- Complex problem solving
- Autonomous agent societies

## Installation

```bash
# AutoGen
pip install pyautogen

# CrewAI
pip install crewai crewai-tools

# CAMEL
pip install camel-ai
```

## Usage

### AutoGen

#### Basic Conversation

```python
from agent_frameworks import AutoGenSystem

system = AutoGenSystem()

# Create agents
assistant = system.create_assistant(
    name="assistant",
    system_message="You are a helpful AI assistant."
)

user_proxy = system.create_user_proxy(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate fibonacci numbers."
)
```

#### Group Chat

```python
from agent_frameworks import AutoGenGroupChat

# Create multiple agents
researcher = system.create_assistant("researcher", "Research expert")
coder = system.create_assistant("coder", "Expert programmer")
reviewer = system.create_assistant("reviewer", "Code reviewer")

# Create group chat
group_chat = AutoGenGroupChat(
    agents=[researcher, coder, reviewer],
    messages=[],
    max_round=10
)

# Start discussion
manager = system.create_group_chat_manager(group_chat)
user_proxy.initiate_chat(
    manager,
    message="Build a web scraper for news articles."
)
```

#### Code Execution

```python
# User proxy with code execution
user_proxy = system.create_user_proxy(
    name="executor",
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": True,
        "timeout": 120
    }
)

# Execute code from assistant
response = assistant.generate_reply(
    messages=[{"role": "user", "content": "Write code to analyze CSV data"}]
)
```

### CrewAI

#### Define Crew

```python
from agent_frameworks import CrewAISystem

system = CrewAISystem()

# Create agents with roles
researcher = system.create_agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="Expert in AI research with 10 years experience",
    tools=[search_tool, scrape_tool]
)

writer = system.create_agent(
    role="Tech Content Strategist",
    goal="Craft compelling content on tech advancements",
    backstory="Award-winning tech writer",
    tools=[write_tool]
)

# Define tasks
research_task = system.create_task(
    description="Research latest developments in LLMs",
    agent=researcher,
    expected_output="Research report with key findings"
)

write_task = system.create_task(
    description="Write an article based on research",
    agent=writer,
    expected_output="1500-word article"
)

# Create and run crew
crew = system.create_crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process="sequential"
)

result = crew.kickoff()
```

#### Parallel Execution

```python
# Parallel tasks
crew = system.create_crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process="parallel"
)

result = crew.kickoff()
```

#### Memory and Context

```python
# Crew with shared memory
crew = system.create_crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    memory=True,
    verbose=True
)
```

### CAMEL

#### Role-Playing Scenario

```python
from agent_frameworks import CAMELSystem

system = CAMELSystem()

# Define roles
ai_assistant = system.create_role(
    name="AI Assistant",
    role_type="assistant",
    inception_prompt="You are helpful AI assistant specialized in {domain}."
)

ai_user = system.create_role(
    name="AI User",
    role_type="user",
    inception_prompt="You are asking for help with {task}."
)

# Create society
society = system.create_society(
    assistant_role=ai_assistant,
    user_role=ai_user,
    task_prompt="Build a recommendation system",
    domain="machine learning"
)

# Run conversation
messages = society.run(max_turns=10)
```

#### Task Decomposition

```python
# CAMEL task decomposition
decomposer = system.create_task_decomposer()

subtasks = decomposer.decompose(
    task="Build a chatbot with sentiment analysis",
    num_subtasks=5
)

for subtask in subtasks:
    print(f"- {subtask}")
```

#### Multi-Agent Collaboration

```python
# Multiple agents with different roles
agents = [
    system.create_role("Architect", "assistant", "Design system architecture"),
    system.create_role("Developer", "assistant", "Implement features"),
    system.create_role("Tester", "assistant", "Test and validate")
]

# Collaborative problem solving
collaboration = system.create_collaboration(
    agents=agents,
    task="Build microservices architecture"
)

result = collaboration.solve()
```

## Framework Comparison

### Feature Matrix

| Feature | AutoGen | CrewAI | CAMEL |
|---------|---------|--------|-------|
| Multi-agent | ✅ | ✅ | ✅ |
| Code execution | ✅ | ❌ | ❌ |
| Human-in-loop | ✅ | ⚠️ | ❌ |
| Role definitions | ⚠️ | ✅ | ✅ |
| Tool use | ✅ | ✅ | ⚠️ |
| Memory | ⚠️ | ✅ | ❌ |
| Visualization | ❌ | ⚠️ | ❌ |
| Ease of use | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

### Use Case Recommendations

**Choose AutoGen for:**
- Code generation and execution
- Complex multi-turn conversations
- Human-in-the-loop workflows
- Research and development

**Choose CrewAI for:**
- Business process automation
- Content creation pipelines
- Marketing and sales workflows
- Team-based task execution

**Choose CAMEL for:**
- Research and exploration
- Academic projects
- Role-playing simulations
- Autonomous agent societies

## Unified Interface

### Cross-Framework Agent

```python
from agent_frameworks import UnifiedAgent

# Works with any framework
agent = UnifiedAgent(
    framework="autogen",  # or "crewai", "camel"
    name="assistant",
    role="Helper",
    capabilities=["code", "research", "writing"]
)

# Unified API
response = agent.execute(
    task="Write a function to sort a list",
    context={"language": "python"}
)
```

### Framework Migration

```python
from agent_frameworks import migrate_agent

# Migrate from AutoGen to CrewAI
autogen_agent = create_autogen_agent()

crewai_agent = migrate_agent(
    agent=autogen_agent,
    from_framework="autogen",
    to_framework="crewai"
)
```

## Advanced Patterns

### Hybrid Workflows

```python
# Combine frameworks
from agent_frameworks import HybridSystem

hybrid = HybridSystem()

# AutoGen for code
code_agent = hybrid.create_agent("autogen", "coder")

# CrewAI for content
content_crew = hybrid.create_crew("crewai", ["writer", "editor"])

# Orchestrate
result = hybrid.orchestrate([
    ("code_agent", "Generate API client"),
    ("content_crew", "Document the API")
])
```

### Framework Comparison Tool

```python
from agent_frameworks import compare_frameworks

# Compare performance
comparison = compare_frameworks(
    task="Build a recommendation system",
    frameworks=["autogen", "crewai", "camel"],
    metrics=["time", "quality", "cost"]
)

print(comparison.best_framework)  # crewai
print(comparison.reasoning)  # Best for business workflows
```

## Use Cases

### Code Review System (AutoGen)

```python
system = AutoGenSystem()

reviewer = system.create_assistant("reviewer", "Code review expert")
developer = system.create_assistant("developer", "Senior developer")

# Review loop
group_chat = AutoGenGroupChat([reviewer, developer])
result = group_chat.discuss("Review this pull request: ...")
```

### Content Creation Team (CrewAI)

```python
system = CrewAISystem()

# Define crew
researcher = system.create_agent("Researcher", "Research topics")
writer = system.create_agent("Writer", "Write articles")
editor = system.create_agent("Editor", "Edit and refine")

crew = system.create_crew([researcher, writer, editor])
article = crew.kickoff(topic="Future of AI")
```

### Research Assistant (CAMEL)

```python
system = CAMELSystem()

# AI researcher and domain expert
society = system.create_society(
    assistant_role="ML Researcher",
    user_role="Domain Expert",
    task="Explore neural architecture search"
)

findings = society.run(max_turns=20)
```

## Best Practices

### AutoGen
✅ Use human proxy for code execution
✅ Implement termination conditions
✅ Set reasonable max_consecutive_auto_reply
✅ Use group chat for complex scenarios
✅ Enable Docker for code safety

### CrewAI
✅ Define clear roles and goals
✅ Use specific backstories
✅ Enable memory for continuity
✅ Choose appropriate process type
✅ Provide relevant tools

### CAMEL
✅ Craft detailed inception prompts
✅ Set appropriate task complexity
✅ Monitor conversation quality
✅ Use task decomposition
✅ Limit max turns

## Performance

### Execution Time

| Framework | Simple Task | Complex Task | Multi-Agent |
|-----------|-------------|--------------|-------------|
| AutoGen | 5-10s | 30-60s | 60-180s |
| CrewAI | 10-15s | 45-90s | 90-240s |
| CAMEL | 15-25s | 60-120s | 120-300s |

### Cost (API Calls)

| Framework | Task Completion | Avg Tokens |
|-----------|-----------------|------------|
| AutoGen | 3-8 calls | 5K-15K |
| CrewAI | 5-12 calls | 8K-25K |
| CAMEL | 10-30 calls | 15K-50K |

## Technologies

- **AutoGen**: Microsoft Research framework
- **CrewAI**: Agent orchestration platform
- **CAMEL**: CMU research project
- **LangChain**: Tool integration
- **OpenAI**: LLM backend
- **Anthropic**: Claude models

## References

- AutoGen: https://microsoft.github.io/autogen/
- CrewAI: https://www.crewai.io/
- CAMEL: https://www.camel-ai.org/
- Multi-Agent Systems: https://arxiv.org/abs/2308.08155
