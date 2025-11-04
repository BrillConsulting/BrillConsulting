# Agent Memory System

Long-term and short-term memory for AI agents.

## Features

- Short-term working memory with limited capacity
- Long-term episodic memory (events and experiences)
- Semantic memory (facts and knowledge)
- Memory consolidation (short-term → long-term)
- Retrieval with relevance scoring
- Memory pruning and maintenance

## Usage

```python
from agent_memory import AgentMemory

memory = AgentMemory(short_term_capacity=10)

# Store memories
memory.remember("User asked about weather", "short_term")
memory.remember("User prefers detailed explanations", "long_term")

# Store knowledge
memory.store_knowledge("user_name", "Alice")

# Recall memories
results = memory.recall("user", limit=5)

# Get context
context = memory.get_context(n=5)

# Consolidate memories
memory.consolidate()

# Prune old memories
memory.prune(days=30)
```

## Demo

```bash
python agent_memory.py
```
