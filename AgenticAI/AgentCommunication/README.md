# Agent Communication

Inter-agent messaging protocols and negotiation frameworks for multi-agent coordination.

## Features

- **Message Passing**: Structured communication between agents
- **FIPA ACL Standards**: Agent Communication Language compliance
- **Negotiation Protocols**: Contract Net, auction mechanisms
- **Information Sharing**: Broadcast and targeted messaging
- **Coordination Messages**: Synchronization and task allocation
- **Performatives**: Request, inform, query, propose actions

## Quick Start

```python
from agent_communication import Agent, Message, MessageBus

# Initialize message bus
bus = MessageBus()

# Create agents
agent1 = Agent("agent_1", bus)
agent2 = Agent("agent_2", bus)

# Send message
msg = Message(
    sender="agent_1",
    receiver="agent_2",
    performative="REQUEST",
    content={"action": "process_data", "dataset": "sales_2024"}
)
agent1.send(msg)

# Receive and respond
received = agent2.receive()
response = Message(
    sender="agent_2",
    receiver="agent_1",
    performative="AGREE",
    in_reply_to=msg.id
)
agent2.send(response)
```

## Use Cases

- **Distributed Problem Solving**: Coordinate multi-agent tasks
- **Task Allocation**: Negotiate work distribution
- **Information Retrieval**: Query other agents for data

## Author

Brill Consulting
