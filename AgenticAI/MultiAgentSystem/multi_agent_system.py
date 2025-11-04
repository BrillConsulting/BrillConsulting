"""
Multi-Agent System
==================

Coordinated multi-agent collaboration framework:
- Agent communication protocols
- Task distribution
- Collaborative problem solving
- Consensus mechanisms
- Agent coordination

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Callable
from datetime import datetime
from enum import Enum
import json


class AgentRole(Enum):
    """Agent roles in the system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    OBSERVER = "observer"


class Message:
    """Message for inter-agent communication."""

    def __init__(self, sender: str, receiver: str, content: Dict, msg_type: str = "info"):
        self.id = f"msg_{datetime.now().timestamp()}"
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.type = msg_type
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.type,
            "timestamp": self.timestamp
        }


class Agent:
    """Individual agent in multi-agent system."""

    def __init__(self, agent_id: str, role: AgentRole, capabilities: List[str]):
        self.id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.inbox = []
        self.outbox = []
        self.status = "idle"
        self.current_task = None

    def send_message(self, receiver: str, content: Dict, msg_type: str = "info") -> Message:
        """Send message to another agent."""
        msg = Message(self.id, receiver, content, msg_type)
        self.outbox.append(msg)
        return msg

    def receive_message(self, message: Message):
        """Receive message from another agent."""
        self.inbox.append(message)

    def process_task(self, task: Dict) -> Dict:
        """Process assigned task."""
        self.status = "working"
        self.current_task = task

        result = {
            "agent_id": self.id,
            "task_id": task.get("id"),
            "status": "completed",
            "output": f"Task '{task.get('description')}' completed by {self.id}",
            "timestamp": datetime.now().isoformat()
        }

        self.status = "idle"
        self.current_task = None
        return result

    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle task type."""
        return task_type in self.capabilities

    def get_status(self) -> Dict:
        """Get agent status."""
        return {
            "id": self.id,
            "role": self.role.value,
            "status": self.status,
            "capabilities": self.capabilities,
            "inbox_count": len(self.inbox),
            "outbox_count": len(self.outbox)
        }


class MultiAgentSystem:
    """Multi-agent coordination system."""

    def __init__(self, name: str = "MAS"):
        self.name = name
        self.agents: Dict[str, Agent] = {}
        self.message_queue = []
        self.task_queue = []
        self.completed_tasks = []

    def register_agent(self, agent: Agent):
        """Register agent in the system."""
        self.agents[agent.id] = agent
        print(f"✓ Agent registered: {agent.id} ({agent.role.value})")

    def create_agent(self, agent_id: str, role: AgentRole, capabilities: List[str]) -> Agent:
        """Create and register new agent."""
        agent = Agent(agent_id, role, capabilities)
        self.register_agent(agent)
        return agent

    def send_message(self, sender_id: str, receiver_id: str, content: Dict, msg_type: str = "info"):
        """Route message between agents."""
        if sender_id not in self.agents or receiver_id not in self.agents:
            print(f"✗ Invalid agent ID in message routing")
            return

        sender = self.agents[sender_id]
        receiver = self.agents[receiver_id]

        msg = sender.send_message(receiver_id, content, msg_type)
        receiver.receive_message(msg)
        self.message_queue.append(msg)

        print(f"📨 Message: {sender_id} → {receiver_id} ({msg_type})")

    def broadcast_message(self, sender_id: str, content: Dict, msg_type: str = "info"):
        """Broadcast message to all agents."""
        for agent_id in self.agents:
            if agent_id != sender_id:
                self.send_message(sender_id, agent_id, content, msg_type)

    def assign_task(self, task: Dict) -> Optional[str]:
        """Assign task to capable agent."""
        task_type = task.get("type")

        # Find available agent with capability
        for agent_id, agent in self.agents.items():
            if agent.status == "idle" and agent.can_handle(task_type):
                print(f"✓ Task assigned to {agent_id}")
                result = agent.process_task(task)
                self.completed_tasks.append(result)
                return agent_id

        # No available agent, queue task
        self.task_queue.append(task)
        print(f"⏳ Task queued: {task.get('description')}")
        return None

    def distribute_tasks(self, tasks: List[Dict]):
        """Distribute multiple tasks across agents."""
        print(f"\n📋 Distributing {len(tasks)} tasks...")

        for task in tasks:
            self.assign_task(task)

        # Process queued tasks
        while self.task_queue:
            task = self.task_queue.pop(0)
            assigned = self.assign_task(task)
            if not assigned:
                self.task_queue.append(task)
                break

    def coordinate_task(self, task: Dict) -> Dict:
        """Coordinate complex task requiring multiple agents."""
        print(f"\n🤝 Coordinating task: {task.get('description')}")

        # Find coordinator
        coordinator = None
        for agent in self.agents.values():
            if agent.role == AgentRole.COORDINATOR:
                coordinator = agent
                break

        if not coordinator:
            return {"status": "failed", "error": "No coordinator available"}

        # Coordinator broadcasts task
        self.broadcast_message(
            coordinator.id,
            {"task": task, "request": "collaboration"},
            "task_broadcast"
        )

        # Collect responses (simplified)
        responses = []
        for agent in self.agents.values():
            if agent.id != coordinator.id and agent.can_handle(task.get("type")):
                result = agent.process_task(task)
                responses.append(result)

        # Coordinator aggregates results
        coordination_result = {
            "task": task,
            "coordinator": coordinator.id,
            "participating_agents": len(responses),
            "results": responses,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }

        print(f"✓ Coordination complete: {len(responses)} agents participated")
        return coordination_result

    def consensus(self, topic: str, options: List[str]) -> Dict:
        """Reach consensus among agents."""
        print(f"\n🗳️  Reaching consensus on: {topic}")

        votes = {opt: 0 for opt in options}

        for agent in self.agents.values():
            if agent.role != AgentRole.OBSERVER:
                # Simplified voting: first option
                vote = options[0]
                votes[vote] += 1
                print(f"  {agent.id} voted: {vote}")

        winner = max(votes.items(), key=lambda x: x[1])

        result = {
            "topic": topic,
            "votes": votes,
            "consensus": winner[0],
            "vote_count": winner[1],
            "timestamp": datetime.now().isoformat()
        }

        print(f"✓ Consensus reached: {winner[0]} ({winner[1]} votes)")
        return result

    def get_system_status(self) -> Dict:
        """Get overall system status."""
        return {
            "name": self.name,
            "total_agents": len(self.agents),
            "idle_agents": sum(1 for a in self.agents.values() if a.status == "idle"),
            "working_agents": sum(1 for a in self.agents.values() if a.status == "working"),
            "messages_exchanged": len(self.message_queue),
            "completed_tasks": len(self.completed_tasks),
            "queued_tasks": len(self.task_queue)
        }


def demo():
    """Demo multi-agent system."""
    print("Multi-Agent System Demo")
    print("=" * 60)

    # Create system
    mas = MultiAgentSystem(name="CollaborativeAI")

    # Create agents
    print("\n1. Creating Agents")
    print("-" * 60)

    coordinator = mas.create_agent("coordinator_1", AgentRole.COORDINATOR, ["coordination", "planning"])
    worker1 = mas.create_agent("worker_1", AgentRole.WORKER, ["data_processing", "analysis"])
    worker2 = mas.create_agent("worker_2", AgentRole.WORKER, ["reporting", "visualization"])
    specialist = mas.create_agent("specialist_1", AgentRole.SPECIALIST, ["ml_modeling", "optimization"])

    # Direct messaging
    print("\n2. Agent Communication")
    print("-" * 60)

    mas.send_message(
        "coordinator_1",
        "worker_1",
        {"request": "status_update"},
        "query"
    )

    mas.broadcast_message(
        "coordinator_1",
        {"announcement": "System initialization complete"},
        "broadcast"
    )

    # Task distribution
    print("\n3. Task Distribution")
    print("-" * 60)

    tasks = [
        {"id": 1, "type": "data_processing", "description": "Process customer dataset"},
        {"id": 2, "type": "ml_modeling", "description": "Train prediction model"},
        {"id": 3, "type": "reporting", "description": "Generate performance report"}
    ]

    mas.distribute_tasks(tasks)

    # Coordinated task
    print("\n4. Coordinated Task Execution")
    print("-" * 60)

    complex_task = {
        "id": 4,
        "type": "data_processing",
        "description": "End-to-end ML pipeline",
        "requires_coordination": True
    }

    coordination_result = mas.coordinate_task(complex_task)

    # Consensus
    print("\n5. Consensus Mechanism")
    print("-" * 60)

    consensus_result = mas.consensus(
        "Select deployment strategy",
        ["blue_green", "canary", "rolling"]
    )

    # System status
    print("\n6. System Status")
    print("-" * 60)

    status = mas.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n✓ Multi-Agent System Demo Complete!")


if __name__ == '__main__':
    demo()
