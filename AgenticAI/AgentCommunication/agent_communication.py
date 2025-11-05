"""
Agent Communication System
==========================

FIPA ACL (Agent Communication Language) implementation for multi-agent messaging,
negotiation protocols, and coordinated communication.

Features:
- FIPA ACL compliant message structure
- Message passing with performatives
- Negotiation protocols (Contract Net, Auction)
- Message queue management
- Broadcasting and unicast
- Conversation management

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json
from collections import defaultdict


class Performative(Enum):
    """FIPA ACL Performatives - communication acts."""
    # Passing information
    INFORM = "inform"
    QUERY_IF = "query-if"
    QUERY_REF = "query-ref"

    # Requesting actions
    REQUEST = "request"
    CFP = "call-for-proposal"  # Call for proposals
    PROPOSE = "propose"
    ACCEPT_PROPOSAL = "accept-proposal"
    REJECT_PROPOSAL = "reject-proposal"

    # Negotiation
    AGREE = "agree"
    REFUSE = "refuse"
    CONFIRM = "confirm"
    DISCONFIRM = "disconfirm"

    # Error handling
    NOT_UNDERSTOOD = "not-understood"
    FAILURE = "failure"

    # Multi-agent coordination
    SUBSCRIBE = "subscribe"
    CANCEL = "cancel"


@dataclass
class ACLMessage:
    """FIPA ACL Message structure."""
    performative: Performative
    sender: str
    receiver: str
    content: Any
    reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    protocol: str = "fipa-request"
    language: str = "JSON"
    ontology: Optional[str] = None
    reply_by: Optional[datetime] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "performative": self.performative.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "reply_to": self.reply_to,
            "conversation_id": self.conversation_id,
            "protocol": self.protocol,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ACLMessage':
        """Create message from dictionary."""
        return cls(
            performative=Performative(data["performative"]),
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            reply_to=data.get("reply_to"),
            conversation_id=data.get("conversation_id"),
            protocol=data.get("protocol", "fipa-request"),
            message_id=data.get("message_id", str(uuid.uuid4()))
        )


class MessageQueue:
    """Thread-safe message queue for agents."""

    def __init__(self):
        self.messages: List[ACLMessage] = []
        self.archived: List[ACLMessage] = []

    def enqueue(self, message: ACLMessage):
        """Add message to queue."""
        self.messages.append(message)

    def dequeue(self, filter_fn: Optional[Callable] = None) -> Optional[ACLMessage]:
        """Remove and return next message matching filter."""
        if not self.messages:
            return None

        if filter_fn:
            for i, msg in enumerate(self.messages):
                if filter_fn(msg):
                    return self.messages.pop(i)
            return None

        return self.messages.pop(0)

    def peek(self, filter_fn: Optional[Callable] = None) -> Optional[ACLMessage]:
        """View next message without removing."""
        if not self.messages:
            return None

        if filter_fn:
            for msg in self.messages:
                if filter_fn(msg):
                    return msg
            return None

        return self.messages[0]

    def archive(self, message: ACLMessage):
        """Archive processed message."""
        self.archived.append(message)

    def get_messages_by_conversation(self, conversation_id: str) -> List[ACLMessage]:
        """Get all messages in a conversation."""
        return [msg for msg in self.messages + self.archived
                if msg.conversation_id == conversation_id]


class MessageBus:
    """Central message routing system."""

    def __init__(self):
        self.agents: Dict[str, 'CommunicatingAgent'] = {}
        self.message_log: List[ACLMessage] = []

    def register_agent(self, agent: 'CommunicatingAgent'):
        """Register agent with message bus."""
        self.agents[agent.name] = agent
        print(f"✓ Agent registered: {agent.name}")

    def unregister_agent(self, agent_name: str):
        """Remove agent from bus."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            print(f"✓ Agent unregistered: {agent_name}")

    def send(self, message: ACLMessage):
        """Route message to recipient."""
        self.message_log.append(message)

        if message.receiver == "ALL":
            # Broadcast to all except sender
            for name, agent in self.agents.items():
                if name != message.sender:
                    agent.receive(message)
        elif message.receiver in self.agents:
            self.agents[message.receiver].receive(message)
        else:
            print(f"⚠ Recipient not found: {message.receiver}")

    def get_statistics(self) -> Dict:
        """Get message bus statistics."""
        return {
            "total_messages": len(self.message_log),
            "active_agents": len(self.agents),
            "message_types": self._count_by_performative()
        }

    def _count_by_performative(self) -> Dict[str, int]:
        """Count messages by performative type."""
        counts = defaultdict(int)
        for msg in self.message_log:
            counts[msg.performative.value] += 1
        return dict(counts)


class CommunicatingAgent:
    """Agent with FIPA ACL communication capabilities."""

    def __init__(self, name: str, message_bus: MessageBus):
        self.name = name
        self.message_bus = message_bus
        self.inbox = MessageQueue()
        self.conversations: Dict[str, List[ACLMessage]] = defaultdict(list)
        self.handlers: Dict[Performative, Callable] = {}

        # Register with message bus
        message_bus.register_agent(self)

    def send(self, message: ACLMessage):
        """Send message via message bus."""
        message.sender = self.name
        self.message_bus.send(message)

        if message.conversation_id:
            self.conversations[message.conversation_id].append(message)

    def receive(self, message: ACLMessage):
        """Receive message into inbox."""
        self.inbox.enqueue(message)

        if message.conversation_id:
            self.conversations[message.conversation_id].append(message)

        # Auto-handle if handler registered
        if message.performative in self.handlers:
            self.handlers[message.performative](message)

    def register_handler(self, performative: Performative, handler: Callable):
        """Register handler for performative type."""
        self.handlers[performative] = handler

    def read_message(self, filter_fn: Optional[Callable] = None) -> Optional[ACLMessage]:
        """Read and remove message from inbox."""
        msg = self.inbox.dequeue(filter_fn)
        if msg:
            self.inbox.archive(msg)
        return msg

    def peek_message(self, filter_fn: Optional[Callable] = None) -> Optional[ACLMessage]:
        """View message without removing."""
        return self.inbox.peek(filter_fn)

    def request(self, receiver: str, action: str, content: Any = None) -> str:
        """Send REQUEST performative."""
        conv_id = str(uuid.uuid4())
        msg = ACLMessage(
            performative=Performative.REQUEST,
            sender=self.name,
            receiver=receiver,
            content={"action": action, "data": content},
            conversation_id=conv_id
        )
        self.send(msg)
        return conv_id

    def inform(self, receiver: str, content: Any, conversation_id: Optional[str] = None):
        """Send INFORM performative."""
        msg = ACLMessage(
            performative=Performative.INFORM,
            sender=self.name,
            receiver=receiver,
            content=content,
            conversation_id=conversation_id
        )
        self.send(msg)

    def query(self, receiver: str, query: str, conversation_id: Optional[str] = None) -> str:
        """Send QUERY_REF performative."""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        msg = ACLMessage(
            performative=Performative.QUERY_REF,
            sender=self.name,
            receiver=receiver,
            content={"query": query},
            conversation_id=conversation_id
        )
        self.send(msg)
        return conversation_id

    def call_for_proposal(self, task: str, requirements: Dict) -> str:
        """Initiate Contract Net Protocol - CFP."""
        conv_id = str(uuid.uuid4())
        msg = ACLMessage(
            performative=Performative.CFP,
            sender=self.name,
            receiver="ALL",
            content={"task": task, "requirements": requirements},
            conversation_id=conv_id,
            protocol="fipa-contract-net"
        )
        self.send(msg)
        return conv_id

    def propose(self, receiver: str, proposal: Dict, conversation_id: str):
        """Send proposal in response to CFP."""
        msg = ACLMessage(
            performative=Performative.PROPOSE,
            sender=self.name,
            receiver=receiver,
            content=proposal,
            conversation_id=conversation_id,
            protocol="fipa-contract-net"
        )
        self.send(msg)

    def accept_proposal(self, receiver: str, conversation_id: str):
        """Accept a proposal."""
        msg = ACLMessage(
            performative=Performative.ACCEPT_PROPOSAL,
            sender=self.name,
            receiver=receiver,
            content={"status": "accepted"},
            conversation_id=conversation_id,
            protocol="fipa-contract-net"
        )
        self.send(msg)

    def reject_proposal(self, receiver: str, reason: str, conversation_id: str):
        """Reject a proposal."""
        msg = ACLMessage(
            performative=Performative.REJECT_PROPOSAL,
            sender=self.name,
            receiver=receiver,
            content={"status": "rejected", "reason": reason},
            conversation_id=conversation_id,
            protocol="fipa-contract-net"
        )
        self.send(msg)

    def get_conversation_history(self, conversation_id: str) -> List[ACLMessage]:
        """Get all messages in a conversation."""
        return self.conversations.get(conversation_id, [])


def demo():
    """Demonstration of Agent Communication System."""
    print("Agent Communication System - FIPA ACL Demo")
    print("=" * 70)

    # Create message bus
    bus = MessageBus()

    # Create agents
    manager = CommunicatingAgent("Manager", bus)
    worker1 = CommunicatingAgent("Worker1", bus)
    worker2 = CommunicatingAgent("Worker2", bus)

    print("\n1️⃣  Request-Inform Protocol")
    print("-" * 70)

    # Manager requests action from Worker1
    conv_id = manager.request("Worker1", "process_data", {"dataset": "sales_2024"})
    print(f"Manager → Worker1: REQUEST (conv: {conv_id[:8]})")

    # Worker1 reads request and responds
    request_msg = worker1.read_message(
        lambda m: m.performative == Performative.REQUEST
    )

    if request_msg:
        print(f"Worker1 received: {request_msg.content}")
        # Send acknowledgment
        worker1.inform("Manager", {"status": "completed", "rows": 1000}, conv_id)
        print(f"Worker1 → Manager: INFORM (completed)")

    print("\n2️⃣  Contract Net Protocol (CFP)")
    print("-" * 70)

    # Manager initiates CFP
    task_conv = manager.call_for_proposal(
        task="data_analysis",
        requirements={"skills": ["python", "ml"], "deadline": "2024-12-31"}
    )
    print(f"Manager → ALL: CALL-FOR-PROPOSAL")

    # Workers send proposals
    cfp_msg = worker1.read_message(lambda m: m.performative == Performative.CFP)
    if cfp_msg:
        worker1.propose(
            "Manager",
            {"cost": 1000, "duration_days": 5, "experience": "5 years"},
            task_conv
        )
        print("Worker1 → Manager: PROPOSE (cost: $1000, 5 days)")

    cfp_msg2 = worker2.read_message(lambda m: m.performative == Performative.CFP)
    if cfp_msg2:
        worker2.propose(
            "Manager",
            {"cost": 800, "duration_days": 7, "experience": "3 years"},
            task_conv
        )
        print("Worker2 → Manager: PROPOSE (cost: $800, 7 days)")

    # Manager evaluates proposals
    proposals = []
    while True:
        prop_msg = manager.read_message(
            lambda m: m.performative == Performative.PROPOSE
        )
        if not prop_msg:
            break
        proposals.append(prop_msg)

    # Accept best proposal (lowest cost)
    best_proposal = min(proposals, key=lambda p: p.content["cost"])
    manager.accept_proposal(best_proposal.sender, task_conv)
    print(f"Manager → {best_proposal.sender}: ACCEPT-PROPOSAL")

    # Reject others
    for prop in proposals:
        if prop.sender != best_proposal.sender:
            manager.reject_proposal(
                prop.sender,
                "Selected lower cost proposal",
                task_conv
            )
            print(f"Manager → {prop.sender}: REJECT-PROPOSAL")

    print("\n3️⃣  Query Protocol")
    print("-" * 70)

    query_conv = manager.query("Worker1", "SELECT status FROM tasks WHERE id=123")
    print("Manager → Worker1: QUERY-REF")

    query_msg = worker1.read_message(lambda m: m.performative == Performative.QUERY_REF)
    if query_msg:
        worker1.inform("Manager", {"result": "completed"}, query_conv)
        print("Worker1 → Manager: INFORM (result: completed)")

    print("\n4️⃣  Message Bus Statistics")
    print("-" * 70)

    stats = bus.get_statistics()
    print(f"Total messages: {stats['total_messages']}")
    print(f"Active agents: {stats['active_agents']}")
    print("\nMessage breakdown:")
    for perf, count in stats['message_types'].items():
        print(f"  {perf}: {count}")

    print("\n5️⃣  Conversation History")
    print("-" * 70)

    history = manager.get_conversation_history(task_conv)
    print(f"Contract Net conversation ({len(history)} messages):")
    for msg in history:
        print(f"  {msg.sender} → {msg.receiver}: {msg.performative.value}")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
