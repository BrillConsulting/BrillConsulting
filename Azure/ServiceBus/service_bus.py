"""
Azure Service Bus
Author: BrillConsulting
Description: Enterprise message broker with queues, topics, and advanced messaging patterns
             supporting publish-subscribe, message sessions, and reliable message delivery
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import uuid


class MessageState(Enum):
    """Message states"""
    ACTIVE = "Active"
    DEFERRED = "Deferred"
    SCHEDULED = "Scheduled"
    DEAD_LETTERED = "DeadLettered"


class EntityStatus(Enum):
    """Entity status"""
    ACTIVE = "Active"
    DISABLED = "Disabled"
    SEND_DISABLED = "SendDisabled"
    RECEIVE_DISABLED = "ReceiveDisabled"


class FilterType(Enum):
    """Subscription filter types"""
    SQL_FILTER = "SqlFilter"
    CORRELATION_FILTER = "CorrelationFilter"
    TRUE_FILTER = "TrueFilter"


@dataclass
class ServiceBusMessage:
    """Service Bus message"""
    message_id: str
    body: str
    content_type: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    reply_to: Optional[str] = None
    reply_to_session_id: Optional[str] = None
    label: Optional[str] = None
    to: Optional[str] = None
    time_to_live: Optional[timedelta] = None
    scheduled_enqueue_time: Optional[datetime] = None
    partition_key: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    enqueued_time: Optional[datetime] = None
    sequence_number: Optional[int] = None
    delivery_count: int = 0
    lock_token: Optional[str] = None
    state: MessageState = MessageState.ACTIVE


@dataclass
class QueueDescription:
    """Queue configuration"""
    name: str
    max_size_in_megabytes: int = 1024
    default_message_time_to_live: timedelta = timedelta(days=14)
    lock_duration: timedelta = timedelta(seconds=60)
    max_delivery_count: int = 10
    requires_duplicate_detection: bool = False
    duplicate_detection_history_time_window: timedelta = timedelta(minutes=10)
    requires_session: bool = False
    dead_lettering_on_message_expiration: bool = False
    enable_batched_operations: bool = True
    enable_partitioning: bool = False
    status: EntityStatus = EntityStatus.ACTIVE
    created_at: Optional[str] = None


@dataclass
class TopicDescription:
    """Topic configuration"""
    name: str
    max_size_in_megabytes: int = 1024
    default_message_time_to_live: timedelta = timedelta(days=14)
    requires_duplicate_detection: bool = False
    duplicate_detection_history_time_window: timedelta = timedelta(minutes=10)
    enable_batched_operations: bool = True
    enable_partitioning: bool = False
    support_ordering: bool = False
    status: EntityStatus = EntityStatus.ACTIVE
    created_at: Optional[str] = None


@dataclass
class SubscriptionDescription:
    """Subscription configuration"""
    name: str
    topic_name: str
    lock_duration: timedelta = timedelta(seconds=60)
    requires_session: bool = False
    default_message_time_to_live: timedelta = timedelta(days=14)
    dead_lettering_on_message_expiration: bool = False
    dead_lettering_on_filter_evaluation_exceptions: bool = True
    max_delivery_count: int = 10
    enable_batched_operations: bool = True
    status: EntityStatus = EntityStatus.ACTIVE
    created_at: Optional[str] = None


@dataclass
class RuleDescription:
    """Subscription rule (filter)"""
    name: str
    filter_type: FilterType
    filter_expression: Optional[str] = None
    correlation_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageSession:
    """Message session"""
    session_id: str
    locked_until: datetime
    state: Optional[bytes] = None


class ServiceBusManager:
    """
    Comprehensive Azure Service Bus manager

    Features:
    - Queue and topic management
    - Send and receive messages
    - Message sessions for ordered processing
    - Scheduled messages
    - Dead-letter queue handling
    - Message deferral
    - Batch operations
    - Topic subscriptions with filters
    - Transaction support
    - Monitoring and diagnostics
    """

    def __init__(
        self,
        namespace: str,
        shared_access_key_name: str,
        shared_access_key: str
    ):
        """
        Initialize Service Bus manager

        Args:
            namespace: Service Bus namespace
            shared_access_key_name: Shared access policy name
            shared_access_key: Shared access key
        """
        self.namespace = namespace
        self.shared_access_key_name = shared_access_key_name
        self.shared_access_key = shared_access_key
        self.queues: Dict[str, QueueDescription] = {}
        self.topics: Dict[str, TopicDescription] = {}
        self.subscriptions: Dict[str, Dict[str, SubscriptionDescription]] = {}
        self.rules: Dict[str, Dict[str, List[RuleDescription]]] = {}
        self.queue_messages: Dict[str, List[ServiceBusMessage]] = {}
        self.topic_messages: Dict[str, Dict[str, List[ServiceBusMessage]]] = {}
        self.dead_letter_messages: Dict[str, List[ServiceBusMessage]] = {}
        self.deferred_messages: Dict[str, List[ServiceBusMessage]] = {}
        self.sessions: Dict[str, Dict[str, MessageSession]] = {}

    # ===========================================
    # Queue Management
    # ===========================================

    def create_queue(
        self,
        queue_name: str,
        max_size_mb: int = 1024,
        message_ttl_days: int = 14,
        lock_duration_seconds: int = 60,
        max_delivery_count: int = 10,
        requires_session: bool = False,
        enable_dead_lettering: bool = False,
        enable_partitioning: bool = False,
        enable_batched_operations: bool = True
    ) -> QueueDescription:
        """
        Create a Service Bus queue

        Args:
            queue_name: Queue name
            max_size_mb: Maximum size in megabytes
            message_ttl_days: Default message time-to-live
            lock_duration_seconds: Message lock duration
            max_delivery_count: Maximum delivery attempts
            requires_session: Enable session support
            enable_dead_lettering: Enable dead-letter on expiration
            enable_partitioning: Enable partitioning
            enable_batched_operations: Enable batched operations

        Returns:
            QueueDescription object
        """
        if queue_name in self.queues:
            raise ValueError(f"Queue '{queue_name}' already exists")

        queue = QueueDescription(
            name=queue_name,
            max_size_in_megabytes=max_size_mb,
            default_message_time_to_live=timedelta(days=message_ttl_days),
            lock_duration=timedelta(seconds=lock_duration_seconds),
            max_delivery_count=max_delivery_count,
            requires_session=requires_session,
            dead_lettering_on_message_expiration=enable_dead_lettering,
            enable_partitioning=enable_partitioning,
            enable_batched_operations=enable_batched_operations,
            created_at=datetime.now().isoformat()
        )

        self.queues[queue_name] = queue
        self.queue_messages[queue_name] = []
        self.dead_letter_messages[queue_name] = []
        self.deferred_messages[queue_name] = []

        if requires_session:
            self.sessions[queue_name] = {}

        return queue

    def delete_queue(self, queue_name: str) -> Dict[str, Any]:
        """Delete a queue"""
        if queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not found")

        del self.queues[queue_name]
        del self.queue_messages[queue_name]
        del self.dead_letter_messages[queue_name]
        del self.deferred_messages[queue_name]

        if queue_name in self.sessions:
            del self.sessions[queue_name]

        return {
            "status": "deleted",
            "queue_name": queue_name,
            "deleted_at": datetime.now().isoformat()
        }

    def list_queues(self) -> List[QueueDescription]:
        """List all queues"""
        return list(self.queues.values())

    def get_queue_runtime_info(self, queue_name: str) -> Dict[str, Any]:
        """Get queue runtime information"""
        if queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not found")

        active = len(self.queue_messages[queue_name])
        dead_letter = len(self.dead_letter_messages.get(queue_name, []))
        deferred = len(self.deferred_messages.get(queue_name, []))

        return {
            "queue_name": queue_name,
            "active_message_count": active,
            "dead_letter_message_count": dead_letter,
            "scheduled_message_count": 0,
            "transfer_message_count": 0,
            "transfer_dead_letter_message_count": 0,
            "size_in_bytes": active * 1024,  # Simulated
            "accessed_at": datetime.now().isoformat()
        }

    # ===========================================
    # Topic Management
    # ===========================================

    def create_topic(
        self,
        topic_name: str,
        max_size_mb: int = 1024,
        message_ttl_days: int = 14,
        enable_partitioning: bool = False,
        support_ordering: bool = False
    ) -> TopicDescription:
        """
        Create a Service Bus topic

        Args:
            topic_name: Topic name
            max_size_mb: Maximum size in megabytes
            message_ttl_days: Default message time-to-live
            enable_partitioning: Enable partitioning
            support_ordering: Support message ordering

        Returns:
            TopicDescription object
        """
        if topic_name in self.topics:
            raise ValueError(f"Topic '{topic_name}' already exists")

        topic = TopicDescription(
            name=topic_name,
            max_size_in_megabytes=max_size_mb,
            default_message_time_to_live=timedelta(days=message_ttl_days),
            enable_partitioning=enable_partitioning,
            support_ordering=support_ordering,
            created_at=datetime.now().isoformat()
        )

        self.topics[topic_name] = topic
        self.subscriptions[topic_name] = {}
        self.rules[topic_name] = {}
        self.topic_messages[topic_name] = {}

        return topic

    def delete_topic(self, topic_name: str) -> Dict[str, Any]:
        """Delete a topic"""
        if topic_name not in self.topics:
            raise ValueError(f"Topic '{topic_name}' not found")

        del self.topics[topic_name]
        del self.subscriptions[topic_name]
        del self.rules[topic_name]
        del self.topic_messages[topic_name]

        return {
            "status": "deleted",
            "topic_name": topic_name,
            "deleted_at": datetime.now().isoformat()
        }

    def list_topics(self) -> List[TopicDescription]:
        """List all topics"""
        return list(self.topics.values())

    # ===========================================
    # Subscription Management
    # ===========================================

    def create_subscription(
        self,
        topic_name: str,
        subscription_name: str,
        lock_duration_seconds: int = 60,
        requires_session: bool = False,
        max_delivery_count: int = 10,
        enable_dead_lettering: bool = False
    ) -> SubscriptionDescription:
        """
        Create a topic subscription

        Args:
            topic_name: Topic name
            subscription_name: Subscription name
            lock_duration_seconds: Message lock duration
            requires_session: Enable session support
            max_delivery_count: Maximum delivery attempts
            enable_dead_lettering: Enable dead-letter on expiration

        Returns:
            SubscriptionDescription object
        """
        if topic_name not in self.topics:
            raise ValueError(f"Topic '{topic_name}' not found")

        if subscription_name in self.subscriptions[topic_name]:
            raise ValueError(f"Subscription '{subscription_name}' already exists")

        subscription = SubscriptionDescription(
            name=subscription_name,
            topic_name=topic_name,
            lock_duration=timedelta(seconds=lock_duration_seconds),
            requires_session=requires_session,
            max_delivery_count=max_delivery_count,
            dead_lettering_on_message_expiration=enable_dead_lettering,
            created_at=datetime.now().isoformat()
        )

        self.subscriptions[topic_name][subscription_name] = subscription
        self.rules[topic_name][subscription_name] = []
        self.topic_messages[topic_name][subscription_name] = []

        # Add default rule
        default_rule = RuleDescription(
            name="$Default",
            filter_type=FilterType.TRUE_FILTER
        )
        self.rules[topic_name][subscription_name].append(default_rule)

        return subscription

    def delete_subscription(
        self,
        topic_name: str,
        subscription_name: str
    ) -> Dict[str, Any]:
        """Delete a subscription"""
        if topic_name not in self.topics:
            raise ValueError(f"Topic '{topic_name}' not found")

        if subscription_name not in self.subscriptions[topic_name]:
            raise ValueError(f"Subscription '{subscription_name}' not found")

        del self.subscriptions[topic_name][subscription_name]
        del self.rules[topic_name][subscription_name]
        del self.topic_messages[topic_name][subscription_name]

        return {
            "status": "deleted",
            "subscription_name": subscription_name,
            "deleted_at": datetime.now().isoformat()
        }

    def list_subscriptions(self, topic_name: str) -> List[SubscriptionDescription]:
        """List all subscriptions for a topic"""
        if topic_name not in self.topics:
            raise ValueError(f"Topic '{topic_name}' not found")

        return list(self.subscriptions[topic_name].values())

    # ===========================================
    # Subscription Rules and Filters
    # ===========================================

    def create_rule(
        self,
        topic_name: str,
        subscription_name: str,
        rule_name: str,
        filter_type: FilterType = FilterType.SQL_FILTER,
        filter_expression: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> RuleDescription:
        """
        Create a subscription rule (filter)

        Args:
            topic_name: Topic name
            subscription_name: Subscription name
            rule_name: Rule name
            filter_type: Filter type (SQL, Correlation, or True)
            filter_expression: SQL filter expression
            correlation_id: Correlation ID for correlation filter

        Returns:
            RuleDescription object
        """
        if topic_name not in self.topics:
            raise ValueError(f"Topic '{topic_name}' not found")

        if subscription_name not in self.subscriptions[topic_name]:
            raise ValueError(f"Subscription '{subscription_name}' not found")

        rule = RuleDescription(
            name=rule_name,
            filter_type=filter_type,
            filter_expression=filter_expression,
            correlation_id=correlation_id
        )

        self.rules[topic_name][subscription_name].append(rule)

        return rule

    def delete_rule(
        self,
        topic_name: str,
        subscription_name: str,
        rule_name: str
    ) -> Dict[str, Any]:
        """Delete a subscription rule"""
        if topic_name not in self.topics:
            raise ValueError(f"Topic '{topic_name}' not found")

        if subscription_name not in self.subscriptions[topic_name]:
            raise ValueError(f"Subscription '{subscription_name}' not found")

        rules = self.rules[topic_name][subscription_name]
        for i, rule in enumerate(rules):
            if rule.name == rule_name:
                del rules[i]
                return {
                    "status": "deleted",
                    "rule_name": rule_name,
                    "deleted_at": datetime.now().isoformat()
                }

        raise ValueError(f"Rule '{rule_name}' not found")

    # ===========================================
    # Message Operations - Queue
    # ===========================================

    def send_message(
        self,
        queue_name: str,
        message_body: str,
        content_type: Optional[str] = None,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        label: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        time_to_live: Optional[timedelta] = None
    ) -> ServiceBusMessage:
        """
        Send a message to a queue

        Args:
            queue_name: Queue name
            message_body: Message body
            content_type: Content type
            correlation_id: Correlation ID
            session_id: Session ID (if queue requires sessions)
            label: Message label
            properties: Custom properties
            time_to_live: Message time-to-live

        Returns:
            ServiceBusMessage object
        """
        if queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not found")

        queue = self.queues[queue_name]

        if queue.requires_session and not session_id:
            raise ValueError("Queue requires session ID")

        message = ServiceBusMessage(
            message_id=str(uuid.uuid4()),
            body=message_body,
            content_type=content_type,
            correlation_id=correlation_id,
            session_id=session_id,
            label=label,
            properties=properties or {},
            time_to_live=time_to_live or queue.default_message_time_to_live,
            enqueued_time=datetime.now(),
            sequence_number=len(self.queue_messages[queue_name]) + 1
        )

        self.queue_messages[queue_name].append(message)

        return message

    def receive_messages(
        self,
        queue_name: str,
        max_message_count: int = 1,
        max_wait_time_seconds: int = 1
    ) -> List[ServiceBusMessage]:
        """
        Receive messages from a queue

        Args:
            queue_name: Queue name
            max_message_count: Maximum messages to receive
            max_wait_time_seconds: Maximum wait time

        Returns:
            List of ServiceBusMessage objects
        """
        if queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not found")

        messages = self.queue_messages[queue_name][:max_message_count]

        # Lock messages
        for msg in messages:
            msg.lock_token = str(uuid.uuid4())

        return messages

    def complete_message(
        self,
        queue_name: str,
        message: ServiceBusMessage
    ) -> Dict[str, Any]:
        """
        Complete (remove) a message from the queue

        Args:
            queue_name: Queue name
            message: Message to complete

        Returns:
            Completion result
        """
        if queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not found")

        messages = self.queue_messages[queue_name]
        for i, msg in enumerate(messages):
            if msg.message_id == message.message_id:
                del messages[i]
                return {
                    "status": "completed",
                    "message_id": message.message_id,
                    "completed_at": datetime.now().isoformat()
                }

        raise ValueError(f"Message '{message.message_id}' not found")

    def abandon_message(
        self,
        queue_name: str,
        message: ServiceBusMessage
    ) -> Dict[str, Any]:
        """
        Abandon a message (return to queue)

        Args:
            queue_name: Queue name
            message: Message to abandon

        Returns:
            Abandon result
        """
        message.lock_token = None
        message.delivery_count += 1

        queue = self.queues[queue_name]

        # Move to dead-letter if max delivery count exceeded
        if message.delivery_count >= queue.max_delivery_count:
            self._move_to_dead_letter(queue_name, message, "MaxDeliveryCountExceeded")
            return {
                "status": "dead_lettered",
                "message_id": message.message_id,
                "reason": "MaxDeliveryCountExceeded"
            }

        return {
            "status": "abandoned",
            "message_id": message.message_id,
            "delivery_count": message.delivery_count
        }

    def dead_letter_message(
        self,
        queue_name: str,
        message: ServiceBusMessage,
        reason: str = "Manual",
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Move message to dead-letter queue

        Args:
            queue_name: Queue name
            message: Message to dead-letter
            reason: Dead-letter reason
            description: Optional description

        Returns:
            Dead-letter result
        """
        self._move_to_dead_letter(queue_name, message, reason)

        return {
            "status": "dead_lettered",
            "message_id": message.message_id,
            "reason": reason,
            "description": description,
            "dead_lettered_at": datetime.now().isoformat()
        }

    def _move_to_dead_letter(
        self,
        queue_name: str,
        message: ServiceBusMessage,
        reason: str
    ):
        """Internal method to move message to dead-letter queue"""
        message.state = MessageState.DEAD_LETTERED
        message.properties["DeadLetterReason"] = reason

        # Remove from active messages
        messages = self.queue_messages[queue_name]
        for i, msg in enumerate(messages):
            if msg.message_id == message.message_id:
                del messages[i]
                break

        # Add to dead-letter queue
        self.dead_letter_messages[queue_name].append(message)

    def receive_dead_letter_messages(
        self,
        queue_name: str,
        max_message_count: int = 10
    ) -> List[ServiceBusMessage]:
        """Receive messages from dead-letter queue"""
        if queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not found")

        return self.dead_letter_messages[queue_name][:max_message_count]

    # ===========================================
    # Message Operations - Deferred Messages
    # ===========================================

    def defer_message(
        self,
        queue_name: str,
        message: ServiceBusMessage
    ) -> Dict[str, Any]:
        """
        Defer a message for later processing

        Args:
            queue_name: Queue name
            message: Message to defer

        Returns:
            Defer result with sequence number
        """
        message.state = MessageState.DEFERRED

        # Remove from active messages
        messages = self.queue_messages[queue_name]
        for i, msg in enumerate(messages):
            if msg.message_id == message.message_id:
                del messages[i]
                break

        # Add to deferred messages
        self.deferred_messages[queue_name].append(message)

        return {
            "status": "deferred",
            "message_id": message.message_id,
            "sequence_number": message.sequence_number,
            "deferred_at": datetime.now().isoformat()
        }

    def receive_deferred_message(
        self,
        queue_name: str,
        sequence_number: int
    ) -> Optional[ServiceBusMessage]:
        """
        Receive a deferred message by sequence number

        Args:
            queue_name: Queue name
            sequence_number: Message sequence number

        Returns:
            ServiceBusMessage or None
        """
        if queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not found")

        for msg in self.deferred_messages[queue_name]:
            if msg.sequence_number == sequence_number:
                return msg

        return None

    # ===========================================
    # Scheduled Messages
    # ===========================================

    def schedule_message(
        self,
        queue_name: str,
        message_body: str,
        scheduled_time: datetime,
        properties: Optional[Dict[str, Any]] = None
    ) -> ServiceBusMessage:
        """
        Schedule a message for future delivery

        Args:
            queue_name: Queue name
            message_body: Message body
            scheduled_time: Time to deliver message
            properties: Custom properties

        Returns:
            ServiceBusMessage object
        """
        message = self.send_message(
            queue_name,
            message_body,
            properties=properties
        )

        message.scheduled_enqueue_time = scheduled_time
        message.state = MessageState.SCHEDULED

        return message

    def cancel_scheduled_message(
        self,
        queue_name: str,
        sequence_number: int
    ) -> Dict[str, Any]:
        """Cancel a scheduled message"""
        messages = self.queue_messages[queue_name]

        for i, msg in enumerate(messages):
            if msg.sequence_number == sequence_number and msg.state == MessageState.SCHEDULED:
                del messages[i]
                return {
                    "status": "cancelled",
                    "sequence_number": sequence_number,
                    "cancelled_at": datetime.now().isoformat()
                }

        raise ValueError(f"Scheduled message {sequence_number} not found")

    # ===========================================
    # Topic Message Operations
    # ===========================================

    def send_message_to_topic(
        self,
        topic_name: str,
        message_body: str,
        content_type: Optional[str] = None,
        correlation_id: Optional[str] = None,
        label: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> ServiceBusMessage:
        """
        Send a message to a topic

        Args:
            topic_name: Topic name
            message_body: Message body
            content_type: Content type
            correlation_id: Correlation ID
            label: Message label
            properties: Custom properties

        Returns:
            ServiceBusMessage object
        """
        if topic_name not in self.topics:
            raise ValueError(f"Topic '{topic_name}' not found")

        message = ServiceBusMessage(
            message_id=str(uuid.uuid4()),
            body=message_body,
            content_type=content_type,
            correlation_id=correlation_id,
            label=label,
            properties=properties or {},
            enqueued_time=datetime.now()
        )

        # Deliver to all subscriptions that match filters
        for subscription_name, rules in self.rules[topic_name].items():
            if self._message_matches_filters(message, rules):
                self.topic_messages[topic_name][subscription_name].append(message)

        return message

    def _message_matches_filters(
        self,
        message: ServiceBusMessage,
        rules: List[RuleDescription]
    ) -> bool:
        """Check if message matches subscription filters"""
        for rule in rules:
            if rule.filter_type == FilterType.TRUE_FILTER:
                return True
            elif rule.filter_type == FilterType.CORRELATION_FILTER:
                if message.correlation_id == rule.correlation_id:
                    return True
            # SQL filters would be evaluated here in production

        return False

    def receive_subscription_messages(
        self,
        topic_name: str,
        subscription_name: str,
        max_message_count: int = 1
    ) -> List[ServiceBusMessage]:
        """
        Receive messages from a subscription

        Args:
            topic_name: Topic name
            subscription_name: Subscription name
            max_message_count: Maximum messages to receive

        Returns:
            List of ServiceBusMessage objects
        """
        if topic_name not in self.topics:
            raise ValueError(f"Topic '{topic_name}' not found")

        if subscription_name not in self.subscriptions[topic_name]:
            raise ValueError(f"Subscription '{subscription_name}' not found")

        messages = self.topic_messages[topic_name][subscription_name][:max_message_count]

        # Lock messages
        for msg in messages:
            msg.lock_token = str(uuid.uuid4())

        return messages

    # ===========================================
    # Batch Operations
    # ===========================================

    def send_messages_batch(
        self,
        queue_name: str,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Send multiple messages in a batch

        Args:
            queue_name: Queue name
            messages: List of message dictionaries

        Returns:
            Batch send result
        """
        sent_messages = []

        for msg_data in messages:
            message = self.send_message(
                queue_name,
                msg_data.get("body", ""),
                content_type=msg_data.get("content_type"),
                correlation_id=msg_data.get("correlation_id"),
                properties=msg_data.get("properties")
            )
            sent_messages.append(message)

        return {
            "status": "success",
            "count": len(sent_messages),
            "message_ids": [msg.message_id for msg in sent_messages],
            "sent_at": datetime.now().isoformat()
        }

    # ===========================================
    # Session Operations
    # ===========================================

    def accept_session(
        self,
        queue_name: str,
        session_id: str,
        timeout_seconds: int = 60
    ) -> MessageSession:
        """
        Accept a message session

        Args:
            queue_name: Queue name
            session_id: Session ID
            timeout_seconds: Session timeout

        Returns:
            MessageSession object
        """
        if queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not found")

        queue = self.queues[queue_name]
        if not queue.requires_session:
            raise ValueError("Queue does not support sessions")

        session = MessageSession(
            session_id=session_id,
            locked_until=datetime.now() + timedelta(seconds=timeout_seconds)
        )

        self.sessions[queue_name][session_id] = session

        return session

    def get_session_state(
        self,
        queue_name: str,
        session_id: str
    ) -> Optional[bytes]:
        """Get session state"""
        if queue_name in self.sessions and session_id in self.sessions[queue_name]:
            return self.sessions[queue_name][session_id].state

        return None

    def set_session_state(
        self,
        queue_name: str,
        session_id: str,
        state: bytes
    ) -> Dict[str, Any]:
        """Set session state"""
        if queue_name not in self.sessions:
            raise ValueError("Queue does not support sessions")

        if session_id not in self.sessions[queue_name]:
            raise ValueError(f"Session '{session_id}' not found")

        self.sessions[queue_name][session_id].state = state

        return {
            "status": "updated",
            "session_id": session_id,
            "updated_at": datetime.now().isoformat()
        }

    # ===========================================
    # Monitoring and Diagnostics
    # ===========================================

    def get_namespace_info(self) -> Dict[str, Any]:
        """Get Service Bus namespace information"""
        return {
            "namespace": self.namespace,
            "queue_count": len(self.queues),
            "topic_count": len(self.topics),
            "messaging_tier": "Standard",
            "created_at": "2024-01-01T00:00:00"
        }

    def get_entity_metrics(
        self,
        entity_name: str,
        entity_type: str = "queue"
    ) -> Dict[str, Any]:
        """
        Get metrics for a queue or topic

        Args:
            entity_name: Entity name
            entity_type: "queue" or "topic"

        Returns:
            Metrics dictionary
        """
        if entity_type == "queue":
            if entity_name not in self.queues:
                raise ValueError(f"Queue '{entity_name}' not found")

            return {
                "entity_name": entity_name,
                "entity_type": "queue",
                "active_messages": len(self.queue_messages[entity_name]),
                "dead_letter_messages": len(self.dead_letter_messages.get(entity_name, [])),
                "scheduled_messages": sum(1 for m in self.queue_messages[entity_name]
                                         if m.state == MessageState.SCHEDULED),
                "size_bytes": len(self.queue_messages[entity_name]) * 1024,
                "incoming_messages_per_second": 10.5,
                "outgoing_messages_per_second": 8.3,
                "timestamp": datetime.now().isoformat()
            }
        else:
            if entity_name not in self.topics:
                raise ValueError(f"Topic '{entity_name}' not found")

            total_messages = sum(
                len(msgs) for msgs in self.topic_messages[entity_name].values()
            )

            return {
                "entity_name": entity_name,
                "entity_type": "topic",
                "subscription_count": len(self.subscriptions[entity_name]),
                "messages": total_messages,
                "incoming_messages_per_second": 15.2,
                "size_bytes": total_messages * 1024,
                "timestamp": datetime.now().isoformat()
            }


# ===========================================
# Demo Functions
# ===========================================

def demo_queue_operations():
    """Demonstrate queue operations"""
    print("=== Queue Operations Demo ===\n")

    manager = ServiceBusManager(
        namespace="myservicebus",
        shared_access_key_name="RootManageSharedAccessKey",
        shared_access_key="your-key"
    )

    # Create queue
    queue = manager.create_queue(
        queue_name="orders",
        max_size_mb=1024,
        max_delivery_count=10,
        enable_dead_lettering=True
    )
    print(f"Created queue: {queue.name}")
    print(f"Max delivery count: {queue.max_delivery_count}")
    print(f"Lock duration: {queue.lock_duration.seconds}s\n")

    # Send messages
    for i in range(1, 4):
        message = manager.send_message(
            "orders",
            json.dumps({"order_id": i, "amount": 100.0 * i}),
            content_type="application/json",
            label="order",
            properties={"priority": "high"}
        )
        print(f"Sent message: {message.message_id}")

    print()

    # Get queue info
    info = manager.get_queue_runtime_info("orders")
    print(f"Queue runtime info:")
    print(f"Active messages: {info['active_message_count']}")
    print(f"Dead letter messages: {info['dead_letter_message_count']}\n")


def demo_message_receiving():
    """Demonstrate message receiving and completion"""
    print("=== Message Receiving Demo ===\n")

    manager = ServiceBusManager(
        namespace="myservicebus",
        shared_access_key_name="RootManageSharedAccessKey",
        shared_access_key="your-key"
    )

    # Setup
    manager.create_queue("processing", max_delivery_count=3)

    # Send messages
    for i in range(3):
        manager.send_message(
            "processing",
            f"Task {i+1}",
            properties={"task_id": i+1}
        )

    # Receive and process messages
    messages = manager.receive_messages("processing", max_message_count=3)

    for msg in messages:
        print(f"Received message: {msg.message_id}")
        print(f"Body: {msg.body}")
        print(f"Properties: {msg.properties}")

        # Complete the message
        result = manager.complete_message("processing", msg)
        print(f"Status: {result['status']}\n")

    # Check queue is empty
    info = manager.get_queue_runtime_info("processing")
    print(f"Remaining messages: {info['active_message_count']}\n")


def demo_dead_letter_queue():
    """Demonstrate dead-letter queue handling"""
    print("=== Dead-Letter Queue Demo ===\n")

    manager = ServiceBusManager(
        namespace="myservicebus",
        shared_access_key_name="RootManageSharedAccessKey",
        shared_access_key="your-key"
    )

    # Create queue with low max delivery count
    manager.create_queue(
        "dlq_test",
        max_delivery_count=2,
        enable_dead_lettering=True
    )

    # Send a message
    message = manager.send_message("dlq_test", "Test message for DLQ")
    print(f"Sent message: {message.message_id}\n")

    # Receive and abandon multiple times
    for attempt in range(3):
        messages = manager.receive_messages("dlq_test", max_message_count=1)

        if messages:
            msg = messages[0]
            result = manager.abandon_message("dlq_test", msg)
            print(f"Attempt {attempt + 1}: {result['status']}")

            if result['status'] == 'dead_lettered':
                print(f"Message moved to DLQ: {result['reason']}\n")
                break
        else:
            print("No messages available (already in DLQ)\n")
            break

    # Receive from dead-letter queue
    dlq_messages = manager.receive_dead_letter_messages("dlq_test")
    print(f"Dead-letter queue messages: {len(dlq_messages)}")
    for msg in dlq_messages:
        print(f"  - {msg.message_id}: {msg.properties.get('DeadLetterReason')}\n")


def demo_scheduled_messages():
    """Demonstrate scheduled message delivery"""
    print("=== Scheduled Messages Demo ===\n")

    manager = ServiceBusManager(
        namespace="myservicebus",
        shared_access_key_name="RootManageSharedAccessKey",
        shared_access_key="your-key"
    )

    # Create queue
    manager.create_queue("scheduled")

    # Schedule messages
    schedule_times = [
        datetime.now() + timedelta(minutes=5),
        datetime.now() + timedelta(minutes=10),
        datetime.now() + timedelta(hours=1)
    ]

    for i, schedule_time in enumerate(schedule_times, 1):
        message = manager.schedule_message(
            "scheduled",
            f"Scheduled task {i}",
            schedule_time,
            properties={"task_id": i}
        )
        print(f"Scheduled message {message.message_id}")
        print(f"  Delivery time: {schedule_time.isoformat()}")
        print(f"  Sequence number: {message.sequence_number}\n")

    # Get queue info
    info = manager.get_queue_runtime_info("scheduled")
    print(f"Total messages in queue: {info['active_message_count']}\n")


def demo_topic_subscriptions():
    """Demonstrate topic and subscription operations"""
    print("=== Topic and Subscription Demo ===\n")

    manager = ServiceBusManager(
        namespace="myservicebus",
        shared_access_key_name="RootManageSharedAccessKey",
        shared_access_key="your-key"
    )

    # Create topic
    topic = manager.create_topic(
        "notifications",
        max_size_mb=2048,
        support_ordering=True
    )
    print(f"Created topic: {topic.name}\n")

    # Create subscriptions
    sub1 = manager.create_subscription("notifications", "email_subscription")
    sub2 = manager.create_subscription("notifications", "sms_subscription")
    sub3 = manager.create_subscription("notifications", "all_notifications")

    print(f"Created subscriptions:")
    for sub in [sub1, sub2, sub3]:
        print(f"  - {sub.name}")
    print()

    # Add filters
    manager.create_rule(
        "notifications",
        "email_subscription",
        "EmailFilter",
        FilterType.SQL_FILTER,
        filter_expression="type = 'email'"
    )

    manager.create_rule(
        "notifications",
        "sms_subscription",
        "SmsFilter",
        FilterType.SQL_FILTER,
        filter_expression="type = 'sms'"
    )

    print("Added filters to subscriptions\n")

    # Send messages to topic
    message_types = ["email", "sms", "push"]
    for msg_type in message_types:
        message = manager.send_message_to_topic(
            "notifications",
            f"Notification: {msg_type}",
            label=msg_type,
            properties={"type": msg_type}
        )
        print(f"Sent {msg_type} message: {message.message_id}")

    print()

    # Receive from subscriptions
    for sub_name in ["email_subscription", "sms_subscription", "all_notifications"]:
        messages = manager.receive_subscription_messages(
            "notifications",
            sub_name,
            max_message_count=10
        )
        print(f"{sub_name}: {len(messages)} messages")


def demo_batch_operations():
    """Demonstrate batch message operations"""
    print("=== Batch Operations Demo ===\n")

    manager = ServiceBusManager(
        namespace="myservicebus",
        shared_access_key_name="RootManageSharedAccessKey",
        shared_access_key="your-key"
    )

    # Create queue
    manager.create_queue("batch_queue", enable_batched_operations=True)

    # Prepare batch messages
    batch_messages = []
    for i in range(1, 101):
        batch_messages.append({
            "body": json.dumps({"batch_id": i, "data": f"item_{i}"}),
            "content_type": "application/json",
            "properties": {"sequence": i}
        })

    # Send batch
    result = manager.send_messages_batch("batch_queue", batch_messages)

    print(f"Batch send result:")
    print(f"Status: {result['status']}")
    print(f"Messages sent: {result['count']}")
    print(f"Sent at: {result['sent_at']}\n")

    # Get queue metrics
    metrics = manager.get_entity_metrics("batch_queue", "queue")
    print(f"Queue metrics:")
    print(f"Active messages: {metrics['active_messages']}")
    print(f"Size: {metrics['size_bytes']} bytes\n")


def demo_message_sessions():
    """Demonstrate message sessions"""
    print("=== Message Sessions Demo ===\n")

    manager = ServiceBusManager(
        namespace="myservicebus",
        shared_access_key_name="RootManageSharedAccessKey",
        shared_access_key="your-key"
    )

    # Create session-enabled queue
    queue = manager.create_queue(
        "session_queue",
        requires_session=True
    )
    print(f"Created session-enabled queue: {queue.name}\n")

    # Send messages to different sessions
    sessions = ["session1", "session2", "session3"]

    for session_id in sessions:
        for i in range(3):
            message = manager.send_message(
                "session_queue",
                f"Message {i+1} for {session_id}",
                session_id=session_id,
                properties={"message_number": i+1}
            )
            print(f"Sent to {session_id}: {message.message_id}")

    print()

    # Accept and process session
    session = manager.accept_session("session_queue", "session1")
    print(f"Accepted session: {session.session_id}")
    print(f"Locked until: {session.locked_until.isoformat()}\n")

    # Set session state
    session_state = json.dumps({"processed": 0, "last_message": None}).encode()
    manager.set_session_state("session_queue", "session1", session_state)
    print("Set session state\n")

    # Get session state
    state = manager.get_session_state("session_queue", "session1")
    print(f"Session state: {state.decode() if state else 'None'}\n")


if __name__ == "__main__":
    print("Azure Service Bus - Advanced Implementation")
    print("=" * 60)
    print()

    # Run all demos
    demo_queue_operations()
    demo_message_receiving()
    demo_dead_letter_queue()
    demo_scheduled_messages()
    demo_topic_subscriptions()
    demo_batch_operations()
    demo_message_sessions()

    print("=" * 60)
    print("All demos completed successfully!")
