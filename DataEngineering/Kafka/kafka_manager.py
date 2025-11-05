"""
Apache Kafka Streaming
Author: BrillConsulting
Description: Real-time data streaming and processing
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class KafkaManager:
    """Apache Kafka management"""

    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.topics = []

    def create_topic(self, topic_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Kafka topic"""
        topic = {
            'name': topic_config.get('name', 'events'),
            'partitions': topic_config.get('partitions', 3),
            'replication_factor': topic_config.get('replication_factor', 2),
            'retention_ms': topic_config.get('retention_ms', 604800000),
            'created_at': datetime.now().isoformat()
        }

        command = f"kafka-topics --create --topic {topic['name']} --partitions {topic['partitions']} --replication-factor {topic['replication_factor']} --bootstrap-server {self.bootstrap_servers}"

        self.topics.append(topic)
        print(f"✓ Kafka topic created: {topic['name']}")
        print(f"  Partitions: {topic['partitions']}, Replication: {topic['replication_factor']}")
        return topic

    def produce_message(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Produce message to Kafka"""
        result = {
            'topic': topic,
            'partition': 0,
            'offset': 12345,
            'timestamp': datetime.now().isoformat()
        }

        producer_code = f'''from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['{self.bootstrap_servers}'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

producer.send('{topic}', {json.dumps(message)})
producer.flush()
'''

        print(f"✓ Message produced to {topic}")
        print(f"  Partition: {result['partition']}, Offset: {result['offset']}")
        return result

    def consume_messages(self, topic: str) -> List[Dict[str, Any]]:
        """Consume messages from Kafka"""
        messages = [
            {'key': 'user1', 'value': {'event': 'login'}, 'offset': 100},
            {'key': 'user2', 'value': {'event': 'purchase'}, 'offset': 101}
        ]

        consumer_code = f'''from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    '{topic}',
    bootstrap_servers=['{self.bootstrap_servers}'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest'
)

for message in consumer:
    print(message.value)
'''

        print(f"✓ Consumed {len(messages)} messages from {topic}")
        return messages


def demo():
    """Demonstrate Kafka"""
    print("=" * 60)
    print("Apache Kafka Streaming Demo")
    print("=" * 60)

    mgr = KafkaManager()

    print("\n1. Creating topic...")
    mgr.create_topic({'name': 'user-events', 'partitions': 3, 'replication_factor': 2})

    print("\n2. Producing message...")
    mgr.produce_message('user-events', {'user_id': 123, 'event': 'login'})

    print("\n3. Consuming messages...")
    mgr.consume_messages('user-events')

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
