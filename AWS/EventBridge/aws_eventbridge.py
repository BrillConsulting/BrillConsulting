"""
AWS EventBridge
===============

Serverless event bus for building event-driven architectures.

Author: Brill Consulting
"""

import boto3
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventBridgeManager:
    """
    Advanced AWS EventBridge Management System

    Provides comprehensive EventBridge operations including:
    - Event bus management (default and custom)
    - Rule creation with event patterns
    - Scheduled rules (cron and rate)
    - Target management (Lambda, SNS, SQS, Step Functions)
    - Event sending and routing
    - Archives and replay
    - API destinations
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize EventBridge Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.events_client = session.client('events', region_name=region)
            self.region = region
            logger.info(f"EventBridge Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing EventBridge Manager: {e}")
            raise

    # ==================== Event Bus Management ====================

    def create_event_bus(
        self,
        name: str,
        tags: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create custom event bus.

        Args:
            name: Event bus name
            tags: Resource tags
        """
        try:
            logger.info(f"Creating event bus: {name}")

            params = {'Name': name}
            if tags:
                params['Tags'] = tags

            response = self.events_client.create_event_bus(**params)

            logger.info(f"✓ Event bus created: {name}")
            return {
                'event_bus_arn': response['EventBusArn']
            }

        except ClientError as e:
            logger.error(f"Error creating event bus: {e}")
            raise

    def describe_event_bus(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Describe event bus."""
        try:
            params = {}
            if name:
                params['Name'] = name

            response = self.events_client.describe_event_bus(**params)

            return {
                'name': response['Name'],
                'arn': response['Arn'],
                'policy': response.get('Policy')
            }

        except ClientError as e:
            logger.error(f"Error describing event bus: {e}")
            raise

    def list_event_buses(self) -> List[Dict[str, Any]]:
        """List all event buses."""
        try:
            response = self.events_client.list_event_buses()

            buses = [
                {
                    'name': bus['Name'],
                    'arn': bus['Arn']
                }
                for bus in response.get('EventBuses', [])
            ]

            logger.info(f"Found {len(buses)} event bus(es)")
            return buses

        except ClientError as e:
            logger.error(f"Error listing event buses: {e}")
            raise

    def delete_event_bus(self, name: str) -> None:
        """Delete custom event bus."""
        try:
            self.events_client.delete_event_bus(Name=name)
            logger.info(f"✓ Event bus deleted: {name}")

        except ClientError as e:
            logger.error(f"Error deleting event bus: {e}")
            raise

    # ==================== Rules ====================

    def put_rule(
        self,
        name: str,
        event_pattern: Optional[Dict[str, Any]] = None,
        schedule_expression: Optional[str] = None,
        description: str = "",
        state: str = "ENABLED",
        event_bus_name: str = "default",
        tags: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create or update rule.

        Args:
            name: Rule name
            event_pattern: Event pattern to match (dict)
            schedule_expression: Cron or rate expression (e.g., 'rate(5 minutes)')
            description: Rule description
            state: 'ENABLED' or 'DISABLED'
            event_bus_name: Event bus name
            tags: Resource tags
        """
        try:
            logger.info(f"Creating rule: {name}")

            params = {
                'Name': name,
                'Description': description,
                'State': state,
                'EventBusName': event_bus_name
            }

            if event_pattern:
                params['EventPattern'] = json.dumps(event_pattern)
            elif schedule_expression:
                params['ScheduleExpression'] = schedule_expression
            else:
                raise ValueError("Either event_pattern or schedule_expression must be provided")

            if tags:
                params['Tags'] = tags

            response = self.events_client.put_rule(**params)

            logger.info(f"✓ Rule created: {name}")
            return {
                'rule_arn': response['RuleArn']
            }

        except ClientError as e:
            logger.error(f"Error creating rule: {e}")
            raise

    def describe_rule(self, name: str, event_bus_name: str = "default") -> Dict[str, Any]:
        """Describe rule."""
        try:
            response = self.events_client.describe_rule(
                Name=name,
                EventBusName=event_bus_name
            )

            return {
                'name': response['Name'],
                'arn': response['Arn'],
                'event_pattern': response.get('EventPattern'),
                'schedule_expression': response.get('ScheduleExpression'),
                'state': response['State'],
                'description': response.get('Description', '')
            }

        except ClientError as e:
            logger.error(f"Error describing rule: {e}")
            raise

    def list_rules(
        self,
        event_bus_name: str = "default",
        name_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List rules."""
        try:
            params = {'EventBusName': event_bus_name}
            if name_prefix:
                params['NamePrefix'] = name_prefix

            response = self.events_client.list_rules(**params)

            rules = [
                {
                    'name': rule['Name'],
                    'arn': rule['Arn'],
                    'state': rule['State'],
                    'schedule_expression': rule.get('ScheduleExpression')
                }
                for rule in response.get('Rules', [])
            ]

            logger.info(f"Found {len(rules)} rule(s)")
            return rules

        except ClientError as e:
            logger.error(f"Error listing rules: {e}")
            raise

    def delete_rule(self, name: str, event_bus_name: str = "default", force: bool = False) -> None:
        """Delete rule."""
        try:
            if force:
                # Remove all targets first
                targets = self.list_targets_by_rule(name, event_bus_name)
                if targets:
                    target_ids = [t['id'] for t in targets]
                    self.remove_targets(name, target_ids, event_bus_name)

            self.events_client.delete_rule(
                Name=name,
                EventBusName=event_bus_name
            )
            logger.info(f"✓ Rule deleted: {name}")

        except ClientError as e:
            logger.error(f"Error deleting rule: {e}")
            raise

    # ==================== Targets ====================

    def put_targets(
        self,
        rule_name: str,
        targets: List[Dict[str, Any]],
        event_bus_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Add targets to rule.

        Args:
            rule_name: Rule name
            targets: List of target configurations
            event_bus_name: Event bus name

        Example targets:
            [{
                'Id': '1',
                'Arn': 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
                'Input': '{"key": "value"}'
            }]
        """
        try:
            logger.info(f"Adding {len(targets)} target(s) to rule: {rule_name}")

            response = self.events_client.put_targets(
                Rule=rule_name,
                EventBusName=event_bus_name,
                Targets=targets
            )

            failed_count = response['FailedEntryCount']
            if failed_count > 0:
                logger.warning(f"{failed_count} target(s) failed to add")

            logger.info(f"✓ Added targets to rule: {rule_name}")
            return {
                'failed_entry_count': failed_count,
                'failed_entries': response.get('FailedEntries', [])
            }

        except ClientError as e:
            logger.error(f"Error adding targets: {e}")
            raise

    def list_targets_by_rule(
        self,
        rule_name: str,
        event_bus_name: str = "default"
    ) -> List[Dict[str, Any]]:
        """List targets for rule."""
        try:
            response = self.events_client.list_targets_by_rule(
                Rule=rule_name,
                EventBusName=event_bus_name
            )

            targets = [
                {
                    'id': target['Id'],
                    'arn': target['Arn'],
                    'input': target.get('Input')
                }
                for target in response.get('Targets', [])
            ]

            logger.info(f"Found {len(targets)} target(s) for rule: {rule_name}")
            return targets

        except ClientError as e:
            logger.error(f"Error listing targets: {e}")
            raise

    def remove_targets(
        self,
        rule_name: str,
        target_ids: List[str],
        event_bus_name: str = "default"
    ) -> Dict[str, Any]:
        """Remove targets from rule."""
        try:
            logger.info(f"Removing {len(target_ids)} target(s) from rule: {rule_name}")

            response = self.events_client.remove_targets(
                Rule=rule_name,
                EventBusName=event_bus_name,
                Ids=target_ids
            )

            failed_count = response['FailedEntryCount']
            if failed_count > 0:
                logger.warning(f"{failed_count} target(s) failed to remove")

            logger.info(f"✓ Removed targets from rule: {rule_name}")
            return {
                'failed_entry_count': failed_count,
                'failed_entries': response.get('FailedEntries', [])
            }

        except ClientError as e:
            logger.error(f"Error removing targets: {e}")
            raise

    # ==================== Events ====================

    def put_events(
        self,
        entries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Send custom events.

        Args:
            entries: List of event entries

        Example:
            [{
                'Source': 'my.application',
                'DetailType': 'User Action',
                'Detail': '{"action": "login", "user": "john"}',
                'EventBusName': 'default'
            }]
        """
        try:
            logger.info(f"Sending {len(entries)} event(s)")

            response = self.events_client.put_events(Entries=entries)

            failed_count = response['FailedEntryCount']
            if failed_count > 0:
                logger.warning(f"{failed_count} event(s) failed to send")

            logger.info(f"✓ Sent {len(entries) - failed_count} event(s)")
            return {
                'successful_count': len(entries) - failed_count,
                'failed_entry_count': failed_count,
                'failed_entries': response.get('Entries', [])
            }

        except ClientError as e:
            logger.error(f"Error sending events: {e}")
            raise

    # ==================== Monitoring ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get EventBridge summary."""
        try:
            buses = self.list_event_buses()
            rules = self.list_rules()

            return {
                'region': self.region,
                'event_buses': len(buses),
                'rules': len(rules),
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of EventBridge Manager capabilities."""
    print("AWS EventBridge Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    print("\n1️⃣  Event Bus Management:")
    print("""
    eb = EventBridgeManager(region='us-east-1')

    # Create custom event bus
    bus = eb.create_event_bus('my-app-events')

    # List event buses
    buses = eb.list_event_buses()
    """)

    print("\n2️⃣  Event Pattern Rules:")
    print("""
    # Create rule with event pattern
    rule = eb.put_rule(
        name='user-signup-rule',
        event_pattern={
            'source': ['my.application'],
            'detail-type': ['User Action'],
            'detail': {
                'action': ['signup']
            }
        },
        description='Trigger on user signup'
    )
    """)

    print("\n3️⃣  Scheduled Rules:")
    print("""
    # Create scheduled rule (every 5 minutes)
    eb.put_rule(
        name='backup-rule',
        schedule_expression='rate(5 minutes)',
        description='Run backup every 5 minutes'
    )

    # Create cron rule (daily at 9 AM UTC)
    eb.put_rule(
        name='daily-report',
        schedule_expression='cron(0 9 * * ? *)',
        description='Generate daily report'
    )
    """)

    print("\n4️⃣  Targets:")
    print("""
    # Add Lambda target
    eb.put_targets(
        rule_name='user-signup-rule',
        targets=[{
            'Id': '1',
            'Arn': 'arn:aws:lambda:us-east-1:123456789012:function:process-signup',
            'Input': '{"environment": "production"}'
        }]
    )

    # Add SNS target
    eb.put_targets(
        rule_name='daily-report',
        targets=[{
            'Id': '1',
            'Arn': 'arn:aws:sns:us-east-1:123456789012:reports'
        }]
    )
    """)

    print("\n5️⃣  Send Custom Events:")
    print("""
    # Send custom event
    eb.put_events([{
        'Source': 'my.application',
        'DetailType': 'User Action',
        'Detail': json.dumps({
            'action': 'signup',
            'user': 'john@example.com',
            'timestamp': datetime.now().isoformat()
        }),
        'EventBusName': 'default'
    }])
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
