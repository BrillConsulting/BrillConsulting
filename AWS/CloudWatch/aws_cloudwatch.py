"""
AWS CloudWatch Management
=========================

Comprehensive monitoring, logging, and alerting with CloudWatch.

Author: Brill Consulting
"""

import boto3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from botocore.exceptions import ClientError, NoCredentialsError


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CloudWatchManager:
    """
    Advanced AWS CloudWatch Management System

    Features:
    - Metric management and publishing
    - Alarms configuration
    - Log group and stream operations
    - Dashboard creation
    - Insights queries
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize CloudWatch Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.cloudwatch_client = session.client('cloudwatch', region_name=region)
            self.logs_client = session.client('logs', region_name=region)
            self.region = region
            logger.info(f"CloudWatch Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise

    # ==================== Metrics ====================

    def put_metric_data(
        self,
        namespace: str,
        metric_name: str,
        value: float,
        unit: str = 'None',
        dimensions: Optional[List[Dict[str, str]]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Publish custom metric."""
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': timestamp or datetime.utcnow()
            }

            if dimensions:
                metric_data['Dimensions'] = dimensions

            self.cloudwatch_client.put_metric_data(
                Namespace=namespace,
                MetricData=[metric_data]
            )
            logger.info(f"✓ Metric published: {namespace}/{metric_name} = {value}")

        except ClientError as e:
            logger.error(f"Error putting metric: {e}")
            raise

    def get_metric_statistics(
        self,
        namespace: str,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int,
        statistics: List[str],
        dimensions: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """Get metric statistics."""
        try:
            params = {
                'Namespace': namespace,
                'MetricName': metric_name,
                'StartTime': start_time,
                'EndTime': end_time,
                'Period': period,
                'Statistics': statistics
            }

            if dimensions:
                params['Dimensions'] = dimensions

            response = self.cloudwatch_client.get_metric_statistics(**params)

            datapoints = response.get('Datapoints', [])
            logger.info(f"Retrieved {len(datapoints)} datapoint(s)")
            return sorted(datapoints, key=lambda x: x['Timestamp'])

        except ClientError as e:
            logger.error(f"Error getting metric statistics: {e}")
            raise

    # ==================== Alarms ====================

    def put_metric_alarm(
        self,
        alarm_name: str,
        comparison_operator: str,
        evaluation_periods: int,
        metric_name: str,
        namespace: str,
        period: int,
        statistic: str,
        threshold: float,
        actions_enabled: bool = True,
        alarm_actions: Optional[List[str]] = None,
        dimensions: Optional[List[Dict[str, str]]] = None,
        description: str = ""
    ) -> None:
        """Create or update metric alarm."""
        try:
            params = {
                'AlarmName': alarm_name,
                'ComparisonOperator': comparison_operator,
                'EvaluationPeriods': evaluation_periods,
                'MetricName': metric_name,
                'Namespace': namespace,
                'Period': period,
                'Statistic': statistic,
                'Threshold': threshold,
                'ActionsEnabled': actions_enabled,
                'AlarmDescription': description
            }

            if alarm_actions:
                params['AlarmActions'] = alarm_actions
            if dimensions:
                params['Dimensions'] = dimensions

            self.cloudwatch_client.put_metric_alarm(**params)
            logger.info(f"✓ Alarm created: {alarm_name}")

        except ClientError as e:
            logger.error(f"Error creating alarm: {e}")
            raise

    def describe_alarms(
        self,
        alarm_names: Optional[List[str]] = None,
        state_value: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Describe alarms."""
        try:
            params = {}
            if alarm_names:
                params['AlarmNames'] = alarm_names
            if state_value:
                params['StateValue'] = state_value

            response = self.cloudwatch_client.describe_alarms(**params)

            alarms = []
            for alarm in response.get('MetricAlarms', []):
                alarms.append({
                    'alarm_name': alarm['AlarmName'],
                    'state': alarm['StateValue'],
                    'metric_name': alarm['MetricName'],
                    'threshold': alarm['Threshold'],
                    'comparison': alarm['ComparisonOperator']
                })

            logger.info(f"Found {len(alarms)} alarm(s)")
            return alarms

        except ClientError as e:
            logger.error(f"Error describing alarms: {e}")
            raise

    def delete_alarms(self, alarm_names: List[str]) -> None:
        """Delete alarms."""
        try:
            self.cloudwatch_client.delete_alarms(AlarmNames=alarm_names)
            logger.info(f"✓ Deleted {len(alarm_names)} alarm(s)")
        except ClientError as e:
            logger.error(f"Error deleting alarms: {e}")
            raise

    # ==================== Logs ====================

    def create_log_group(self, log_group_name: str) -> None:
        """Create log group."""
        try:
            self.logs_client.create_log_group(logGroupName=log_group_name)
            logger.info(f"✓ Log group created: {log_group_name}")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                logger.error(f"Error creating log group: {e}")
                raise

    def create_log_stream(self, log_group_name: str, log_stream_name: str) -> None:
        """Create log stream."""
        try:
            self.logs_client.create_log_stream(
                logGroupName=log_group_name,
                logStreamName=log_stream_name
            )
            logger.info(f"✓ Log stream created: {log_stream_name}")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                logger.error(f"Error creating log stream: {e}")
                raise

    def put_log_events(
        self,
        log_group_name: str,
        log_stream_name: str,
        messages: List[str],
        sequence_token: Optional[str] = None
    ) -> str:
        """Put log events."""
        try:
            log_events = [
                {'timestamp': int(datetime.utcnow().timestamp() * 1000), 'message': msg}
                for msg in messages
            ]

            params = {
                'logGroupName': log_group_name,
                'logStreamName': log_stream_name,
                'logEvents': log_events
            }

            if sequence_token:
                params['sequenceToken'] = sequence_token

            response = self.logs_client.put_log_events(**params)

            logger.info(f"✓ Put {len(messages)} log event(s)")
            return response.get('nextSequenceToken', '')

        except ClientError as e:
            logger.error(f"Error putting log events: {e}")
            raise

    def filter_log_events(
        self,
        log_group_name: str,
        filter_pattern: str = "",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Filter log events."""
        try:
            params = {
                'logGroupName': log_group_name,
                'filterPattern': filter_pattern,
                'limit': limit
            }

            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            response = self.logs_client.filter_log_events(**params)

            events = response.get('events', [])
            logger.info(f"Found {len(events)} log event(s)")
            return events

        except ClientError as e:
            logger.error(f"Error filtering log events: {e}")
            raise

    # ==================== Dashboards ====================

    def put_dashboard(self, dashboard_name: str, dashboard_body: str) -> None:
        """Create or update dashboard."""
        try:
            self.cloudwatch_client.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=dashboard_body
            )
            logger.info(f"✓ Dashboard created: {dashboard_name}")
        except ClientError as e:
            logger.error(f"Error creating dashboard: {e}")
            raise

    def list_dashboards(self) -> List[str]:
        """List dashboards."""
        try:
            response = self.cloudwatch_client.list_dashboards()
            dashboards = [d['DashboardName'] for d in response.get('DashboardEntries', [])]
            logger.info(f"Found {len(dashboards)} dashboard(s)")
            return dashboards
        except ClientError as e:
            logger.error(f"Error listing dashboards: {e}")
            raise

    def get_summary(self) -> Dict[str, Any]:
        """Get CloudWatch summary."""
        try:
            alarms = self.describe_alarms()
            dashboards = self.list_dashboards()

            return {
                'region': self.region,
                'total_alarms': len(alarms),
                'total_dashboards': len(dashboards),
                'timestamp': datetime.now().isoformat()
            }
        except ClientError as e:
            return {'error': str(e)}


def demo():
    """Demo CloudWatch Manager."""
    print("AWS CloudWatch Manager - Demo")
    print("=" * 70)
    print("\n📋 Usage Examples:")
    print("""
    cw = CloudWatchManager(region='us-east-1')

    # Publish metric
    cw.put_metric_data('MyApp', 'RequestCount', 100, unit='Count')

    # Create alarm
    cw.put_metric_alarm(
        alarm_name='HighCPU',
        comparison_operator='GreaterThanThreshold',
        evaluation_periods=2,
        metric_name='CPUUtilization',
        namespace='AWS/EC2',
        period=300,
        statistic='Average',
        threshold=80.0,
        alarm_actions=['arn:aws:sns:us-east-1:123456789012:alerts']
    )

    # Create log group and put logs
    cw.create_log_group('/aws/myapp')
    cw.create_log_stream('/aws/myapp', '2024-11-05')
    cw.put_log_events('/aws/myapp', '2024-11-05', ['App started', 'Request processed'])
    """)
    print("\n✓ Demo Complete!")


if __name__ == '__main__':
    demo()
