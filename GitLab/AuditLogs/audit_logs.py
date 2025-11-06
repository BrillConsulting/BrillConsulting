"""
AuditLogs - Compliance and Audit Logging
Author: BrillConsulting
Description: Comprehensive GitLab audit logging for compliance, security monitoring, and event tracking
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib


class EventType(Enum):
    """Audit event types."""
    # User events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_FAILED_LOGIN = "user_failed_login"
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    USER_UPDATED = "user_updated"

    # Project events
    PROJECT_CREATED = "project_created"
    PROJECT_DELETED = "project_deleted"
    PROJECT_TRANSFERRED = "project_transferred"
    PROJECT_SETTINGS_CHANGED = "project_settings_changed"

    # Access control events
    MEMBER_ADDED = "member_added"
    MEMBER_REMOVED = "member_removed"
    MEMBER_ACCESS_CHANGED = "member_access_changed"

    # Repository events
    PUSH = "push"
    MERGE_REQUEST_CREATED = "merge_request_created"
    MERGE_REQUEST_MERGED = "merge_request_merged"
    MERGE_REQUEST_APPROVED = "merge_request_approved"
    BRANCH_CREATED = "branch_created"
    BRANCH_DELETED = "branch_deleted"
    TAG_CREATED = "tag_created"
    TAG_DELETED = "tag_deleted"

    # Security events
    DEPLOY_KEY_ADDED = "deploy_key_added"
    DEPLOY_KEY_REMOVED = "deploy_key_removed"
    ACCESS_TOKEN_CREATED = "access_token_created"
    ACCESS_TOKEN_REVOKED = "access_token_revoked"
    SSH_KEY_ADDED = "ssh_key_added"
    SSH_KEY_REMOVED = "ssh_key_removed"
    TWO_FACTOR_ENABLED = "two_factor_enabled"
    TWO_FACTOR_DISABLED = "two_factor_disabled"

    # Admin events
    ADMIN_ACTION = "admin_action"
    SYSTEM_SETTINGS_CHANGED = "system_settings_changed"
    GROUP_CREATED = "group_created"
    GROUP_DELETED = "group_deleted"

    # Compliance events
    PROTECTED_BRANCH_CREATED = "protected_branch_created"
    PROTECTED_BRANCH_REMOVED = "protected_branch_removed"
    APPROVAL_RULE_CREATED = "approval_rule_created"
    APPROVAL_RULE_DELETED = "approval_rule_deleted"


class EventSeverity(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEventManager:
    """Manage audit event creation and tracking."""

    def __init__(self, project_id: str = None):
        self.project_id = project_id
        self.events: Dict[str, Dict[str, Any]] = {}
        self.event_counter = 1

    def log_event(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log an audit event.

        Config:
        - event_type: EventType
        - actor_id: User ID performing action
        - target_id: Target user/project/resource ID
        - details: Additional event details
        - ip_address: IP address of actor
        - user_agent: User agent string
        - severity: EventSeverity (default: INFO)
        """
        event_type = config.get('event_type')
        actor_id = config.get('actor_id')
        target_id = config.get('target_id')
        details = config.get('details', {})
        ip_address = config.get('ip_address')
        user_agent = config.get('user_agent')
        severity = config.get('severity', EventSeverity.INFO)

        event_id = f"event-{self.event_counter}"
        self.event_counter += 1

        timestamp = datetime.now().isoformat()

        # Create event hash for integrity verification
        event_data = f"{event_id}{event_type}{actor_id}{target_id}{timestamp}"
        event_hash = hashlib.sha256(event_data.encode()).hexdigest()

        event = {
            "event_id": event_id,
            "event_type": event_type.value if isinstance(event_type, EventType) else event_type,
            "severity": severity.value if isinstance(severity, EventSeverity) else severity,
            "actor_id": actor_id,
            "target_id": target_id,
            "project_id": self.project_id,
            "details": details,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": timestamp,
            "event_hash": event_hash
        }

        self.events[event_id] = event
        return event

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get specific audit event."""
        return self.events.get(event_id)

    def list_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent audit events."""
        all_events = list(self.events.values())
        # Sort by timestamp (most recent first)
        all_events.sort(key=lambda x: x['timestamp'], reverse=True)
        return all_events[:limit]

    def verify_event_integrity(self, event_id: str) -> bool:
        """Verify event integrity using hash."""
        event = self.events.get(event_id)
        if not event:
            return False

        # Recalculate hash
        event_data = f"{event['event_id']}{event['event_type']}{event['actor_id']}{event['target_id']}{event['timestamp']}"
        calculated_hash = hashlib.sha256(event_data.encode()).hexdigest()

        return calculated_hash == event['event_hash']


class EventQueryManager:
    """Query and filter audit events."""

    def __init__(self, audit_events: AuditEventManager):
        self.audit_events = audit_events

    def query_by_event_type(self, event_type: EventType) -> List[Dict[str, Any]]:
        """Query events by event type."""
        event_type_value = event_type.value if isinstance(event_type, EventType) else event_type
        return [
            event for event in self.audit_events.events.values()
            if event['event_type'] == event_type_value
        ]

    def query_by_actor(self, actor_id: int) -> List[Dict[str, Any]]:
        """Query events by actor (user who performed action)."""
        return [
            event for event in self.audit_events.events.values()
            if event['actor_id'] == actor_id
        ]

    def query_by_target(self, target_id: Any) -> List[Dict[str, Any]]:
        """Query events by target (affected resource)."""
        return [
            event for event in self.audit_events.events.values()
            if event['target_id'] == target_id
        ]

    def query_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Query events by date range."""
        return [
            event for event in self.audit_events.events.values()
            if start_date <= event['timestamp'] <= end_date
        ]

    def query_by_severity(self, severity: EventSeverity) -> List[Dict[str, Any]]:
        """Query events by severity level."""
        severity_value = severity.value if isinstance(severity, EventSeverity) else severity
        return [
            event for event in self.audit_events.events.values()
            if event['severity'] == severity_value
        ]

    def query_by_ip_address(self, ip_address: str) -> List[Dict[str, Any]]:
        """Query events by IP address."""
        return [
            event for event in self.audit_events.events.values()
            if event['ip_address'] == ip_address
        ]

    def advanced_query(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Advanced query with multiple filters.

        Filters:
        - event_types: List of EventType
        - actor_ids: List of actor IDs
        - severity: EventSeverity
        - start_date: ISO format date
        - end_date: ISO format date
        - ip_address: IP address
        """
        results = list(self.audit_events.events.values())

        # Filter by event types
        if 'event_types' in filters:
            event_type_values = [
                et.value if isinstance(et, EventType) else et
                for et in filters['event_types']
            ]
            results = [e for e in results if e['event_type'] in event_type_values]

        # Filter by actor IDs
        if 'actor_ids' in filters:
            results = [e for e in results if e['actor_id'] in filters['actor_ids']]

        # Filter by severity
        if 'severity' in filters:
            severity = filters['severity']
            severity_value = severity.value if isinstance(severity, EventSeverity) else severity
            results = [e for e in results if e['severity'] == severity_value]

        # Filter by date range
        if 'start_date' in filters:
            results = [e for e in results if e['timestamp'] >= filters['start_date']]
        if 'end_date' in filters:
            results = [e for e in results if e['timestamp'] <= filters['end_date']]

        # Filter by IP address
        if 'ip_address' in filters:
            results = [e for e in results if e['ip_address'] == filters['ip_address']]

        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x['timestamp'], reverse=True)

        return results


class ComplianceReportManager:
    """Generate compliance and audit reports."""

    def __init__(self, audit_events: AuditEventManager):
        self.audit_events = audit_events

    def generate_user_activity_report(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Generate user activity report for specified period."""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        user_events = [
            event for event in self.audit_events.events.values()
            if event['actor_id'] == user_id and event['timestamp'] >= start_date
        ]

        # Categorize events
        event_counts = {}
        for event in user_events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "user_id": user_id,
            "report_period_days": days,
            "start_date": start_date,
            "end_date": datetime.now().isoformat(),
            "total_events": len(user_events),
            "event_breakdown": event_counts,
            "events": user_events
        }

    def generate_security_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate security events report."""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        security_event_types = [
            EventType.USER_FAILED_LOGIN,
            EventType.DEPLOY_KEY_ADDED,
            EventType.DEPLOY_KEY_REMOVED,
            EventType.ACCESS_TOKEN_CREATED,
            EventType.ACCESS_TOKEN_REVOKED,
            EventType.SSH_KEY_ADDED,
            EventType.SSH_KEY_REMOVED,
            EventType.TWO_FACTOR_ENABLED,
            EventType.TWO_FACTOR_DISABLED
        ]

        security_events = [
            event for event in self.audit_events.events.values()
            if event['timestamp'] >= start_date and
            any(event['event_type'] == et.value for et in security_event_types)
        ]

        # Critical events
        critical_events = [
            event for event in security_events
            if event['severity'] in [EventSeverity.CRITICAL.value, EventSeverity.ERROR.value]
        ]

        # Failed logins by user
        failed_logins = {}
        for event in security_events:
            if event['event_type'] == EventType.USER_FAILED_LOGIN.value:
                user_id = event['actor_id']
                failed_logins[user_id] = failed_logins.get(user_id, 0) + 1

        return {
            "report_type": "security",
            "period_days": days,
            "start_date": start_date,
            "end_date": datetime.now().isoformat(),
            "total_security_events": len(security_events),
            "critical_events_count": len(critical_events),
            "failed_logins_by_user": failed_logins,
            "security_events": security_events,
            "critical_events": critical_events
        }

    def generate_compliance_report(self, days: int = 90) -> Dict[str, Any]:
        """Generate compliance audit report (SOC2, GDPR, HIPAA)."""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        all_events = [
            event for event in self.audit_events.events.values()
            if event['timestamp'] >= start_date
        ]

        # Access control changes
        access_events = [
            event for event in all_events
            if event['event_type'] in [
                EventType.MEMBER_ADDED.value,
                EventType.MEMBER_REMOVED.value,
                EventType.MEMBER_ACCESS_CHANGED.value
            ]
        ]

        # Data access events
        data_access_events = [
            event for event in all_events
            if event['event_type'] in [
                EventType.PUSH.value,
                EventType.BRANCH_CREATED.value,
                EventType.MERGE_REQUEST_MERGED.value
            ]
        ]

        # Administrative actions
        admin_events = [
            event for event in all_events
            if event['event_type'] in [
                EventType.ADMIN_ACTION.value,
                EventType.SYSTEM_SETTINGS_CHANGED.value,
                EventType.PROJECT_DELETED.value,
                EventType.USER_DELETED.value
            ]
        ]

        # Protected resource changes
        protection_events = [
            event for event in all_events
            if event['event_type'] in [
                EventType.PROTECTED_BRANCH_CREATED.value,
                EventType.PROTECTED_BRANCH_REMOVED.value,
                EventType.APPROVAL_RULE_CREATED.value
            ]
        ]

        return {
            "report_type": "compliance",
            "period_days": days,
            "start_date": start_date,
            "end_date": datetime.now().isoformat(),
            "total_events": len(all_events),
            "access_control_changes": len(access_events),
            "data_access_events": len(data_access_events),
            "administrative_actions": len(admin_events),
            "protection_changes": len(protection_events),
            "events_by_category": {
                "access_control": access_events,
                "data_access": data_access_events,
                "administrative": admin_events,
                "protection": protection_events
            }
        }

    def generate_project_activity_report(self, project_id: str, days: int = 30) -> Dict[str, Any]:
        """Generate project activity report."""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        project_events = [
            event for event in self.audit_events.events.values()
            if event['project_id'] == project_id and event['timestamp'] >= start_date
        ]

        # Count unique contributors
        contributors = set(event['actor_id'] for event in project_events)

        # Event breakdown
        event_counts = {}
        for event in project_events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "project_id": project_id,
            "report_period_days": days,
            "start_date": start_date,
            "end_date": datetime.now().isoformat(),
            "total_events": len(project_events),
            "unique_contributors": len(contributors),
            "event_breakdown": event_counts,
            "events": project_events
        }


class LogRetentionManager:
    """Manage audit log retention and cleanup."""

    def __init__(self, audit_events: AuditEventManager):
        self.audit_events = audit_events
        self.retention_policies: Dict[str, int] = {}
        self.policy_counter = 1

    def create_retention_policy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create log retention policy.

        Config:
        - name: Policy name
        - retention_days: Number of days to retain logs
        - event_types: List of EventType (optional, applies to all if not specified)
        - severity: EventSeverity (optional)
        """
        policy_id = f"policy-{self.policy_counter}"
        self.policy_counter += 1

        policy = {
            "policy_id": policy_id,
            "name": config.get('name'),
            "retention_days": config.get('retention_days', 90),
            "event_types": config.get('event_types', []),
            "severity": config.get('severity'),
            "created_at": datetime.now().isoformat()
        }

        self.retention_policies[policy_id] = policy
        return policy

    def apply_retention_policies(self) -> Dict[str, Any]:
        """Apply retention policies and clean up old logs."""
        deleted_count = 0
        retained_count = 0

        for event_id, event in list(self.audit_events.events.items()):
            should_delete = False

            for policy in self.retention_policies.values():
                # Check if event matches policy
                if policy['event_types'] and event['event_type'] not in policy['event_types']:
                    continue

                if policy['severity'] and event['severity'] != policy['severity']:
                    continue

                # Check retention period
                event_date = datetime.fromisoformat(event['timestamp'])
                retention_date = datetime.now() - timedelta(days=policy['retention_days'])

                if event_date < retention_date:
                    should_delete = True
                    break

            if should_delete:
                del self.audit_events.events[event_id]
                deleted_count += 1
            else:
                retained_count += 1

        return {
            "deleted_events": deleted_count,
            "retained_events": retained_count,
            "policies_applied": len(self.retention_policies),
            "timestamp": datetime.now().isoformat()
        }

    def get_retention_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get retention policy details."""
        return self.retention_policies.get(policy_id)

    def delete_retention_policy(self, policy_id: str) -> Dict[str, Any]:
        """Delete retention policy."""
        if policy_id in self.retention_policies:
            del self.retention_policies[policy_id]
            return {"status": "deleted", "policy_id": policy_id}
        return {"status": "not_found", "policy_id": policy_id}


class LogExportManager:
    """Export audit logs in various formats."""

    def __init__(self, audit_events: AuditEventManager):
        self.audit_events = audit_events

    def export_to_json(self, events: List[Dict[str, Any]]) -> str:
        """Export events to JSON format."""
        return json.dumps(events, indent=2)

    def export_to_csv(self, events: List[Dict[str, Any]]) -> str:
        """Export events to CSV format."""
        if not events:
            return ""

        # CSV header
        headers = ["event_id", "event_type", "severity", "actor_id", "target_id",
                  "project_id", "ip_address", "timestamp"]
        csv_lines = [",".join(headers)]

        # CSV rows
        for event in events:
            row = [
                str(event.get(header, '')) for header in headers
            ]
            csv_lines.append(",".join(row))

        return "\n".join(csv_lines)

    def export_to_syslog(self, events: List[Dict[str, Any]]) -> List[str]:
        """Export events to syslog format."""
        syslog_messages = []

        for event in events:
            # Syslog format: <timestamp> <hostname> <app>: <message>
            message = (
                f"{event['timestamp']} gitlab audit[{event['event_id']}]: "
                f"type={event['event_type']} severity={event['severity']} "
                f"actor={event['actor_id']} target={event['target_id']} "
                f"ip={event['ip_address']}"
            )
            syslog_messages.append(message)

        return syslog_messages

    def export_all_events(self, format: str = 'json') -> str:
        """Export all events in specified format."""
        all_events = list(self.audit_events.events.values())

        if format == 'json':
            return self.export_to_json(all_events)
        elif format == 'csv':
            return self.export_to_csv(all_events)
        elif format == 'syslog':
            return "\n".join(self.export_to_syslog(all_events))
        else:
            return json.dumps({"error": f"Unsupported format: {format}"})


class SIEMIntegrationManager:
    """Integration with SIEM (Security Information and Event Management) systems."""

    def __init__(self, audit_events: AuditEventManager):
        self.audit_events = audit_events
        self.siem_configs: Dict[str, Dict[str, Any]] = {}
        self.config_counter = 1

    def configure_siem(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure SIEM integration.

        Config:
        - name: SIEM system name (Splunk, ELK, ArcSight, QRadar)
        - endpoint_url: SIEM API endpoint
        - api_key: Authentication key
        - event_filters: List of EventType to forward
        - batch_size: Number of events per batch (default: 100)
        """
        config_id = f"siem-config-{self.config_counter}"
        self.config_counter += 1

        siem_config = {
            "config_id": config_id,
            "name": config.get('name'),
            "endpoint_url": config.get('endpoint_url'),
            "api_key": config.get('api_key', '***'),
            "event_filters": config.get('event_filters', []),
            "batch_size": config.get('batch_size', 100),
            "enabled": True,
            "created_at": datetime.now().isoformat()
        }

        self.siem_configs[config_id] = siem_config
        return siem_config

    def forward_events_to_siem(self, config_id: str) -> Dict[str, Any]:
        """Forward events to configured SIEM system."""
        config = self.siem_configs.get(config_id)
        if not config or not config['enabled']:
            return {"status": "error", "message": "SIEM config not found or disabled"}

        # Get events to forward
        events = list(self.audit_events.events.values())

        # Apply filters
        if config['event_filters']:
            event_type_values = [
                et.value if isinstance(et, EventType) else et
                for et in config['event_filters']
            ]
            events = [e for e in events if e['event_type'] in event_type_values]

        # Simulate forwarding in batches
        batch_size = config['batch_size']
        batches_sent = (len(events) + batch_size - 1) // batch_size

        return {
            "status": "success",
            "config_id": config_id,
            "siem_system": config['name'],
            "events_forwarded": len(events),
            "batches_sent": batches_sent,
            "timestamp": datetime.now().isoformat()
        }

    def disable_siem(self, config_id: str) -> Dict[str, Any]:
        """Disable SIEM integration."""
        if config_id in self.siem_configs:
            self.siem_configs[config_id]['enabled'] = False
            return {"status": "disabled", "config_id": config_id}
        return {"status": "not_found", "config_id": config_id}


class AuditLogsManager:
    """Main audit logs manager integrating all components."""

    def __init__(self, project_id: str = None, gitlab_url: str = 'https://gitlab.com'):
        self.project_id = project_id
        self.gitlab_url = gitlab_url

        # Initialize all managers
        self.events = AuditEventManager(project_id)
        self.query = EventQueryManager(self.events)
        self.reports = ComplianceReportManager(self.events)
        self.retention = LogRetentionManager(self.events)
        self.export = LogExportManager(self.events)
        self.siem = SIEMIntegrationManager(self.events)

    def info(self) -> Dict[str, Any]:
        """Get audit logs system information."""
        return {
            "project_id": self.project_id,
            "gitlab_url": self.gitlab_url,
            "total_events": len(self.events.events),
            "retention_policies": len(self.retention.retention_policies),
            "siem_integrations": len(self.siem.siem_configs),
            "capabilities": [
                "event_logging",
                "event_querying",
                "compliance_reporting",
                "log_retention",
                "multi_format_export",
                "siem_integration",
                "integrity_verification"
            ]
        }


def demo():
    """Demonstrate audit logs capabilities."""
    print("=" * 80)
    print("GitLab Audit Logs - Comprehensive Demo")
    print("=" * 80)

    # Initialize manager
    mgr = AuditLogsManager(project_id='myorg/myproject')

    print("\n📋 1. Logging Audit Events")
    print("-" * 80)

    # Log various events
    login_event = mgr.events.log_event({
        'event_type': EventType.USER_LOGIN,
        'actor_id': 101,
        'target_id': 101,
        'details': {'method': 'password'},
        'ip_address': '192.168.1.100',
        'user_agent': 'Mozilla/5.0',
        'severity': EventSeverity.INFO
    })
    print(f"✓ Logged login event: {login_event['event_id']}")

    # Failed login (security event)
    failed_login = mgr.events.log_event({
        'event_type': EventType.USER_FAILED_LOGIN,
        'actor_id': 999,
        'target_id': 999,
        'details': {'reason': 'invalid_password', 'attempts': 3},
        'ip_address': '10.0.0.50',
        'user_agent': 'curl/7.68.0',
        'severity': EventSeverity.WARNING
    })
    print(f"✓ Logged failed login: {failed_login['event_id']} (severity: {failed_login['severity']})")

    # Member added event
    member_event = mgr.events.log_event({
        'event_type': EventType.MEMBER_ADDED,
        'actor_id': 1,
        'target_id': 102,
        'details': {'access_level': 30, 'role': 'Developer'},
        'ip_address': '192.168.1.10',
        'severity': EventSeverity.INFO
    })
    print(f"✓ Logged member added: {member_event['event_id']}")

    # Deploy key added (security event)
    deploy_key_event = mgr.events.log_event({
        'event_type': EventType.DEPLOY_KEY_ADDED,
        'actor_id': 1,
        'target_id': 'deploy-key-1',
        'details': {'key_title': 'CI Deploy Key', 'can_push': True},
        'ip_address': '192.168.1.10',
        'severity': EventSeverity.INFO
    })
    print(f"✓ Logged deploy key added: {deploy_key_event['event_id']}")

    # Merge request merged
    merge_event = mgr.events.log_event({
        'event_type': EventType.MERGE_REQUEST_MERGED,
        'actor_id': 102,
        'target_id': 'MR-123',
        'details': {'source_branch': 'feature/new-feature', 'target_branch': 'main'},
        'ip_address': '192.168.1.100',
        'severity': EventSeverity.INFO
    })
    print(f"✓ Logged merge request merged: {merge_event['event_id']}")

    # Admin action (critical)
    admin_event = mgr.events.log_event({
        'event_type': EventType.SYSTEM_SETTINGS_CHANGED,
        'actor_id': 1,
        'target_id': 'system',
        'details': {'setting': 'allow_local_requests', 'old_value': False, 'new_value': True},
        'ip_address': '192.168.1.10',
        'severity': EventSeverity.CRITICAL
    })
    print(f"✓ Logged admin action: {admin_event['event_id']} (severity: {admin_event['severity']})")

    print("\n🔍 2. Querying Audit Events")
    print("-" * 80)

    # Query by event type
    login_events = mgr.query.query_by_event_type(EventType.USER_LOGIN)
    print(f"✓ Found {len(login_events)} login events")

    # Query by actor
    admin_events = mgr.query.query_by_actor(1)
    print(f"✓ Found {len(admin_events)} events by admin user (ID: 1)")

    # Query by severity
    critical_events = mgr.query.query_by_severity(EventSeverity.CRITICAL)
    print(f"✓ Found {len(critical_events)} critical events")

    # Advanced query
    security_query = mgr.query.advanced_query({
        'event_types': [EventType.USER_FAILED_LOGIN, EventType.DEPLOY_KEY_ADDED],
        'severity': EventSeverity.WARNING
    })
    print(f"✓ Advanced query found {len(security_query)} matching events")

    print("\n🔐 3. Event Integrity Verification")
    print("-" * 80)

    # Verify event integrity
    for event_id in list(mgr.events.events.keys())[:3]:
        is_valid = mgr.events.verify_event_integrity(event_id)
        print(f"✓ Event {event_id}: {'VALID' if is_valid else 'INVALID'}")

    print("\n📊 4. Compliance Reports")
    print("-" * 80)

    # User activity report
    user_report = mgr.reports.generate_user_activity_report(user_id=102, days=30)
    print(f"✓ User activity report: {user_report['total_events']} events in {user_report['report_period_days']} days")

    # Security report
    security_report = mgr.reports.generate_security_report(days=7)
    print(f"✓ Security report: {security_report['total_security_events']} security events, "
          f"{security_report['critical_events_count']} critical")

    # Compliance report
    compliance_report = mgr.reports.generate_compliance_report(days=90)
    print(f"✓ Compliance report: {compliance_report['total_events']} total events")
    print(f"  - Access control changes: {compliance_report['access_control_changes']}")
    print(f"  - Administrative actions: {compliance_report['administrative_actions']}")

    # Project activity report
    project_report = mgr.reports.generate_project_activity_report('myorg/myproject', days=30)
    print(f"✓ Project activity: {project_report['total_events']} events, "
          f"{project_report['unique_contributors']} contributors")

    print("\n📅 5. Log Retention Policies")
    print("-" * 80)

    # Create retention policy for general logs (90 days)
    general_policy = mgr.retention.create_retention_policy({
        'name': 'general-logs-90d',
        'retention_days': 90
    })
    print(f"✓ Created retention policy: {general_policy['name']} ({general_policy['retention_days']} days)")

    # Create retention policy for security logs (365 days)
    security_policy = mgr.retention.create_retention_policy({
        'name': 'security-logs-1y',
        'retention_days': 365,
        'event_types': [EventType.USER_FAILED_LOGIN, EventType.ACCESS_TOKEN_CREATED]
    })
    print(f"✓ Created security retention policy: {security_policy['name']} ({security_policy['retention_days']} days)")

    # Apply retention policies
    retention_result = mgr.retention.apply_retention_policies()
    print(f"✓ Applied retention policies: {retention_result['deleted_events']} deleted, "
          f"{retention_result['retained_events']} retained")

    print("\n💾 6. Export Audit Logs")
    print("-" * 80)

    recent_events = mgr.events.list_events(limit=3)

    # Export to JSON
    json_export = mgr.export.export_to_json(recent_events)
    print(f"✓ Exported {len(recent_events)} events to JSON ({len(json_export)} bytes)")

    # Export to CSV
    csv_export = mgr.export.export_to_csv(recent_events)
    print(f"✓ Exported {len(recent_events)} events to CSV ({len(csv_export)} bytes)")

    # Export to syslog
    syslog_export = mgr.export.export_to_syslog(recent_events)
    print(f"✓ Exported {len(recent_events)} events to syslog format ({len(syslog_export)} messages)")

    print("\n🔗 7. SIEM Integration")
    print("-" * 80)

    # Configure Splunk integration
    splunk_config = mgr.siem.configure_siem({
        'name': 'Splunk',
        'endpoint_url': 'https://splunk.example.com:8088/services/collector',
        'api_key': 'splunk-hec-token-xxx',
        'event_filters': [EventType.USER_FAILED_LOGIN, EventType.ADMIN_ACTION],
        'batch_size': 50
    })
    print(f"✓ Configured SIEM: {splunk_config['name']} (batch size: {splunk_config['batch_size']})")

    # Configure ELK Stack integration
    elk_config = mgr.siem.configure_siem({
        'name': 'ELK Stack',
        'endpoint_url': 'https://elasticsearch.example.com:9200',
        'api_key': 'elk-api-key-xxx',
        'batch_size': 100
    })
    print(f"✓ Configured SIEM: {elk_config['name']}")

    # Forward events
    forward_result = mgr.siem.forward_events_to_siem(splunk_config['config_id'])
    print(f"✓ Forwarded events to {forward_result['siem_system']}: "
          f"{forward_result['events_forwarded']} events in {forward_result['batches_sent']} batches")

    print("\n📈 8. Summary Statistics")
    print("-" * 80)

    info = mgr.info()
    print(f"Total events logged: {info['total_events']}")
    print(f"Retention policies: {info['retention_policies']}")
    print(f"SIEM integrations: {info['siem_integrations']}")
    print(f"Capabilities: {', '.join(info['capabilities'])}")

    print("\n✅ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo()
