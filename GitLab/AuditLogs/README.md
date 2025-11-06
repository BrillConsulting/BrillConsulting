# Audit Logs - Compliance and Security Monitoring

Comprehensive GitLab audit logging system for compliance reporting, security monitoring, and event tracking with SIEM integration, log retention management, and multi-format export capabilities.

## Features

### Event Logging
- **40+ Event Types**: User, project, repository, security, admin, and compliance events
- **Severity Levels**: INFO, WARNING, ERROR, CRITICAL classification
- **Rich Event Data**: Actor, target, IP address, user agent, details
- **Integrity Verification**: SHA256 hash-based event validation
- **Automatic Timestamping**: ISO format timestamps for all events
- **Event Metadata**: Detailed context for each logged action

### Event Querying
- **Query by Event Type**: Filter by specific event categories
- **Query by Actor**: Find all actions by a user
- **Query by Target**: Track changes to specific resources
- **Date Range Queries**: Time-based event filtering
- **Severity Filtering**: Find critical/warning events
- **IP Address Queries**: Track actions from specific locations
- **Advanced Multi-Filter**: Combine multiple criteria

### Compliance Reporting
- **User Activity Reports**: Individual user action tracking
- **Security Reports**: Failed logins, key changes, 2FA events
- **Compliance Reports**: SOC2, GDPR, HIPAA audit trails
- **Project Activity Reports**: Repository activity tracking
- **Event Categorization**: Automatic event type classification
- **Contributor Analytics**: Unique contributor tracking

### Log Retention Management
- **Retention Policies**: Configurable retention periods
- **Event Type Filtering**: Different retention for event types
- **Severity-Based Retention**: Longer retention for critical events
- **Automated Cleanup**: Scheduled log purging
- **Policy Management**: Create, update, delete policies
- **Compliance Support**: Meet regulatory requirements

### Multi-Format Export
- **JSON Export**: Structured JSON output
- **CSV Export**: Spreadsheet-compatible format
- **Syslog Export**: Standard syslog format
- **Batch Export**: Export all or filtered events
- **Streaming Support**: Real-time event streaming

### SIEM Integration
- **Splunk Integration**: HEC (HTTP Event Collector) support
- **ELK Stack**: Elasticsearch integration
- **ArcSight**: HP ArcSight SIEM support
- **QRadar**: IBM QRadar integration
- **Event Filtering**: Forward specific event types
- **Batch Forwarding**: Configurable batch sizes
- **Enable/Disable**: Control SIEM forwarding

### Integrity Verification
- **SHA256 Hashing**: Cryptographic event integrity
- **Tamper Detection**: Verify events haven't been modified
- **Audit Chain**: Verify entire event log integrity

## Usage Example

```python
from audit_logs import AuditLogsManager, EventType, EventSeverity

# Initialize manager
mgr = AuditLogsManager(project_id='myorg/myproject')

# 1. Log events with different severity levels
# Successful login (INFO)
login = mgr.events.log_event({
    'event_type': EventType.USER_LOGIN,
    'actor_id': 101,
    'target_id': 101,
    'details': {'method': 'password', 'session_id': 'abc123'},
    'ip_address': '192.168.1.100',
    'user_agent': 'Mozilla/5.0',
    'severity': EventSeverity.INFO
})

# Failed login (WARNING)
failed_login = mgr.events.log_event({
    'event_type': EventType.USER_FAILED_LOGIN,
    'actor_id': 999,
    'target_id': 999,
    'details': {'reason': 'invalid_password', 'attempts': 3},
    'ip_address': '10.0.0.50',
    'user_agent': 'curl/7.68.0',
    'severity': EventSeverity.WARNING
})

# System settings changed (CRITICAL)
admin_action = mgr.events.log_event({
    'event_type': EventType.SYSTEM_SETTINGS_CHANGED,
    'actor_id': 1,
    'target_id': 'system',
    'details': {'setting': 'allow_local_requests', 'old_value': False, 'new_value': True},
    'ip_address': '192.168.1.10',
    'severity': EventSeverity.CRITICAL
})

# Deploy key added (INFO)
deploy_key = mgr.events.log_event({
    'event_type': EventType.DEPLOY_KEY_ADDED,
    'actor_id': 1,
    'target_id': 'deploy-key-1',
    'details': {'key_title': 'CI Deploy Key', 'can_push': True},
    'ip_address': '192.168.1.10',
    'severity': EventSeverity.INFO
})

# 2. Query audit events
# Find all login events
logins = mgr.query.query_by_event_type(EventType.USER_LOGIN)

# Find all actions by admin user
admin_events = mgr.query.query_by_actor(actor_id=1)

# Find all critical events
critical = mgr.query.query_by_severity(EventSeverity.CRITICAL)

# Query by date range
from datetime import datetime, timedelta
start_date = (datetime.now() - timedelta(days=7)).isoformat()
end_date = datetime.now().isoformat()
recent = mgr.query.query_by_date_range(start_date, end_date)

# Advanced multi-filter query
security_events = mgr.query.advanced_query({
    'event_types': [EventType.USER_FAILED_LOGIN, EventType.DEPLOY_KEY_ADDED],
    'severity': EventSeverity.WARNING,
    'start_date': start_date,
    'ip_address': '10.0.0.50'
})

# 3. Verify event integrity
is_valid = mgr.events.verify_event_integrity('event-1')
print(f"Event integrity: {'VALID' if is_valid else 'TAMPERED'}")

# 4. Generate compliance reports
# User activity report (30 days)
user_report = mgr.reports.generate_user_activity_report(
    user_id=102,
    days=30
)
# Returns: total_events, event_breakdown, events

# Security report (7 days)
security_report = mgr.reports.generate_security_report(days=7)
# Returns: total_security_events, critical_events_count, failed_logins_by_user

# Compliance report for auditors (90 days)
compliance_report = mgr.reports.generate_compliance_report(days=90)
# Returns: access_control_changes, data_access_events, administrative_actions

# Project activity report
project_report = mgr.reports.generate_project_activity_report(
    project_id='myorg/myproject',
    days=30
)
# Returns: total_events, unique_contributors, event_breakdown

# 5. Configure log retention policies
# General logs - 90 days
general_policy = mgr.retention.create_retention_policy({
    'name': 'general-logs-90d',
    'retention_days': 90
})

# Security logs - 1 year
security_policy = mgr.retention.create_retention_policy({
    'name': 'security-logs-1y',
    'retention_days': 365,
    'event_types': [EventType.USER_FAILED_LOGIN, EventType.ACCESS_TOKEN_CREATED]
})

# Critical events - 2 years
critical_policy = mgr.retention.create_retention_policy({
    'name': 'critical-events-2y',
    'retention_days': 730,
    'severity': EventSeverity.CRITICAL
})

# Apply retention policies (delete old logs)
result = mgr.retention.apply_retention_policies()
# Returns: deleted_events, retained_events

# 6. Export audit logs
recent_events = mgr.events.list_events(limit=100)

# Export to JSON
json_data = mgr.export.export_to_json(recent_events)

# Export to CSV
csv_data = mgr.export.export_to_csv(recent_events)

# Export to syslog format
syslog_messages = mgr.export.export_to_syslog(recent_events)

# Export all events
all_json = mgr.export.export_all_events(format='json')
all_csv = mgr.export.export_all_events(format='csv')
all_syslog = mgr.export.export_all_events(format='syslog')

# 7. SIEM integration
# Configure Splunk
splunk_config = mgr.siem.configure_siem({
    'name': 'Splunk',
    'endpoint_url': 'https://splunk.example.com:8088/services/collector',
    'api_key': 'splunk-hec-token-xxx',
    'event_filters': [EventType.USER_FAILED_LOGIN, EventType.ADMIN_ACTION],
    'batch_size': 50
})

# Configure ELK Stack
elk_config = mgr.siem.configure_siem({
    'name': 'ELK Stack',
    'endpoint_url': 'https://elasticsearch.example.com:9200',
    'api_key': 'elk-api-key-xxx',
    'batch_size': 100
})

# Forward events to SIEM
result = mgr.siem.forward_events_to_siem(splunk_config['config_id'])
# Returns: events_forwarded, batches_sent

# Disable SIEM integration
mgr.siem.disable_siem(elk_config['config_id'])
```

## Event Types

### User Events
- **USER_LOGIN**: User logged in successfully
- **USER_LOGOUT**: User logged out
- **USER_FAILED_LOGIN**: Failed login attempt (security event)
- **USER_CREATED**: New user account created
- **USER_DELETED**: User account deleted
- **USER_UPDATED**: User account updated

### Project Events
- **PROJECT_CREATED**: New project created
- **PROJECT_DELETED**: Project deleted
- **PROJECT_TRANSFERRED**: Project ownership transferred
- **PROJECT_SETTINGS_CHANGED**: Project settings modified

### Access Control Events
- **MEMBER_ADDED**: Member added to project/group
- **MEMBER_REMOVED**: Member removed from project/group
- **MEMBER_ACCESS_CHANGED**: Member access level changed

### Repository Events
- **PUSH**: Code pushed to repository
- **MERGE_REQUEST_CREATED**: New merge request
- **MERGE_REQUEST_MERGED**: Merge request merged
- **MERGE_REQUEST_APPROVED**: Merge request approved
- **BRANCH_CREATED**: New branch created
- **BRANCH_DELETED**: Branch deleted
- **TAG_CREATED**: New tag created
- **TAG_DELETED**: Tag deleted

### Security Events
- **DEPLOY_KEY_ADDED**: SSH deploy key added
- **DEPLOY_KEY_REMOVED**: SSH deploy key removed
- **ACCESS_TOKEN_CREATED**: Access token created
- **ACCESS_TOKEN_REVOKED**: Access token revoked
- **SSH_KEY_ADDED**: SSH key added to user
- **SSH_KEY_REMOVED**: SSH key removed
- **TWO_FACTOR_ENABLED**: 2FA enabled for user
- **TWO_FACTOR_DISABLED**: 2FA disabled for user

### Admin Events
- **ADMIN_ACTION**: Administrative action performed
- **SYSTEM_SETTINGS_CHANGED**: System-wide settings changed
- **GROUP_CREATED**: New group created
- **GROUP_DELETED**: Group deleted

### Compliance Events
- **PROTECTED_BRANCH_CREATED**: Branch protection added
- **PROTECTED_BRANCH_REMOVED**: Branch protection removed
- **APPROVAL_RULE_CREATED**: Approval rule created
- **APPROVAL_RULE_DELETED**: Approval rule deleted

## Severity Levels

### INFO
- Normal operations
- Successful actions
- Non-security-related changes

### WARNING
- Failed login attempts
- Unusual activity patterns
- Deprecated feature usage

### ERROR
- Failed operations
- Permission errors
- Configuration errors

### CRITICAL
- System settings changes
- User deletions
- Security configuration changes
- Admin actions

## Compliance Standards

### SOC 2 (Service Organization Control 2)
- User access logging
- Administrative action tracking
- Security event monitoring
- Change management audit trail

### GDPR (General Data Protection Regulation)
- User data access logging
- Data deletion tracking
- User consent tracking
- Data export audit trail

### HIPAA (Health Insurance Portability and Accountability Act)
- Access control logging
- Audit trail requirements
- Security incident tracking
- Data access monitoring

### PCI DSS (Payment Card Industry Data Security Standard)
- Access control logging
- Security event monitoring
- Log retention (1 year)
- Regular log review

## Log Retention Recommendations

### General Events
- **Retention**: 90 days minimum
- **Events**: INFO level user actions
- **Use Case**: Day-to-day operations

### Security Events
- **Retention**: 1 year minimum
- **Events**: Failed logins, key changes, 2FA events
- **Use Case**: Security incident investigation

### Critical Events
- **Retention**: 2-7 years
- **Events**: Admin actions, system changes, deletions
- **Use Case**: Compliance audits

### Compliance Events
- **Retention**: Per regulatory requirements
- **SOC 2**: 1 year minimum
- **GDPR**: 3 years recommended
- **HIPAA**: 6 years required
- **PCI DSS**: 1 year minimum

## Export Formats

### JSON
```json
[
  {
    "event_id": "event-1",
    "event_type": "user_login",
    "severity": "info",
    "actor_id": 101,
    "target_id": 101,
    "project_id": "myorg/myproject",
    "details": {"method": "password"},
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0",
    "timestamp": "2025-11-06T10:30:00.000Z",
    "event_hash": "abc123..."
  }
]
```

### CSV
```csv
event_id,event_type,severity,actor_id,target_id,project_id,ip_address,timestamp
event-1,user_login,info,101,101,myorg/myproject,192.168.1.100,2025-11-06T10:30:00.000Z
```

### Syslog
```
2025-11-06T10:30:00.000Z gitlab audit[event-1]: type=user_login severity=info actor=101 target=101 ip=192.168.1.100
```

## SIEM Systems

### Splunk
- **Integration**: HTTP Event Collector (HEC)
- **Format**: JSON
- **Batch**: 50-100 events
- **Use Case**: Enterprise SIEM

### ELK Stack (Elasticsearch, Logstash, Kibana)
- **Integration**: Elasticsearch HTTP API
- **Format**: JSON
- **Batch**: 100-1000 events
- **Use Case**: Open-source SIEM

### ArcSight
- **Integration**: CEF (Common Event Format)
- **Format**: Syslog
- **Batch**: 100 events
- **Use Case**: HP/Micro Focus SIEM

### QRadar
- **Integration**: Syslog or LEEF
- **Format**: Syslog/LEEF
- **Batch**: 100 events
- **Use Case**: IBM SIEM

## Best Practices

### Event Logging
1. **Log all security events** - Failed logins, key changes, 2FA events
2. **Include context** - IP address, user agent, details
3. **Use appropriate severity** - INFO for normal, CRITICAL for admin actions
4. **Log administrative actions** - All system changes and user management
5. **Include actor and target** - Track who did what to whom

### Event Querying
1. **Use indexes** - For faster queries on large datasets
2. **Query by date range** - Limit result sets for performance
3. **Use advanced filters** - Combine multiple criteria for precision
4. **Cache frequent queries** - Improve performance for common queries

### Compliance Reporting
1. **Generate regular reports** - Weekly security, monthly compliance
2. **Review critical events** - Daily review of CRITICAL severity
3. **Track failed logins** - Monitor for brute force attacks
4. **Audit access changes** - Weekly review of member additions/removals
5. **Document findings** - Keep records of report reviews

### Log Retention
1. **Follow regulations** - SOC 2 (1 year), HIPAA (6 years), GDPR (3 years)
2. **Separate policies** - Different retention for different event types
3. **Longer for critical** - Keep critical events longer than INFO
4. **Automated cleanup** - Schedule regular retention policy application
5. **Archive old logs** - Move to cold storage before deletion

### SIEM Integration
1. **Filter events** - Only forward relevant events to reduce costs
2. **Use batching** - Batch 50-100 events for efficiency
3. **Monitor forwarding** - Alert on failed SIEM forwards
4. **Test configuration** - Verify events arrive in SIEM
5. **Secure credentials** - Store API keys in secret managers

### Security
1. **Verify integrity** - Regularly check event hashes
2. **Restrict access** - Limit who can query audit logs
3. **Encrypt in transit** - Use HTTPS for SIEM forwarding
4. **Encrypt at rest** - Encrypt audit log storage
5. **Monitor tampering** - Alert on integrity verification failures

## Common Use Cases

### Security Incident Investigation
```python
# Find all events from suspicious IP
suspicious_events = mgr.query.query_by_ip_address('10.0.0.50')

# Find all failed logins in last 24 hours
failed_logins = mgr.query.advanced_query({
    'event_types': [EventType.USER_FAILED_LOGIN],
    'start_date': (datetime.now() - timedelta(days=1)).isoformat()
})

# Generate security report
security_report = mgr.reports.generate_security_report(days=7)
```

### Compliance Audit
```python
# Generate 90-day compliance report
compliance_report = mgr.reports.generate_compliance_report(days=90)

# Export to JSON for auditors
audit_data = mgr.export.export_all_events(format='json')

# Verify event integrity
for event_id in mgr.events.events.keys():
    if not mgr.events.verify_event_integrity(event_id):
        print(f"ALERT: Event {event_id} integrity check failed!")
```

### User Activity Tracking
```python
# Track specific user's activity
user_report = mgr.reports.generate_user_activity_report(
    user_id=102,
    days=30
)

# Find all actions by user
user_events = mgr.query.query_by_actor(102)

# Find all changes to user
target_events = mgr.query.query_by_target(102)
```

### Automated Monitoring
```python
# Daily security report
security_report = mgr.reports.generate_security_report(days=1)

# Alert on critical events
critical = mgr.query.query_by_severity(EventSeverity.CRITICAL)
if len(critical) > 0:
    send_alert(f"{len(critical)} critical events in last 24 hours")

# Forward to SIEM
mgr.siem.forward_events_to_siem('siem-config-1')
```

## Requirements

```
hashlib (standard library)
json (standard library)
datetime (standard library)
```

No external dependencies required.

## Configuration

### Environment Variables
```bash
export GITLAB_URL="https://gitlab.com"
export GITLAB_PROJECT_ID="myorg/myproject"
export SIEM_ENDPOINT="https://splunk.example.com:8088"
export SIEM_API_KEY="your-siem-api-key"
```

### Python Configuration
```python
from audit_logs import AuditLogsManager

mgr = AuditLogsManager(
    project_id='myorg/myproject',
    gitlab_url='https://gitlab.com'
)
```

## Integration Examples

### With CI/CD
```python
# Log deployment events
mgr.events.log_event({
    'event_type': EventType.PUSH,
    'actor_id': ci_user_id,
    'target_id': 'main',
    'details': {'commit_sha': commit_sha, 'pipeline_id': pipeline_id},
    'ip_address': ci_ip,
    'severity': EventSeverity.INFO
})
```

### With Monitoring Systems
```python
# Forward critical events to monitoring
critical_events = mgr.query.query_by_severity(EventSeverity.CRITICAL)
for event in critical_events:
    monitoring_system.send_alert(event)
```

### With Backup Systems
```python
# Daily backup of audit logs
daily_events = mgr.query.query_by_date_range(
    start_date=(datetime.now() - timedelta(days=1)).isoformat(),
    end_date=datetime.now().isoformat()
)
backup_system.store(mgr.export.export_to_json(daily_events))
```

## Troubleshooting

### Issue: Events not appearing in queries
- **Solution**: Check event timestamps and date range filters
- **Solution**: Verify events were logged with `mgr.events.list_events()`

### Issue: SIEM integration failing
- **Solution**: Verify endpoint URL and API key
- **Solution**: Check network connectivity to SIEM endpoint
- **Solution**: Review SIEM logs for error messages

### Issue: Retention policy not deleting logs
- **Solution**: Ensure policy matches event types/severity
- **Solution**: Verify `apply_retention_policies()` is called
- **Solution**: Check retention period is in the past

### Issue: Event integrity verification failing
- **Solution**: Check if events were modified externally
- **Solution**: Verify event hash matches expected format
- **Solution**: Review system logs for tampering indicators

## Performance Considerations

### Large Datasets
- Use date range queries to limit result sets
- Implement pagination for listing events
- Consider archiving old events to separate storage

### SIEM Forwarding
- Use appropriate batch sizes (50-100 events)
- Implement retry logic for failed forwards
- Monitor SIEM API rate limits

### Report Generation
- Generate reports during off-peak hours
- Cache frequently accessed reports
- Use background jobs for long-running reports

## Author

BrillConsulting - Enterprise Cloud Solutions
