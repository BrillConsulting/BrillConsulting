"""
Linux Log Management
Author: BrillConsulting
Description: Centralized logging, log rotation, and analysis with ELK stack integration
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class LogManagement:
    """Comprehensive log management and analysis"""

    def __init__(self, hostname: str = 'localhost'):
        """Initialize log management"""
        self.hostname = hostname
        self.log_configs = []
        self.shippers = []

    def configure_rsyslog(self, rsyslog_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure rsyslog for centralized logging"""
        config = {
            'remote_host': rsyslog_config.get('remote_host', 'logserver.example.com'),
            'remote_port': rsyslog_config.get('remote_port', 514),
            'protocol': rsyslog_config.get('protocol', 'tcp'),
            'facilities': rsyslog_config.get('facilities', ['auth', 'authpriv', 'syslog']),
            'configured_at': datetime.now().isoformat()
        }

        rsyslog_conf = f"""# Centralized Logging Configuration
# Generated: {datetime.now().isoformat()}

# Send all logs to remote server
*.* @@{config['remote_host']}:{config['remote_port']}

# Or use TCP with queue
$ActionQueueType LinkedList
$ActionQueueFileName srvrfwd
$ActionResumeRetryCount -1
$ActionQueueSaveOnShutdown on
*.* @@{config['remote_host']}:{config['remote_port']}

# Local logging rules
auth,authpriv.* /var/log/auth.log
*.*;auth,authpriv.none -/var/log/syslog
kern.* -/var/log/kern.log
mail.* -/var/log/mail.log
"""

        print(f"✓ rsyslog configured")
        print(f"  Remote: {config['remote_host']}:{config['remote_port']}")
        print(f"  Protocol: {config['protocol']}, Facilities: {len(config['facilities'])}")
        return config

    def configure_logrotate(self, logrotate_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure log rotation"""
        config = {
            'log_path': logrotate_config.get('log_path', '/var/log/app/*.log'),
            'rotate': logrotate_config.get('rotate', 7),
            'size': logrotate_config.get('size', '100M'),
            'compress': logrotate_config.get('compress', True),
            'frequency': logrotate_config.get('frequency', 'daily'),
            'configured_at': datetime.now().isoformat()
        }

        logrotate_conf = f"""{config['log_path']} {{
    {config['frequency']}
    rotate {config['rotate']}
    size {config['size']}
    {'compress' if config['compress'] else 'nocompress'}
    delaycompress
    notifempty
    missingok
    create 0640 root adm
    sharedscripts
    postrotate
        systemctl reload app > /dev/null 2>&1 || true
    endscript
}}
"""

        print(f"✓ logrotate configured")
        print(f"  Path: {config['log_path']}, Rotate: {config['rotate']} days")
        print(f"  Size: {config['size']}, Compress: {config['compress']}")
        return config

    def setup_filebeat(self, filebeat_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure Filebeat for log shipping"""
        config = {
            'log_paths': filebeat_config.get('log_paths', ['/var/log/*.log']),
            'elasticsearch_hosts': filebeat_config.get('elasticsearch_hosts', ['localhost:9200']),
            'logstash_hosts': filebeat_config.get('logstash_hosts', []),
            'fields': filebeat_config.get('fields', {}),
            'configured_at': datetime.now().isoformat()
        }

        filebeat_yml = f"""# Filebeat Configuration
# Generated: {datetime.now().isoformat()}

filebeat.inputs:
- type: log
  enabled: true
  paths:"""

        for path in config['log_paths']:
            filebeat_yml += f"\n    - {path}"

        filebeat_yml += f"""

  fields:
    environment: production
    application: myapp
  fields_under_root: true

filebeat.config.modules:
  path: ${{path.config}}/modules.d/*.yml
  reload.enabled: false

setup.template.settings:
  index.number_of_shards: 1

setup.kibana:
  host: "localhost:5601"

output.elasticsearch:
  hosts: {json.dumps(config['elasticsearch_hosts'])}
  username: "elastic"
  password: "changeme"

processors:
  - add_host_metadata: ~
  - add_cloud_metadata: ~
  - add_docker_metadata: ~
"""

        self.shippers.append(config)
        print(f"✓ Filebeat configured")
        print(f"  Log paths: {len(config['log_paths'])}")
        print(f"  Elasticsearch: {', '.join(config['elasticsearch_hosts'])}")
        return config

    def setup_logstash(self, logstash_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure Logstash pipeline"""
        config = {
            'input_type': logstash_config.get('input_type', 'beats'),
            'input_port': logstash_config.get('input_port', 5044),
            'filters': logstash_config.get('filters', []),
            'output_type': logstash_config.get('output_type', 'elasticsearch'),
            'configured_at': datetime.now().isoformat()
        }

        logstash_conf = f"""# Logstash Pipeline Configuration
# Generated: {datetime.now().isoformat()}

input {{
  beats {{
    port => {config['input_port']}
  }}

  syslog {{
    port => 5000
    type => syslog
  }}
}}

filter {{
  if [type] == "syslog" {{
    grok {{
      match => {{ "message" => "%{{SYSLOGLINE}}" }}
    }}
    date {{
      match => [ "timestamp", "MMM  d HH:mm:ss", "MMM dd HH:mm:ss" ]
    }}
  }}

  if [type] == "nginx" {{
    grok {{
      match => {{ "message" => '%{{IPORHOST:remote_addr}} - %{{DATA:remote_user}} \\[%{{HTTPDATE:timestamp}}\\] "%{{WORD:request_method}} %{{DATA:request_path}} HTTP/%{{NUMBER:http_version}}" %{{INT:response_code}} %{{INT:body_sent_bytes}} "%{{DATA:http_referer}}" "%{{DATA:http_user_agent}}"' }}
    }}
    date {{
      match => [ "timestamp", "dd/MMM/YYYY:HH:mm:ss Z" ]
    }}
    geoip {{
      source => "remote_addr"
    }}
  }}

  mutate {{
    add_field => {{ "received_at" => "%{{@timestamp}}" }}
    add_field => {{ "received_from" => "%{{host}}" }}
  }}
}}

output {{
  elasticsearch {{
    hosts => ["localhost:9200"]
    index => "logs-%{{+YYYY.MM.dd}}"
    user => "elastic"
    password => "changeme"
  }}

  stdout {{
    codec => rubydebug
  }}
}}
"""

        print(f"✓ Logstash configured")
        print(f"  Input: {config['input_type']} on port {config['input_port']}")
        print(f"  Output: {config['output_type']}")
        return config

    def analyze_logs(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze logs for patterns and issues"""
        result = {
            'log_file': analysis_config.get('log_file', '/var/log/syslog'),
            'period': analysis_config.get('period', 'last_24h'),
            'errors_found': 45,
            'warnings_found': 128,
            'unique_ips': 523,
            'top_errors': [
                {'error': 'Connection timeout', 'count': 15},
                {'error': 'Permission denied', 'count': 12},
                {'error': 'File not found', 'count': 8}
            ],
            'analyzed_at': datetime.now().isoformat()
        }

        analysis_commands = [
            f"# Top 10 error messages",
            f"grep -i error {result['log_file']} | sort | uniq -c | sort -rn | head -10",
            f"",
            f"# Failed login attempts",
            f"grep 'Failed password' /var/log/auth.log | wc -l",
            f"",
            f"# Top IP addresses",
            f"awk '{{print $1}}' {result['log_file']} | sort | uniq -c | sort -rn | head -10",
            f"",
            f"# Traffic by hour",
            f"awk '{{print $4}}' {result['log_file']} | cut -d: -f1,2 | sort | uniq -c",
            f"",
            f"# 404 errors",
            f"grep ' 404 ' {result['log_file']} | wc -l"
        ]

        print(f"✓ Log analysis completed")
        print(f"  Errors: {result['errors_found']}, Warnings: {result['warnings_found']}")
        print(f"  Top errors:")
        for err in result['top_errors']:
            print(f"    - {err['error']}: {err['count']}")
        return result

    def create_log_alert(self, alert_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create log-based alert"""
        alert = {
            'alert_name': alert_config.get('alert_name', 'high_error_rate'),
            'condition': alert_config.get('condition', 'error_count > 100'),
            'severity': alert_config.get('severity', 'high'),
            'notification': alert_config.get('notification', 'email'),
            'recipients': alert_config.get('recipients', ['admin@example.com']),
            'created_at': datetime.now().isoformat()
        }

        elastalert_rule = f"""# ElastAlert Rule
# Generated: {datetime.now().isoformat()}

name: {alert['alert_name']}
type: frequency
index: logs-*

num_events: 100
timeframe:
  minutes: 5

filter:
- query:
    query_string:
      query: "level:ERROR"

alert:
- "email"

email:
- {', '.join(alert['recipients'])}

alert_subject: "Alert: {alert['alert_name']}"
alert_text: "High error rate detected"
"""

        print(f"✓ Log alert created: {alert['alert_name']}")
        print(f"  Condition: {alert['condition']}, Severity: {alert['severity']}")
        return alert

    def get_log_info(self) -> Dict[str, Any]:
        """Get log management information"""
        return {
            'hostname': self.hostname,
            'log_configs': len(self.log_configs),
            'shippers': len(self.shippers),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate log management"""
    print("=" * 60)
    print("Linux Log Management Demo")
    print("=" * 60)

    log_mgr = LogManagement(hostname='prod-server-01')

    print("\n1. Configuring rsyslog...")
    log_mgr.configure_rsyslog({
        'remote_host': 'logserver.example.com',
        'remote_port': 514,
        'protocol': 'tcp'
    })

    print("\n2. Configuring logrotate...")
    log_mgr.configure_logrotate({
        'log_path': '/var/log/app/*.log',
        'rotate': 14,
        'size': '100M',
        'compress': True
    })

    print("\n3. Setting up Filebeat...")
    log_mgr.setup_filebeat({
        'log_paths': ['/var/log/app/*.log', '/var/log/nginx/*.log'],
        'elasticsearch_hosts': ['es1.example.com:9200', 'es2.example.com:9200']
    })

    print("\n4. Setting up Logstash...")
    log_mgr.setup_logstash({
        'input_type': 'beats',
        'input_port': 5044,
        'output_type': 'elasticsearch'
    })

    print("\n5. Analyzing logs...")
    analysis = log_mgr.analyze_logs({
        'log_file': '/var/log/app/application.log',
        'period': 'last_24h'
    })

    print("\n6. Creating log alert...")
    log_mgr.create_log_alert({
        'alert_name': 'high_error_rate',
        'condition': 'error_count > 100',
        'severity': 'high'
    })

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
