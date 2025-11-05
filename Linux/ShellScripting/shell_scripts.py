"""
Linux Shell Scripting
Advanced Bash scripting and automation toolkit
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class ShellScriptGenerator:
    """Comprehensive shell script generation and automation"""

    def __init__(self):
        """Initialize shell script generator"""
        self.scripts = []

    def generate_backup_script(self, backup_config: Dict[str, Any]) -> str:
        """
        Generate backup script

        Args:
            backup_config: Backup configuration

        Returns:
            Backup script content
        """
        source = backup_config.get('source', '/var/www')
        destination = backup_config.get('destination', '/backup')
        retention_days = backup_config.get('retention_days', 7)
        compress = backup_config.get('compress', True)

        script = f"""#!/bin/bash
# Backup Script
# Generated: {datetime.now().isoformat()}

set -euo pipefail

# Configuration
SOURCE="{source}"
DESTINATION="{destination}"
RETENTION_DAYS={retention_days}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_${{TIMESTAMP}}.tar{'gz' if compress else ''}"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Logging
log_info() {{
    echo -e "${{GREEN}}[INFO]${{NC}} $1"
}}

log_error() {{
    echo -e "${{RED}}[ERROR]${{NC}} $1" >&2
}}

log_warning() {{
    echo -e "${{YELLOW}}[WARNING]${{NC}} $1"
}}

# Create backup directory
log_info "Creating backup directory..."
mkdir -p "$DESTINATION"

# Perform backup
log_info "Starting backup of $SOURCE..."
if [ -d "$SOURCE" ]; then
    tar {'czf' if compress else 'cf'} "$DESTINATION/$BACKUP_NAME" -C "$(dirname $SOURCE)" "$(basename $SOURCE)"
    log_info "Backup created: $BACKUP_NAME"
    log_info "Backup size: $(du -h $DESTINATION/$BACKUP_NAME | cut -f1)"
else
    log_error "Source directory does not exist: $SOURCE"
    exit 1
fi

# Remove old backups
log_info "Removing backups older than $RETENTION_DAYS days..."
find "$DESTINATION" -name "backup_*.tar*" -type f -mtime +$RETENTION_DAYS -delete
log_info "Cleanup completed"

# Summary
BACKUP_COUNT=$(find "$DESTINATION" -name "backup_*.tar*" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$DESTINATION" | cut -f1)
log_info "Backup summary:"
log_info "  Total backups: $BACKUP_COUNT"
log_info "  Total size: $TOTAL_SIZE"

log_info "Backup completed successfully!"
"""

        self.scripts.append({
            'name': 'backup_script',
            'type': 'backup',
            'created_at': datetime.now().isoformat()
        })

        print(f"✓ Backup script generated")
        print(f"  Source: {source}, Destination: {destination}")
        print(f"  Retention: {retention_days} days, Compression: {compress}")
        return script

    def generate_monitoring_script(self, monitoring_config: Dict[str, Any]) -> str:
        """
        Generate system monitoring script

        Args:
            monitoring_config: Monitoring configuration

        Returns:
            Monitoring script content
        """
        cpu_threshold = monitoring_config.get('cpu_threshold', 80)
        memory_threshold = monitoring_config.get('memory_threshold', 80)
        disk_threshold = monitoring_config.get('disk_threshold', 90)
        email = monitoring_config.get('email', 'admin@example.com')

        script = f"""#!/bin/bash
# System Monitoring Script
# Generated: {datetime.now().isoformat()}

set -euo pipefail

# Thresholds
CPU_THRESHOLD={cpu_threshold}
MEMORY_THRESHOLD={memory_threshold}
DISK_THRESHOLD={disk_threshold}
EMAIL="{email}"

# Get system metrics
get_cpu_usage() {{
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\\([0-9.]*\\)%* id.*/\\1/" | awk '{{print 100 - $1}}'
}}

get_memory_usage() {{
    free | grep Mem | awk '{{print ($3/$2) * 100.0}}'
}}

get_disk_usage() {{
    df -h / | awk 'NR==2 {{print $5}}' | sed 's/%//'
}}

# Check thresholds
check_cpu() {{
    CPU_USAGE=$(get_cpu_usage)
    if (( $(echo "$CPU_USAGE > $CPU_THRESHOLD" | bc -l) )); then
        echo "WARNING: CPU usage is ${{CPU_USAGE}}% (threshold: ${{CPU_THRESHOLD}}%)"
        return 1
    fi
    echo "OK: CPU usage is ${{CPU_USAGE}}%"
    return 0
}}

check_memory() {{
    MEMORY_USAGE=$(get_memory_usage)
    if (( $(echo "$MEMORY_USAGE > $MEMORY_THRESHOLD" | bc -l) )); then
        echo "WARNING: Memory usage is ${{MEMORY_USAGE}}% (threshold: ${{MEMORY_THRESHOLD}}%)"
        return 1
    fi
    echo "OK: Memory usage is ${{MEMORY_USAGE}}%"
    return 0
}}

check_disk() {{
    DISK_USAGE=$(get_disk_usage)
    if [ "$DISK_USAGE" -gt "$DISK_THRESHOLD" ]; then
        echo "WARNING: Disk usage is ${{DISK_USAGE}}% (threshold: ${{DISK_THRESHOLD}}%)"
        return 1
    fi
    echo "OK: Disk usage is ${{DISK_USAGE}}%"
    return 0
}}

# Main monitoring
echo "=== System Monitoring Report ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

ALERTS=()

if ! check_cpu; then
    ALERTS+=("CPU")
fi

if ! check_memory; then
    ALERTS+=("Memory")
fi

if ! check_disk; then
    ALERTS+=("Disk")
fi

# Send alert if thresholds exceeded
if [ ${{#ALERTS[@]}} -gt 0 ]; then
    echo ""
    echo "ALERTS TRIGGERED: ${{ALERTS[*]}}"
    # Uncomment to send email
    # echo "System alert: ${{ALERTS[*]}}" | mail -s "System Alert: $(hostname)" "$EMAIL"
    exit 1
fi

echo ""
echo "All checks passed!"
exit 0
"""

        self.scripts.append({
            'name': 'monitoring_script',
            'type': 'monitoring',
            'created_at': datetime.now().isoformat()
        })

        print(f"✓ Monitoring script generated")
        print(f"  CPU threshold: {cpu_threshold}%, Memory: {memory_threshold}%, Disk: {disk_threshold}%")
        return script

    def generate_deployment_script(self, deploy_config: Dict[str, Any]) -> str:
        """
        Generate application deployment script

        Args:
            deploy_config: Deployment configuration

        Returns:
            Deployment script content
        """
        app_name = deploy_config.get('app_name', 'myapp')
        git_repo = deploy_config.get('git_repo', 'https://github.com/user/repo.git')
        branch = deploy_config.get('branch', 'main')
        deploy_path = deploy_config.get('deploy_path', '/opt/myapp')

        script = f"""#!/bin/bash
# Application Deployment Script
# Generated: {datetime.now().isoformat()}

set -euo pipefail

# Configuration
APP_NAME="{app_name}"
GIT_REPO="{git_repo}"
BRANCH="{branch}"
DEPLOY_PATH="{deploy_path}"
BACKUP_PATH="/backup/${{APP_NAME}}"

# Colors
GREEN='\\033[0;32m'
RED='\\033[0;31m'
BLUE='\\033[0;34m'
NC='\\033[0m'

log_step() {{
    echo -e "${{BLUE}}===>${{NC}} $1"
}}

log_success() {{
    echo -e "${{GREEN}}✓${{NC}} $1"
}}

log_error() {{
    echo -e "${{RED}}✗${{NC}} $1" >&2
}}

# Backup current version
backup_current() {{
    log_step "Backing up current version..."
    if [ -d "$DEPLOY_PATH" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        mkdir -p "$BACKUP_PATH"
        tar czf "$BACKUP_PATH/backup_${{TIMESTAMP}}.tar.gz" -C "$(dirname $DEPLOY_PATH)" "$(basename $DEPLOY_PATH)"
        log_success "Backup created: backup_${{TIMESTAMP}}.tar.gz"
    else
        log_step "No existing deployment to backup"
    fi
}}

# Clone or update repository
deploy_code() {{
    log_step "Deploying code from $GIT_REPO ($BRANCH)..."

    if [ -d "$DEPLOY_PATH/.git" ]; then
        log_step "Updating existing repository..."
        cd "$DEPLOY_PATH"
        git fetch origin
        git checkout "$BRANCH"
        git pull origin "$BRANCH"
    else
        log_step "Cloning repository..."
        rm -rf "$DEPLOY_PATH"
        git clone -b "$BRANCH" "$GIT_REPO" "$DEPLOY_PATH"
    fi

    log_success "Code deployed successfully"
}}

# Install dependencies
install_dependencies() {{
    log_step "Installing dependencies..."
    cd "$DEPLOY_PATH"

    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    elif [ -f "package.json" ]; then
        npm install
    fi

    log_success "Dependencies installed"
}}

# Run tests
run_tests() {{
    log_step "Running tests..."
    cd "$DEPLOY_PATH"

    if [ -f "pytest.ini" ]; then
        pytest || {{
            log_error "Tests failed!"
            return 1
        }}
    fi

    log_success "Tests passed"
}}

# Restart service
restart_service() {{
    log_step "Restarting service..."
    systemctl restart "$APP_NAME" || {{
        log_error "Failed to restart service"
        return 1
    }}
    log_success "Service restarted"
}}

# Main deployment process
main() {{
    log_step "Starting deployment of $APP_NAME..."
    echo ""

    backup_current
    deploy_code
    install_dependencies
    run_tests
    restart_service

    echo ""
    log_success "Deployment completed successfully!"
    log_step "Deployed version: $(cd $DEPLOY_PATH && git rev-parse --short HEAD)"
}}

# Error handling
trap 'log_error "Deployment failed!"; exit 1' ERR

main "$@"
"""

        self.scripts.append({
            'name': 'deployment_script',
            'type': 'deployment',
            'created_at': datetime.now().isoformat()
        })

        print(f"✓ Deployment script generated")
        print(f"  App: {app_name}, Branch: {branch}")
        return script

    def generate_log_analyzer_script(self, analyzer_config: Dict[str, Any]) -> str:
        """
        Generate log analysis script

        Args:
            analyzer_config: Analyzer configuration

        Returns:
            Log analyzer script content
        """
        log_file = analyzer_config.get('log_file', '/var/log/apache2/access.log')
        output_file = analyzer_config.get('output_file', '/tmp/log_report.txt')

        script = f"""#!/bin/bash
# Log Analyzer Script
# Generated: {datetime.now().isoformat()}

set -euo pipefail

LOG_FILE="{log_file}"
OUTPUT_FILE="{output_file}"

echo "=== Log Analysis Report ===" > "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "Log file: $LOG_FILE" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Total requests
echo "Total Requests:" >> "$OUTPUT_FILE"
wc -l "$LOG_FILE" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Top 10 IP addresses
echo "Top 10 IP Addresses:" >> "$OUTPUT_FILE"
awk '{{print $1}}' "$LOG_FILE" | sort | uniq -c | sort -nr | head -10 >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Top 10 requested URLs
echo "Top 10 Requested URLs:" >> "$OUTPUT_FILE"
awk '{{print $7}}' "$LOG_FILE" | sort | uniq -c | sort -nr | head -10 >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# HTTP status codes
echo "HTTP Status Codes:" >> "$OUTPUT_FILE"
awk '{{print $9}}' "$LOG_FILE" | sort | uniq -c | sort -nr >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Errors (4xx and 5xx)
echo "Error Requests (4xx, 5xx):" >> "$OUTPUT_FILE"
awk '$9 ~ /^[45]/ {{print $0}}' "$LOG_FILE" | wc -l >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# User agents
echo "Top 10 User Agents:" >> "$OUTPUT_FILE"
awk -F'"' '{{print $6}}' "$LOG_FILE" | sort | uniq -c | sort -nr | head -10 >> "$OUTPUT_FILE"

echo ""
echo "Report generated: $OUTPUT_FILE"
cat "$OUTPUT_FILE"
"""

        self.scripts.append({
            'name': 'log_analyzer_script',
            'type': 'analyzer',
            'created_at': datetime.now().isoformat()
        })

        print(f"✓ Log analyzer script generated")
        print(f"  Log file: {log_file}")
        return script

    def generate_database_backup_script(self, db_config: Dict[str, Any]) -> str:
        """
        Generate database backup script

        Args:
            db_config: Database configuration

        Returns:
            Database backup script content
        """
        db_type = db_config.get('db_type', 'postgresql')
        db_name = db_config.get('db_name', 'mydb')
        backup_path = db_config.get('backup_path', '/backup/db')
        retention_days = db_config.get('retention_days', 7)

        if db_type == 'postgresql':
            backup_cmd = f"pg_dump {db_name}"
        elif db_type == 'mysql':
            backup_cmd = f"mysqldump {db_name}"
        else:
            backup_cmd = f"# Backup command for {db_type}"

        script = f"""#!/bin/bash
# Database Backup Script ({db_type})
# Generated: {datetime.now().isoformat()}

set -euo pipefail

# Configuration
DB_TYPE="{db_type}"
DB_NAME="{db_name}"
BACKUP_PATH="{backup_path}"
RETENTION_DAYS={retention_days}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${{BACKUP_PATH}}/${{DB_NAME}}_${{TIMESTAMP}}.sql.gz"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Perform backup
echo "Starting backup of $DB_NAME database..."
{backup_cmd} | gzip > "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "✓ Backup completed: $BACKUP_FILE"
    echo "  Size: $(du -h $BACKUP_FILE | cut -f1)"
else
    echo "✗ Backup failed!"
    exit 1
fi

# Remove old backups
echo "Removing backups older than $RETENTION_DAYS days..."
find "$BACKUP_PATH" -name "${{DB_NAME}}_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete

# Summary
BACKUP_COUNT=$(find "$BACKUP_PATH" -name "${{DB_NAME}}_*.sql.gz" -type f | wc -l)
echo "Total backups: $BACKUP_COUNT"
echo "Backup completed successfully!"
"""

        self.scripts.append({
            'name': 'database_backup_script',
            'type': 'database_backup',
            'created_at': datetime.now().isoformat()
        })

        print(f"✓ Database backup script generated")
        print(f"  Database: {db_name} ({db_type}), Retention: {retention_days} days")
        return script

    def get_generator_info(self) -> Dict[str, Any]:
        """Get generator information"""
        return {
            'scripts_generated': len(self.scripts),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate shell script generation"""

    print("=" * 60)
    print("Linux Shell Scripting Demo")
    print("=" * 60)

    generator = ShellScriptGenerator()

    print("\n1. Generating backup script...")
    backup_script = generator.generate_backup_script({
        'source': '/var/www/html',
        'destination': '/backup/www',
        'retention_days': 14,
        'compress': True
    })
    print(backup_script[:300] + "...\n")

    print("\n2. Generating monitoring script...")
    monitoring_script = generator.generate_monitoring_script({
        'cpu_threshold': 80,
        'memory_threshold': 85,
        'disk_threshold': 90,
        'email': 'admin@example.com'
    })
    print(monitoring_script[:300] + "...\n")

    print("\n3. Generating deployment script...")
    deployment_script = generator.generate_deployment_script({
        'app_name': 'webapp',
        'git_repo': 'https://github.com/mycompany/webapp.git',
        'branch': 'production',
        'deploy_path': '/opt/webapp'
    })
    print(deployment_script[:300] + "...\n")

    print("\n4. Generating log analyzer script...")
    log_analyzer = generator.generate_log_analyzer_script({
        'log_file': '/var/log/nginx/access.log',
        'output_file': '/tmp/nginx_report.txt'
    })
    print(log_analyzer[:300] + "...\n")

    print("\n5. Generating database backup script...")
    db_backup = generator.generate_database_backup_script({
        'db_type': 'postgresql',
        'db_name': 'production_db',
        'backup_path': '/backup/postgresql',
        'retention_days': 30
    })
    print(db_backup[:300] + "...\n")

    print("\n6. Generator summary:")
    info = generator.get_generator_info()
    print(f"  Scripts generated: {info['scripts_generated']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
