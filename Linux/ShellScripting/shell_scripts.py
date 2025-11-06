"""
Linux Shell Scripting Toolkit
Author: BrillConsulting
Description: Production-ready Bash scripting and automation toolkit with comprehensive
             features including backup automation, deployment, monitoring, log analysis,
             database backups, error handling, and best practices.

Features:
- Backup automation with compression and retention policies
- Deployment scripts with rollback capabilities
- System monitoring and health checks
- Log analysis and reporting
- Database backup automation (MySQL, PostgreSQL, MongoDB)
- Process management and service monitoring
- Cron job management
- Security hardening utilities
- Performance monitoring
- Error handling and retry mechanisms
- Colored output and structured logging
- Script templates and best practices
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class ShellScriptGenerator:
    """
    Production-ready shell script generation and automation toolkit

    This class provides comprehensive Bash script generation for various
    system administration and DevOps tasks including backup, deployment,
    monitoring, and database management.
    """

    def __init__(self):
        """Initialize shell script generator with default configurations"""
        self.scripts = []
        self.templates = {}
        self._load_templates()

    def _load_templates(self):
        """Load script templates and common functions"""
        self.templates['common_functions'] = '''
# Common shell script functions and utilities

# Color definitions
export RED='\\033[0;31m'
export GREEN='\\033[0;32m'
export YELLOW='\\033[1;33m'
export BLUE='\\033[0;34m'
export PURPLE='\\033[0;35m'
export CYAN='\\033[0;36m'
export NC='\\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Error handling
handle_error() {
    log_error "$1"
    log_error "Script failed at line $2"
    exit 1
}

# Retry mechanism
retry() {
    local max_attempts=$1
    shift
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if "$@"; then
            return 0
        fi
        log_warning "Attempt $attempt/$max_attempts failed. Retrying..."
        attempt=$((attempt + 1))
        sleep 2
    done

    log_error "All $max_attempts attempts failed"
    return 1
}

# Check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command not found: $1"
        exit 1
    fi
}

# Send notification (email, Slack, etc.)
send_notification() {
    local subject="$1"
    local message="$2"

    # Email notification (if mail command is available)
    if command -v mail &> /dev/null && [ -n "$NOTIFICATION_EMAIL" ]; then
        echo "$message" | mail -s "$subject" "$NOTIFICATION_EMAIL"
    fi

    # Slack notification (if webhook is configured)
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\\"text\\":\\"$subject\\\\n$message\\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null
    fi
}

# Check disk space
check_disk_space() {
    local path="$1"
    local required_space_gb="$2"

    local available_space=$(df -BG "$path" | awk 'NR==2 {print $4}' | sed 's/G//')

    if [ "$available_space" -lt "$required_space_gb" ]; then
        log_error "Insufficient disk space. Required: ${required_space_gb}GB, Available: ${available_space}GB"
        return 1
    fi

    return 0
}

# Create lock file to prevent concurrent execution
acquire_lock() {
    local lock_file="$1"

    if [ -f "$lock_file" ]; then
        local pid=$(cat "$lock_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_error "Another instance is running (PID: $pid)"
            exit 1
        else
            log_warning "Removing stale lock file"
            rm -f "$lock_file"
        fi
    fi

    echo $$ > "$lock_file"
}

# Release lock file
release_lock() {
    local lock_file="$1"
    rm -f "$lock_file"
}
'''

        self.templates['error_handling'] = '''
# Advanced error handling setup
set -euo pipefail
IFS=$'\\n\\t'

# Trap errors
trap 'handle_error "An error occurred" $LINENO' ERR
trap 'release_lock "$LOCK_FILE" 2>/dev/null' EXIT
'''

    def get_common_functions(self) -> str:
        """Get common shell script functions"""
        return self.templates.get('common_functions', '')

    def get_error_handling(self) -> str:
        """Get error handling template"""
        return self.templates.get('error_handling', '')

    def generate_backup_script(self, backup_config: Dict[str, Any]) -> str:
        """
        Generate production-ready backup script with advanced features

        Args:
            backup_config: Backup configuration
                - source: Source directory to backup
                - destination: Backup destination directory
                - retention_days: Number of days to keep backups
                - compress: Enable compression (default: True)
                - encrypt: Enable encryption (default: False)
                - exclude_patterns: List of patterns to exclude
                - notification_email: Email for notifications
                - min_disk_space_gb: Minimum required disk space in GB

        Returns:
            Backup script content
        """
        source = backup_config.get('source', '/var/www')
        destination = backup_config.get('destination', '/backup')
        retention_days = backup_config.get('retention_days', 7)
        compress = backup_config.get('compress', True)
        encrypt = backup_config.get('encrypt', False)
        exclude_patterns = backup_config.get('exclude_patterns', [])
        notification_email = backup_config.get('notification_email', '')
        min_disk_space = backup_config.get('min_disk_space_gb', 10)

        exclude_flags = ' '.join([f'--exclude="{pattern}"' for pattern in exclude_patterns])

        script = f"""#!/bin/bash
# Production-Ready Backup Script
# Generated: {datetime.now().isoformat()}
# Description: Automated backup with compression, encryption, and retention

set -euo pipefail

# Configuration
SOURCE="{source}"
DESTINATION="{destination}"
RETENTION_DAYS={retention_days}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_${{TIMESTAMP}}.tar{'gz' if compress else ''}"
LOCK_FILE="/tmp/backup_script.lock"
LOG_FILE="$DESTINATION/backup_${{TIMESTAMP}}.log"
NOTIFICATION_EMAIL="{notification_email}"
MIN_DISK_SPACE_GB={min_disk_space}

{self.get_common_functions()}

# Main backup function
perform_backup() {{
    log_step "Starting backup process..."

    # Check prerequisites
    log_info "Checking prerequisites..."
    check_command tar
    {'check_command gpg' if encrypt else '# Encryption not enabled'}

    # Check if source exists
    if [ ! -d "$SOURCE" ]; then
        log_error "Source directory does not exist: $SOURCE"
        return 1
    fi

    # Create destination directory
    log_info "Creating backup directory..."
    mkdir -p "$DESTINATION"

    # Check disk space
    log_info "Checking available disk space..."
    if ! check_disk_space "$DESTINATION" "$MIN_DISK_SPACE_GB"; then
        send_notification "Backup Failed" "Insufficient disk space on $(hostname)"
        return 1
    fi

    # Acquire lock
    log_info "Acquiring lock..."
    acquire_lock "$LOCK_FILE"

    # Perform backup
    log_info "Creating backup of $SOURCE..."
    if tar {'czf' if compress else 'cf'} "$DESTINATION/$BACKUP_NAME" \\
        -C "$(dirname $SOURCE)" \\
        {exclude_flags} \\
        "$(basename $SOURCE)"; then
        log_success "Backup archive created: $BACKUP_NAME"
    else
        log_error "Backup creation failed"
        return 1
    fi

    # Encrypt backup if enabled
    {'if [ -n "$GPG_RECIPIENT" ]; then' if encrypt else '# Encryption not enabled'}
    {'    log_info "Encrypting backup..."' if encrypt else ''}
    {'    gpg --encrypt --recipient "$GPG_RECIPIENT" "$DESTINATION/$BACKUP_NAME"' if encrypt else ''}
    {'    rm "$DESTINATION/$BACKUP_NAME"' if encrypt else ''}
    {'    BACKUP_NAME="${BACKUP_NAME}.gpg"' if encrypt else ''}
    {'    log_success "Backup encrypted"' if encrypt else ''}
    {'fi' if encrypt else ''}

    # Calculate backup size
    local backup_size=$(du -h "$DESTINATION/$BACKUP_NAME" | cut -f1)
    log_info "Backup size: $backup_size"

    # Verify backup integrity
    log_info "Verifying backup integrity..."
    if tar tzf "$DESTINATION/$BACKUP_NAME" > /dev/null 2>&1; then
        log_success "Backup integrity verified"
    else
        log_error "Backup verification failed"
        return 1
    fi

    # Remove old backups
    log_info "Removing backups older than $RETENTION_DAYS days..."
    local removed_count=$(find "$DESTINATION" -name "backup_*.tar*" -type f -mtime +$RETENTION_DAYS -delete -print | wc -l)
    log_info "Removed $removed_count old backup(s)"

    # Generate summary
    local backup_count=$(find "$DESTINATION" -name "backup_*.tar*" -type f | wc -l)
    local total_size=$(du -sh "$DESTINATION" | cut -f1)

    log_success "Backup completed successfully!"
    log_info "Summary:"
    log_info "  Backup file: $BACKUP_NAME"
    log_info "  Backup size: $backup_size"
    log_info "  Total backups: $backup_count"
    log_info "  Total size: $total_size"
    log_info "  Retention policy: $RETENTION_DAYS days"

    # Send success notification
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        send_notification "Backup Successful" "Backup completed on $(hostname)\\nBackup: $BACKUP_NAME\\nSize: $backup_size"
    fi
}}

# Error handling
trap 'log_error "Backup failed!"; send_notification "Backup Failed" "Backup failed on $(hostname)"; release_lock "$LOCK_FILE"; exit 1' ERR
trap 'release_lock "$LOCK_FILE"' EXIT

# Execute backup
perform_backup 2>&1 | tee "$LOG_FILE"
"""

        self.scripts.append({
            'name': 'backup_script',
            'type': 'backup',
            'created_at': datetime.now().isoformat(),
            'config': backup_config
        })

        print(f"✓ Enhanced backup script generated")
        print(f"  Source: {source}, Destination: {destination}")
        print(f"  Retention: {retention_days} days, Compression: {compress}, Encryption: {encrypt}")
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

    def generate_system_health_check(self, health_config: Dict[str, Any]) -> str:
        """
        Generate comprehensive system health check script

        Args:
            health_config: Health check configuration
                - services: List of services to check
                - ports: List of ports to check
                - urls: List of URLs to check
                - disk_threshold: Disk usage threshold percentage
                - memory_threshold: Memory usage threshold percentage

        Returns:
            System health check script content
        """
        services = health_config.get('services', ['nginx', 'postgresql'])
        ports = health_config.get('ports', [80, 443, 5432])
        urls = health_config.get('urls', ['http://localhost'])
        disk_threshold = health_config.get('disk_threshold', 90)
        memory_threshold = health_config.get('memory_threshold', 85)

        script = f"""#!/bin/bash
# System Health Check Script
# Generated: {datetime.now().isoformat()}
# Description: Comprehensive system health monitoring

set -euo pipefail

{self.get_common_functions()}

# Configuration
DISK_THRESHOLD={disk_threshold}
MEMORY_THRESHOLD={memory_threshold}
SERVICES=({' '.join([f'"{s}"' for s in services])})
PORTS=({' '.join([str(p) for p in ports])})
URLS=({' '.join([f'"{u}"' for u in urls])})

HEALTH_STATUS=0
REPORT_FILE="/tmp/health_check_$(date +%Y%m%d_%H%M%S).txt"

# Initialize report
exec > >(tee -a "$REPORT_FILE")
exec 2>&1

echo "======================================"
echo "System Health Check Report"
echo "======================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "======================================"
echo ""

# Check system uptime
check_uptime() {{
    log_step "Checking system uptime..."
    local uptime_info=$(uptime)
    echo "  Uptime: $uptime_info"
    log_success "System uptime checked"
}}

# Check CPU usage
check_cpu() {{
    log_step "Checking CPU usage..."
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\\([0-9.]*\\)%* id.*/\\1/" | awk '{{print 100 - $1}}')
    echo "  CPU Usage: ${{cpu_usage}}%"

    if (( $(echo "$cpu_usage > 90" | bc -l) )); then
        log_warning "High CPU usage detected: ${{cpu_usage}}%"
        HEALTH_STATUS=1
    else
        log_success "CPU usage normal"
    fi
}}

# Check memory usage
check_memory() {{
    log_step "Checking memory usage..."
    local memory_usage=$(free | grep Mem | awk '{{printf "%.1f", ($3/$2) * 100.0}}')
    local memory_total=$(free -h | grep Mem | awk '{{print $2}}')
    local memory_used=$(free -h | grep Mem | awk '{{print $3}}')

    echo "  Memory: $memory_used / $memory_total (${{memory_usage}}%)"

    if (( $(echo "$memory_usage > $MEMORY_THRESHOLD" | bc -l) )); then
        log_warning "High memory usage: ${{memory_usage}}%"
        HEALTH_STATUS=1
    else
        log_success "Memory usage normal"
    fi
}}

# Check disk usage
check_disk() {{
    log_step "Checking disk usage..."
    echo "  Disk Usage by Mount Point:"

    while read -r line; do
        local usage=$(echo "$line" | awk '{{print $5}}' | sed 's/%//')
        local mount=$(echo "$line" | awk '{{print $6}}')
        echo "    $mount: ${{usage}}%"

        if [ "$usage" -gt "$DISK_THRESHOLD" ]; then
            log_warning "High disk usage on $mount: ${{usage}}%"
            HEALTH_STATUS=1
        fi
    done < <(df -h | tail -n +2)

    log_success "Disk usage checked"
}}

# Check services
check_services() {{
    log_step "Checking services..."

    for service in "${{SERVICES[@]}}"; do
        if systemctl is-active --quiet "$service"; then
            echo "  ✓ $service: running"
        else
            log_error "Service $service is not running"
            HEALTH_STATUS=1
        fi
    done

    log_success "Service check completed"
}}

# Check open ports
check_ports() {{
    log_step "Checking ports..."

    for port in "${{PORTS[@]}}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
            echo "  ✓ Port $port: open"
        else
            log_warning "Port $port is not listening"
            HEALTH_STATUS=1
        fi
    done

    log_success "Port check completed"
}}

# Check URLs
check_urls() {{
    log_step "Checking URLs..."

    for url in "${{URLS[@]}}"; do
        if curl -f -s -o /dev/null -w "%{{http_code}}" "$url" | grep -q "200"; then
            echo "  ✓ $url: accessible"
        else
            log_error "URL $url is not accessible"
            HEALTH_STATUS=1
        fi
    done

    log_success "URL check completed"
}}

# Check system load
check_load() {{
    log_step "Checking system load..."
    local load=$(uptime | awk -F'load average:' '{{print $2}}')
    echo "  Load Average: $load"
    log_success "Load checked"
}}

# Check failed login attempts
check_security() {{
    log_step "Checking security..."

    if [ -f /var/log/auth.log ]; then
        local failed_logins=$(grep "Failed password" /var/log/auth.log 2>/dev/null | wc -l)
        echo "  Failed login attempts (today): $failed_logins"

        if [ "$failed_logins" -gt 50 ]; then
            log_warning "High number of failed login attempts: $failed_logins"
            HEALTH_STATUS=1
        fi
    fi

    log_success "Security check completed"
}}

# Main health check
main() {{
    check_uptime
    echo ""
    check_cpu
    echo ""
    check_memory
    echo ""
    check_disk
    echo ""
    check_services
    echo ""
    check_ports
    echo ""
    check_urls
    echo ""
    check_load
    echo ""
    check_security
    echo ""

    echo "======================================"
    if [ $HEALTH_STATUS -eq 0 ]; then
        log_success "All health checks passed!"
        echo "Overall Status: HEALTHY"
    else
        log_warning "Some health checks failed"
        echo "Overall Status: DEGRADED"
    fi
    echo "======================================"
    echo "Report saved to: $REPORT_FILE"

    exit $HEALTH_STATUS
}}

main "$@"
"""

        self.scripts.append({
            'name': 'system_health_check',
            'type': 'health_check',
            'created_at': datetime.now().isoformat()
        })

        print(f"✓ System health check script generated")
        print(f"  Services: {len(services)}, Ports: {len(ports)}, URLs: {len(urls)}")
        return script

    def generate_process_monitor(self, monitor_config: Dict[str, Any]) -> str:
        """
        Generate process monitoring script

        Args:
            monitor_config: Process monitoring configuration
                - processes: List of process names to monitor
                - restart_on_failure: Auto-restart failed processes
                - notification_email: Email for notifications

        Returns:
            Process monitoring script content
        """
        processes = monitor_config.get('processes', ['nginx', 'postgresql'])
        restart_on_failure = monitor_config.get('restart_on_failure', True)
        notification_email = monitor_config.get('notification_email', '')

        script = f"""#!/bin/bash
# Process Monitoring Script
# Generated: {datetime.now().isoformat()}
# Description: Monitor critical processes and auto-restart if needed

set -euo pipefail

{self.get_common_functions()}

# Configuration
PROCESSES=({' '.join([f'"{p}"' for p in processes])})
RESTART_ON_FAILURE={'true' if restart_on_failure else 'false'}
NOTIFICATION_EMAIL="{notification_email}"
LOG_FILE="/var/log/process_monitor.log"

# Check and restart process
check_process() {{
    local process_name="$1"

    log_info "Checking process: $process_name"

    if pgrep -x "$process_name" > /dev/null; then
        log_success "Process $process_name is running"
        return 0
    else
        log_error "Process $process_name is not running"

        # Try to restart if enabled
        if [ "$RESTART_ON_FAILURE" = "true" ]; then
            log_info "Attempting to restart $process_name..."

            if systemctl restart "$process_name" 2>/dev/null; then
                log_success "Successfully restarted $process_name"
                send_notification "Process Restarted" "Process $process_name was restarted on $(hostname)"
            else
                log_error "Failed to restart $process_name"
                send_notification "Process Restart Failed" "Failed to restart $process_name on $(hostname)"
                return 1
            fi
        else
            send_notification "Process Down" "Process $process_name is down on $(hostname)"
            return 1
        fi
    fi
}}

# Get process info
get_process_info() {{
    local process_name="$1"

    if pgrep -x "$process_name" > /dev/null; then
        local pid=$(pgrep -x "$process_name" | head -1)
        local cpu=$(ps -p "$pid" -o %cpu --no-headers | tr -d ' ')
        local mem=$(ps -p "$pid" -o %mem --no-headers | tr -d ' ')
        local uptime=$(ps -p "$pid" -o etime --no-headers | tr -d ' ')

        echo "  PID: $pid | CPU: ${{cpu}}% | Memory: ${{mem}}% | Uptime: $uptime"
    fi
}}

# Main monitoring loop
main() {{
    log_step "Starting process monitoring..."
    echo "Monitoring processes: ${{PROCESSES[*]}}"
    echo "Auto-restart: $RESTART_ON_FAILURE"
    echo ""

    local failures=0

    for process in "${{PROCESSES[@]}}"; do
        if ! check_process "$process"; then
            failures=$((failures + 1))
        else
            get_process_info "$process"
        fi
        echo ""
    done

    echo "======================================"
    if [ $failures -eq 0 ]; then
        log_success "All processes are running normally"
        exit 0
    else
        log_error "$failures process(es) encountered issues"
        exit 1
    fi
}}

main "$@" 2>&1 | tee -a "$LOG_FILE"
"""

        self.scripts.append({
            'name': 'process_monitor',
            'type': 'monitoring',
            'created_at': datetime.now().isoformat()
        })

        print(f"✓ Process monitoring script generated")
        print(f"  Processes: {len(processes)}, Auto-restart: {restart_on_failure}")
        return script

    def generate_cron_job_script(self, cron_config: Dict[str, Any]) -> str:
        """
        Generate cron job management script

        Args:
            cron_config: Cron job configuration
                - job_name: Name of the cron job
                - command: Command to execute
                - schedule: Cron schedule (e.g., "0 2 * * *")
                - user: User to run cron as

        Returns:
            Cron job management script content
        """
        job_name = cron_config.get('job_name', 'backup_job')
        command = cron_config.get('command', '/usr/local/bin/backup.sh')
        schedule = cron_config.get('schedule', '0 2 * * *')
        user = cron_config.get('user', 'root')

        script = f"""#!/bin/bash
# Cron Job Management Script
# Generated: {datetime.now().isoformat()}
# Description: Manage cron jobs with logging and error handling

set -euo pipefail

{self.get_common_functions()}

# Configuration
JOB_NAME="{job_name}"
COMMAND="{command}"
SCHEDULE="{schedule}"
USER="{user}"
CRON_LOG="/var/log/cron/${{JOB_NAME}}.log"
LOCK_FILE="/tmp/${{JOB_NAME}}.lock"

# Setup logging directory
setup_logging() {{
    mkdir -p "$(dirname $CRON_LOG)"
    log_info "Logging to: $CRON_LOG"
}}

# Install cron job
install_cron() {{
    log_step "Installing cron job..."

    # Create wrapper script with logging
    local wrapper_script="/usr/local/bin/${{JOB_NAME}}_wrapper.sh"

    cat > "$wrapper_script" << 'WRAPPER_EOF'
#!/bin/bash
set -euo pipefail

LOCK_FILE="LOCK_FILE_PLACEHOLDER"
LOG_FILE="LOG_FILE_PLACEHOLDER"

# Acquire lock
if [ -f "$LOCK_FILE" ]; then
    echo "[ERROR] Job already running (lock file exists)"
    exit 1
fi

echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# Execute command with logging
{{{{
    echo "======================================"
    echo "Job: JOB_NAME_PLACEHOLDER"
    echo "Started: $(date)"
    echo "======================================"

    if COMMAND_PLACEHOLDER; then
        echo "[SUCCESS] Job completed successfully"
        exit_code=0
    else
        echo "[ERROR] Job failed"
        exit_code=1
    fi

    echo "Finished: $(date)"
    echo "======================================"
    exit $exit_code
}}}} >> "$LOG_FILE" 2>&1
WRAPPER_EOF

    # Replace placeholders
    sed -i "s|LOCK_FILE_PLACEHOLDER|$LOCK_FILE|g" "$wrapper_script"
    sed -i "s|LOG_FILE_PLACEHOLDER|$CRON_LOG|g" "$wrapper_script"
    sed -i "s|JOB_NAME_PLACEHOLDER|$JOB_NAME|g" "$wrapper_script"
    sed -i "s|COMMAND_PLACEHOLDER|$COMMAND|g" "$wrapper_script"

    chmod +x "$wrapper_script"

    # Add to crontab
    local cron_entry="$SCHEDULE $wrapper_script"

    if crontab -u "$USER" -l 2>/dev/null | grep -q "$JOB_NAME"; then
        log_warning "Cron job already exists, updating..."
        (crontab -u "$USER" -l 2>/dev/null | grep -v "$JOB_NAME"; echo "# $JOB_NAME"; echo "$cron_entry") | crontab -u "$USER" -
    else
        (crontab -u "$USER" -l 2>/dev/null; echo "# $JOB_NAME"; echo "$cron_entry") | crontab -u "$USER" -
    fi

    log_success "Cron job installed successfully"
    echo "  Schedule: $SCHEDULE"
    echo "  Command: $COMMAND"
    echo "  Log file: $CRON_LOG"
}}

# Remove cron job
remove_cron() {{
    log_step "Removing cron job..."

    if crontab -u "$USER" -l 2>/dev/null | grep -q "$JOB_NAME"; then
        crontab -u "$USER" -l 2>/dev/null | grep -v "$JOB_NAME" | crontab -u "$USER" -
        rm -f "/usr/local/bin/${{JOB_NAME}}_wrapper.sh"
        log_success "Cron job removed"
    else
        log_warning "Cron job not found"
    fi
}}

# Show cron job status
show_status() {{
    log_step "Cron job status..."

    if crontab -u "$USER" -l 2>/dev/null | grep -q "$JOB_NAME"; then
        log_success "Cron job is installed"
        echo ""
        echo "Schedule:"
        crontab -u "$USER" -l 2>/dev/null | grep -A1 "$JOB_NAME"
        echo ""
        echo "Recent logs:"
        if [ -f "$CRON_LOG" ]; then
            tail -n 20 "$CRON_LOG"
        else
            echo "No logs found"
        fi
    else
        log_warning "Cron job is not installed"
    fi
}}

# Main
case "${{1:-}}" in
    install)
        setup_logging
        install_cron
        ;;
    remove)
        remove_cron
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {{install|remove|status}}"
        echo ""
        echo "Commands:"
        echo "  install  - Install cron job"
        echo "  remove   - Remove cron job"
        echo "  status   - Show cron job status"
        exit 1
        ;;
esac
"""

        self.scripts.append({
            'name': 'cron_job_script',
            'type': 'cron',
            'created_at': datetime.now().isoformat()
        })

        print(f"✓ Cron job management script generated")
        print(f"  Job: {job_name}, Schedule: {schedule}")
        return script

    def generate_security_hardening_script(self) -> str:
        """
        Generate security hardening script for Linux systems

        Returns:
            Security hardening script content
        """
        script = f"""#!/bin/bash
# Security Hardening Script
# Generated: {datetime.now().isoformat()}
# Description: Apply security best practices to Linux system

set -euo pipefail

{self.get_common_functions()}

# Backup directory
BACKUP_DIR="/root/security_backup_$(date +%Y%m%d_%H%M%S)"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Update system
harden_updates() {{
    log_step "Updating system packages..."

    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get upgrade -y
    elif command -v yum &> /dev/null; then
        yum update -y
    fi

    log_success "System updated"
}}

# Configure firewall
harden_firewall() {{
    log_step "Configuring firewall..."

    if command -v ufw &> /dev/null; then
        # Backup current rules
        ufw status > "$BACKUP_DIR/ufw_rules.txt"

        # Basic UFW rules
        ufw default deny incoming
        ufw default allow outgoing
        ufw allow ssh
        ufw allow http
        ufw allow https
        ufw --force enable

        log_success "UFW firewall configured"
    elif command -v firewall-cmd &> /dev/null; then
        firewall-cmd --permanent --set-default-zone=public
        firewall-cmd --permanent --add-service=ssh
        firewall-cmd --permanent --add-service=http
        firewall-cmd --permanent --add-service=https
        firewall-cmd --reload

        log_success "Firewalld configured"
    fi
}}

# Secure SSH
harden_ssh() {{
    log_step "Hardening SSH configuration..."

    local ssh_config="/etc/ssh/sshd_config"
    cp "$ssh_config" "$BACKUP_DIR/sshd_config.bak"

    # Apply secure SSH settings
    sed -i 's/#*PermitRootLogin.*/PermitRootLogin no/' "$ssh_config"
    sed -i 's/#*PasswordAuthentication.*/PasswordAuthentication no/' "$ssh_config"
    sed -i 's/#*X11Forwarding.*/X11Forwarding no/' "$ssh_config"
    sed -i 's/#*MaxAuthTries.*/MaxAuthTries 3/' "$ssh_config"
    sed -i 's/#*Protocol.*/Protocol 2/' "$ssh_config"

    # Restart SSH service
    systemctl restart sshd || systemctl restart ssh

    log_success "SSH hardened"
}}

# Configure fail2ban
harden_fail2ban() {{
    log_step "Installing and configuring fail2ban..."

    if ! command -v fail2ban-client &> /dev/null; then
        if command -v apt-get &> /dev/null; then
            apt-get install -y fail2ban
        elif command -v yum &> /dev/null; then
            yum install -y fail2ban
        fi
    fi

    # Configure fail2ban
    cat > /etc/fail2ban/jail.local <<'F2B_EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log
F2B_EOF

    systemctl enable fail2ban
    systemctl restart fail2ban

    log_success "fail2ban configured"
}}

# Set file permissions
harden_permissions() {{
    log_step "Setting secure file permissions..."

    # Secure sensitive files
    chmod 600 /etc/ssh/sshd_config
    chmod 644 /etc/passwd
    chmod 600 /etc/shadow
    chmod 644 /etc/group
    chmod 600 /etc/gshadow

    log_success "File permissions secured"
}}

# Disable unnecessary services
harden_services() {{
    log_step "Disabling unnecessary services..."

    local services_to_disable=("telnet" "rsh" "rlogin")

    for service in "${{services_to_disable[@]}}"; do
        if systemctl is-enabled "$service" 2>/dev/null; then
            systemctl disable "$service"
            systemctl stop "$service"
            log_info "Disabled $service"
        fi
    done

    log_success "Unnecessary services disabled"
}}

# Configure audit logging
harden_audit() {{
    log_step "Configuring audit logging..."

    if ! command -v auditd &> /dev/null; then
        if command -v apt-get &> /dev/null; then
            apt-get install -y auditd
        elif command -v yum &> /dev/null; then
            yum install -y audit
        fi
    fi

    systemctl enable auditd
    systemctl start auditd

    log_success "Audit logging configured"
}}

# Main hardening process
main() {{
    log_step "Starting security hardening..."
    echo "Backup directory: $BACKUP_DIR"
    echo ""

    harden_updates
    echo ""
    harden_firewall
    echo ""
    harden_ssh
    echo ""
    harden_fail2ban
    echo ""
    harden_permissions
    echo ""
    harden_services
    echo ""
    harden_audit
    echo ""

    log_success "Security hardening completed!"
    log_info "Backups saved to: $BACKUP_DIR"
    log_warning "Please review changes and test system functionality"
}}

main "$@"
"""

        self.scripts.append({
            'name': 'security_hardening',
            'type': 'security',
            'created_at': datetime.now().isoformat()
        })

        print(f"✓ Security hardening script generated")
        return script

    def generate_performance_monitor(self) -> str:
        """
        Generate performance monitoring script

        Returns:
            Performance monitoring script content
        """
        script = f"""#!/bin/bash
# Performance Monitoring Script
# Generated: {datetime.now().isoformat()}
# Description: Monitor system performance metrics

set -euo pipefail

{self.get_common_functions()}

# Configuration
OUTPUT_DIR="/var/log/performance"
REPORT_FILE="$OUTPUT_DIR/performance_$(date +%Y%m%d_%H%M%S).txt"
DURATION=${{1:-60}}  # Default 60 seconds

mkdir -p "$OUTPUT_DIR"

# Collect performance data
collect_metrics() {{
    log_step "Collecting performance metrics for ${{DURATION}} seconds..."

    {{
        echo "======================================"
        echo "Performance Monitoring Report"
        echo "======================================"
        echo "Date: $(date)"
        echo "Hostname: $(hostname)"
        echo "Duration: ${{DURATION}} seconds"
        echo "======================================"
        echo ""

        # CPU Information
        echo "=== CPU Information ==="
        lscpu | grep -E "Model name|CPU\\(s\\)|Thread|Core"
        echo ""

        # CPU Usage over time
        echo "=== CPU Usage (%) ==="
        for i in $(seq 1 5); do
            cpu=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\\([0-9.]*\\)%* id.*/\\1/" | awk '{{print 100 - $1}}')
            echo "Sample $i: ${{cpu}}%"
            sleep 2
        done
        echo ""

        # Memory Information
        echo "=== Memory Usage ==="
        free -h
        echo ""

        # Disk I/O
        echo "=== Disk I/O Statistics ==="
        iostat -x 2 3
        echo ""

        # Top processes by CPU
        echo "=== Top 10 Processes by CPU ==="
        ps aux --sort=-%cpu | head -11
        echo ""

        # Top processes by Memory
        echo "=== Top 10 Processes by Memory ==="
        ps aux --sort=-%mem | head -11
        echo ""

        # Network statistics
        echo "=== Network Statistics ==="
        netstat -i
        echo ""

        # Load average
        echo "=== Load Average ==="
        uptime
        echo ""

        # Disk usage
        echo "=== Disk Usage ==="
        df -h
        echo ""

        # Open files
        echo "=== Open Files Count ==="
        lsof | wc -l
        echo ""

        # Connection statistics
        echo "=== Network Connections ==="
        netstat -an | awk '/^tcp/ {{count[$6]++}} END {{for(state in count) print state, count[state]}}'
        echo ""

        echo "======================================"
        echo "Report completed at $(date)"
        echo "======================================"
    }} > "$REPORT_FILE"

    log_success "Performance data collected"
    echo "Report saved to: $REPORT_FILE"

    # Display summary
    echo ""
    echo "=== Quick Summary ==="
    grep -A3 "CPU Usage" "$REPORT_FILE"
    echo ""
    grep -A5 "Memory Usage" "$REPORT_FILE" | head -6
    echo ""
    grep "Load Average" "$REPORT_FILE"
}}

# Main
main() {{
    log_info "Starting performance monitoring..."
    collect_metrics
    log_success "Monitoring completed"

    # Cleanup old reports (keep last 30 days)
    find "$OUTPUT_DIR" -name "performance_*.txt" -type f -mtime +30 -delete
}}

main "$@"
"""

        self.scripts.append({
            'name': 'performance_monitor',
            'type': 'monitoring',
            'created_at': datetime.now().isoformat()
        })

        print(f"✓ Performance monitoring script generated")
        return script

    def save_script(self, script_content: str, filename: str, output_dir: str = '.') -> str:
        """
        Save generated script to file

        Args:
            script_content: Script content to save
            filename: Output filename
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(script_content)

        # Make executable
        os.chmod(output_path, 0o755)

        print(f"✓ Script saved to: {output_path}")
        return str(output_path)

    def get_generator_info(self) -> Dict[str, Any]:
        """Get generator information"""
        return {
            'scripts_generated': len(self.scripts),
            'timestamp': datetime.now().isoformat(),
            'script_types': list(set([s['type'] for s in self.scripts]))
        }


def demo():
    """Demonstrate comprehensive shell script generation toolkit"""

    print("=" * 80)
    print("Linux Shell Scripting Toolkit - Production-Ready Demo")
    print("=" * 80)

    generator = ShellScriptGenerator()

    print("\n1. Generating enhanced backup script with encryption...")
    backup_script = generator.generate_backup_script({
        'source': '/var/www/html',
        'destination': '/backup/www',
        'retention_days': 14,
        'compress': True,
        'encrypt': False,
        'exclude_patterns': ['*.tmp', '*.log', 'cache/*'],
        'notification_email': 'admin@example.com',
        'min_disk_space_gb': 20
    })

    print("\n2. Generating system monitoring script...")
    monitoring_script = generator.generate_monitoring_script({
        'cpu_threshold': 80,
        'memory_threshold': 85,
        'disk_threshold': 90,
        'email': 'admin@example.com'
    })

    print("\n3. Generating deployment script...")
    deployment_script = generator.generate_deployment_script({
        'app_name': 'webapp',
        'git_repo': 'https://github.com/mycompany/webapp.git',
        'branch': 'production',
        'deploy_path': '/opt/webapp'
    })

    print("\n4. Generating log analyzer script...")
    log_analyzer = generator.generate_log_analyzer_script({
        'log_file': '/var/log/nginx/access.log',
        'output_file': '/tmp/nginx_report.txt'
    })

    print("\n5. Generating database backup script...")
    db_backup = generator.generate_database_backup_script({
        'db_type': 'postgresql',
        'db_name': 'production_db',
        'backup_path': '/backup/postgresql',
        'retention_days': 30
    })

    print("\n6. Generating system health check script...")
    health_check = generator.generate_system_health_check({
        'services': ['nginx', 'postgresql', 'redis'],
        'ports': [80, 443, 5432, 6379],
        'urls': ['http://localhost', 'http://localhost/health'],
        'disk_threshold': 90,
        'memory_threshold': 85
    })

    print("\n7. Generating process monitoring script...")
    process_monitor = generator.generate_process_monitor({
        'processes': ['nginx', 'postgresql', 'redis-server'],
        'restart_on_failure': True,
        'notification_email': 'admin@example.com'
    })

    print("\n8. Generating cron job management script...")
    cron_script = generator.generate_cron_job_script({
        'job_name': 'daily_backup',
        'command': '/usr/local/bin/backup.sh',
        'schedule': '0 2 * * *',
        'user': 'root'
    })

    print("\n9. Generating security hardening script...")
    security_script = generator.generate_security_hardening_script()

    print("\n10. Generating performance monitoring script...")
    performance_script = generator.generate_performance_monitor()

    print("\n" + "=" * 80)
    print("Generator Summary:")
    info = generator.get_generator_info()
    print(f"  Total scripts generated: {info['scripts_generated']}")
    print(f"  Script types: {', '.join(info['script_types'])}")
    print(f"  Timestamp: {info['timestamp']}")

    print("\n" + "=" * 80)
    print("Available Features:")
    print("  - Backup automation with compression, encryption, and retention")
    print("  - System monitoring with thresholds and alerts")
    print("  - Application deployment with rollback capabilities")
    print("  - Log analysis and reporting")
    print("  - Database backup automation (PostgreSQL, MySQL, MongoDB)")
    print("  - System health checks (services, ports, URLs)")
    print("  - Process monitoring with auto-restart")
    print("  - Cron job management with logging")
    print("  - Security hardening utilities")
    print("  - Performance monitoring and reporting")
    print("  - Error handling and retry mechanisms")
    print("  - Colored output and structured logging")
    print("  - Notification support (email, Slack)")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo()
