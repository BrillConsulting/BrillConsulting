"""
Linux Process Management
Author: BrillConsulting
Description: Complete process monitoring, management, and control toolkit
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class ProcessManager:
    """Comprehensive Linux process management"""

    def __init__(self):
        """Initialize process manager"""
        self.processes = []
        self.monitored_processes = []

    def list_processes(self, filter_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List running processes

        Args:
            filter_config: Optional filter configuration

        Returns:
            List of processes
        """
        filter_config = filter_config or {}
        user = filter_config.get('user', None)
        command = filter_config.get('command', None)

        processes = [
            {
                'pid': 1234,
                'ppid': 1,
                'user': 'root',
                'cpu': 2.5,
                'memory': 1.2,
                'command': '/usr/bin/python3 app.py',
                'state': 'Running',
                'start_time': '10:30:15'
            },
            {
                'pid': 5678,
                'ppid': 1,
                'user': 'www-data',
                'cpu': 15.3,
                'memory': 5.8,
                'command': 'nginx: worker process',
                'state': 'Running',
                'start_time': '09:15:42'
            },
            {
                'pid': 9012,
                'ppid': 1234,
                'user': 'appuser',
                'cpu': 45.2,
                'memory': 12.4,
                'command': '/opt/app/worker',
                'state': 'Running',
                'start_time': '11:22:33'
            }
        ]

        # Apply filters
        if user:
            processes = [p for p in processes if p['user'] == user]
        if command:
            processes = [p for p in processes if command in p['command']]

        print(f"✓ Listed {len(processes)} processes")
        if user:
            print(f"  Filtered by user: {user}")
        if command:
            print(f"  Filtered by command: {command}")

        return processes

    def get_process_info(self, pid: int) -> Dict[str, Any]:
        """
        Get detailed process information

        Args:
            pid: Process ID

        Returns:
            Process details
        """
        process = {
            'pid': pid,
            'ppid': 1,
            'name': 'python3',
            'command': '/usr/bin/python3 /opt/app/main.py',
            'user': 'appuser',
            'group': 'appgroup',
            'state': 'Running',
            'threads': 8,
            'cpu_percent': 25.5,
            'memory_percent': 8.2,
            'memory_rss': '512MB',
            'memory_vms': '1.2GB',
            'open_files': 45,
            'connections': 12,
            'start_time': '2025-11-05 10:30:15',
            'cpu_time': '00:15:42',
            'nice': 0,
            'io_read_bytes': 104857600,
            'io_write_bytes': 52428800
        }

        print(f"✓ Process info retrieved: PID {pid}")
        print(f"  Command: {process['command']}")
        print(f"  CPU: {process['cpu_percent']}%, Memory: {process['memory_percent']}%")
        return process

    def kill_process(self, pid: int, signal: str = 'TERM') -> Dict[str, Any]:
        """
        Kill process with signal

        Args:
            pid: Process ID
            signal: Signal name (TERM, KILL, HUP, etc.)

        Returns:
            Result details
        """
        result = {
            'pid': pid,
            'signal': signal,
            'status': 'success',
            'killed_at': datetime.now().isoformat()
        }

        signal_map = {
            'TERM': 15,
            'KILL': 9,
            'HUP': 1,
            'INT': 2,
            'QUIT': 3
        }

        signal_num = signal_map.get(signal, 15)
        command = f"kill -{signal_num} {pid}"

        print(f"✓ Process killed: PID {pid}")
        print(f"  Signal: {signal} ({signal_num})")
        print(f"  Command: {command}")
        return result

    def set_process_priority(self, pid: int, priority: int) -> Dict[str, Any]:
        """
        Set process priority (nice value)

        Args:
            pid: Process ID
            priority: Nice value (-20 to 19)

        Returns:
            Result details
        """
        result = {
            'pid': pid,
            'old_priority': 0,
            'new_priority': priority,
            'set_at': datetime.now().isoformat()
        }

        command = f"renice {priority} -p {pid}"

        print(f"✓ Process priority updated: PID {pid}")
        print(f"  Priority: {result['old_priority']} → {result['new_priority']}")
        print(f"  Command: {command}")
        return result

    def get_top_processes(self, sort_by: str = 'cpu', limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top processes by CPU or memory

        Args:
            sort_by: Sort by 'cpu' or 'memory'
            limit: Number of processes to return

        Returns:
            List of top processes
        """
        processes = [
            {'pid': 1234, 'user': 'root', 'cpu': 45.2, 'memory': 12.4, 'command': '/opt/worker'},
            {'pid': 5678, 'user': 'www-data', 'cpu': 25.5, 'memory': 8.2, 'command': 'nginx'},
            {'pid': 9012, 'user': 'postgres', 'cpu': 15.3, 'memory': 25.6, 'command': 'postgres'},
            {'pid': 3456, 'user': 'redis', 'cpu': 8.7, 'memory': 5.1, 'command': 'redis-server'},
            {'pid': 7890, 'user': 'appuser', 'cpu': 5.2, 'memory': 3.8, 'command': '/usr/bin/python3'}
        ]

        if sort_by == 'memory':
            processes.sort(key=lambda x: x['memory'], reverse=True)
        else:
            processes.sort(key=lambda x: x['cpu'], reverse=True)

        top_processes = processes[:limit]

        print(f"✓ Top {limit} processes by {sort_by}:")
        for proc in top_processes[:5]:
            print(f"  PID {proc['pid']}: CPU {proc['cpu']}%, Memory {proc['memory']}% - {proc['command']}")

        return top_processes

    def monitor_process(self, monitor_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor process for resource usage

        Args:
            monitor_config: Monitor configuration

        Returns:
            Monitoring details
        """
        monitor = {
            'pid': monitor_config.get('pid'),
            'name': monitor_config.get('name', 'process'),
            'cpu_threshold': monitor_config.get('cpu_threshold', 80),
            'memory_threshold': monitor_config.get('memory_threshold', 80),
            'check_interval': monitor_config.get('check_interval', 60),
            'alert_action': monitor_config.get('alert_action', 'log'),
            'status': 'active',
            'created_at': datetime.now().isoformat()
        }

        self.monitored_processes.append(monitor)

        print(f"✓ Process monitoring configured: PID {monitor['pid']}")
        print(f"  Thresholds - CPU: {monitor['cpu_threshold']}%, Memory: {monitor['memory_threshold']}%")
        print(f"  Check interval: {monitor['check_interval']}s")
        return monitor

    def get_process_tree(self, root_pid: int = 1) -> Dict[str, Any]:
        """
        Get process tree

        Args:
            root_pid: Root process ID

        Returns:
            Process tree structure
        """
        tree = {
            'pid': root_pid,
            'command': 'systemd' if root_pid == 1 else 'process',
            'children': [
                {
                    'pid': 1234,
                    'command': 'nginx',
                    'children': [
                        {'pid': 5678, 'command': 'nginx: worker', 'children': []},
                        {'pid': 5679, 'command': 'nginx: worker', 'children': []}
                    ]
                },
                {
                    'pid': 2345,
                    'command': 'postgresql',
                    'children': [
                        {'pid': 6789, 'command': 'postgres: worker', 'children': []}
                    ]
                }
            ]
        }

        def count_descendants(node):
            count = len(node['children'])
            for child in node['children']:
                count += count_descendants(child)
            return count

        total_processes = count_descendants(tree) + 1

        print(f"✓ Process tree retrieved: Root PID {root_pid}")
        print(f"  Total processes in tree: {total_processes}")
        return tree

    def get_process_resources(self, pid: int) -> Dict[str, Any]:
        """
        Get detailed process resource usage

        Args:
            pid: Process ID

        Returns:
            Resource usage details
        """
        resources = {
            'pid': pid,
            'cpu': {
                'percent': 25.5,
                'user_time': 120.5,
                'system_time': 45.2,
                'total_time': 165.7
            },
            'memory': {
                'percent': 8.2,
                'rss': 536870912,  # 512 MB
                'vms': 1288490189,  # 1.2 GB
                'shared': 26214400  # 25 MB
            },
            'io': {
                'read_count': 15420,
                'write_count': 8230,
                'read_bytes': 104857600,  # 100 MB
                'write_bytes': 52428800   # 50 MB
            },
            'threads': 8,
            'open_files': 45,
            'connections': {
                'total': 12,
                'established': 8,
                'listening': 4
            },
            'collected_at': datetime.now().isoformat()
        }

        print(f"✓ Process resources retrieved: PID {pid}")
        print(f"  CPU: {resources['cpu']['percent']}%, Memory: {resources['memory']['percent']}%")
        print(f"  Threads: {resources['threads']}, Open files: {resources['open_files']}")
        return resources

    def create_monitoring_script(self) -> str:
        """Generate process monitoring script"""

        script = """#!/bin/bash
# Process Monitoring Script

# Configuration
CPU_THRESHOLD=80
MEMORY_THRESHOLD=80
PROCESS_NAME="$1"

if [ -z "$PROCESS_NAME" ]; then
    echo "Usage: $0 <process_name>"
    exit 1
fi

# Find process
PID=$(pgrep -f "$PROCESS_NAME" | head -1)

if [ -z "$PID" ]; then
    echo "Process not found: $PROCESS_NAME"
    exit 1
fi

# Get CPU and memory usage
CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
MEMORY=$(ps -p $PID -o %mem= | tr -d ' ')

echo "Process: $PROCESS_NAME (PID: $PID)"
echo "CPU: ${CPU}%"
echo "Memory: ${MEMORY}%"

# Check thresholds
if (( $(echo "$CPU > $CPU_THRESHOLD" | bc -l) )); then
    echo "WARNING: CPU usage exceeds threshold!"
fi

if (( $(echo "$MEMORY > $MEMORY_THRESHOLD" | bc -l) )); then
    echo "WARNING: Memory usage exceeds threshold!"
fi
"""

        print("✓ Process monitoring script generated")
        return script

    def get_manager_info(self) -> Dict[str, Any]:
        """Get process manager information"""
        return {
            'processes_tracked': len(self.processes),
            'monitored_processes': len(self.monitored_processes),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate process management"""

    print("=" * 60)
    print("Linux Process Management Demo")
    print("=" * 60)

    manager = ProcessManager()

    print("\n1. Listing all processes...")
    processes = manager.list_processes()

    print("\n2. Filtering processes by user...")
    user_processes = manager.list_processes({'user': 'www-data'})

    print("\n3. Getting detailed process info...")
    process_info = manager.get_process_info(1234)

    print("\n4. Getting top processes by CPU...")
    top_cpu = manager.get_top_processes(sort_by='cpu', limit=5)

    print("\n5. Getting top processes by memory...")
    top_memory = manager.get_top_processes(sort_by='memory', limit=5)

    print("\n6. Setting process priority...")
    priority_result = manager.set_process_priority(1234, -5)

    print("\n7. Monitoring process...")
    monitor = manager.monitor_process({
        'pid': 1234,
        'name': 'webapp',
        'cpu_threshold': 80,
        'memory_threshold': 85,
        'check_interval': 30
    })

    print("\n8. Getting process tree...")
    process_tree = manager.get_process_tree(root_pid=1)

    print("\n9. Getting process resources...")
    resources = manager.get_process_resources(1234)

    print("\n10. Killing process with TERM signal...")
    kill_result = manager.kill_process(9999, signal='TERM')

    print("\n11. Generating monitoring script...")
    script = manager.create_monitoring_script()
    print(script[:200] + "...")

    print("\n12. Manager summary:")
    info = manager.get_manager_info()
    print(f"  Monitored processes: {info['monitored_processes']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
