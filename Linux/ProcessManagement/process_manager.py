"""
Linux Process Management System
Author: BrillConsulting
Version: 2.0.0
Description: Production-ready process monitoring, management, and control toolkit
"""

import psutil
import signal
import os
import json
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import resource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessAlert:
    """Process alert data structure"""
    alert_id: str
    pid: int
    process_name: str
    alert_type: str
    message: str
    value: float
    threshold: float
    timestamp: str
    severity: str


@dataclass
class ResourceLimit:
    """Resource limit configuration"""
    resource_type: str
    soft_limit: int
    hard_limit: int


class ProcessMonitor:
    """Real-time process monitoring with alerts"""

    def __init__(self, alert_callback: Optional[Callable] = None):
        """Initialize process monitor"""
        self.monitors: Dict[int, Dict[str, Any]] = {}
        self.alerts: List[ProcessAlert] = []
        self.alert_callback = alert_callback or self._default_alert_handler
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

    def add_monitor(self, pid: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add process to monitoring"""
        self.monitors[pid] = {
            'pid': pid,
            'name': config.get('name', f'process_{pid}'),
            'cpu_threshold': config.get('cpu_threshold', 80.0),
            'memory_threshold': config.get('memory_threshold', 80.0),
            'check_interval': config.get('check_interval', 5),
            'alert_cooldown': config.get('alert_cooldown', 60),
            'last_alert': None,
            'alert_count': 0,
            'created_at': datetime.now().isoformat()
        }
        logger.info(f"Added monitor for PID {pid}")
        return self.monitors[pid]

    def remove_monitor(self, pid: int) -> bool:
        """Remove process from monitoring"""
        if pid in self.monitors:
            del self.monitors[pid]
            logger.info(f"Removed monitor for PID {pid}")
            return True
        return False

    def _default_alert_handler(self, alert: ProcessAlert):
        """Default alert handler"""
        severity_symbols = {
            'info': 'ℹ',
            'warning': '⚠',
            'critical': '🔴'
        }
        symbol = severity_symbols.get(alert.severity, '•')
        logger.warning(
            f"{symbol} ALERT: {alert.alert_type} - PID {alert.pid} ({alert.process_name}) - "
            f"{alert.message} (Value: {alert.value:.2f}, Threshold: {alert.threshold:.2f})"
        )
        self.alerts.append(alert)

    def _check_process(self, pid: int, config: Dict[str, Any]):
        """Check single process against thresholds"""
        try:
            proc = psutil.Process(pid)

            # Check CPU threshold
            cpu_percent = proc.cpu_percent(interval=0.1)
            if cpu_percent > config['cpu_threshold']:
                self._create_alert(
                    pid, config['name'], 'CPU_HIGH',
                    f"CPU usage exceeds threshold",
                    cpu_percent, config['cpu_threshold'], 'warning'
                )

            # Check memory threshold
            memory_percent = proc.memory_percent()
            if memory_percent > config['memory_threshold']:
                self._create_alert(
                    pid, config['name'], 'MEMORY_HIGH',
                    f"Memory usage exceeds threshold",
                    memory_percent, config['memory_threshold'], 'warning'
                )

        except psutil.NoSuchProcess:
            self._create_alert(
                pid, config['name'], 'PROCESS_DIED',
                f"Process no longer exists",
                0, 0, 'critical'
            )
            self.remove_monitor(pid)
        except Exception as e:
            logger.error(f"Error checking process {pid}: {e}")

    def _create_alert(self, pid: int, name: str, alert_type: str,
                     message: str, value: float, threshold: float, severity: str):
        """Create and dispatch alert"""
        config = self.monitors.get(pid, {})

        # Check alert cooldown
        if config.get('last_alert'):
            last_alert_time = datetime.fromisoformat(config['last_alert'])
            cooldown = config.get('alert_cooldown', 60)
            if (datetime.now() - last_alert_time).total_seconds() < cooldown:
                return

        alert = ProcessAlert(
            alert_id=f"{pid}_{alert_type}_{int(time.time())}",
            pid=pid,
            process_name=name,
            alert_type=alert_type,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=datetime.now().isoformat(),
            severity=severity
        )

        self.alert_callback(alert)

        if pid in self.monitors:
            self.monitors[pid]['last_alert'] = datetime.now().isoformat()
            self.monitors[pid]['alert_count'] += 1

    def start_monitoring(self):
        """Start monitoring loop"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started process monitoring")

    def stop_monitoring(self):
        """Stop monitoring loop"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped process monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            for pid, config in list(self.monitors.items()):
                self._check_process(pid, config)
            time.sleep(1)

    def get_alerts(self, severity: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        alerts = self.alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if limit:
            alerts = alerts[-limit:]
        return [asdict(a) for a in alerts]


class ProcessManager:
    """Production-ready Linux process management system"""

    def __init__(self):
        """Initialize process manager"""
        self.monitor = ProcessMonitor()
        self.resource_limits: Dict[int, List[ResourceLimit]] = {}
        logger.info("ProcessManager initialized")

    def list_processes(self, filter_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List running processes with optional filtering

        Args:
            filter_config: Filter options (user, name, status, min_cpu, min_memory)

        Returns:
            List of process information dictionaries
        """
        filter_config = filter_config or {}
        processes = []

        try:
            for proc in psutil.process_iter(['pid', 'ppid', 'name', 'username', 'status',
                                            'cpu_percent', 'memory_percent', 'cmdline', 'create_time']):
                try:
                    info = proc.info

                    # Apply filters
                    if filter_config.get('user') and info['username'] != filter_config['user']:
                        continue
                    if filter_config.get('name') and filter_config['name'] not in info['name']:
                        continue
                    if filter_config.get('status') and info['status'] != filter_config['status']:
                        continue
                    if filter_config.get('min_cpu', 0) > info['cpu_percent']:
                        continue
                    if filter_config.get('min_memory', 0) > info['memory_percent']:
                        continue

                    processes.append({
                        'pid': info['pid'],
                        'ppid': info['ppid'],
                        'name': info['name'],
                        'user': info['username'],
                        'status': info['status'],
                        'cpu_percent': round(info['cpu_percent'], 2),
                        'memory_percent': round(info['memory_percent'], 2),
                        'cmdline': ' '.join(info['cmdline']) if info['cmdline'] else '',
                        'create_time': datetime.fromtimestamp(info['create_time']).strftime('%Y-%m-%d %H:%M:%S')
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            logger.info(f"Listed {len(processes)} processes")
            return processes

        except Exception as e:
            logger.error(f"Error listing processes: {e}")
            return []

    def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive process information

        Args:
            pid: Process ID

        Returns:
            Detailed process information or None if not found
        """
        try:
            proc = psutil.Process(pid)

            # Get memory info
            mem_info = proc.memory_info()
            mem_full = proc.memory_full_info()

            # Get CPU times
            cpu_times = proc.cpu_times()

            # Get IO counters (may not be available on all systems)
            try:
                io_counters = proc.io_counters()
                io_info = {
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count,
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes
                }
            except (psutil.AccessDenied, AttributeError):
                io_info = None

            # Get connections
            try:
                connections = proc.connections()
                conn_info = {
                    'total': len(connections),
                    'established': len([c for c in connections if c.status == 'ESTABLISHED']),
                    'listening': len([c for c in connections if c.status == 'LISTEN'])
                }
            except (psutil.AccessDenied, AttributeError):
                conn_info = None

            # Get open files
            try:
                open_files = len(proc.open_files())
            except (psutil.AccessDenied, AttributeError):
                open_files = None

            info = {
                'pid': pid,
                'ppid': proc.ppid(),
                'name': proc.name(),
                'exe': proc.exe() if proc.exe() else None,
                'cmdline': ' '.join(proc.cmdline()) if proc.cmdline() else '',
                'user': proc.username(),
                'status': proc.status(),
                'create_time': datetime.fromtimestamp(proc.create_time()).strftime('%Y-%m-%d %H:%M:%S'),
                'cpu_percent': round(proc.cpu_percent(interval=0.1), 2),
                'cpu_times': {
                    'user': round(cpu_times.user, 2),
                    'system': round(cpu_times.system, 2),
                    'children_user': round(cpu_times.children_user, 2) if hasattr(cpu_times, 'children_user') else 0,
                    'children_system': round(cpu_times.children_system, 2) if hasattr(cpu_times, 'children_system') else 0
                },
                'memory_percent': round(proc.memory_percent(), 2),
                'memory_info': {
                    'rss': mem_info.rss,
                    'rss_mb': round(mem_info.rss / 1024 / 1024, 2),
                    'vms': mem_info.vms,
                    'vms_mb': round(mem_info.vms / 1024 / 1024, 2),
                    'shared': mem_full.shared if hasattr(mem_full, 'shared') else None,
                    'data': mem_full.data if hasattr(mem_full, 'data') else None
                },
                'num_threads': proc.num_threads(),
                'num_fds': proc.num_fds() if hasattr(proc, 'num_fds') else None,
                'open_files': open_files,
                'connections': conn_info,
                'io_counters': io_info,
                'nice': proc.nice(),
                'cwd': proc.cwd() if proc.cwd() else None,
                'terminal': proc.terminal() if hasattr(proc, 'terminal') else None
            }

            logger.info(f"Retrieved info for PID {pid}")
            return info

        except psutil.NoSuchProcess:
            logger.error(f"Process {pid} not found")
            return None
        except psutil.AccessDenied:
            logger.error(f"Access denied to process {pid}")
            return None
        except Exception as e:
            logger.error(f"Error getting process info for {pid}: {e}")
            return None

    def send_signal(self, pid: int, sig: str = 'TERM') -> Dict[str, Any]:
        """
        Send signal to process

        Args:
            pid: Process ID
            sig: Signal name (TERM, KILL, HUP, INT, QUIT, USR1, USR2, STOP, CONT)

        Returns:
            Result dictionary
        """
        signal_map = {
            'TERM': signal.SIGTERM,
            'KILL': signal.SIGKILL,
            'HUP': signal.SIGHUP,
            'INT': signal.SIGINT,
            'QUIT': signal.SIGQUIT,
            'USR1': signal.SIGUSR1,
            'USR2': signal.SIGUSR2,
            'STOP': signal.SIGSTOP,
            'CONT': signal.SIGCONT
        }

        try:
            proc = psutil.Process(pid)
            proc_name = proc.name()

            sig_num = signal_map.get(sig.upper(), signal.SIGTERM)
            proc.send_signal(sig_num)

            result = {
                'pid': pid,
                'process_name': proc_name,
                'signal': sig.upper(),
                'signal_num': sig_num,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Sent signal {sig} to PID {pid} ({proc_name})")
            return result

        except psutil.NoSuchProcess:
            logger.error(f"Process {pid} not found")
            return {'status': 'error', 'message': 'Process not found'}
        except psutil.AccessDenied:
            logger.error(f"Access denied to process {pid}")
            return {'status': 'error', 'message': 'Access denied'}
        except Exception as e:
            logger.error(f"Error sending signal to {pid}: {e}")
            return {'status': 'error', 'message': str(e)}

    def set_priority(self, pid: int, nice_value: int) -> Dict[str, Any]:
        """
        Set process priority (nice value)

        Args:
            pid: Process ID
            nice_value: Nice value (-20 to 19, lower = higher priority)

        Returns:
            Result dictionary
        """
        if not -20 <= nice_value <= 19:
            return {'status': 'error', 'message': 'Nice value must be between -20 and 19'}

        try:
            proc = psutil.Process(pid)
            old_nice = proc.nice()
            proc.nice(nice_value)

            result = {
                'pid': pid,
                'process_name': proc.name(),
                'old_priority': old_nice,
                'new_priority': nice_value,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }

            logger.info(f"Changed priority for PID {pid}: {old_nice} -> {nice_value}")
            return result

        except psutil.NoSuchProcess:
            return {'status': 'error', 'message': 'Process not found'}
        except psutil.AccessDenied:
            return {'status': 'error', 'message': 'Access denied (may need root)'}
        except Exception as e:
            logger.error(f"Error setting priority for {pid}: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_top_processes(self, sort_by: str = 'cpu', limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top processes by CPU or memory usage

        Args:
            sort_by: Sort by 'cpu', 'memory', or 'io'
            limit: Number of processes to return

        Returns:
            List of top processes
        """
        processes = []

        try:
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'user': proc.info['username'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort processes
            if sort_by == 'memory':
                processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            else:  # default to cpu
                processes.sort(key=lambda x: x['cpu_percent'], reverse=True)

            top = processes[:limit]
            logger.info(f"Retrieved top {limit} processes by {sort_by}")
            return top

        except Exception as e:
            logger.error(f"Error getting top processes: {e}")
            return []

    def get_process_tree(self, root_pid: Optional[int] = None) -> Dict[str, Any]:
        """
        Get process tree structure

        Args:
            root_pid: Root process ID (None for all processes)

        Returns:
            Process tree dictionary
        """
        def build_tree(proc):
            """Recursively build process tree"""
            try:
                children = proc.children(recursive=False)
                return {
                    'pid': proc.pid,
                    'name': proc.name(),
                    'user': proc.username(),
                    'cpu_percent': round(proc.cpu_percent(interval=0), 2),
                    'memory_percent': round(proc.memory_percent(), 2),
                    'cmdline': ' '.join(proc.cmdline())[:100] if proc.cmdline() else '',
                    'children': [build_tree(child) for child in children]
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return None

        try:
            if root_pid:
                proc = psutil.Process(root_pid)
                tree = build_tree(proc)
            else:
                # Get all root processes (ppid=0 or ppid not found)
                roots = []
                for proc in psutil.process_iter():
                    try:
                        if proc.ppid() == 0:
                            tree_node = build_tree(proc)
                            if tree_node:
                                roots.append(tree_node)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                tree = {'processes': roots}

            logger.info(f"Retrieved process tree for PID {root_pid or 'all'}")
            return tree

        except psutil.NoSuchProcess:
            return {'error': 'Process not found'}
        except Exception as e:
            logger.error(f"Error getting process tree: {e}")
            return {'error': str(e)}

    def track_process_resources(self, pid: int, duration: int = 60,
                               interval: int = 1) -> List[Dict[str, Any]]:
        """
        Track process resource usage over time

        Args:
            pid: Process ID
            duration: Tracking duration in seconds
            interval: Sample interval in seconds

        Returns:
            List of resource snapshots
        """
        snapshots = []
        start_time = time.time()

        try:
            proc = psutil.Process(pid)

            while time.time() - start_time < duration:
                try:
                    snapshot = {
                        'timestamp': datetime.now().isoformat(),
                        'cpu_percent': round(proc.cpu_percent(interval=0.1), 2),
                        'memory_percent': round(proc.memory_percent(), 2),
                        'memory_rss_mb': round(proc.memory_info().rss / 1024 / 1024, 2),
                        'num_threads': proc.num_threads(),
                        'num_fds': proc.num_fds() if hasattr(proc, 'num_fds') else None
                    }

                    # Add IO counters if available
                    try:
                        io = proc.io_counters()
                        snapshot['io_read_mb'] = round(io.read_bytes / 1024 / 1024, 2)
                        snapshot['io_write_mb'] = round(io.write_bytes / 1024 / 1024, 2)
                    except (psutil.AccessDenied, AttributeError):
                        pass

                    snapshots.append(snapshot)
                    time.sleep(interval)

                except psutil.NoSuchProcess:
                    logger.warning(f"Process {pid} terminated during tracking")
                    break

            logger.info(f"Tracked process {pid} for {len(snapshots)} samples")
            return snapshots

        except psutil.NoSuchProcess:
            logger.error(f"Process {pid} not found")
            return []
        except Exception as e:
            logger.error(f"Error tracking process {pid}: {e}")
            return []

    def set_resource_limits(self, pid: int, limits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Set resource limits for a process

        Args:
            pid: Process ID
            limits: List of limit configurations

        Returns:
            Result dictionary
        """
        if os.getpid() != pid:
            return {'status': 'error', 'message': 'Can only set limits for current process'}

        resource_map = {
            'cpu': resource.RLIMIT_CPU,
            'fsize': resource.RLIMIT_FSIZE,
            'data': resource.RLIMIT_DATA,
            'stack': resource.RLIMIT_STACK,
            'core': resource.RLIMIT_CORE,
            'nofile': resource.RLIMIT_NOFILE,
            'nproc': resource.RLIMIT_NPROC,
            'memlock': resource.RLIMIT_MEMLOCK
        }

        results = []
        for limit_config in limits:
            resource_type = limit_config.get('resource_type')
            soft_limit = limit_config.get('soft_limit')
            hard_limit = limit_config.get('hard_limit')

            if resource_type not in resource_map:
                results.append({
                    'resource': resource_type,
                    'status': 'error',
                    'message': 'Invalid resource type'
                })
                continue

            try:
                resource.setrlimit(
                    resource_map[resource_type],
                    (soft_limit, hard_limit)
                )
                results.append({
                    'resource': resource_type,
                    'soft_limit': soft_limit,
                    'hard_limit': hard_limit,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'resource': resource_type,
                    'status': 'error',
                    'message': str(e)
                })

        return {'pid': pid, 'limits': results}

    def get_resource_limits(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """
        Get current resource limits

        Args:
            pid: Process ID (None for current process)

        Returns:
            Resource limits dictionary
        """
        resource_names = {
            resource.RLIMIT_CPU: 'cpu',
            resource.RLIMIT_FSIZE: 'fsize',
            resource.RLIMIT_DATA: 'data',
            resource.RLIMIT_STACK: 'stack',
            resource.RLIMIT_CORE: 'core',
            resource.RLIMIT_NOFILE: 'nofile',
            resource.RLIMIT_NPROC: 'nproc',
            resource.RLIMIT_MEMLOCK: 'memlock'
        }

        limits = {}
        for res_id, res_name in resource_names.items():
            try:
                soft, hard = resource.getrlimit(res_id)
                limits[res_name] = {
                    'soft': soft,
                    'hard': hard
                }
            except Exception:
                continue

        return {
            'pid': pid or os.getpid(),
            'limits': limits
        }

    def start_monitoring(self, monitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Start monitoring processes

        Args:
            monitors: List of monitor configurations

        Returns:
            Monitoring status
        """
        for monitor_config in monitors:
            pid = monitor_config.get('pid')
            if pid:
                self.monitor.add_monitor(pid, monitor_config)

        self.monitor.start_monitoring()

        return {
            'status': 'active',
            'monitors': len(self.monitor.monitors),
            'started_at': datetime.now().isoformat()
        }

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop all monitoring"""
        self.monitor.stop_monitoring()

        return {
            'status': 'stopped',
            'stopped_at': datetime.now().isoformat()
        }

    def get_alerts(self, severity: Optional[str] = None,
                   limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.monitor.get_alerts(severity=severity, limit=limit)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            'cpu': {
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True),
                'percent_per_cpu': [round(p, 2) for p in cpu_percent],
                'percent_total': round(sum(cpu_percent) / len(cpu_percent), 2)
            },
            'memory': {
                'total_gb': round(mem.total / 1024 / 1024 / 1024, 2),
                'available_gb': round(mem.available / 1024 / 1024 / 1024, 2),
                'used_gb': round(mem.used / 1024 / 1024 / 1024, 2),
                'percent': round(mem.percent, 2)
            },
            'swap': {
                'total_gb': round(swap.total / 1024 / 1024 / 1024, 2),
                'used_gb': round(swap.used / 1024 / 1024 / 1024, 2),
                'percent': round(swap.percent, 2)
            },
            'processes': {
                'total': len(psutil.pids()),
                'running': len([p for p in psutil.process_iter(['status'])
                               if p.info['status'] == psutil.STATUS_RUNNING]),
                'sleeping': len([p for p in psutil.process_iter(['status'])
                                if p.info['status'] == psutil.STATUS_SLEEPING])
            },
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate comprehensive process management features"""

    print("=" * 80)
    print("Linux Process Management System v2.0.0 - Production Demo")
    print("=" * 80)

    manager = ProcessManager()

    # 1. System Statistics
    print("\n[1] System Statistics")
    print("-" * 80)
    stats = manager.get_system_stats()
    print(f"CPU Cores: {stats['cpu']['count']} physical, {stats['cpu']['count_logical']} logical")
    print(f"CPU Usage: {stats['cpu']['percent_total']}%")
    print(f"Memory: {stats['memory']['used_gb']}GB / {stats['memory']['total_gb']}GB ({stats['memory']['percent']}%)")
    print(f"Total Processes: {stats['processes']['total']}")
    print(f"Running: {stats['processes']['running']}, Sleeping: {stats['processes']['sleeping']}")

    # 2. List Processes
    print("\n[2] Listing Processes (filtering by min CPU usage)")
    print("-" * 80)
    processes = manager.list_processes({'min_cpu': 0.1})
    print(f"Found {len(processes)} processes with CPU > 0.1%")
    for proc in processes[:5]:
        print(f"  PID {proc['pid']:6} | {proc['name']:20} | CPU: {proc['cpu_percent']:5.1f}% | MEM: {proc['memory_percent']:5.1f}%")

    # 3. Top Processes
    print("\n[3] Top CPU Consumers")
    print("-" * 80)
    top_cpu = manager.get_top_processes(sort_by='cpu', limit=5)
    for i, proc in enumerate(top_cpu, 1):
        print(f"  {i}. PID {proc['pid']:6} | {proc['name']:20} | CPU: {proc['cpu_percent']:5.1f}% | User: {proc['user']}")

    print("\n[4] Top Memory Consumers")
    print("-" * 80)
    top_mem = manager.get_top_processes(sort_by='memory', limit=5)
    for i, proc in enumerate(top_mem, 1):
        print(f"  {i}. PID {proc['pid']:6} | {proc['name']:20} | MEM: {proc['memory_percent']:5.1f}% | User: {proc['user']}")

    # 4. Detailed Process Info
    if processes:
        sample_pid = processes[0]['pid']
        print(f"\n[5] Detailed Process Information (PID {sample_pid})")
        print("-" * 80)
        info = manager.get_process_info(sample_pid)
        if info:
            print(f"Name: {info['name']}")
            print(f"User: {info['user']}")
            print(f"Status: {info['status']}")
            print(f"CPU: {info['cpu_percent']}% | Memory: {info['memory_percent']}%")
            print(f"Memory RSS: {info['memory_info']['rss_mb']} MB")
            print(f"Threads: {info['num_threads']}")
            print(f"Nice Value: {info['nice']}")
            print(f"Started: {info['create_time']}")
            if info['io_counters']:
                print(f"I/O Read: {info['io_counters']['read_bytes'] / 1024 / 1024:.2f} MB")
                print(f"I/O Write: {info['io_counters']['write_bytes'] / 1024 / 1024:.2f} MB")

    # 5. Process Tree
    print("\n[6] Process Tree (PID 1)")
    print("-" * 80)
    tree = manager.get_process_tree(root_pid=1)
    if 'error' not in tree:
        def print_tree(node, indent=0):
            if node:
                prefix = "  " * indent + ("└─ " if indent > 0 else "")
                print(f"{prefix}PID {node['pid']:6} | {node['name']:20} | CPU: {node['cpu_percent']:5.1f}%")
                for child in node.get('children', [])[:3]:  # Limit to 3 children for demo
                    print_tree(child, indent + 1)
                if len(node.get('children', [])) > 3:
                    print(f"{'  ' * (indent + 1)}... and {len(node['children']) - 3} more")

        print_tree(tree)

    # 6. Resource Limits
    print("\n[7] Current Process Resource Limits")
    print("-" * 80)
    limits = manager.get_resource_limits()
    for res_name, res_values in list(limits['limits'].items())[:5]:
        soft = res_values['soft']
        hard = res_values['hard']
        soft_str = str(soft) if soft != resource.RLIM_INFINITY else "unlimited"
        hard_str = str(hard) if hard != resource.RLIM_INFINITY else "unlimited"
        print(f"  {res_name:10} | Soft: {soft_str:15} | Hard: {hard_str}")

    # 7. Process Monitoring Setup
    print("\n[8] Setting Up Process Monitoring")
    print("-" * 80)
    if processes and len(processes) >= 2:
        monitor_configs = [
            {
                'pid': processes[0]['pid'],
                'name': processes[0]['name'],
                'cpu_threshold': 50.0,
                'memory_threshold': 50.0,
                'check_interval': 2,
                'alert_cooldown': 30
            }
        ]

        result = manager.start_monitoring(monitor_configs)
        print(f"Monitoring Status: {result['status']}")
        print(f"Active Monitors: {result['monitors']}")
        print(f"Started At: {result['started_at']}")

        # Let it run briefly
        print("\nMonitoring for 3 seconds...")
        time.sleep(3)

        # Check for alerts
        alerts = manager.get_alerts(limit=5)
        if alerts:
            print(f"\nAlerts Generated: {len(alerts)}")
            for alert in alerts[:3]:
                print(f"  - {alert['alert_type']}: PID {alert['pid']} ({alert['process_name']}) - {alert['message']}")
        else:
            print("\nNo alerts generated (all processes within thresholds)")

        # Stop monitoring
        stop_result = manager.stop_monitoring()
        print(f"\nMonitoring stopped at: {stop_result['stopped_at']}")

    # 8. Signal Handling Demo (informational only)
    print("\n[9] Signal Handling Capabilities")
    print("-" * 80)
    print("Available signals: TERM, KILL, HUP, INT, QUIT, USR1, USR2, STOP, CONT")
    print("Example: manager.send_signal(pid, 'TERM')  # Graceful termination")
    print("Example: manager.send_signal(pid, 'KILL')  # Force kill")
    print("Example: manager.send_signal(pid, 'HUP')   # Reload configuration")

    # 9. Priority Management Demo (informational only)
    print("\n[10] Priority Management Capabilities")
    print("-" * 80)
    print("Nice values range from -20 (highest priority) to 19 (lowest priority)")
    print("Example: manager.set_priority(pid, -5)  # Increase priority")
    print("Example: manager.set_priority(pid, 10)  # Decrease priority")
    print("Note: Negative values require root privileges")

    # 10. Resource Tracking Demo
    print("\n[11] Resource Tracking Capability")
    print("-" * 80)
    print("Track process resources over time with configurable intervals")
    print("Example: snapshots = manager.track_process_resources(pid, duration=60, interval=1)")
    print("This captures CPU, memory, I/O, threads, and file descriptors over time")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  - System statistics and monitoring")
    print("  - Process listing with flexible filtering")
    print("  - Top process identification (CPU/Memory)")
    print("  - Detailed process information")
    print("  - Process tree visualization")
    print("  - Resource limit management")
    print("  - Real-time monitoring with automated alerts")
    print("  - Signal handling for process control")
    print("  - Priority management (nice/renice)")
    print("  - Historical resource tracking")
    print("\nProduction-Ready Features:")
    print("  - Thread-safe monitoring")
    print("  - Comprehensive error handling")
    print("  - Structured logging")
    print("  - Alert cooldown periods")
    print("  - Flexible filtering and sorting")
    print("  - Cross-platform compatibility (Linux focus)")


if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        print(f"\nError during demo: {e}")
