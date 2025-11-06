"""
KernelTuning - Production-Ready Linux Kernel Optimization System
Author: BrillConsulting
Description: Comprehensive kernel parameter optimization, module management, and performance tuning

Features:
- sysctl parameter management and optimization
- Kernel module loading and configuration
- Kernel compilation automation
- Performance tuning profiles (web, database, realtime, HPC)
- Boot parameter configuration (GRUB)
- Real-time kernel setup and validation
- NUMA topology optimization
"""

import os
import sys
import subprocess
import json
import logging
import shutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import re


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TuningProfile(Enum):
    """Predefined kernel tuning profiles for different workloads"""
    WEB_SERVER = "web_server"
    DATABASE = "database"
    REALTIME = "realtime"
    HPC = "hpc"
    STORAGE = "storage"
    NETWORK = "network"
    DESKTOP = "desktop"
    CUSTOM = "custom"


@dataclass
class SysctlParameter:
    """Represents a sysctl kernel parameter"""
    name: str
    value: str
    description: str
    category: str
    requires_reboot: bool = False


@dataclass
class KernelModule:
    """Represents a kernel module configuration"""
    name: str
    parameters: Dict[str, str]
    load_on_boot: bool = True
    description: str = ""


@dataclass
class TuningResult:
    """Result of a tuning operation"""
    success: bool
    operation: str
    details: str
    timestamp: str
    changes: List[str]


class SysctlManager:
    """Manages sysctl kernel parameters"""

    def __init__(self, backup_dir: str = "/var/backups/kernel-tuning"):
        self.backup_dir = Path(backup_dir)
        self.sysctl_conf = Path("/etc/sysctl.conf")
        self.sysctl_d = Path("/etc/sysctl.d")
        self._ensure_backup_dir()

    def _ensure_backup_dir(self):
        """Create backup directory if it doesn't exist"""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create backup directory: {e}")

    def get_parameter(self, param: str) -> Optional[str]:
        """Get current value of a sysctl parameter"""
        try:
            result = subprocess.run(
                ['sysctl', '-n', param],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception as e:
            logger.error(f"Error getting parameter {param}: {e}")
            return None

    def set_parameter(self, param: str, value: str, persistent: bool = True) -> bool:
        """Set a sysctl parameter"""
        try:
            # Set runtime value
            result = subprocess.run(
                ['sysctl', '-w', f'{param}={value}'],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                logger.error(f"Failed to set {param}={value}: {result.stderr}")
                return False

            # Make persistent if requested
            if persistent:
                self._persist_parameter(param, value)

            logger.info(f"Set {param}={value}")
            return True
        except Exception as e:
            logger.error(f"Error setting parameter {param}: {e}")
            return False

    def _persist_parameter(self, param: str, value: str):
        """Persist parameter to sysctl configuration"""
        config_file = self.sysctl_d / "99-kernel-tuning.conf"

        try:
            # Create or append to config file
            with open(config_file, 'a') as f:
                f.write(f"{param} = {value}\n")
            logger.info(f"Persisted {param} to {config_file}")
        except Exception as e:
            logger.error(f"Error persisting parameter: {e}")

    def backup_current_config(self) -> str:
        """Backup current sysctl configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"sysctl_backup_{timestamp}.conf"

        try:
            result = subprocess.run(
                ['sysctl', '-a'],
                capture_output=True,
                text=True,
                check=True
            )

            with open(backup_file, 'w') as f:
                f.write(result.stdout)

            logger.info(f"Backed up sysctl config to {backup_file}")
            return str(backup_file)
        except Exception as e:
            logger.error(f"Error backing up config: {e}")
            return ""

    def apply_profile_parameters(self, parameters: List[SysctlParameter]) -> List[str]:
        """Apply a list of sysctl parameters"""
        changes = []
        for param in parameters:
            current = self.get_parameter(param.name)
            if current != param.value:
                if self.set_parameter(param.name, param.value):
                    changes.append(f"{param.name}: {current} -> {param.value}")
        return changes


class KernelModuleManager:
    """Manages kernel module loading and configuration"""

    def __init__(self):
        self.modules_load_d = Path("/etc/modules-load.d")
        self.modprobe_d = Path("/etc/modprobe.d")

    def is_module_loaded(self, module_name: str) -> bool:
        """Check if a kernel module is loaded"""
        try:
            result = subprocess.run(
                ['lsmod'],
                capture_output=True,
                text=True,
                check=True
            )
            return module_name in result.stdout
        except Exception as e:
            logger.error(f"Error checking module {module_name}: {e}")
            return False

    def load_module(self, module: KernelModule) -> bool:
        """Load a kernel module with parameters"""
        try:
            # Build modprobe command with parameters
            cmd = ['modprobe', module.name]
            for key, value in module.parameters.items():
                cmd.append(f"{key}={value}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                logger.error(f"Failed to load module {module.name}: {result.stderr}")
                return False

            logger.info(f"Loaded module {module.name}")

            # Configure for boot loading if requested
            if module.load_on_boot:
                self._configure_boot_loading(module)

            return True
        except Exception as e:
            logger.error(f"Error loading module {module.name}: {e}")
            return False

    def _configure_boot_loading(self, module: KernelModule):
        """Configure module to load at boot"""
        try:
            # Add to modules-load.d
            load_file = self.modules_load_d / f"{module.name}.conf"
            with open(load_file, 'w') as f:
                f.write(f"# Auto-configured by KernelTuning\n")
                f.write(f"{module.name}\n")

            # Add parameters to modprobe.d
            if module.parameters:
                param_file = self.modprobe_d / f"{module.name}.conf"
                with open(param_file, 'w') as f:
                    f.write(f"# Auto-configured by KernelTuning\n")
                    params = " ".join([f"{k}={v}" for k, v in module.parameters.items()])
                    f.write(f"options {module.name} {params}\n")

            logger.info(f"Configured {module.name} for boot loading")
        except Exception as e:
            logger.error(f"Error configuring boot loading: {e}")

    def unload_module(self, module_name: str) -> bool:
        """Unload a kernel module"""
        try:
            result = subprocess.run(
                ['modprobe', '-r', module_name],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                logger.error(f"Failed to unload module {module_name}: {result.stderr}")
                return False

            logger.info(f"Unloaded module {module_name}")
            return True
        except Exception as e:
            logger.error(f"Error unloading module {module_name}: {e}")
            return False

    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """Get information about a kernel module"""
        try:
            result = subprocess.run(
                ['modinfo', module_name],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                return {}

            info = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()

            return info
        except Exception as e:
            logger.error(f"Error getting module info: {e}")
            return {}


class GRUBManager:
    """Manages GRUB bootloader configuration and kernel boot parameters"""

    def __init__(self):
        self.grub_default = Path("/etc/default/grub")
        self.grub_cfg = Path("/boot/grub/grub.cfg")

    def get_cmdline_parameters(self) -> List[str]:
        """Get current kernel command line parameters"""
        try:
            with open('/proc/cmdline', 'r') as f:
                return f.read().strip().split()
        except Exception as e:
            logger.error(f"Error reading cmdline: {e}")
            return []

    def add_boot_parameter(self, parameter: str, update_grub: bool = True) -> bool:
        """Add a boot parameter to GRUB configuration"""
        try:
            # Backup GRUB config
            backup = f"{self.grub_default}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(self.grub_default, backup)
            logger.info(f"Backed up GRUB config to {backup}")

            # Read current config
            with open(self.grub_default, 'r') as f:
                lines = f.readlines()

            # Update GRUB_CMDLINE_LINUX_DEFAULT
            updated = False
            for i, line in enumerate(lines):
                if line.startswith('GRUB_CMDLINE_LINUX_DEFAULT='):
                    # Extract current parameters
                    match = re.search(r'GRUB_CMDLINE_LINUX_DEFAULT="([^"]*)"', line)
                    if match:
                        current_params = match.group(1)
                        if parameter not in current_params:
                            new_params = f"{current_params} {parameter}".strip()
                            lines[i] = f'GRUB_CMDLINE_LINUX_DEFAULT="{new_params}"\n'
                            updated = True
                    break

            if updated:
                with open(self.grub_default, 'w') as f:
                    f.writelines(lines)
                logger.info(f"Added boot parameter: {parameter}")

                if update_grub:
                    self._update_grub()

                return True
            return False
        except Exception as e:
            logger.error(f"Error adding boot parameter: {e}")
            return False

    def _update_grub(self) -> bool:
        """Update GRUB configuration"""
        try:
            # Try different update-grub commands based on distribution
            commands = [
                ['update-grub'],
                ['grub2-mkconfig', '-o', '/boot/grub2/grub.cfg'],
                ['grub-mkconfig', '-o', '/boot/grub/grub.cfg']
            ]

            for cmd in commands:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    logger.info("Updated GRUB configuration")
                    return True

            logger.warning("Could not update GRUB configuration")
            return False
        except Exception as e:
            logger.error(f"Error updating GRUB: {e}")
            return False


class NUMAManager:
    """Manages NUMA (Non-Uniform Memory Access) topology and optimization"""

    def __init__(self):
        self.numa_available = self._check_numa_available()

    def _check_numa_available(self) -> bool:
        """Check if NUMA is available on the system"""
        return Path("/sys/devices/system/node").exists()

    def get_numa_topology(self) -> Dict[str, Any]:
        """Get NUMA topology information"""
        if not self.numa_available:
            return {"available": False}

        topology = {"available": True, "nodes": []}

        try:
            result = subprocess.run(
                ['numactl', '--hardware'],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                topology["raw_info"] = result.stdout

            # Parse node information
            node_dirs = Path("/sys/devices/system/node").glob("node*")
            for node_dir in sorted(node_dirs):
                if node_dir.is_dir() and node_dir.name.startswith("node"):
                    node_id = node_dir.name.replace("node", "")
                    node_info = self._get_node_info(node_dir)
                    node_info["id"] = node_id
                    topology["nodes"].append(node_info)

            return topology
        except Exception as e:
            logger.error(f"Error getting NUMA topology: {e}")
            return {"available": True, "error": str(e)}

    def _get_node_info(self, node_path: Path) -> Dict[str, Any]:
        """Get information about a specific NUMA node"""
        info = {}

        try:
            # Get CPU list
            cpulist_file = node_path / "cpulist"
            if cpulist_file.exists():
                with open(cpulist_file, 'r') as f:
                    info["cpulist"] = f.read().strip()

            # Get memory info
            meminfo_file = node_path / "meminfo"
            if meminfo_file.exists():
                with open(meminfo_file, 'r') as f:
                    info["meminfo"] = f.read()

            return info
        except Exception as e:
            logger.error(f"Error getting node info: {e}")
            return {}

    def optimize_numa_settings(self) -> List[str]:
        """Apply NUMA optimization settings"""
        changes = []

        if not self.numa_available:
            logger.warning("NUMA not available on this system")
            return changes

        sysctl_mgr = SysctlManager()

        # NUMA balancing settings
        numa_params = [
            SysctlParameter(
                name="kernel.numa_balancing",
                value="1",
                description="Enable automatic NUMA balancing",
                category="numa"
            ),
            SysctlParameter(
                name="kernel.numa_balancing_scan_delay_ms",
                value="1000",
                description="NUMA scan delay",
                category="numa"
            ),
            SysctlParameter(
                name="kernel.numa_balancing_scan_period_min_ms",
                value="1000",
                description="Minimum NUMA scan period",
                category="numa"
            ),
            SysctlParameter(
                name="kernel.numa_balancing_scan_period_max_ms",
                value="60000",
                description="Maximum NUMA scan period",
                category="numa"
            )
        ]

        changes = sysctl_mgr.apply_profile_parameters(numa_params)
        return changes


class RealtimeKernelManager:
    """Manages real-time kernel configuration and validation"""

    def __init__(self):
        self.is_rt_kernel = self._check_rt_kernel()

    def _check_rt_kernel(self) -> bool:
        """Check if running a real-time kernel"""
        try:
            with open('/proc/version', 'r') as f:
                version = f.read().lower()
                return 'preempt_rt' in version or 'rt' in version
        except Exception:
            return False

    def get_rt_info(self) -> Dict[str, Any]:
        """Get real-time kernel information"""
        info = {
            "is_rt_kernel": self.is_rt_kernel,
            "kernel_version": "",
            "preemption_model": ""
        }

        try:
            # Get kernel version
            with open('/proc/version', 'r') as f:
                info["kernel_version"] = f.read().strip()

            # Get preemption model
            preempt_file = Path("/sys/kernel/debug/sched_features")
            if preempt_file.exists():
                with open(preempt_file, 'r') as f:
                    info["sched_features"] = f.read().strip()

            return info
        except Exception as e:
            logger.error(f"Error getting RT info: {e}")
            return info

    def configure_rt_parameters(self) -> List[str]:
        """Configure parameters for real-time performance"""
        sysctl_mgr = SysctlManager()

        rt_params = [
            SysctlParameter(
                name="kernel.sched_rt_period_us",
                value="1000000",
                description="RT scheduling period",
                category="realtime"
            ),
            SysctlParameter(
                name="kernel.sched_rt_runtime_us",
                value="950000",
                description="RT runtime allocation",
                category="realtime"
            ),
            SysctlParameter(
                name="kernel.sched_latency_ns",
                value="10000000",
                description="Scheduler latency",
                category="realtime"
            ),
            SysctlParameter(
                name="kernel.sched_min_granularity_ns",
                value="2000000",
                description="Minimum scheduling granularity",
                category="realtime"
            ),
            SysctlParameter(
                name="kernel.sched_wakeup_granularity_ns",
                value="3000000",
                description="Wakeup granularity",
                category="realtime"
            )
        ]

        changes = sysctl_mgr.apply_profile_parameters(rt_params)
        return changes


class KernelCompilationManager:
    """Manages kernel compilation and configuration"""

    def __init__(self, work_dir: str = "/usr/src"):
        self.work_dir = Path(work_dir)
        self.kernel_config = Path("/boot/config-" + os.uname().release)

    def get_current_config(self) -> Optional[str]:
        """Get current kernel configuration"""
        if self.kernel_config.exists():
            return str(self.kernel_config)

        # Try alternative locations
        alt_configs = [
            Path("/proc/config.gz"),
            Path("/boot/config-$(uname -r)")
        ]

        for config in alt_configs:
            if config.exists():
                return str(config)

        return None

    def prepare_compilation_env(self) -> Dict[str, Any]:
        """Prepare environment for kernel compilation"""
        result = {
            "dependencies_installed": False,
            "source_available": False,
            "config_available": False
        }

        try:
            # Check for required packages
            required_packages = [
                'build-essential', 'libncurses-dev', 'bison', 'flex',
                'libssl-dev', 'libelf-dev', 'bc'
            ]

            logger.info("Checking compilation dependencies...")
            result["required_packages"] = required_packages

            # Check if source directory exists
            if self.work_dir.exists():
                result["source_available"] = True

            # Check if config is available
            if self.get_current_config():
                result["config_available"] = True

            return result
        except Exception as e:
            logger.error(f"Error preparing compilation environment: {e}")
            return result

    def generate_custom_config(self, profile: TuningProfile) -> str:
        """Generate custom kernel configuration based on profile"""
        config_options = {
            TuningProfile.REALTIME: [
                "CONFIG_PREEMPT_RT=y",
                "CONFIG_PREEMPT=y",
                "CONFIG_NO_HZ_FULL=y",
                "CONFIG_RCU_NOCB_CPU=y",
                "CONFIG_IRQ_FORCED_THREADING=y"
            ],
            TuningProfile.HPC: [
                "CONFIG_NUMA=y",
                "CONFIG_TRANSPARENT_HUGEPAGE=y",
                "CONFIG_HUGETLBFS=y",
                "CONFIG_CGROUP_CPUACCT=y"
            ],
            TuningProfile.NETWORK: [
                "CONFIG_NET_SCHED=y",
                "CONFIG_NET_CLS_CGROUP=y",
                "CONFIG_XPS=y",
                "CONFIG_RPS=y"
            ]
        }

        options = config_options.get(profile, [])
        return "\n".join(options)


class TuningProfileManager:
    """Manages predefined tuning profiles for different workloads"""

    def __init__(self):
        self.sysctl_mgr = SysctlManager()
        self.module_mgr = KernelModuleManager()
        self.grub_mgr = GRUBManager()
        self.numa_mgr = NUMAManager()
        self.rt_mgr = RealtimeKernelManager()

    def get_profile_parameters(self, profile: TuningProfile) -> List[SysctlParameter]:
        """Get sysctl parameters for a specific profile"""

        if profile == TuningProfile.WEB_SERVER:
            return self._get_web_server_params()
        elif profile == TuningProfile.DATABASE:
            return self._get_database_params()
        elif profile == TuningProfile.REALTIME:
            return self._get_realtime_params()
        elif profile == TuningProfile.HPC:
            return self._get_hpc_params()
        elif profile == TuningProfile.STORAGE:
            return self._get_storage_params()
        elif profile == TuningProfile.NETWORK:
            return self._get_network_params()
        else:
            return []

    def _get_web_server_params(self) -> List[SysctlParameter]:
        """Web server optimization parameters"""
        return [
            SysctlParameter(
                name="net.core.somaxconn",
                value="65535",
                description="Maximum socket connections",
                category="network"
            ),
            SysctlParameter(
                name="net.ipv4.tcp_max_syn_backlog",
                value="8192",
                description="TCP SYN backlog",
                category="network"
            ),
            SysctlParameter(
                name="net.ipv4.tcp_tw_reuse",
                value="1",
                description="Reuse TIME-WAIT sockets",
                category="network"
            ),
            SysctlParameter(
                name="net.ipv4.ip_local_port_range",
                value="10000 65535",
                description="Local port range",
                category="network"
            ),
            SysctlParameter(
                name="net.core.netdev_max_backlog",
                value="5000",
                description="Network device backlog",
                category="network"
            ),
            SysctlParameter(
                name="vm.swappiness",
                value="10",
                description="Reduce swap usage",
                category="memory"
            )
        ]

    def _get_database_params(self) -> List[SysctlParameter]:
        """Database server optimization parameters"""
        return [
            SysctlParameter(
                name="vm.swappiness",
                value="1",
                description="Minimize swapping",
                category="memory"
            ),
            SysctlParameter(
                name="vm.dirty_ratio",
                value="15",
                description="Dirty page cache ratio",
                category="memory"
            ),
            SysctlParameter(
                name="vm.dirty_background_ratio",
                value="5",
                description="Background dirty ratio",
                category="memory"
            ),
            SysctlParameter(
                name="kernel.shmmax",
                value="68719476736",
                description="Maximum shared memory segment (64GB)",
                category="memory"
            ),
            SysctlParameter(
                name="kernel.shmall",
                value="4294967296",
                description="Total shared memory pages",
                category="memory"
            ),
            SysctlParameter(
                name="kernel.sem",
                value="250 32000 100 128",
                description="Semaphore limits",
                category="ipc"
            ),
            SysctlParameter(
                name="fs.file-max",
                value="2097152",
                description="Maximum file handles",
                category="filesystem"
            )
        ]

    def _get_realtime_params(self) -> List[SysctlParameter]:
        """Real-time application parameters"""
        return [
            SysctlParameter(
                name="kernel.sched_rt_runtime_us",
                value="950000",
                description="RT runtime allocation",
                category="realtime"
            ),
            SysctlParameter(
                name="kernel.sched_latency_ns",
                value="10000000",
                description="Scheduler latency",
                category="realtime"
            ),
            SysctlParameter(
                name="kernel.sched_min_granularity_ns",
                value="2000000",
                description="Minimum scheduling granularity",
                category="realtime"
            ),
            SysctlParameter(
                name="vm.swappiness",
                value="0",
                description="Disable swapping",
                category="memory"
            ),
            SysctlParameter(
                name="kernel.sched_migration_cost_ns",
                value="5000000",
                description="CPU migration cost",
                category="realtime"
            )
        ]

    def _get_hpc_params(self) -> List[SysctlParameter]:
        """High-Performance Computing parameters"""
        return [
            SysctlParameter(
                name="vm.swappiness",
                value="1",
                description="Minimize swapping",
                category="memory"
            ),
            SysctlParameter(
                name="kernel.sched_autogroup_enabled",
                value="0",
                description="Disable automatic task grouping",
                category="scheduler"
            ),
            SysctlParameter(
                name="kernel.numa_balancing",
                value="1",
                description="Enable NUMA balancing",
                category="numa"
            ),
            SysctlParameter(
                name="vm.zone_reclaim_mode",
                value="0",
                description="NUMA zone reclaim",
                category="numa"
            ),
            SysctlParameter(
                name="kernel.sched_latency_ns",
                value="24000000",
                description="Scheduler latency for throughput",
                category="scheduler"
            )
        ]

    def _get_storage_params(self) -> List[SysctlParameter]:
        """Storage server optimization parameters"""
        return [
            SysctlParameter(
                name="vm.dirty_ratio",
                value="40",
                description="Dirty page cache ratio",
                category="memory"
            ),
            SysctlParameter(
                name="vm.dirty_background_ratio",
                value="10",
                description="Background dirty ratio",
                category="memory"
            ),
            SysctlParameter(
                name="vm.dirty_expire_centisecs",
                value="3000",
                description="Dirty data expiration",
                category="memory"
            ),
            SysctlParameter(
                name="vm.dirty_writeback_centisecs",
                value="500",
                description="Writeback interval",
                category="memory"
            ),
            SysctlParameter(
                name="fs.file-max",
                value="2097152",
                description="Maximum file handles",
                category="filesystem"
            ),
            SysctlParameter(
                name="fs.aio-max-nr",
                value="1048576",
                description="Maximum async I/O requests",
                category="filesystem"
            )
        ]

    def _get_network_params(self) -> List[SysctlParameter]:
        """Network-intensive workload parameters"""
        return [
            SysctlParameter(
                name="net.core.rmem_max",
                value="134217728",
                description="Maximum receive buffer",
                category="network"
            ),
            SysctlParameter(
                name="net.core.wmem_max",
                value="134217728",
                description="Maximum send buffer",
                category="network"
            ),
            SysctlParameter(
                name="net.ipv4.tcp_rmem",
                value="4096 87380 67108864",
                description="TCP receive buffer",
                category="network"
            ),
            SysctlParameter(
                name="net.ipv4.tcp_wmem",
                value="4096 65536 67108864",
                description="TCP send buffer",
                category="network"
            ),
            SysctlParameter(
                name="net.core.netdev_max_backlog",
                value="250000",
                description="Network device backlog",
                category="network"
            ),
            SysctlParameter(
                name="net.ipv4.tcp_congestion_control",
                value="bbr",
                description="TCP congestion control algorithm",
                category="network"
            ),
            SysctlParameter(
                name="net.core.default_qdisc",
                value="fq",
                description="Default queueing discipline",
                category="network"
            )
        ]

    def apply_profile(self, profile: TuningProfile) -> TuningResult:
        """Apply a complete tuning profile"""
        changes = []

        try:
            # Backup current configuration
            backup_file = self.sysctl_mgr.backup_current_config()
            if backup_file:
                changes.append(f"Backed up config to {backup_file}")

            # Get and apply profile parameters
            parameters = self.get_profile_parameters(profile)
            param_changes = self.sysctl_mgr.apply_profile_parameters(parameters)
            changes.extend(param_changes)

            # Apply special configurations based on profile
            if profile == TuningProfile.REALTIME:
                rt_changes = self.rt_mgr.configure_rt_parameters()
                changes.extend(rt_changes)

            if profile == TuningProfile.HPC and self.numa_mgr.numa_available:
                numa_changes = self.numa_mgr.optimize_numa_settings()
                changes.extend(numa_changes)

            return TuningResult(
                success=True,
                operation=f"Apply {profile.value} profile",
                details=f"Applied {len(changes)} changes",
                timestamp=datetime.now().isoformat(),
                changes=changes
            )
        except Exception as e:
            logger.error(f"Error applying profile: {e}")
            return TuningResult(
                success=False,
                operation=f"Apply {profile.value} profile",
                details=f"Error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                changes=changes
            )


class KernelTuningManager:
    """Main kernel tuning management system"""

    def __init__(self):
        self.sysctl_mgr = SysctlManager()
        self.module_mgr = KernelModuleManager()
        self.grub_mgr = GRUBManager()
        self.numa_mgr = NUMAManager()
        self.rt_mgr = RealtimeKernelManager()
        self.compile_mgr = KernelCompilationManager()
        self.profile_mgr = TuningProfileManager()

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            "kernel": {
                "version": os.uname().release,
                "architecture": os.uname().machine,
                "boot_parameters": self.grub_mgr.get_cmdline_parameters()
            },
            "realtime": self.rt_mgr.get_rt_info(),
            "numa": self.numa_mgr.get_numa_topology(),
            "resources": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available
            },
            "timestamp": datetime.now().isoformat()
        }
        return info

    def analyze_current_tuning(self) -> Dict[str, Any]:
        """Analyze current kernel tuning configuration"""
        analysis = {
            "sysctl_parameters": {},
            "loaded_modules": [],
            "recommendations": []
        }

        try:
            # Sample important sysctl parameters
            important_params = [
                "vm.swappiness",
                "vm.dirty_ratio",
                "net.core.somaxconn",
                "kernel.sched_latency_ns",
                "fs.file-max"
            ]

            for param in important_params:
                value = self.sysctl_mgr.get_parameter(param)
                if value:
                    analysis["sysctl_parameters"][param] = value

            # Get loaded modules
            result = subprocess.run(
                ['lsmod'],
                capture_output=True,
                text=True,
                check=True
            )

            module_lines = result.stdout.split('\n')[1:]  # Skip header
            for line in module_lines:
                if line.strip():
                    parts = line.split()
                    if parts:
                        analysis["loaded_modules"].append(parts[0])

            # Generate recommendations based on system type
            mem_total_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()

            if cpu_count >= 16:
                analysis["recommendations"].append(
                    "High CPU count detected: Consider HPC or NUMA tuning"
                )

            if mem_total_gb >= 64:
                analysis["recommendations"].append(
                    "Large memory detected: Consider database or HPC profile"
                )

            swappiness = int(analysis["sysctl_parameters"].get("vm.swappiness", 60))
            if swappiness > 30:
                analysis["recommendations"].append(
                    f"High swappiness ({swappiness}): Consider reducing for better performance"
                )

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing tuning: {e}")
            return analysis

    def execute(self,
                operation: str = "info",
                profile: Optional[TuningProfile] = None,
                **kwargs) -> Dict[str, Any]:
        """Execute kernel tuning operations"""

        result = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "data": {}
        }

        try:
            if operation == "info":
                result["data"] = self.get_system_info()
                result["success"] = True

            elif operation == "analyze":
                result["data"] = self.analyze_current_tuning()
                result["success"] = True

            elif operation == "apply_profile":
                if not profile:
                    result["error"] = "Profile not specified"
                    return result

                tuning_result = self.profile_mgr.apply_profile(profile)
                result["data"] = asdict(tuning_result)
                result["success"] = tuning_result.success

            elif operation == "set_parameter":
                param = kwargs.get("parameter")
                value = kwargs.get("value")
                if not param or not value:
                    result["error"] = "Parameter and value required"
                    return result

                success = self.sysctl_mgr.set_parameter(param, value)
                result["success"] = success
                result["data"] = {"parameter": param, "value": value}

            elif operation == "load_module":
                module_name = kwargs.get("module")
                if not module_name:
                    result["error"] = "Module name required"
                    return result

                module = KernelModule(
                    name=module_name,
                    parameters=kwargs.get("parameters", {}),
                    load_on_boot=kwargs.get("load_on_boot", True)
                )
                success = self.module_mgr.load_module(module)
                result["success"] = success
                result["data"] = {"module": module_name}

            elif operation == "numa_optimize":
                if not self.numa_mgr.numa_available:
                    result["error"] = "NUMA not available"
                    return result

                changes = self.numa_mgr.optimize_numa_settings()
                result["success"] = True
                result["data"] = {"changes": changes}

            elif operation == "rt_configure":
                changes = self.rt_mgr.configure_rt_parameters()
                result["success"] = True
                result["data"] = {
                    "is_rt_kernel": self.rt_mgr.is_rt_kernel,
                    "changes": changes
                }

            elif operation == "add_boot_param":
                param = kwargs.get("parameter")
                if not param:
                    result["error"] = "Boot parameter required"
                    return result

                success = self.grub_mgr.add_boot_parameter(param)
                result["success"] = success
                result["data"] = {"parameter": param}

            else:
                result["error"] = f"Unknown operation: {operation}"

            return result

        except Exception as e:
            logger.error(f"Error executing {operation}: {e}")
            result["error"] = str(e)
            return result


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Linux Kernel Tuning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get system information
  python kernel_tuning.py --operation info

  # Analyze current tuning
  python kernel_tuning.py --operation analyze

  # Apply web server profile
  python kernel_tuning.py --operation apply_profile --profile web_server

  # Apply database profile
  python kernel_tuning.py --operation apply_profile --profile database

  # Set specific parameter
  python kernel_tuning.py --operation set_parameter --param vm.swappiness --value 10

  # Optimize NUMA
  python kernel_tuning.py --operation numa_optimize

  # Configure real-time parameters
  python kernel_tuning.py --operation rt_configure
        """
    )

    parser.add_argument(
        '--operation',
        choices=['info', 'analyze', 'apply_profile', 'set_parameter',
                 'load_module', 'numa_optimize', 'rt_configure', 'add_boot_param'],
        default='info',
        help='Operation to perform'
    )

    parser.add_argument(
        '--profile',
        choices=['web_server', 'database', 'realtime', 'hpc', 'storage', 'network', 'desktop'],
        help='Tuning profile to apply'
    )

    parser.add_argument('--param', help='Parameter name')
    parser.add_argument('--value', help='Parameter value')
    parser.add_argument('--module', help='Module name')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')

    args = parser.parse_args()

    # Create manager
    manager = KernelTuningManager()

    # Prepare kwargs
    kwargs = {}
    if args.param:
        kwargs['parameter'] = args.param
    if args.value:
        kwargs['value'] = args.value
    if args.module:
        kwargs['module'] = args.module

    # Convert profile string to enum
    profile = None
    if args.profile:
        profile = TuningProfile(args.profile)

    # Execute operation
    result = manager.execute(
        operation=args.operation,
        profile=profile,
        **kwargs
    )

    # Output result
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nKernel Tuning System - {result['operation'].upper()}")
        print("=" * 60)
        print(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"Timestamp: {result['timestamp']}")

        if 'error' in result:
            print(f"\nError: {result['error']}")

        if result['data']:
            print("\nResults:")
            print(json.dumps(result['data'], indent=2))

    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
