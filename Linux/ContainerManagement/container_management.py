"""
ContainerManagement - Production-Ready Container Orchestration System
Author: BrillConsulting
Description: Comprehensive Docker and Podman management with lifecycle, networking,
            volumes, compose orchestration, security scanning, and monitoring
Version: 2.0.0
"""

import subprocess
import json
import yaml
import shlex
import logging
import psutil
import time
import re
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContainerRuntime(Enum):
    """Supported container runtimes"""
    DOCKER = "docker"
    PODMAN = "podman"


class ContainerState(Enum):
    """Container lifecycle states"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"


class HealthStatus(Enum):
    """Container health check status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    NONE = "none"


@dataclass
class ResourceLimits:
    """Container resource constraints"""
    cpu_shares: Optional[int] = None  # CPU shares (relative weight)
    cpu_period: Optional[int] = None  # CPU CFS (Completely Fair Scheduler) period
    cpu_quota: Optional[int] = None   # CPU CFS quota
    cpuset_cpus: Optional[str] = None  # CPUs in which to allow execution (0-3, 0,1)
    memory: Optional[str] = None       # Memory limit (e.g., "512m", "2g")
    memory_swap: Optional[str] = None  # Total memory (memory + swap)
    memory_reservation: Optional[str] = None  # Memory soft limit
    kernel_memory: Optional[str] = None  # Kernel memory limit
    pids_limit: Optional[int] = None   # Maximum number of PIDs

    def to_docker_args(self) -> List[str]:
        """Convert to Docker CLI arguments"""
        args = []
        if self.cpu_shares:
            args.extend(['--cpu-shares', str(self.cpu_shares)])
        if self.cpu_period:
            args.extend(['--cpu-period', str(self.cpu_period)])
        if self.cpu_quota:
            args.extend(['--cpu-quota', str(self.cpu_quota)])
        if self.cpuset_cpus:
            args.extend(['--cpuset-cpus', self.cpuset_cpus])
        if self.memory:
            args.extend(['--memory', self.memory])
        if self.memory_swap:
            args.extend(['--memory-swap', self.memory_swap])
        if self.memory_reservation:
            args.extend(['--memory-reservation', self.memory_reservation])
        if self.kernel_memory:
            args.extend(['--kernel-memory', self.kernel_memory])
        if self.pids_limit:
            args.extend(['--pids-limit', str(self.pids_limit)])
        return args


@dataclass
class HealthCheck:
    """Container health check configuration"""
    test: List[str]  # Health check command
    interval: str = "30s"  # Time between checks
    timeout: str = "30s"   # Timeout for each check
    retries: int = 3       # Consecutive failures needed to be unhealthy
    start_period: str = "0s"  # Start period before first check

    def to_docker_args(self) -> List[str]:
        """Convert to Docker CLI arguments"""
        test_cmd = ' '.join(self.test)
        return [
            '--health-cmd', test_cmd,
            '--health-interval', self.interval,
            '--health-timeout', self.timeout,
            '--health-retries', str(self.retries),
            '--health-start-period', self.start_period
        ]


@dataclass
class NetworkConfig:
    """Container network configuration"""
    network: str = "bridge"
    ip_address: Optional[str] = None
    hostname: Optional[str] = None
    dns: List[str] = field(default_factory=list)
    extra_hosts: Dict[str, str] = field(default_factory=dict)
    ports: Dict[str, str] = field(default_factory=dict)  # container_port: host_port

    def to_docker_args(self) -> List[str]:
        """Convert to Docker CLI arguments"""
        args = ['--network', self.network]
        if self.ip_address:
            args.extend(['--ip', self.ip_address])
        if self.hostname:
            args.extend(['--hostname', self.hostname])
        for dns in self.dns:
            args.extend(['--dns', dns])
        for host, ip in self.extra_hosts.items():
            args.extend(['--add-host', f"{host}:{ip}"])
        for container_port, host_port in self.ports.items():
            args.extend(['-p', f"{host_port}:{container_port}"])
        return args


@dataclass
class VolumeMount:
    """Volume mount configuration"""
    source: str  # Host path or volume name
    target: str  # Container path
    read_only: bool = False
    volume_type: str = "bind"  # bind, volume, tmpfs

    def to_docker_arg(self) -> str:
        """Convert to Docker mount argument"""
        options = []
        if self.read_only:
            options.append("ro")

        if self.volume_type == "bind":
            return f"{self.source}:{self.target}" + (",ro" if self.read_only else "")
        elif self.volume_type == "volume":
            return f"{self.source}:{self.target}" + (",ro" if self.read_only else "")
        elif self.volume_type == "tmpfs":
            return self.target
        return f"{self.source}:{self.target}"


@dataclass
class ContainerConfig:
    """Complete container configuration"""
    name: str
    image: str
    command: Optional[List[str]] = None
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[VolumeMount] = field(default_factory=list)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    health_check: Optional[HealthCheck] = None
    restart_policy: str = "no"  # no, always, on-failure, unless-stopped
    detach: bool = True
    remove: bool = False
    privileged: bool = False
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class ContainerStats:
    """Container resource usage statistics"""
    container_id: str
    cpu_percent: float
    memory_usage: int
    memory_limit: int
    memory_percent: float
    network_rx_bytes: int
    network_tx_bytes: int
    block_read_bytes: int
    block_write_bytes: int
    pids: int
    timestamp: datetime = field(default_factory=datetime.now)


class ContainerRuntimeManager:
    """Base class for container runtime operations"""

    def __init__(self, runtime: ContainerRuntime):
        self.runtime = runtime
        self.runtime_cmd = runtime.value
        self._check_runtime_available()

    def _check_runtime_available(self):
        """Verify container runtime is installed"""
        try:
            result = subprocess.run(
                [self.runtime_cmd, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"{self.runtime.value} is not available")
            logger.info(f"Using {self.runtime.value}: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(f"{self.runtime.value} is not installed")

    def _execute_command(self, args: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Execute container runtime command"""
        cmd = [self.runtime_cmd] + args
        logger.debug(f"Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s")
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout}s',
                'returncode': -1
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }


class ContainerLifecycleManager(ContainerRuntimeManager):
    """Manages container lifecycle operations"""

    def create(self, config: ContainerConfig) -> Dict[str, Any]:
        """Create a new container"""
        args = ['create', '--name', config.name]

        # Add restart policy
        args.extend(['--restart', config.restart_policy])

        # Add environment variables
        for key, value in config.environment.items():
            args.extend(['-e', f"{key}={value}"])

        # Add volumes
        for volume in config.volumes:
            if volume.volume_type == "tmpfs":
                args.extend(['--tmpfs', volume.target])
            else:
                args.extend(['-v', volume.to_docker_arg()])

        # Add network configuration
        args.extend(config.network.to_docker_args())

        # Add resource limits
        args.extend(config.resources.to_docker_args())

        # Add health check
        if config.health_check:
            args.extend(config.health_check.to_docker_args())

        # Add labels
        for key, value in config.labels.items():
            args.extend(['--label', f"{key}={value}"])

        # Add privileged mode
        if config.privileged:
            args.append('--privileged')

        # Add image
        args.append(config.image)

        # Add command
        if config.command:
            args.extend(config.command)

        result = self._execute_command(args)

        if result['success']:
            container_id = result['stdout']
            logger.info(f"Container created: {config.name} ({container_id[:12]})")
            return {
                'success': True,
                'container_id': container_id,
                'name': config.name
            }
        else:
            logger.error(f"Failed to create container: {result['stderr']}")
            return {
                'success': False,
                'error': result['stderr']
            }

    def run(self, config: ContainerConfig) -> Dict[str, Any]:
        """Run a container (create and start)"""
        args = ['run']

        if config.detach:
            args.append('-d')

        if config.remove:
            args.append('--rm')

        args.extend(['--name', config.name])
        args.extend(['--restart', config.restart_policy])

        for key, value in config.environment.items():
            args.extend(['-e', f"{key}={value}"])

        for volume in config.volumes:
            if volume.volume_type == "tmpfs":
                args.extend(['--tmpfs', volume.target])
            else:
                args.extend(['-v', volume.to_docker_arg()])

        args.extend(config.network.to_docker_args())
        args.extend(config.resources.to_docker_args())

        if config.health_check:
            args.extend(config.health_check.to_docker_args())

        for key, value in config.labels.items():
            args.extend(['--label', f"{key}={value}"])

        if config.privileged:
            args.append('--privileged')

        args.append(config.image)

        if config.command:
            args.extend(config.command)

        result = self._execute_command(args, timeout=60)

        if result['success']:
            container_id = result['stdout']
            logger.info(f"Container started: {config.name} ({container_id[:12]})")
            return {
                'success': True,
                'container_id': container_id,
                'name': config.name
            }
        else:
            logger.error(f"Failed to run container: {result['stderr']}")
            return {
                'success': False,
                'error': result['stderr']
            }

    def start(self, container: str) -> Dict[str, Any]:
        """Start a stopped container"""
        result = self._execute_command(['start', container])

        if result['success']:
            logger.info(f"Container started: {container}")
            return {'success': True, 'container': container}
        else:
            logger.error(f"Failed to start container: {result['stderr']}")
            return {'success': False, 'error': result['stderr']}

    def stop(self, container: str, timeout: int = 10) -> Dict[str, Any]:
        """Stop a running container"""
        result = self._execute_command(['stop', '-t', str(timeout), container])

        if result['success']:
            logger.info(f"Container stopped: {container}")
            return {'success': True, 'container': container}
        else:
            logger.error(f"Failed to stop container: {result['stderr']}")
            return {'success': False, 'error': result['stderr']}

    def restart(self, container: str, timeout: int = 10) -> Dict[str, Any]:
        """Restart a container"""
        result = self._execute_command(['restart', '-t', str(timeout), container])

        if result['success']:
            logger.info(f"Container restarted: {container}")
            return {'success': True, 'container': container}
        else:
            logger.error(f"Failed to restart container: {result['stderr']}")
            return {'success': False, 'error': result['stderr']}

    def pause(self, container: str) -> Dict[str, Any]:
        """Pause a running container"""
        result = self._execute_command(['pause', container])

        if result['success']:
            logger.info(f"Container paused: {container}")
            return {'success': True, 'container': container}
        else:
            return {'success': False, 'error': result['stderr']}

    def unpause(self, container: str) -> Dict[str, Any]:
        """Unpause a paused container"""
        result = self._execute_command(['unpause', container])

        if result['success']:
            logger.info(f"Container unpaused: {container}")
            return {'success': True, 'container': container}
        else:
            return {'success': False, 'error': result['stderr']}

    def remove(self, container: str, force: bool = False, volumes: bool = False) -> Dict[str, Any]:
        """Remove a container"""
        args = ['rm']
        if force:
            args.append('-f')
        if volumes:
            args.append('-v')
        args.append(container)

        result = self._execute_command(args)

        if result['success']:
            logger.info(f"Container removed: {container}")
            return {'success': True, 'container': container}
        else:
            logger.error(f"Failed to remove container: {result['stderr']}")
            return {'success': False, 'error': result['stderr']}

    def kill(self, container: str, signal: str = "SIGKILL") -> Dict[str, Any]:
        """Kill a running container"""
        result = self._execute_command(['kill', '-s', signal, container])

        if result['success']:
            logger.info(f"Container killed: {container}")
            return {'success': True, 'container': container}
        else:
            return {'success': False, 'error': result['stderr']}

    def list_containers(self, all_containers: bool = False) -> Dict[str, Any]:
        """List containers"""
        args = ['ps', '--format', 'json']
        if all_containers:
            args.append('-a')

        result = self._execute_command(args)

        if result['success']:
            try:
                # Handle both single JSON object and newline-delimited JSON
                containers = []
                if result['stdout']:
                    for line in result['stdout'].split('\n'):
                        if line.strip():
                            containers.append(json.loads(line))

                return {
                    'success': True,
                    'containers': containers,
                    'count': len(containers)
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse container list: {e}")
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': result['stderr']}

    def inspect(self, container: str) -> Dict[str, Any]:
        """Get detailed container information"""
        result = self._execute_command(['inspect', container])

        if result['success']:
            try:
                info = json.loads(result['stdout'])
                return {
                    'success': True,
                    'info': info[0] if isinstance(info, list) else info
                }
            except json.JSONDecodeError as e:
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': result['stderr']}

    def logs(self, container: str, tail: int = 100, follow: bool = False) -> Dict[str, Any]:
        """Get container logs"""
        args = ['logs']
        if tail:
            args.extend(['--tail', str(tail)])
        if follow:
            args.append('-f')
        args.append(container)

        result = self._execute_command(args, timeout=60)

        if result['success']:
            return {
                'success': True,
                'logs': result['stdout']
            }
        else:
            return {'success': False, 'error': result['stderr']}

    def exec(self, container: str, command: List[str], interactive: bool = False) -> Dict[str, Any]:
        """Execute command in running container"""
        args = ['exec']
        if interactive:
            args.append('-it')
        args.append(container)
        args.extend(command)

        result = self._execute_command(args)

        if result['success']:
            return {
                'success': True,
                'output': result['stdout']
            }
        else:
            return {'success': False, 'error': result['stderr']}


class ImageManager(ContainerRuntimeManager):
    """Manages container images"""

    def pull(self, image: str, tag: str = "latest") -> Dict[str, Any]:
        """Pull an image from registry"""
        full_image = f"{image}:{tag}"
        result = self._execute_command(['pull', full_image], timeout=300)

        if result['success']:
            logger.info(f"Image pulled: {full_image}")
            return {'success': True, 'image': full_image}
        else:
            logger.error(f"Failed to pull image: {result['stderr']}")
            return {'success': False, 'error': result['stderr']}

    def push(self, image: str, tag: str = "latest") -> Dict[str, Any]:
        """Push an image to registry"""
        full_image = f"{image}:{tag}"
        result = self._execute_command(['push', full_image], timeout=300)

        if result['success']:
            logger.info(f"Image pushed: {full_image}")
            return {'success': True, 'image': full_image}
        else:
            return {'success': False, 'error': result['stderr']}

    def build(self, path: str, tag: str, dockerfile: str = "Dockerfile",
              build_args: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Build an image from Dockerfile"""
        args = ['build', '-t', tag, '-f', dockerfile]

        if build_args:
            for key, value in build_args.items():
                args.extend(['--build-arg', f"{key}={value}"])

        args.append(path)

        result = self._execute_command(args, timeout=600)

        if result['success']:
            logger.info(f"Image built: {tag}")
            return {'success': True, 'tag': tag}
        else:
            logger.error(f"Failed to build image: {result['stderr']}")
            return {'success': False, 'error': result['stderr']}

    def tag(self, source_image: str, target_image: str) -> Dict[str, Any]:
        """Tag an image"""
        result = self._execute_command(['tag', source_image, target_image])

        if result['success']:
            logger.info(f"Image tagged: {source_image} -> {target_image}")
            return {'success': True, 'source': source_image, 'target': target_image}
        else:
            return {'success': False, 'error': result['stderr']}

    def remove(self, image: str, force: bool = False) -> Dict[str, Any]:
        """Remove an image"""
        args = ['rmi']
        if force:
            args.append('-f')
        args.append(image)

        result = self._execute_command(args)

        if result['success']:
            logger.info(f"Image removed: {image}")
            return {'success': True, 'image': image}
        else:
            return {'success': False, 'error': result['stderr']}

    def list_images(self) -> Dict[str, Any]:
        """List all images"""
        result = self._execute_command(['images', '--format', 'json'])

        if result['success']:
            try:
                images = []
                if result['stdout']:
                    for line in result['stdout'].split('\n'):
                        if line.strip():
                            images.append(json.loads(line))

                return {
                    'success': True,
                    'images': images,
                    'count': len(images)
                }
            except json.JSONDecodeError as e:
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': result['stderr']}

    def inspect(self, image: str) -> Dict[str, Any]:
        """Get detailed image information"""
        result = self._execute_command(['inspect', image])

        if result['success']:
            try:
                info = json.loads(result['stdout'])
                return {
                    'success': True,
                    'info': info[0] if isinstance(info, list) else info
                }
            except json.JSONDecodeError as e:
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': result['stderr']}

    def prune(self, all_images: bool = False) -> Dict[str, Any]:
        """Remove unused images"""
        args = ['image', 'prune', '-f']
        if all_images:
            args.append('-a')

        result = self._execute_command(args)

        if result['success']:
            logger.info("Unused images pruned")
            return {'success': True, 'output': result['stdout']}
        else:
            return {'success': False, 'error': result['stderr']}


class NetworkManager(ContainerRuntimeManager):
    """Manages container networks"""

    def create(self, name: str, driver: str = "bridge",
               subnet: Optional[str] = None, gateway: Optional[str] = None,
               options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Create a network"""
        args = ['network', 'create', '--driver', driver]

        if subnet:
            args.extend(['--subnet', subnet])
        if gateway:
            args.extend(['--gateway', gateway])
        if options:
            for key, value in options.items():
                args.extend(['--opt', f"{key}={value}"])

        args.append(name)

        result = self._execute_command(args)

        if result['success']:
            network_id = result['stdout']
            logger.info(f"Network created: {name} ({network_id[:12]})")
            return {
                'success': True,
                'network_id': network_id,
                'name': name
            }
        else:
            logger.error(f"Failed to create network: {result['stderr']}")
            return {'success': False, 'error': result['stderr']}

    def connect(self, network: str, container: str,
                ip_address: Optional[str] = None) -> Dict[str, Any]:
        """Connect container to network"""
        args = ['network', 'connect']
        if ip_address:
            args.extend(['--ip', ip_address])
        args.extend([network, container])

        result = self._execute_command(args)

        if result['success']:
            logger.info(f"Container {container} connected to network {network}")
            return {'success': True, 'network': network, 'container': container}
        else:
            return {'success': False, 'error': result['stderr']}

    def disconnect(self, network: str, container: str, force: bool = False) -> Dict[str, Any]:
        """Disconnect container from network"""
        args = ['network', 'disconnect']
        if force:
            args.append('-f')
        args.extend([network, container])

        result = self._execute_command(args)

        if result['success']:
            logger.info(f"Container {container} disconnected from network {network}")
            return {'success': True, 'network': network, 'container': container}
        else:
            return {'success': False, 'error': result['stderr']}

    def remove(self, network: str) -> Dict[str, Any]:
        """Remove a network"""
        result = self._execute_command(['network', 'rm', network])

        if result['success']:
            logger.info(f"Network removed: {network}")
            return {'success': True, 'network': network}
        else:
            return {'success': False, 'error': result['stderr']}

    def list_networks(self) -> Dict[str, Any]:
        """List all networks"""
        result = self._execute_command(['network', 'ls', '--format', 'json'])

        if result['success']:
            try:
                networks = []
                if result['stdout']:
                    for line in result['stdout'].split('\n'):
                        if line.strip():
                            networks.append(json.loads(line))

                return {
                    'success': True,
                    'networks': networks,
                    'count': len(networks)
                }
            except json.JSONDecodeError as e:
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': result['stderr']}

    def inspect(self, network: str) -> Dict[str, Any]:
        """Get detailed network information"""
        result = self._execute_command(['network', 'inspect', network])

        if result['success']:
            try:
                info = json.loads(result['stdout'])
                return {
                    'success': True,
                    'info': info[0] if isinstance(info, list) else info
                }
            except json.JSONDecodeError as e:
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': result['stderr']}

    def prune(self) -> Dict[str, Any]:
        """Remove unused networks"""
        result = self._execute_command(['network', 'prune', '-f'])

        if result['success']:
            logger.info("Unused networks pruned")
            return {'success': True, 'output': result['stdout']}
        else:
            return {'success': False, 'error': result['stderr']}


class VolumeManager(ContainerRuntimeManager):
    """Manages container volumes"""

    def create(self, name: str, driver: str = "local",
               options: Optional[Dict[str, str]] = None,
               labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Create a volume"""
        args = ['volume', 'create', '--driver', driver]

        if options:
            for key, value in options.items():
                args.extend(['--opt', f"{key}={value}"])

        if labels:
            for key, value in labels.items():
                args.extend(['--label', f"{key}={value}"])

        args.append(name)

        result = self._execute_command(args)

        if result['success']:
            logger.info(f"Volume created: {name}")
            return {'success': True, 'name': name}
        else:
            logger.error(f"Failed to create volume: {result['stderr']}")
            return {'success': False, 'error': result['stderr']}

    def remove(self, volume: str, force: bool = False) -> Dict[str, Any]:
        """Remove a volume"""
        args = ['volume', 'rm']
        if force:
            args.append('-f')
        args.append(volume)

        result = self._execute_command(args)

        if result['success']:
            logger.info(f"Volume removed: {volume}")
            return {'success': True, 'volume': volume}
        else:
            return {'success': False, 'error': result['stderr']}

    def list_volumes(self) -> Dict[str, Any]:
        """List all volumes"""
        result = self._execute_command(['volume', 'ls', '--format', 'json'])

        if result['success']:
            try:
                volumes = []
                if result['stdout']:
                    for line in result['stdout'].split('\n'):
                        if line.strip():
                            volumes.append(json.loads(line))

                return {
                    'success': True,
                    'volumes': volumes,
                    'count': len(volumes)
                }
            except json.JSONDecodeError as e:
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': result['stderr']}

    def inspect(self, volume: str) -> Dict[str, Any]:
        """Get detailed volume information"""
        result = self._execute_command(['volume', 'inspect', volume])

        if result['success']:
            try:
                info = json.loads(result['stdout'])
                return {
                    'success': True,
                    'info': info[0] if isinstance(info, list) else info
                }
            except json.JSONDecodeError as e:
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': result['stderr']}

    def prune(self) -> Dict[str, Any]:
        """Remove unused volumes"""
        result = self._execute_command(['volume', 'prune', '-f'])

        if result['success']:
            logger.info("Unused volumes pruned")
            return {'success': True, 'output': result['stdout']}
        else:
            return {'success': False, 'error': result['stderr']}


class ComposeOrchestrator:
    """Manages Docker Compose orchestration"""

    def __init__(self, compose_file: str = "docker-compose.yml"):
        self.compose_file = compose_file
        self._check_compose_available()

    def _check_compose_available(self):
        """Check if docker-compose or docker compose is available"""
        try:
            # Try docker compose (newer)
            result = subprocess.run(
                ['docker', 'compose', 'version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.compose_cmd = ['docker', 'compose']
                logger.info(f"Using docker compose: {result.stdout.strip()}")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            # Try docker-compose (older)
            result = subprocess.run(
                ['docker-compose', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.compose_cmd = ['docker-compose']
                logger.info(f"Using docker-compose: {result.stdout.strip()}")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        raise RuntimeError("Docker Compose is not available")

    def _execute_command(self, args: List[str], timeout: int = 60) -> Dict[str, Any]:
        """Execute compose command"""
        cmd = self.compose_cmd + ['-f', self.compose_file] + args
        logger.debug(f"Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout}s',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }

    def up(self, detach: bool = True, build: bool = False,
           scale: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Start services defined in compose file"""
        args = ['up']
        if detach:
            args.append('-d')
        if build:
            args.append('--build')
        if scale:
            for service, count in scale.items():
                args.extend(['--scale', f"{service}={count}"])

        result = self._execute_command(args, timeout=300)

        if result['success']:
            logger.info("Compose services started")
            return {'success': True, 'output': result['stdout']}
        else:
            logger.error(f"Failed to start compose services: {result['stderr']}")
            return {'success': False, 'error': result['stderr']}

    def down(self, volumes: bool = False, remove_orphans: bool = False) -> Dict[str, Any]:
        """Stop and remove services"""
        args = ['down']
        if volumes:
            args.append('-v')
        if remove_orphans:
            args.append('--remove-orphans')

        result = self._execute_command(args)

        if result['success']:
            logger.info("Compose services stopped")
            return {'success': True, 'output': result['stdout']}
        else:
            return {'success': False, 'error': result['stderr']}

    def start(self) -> Dict[str, Any]:
        """Start existing services"""
        result = self._execute_command(['start'])

        if result['success']:
            logger.info("Compose services started")
            return {'success': True, 'output': result['stdout']}
        else:
            return {'success': False, 'error': result['stderr']}

    def stop(self, timeout: int = 10) -> Dict[str, Any]:
        """Stop running services"""
        result = self._execute_command(['stop', '-t', str(timeout)])

        if result['success']:
            logger.info("Compose services stopped")
            return {'success': True, 'output': result['stdout']}
        else:
            return {'success': False, 'error': result['stderr']}

    def restart(self, timeout: int = 10) -> Dict[str, Any]:
        """Restart services"""
        result = self._execute_command(['restart', '-t', str(timeout)])

        if result['success']:
            logger.info("Compose services restarted")
            return {'success': True, 'output': result['stdout']}
        else:
            return {'success': False, 'error': result['stderr']}

    def ps(self) -> Dict[str, Any]:
        """List services"""
        result = self._execute_command(['ps', '--format', 'json'])

        if result['success']:
            try:
                services = []
                if result['stdout']:
                    # Handle different compose output formats
                    if result['stdout'].startswith('['):
                        services = json.loads(result['stdout'])
                    else:
                        for line in result['stdout'].split('\n'):
                            if line.strip():
                                services.append(json.loads(line))

                return {
                    'success': True,
                    'services': services,
                    'count': len(services)
                }
            except json.JSONDecodeError as e:
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': result['stderr']}

    def logs(self, service: Optional[str] = None, tail: int = 100,
             follow: bool = False) -> Dict[str, Any]:
        """Get service logs"""
        args = ['logs']
        if tail:
            args.extend(['--tail', str(tail)])
        if follow:
            args.append('-f')
        if service:
            args.append(service)

        result = self._execute_command(args, timeout=60)

        if result['success']:
            return {'success': True, 'logs': result['stdout']}
        else:
            return {'success': False, 'error': result['stderr']}

    def build(self, no_cache: bool = False) -> Dict[str, Any]:
        """Build or rebuild services"""
        args = ['build']
        if no_cache:
            args.append('--no-cache')

        result = self._execute_command(args, timeout=600)

        if result['success']:
            logger.info("Compose services built")
            return {'success': True, 'output': result['stdout']}
        else:
            return {'success': False, 'error': result['stderr']}

    def pull(self) -> Dict[str, Any]:
        """Pull service images"""
        result = self._execute_command(['pull'], timeout=300)

        if result['success']:
            logger.info("Compose images pulled")
            return {'success': True, 'output': result['stdout']}
        else:
            return {'success': False, 'error': result['stderr']}


class SecurityScanner:
    """Security vulnerability scanning for images"""

    def __init__(self):
        self.scanner = self._detect_scanner()

    def _detect_scanner(self) -> Optional[str]:
        """Detect available security scanner"""
        scanners = ['trivy', 'grype', 'snyk']

        for scanner in scanners:
            try:
                result = subprocess.run(
                    [scanner, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"Using security scanner: {scanner}")
                    return scanner
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        logger.warning("No security scanner available (trivy, grype, snyk)")
        return None

    def scan_image(self, image: str, severity: str = "HIGH,CRITICAL") -> Dict[str, Any]:
        """Scan image for vulnerabilities"""
        if not self.scanner:
            return {
                'success': False,
                'error': 'No security scanner available'
            }

        try:
            if self.scanner == 'trivy':
                cmd = ['trivy', 'image', '--severity', severity, '--format', 'json', image]
            elif self.scanner == 'grype':
                cmd = ['grype', image, '-o', 'json']
            elif self.scanner == 'snyk':
                cmd = ['snyk', 'container', 'test', image, '--json']
            else:
                return {'success': False, 'error': 'Unknown scanner'}

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.stdout:
                try:
                    scan_results = json.loads(result.stdout)
                    logger.info(f"Security scan completed for {image}")
                    return {
                        'success': True,
                        'image': image,
                        'scanner': self.scanner,
                        'results': scan_results
                    }
                except json.JSONDecodeError:
                    return {
                        'success': True,
                        'image': image,
                        'scanner': self.scanner,
                        'results': result.stdout
                    }
            else:
                return {
                    'success': False,
                    'error': result.stderr or 'Scan produced no output'
                }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Scan timed out'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_vulnerability_summary(self, scan_results: Dict[str, Any]) -> Dict[str, int]:
        """Extract vulnerability summary from scan results"""
        summary = defaultdict(int)

        try:
            if self.scanner == 'trivy':
                for result in scan_results.get('Results', []):
                    for vuln in result.get('Vulnerabilities', []):
                        severity = vuln.get('Severity', 'UNKNOWN')
                        summary[severity] += 1
            elif self.scanner == 'grype':
                for match in scan_results.get('matches', []):
                    severity = match.get('vulnerability', {}).get('severity', 'UNKNOWN')
                    summary[severity] += 1
            elif self.scanner == 'snyk':
                for vuln in scan_results.get('vulnerabilities', []):
                    severity = vuln.get('severity', 'unknown').upper()
                    summary[severity] += 1
        except Exception as e:
            logger.error(f"Failed to parse vulnerability summary: {e}")

        return dict(summary)


class MonitoringManager(ContainerRuntimeManager):
    """Container monitoring and metrics collection"""

    def get_container_stats(self, container: str) -> Optional[ContainerStats]:
        """Get real-time container statistics"""
        result = self._execute_command(['stats', '--no-stream', '--format', 'json', container])

        if result['success'] and result['stdout']:
            try:
                stats = json.loads(result['stdout'])

                # Parse CPU percentage
                cpu_str = stats.get('CPUPerc', '0%').rstrip('%')
                cpu_percent = float(cpu_str) if cpu_str else 0.0

                # Parse memory
                mem_usage_str = stats.get('MemUsage', '0B / 0B')
                mem_parts = mem_usage_str.split(' / ')
                memory_usage = self._parse_size(mem_parts[0])
                memory_limit = self._parse_size(mem_parts[1]) if len(mem_parts) > 1 else 0

                mem_perc_str = stats.get('MemPerc', '0%').rstrip('%')
                memory_percent = float(mem_perc_str) if mem_perc_str else 0.0

                # Parse network I/O
                net_io_str = stats.get('NetIO', '0B / 0B')
                net_parts = net_io_str.split(' / ')
                network_rx = self._parse_size(net_parts[0])
                network_tx = self._parse_size(net_parts[1]) if len(net_parts) > 1 else 0

                # Parse block I/O
                block_io_str = stats.get('BlockIO', '0B / 0B')
                block_parts = block_io_str.split(' / ')
                block_read = self._parse_size(block_parts[0])
                block_write = self._parse_size(block_parts[1]) if len(block_parts) > 1 else 0

                # Parse PIDs
                pids_str = stats.get('PIDs', '0')
                pids = int(pids_str) if pids_str.isdigit() else 0

                return ContainerStats(
                    container_id=stats.get('ID', container),
                    cpu_percent=cpu_percent,
                    memory_usage=memory_usage,
                    memory_limit=memory_limit,
                    memory_percent=memory_percent,
                    network_rx_bytes=network_rx,
                    network_tx_bytes=network_tx,
                    block_read_bytes=block_read,
                    block_write_bytes=block_write,
                    pids=pids
                )
            except Exception as e:
                logger.error(f"Failed to parse container stats: {e}")
                return None

        return None

    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes"""
        size_str = size_str.strip()
        if not size_str or size_str == '0B':
            return 0

        units = {
            'B': 1,
            'KB': 1024,
            'MB': 1024**2,
            'GB': 1024**3,
            'TB': 1024**4,
            'KiB': 1024,
            'MiB': 1024**2,
            'GiB': 1024**3,
            'TiB': 1024**4
        }

        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                try:
                    value = float(size_str[:-len(unit)])
                    return int(value * multiplier)
                except ValueError:
                    return 0

        try:
            return int(float(size_str))
        except ValueError:
            return 0

    def get_health_status(self, container: str) -> HealthStatus:
        """Get container health status"""
        inspect_result = self._execute_command(['inspect', container])

        if inspect_result['success']:
            try:
                info = json.loads(inspect_result['stdout'])
                container_info = info[0] if isinstance(info, list) else info

                state = container_info.get('State', {})
                health = state.get('Health', {})

                if health:
                    status = health.get('Status', '').lower()
                    if status == 'healthy':
                        return HealthStatus.HEALTHY
                    elif status == 'unhealthy':
                        return HealthStatus.UNHEALTHY
                    elif status == 'starting':
                        return HealthStatus.STARTING

                return HealthStatus.NONE
            except Exception as e:
                logger.error(f"Failed to get health status: {e}")
                return HealthStatus.NONE

        return HealthStatus.NONE

    def monitor_containers(self, interval: int = 5, duration: int = 60) -> List[ContainerStats]:
        """Monitor all running containers over time"""
        stats_collection = []
        end_time = time.time() + duration

        while time.time() < end_time:
            # Get list of running containers
            list_result = self._execute_command(['ps', '-q'])

            if list_result['success'] and list_result['stdout']:
                container_ids = list_result['stdout'].split('\n')

                for container_id in container_ids:
                    if container_id.strip():
                        stats = self.get_container_stats(container_id.strip())
                        if stats:
                            stats_collection.append(stats)

            time.sleep(interval)

        return stats_collection


class ContainerManagementSystem:
    """Main container management system"""

    def __init__(self, runtime: ContainerRuntime = ContainerRuntime.DOCKER):
        """Initialize container management system"""
        self.runtime = runtime
        logger.info(f"Initializing Container Management System with {runtime.value}")

        # Initialize managers
        self.lifecycle = ContainerLifecycleManager(runtime)
        self.images = ImageManager(runtime)
        self.networks = NetworkManager(runtime)
        self.volumes = VolumeManager(runtime)
        self.monitoring = MonitoringManager(runtime)
        self.security = SecurityScanner()

        # Compose orchestrator (optional)
        try:
            self.compose = ComposeOrchestrator()
        except RuntimeError:
            self.compose = None
            logger.warning("Docker Compose not available")

        logger.info("Container Management System initialized successfully")

    def get_system_info(self) -> Dict[str, Any]:
        """Get container runtime system information"""
        result = self.lifecycle._execute_command(['info', '--format', 'json'])

        if result['success']:
            try:
                info = json.loads(result['stdout'])
                return {
                    'success': True,
                    'runtime': self.runtime.value,
                    'info': info
                }
            except json.JSONDecodeError:
                return {
                    'success': True,
                    'runtime': self.runtime.value,
                    'info': result['stdout']
                }
        else:
            return {'success': False, 'error': result['stderr']}

    def deploy_application(self, config: ContainerConfig) -> Dict[str, Any]:
        """Deploy a complete application with all configurations"""
        logger.info(f"Deploying application: {config.name}")

        # Pull image if needed
        image_parts = config.image.split(':')
        image_name = image_parts[0]
        image_tag = image_parts[1] if len(image_parts) > 1 else 'latest'

        logger.info(f"Pulling image: {config.image}")
        pull_result = self.images.pull(image_name, image_tag)
        if not pull_result['success']:
            logger.warning(f"Failed to pull image: {pull_result.get('error')}")

        # Create volumes if needed
        for volume in config.volumes:
            if volume.volume_type == "volume":
                self.volumes.create(volume.source)

        # Create network if custom network specified
        if config.network.network != "bridge" and config.network.network != "host":
            self.networks.create(config.network.network)

        # Run container
        run_result = self.lifecycle.run(config)

        if run_result['success']:
            container_id = run_result['container_id']

            # Wait for container to be healthy (if health check configured)
            if config.health_check:
                logger.info("Waiting for health check...")
                max_wait = 60
                elapsed = 0
                while elapsed < max_wait:
                    health = self.monitoring.get_health_status(container_id)
                    if health == HealthStatus.HEALTHY:
                        logger.info("Container is healthy")
                        break
                    elif health == HealthStatus.UNHEALTHY:
                        logger.error("Container is unhealthy")
                        break
                    time.sleep(2)
                    elapsed += 2

            return {
                'success': True,
                'container_id': container_id,
                'name': config.name
            }
        else:
            return run_result

    def cleanup_resources(self, all_resources: bool = False) -> Dict[str, Any]:
        """Clean up unused container resources"""
        logger.info("Cleaning up unused resources")

        results = {
            'containers': None,
            'images': None,
            'volumes': None,
            'networks': None
        }

        # Remove stopped containers
        list_result = self.lifecycle.list_containers(all_containers=True)
        if list_result['success']:
            for container in list_result['containers']:
                state = container.get('State', '').lower()
                if state in ['exited', 'dead', 'created']:
                    self.lifecycle.remove(container.get('ID', container.get('Names', '')))

        # Prune images
        results['images'] = self.images.prune(all_images=all_resources)

        # Prune volumes
        results['volumes'] = self.volumes.prune()

        # Prune networks
        results['networks'] = self.networks.prune()

        logger.info("Resource cleanup completed")
        return {
            'success': True,
            'results': results
        }

    def export_metrics(self, output_file: str = "container_metrics.json") -> Dict[str, Any]:
        """Export container metrics to file"""
        logger.info("Collecting container metrics")

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'runtime': self.runtime.value,
            'containers': [],
            'images': [],
            'volumes': [],
            'networks': []
        }

        # Get container stats
        list_result = self.lifecycle.list_containers(all_containers=False)
        if list_result['success']:
            for container in list_result['containers']:
                container_id = container.get('ID', container.get('Names', ''))
                stats = self.monitoring.get_container_stats(container_id)
                if stats:
                    metrics['containers'].append(asdict(stats))

        # Get images
        images_result = self.images.list_images()
        if images_result['success']:
            metrics['images'] = images_result['images']

        # Get volumes
        volumes_result = self.volumes.list_volumes()
        if volumes_result['success']:
            metrics['volumes'] = volumes_result['volumes']

        # Get networks
        networks_result = self.networks.list_networks()
        if networks_result['success']:
            metrics['networks'] = networks_result['networks']

        # Write to file
        try:
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics exported to {output_file}")
            return {'success': True, 'file': output_file}
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return {'success': False, 'error': str(e)}


def demo_container_management():
    """Demonstration of container management capabilities"""
    print("=" * 80)
    print("Container Management System - Production Demo")
    print("=" * 80)

    try:
        # Initialize system
        cms = ContainerManagementSystem(ContainerRuntime.DOCKER)

        # Get system info
        print("\n[1] System Information:")
        info = cms.get_system_info()
        if info['success']:
            print(f"   Runtime: {info['runtime']}")
            print(f"   Status: Available")

        # List existing containers
        print("\n[2] Existing Containers:")
        containers = cms.lifecycle.list_containers(all_containers=True)
        if containers['success']:
            print(f"   Found {containers['count']} containers")
            for container in containers['containers'][:5]:
                name = container.get('Names', container.get('Name', 'N/A'))
                state = container.get('State', container.get('Status', 'N/A'))
                print(f"   - {name}: {state}")

        # List images
        print("\n[3] Available Images:")
        images = cms.images.list_images()
        if images['success']:
            print(f"   Found {images['count']} images")
            for image in images['images'][:5]:
                repo = image.get('Repository', 'N/A')
                tag = image.get('Tag', 'N/A')
                size = image.get('Size', 'N/A')
                print(f"   - {repo}:{tag} ({size})")

        # List volumes
        print("\n[4] Volumes:")
        volumes = cms.volumes.list_volumes()
        if volumes['success']:
            print(f"   Found {volumes['count']} volumes")

        # List networks
        print("\n[5] Networks:")
        networks = cms.networks.list_networks()
        if networks['success']:
            print(f"   Found {networks['count']} networks")
            for network in networks['networks'][:5]:
                name = network.get('Name', 'N/A')
                driver = network.get('Driver', 'N/A')
                print(f"   - {name} ({driver})")

        # Security scanning
        print("\n[6] Security Scanning:")
        if cms.security.scanner:
            print(f"   Scanner available: {cms.security.scanner}")
        else:
            print("   No security scanner installed")

        # Monitoring
        print("\n[7] Container Monitoring:")
        running = cms.lifecycle.list_containers(all_containers=False)
        if running['success'] and running['count'] > 0:
            container = running['containers'][0]
            container_id = container.get('ID', container.get('Names', ''))
            stats = cms.monitoring.get_container_stats(container_id)
            if stats:
                print(f"   Container: {container_id[:12]}")
                print(f"   CPU: {stats.cpu_percent:.2f}%")
                print(f"   Memory: {stats.memory_usage / 1024**2:.2f}MB / "
                      f"{stats.memory_limit / 1024**2:.2f}MB ({stats.memory_percent:.2f}%)")
                print(f"   Network RX: {stats.network_rx_bytes / 1024:.2f}KB")
                print(f"   Network TX: {stats.network_tx_bytes / 1024:.2f}KB")

        # Compose orchestration
        print("\n[8] Compose Orchestration:")
        if cms.compose:
            print("   Docker Compose available")
        else:
            print("   Docker Compose not available")

        print("\n" + "=" * 80)
        print("Container Management System operational")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    demo_container_management()
