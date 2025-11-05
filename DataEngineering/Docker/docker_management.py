"""
Docker Container Management
Complete Docker containerization and image management
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class DockerManager:
    """Comprehensive Docker container and image management"""

    def __init__(self, host: str = 'unix:///var/run/docker.sock'):
        """
        Initialize Docker manager

        Args:
            host: Docker daemon host
        """
        self.host = host
        self.containers = []
        self.images = []
        self.networks = []
        self.volumes = []
        self.services = []

    def build_image(self, build_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Docker image

        Args:
            build_config: Build configuration

        Returns:
            Image details
        """
        image = {
            'image_id': f"sha256:{'a' * 64}",
            'repository': build_config.get('repository', 'myapp'),
            'tag': build_config.get('tag', 'latest'),
            'dockerfile': build_config.get('dockerfile', 'Dockerfile'),
            'context': build_config.get('context', '.'),
            'build_args': build_config.get('build_args', {}),
            'labels': build_config.get('labels', {}),
            'size': build_config.get('size', 512000000),
            'layers': build_config.get('layers', 12),
            'created_at': datetime.now().isoformat()
        }

        image['full_name'] = f"{image['repository']}:{image['tag']}"
        self.images.append(image)
        print(f"✓ Image built: {image['full_name']}")
        print(f"  Size: {image['size'] / 1024 / 1024:.2f} MB, Layers: {image['layers']}")
        return image

    def create_dockerfile(self, dockerfile_config: Dict[str, Any]) -> str:
        """
        Generate Dockerfile

        Args:
            dockerfile_config: Dockerfile configuration

        Returns:
            Dockerfile content
        """
        base_image = dockerfile_config.get('base_image', 'python:3.11-slim')
        workdir = dockerfile_config.get('workdir', '/app')
        commands = dockerfile_config.get('commands', [])
        expose_ports = dockerfile_config.get('expose_ports', [8000])
        entrypoint = dockerfile_config.get('entrypoint', ['python', 'app.py'])

        dockerfile = f"""FROM {base_image}

WORKDIR {workdir}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

"""

        # Add custom commands
        for cmd in commands:
            dockerfile += f"RUN {cmd}\n"

        # Add exposed ports
        for port in expose_ports:
            dockerfile += f"\nEXPOSE {port}"

        # Add entrypoint
        dockerfile += f'\n\nENTRYPOINT {json.dumps(entrypoint)}'

        print(f"✓ Dockerfile generated")
        return dockerfile

    def create_container(self, container_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Docker container

        Args:
            container_config: Container configuration

        Returns:
            Container details
        """
        container = {
            'container_id': f"{'c' * 12}",
            'name': container_config.get('name', 'app-container'),
            'image': container_config.get('image', 'myapp:latest'),
            'command': container_config.get('command', None),
            'environment': container_config.get('environment', {}),
            'ports': container_config.get('ports', {}),
            'volumes': container_config.get('volumes', {}),
            'network': container_config.get('network', 'bridge'),
            'restart_policy': container_config.get('restart_policy', 'unless-stopped'),
            'labels': container_config.get('labels', {}),
            'state': 'running',
            'status': 'Up 2 seconds',
            'created_at': datetime.now().isoformat()
        }

        self.containers.append(container)
        print(f"✓ Container created: {container['name']}")
        print(f"  Image: {container['image']}, Network: {container['network']}")
        return container

    def create_docker_compose(self, compose_config: Dict[str, Any]) -> str:
        """
        Generate docker-compose.yml

        Args:
            compose_config: Compose configuration

        Returns:
            docker-compose.yml content
        """
        services = compose_config.get('services', {})
        networks = compose_config.get('networks', {})
        volumes = compose_config.get('volumes', {})

        compose = "version: '3.8'\n\n"
        compose += "services:\n"

        for service_name, service_config in services.items():
            compose += f"  {service_name}:\n"
            compose += f"    image: {service_config.get('image', '')}\n"

            if 'build' in service_config:
                compose += f"    build:\n"
                compose += f"      context: {service_config['build'].get('context', '.')}\n"
                compose += f"      dockerfile: {service_config['build'].get('dockerfile', 'Dockerfile')}\n"

            if 'ports' in service_config:
                compose += "    ports:\n"
                for port in service_config['ports']:
                    compose += f"      - \"{port}\"\n"

            if 'environment' in service_config:
                compose += "    environment:\n"
                for key, value in service_config['environment'].items():
                    compose += f"      - {key}={value}\n"

            if 'volumes' in service_config:
                compose += "    volumes:\n"
                for volume in service_config['volumes']:
                    compose += f"      - {volume}\n"

            if 'depends_on' in service_config:
                compose += "    depends_on:\n"
                for dep in service_config['depends_on']:
                    compose += f"      - {dep}\n"

            compose += "\n"

        if networks:
            compose += "networks:\n"
            for network_name, network_config in networks.items():
                compose += f"  {network_name}:\n"
                if network_config:
                    compose += f"    driver: {network_config.get('driver', 'bridge')}\n"

        if volumes:
            compose += "\nvolumes:\n"
            for volume_name in volumes:
                compose += f"  {volume_name}:\n"

        print(f"✓ docker-compose.yml generated")
        return compose

    def create_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Docker network

        Args:
            network_config: Network configuration

        Returns:
            Network details
        """
        network = {
            'network_id': f"{'n' * 12}",
            'name': network_config.get('name', 'app-network'),
            'driver': network_config.get('driver', 'bridge'),
            'scope': network_config.get('scope', 'local'),
            'subnet': network_config.get('subnet', '172.18.0.0/16'),
            'gateway': network_config.get('gateway', '172.18.0.1'),
            'attachable': network_config.get('attachable', True),
            'created_at': datetime.now().isoformat()
        }

        self.networks.append(network)
        print(f"✓ Network created: {network['name']} (Driver: {network['driver']})")
        return network

    def create_volume(self, volume_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Docker volume

        Args:
            volume_config: Volume configuration

        Returns:
            Volume details
        """
        volume = {
            'volume_id': f"{'v' * 64}",
            'name': volume_config.get('name', 'app-data'),
            'driver': volume_config.get('driver', 'local'),
            'mountpoint': volume_config.get('mountpoint', f"/var/lib/docker/volumes/{volume_config.get('name', 'app-data')}/_data"),
            'labels': volume_config.get('labels', {}),
            'options': volume_config.get('options', {}),
            'created_at': datetime.now().isoformat()
        }

        self.volumes.append(volume)
        print(f"✓ Volume created: {volume['name']}")
        return volume

    def push_image(self, image_name: str, registry: str = 'docker.io') -> Dict[str, Any]:
        """
        Push image to registry

        Args:
            image_name: Image name with tag
            registry: Registry URL

        Returns:
            Push details
        """
        result = {
            'image': image_name,
            'registry': registry,
            'full_name': f"{registry}/{image_name}",
            'status': 'PUSHED',
            'layers_pushed': 12,
            'size_pushed': 512000000,
            'pushed_at': datetime.now().isoformat()
        }

        print(f"✓ Image pushed: {result['full_name']}")
        print(f"  Layers: {result['layers_pushed']}, Size: {result['size_pushed'] / 1024 / 1024:.2f} MB")
        return result

    def create_multi_stage_dockerfile(self) -> str:
        """Generate multi-stage Dockerfile for optimized builds"""

        dockerfile = """# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary artifacts from builder
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

ENTRYPOINT ["python", "app.py"]
"""

        print("✓ Multi-stage Dockerfile generated")
        return dockerfile

    def start_container(self, container_name: str) -> Dict[str, Any]:
        """Start container"""
        container = next((c for c in self.containers if c['name'] == container_name), None)
        if container:
            container['state'] = 'running'
            print(f"✓ Container started: {container_name}")
            return container
        return {'error': 'Container not found'}

    def stop_container(self, container_name: str) -> Dict[str, Any]:
        """Stop container"""
        container = next((c for c in self.containers if c['name'] == container_name), None)
        if container:
            container['state'] = 'exited'
            print(f"✓ Container stopped: {container_name}")
            return container
        return {'error': 'Container not found'}

    def get_manager_info(self) -> Dict[str, Any]:
        """Get Docker manager information"""
        return {
            'host': self.host,
            'images': len(self.images),
            'containers': len(self.containers),
            'networks': len(self.networks),
            'volumes': len(self.volumes),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Docker container management"""

    print("=" * 60)
    print("Docker Container Management Demo")
    print("=" * 60)

    # Initialize manager
    docker = DockerManager()

    print("\n1. Generating Dockerfile...")
    dockerfile = docker.create_dockerfile({
        'base_image': 'python:3.11-slim',
        'workdir': '/app',
        'expose_ports': [8000, 8080],
        'entrypoint': ['python', 'main.py'],
        'commands': [
            'apt-get update && apt-get install -y curl',
            'pip install --upgrade pip'
        ]
    })
    print(dockerfile[:200] + "...")

    print("\n2. Building Docker image...")
    image = docker.build_image({
        'repository': 'myapp',
        'tag': 'v1.0.0',
        'dockerfile': 'Dockerfile',
        'context': '.',
        'build_args': {
            'VERSION': '1.0.0',
            'ENVIRONMENT': 'production'
        },
        'labels': {
            'maintainer': 'devops@example.com',
            'version': '1.0.0'
        }
    })

    print("\n3. Creating Docker network...")
    network = docker.create_network({
        'name': 'app-network',
        'driver': 'bridge',
        'subnet': '172.20.0.0/16',
        'gateway': '172.20.0.1'
    })

    print("\n4. Creating Docker volume...")
    volume = docker.create_volume({
        'name': 'app-data',
        'driver': 'local',
        'labels': {
            'environment': 'production',
            'backup': 'enabled'
        }
    })

    print("\n5. Creating container...")
    container = docker.create_container({
        'name': 'web-app',
        'image': 'myapp:v1.0.0',
        'environment': {
            'DATABASE_URL': 'postgresql://localhost:5432/db',
            'REDIS_URL': 'redis://localhost:6379',
            'LOG_LEVEL': 'INFO'
        },
        'ports': {
            '8000/tcp': 8000,
            '8080/tcp': 8080
        },
        'volumes': {
            'app-data': {'bind': '/app/data', 'mode': 'rw'}
        },
        'network': 'app-network',
        'restart_policy': 'unless-stopped'
    })

    print("\n6. Generating docker-compose.yml...")
    compose = docker.create_docker_compose({
        'services': {
            'web': {
                'build': {
                    'context': '.',
                    'dockerfile': 'Dockerfile'
                },
                'ports': ['8000:8000'],
                'environment': {
                    'DATABASE_URL': 'postgresql://db:5432/mydb',
                    'REDIS_URL': 'redis://redis:6379'
                },
                'volumes': ['./app:/app', 'static-data:/app/static'],
                'depends_on': ['db', 'redis']
            },
            'db': {
                'image': 'postgres:15',
                'environment': {
                    'POSTGRES_DB': 'mydb',
                    'POSTGRES_USER': 'user',
                    'POSTGRES_PASSWORD': 'password'
                },
                'volumes': ['postgres-data:/var/lib/postgresql/data']
            },
            'redis': {
                'image': 'redis:7-alpine',
                'ports': ['6379:6379']
            }
        },
        'volumes': ['postgres-data', 'static-data'],
        'networks': {
            'default': {'driver': 'bridge'}
        }
    })
    print(compose[:300] + "...")

    print("\n7. Generating multi-stage Dockerfile...")
    multi_stage_df = docker.create_multi_stage_dockerfile()
    print(multi_stage_df[:250] + "...")

    print("\n8. Pushing image to registry...")
    push_result = docker.push_image('myapp:v1.0.0', 'registry.example.com')

    print("\n9. Creating additional containers...")
    docker.create_container({
        'name': 'worker-1',
        'image': 'myapp:v1.0.0',
        'command': ['python', 'worker.py'],
        'network': 'app-network'
    })

    docker.create_container({
        'name': 'worker-2',
        'image': 'myapp:v1.0.0',
        'command': ['python', 'worker.py'],
        'network': 'app-network'
    })

    print("\n10. Docker manager summary:")
    info = docker.get_manager_info()
    print(f"  Images: {info['images']}")
    print(f"  Containers: {info['containers']}")
    print(f"  Networks: {info['networks']}")
    print(f"  Volumes: {info['volumes']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
