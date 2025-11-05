"""
Azure Container Apps Service Integration
Author: BrillConsulting
Contact: clientbrill@gmail.com
LinkedIn: brillconsulting
Description: Advanced Container Apps implementation with deployment, scaling, revisions, and traffic management
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import json


class ScaleRuleType(Enum):
    """Scale rule types"""
    HTTP = "http"
    CPU = "cpu"
    MEMORY = "memory"
    CUSTOM = "custom"


class RevisionMode(Enum):
    """Revision activation modes"""
    SINGLE = "Single"
    MULTIPLE = "Multiple"


class IngressType(Enum):
    """Ingress types"""
    EXTERNAL = "External"
    INTERNAL = "Internal"


@dataclass
class Container:
    """Container configuration"""
    name: str
    image: str
    cpu: float
    memory: str
    env: List[Dict[str, str]] = field(default_factory=list)
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None


@dataclass
class ScaleRule:
    """Scaling rule configuration"""
    name: str
    rule_type: ScaleRuleType
    metadata: Dict[str, Any]
    min_replicas: int = 0
    max_replicas: int = 10


@dataclass
class Revision:
    """Container app revision"""
    name: str
    container_app: str
    created_at: str
    active: bool
    traffic_weight: int = 0
    replicas: int = 1
    containers: List[Container] = field(default_factory=list)


@dataclass
class IngressConfig:
    """Ingress configuration"""
    external: bool
    target_port: int
    transport: str = "auto"
    allow_insecure: bool = False
    traffic_weights: List[Dict[str, Any]] = field(default_factory=list)
    custom_domains: List[str] = field(default_factory=list)


@dataclass
class DaprConfig:
    """Dapr configuration"""
    enabled: bool
    app_id: str
    app_port: Optional[int] = None
    app_protocol: str = "http"


class ContainerAppManager:
    """
    Manage Azure Container Apps
    
    Features:
    - Container app deployment
    - Revision management
    - Auto-scaling configuration
    - Traffic splitting
    - Ingress configuration
    """
    
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        environment_name: str
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.environment_name = environment_name
        self.container_apps: Dict[str, Dict[str, Any]] = {}
        self.revisions: Dict[str, Dict[str, Revision]] = {}
    
    def create_container_app(
        self,
        name: str,
        containers: List[Container],
        ingress: Optional[IngressConfig] = None,
        dapr: Optional[DaprConfig] = None,
        scale_rules: Optional[List[ScaleRule]] = None,
        secrets: Optional[List[Dict[str, str]]] = None,
        revision_mode: RevisionMode = RevisionMode.SINGLE
    ) -> Dict[str, Any]:
        """
        Create container app
        
        Args:
            name: App name
            containers: Container configurations
            ingress: Ingress configuration
            dapr: Dapr configuration
            scale_rules: Scaling rules
            secrets: App secrets
            revision_mode: Revision mode
            
        Returns:
            Container app configuration
        """
        app = {
            "name": name,
            "environment": self.environment_name,
            "revision_mode": revision_mode.value,
            "containers": [asdict(c) for c in containers],
            "ingress": asdict(ingress) if ingress else None,
            "dapr": asdict(dapr) if dapr else None,
            "scale_rules": [asdict(r) for r in scale_rules] if scale_rules else [],
            "secrets": secrets or [],
            "created_at": datetime.now().isoformat(),
            "provisioning_state": "Succeeded"
        }
        
        self.container_apps[name] = app
        
        # Create initial revision
        revision = self._create_revision(name, containers)
        
        return app
    
    def _create_revision(
        self,
        container_app: str,
        containers: List[Container]
    ) -> Revision:
        """Create a new revision"""
        timestamp = datetime.now().timestamp()
        revision_name = f"{container_app}--{int(timestamp)}"
        
        revision = Revision(
            name=revision_name,
            container_app=container_app,
            created_at=datetime.now().isoformat(),
            active=True,
            traffic_weight=100,
            replicas=1,
            containers=containers
        )
        
        if container_app not in self.revisions:
            self.revisions[container_app] = {}
        
        self.revisions[container_app][revision_name] = revision
        
        return revision
    
    def update_container_app(
        self,
        name: str,
        containers: Optional[List[Container]] = None,
        ingress: Optional[IngressConfig] = None,
        scale_rules: Optional[List[ScaleRule]] = None
    ) -> Dict[str, Any]:
        """
        Update container app (creates new revision)
        
        Args:
            name: App name
            containers: Updated containers
            ingress: Updated ingress
            scale_rules: Updated scale rules
            
        Returns:
            Updated app configuration
        """
        if name not in self.container_apps:
            raise ValueError(f"Container app '{name}' not found")
        
        app = self.container_apps[name]
        
        if containers:
            app["containers"] = [asdict(c) for c in containers]
            
            # Create new revision
            if app["revision_mode"] == RevisionMode.MULTIPLE.value:
                self._create_revision(name, containers)
        
        if ingress:
            app["ingress"] = asdict(ingress)
        
        if scale_rules:
            app["scale_rules"] = [asdict(r) for r in scale_rules]
        
        app["updated_at"] = datetime.now().isoformat()
        
        return app
    
    def list_revisions(self, container_app: str) -> List[Revision]:
        """List all revisions for a container app"""
        if container_app not in self.revisions:
            return []
        
        return list(self.revisions[container_app].values())
    
    def get_revision(
        self,
        container_app: str,
        revision_name: str
    ) -> Optional[Revision]:
        """Get specific revision"""
        if container_app not in self.revisions:
            return None
        
        return self.revisions[container_app].get(revision_name)
    
    def activate_revision(
        self,
        container_app: str,
        revision_name: str
    ) -> Revision:
        """Activate a revision"""
        revision = self.get_revision(container_app, revision_name)
        if not revision:
            raise ValueError(f"Revision '{revision_name}' not found")
        
        revision.active = True
        return revision
    
    def deactivate_revision(
        self,
        container_app: str,
        revision_name: str
    ) -> Revision:
        """Deactivate a revision"""
        revision = self.get_revision(container_app, revision_name)
        if not revision:
            raise ValueError(f"Revision '{revision_name}' not found")
        
        revision.active = False
        revision.traffic_weight = 0
        return revision


class TrafficManager:
    """
    Manage traffic splitting between revisions
    
    Features:
    - A/B testing
    - Blue-green deployment
    - Canary releases
    - Traffic weight management
    """
    
    def __init__(self, container_app_manager: ContainerAppManager):
        self.app_manager = container_app_manager
    
    def split_traffic(
        self,
        container_app: str,
        traffic_weights: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Split traffic between revisions
        
        Args:
            container_app: Container app name
            traffic_weights: Dict of revision_name -> weight percentage
            
        Returns:
            Traffic configuration
        """
        # Validate weights sum to 100
        total = sum(traffic_weights.values())
        if total != 100:
            raise ValueError(f"Traffic weights must sum to 100, got {total}")
        
        # Update revision traffic weights
        for revision_name, weight in traffic_weights.items():
            revision = self.app_manager.get_revision(container_app, revision_name)
            if revision:
                revision.traffic_weight = weight
        
        return {
            "container_app": container_app,
            "traffic_weights": traffic_weights,
            "updated_at": datetime.now().isoformat()
        }
    
    def canary_deployment(
        self,
        container_app: str,
        current_revision: str,
        canary_revision: str,
        canary_percent: int = 10
    ) -> Dict[str, Any]:
        """
        Setup canary deployment
        
        Args:
            container_app: Container app name
            current_revision: Current stable revision
            canary_revision: New canary revision
            canary_percent: Percentage for canary (1-99)
            
        Returns:
            Canary configuration
        """
        if not 1 <= canary_percent <= 99:
            raise ValueError("Canary percent must be between 1 and 99")
        
        traffic_weights = {
            current_revision: 100 - canary_percent,
            canary_revision: canary_percent
        }
        
        return self.split_traffic(container_app, traffic_weights)
    
    def blue_green_deployment(
        self,
        container_app: str,
        blue_revision: str,
        green_revision: str,
        switch_to_green: bool = False
    ) -> Dict[str, Any]:
        """
        Setup blue-green deployment
        
        Args:
            container_app: Container app name
            blue_revision: Blue (current) revision
            green_revision: Green (new) revision
            switch_to_green: Switch traffic to green
            
        Returns:
            Blue-green configuration
        """
        if switch_to_green:
            traffic_weights = {
                blue_revision: 0,
                green_revision: 100
            }
        else:
            traffic_weights = {
                blue_revision: 100,
                green_revision: 0
            }
        
        return self.split_traffic(container_app, traffic_weights)


class ScalingManager:
    """
    Manage auto-scaling configuration
    
    Features:
    - HTTP-based scaling
    - CPU-based scaling
    - Memory-based scaling
    - Custom metric scaling
    """
    
    def __init__(self):
        pass
    
    def create_http_scale_rule(
        self,
        name: str,
        concurrent_requests: int,
        min_replicas: int = 0,
        max_replicas: int = 10
    ) -> ScaleRule:
        """
        Create HTTP-based scale rule
        
        Args:
            name: Rule name
            concurrent_requests: Requests per replica
            min_replicas: Minimum replicas
            max_replicas: Maximum replicas
            
        Returns:
            ScaleRule object
        """
        return ScaleRule(
            name=name,
            rule_type=ScaleRuleType.HTTP,
            metadata={"concurrentRequests": str(concurrent_requests)},
            min_replicas=min_replicas,
            max_replicas=max_replicas
        )
    
    def create_cpu_scale_rule(
        self,
        name: str,
        cpu_percentage: int,
        min_replicas: int = 1,
        max_replicas: int = 10
    ) -> ScaleRule:
        """
        Create CPU-based scale rule
        
        Args:
            name: Rule name
            cpu_percentage: CPU utilization threshold
            min_replicas: Minimum replicas
            max_replicas: Maximum replicas
            
        Returns:
            ScaleRule object
        """
        return ScaleRule(
            name=name,
            rule_type=ScaleRuleType.CPU,
            metadata={"type": "Utilization", "value": str(cpu_percentage)},
            min_replicas=min_replicas,
            max_replicas=max_replicas
        )
    
    def create_memory_scale_rule(
        self,
        name: str,
        memory_percentage: int,
        min_replicas: int = 1,
        max_replicas: int = 10
    ) -> ScaleRule:
        """
        Create memory-based scale rule
        
        Args:
            name: Rule name
            memory_percentage: Memory utilization threshold
            min_replicas: Minimum replicas
            max_replicas: Maximum replicas
            
        Returns:
            ScaleRule object
        """
        return ScaleRule(
            name=name,
            rule_type=ScaleRuleType.MEMORY,
            metadata={"type": "Utilization", "value": str(memory_percentage)},
            min_replicas=min_replicas,
            max_replicas=max_replicas
        )
    
    def create_custom_scale_rule(
        self,
        name: str,
        metric_type: str,
        metadata: Dict[str, str],
        min_replicas: int = 0,
        max_replicas: int = 10
    ) -> ScaleRule:
        """
        Create custom metric scale rule
        
        Args:
            name: Rule name
            metric_type: Custom metric type
            metadata: Scale rule metadata
            min_replicas: Minimum replicas
            max_replicas: Maximum replicas
            
        Returns:
            ScaleRule object
        """
        return ScaleRule(
            name=name,
            rule_type=ScaleRuleType.CUSTOM,
            metadata={**metadata, "type": metric_type},
            min_replicas=min_replicas,
            max_replicas=max_replicas
        )


class DaprManager:
    """
    Manage Dapr integration
    
    Features:
    - Service-to-service invocation
    - State management
    - Pub/sub messaging
    - Bindings
    """
    
    def __init__(self):
        self.dapr_components: Dict[str, Dict[str, Any]] = {}
    
    def create_state_store_component(
        self,
        name: str,
        component_type: str,
        metadata: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Create Dapr state store component
        
        Args:
            name: Component name
            component_type: State store type (e.g., state.redis)
            metadata: Component metadata
            
        Returns:
            Component configuration
        """
        component = {
            "name": name,
            "type": component_type,
            "version": "v1",
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
        
        self.dapr_components[name] = component
        return component
    
    def create_pubsub_component(
        self,
        name: str,
        component_type: str,
        metadata: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Create Dapr pub/sub component
        
        Args:
            name: Component name
            component_type: Pub/sub type (e.g., pubsub.azure.servicebus)
            metadata: Component metadata
            
        Returns:
            Component configuration
        """
        component = {
            "name": name,
            "type": component_type,
            "version": "v1",
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
        
        self.dapr_components[name] = component
        return component


class ContainerRegistryManager:
    """
    Manage container registry integration
    
    Features:
    - ACR authentication
    - Image pull secrets
    - Private registry support
    """
    
    def __init__(self):
        self.registries: Dict[str, Dict[str, Any]] = {}
    
    def add_registry(
        self,
        registry_url: str,
        username: str,
        password: str
    ) -> Dict[str, Any]:
        """
        Add container registry
        
        Args:
            registry_url: Registry URL
            username: Registry username
            password: Registry password
            
        Returns:
            Registry configuration
        """
        registry = {
            "url": registry_url,
            "username": username,
            "password_secret": "registry-password",
            "added_at": datetime.now().isoformat()
        }
        
        self.registries[registry_url] = registry
        return registry
    
    def create_image_pull_secret(
        self,
        secret_name: str,
        registry_url: str
    ) -> Dict[str, Any]:
        """
        Create image pull secret
        
        Args:
            secret_name: Secret name
            registry_url: Registry URL
            
        Returns:
            Secret configuration
        """
        if registry_url not in self.registries:
            raise ValueError(f"Registry '{registry_url}' not configured")
        
        return {
            "name": secret_name,
            "registry": registry_url,
            "type": "image_pull_secret",
            "created_at": datetime.now().isoformat()
        }


class EnvironmentManager:
    """
    Manage Container Apps Environment
    
    Features:
    - Environment configuration
    - Virtual network integration
    - Log Analytics workspace
    - Custom domains
    """
    
    def __init__(self):
        self.environments: Dict[str, Dict[str, Any]] = {}
    
    def create_environment(
        self,
        name: str,
        location: str,
        vnet_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        dapr_ai_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create Container Apps Environment
        
        Args:
            name: Environment name
            location: Azure region
            vnet_id: Virtual network ID
            workspace_id: Log Analytics workspace ID
            dapr_ai_key: Dapr Application Insights key
            
        Returns:
            Environment configuration
        """
        environment = {
            "name": name,
            "location": location,
            "vnet_id": vnet_id,
            "workspace_id": workspace_id,
            "dapr_ai_key": dapr_ai_key,
            "provisioning_state": "Succeeded",
            "created_at": datetime.now().isoformat()
        }
        
        self.environments[name] = environment
        return environment
    
    def add_custom_domain(
        self,
        environment_name: str,
        domain_name: str,
        certificate_id: str
    ) -> Dict[str, Any]:
        """
        Add custom domain to environment
        
        Args:
            environment_name: Environment name
            domain_name: Custom domain
            certificate_id: Certificate resource ID
            
        Returns:
            Domain configuration
        """
        if environment_name not in self.environments:
            raise ValueError(f"Environment '{environment_name}' not found")
        
        return {
            "environment": environment_name,
            "domain": domain_name,
            "certificate_id": certificate_id,
            "added_at": datetime.now().isoformat()
        }


# Demo functions
def demo_container_app_deployment():
    """Demonstrate container app deployment"""
    print("=== Container App Deployment Demo ===\n")
    
    manager = ContainerAppManager("sub-123", "my-rg", "my-environment")
    
    # Create container configuration
    container = Container(
        name="api",
        image="myregistry.azurecr.io/myapp:latest",
        cpu=0.5,
        memory="1Gi",
        env=[
            {"name": "DATABASE_URL", "value": "postgresql://..."},
            {"name": "API_KEY", "secretRef": "api-key"}
        ]
    )
    
    # Create ingress configuration
    ingress = IngressConfig(
        external=True,
        target_port=8080,
        transport="auto"
    )
    
    # Create container app
    app = manager.create_container_app(
        "my-api",
        containers=[container],
        ingress=ingress,
        revision_mode=RevisionMode.MULTIPLE
    )
    
    print(f"Created container app: {app['name']}")
    print(f"Environment: {app['environment']}")
    print(f"Revision mode: {app['revision_mode']}")
    print(f"Status: {app['provisioning_state']}\n")
    
    # List revisions
    revisions = manager.list_revisions("my-api")
    print(f"Revisions: {len(revisions)}")
    for rev in revisions:
        print(f"  - {rev.name}: Active={rev.active}, Traffic={rev.traffic_weight}%\n")


def demo_traffic_management():
    """Demonstrate traffic splitting"""
    print("=== Traffic Management Demo ===\n")
    
    app_manager = ContainerAppManager("sub-123", "my-rg", "my-environment")
    traffic_manager = TrafficManager(app_manager)
    
    # Create app with containers
    container = Container("app", "myapp:v1", 0.5, "1Gi")
    app_manager.create_container_app("web-app", [container])
    
    # Simulate two revisions
    rev1_name = list(app_manager.revisions["web-app"].keys())[0]
    container_v2 = Container("app", "myapp:v2", 0.5, "1Gi")
    app_manager.update_container_app("web-app", [container_v2])
    rev2_name = list(app_manager.revisions["web-app"].keys())[1]
    
    # Canary deployment
    canary = traffic_manager.canary_deployment(
        "web-app",
        rev1_name,
        rev2_name,
        canary_percent=10
    )
    
    print(f"Canary deployment configured:")
    print(f"  Current revision: {canary['traffic_weights'][rev1_name]}%")
    print(f"  Canary revision: {canary['traffic_weights'][rev2_name]}%\n")
    
    # Blue-green deployment
    blue_green = traffic_manager.blue_green_deployment(
        "web-app",
        rev1_name,
        rev2_name,
        switch_to_green=False
    )
    
    print(f"Blue-green deployment configured:")
    print(f"  Blue: {blue_green['traffic_weights'][rev1_name]}%")
    print(f"  Green: {blue_green['traffic_weights'][rev2_name]}%\n")


def demo_scaling():
    """Demonstrate auto-scaling"""
    print("=== Auto-Scaling Demo ===\n")
    
    scaling = ScalingManager()
    
    # HTTP-based scaling
    http_rule = scaling.create_http_scale_rule(
        "http-scale",
        concurrent_requests=50,
        min_replicas=1,
        max_replicas=10
    )
    print(f"HTTP scale rule: {http_rule.name}")
    print(f"  Concurrent requests: {http_rule.metadata['concurrentRequests']}")
    print(f"  Replicas: {http_rule.min_replicas}-{http_rule.max_replicas}\n")
    
    # CPU-based scaling
    cpu_rule = scaling.create_cpu_scale_rule(
        "cpu-scale",
        cpu_percentage=70,
        min_replicas=2,
        max_replicas=15
    )
    print(f"CPU scale rule: {cpu_rule.name}")
    print(f"  CPU threshold: {cpu_rule.metadata['value']}%")
    print(f"  Replicas: {cpu_rule.min_replicas}-{cpu_rule.max_replicas}\n")
    
    # Memory-based scaling
    memory_rule = scaling.create_memory_scale_rule(
        "memory-scale",
        memory_percentage=80,
        min_replicas=1,
        max_replicas=8
    )
    print(f"Memory scale rule: {memory_rule.name}")
    print(f"  Memory threshold: {memory_rule.metadata['value']}%")
    print(f"  Replicas: {memory_rule.min_replicas}-{memory_rule.max_replicas}\n")


def demo_dapr_integration():
    """Demonstrate Dapr integration"""
    print("=== Dapr Integration Demo ===\n")
    
    dapr = DaprManager()
    
    # Create state store
    state_store = dapr.create_state_store_component(
        "statestore",
        "state.azure.cosmosdb",
        {
            "url": "https://mycosmosdb.documents.azure.com:443/",
            "database": "mydb",
            "collection": "state"
        }
    )
    print(f"Created state store: {state_store['name']}")
    print(f"  Type: {state_store['type']}\n")
    
    # Create pub/sub
    pubsub = dapr.create_pubsub_component(
        "pubsub",
        "pubsub.azure.servicebus",
        {
            "connectionString": "Endpoint=sb://...",
            "consumerID": "myapp"
        }
    )
    print(f"Created pub/sub: {pubsub['name']}")
    print(f"  Type: {pubsub['type']}\n")
    
    # Dapr config for container app
    dapr_config = DaprConfig(
        enabled=True,
        app_id="my-app",
        app_port=8080,
        app_protocol="http"
    )
    print(f"Dapr configuration:")
    print(f"  App ID: {dapr_config.app_id}")
    print(f"  App port: {dapr_config.app_port}")
    print(f"  Protocol: {dapr_config.app_protocol}\n")


def demo_registry_integration():
    """Demonstrate container registry integration"""
    print("=== Container Registry Demo ===\n")
    
    registry = ContainerRegistryManager()
    
    # Add registry
    acr = registry.add_registry(
        "myregistry.azurecr.io",
        "myregistry",
        "password123"
    )
    print(f"Added registry: {acr['url']}")
    print(f"  Username: {acr['username']}\n")
    
    # Create pull secret
    secret = registry.create_image_pull_secret(
        "acr-secret",
        "myregistry.azurecr.io"
    )
    print(f"Created pull secret: {secret['name']}")
    print(f"  Registry: {secret['registry']}\n")


if __name__ == "__main__":
    print("Azure Container Apps - Advanced Implementation")
    print("=" * 60)
    print()
    
    demo_container_app_deployment()
    demo_traffic_management()
    demo_scaling()
    demo_dapr_integration()
    demo_registry_integration()
    
    print("=" * 60)
    print("All demos completed successfully!")
