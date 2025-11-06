"""
Google Cloud Functions - Advanced Serverless Functions
=======================================================

Comprehensive Cloud Functions implementation with:
- HTTP, Pub/Sub, and Cloud Storage triggers
- Advanced deployment configuration (memory, timeout, concurrency)
- Environment variables and secrets management
- IAM and access control
- Function versioning and traffic splitting
- Monitoring and logging integration
- Event-driven architecture patterns

Author: Brill Consulting
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class FunctionManager:
    """Manages Cloud Function deployment and configuration."""

    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.functions = {}

    def create_function(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Cloud Function with advanced configuration.

        Config options:
        - name: Function name
        - runtime: python39, python310, python311, nodejs16, nodejs18, nodejs20, go119, go120, go121, java11, java17
        - entry_point: Entry point function name
        - memory_mb: 128-8192 (128, 256, 512, 1024, 2048, 4096, 8192)
        - timeout_seconds: 1-540 (default: 60)
        - max_instances: 0-1000 (0 = unlimited)
        - min_instances: 0-100 (for reduced cold starts)
        - concurrency: 1-1000 (concurrent requests per instance)
        - environment_variables: Dict of env vars
        - secrets: List of secret references
        - vpc_connector: VPC connector name
        - service_account: Service account email
        """
        name = config.get('name')
        runtime = config.get('runtime', 'python311')
        entry_point = config.get('entry_point', 'main')
        memory_mb = config.get('memory_mb', 256)
        timeout_seconds = config.get('timeout_seconds', 60)
        max_instances = config.get('max_instances', 100)
        min_instances = config.get('min_instances', 0)
        concurrency = config.get('concurrency', 80)

        print(f"\n🚀 Creating Cloud Function: {name}")
        print(f"   Runtime: {runtime}")
        print(f"   Entry Point: {entry_point}")
        print(f"   Memory: {memory_mb}MB")
        print(f"   Timeout: {timeout_seconds}s")
        print(f"   Max Instances: {max_instances}")
        print(f"   Min Instances: {min_instances}")
        print(f"   Concurrency: {concurrency}")

        function_config = {
            "name": name,
            "runtime": runtime,
            "entry_point": entry_point,
            "memory_mb": memory_mb,
            "timeout_seconds": timeout_seconds,
            "max_instances": max_instances,
            "min_instances": min_instances,
            "concurrency": concurrency,
            "environment_variables": config.get('environment_variables', {}),
            "secrets": config.get('secrets', []),
            "vpc_connector": config.get('vpc_connector'),
            "service_account": config.get('service_account', f"{self.project_id}@appspot.gserviceaccount.com"),
            "region": self.region,
            "url": f"https://{self.region}-{self.project_id}.cloudfunctions.net/{name}",
            "created_at": datetime.now().isoformat(),
            "status": "ACTIVE",
            "executions": []
        }

        self.functions[name] = function_config

        print(f"✓ Function created successfully")
        print(f"   URL: {function_config['url']}")

        return function_config

    def update_function(self, name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update function configuration."""
        if name not in self.functions:
            return {"error": f"Function {name} not found"}

        print(f"\n🔄 Updating function: {name}")

        function = self.functions[name]
        for key, value in updates.items():
            if key in function:
                print(f"   {key}: {function[key]} → {value}")
                function[key] = value

        function['updated_at'] = datetime.now().isoformat()

        print(f"✓ Function updated successfully")
        return function

    def delete_function(self, name: str) -> Dict[str, str]:
        """Delete Cloud Function."""
        if name not in self.functions:
            return {"error": f"Function {name} not found"}

        print(f"\n🗑️  Deleting function: {name}")
        del self.functions[name]

        print(f"✓ Function deleted successfully")
        return {"status": "deleted", "name": name}

    def list_functions(self) -> List[Dict[str, Any]]:
        """List all functions."""
        return list(self.functions.values())

    def get_function(self, name: str) -> Optional[Dict[str, Any]]:
        """Get function configuration."""
        return self.functions.get(name)

    def generate_deployment_code(self, name: str) -> str:
        """Generate gcloud deployment command."""
        if name not in self.functions:
            return f"# Error: Function {name} not found"

        func = self.functions[name]

        cmd = f"""# Deploy {name} to Cloud Functions
gcloud functions deploy {name} \\
    --runtime={func['runtime']} \\
    --entry-point={func['entry_point']} \\
    --memory={func['memory_mb']}MB \\
    --timeout={func['timeout_seconds']}s \\
    --max-instances={func['max_instances']} \\
    --min-instances={func['min_instances']} \\
    --service-account={func['service_account']} \\
    --region={func['region']}"""

        if func.get('environment_variables'):
            env_vars = ','.join([f"{k}={v}" for k, v in func['environment_variables'].items()])
            cmd += f" \\\n    --set-env-vars={env_vars}"

        if func.get('vpc_connector'):
            cmd += f" \\\n    --vpc-connector={func['vpc_connector']}"

        return cmd


class TriggerManager:
    """Manages Cloud Function triggers."""

    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.triggers = {}

    def create_http_trigger(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create HTTP trigger.

        Config:
        - function_name: Target function
        - allow_unauthenticated: Public access (default: False)
        - cors: CORS configuration
        """
        function_name = config.get('function_name')
        allow_unauthenticated = config.get('allow_unauthenticated', False)

        print(f"\n🌐 Creating HTTP trigger for: {function_name}")
        print(f"   Public Access: {allow_unauthenticated}")

        trigger = {
            "type": "HTTP",
            "function_name": function_name,
            "allow_unauthenticated": allow_unauthenticated,
            "url": f"https://{self.region}-{self.project_id}.cloudfunctions.net/{function_name}",
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "cors": config.get('cors', {"origins": ["*"], "methods": ["GET", "POST"]}),
            "created_at": datetime.now().isoformat()
        }

        self.triggers[function_name] = trigger

        print(f"✓ HTTP trigger created")
        print(f"   URL: {trigger['url']}")

        return trigger

    def create_pubsub_trigger(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Pub/Sub trigger.

        Config:
        - function_name: Target function
        - topic: Pub/Sub topic name
        - retry_policy: Retry configuration
        """
        function_name = config.get('function_name')
        topic = config.get('topic')

        print(f"\n📨 Creating Pub/Sub trigger for: {function_name}")
        print(f"   Topic: {topic}")

        trigger = {
            "type": "PUBSUB",
            "function_name": function_name,
            "topic": f"projects/{self.project_id}/topics/{topic}",
            "retry_policy": config.get('retry_policy', {
                "retry_attempts": 3,
                "min_backoff": "10s",
                "max_backoff": "600s"
            }),
            "created_at": datetime.now().isoformat()
        }

        self.triggers[function_name] = trigger

        print(f"✓ Pub/Sub trigger created")

        return trigger

    def create_storage_trigger(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Cloud Storage trigger.

        Config:
        - function_name: Target function
        - bucket: Storage bucket name
        - event_type: finalize, delete, archive, metadataUpdate
        """
        function_name = config.get('function_name')
        bucket = config.get('bucket')
        event_type = config.get('event_type', 'finalize')

        print(f"\n🪣 Creating Storage trigger for: {function_name}")
        print(f"   Bucket: {bucket}")
        print(f"   Event Type: {event_type}")

        trigger = {
            "type": "STORAGE",
            "function_name": function_name,
            "bucket": bucket,
            "event_type": f"google.storage.object.{event_type}",
            "created_at": datetime.now().isoformat()
        }

        self.triggers[function_name] = trigger

        print(f"✓ Storage trigger created")

        return trigger

    def create_firestore_trigger(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Firestore trigger.

        Config:
        - function_name: Target function
        - document_path: Document path pattern (e.g., users/{userId})
        - event_type: create, update, delete, write
        """
        function_name = config.get('function_name')
        document_path = config.get('document_path')
        event_type = config.get('event_type', 'write')

        print(f"\n🔥 Creating Firestore trigger for: {function_name}")
        print(f"   Document Path: {document_path}")
        print(f"   Event Type: {event_type}")

        trigger = {
            "type": "FIRESTORE",
            "function_name": function_name,
            "document_path": document_path,
            "event_type": f"providers/cloud.firestore/eventTypes/document.{event_type}",
            "created_at": datetime.now().isoformat()
        }

        self.triggers[function_name] = trigger

        print(f"✓ Firestore trigger created")

        return trigger

    def list_triggers(self) -> List[Dict[str, Any]]:
        """List all triggers."""
        return list(self.triggers.values())

    def get_trigger_code(self, function_name: str) -> str:
        """Generate trigger deployment code."""
        if function_name not in self.triggers:
            return f"# No trigger found for {function_name}"

        trigger = self.triggers[function_name]

        if trigger['type'] == 'HTTP':
            return f"""# HTTP Trigger for {function_name}
gcloud functions deploy {function_name} \\
    --trigger-http \\
    {'--allow-unauthenticated' if trigger['allow_unauthenticated'] else ''}"""

        elif trigger['type'] == 'PUBSUB':
            topic = trigger['topic'].split('/')[-1]
            return f"""# Pub/Sub Trigger for {function_name}
gcloud functions deploy {function_name} \\
    --trigger-topic={topic}"""

        elif trigger['type'] == 'STORAGE':
            return f"""# Storage Trigger for {function_name}
gcloud functions deploy {function_name} \\
    --trigger-bucket={trigger['bucket']} \\
    --trigger-event={trigger['event_type']}"""

        elif trigger['type'] == 'FIRESTORE':
            return f"""# Firestore Trigger for {function_name}
gcloud functions deploy {function_name} \\
    --trigger-event={trigger['event_type']} \\
    --trigger-resource=projects/{self.project_id}/databases/(default)/documents/{trigger['document_path']}"""


class IAMManager:
    """Manages Cloud Function IAM policies."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.policies = {}

    def grant_invoker_access(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grant function invoker permissions.

        Config:
        - function_name: Function name
        - member: allUsers, allAuthenticatedUsers, user:email, serviceAccount:email
        """
        function_name = config.get('function_name')
        member = config.get('member')

        print(f"\n🔐 Granting invoker access: {function_name}")
        print(f"   Member: {member}")

        policy = {
            "function_name": function_name,
            "role": "roles/cloudfunctions.invoker",
            "member": member,
            "granted_at": datetime.now().isoformat()
        }

        if function_name not in self.policies:
            self.policies[function_name] = []

        self.policies[function_name].append(policy)

        print(f"✓ Access granted")

        return policy

    def revoke_invoker_access(self, function_name: str, member: str) -> Dict[str, str]:
        """Revoke function invoker permissions."""
        print(f"\n🔒 Revoking invoker access: {function_name}")
        print(f"   Member: {member}")

        if function_name in self.policies:
            self.policies[function_name] = [
                p for p in self.policies[function_name]
                if p['member'] != member
            ]

        print(f"✓ Access revoked")

        return {"status": "revoked", "function": function_name, "member": member}

    def make_public(self, function_name: str) -> Dict[str, Any]:
        """Make function publicly accessible."""
        return self.grant_invoker_access({
            "function_name": function_name,
            "member": "allUsers"
        })

    def make_private(self, function_name: str) -> Dict[str, str]:
        """Make function private (revoke public access)."""
        return self.revoke_invoker_access(function_name, "allUsers")

    def list_policies(self, function_name: str) -> List[Dict[str, Any]]:
        """List IAM policies for function."""
        return self.policies.get(function_name, [])

    def generate_iam_code(self, function_name: str, member: str) -> str:
        """Generate gcloud IAM command."""
        return f"""# Grant invoker access to {function_name}
gcloud functions add-iam-policy-binding {function_name} \\
    --member="{member}" \\
    --role="roles/cloudfunctions.invoker" \\
    --region=us-central1"""


class VersionManager:
    """Manages Cloud Function versions and traffic splitting."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.versions = {}

    def create_version(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new function version.

        Config:
        - function_name: Function name
        - version_id: Version identifier
        - source_code: Source code location
        - description: Version description
        """
        function_name = config.get('function_name')
        version_id = config.get('version_id')

        print(f"\n📦 Creating version for: {function_name}")
        print(f"   Version ID: {version_id}")

        version = {
            "function_name": function_name,
            "version_id": version_id,
            "source_code": config.get('source_code'),
            "description": config.get('description', ''),
            "created_at": datetime.now().isoformat(),
            "status": "ACTIVE"
        }

        if function_name not in self.versions:
            self.versions[function_name] = []

        self.versions[function_name].append(version)

        print(f"✓ Version created")

        return version

    def split_traffic(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Split traffic between versions.

        Config:
        - function_name: Function name
        - traffic_split: Dict of version_id -> traffic percentage
        """
        function_name = config.get('function_name')
        traffic_split = config.get('traffic_split', {})

        print(f"\n🔀 Configuring traffic split: {function_name}")
        for version_id, percentage in traffic_split.items():
            print(f"   {version_id}: {percentage}%")

        split = {
            "function_name": function_name,
            "traffic_split": traffic_split,
            "configured_at": datetime.now().isoformat()
        }

        print(f"✓ Traffic split configured")

        return split

    def rollback(self, function_name: str, target_version: str) -> Dict[str, Any]:
        """Rollback to previous version."""
        print(f"\n⏮️  Rolling back: {function_name}")
        print(f"   Target Version: {target_version}")

        rollback = {
            "function_name": function_name,
            "from_version": "current",
            "to_version": target_version,
            "rolled_back_at": datetime.now().isoformat()
        }

        print(f"✓ Rollback complete")

        return rollback

    def list_versions(self, function_name: str) -> List[Dict[str, Any]]:
        """List all versions for function."""
        return self.versions.get(function_name, [])


class MonitoringManager:
    """Manages Cloud Function monitoring and logging."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.logs = {}
        self.metrics = {}

    def log_execution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log function execution.

        Config:
        - function_name: Function name
        - execution_id: Unique execution ID
        - status: success, error, timeout
        - duration_ms: Execution duration
        - memory_used_mb: Memory used
        - error: Error message (if failed)
        """
        function_name = config.get('function_name')
        execution_id = config.get('execution_id')
        status = config.get('status', 'success')

        log_entry = {
            "function_name": function_name,
            "execution_id": execution_id,
            "status": status,
            "duration_ms": config.get('duration_ms', 0),
            "memory_used_mb": config.get('memory_used_mb', 0),
            "error": config.get('error'),
            "timestamp": datetime.now().isoformat()
        }

        if function_name not in self.logs:
            self.logs[function_name] = []

        self.logs[function_name].append(log_entry)

        return log_entry

    def get_metrics(self, function_name: str) -> Dict[str, Any]:
        """Get function execution metrics."""
        logs = self.logs.get(function_name, [])

        if not logs:
            return {"error": f"No logs found for {function_name}"}

        total_executions = len(logs)
        successful = sum(1 for log in logs if log['status'] == 'success')
        errors = sum(1 for log in logs if log['status'] == 'error')
        timeouts = sum(1 for log in logs if log['status'] == 'timeout')

        durations = [log['duration_ms'] for log in logs]
        avg_duration = sum(durations) / len(durations) if durations else 0

        memory_used = [log['memory_used_mb'] for log in logs]
        avg_memory = sum(memory_used) / len(memory_used) if memory_used else 0

        metrics = {
            "function_name": function_name,
            "total_executions": total_executions,
            "successful_executions": successful,
            "errors": errors,
            "timeouts": timeouts,
            "success_rate": (successful / total_executions * 100) if total_executions > 0 else 0,
            "avg_duration_ms": round(avg_duration, 2),
            "avg_memory_mb": round(avg_memory, 2)
        }

        print(f"\n📊 Metrics for {function_name}:")
        print(f"   Total Executions: {metrics['total_executions']}")
        print(f"   Success Rate: {metrics['success_rate']:.1f}%")
        print(f"   Avg Duration: {metrics['avg_duration_ms']}ms")
        print(f"   Avg Memory: {metrics['avg_memory_mb']}MB")

        return metrics

    def query_logs(self, function_name: str, filter_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query function logs with optional status filter."""
        logs = self.logs.get(function_name, [])

        if filter_status:
            logs = [log for log in logs if log['status'] == filter_status]

        return logs

    def create_alert(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create monitoring alert.

        Config:
        - function_name: Function name
        - metric: error_rate, latency, invocation_count
        - threshold: Alert threshold
        - notification_channels: List of notification channels
        """
        function_name = config.get('function_name')
        metric = config.get('metric')
        threshold = config.get('threshold')

        print(f"\n🚨 Creating alert for: {function_name}")
        print(f"   Metric: {metric}")
        print(f"   Threshold: {threshold}")

        alert = {
            "function_name": function_name,
            "metric": metric,
            "threshold": threshold,
            "notification_channels": config.get('notification_channels', []),
            "created_at": datetime.now().isoformat(),
            "status": "ACTIVE"
        }

        print(f"✓ Alert created")

        return alert


class CloudFunctionsManager:
    """Main Cloud Functions manager integrating all components."""

    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.functions = FunctionManager(project_id, region)
        self.triggers = TriggerManager(project_id, region)
        self.iam = IAMManager(project_id)
        self.versions = VersionManager(project_id)
        self.monitoring = MonitoringManager(project_id)

    def info(self) -> Dict[str, Any]:
        """Get manager information."""
        return {
            "project_id": self.project_id,
            "region": self.region,
            "total_functions": len(self.functions.functions),
            "total_triggers": len(self.triggers.triggers),
            "supported_runtimes": [
                "python39", "python310", "python311",
                "nodejs16", "nodejs18", "nodejs20",
                "go119", "go120", "go121",
                "java11", "java17"
            ]
        }


def demo():
    """Demo Cloud Functions with advanced features."""
    print("=" * 70)
    print("Google Cloud Functions - Advanced Demo")
    print("=" * 70)

    mgr = CloudFunctionsManager("my-gcp-project", "us-central1")

    # 1. Create HTTP function with configuration
    print("\n1. HTTP Function with Advanced Configuration")
    print("-" * 70)

    http_func = mgr.functions.create_function({
        "name": "api-gateway",
        "runtime": "python311",
        "entry_point": "handle_request",
        "memory_mb": 512,
        "timeout_seconds": 120,
        "max_instances": 50,
        "min_instances": 2,
        "concurrency": 100,
        "environment_variables": {
            "ENV": "production",
            "API_VERSION": "v2"
        }
    })

    # Create HTTP trigger
    mgr.triggers.create_http_trigger({
        "function_name": "api-gateway",
        "allow_unauthenticated": True,
        "cors": {
            "origins": ["https://example.com"],
            "methods": ["GET", "POST", "PUT"]
        }
    })

    # Make it public
    mgr.iam.make_public("api-gateway")

    # 2. Pub/Sub function
    print("\n2. Pub/Sub Event Processing Function")
    print("-" * 70)

    pubsub_func = mgr.functions.create_function({
        "name": "process-events",
        "runtime": "python311",
        "entry_point": "handle_pubsub",
        "memory_mb": 1024,
        "timeout_seconds": 300,
        "max_instances": 100
    })

    mgr.triggers.create_pubsub_trigger({
        "function_name": "process-events",
        "topic": "event-stream",
        "retry_policy": {
            "retry_attempts": 5,
            "min_backoff": "10s",
            "max_backoff": "600s"
        }
    })

    # 3. Storage function for image processing
    print("\n3. Storage Trigger for Image Processing")
    print("-" * 70)

    storage_func = mgr.functions.create_function({
        "name": "process-images",
        "runtime": "python311",
        "entry_point": "process_image",
        "memory_mb": 2048,
        "timeout_seconds": 540,
        "max_instances": 20
    })

    mgr.triggers.create_storage_trigger({
        "function_name": "process-images",
        "bucket": "uploaded-images",
        "event_type": "finalize"
    })

    # 4. Firestore trigger
    print("\n4. Firestore Database Trigger")
    print("-" * 70)

    firestore_func = mgr.functions.create_function({
        "name": "sync-user-data",
        "runtime": "python311",
        "entry_point": "sync_data",
        "memory_mb": 256,
        "timeout_seconds": 60
    })

    mgr.triggers.create_firestore_trigger({
        "function_name": "sync-user-data",
        "document_path": "users/{userId}",
        "event_type": "write"
    })

    # 5. Function versioning and traffic splitting
    print("\n5. Function Versioning and Traffic Split")
    print("-" * 70)

    # Create versions
    mgr.versions.create_version({
        "function_name": "api-gateway",
        "version_id": "v1.0.0",
        "description": "Initial production release"
    })

    mgr.versions.create_version({
        "function_name": "api-gateway",
        "version_id": "v1.1.0",
        "description": "Performance improvements"
    })

    # Split traffic (90% v1.0.0, 10% v1.1.0 for canary testing)
    mgr.versions.split_traffic({
        "function_name": "api-gateway",
        "traffic_split": {
            "v1.0.0": 90,
            "v1.1.0": 10
        }
    })

    # 6. Monitoring and logging
    print("\n6. Function Monitoring")
    print("-" * 70)

    # Simulate some executions
    for i in range(10):
        mgr.monitoring.log_execution({
            "function_name": "api-gateway",
            "execution_id": f"exec-{i}",
            "status": "success" if i < 9 else "error",
            "duration_ms": 150 + (i * 10),
            "memory_used_mb": 128 + (i * 5),
            "error": "Timeout error" if i == 9 else None
        })

    # Get metrics
    metrics = mgr.monitoring.get_metrics("api-gateway")

    # Create alert
    mgr.monitoring.create_alert({
        "function_name": "api-gateway",
        "metric": "error_rate",
        "threshold": 5,
        "notification_channels": ["email:admin@example.com"]
    })

    # 7. List all functions
    print("\n7. Deployed Functions Summary")
    print("-" * 70)

    functions = mgr.functions.list_functions()
    for func in functions:
        print(f"\n  📦 {func['name']}")
        print(f"     Runtime: {func['runtime']}")
        print(f"     Memory: {func['memory_mb']}MB")
        print(f"     Timeout: {func['timeout_seconds']}s")
        print(f"     Max Instances: {func['max_instances']}")
        print(f"     Status: {func['status']}")
        print(f"     URL: {func['url']}")

    # 8. Generate deployment code
    print("\n8. Deployment Code Example")
    print("-" * 70)

    deployment_code = mgr.functions.generate_deployment_code("api-gateway")
    print(f"\n{deployment_code}")

    trigger_code = mgr.triggers.get_trigger_code("api-gateway")
    print(f"\n{trigger_code}")

    iam_code = mgr.iam.generate_iam_code("api-gateway", "allUsers")
    print(f"\n{iam_code}")

    # Manager info
    print("\n9. Manager Information")
    print("-" * 70)

    info = mgr.info()
    print(f"\n  Project: {info['project_id']}")
    print(f"  Region: {info['region']}")
    print(f"  Total Functions: {info['total_functions']}")
    print(f"  Total Triggers: {info['total_triggers']}")
    print(f"  Supported Runtimes: {', '.join(info['supported_runtimes'][:5])}...")

    print("\n" + "=" * 70)
    print("✓ Cloud Functions Advanced Demo Complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
