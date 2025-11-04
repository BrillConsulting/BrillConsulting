"""
Google Cloud Functions
======================

Serverless functions on GCP:
- HTTP triggered functions
- Pub/Sub triggered functions
- Cloud Storage triggered functions
- Firestore triggered functions
- Function deployment

Author: Brill Consulting
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime
import json


class CloudFunction:
    """GCP Cloud Function."""

    def __init__(self, name: str, trigger_type: str, runtime: str = "python311"):
        self.name = name
        self.trigger_type = trigger_type
        self.runtime = runtime
        self.entry_point = "main"
        self.executions = []

    def execute(self, event: Dict, context: Optional[Dict] = None) -> Dict:
        """Execute function."""
        print(f"\n⚡ Executing function: {self.name}")
        print(f"   Trigger: {self.trigger_type}")

        execution = {
            "function": self.name,
            "trigger_type": self.trigger_type,
            "event": event,
            "context": context or {},
            "executed_at": datetime.now().isoformat(),
            "status": "success"
        }

        self.executions.append(execution)
        print(f"✓ Function executed successfully")

        return execution


class HTTPFunction(CloudFunction):
    """HTTP triggered Cloud Function."""

    def __init__(self, name: str):
        super().__init__(name, "http", "python311")

    def handle_request(self, request: Dict) -> Dict:
        """Handle HTTP request."""
        method = request.get("method", "GET")
        path = request.get("path", "/")
        body = request.get("body", {})

        print(f"   HTTP {method} {path}")

        result = self.execute({"method": method, "path": path, "body": body})

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Success", "function": self.name}),
            "headers": {"Content-Type": "application/json"}
        }


class PubSubFunction(CloudFunction):
    """Pub/Sub triggered Cloud Function."""

    def __init__(self, name: str, topic: str):
        super().__init__(name, "pubsub", "python311")
        self.topic = topic

    def handle_message(self, message: Dict) -> Dict:
        """Handle Pub/Sub message."""
        data = message.get("data", "")
        attributes = message.get("attributes", {})

        print(f"   Topic: {self.topic}")
        print(f"   Message: {data}")

        return self.execute({
            "topic": self.topic,
            "data": data,
            "attributes": attributes
        })


class StorageFunction(CloudFunction):
    """Cloud Storage triggered Cloud Function."""

    def __init__(self, name: str, bucket: str, event_type: str = "finalize"):
        super().__init__(name, "storage", "python311")
        self.bucket = bucket
        self.event_type = event_type

    def handle_storage_event(self, file_info: Dict) -> Dict:
        """Handle storage event."""
        file_name = file_info.get("name")
        size = file_info.get("size", 0)

        print(f"   Bucket: {self.bucket}")
        print(f"   File: {file_name}")
        print(f"   Event: {self.event_type}")

        return self.execute({
            "bucket": self.bucket,
            "file": file_name,
            "size": size,
            "event_type": self.event_type
        })


class GCPFunctionManager:
    """GCP Cloud Functions manager."""

    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.functions = {}

    def deploy_function(self, function: CloudFunction) -> Dict:
        """Deploy Cloud Function."""
        print(f"\n🚀 Deploying function: {function.name}")
        print(f"   Runtime: {function.runtime}")
        print(f"   Trigger: {function.trigger_type}")
        print(f"   Region: {self.region}")

        deployment = {
            "name": function.name,
            "runtime": function.runtime,
            "trigger_type": function.trigger_type,
            "region": self.region,
            "url": f"https://{self.region}-{self.project_id}.cloudfunctions.net/{function.name}",
            "deployed_at": datetime.now().isoformat(),
            "status": "ACTIVE"
        }

        self.functions[function.name] = {
            "function": function,
            "deployment": deployment
        }

        print(f"✓ Function deployed successfully")
        print(f"   URL: {deployment['url']}")

        return deployment

    def invoke_function(self, name: str, event: Dict, context: Optional[Dict] = None) -> Dict:
        """Invoke deployed function."""
        if name not in self.functions:
            return {"error": f"Function {name} not found"}

        function = self.functions[name]["function"]
        return function.execute(event, context)

    def list_functions(self) -> List[Dict]:
        """List deployed functions."""
        return [info["deployment"] for info in self.functions.values()]

    def get_function_metrics(self, name: str) -> Dict:
        """Get function execution metrics."""
        if name not in self.functions:
            return {"error": f"Function {name} not found"}

        function = self.functions[name]["function"]

        return {
            "function": name,
            "total_executions": len(function.executions),
            "success_count": sum(1 for e in function.executions if e["status"] == "success"),
            "trigger_type": function.trigger_type
        }


def demo():
    """Demo GCP Cloud Functions."""
    print("Google Cloud Functions Demo")
    print("=" * 60)

    manager = GCPFunctionManager("my-gcp-project", "us-central1")

    # 1. HTTP Function
    print("\n1. HTTP Triggered Function")
    print("-" * 60)

    http_func = HTTPFunction("process-order")
    manager.deploy_function(http_func)

    http_func.handle_request({
        "method": "POST",
        "path": "/orders",
        "body": {"order_id": "12345", "amount": 99.99}
    })

    # 2. Pub/Sub Function
    print("\n2. Pub/Sub Triggered Function")
    print("-" * 60)

    pubsub_func = PubSubFunction("process-message", "my-topic")
    manager.deploy_function(pubsub_func)

    pubsub_func.handle_message({
        "data": "SGVsbG8gV29ybGQ=",  # base64 encoded
        "attributes": {"source": "api", "priority": "high"}
    })

    # 3. Storage Function
    print("\n3. Cloud Storage Triggered Function")
    print("-" * 60)

    storage_func = StorageFunction("process-image", "my-images-bucket", "finalize")
    manager.deploy_function(storage_func)

    storage_func.handle_storage_event({
        "name": "photos/vacation.jpg",
        "size": 1024000,
        "contentType": "image/jpeg"
    })

    # 4. List Functions
    print("\n4. Deployed Functions")
    print("-" * 60)

    functions = manager.list_functions()
    for func in functions:
        print(f"  • {func['name']} ({func['trigger_type']})")
        print(f"    Status: {func['status']}")
        print(f"    URL: {func['url']}")

    # 5. Function Metrics
    print("\n5. Function Metrics")
    print("-" * 60)

    for func_name in ["process-order", "process-message", "process-image"]:
        metrics = manager.get_function_metrics(func_name)
        print(f"\n  {func_name}:")
        print(f"    Total executions: {metrics['total_executions']}")
        print(f"    Success rate: {metrics['success_count']}/{metrics['total_executions']}")

    print("\n✓ GCP Cloud Functions Demo Complete!")


if __name__ == '__main__':
    demo()
