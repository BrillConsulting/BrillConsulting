"""
Azure Functions
===============

Serverless computing with Azure Functions:
- HTTP triggered functions
- Timer triggered functions
- Blob triggered functions
- Queue triggered functions
- Function orchestration

Author: Brill Consulting
"""

from typing import Dict, Optional, Callable, List
from datetime import datetime, timedelta
import json


class HttpTrigger:
    """HTTP triggered Azure Function."""

    def __init__(self, name: str, methods: List[str] = None, auth_level: str = "function"):
        self.name = name
        self.methods = methods or ["GET", "POST"]
        self.auth_level = auth_level

    def __call__(self, req: Dict) -> Dict:
        """Handle HTTP request."""
        method = req.get("method", "GET")
        body = req.get("body", {})
        params = req.get("params", {})

        print(f"🌐 HTTP {method} request to {self.name}")
        print(f"   Params: {params}")
        print(f"   Body: {body}")

        return {
            "status": 200,
            "body": {
                "message": f"Function {self.name} executed successfully",
                "method": method,
                "timestamp": datetime.now().isoformat()
            }
        }


class TimerTrigger:
    """Timer triggered Azure Function."""

    def __init__(self, name: str, schedule: str):
        """
        Initialize timer trigger.

        Args:
            name: Function name
            schedule: Cron expression (e.g., "0 */5 * * * *" for every 5 minutes)
        """
        self.name = name
        self.schedule = schedule
        self.execution_count = 0
        self.last_execution = None

    def __call__(self, timer_info: Dict) -> Dict:
        """Handle timer trigger."""
        self.execution_count += 1
        self.last_execution = datetime.now()

        print(f"⏰ Timer triggered: {self.name}")
        print(f"   Schedule: {self.schedule}")
        print(f"   Execution #{self.execution_count}")

        return {
            "executed_at": self.last_execution.isoformat(),
            "execution_count": self.execution_count,
            "next_execution": timer_info.get("next_execution")
        }


class BlobTrigger:
    """Blob storage triggered Azure Function."""

    def __init__(self, name: str, container: str, path_pattern: str = "{name}"):
        self.name = name
        self.container = container
        self.path_pattern = path_pattern
        self.processed_blobs = []

    def __call__(self, blob: Dict) -> Dict:
        """Handle blob trigger."""
        blob_name = blob.get("name")
        blob_size = blob.get("size", 0)
        blob_uri = blob.get("uri")

        print(f"📦 Blob trigger: {self.name}")
        print(f"   Container: {self.container}")
        print(f"   Blob: {blob_name}")
        print(f"   Size: {blob_size} bytes")

        # Process blob
        result = {
            "blob_name": blob_name,
            "blob_uri": blob_uri,
            "size": blob_size,
            "processed_at": datetime.now().isoformat(),
            "status": "processed"
        }

        self.processed_blobs.append(result)
        print(f"✓ Blob processed successfully")

        return result


class QueueTrigger:
    """Queue storage triggered Azure Function."""

    def __init__(self, name: str, queue_name: str):
        self.name = name
        self.queue_name = queue_name
        self.processed_messages = []

    def __call__(self, message: Dict) -> Dict:
        """Handle queue message."""
        msg_id = message.get("id")
        msg_content = message.get("content")
        dequeue_count = message.get("dequeueCount", 1)

        print(f"📨 Queue trigger: {self.name}")
        print(f"   Queue: {self.queue_name}")
        print(f"   Message ID: {msg_id}")
        print(f"   Dequeue count: {dequeue_count}")

        # Process message
        result = {
            "message_id": msg_id,
            "content": msg_content,
            "processed_at": datetime.now().isoformat(),
            "status": "completed"
        }

        self.processed_messages.append(result)
        print(f"✓ Message processed")

        return result


class DurableFunctionOrchestrator:
    """Durable Function orchestrator for workflows."""

    def __init__(self, name: str):
        self.name = name
        self.activities = []
        self.execution_history = []

    def add_activity(self, activity_name: str, function: Callable):
        """Add activity to orchestration."""
        self.activities.append({
            "name": activity_name,
            "function": function
        })
        print(f"✓ Activity registered: {activity_name}")

    def orchestrate(self, input_data: Dict) -> Dict:
        """Execute orchestration workflow."""
        print(f"\n🎭 Starting orchestration: {self.name}")
        print(f"   Input: {input_data}")

        instance_id = f"orch_{datetime.now().timestamp()}"
        results = []

        for i, activity in enumerate(self.activities, 1):
            print(f"\n   Step {i}: {activity['name']}")

            try:
                result = activity["function"](input_data)
                results.append({
                    "activity": activity["name"],
                    "status": "completed",
                    "result": result
                })
                print(f"   ✓ Step {i} completed")

                # Pass result to next activity
                input_data = result

            except Exception as e:
                results.append({
                    "activity": activity["name"],
                    "status": "failed",
                    "error": str(e)
                })
                print(f"   ✗ Step {i} failed: {e}")
                break

        orchestration_result = {
            "instance_id": instance_id,
            "orchestrator": self.name,
            "status": "completed" if all(r["status"] == "completed" for r in results) else "failed",
            "steps": results,
            "timestamp": datetime.now().isoformat()
        }

        self.execution_history.append(orchestration_result)
        print(f"\n✓ Orchestration completed")

        return orchestration_result


class AzureFunctionApp:
    """Azure Function App manager."""

    def __init__(self, name: str, resource_group: str, region: str = "eastus"):
        self.name = name
        self.resource_group = resource_group
        self.region = region
        self.functions = {}

    def register_function(self, trigger_type: str, function: Callable):
        """Register function with trigger."""
        func_name = function.name if hasattr(function, 'name') else function.__name__
        self.functions[func_name] = {
            "trigger_type": trigger_type,
            "function": function
        }
        print(f"✓ Function registered: {func_name} ({trigger_type})")

    def invoke(self, function_name: str, input_data: Dict) -> Dict:
        """Invoke function."""
        if function_name not in self.functions:
            return {"error": f"Function {function_name} not found"}

        func_info = self.functions[function_name]
        func = func_info["function"]

        print(f"\n🚀 Invoking function: {function_name}")
        result = func(input_data)
        print(f"✓ Function execution completed")

        return result

    def get_function_config(self) -> Dict:
        """Generate function.json configuration."""
        return {
            "version": "2.0",
            "extensionBundle": {
                "id": "Microsoft.Azure.Functions.ExtensionBundle",
                "version": "[3.*, 4.0.0)"
            },
            "functions": {
                name: {
                    "trigger_type": info["trigger_type"],
                    "direction": "in"
                }
                for name, info in self.functions.items()
            }
        }


def demo():
    """Demo Azure Functions."""
    print("Azure Functions Demo")
    print("=" * 60)

    # Create function app
    app = AzureFunctionApp("func-demo-app", "rg-demo", "eastus")

    # 1. HTTP Trigger
    print("\n1. HTTP Triggered Function")
    print("-" * 60)

    http_func = HttpTrigger("ProcessOrder", methods=["POST"], auth_level="function")
    app.register_function("httpTrigger", http_func)

    result = app.invoke("ProcessOrder", {
        "method": "POST",
        "body": {"order_id": "12345", "amount": 99.99},
        "params": {"customer_id": "cust_001"}
    })
    print(f"Response: {result}")

    # 2. Timer Trigger
    print("\n2. Timer Triggered Function")
    print("-" * 60)

    timer_func = TimerTrigger("DailyCleanup", "0 0 2 * * *")
    app.register_function("timerTrigger", timer_func)

    timer_result = app.invoke("DailyCleanup", {
        "schedule": "0 0 2 * * *",
        "next_execution": (datetime.now() + timedelta(days=1)).isoformat()
    })
    print(f"Timer result: {timer_result}")

    # 3. Blob Trigger
    print("\n3. Blob Storage Triggered Function")
    print("-" * 60)

    blob_func = BlobTrigger("ProcessImage", "images", "{name}.jpg")
    app.register_function("blobTrigger", blob_func)

    blob_result = app.invoke("ProcessImage", {
        "name": "photo_2024.jpg",
        "size": 1024000,
        "uri": "https://storageaccount.blob.core.windows.net/images/photo_2024.jpg"
    })

    # 4. Queue Trigger
    print("\n4. Queue Storage Triggered Function")
    print("-" * 60)

    queue_func = QueueTrigger("ProcessPayment", "payment-queue")
    app.register_function("queueTrigger", queue_func)

    queue_result = app.invoke("ProcessPayment", {
        "id": "msg_001",
        "content": {"transaction_id": "txn_12345", "amount": 150.00},
        "dequeueCount": 1
    })

    # 5. Durable Function Orchestration
    print("\n5. Durable Function Orchestration")
    print("-" * 60)

    orchestrator = DurableFunctionOrchestrator("OrderWorkflow")

    # Define activities
    def validate_order(data):
        return {**data, "validated": True}

    def process_payment(data):
        return {**data, "payment_status": "completed"}

    def send_confirmation(data):
        return {**data, "notification_sent": True}

    orchestrator.add_activity("ValidateOrder", validate_order)
    orchestrator.add_activity("ProcessPayment", process_payment)
    orchestrator.add_activity("SendConfirmation", send_confirmation)

    orch_result = orchestrator.orchestrate({"order_id": "ORD_001", "amount": 250.00})

    print(f"\nOrchestration status: {orch_result['status']}")
    print(f"Steps completed: {len(orch_result['steps'])}")

    # Function App Summary
    print("\n6. Function App Summary")
    print("-" * 60)
    print(f"  App Name: {app.name}")
    print(f"  Resource Group: {app.resource_group}")
    print(f"  Region: {app.region}")
    print(f"  Total Functions: {len(app.functions)}")
    print(f"\n  Registered Functions:")
    for name, info in app.functions.items():
        print(f"    • {name} ({info['trigger_type']})")

    print("\n✓ Azure Functions Demo Complete!")


if __name__ == '__main__':
    demo()
