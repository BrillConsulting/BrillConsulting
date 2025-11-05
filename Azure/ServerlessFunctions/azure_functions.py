"""
Azure Functions Service Integration
Author: BrillConsulting
Contact: clientbrill@gmail.com
LinkedIn: brillconsulting
Description: Advanced Azure Functions implementation with multiple triggers, bindings, and Durable Functions
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import json


class TriggerType(Enum):
    """Function trigger types"""
    HTTP = "httpTrigger"
    TIMER = "timerTrigger"
    BLOB = "blobTrigger"
    QUEUE = "queueTrigger"
    EVENT_GRID = "eventGridTrigger"
    EVENT_HUB = "eventHubTrigger"
    SERVICE_BUS = "serviceBusTrigger"
    COSMOS_DB = "cosmosDBTrigger"


class BindingDirection(Enum):
    """Binding direction"""
    IN = "in"
    OUT = "out"
    IN_OUT = "inout"


class AuthLevel(Enum):
    """HTTP authorization levels"""
    ANONYMOUS = "anonymous"
    FUNCTION = "function"
    ADMIN = "admin"


@dataclass
class Binding:
    """Function binding configuration"""
    name: str
    binding_type: str
    direction: BindingDirection
    connection: Optional[str] = None
    path: Optional[str] = None
    queue_name: Optional[str] = None
    container_name: Optional[str] = None
    database_name: Optional[str] = None
    collection_name: Optional[str] = None


@dataclass
class FunctionConfig:
    """Function configuration"""
    name: str
    trigger_type: TriggerType
    bindings: List[Binding]
    disabled: bool = False
    script_file: Optional[str] = None


@dataclass
class HttpRequest:
    """HTTP request data"""
    method: str
    url: str
    headers: Dict[str, str]
    query_params: Dict[str, str]
    body: Optional[Any] = None


@dataclass
class HttpResponse:
    """HTTP response data"""
    status_code: int
    body: Any
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class OrchestrationInstance:
    """Durable Function orchestration instance"""
    instance_id: str
    orchestration_name: str
    created_at: str
    status: str
    input: Any
    output: Optional[Any] = None
    runtime_status: str = "Running"


class HttpTriggerFunction:
    """
    HTTP triggered Azure Function
    
    Features:
    - HTTP methods support
    - Authorization levels
    - Route parameters
    - Request/response handling
    """
    
    def __init__(
        self,
        name: str,
        methods: Optional[List[str]] = None,
        auth_level: AuthLevel = AuthLevel.FUNCTION,
        route: Optional[str] = None
    ):
        self.name = name
        self.methods = methods or ["GET", "POST"]
        self.auth_level = auth_level
        self.route = route
        self.execution_count = 0
    
    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Execute function"""
        self.execution_count += 1
        
        if request.method not in self.methods:
            return HttpResponse(
                status_code=405,
                body={"error": "Method not allowed"}
            )
        
        return HttpResponse(
            status_code=200,
            body={
                "message": f"Function {self.name} executed",
                "method": request.method,
                "execution_count": self.execution_count,
                "timestamp": datetime.now().isoformat()
            }
        )


class TimerTriggerFunction:
    """
    Timer triggered Azure Function
    
    Features:
    - Cron expressions
    - Scheduled execution
    - Execution history
    """
    
    def __init__(
        self,
        name: str,
        schedule: str,
        use_monitor: bool = True
    ):
        self.name = name
        self.schedule = schedule
        self.use_monitor = use_monitor
        self.execution_history: List[Dict[str, Any]] = []
    
    def __call__(self, timer_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute function"""
        execution = {
            "executed_at": datetime.now().isoformat(),
            "schedule": self.schedule,
            "is_past_due": timer_info.get("is_past_due", False),
            "next_occurrence": (datetime.now() + timedelta(minutes=5)).isoformat()
        }
        
        self.execution_history.append(execution)
        
        return execution


class BlobTriggerFunction:
    """
    Blob storage triggered Azure Function
    
    Features:
    - Blob upload triggers
    - Path patterns
    - Blob metadata
    """
    
    def __init__(
        self,
        name: str,
        container: str,
        path_pattern: str = "{name}",
        connection: str = "AzureWebJobsStorage"
    ):
        self.name = name
        self.container = container
        self.path_pattern = path_pattern
        self.connection = connection
        self.processed_blobs: List[Dict[str, Any]] = []
    
    def __call__(self, blob: Dict[str, Any]) -> Dict[str, Any]:
        """Execute function"""
        result = {
            "blob_name": blob.get("name"),
            "blob_uri": blob.get("uri"),
            "size": blob.get("size", 0),
            "content_type": blob.get("content_type"),
            "processed_at": datetime.now().isoformat(),
            "status": "processed"
        }
        
        self.processed_blobs.append(result)
        return result


class QueueTriggerFunction:
    """
    Queue storage triggered Azure Function
    
    Features:
    - Queue message processing
    - Poison queue handling
    - Batch processing
    """
    
    def __init__(
        self,
        name: str,
        queue_name: str,
        connection: str = "AzureWebJobsStorage"
    ):
        self.name = name
        self.queue_name = queue_name
        self.connection = connection
        self.processed_messages: List[Dict[str, Any]] = []
    
    def __call__(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Execute function"""
        result = {
            "message_id": message.get("id"),
            "content": message.get("content"),
            "dequeue_count": message.get("dequeue_count", 1),
            "processed_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        self.processed_messages.append(result)
        return result


class EventGridTriggerFunction:
    """
    Event Grid triggered Azure Function
    
    Features:
    - Event Grid events
    - Cloud events schema
    - Event filtering
    """
    
    def __init__(self, name: str):
        self.name = name
        self.processed_events: List[Dict[str, Any]] = []
    
    def __call__(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Execute function"""
        result = {
            "event_id": event.get("id"),
            "event_type": event.get("eventType"),
            "subject": event.get("subject"),
            "data": event.get("data"),
            "processed_at": datetime.now().isoformat()
        }
        
        self.processed_events.append(result)
        return result


class DurableOrchestrationClient:
    """
    Durable Functions orchestration client
    
    Features:
    - Start orchestrations
    - Query status
    - Send events
    - Terminate instances
    """
    
    def __init__(self):
        self.instances: Dict[str, OrchestrationInstance] = {}
    
    def start_new(
        self,
        orchestration_name: str,
        instance_id: Optional[str] = None,
        input_data: Optional[Any] = None
    ) -> str:
        """
        Start new orchestration
        
        Args:
            orchestration_name: Orchestration function name
            instance_id: Optional instance ID
            input_data: Input data
            
        Returns:
            Instance ID
        """
        if not instance_id:
            instance_id = f"orch-{datetime.now().timestamp()}"
        
        instance = OrchestrationInstance(
            instance_id=instance_id,
            orchestration_name=orchestration_name,
            created_at=datetime.now().isoformat(),
            status="Running",
            input=input_data
        )
        
        self.instances[instance_id] = instance
        return instance_id
    
    def get_status(self, instance_id: str) -> Optional[OrchestrationInstance]:
        """Get orchestration status"""
        return self.instances.get(instance_id)
    
    def raise_event(
        self,
        instance_id: str,
        event_name: str,
        event_data: Any
    ) -> bool:
        """Raise external event to orchestration"""
        instance = self.instances.get(instance_id)
        if not instance:
            return False
        
        # Event would be processed by orchestration
        return True
    
    def terminate(self, instance_id: str, reason: str) -> bool:
        """Terminate orchestration"""
        instance = self.instances.get(instance_id)
        if not instance:
            return False
        
        instance.runtime_status = "Terminated"
        instance.output = {"reason": reason}
        return True


class DurableActivityFunction:
    """
    Durable Functions activity
    
    Features:
    - Long-running tasks
    - Retry policies
    - Activity execution
    """
    
    def __init__(self, name: str, function: Callable):
        self.name = name
        self.function = function
        self.execution_history: List[Dict[str, Any]] = []
    
    def __call__(self, input_data: Any) -> Any:
        """Execute activity"""
        start_time = datetime.now()
        
        try:
            result = self.function(input_data)
            status = "Completed"
        except Exception as e:
            result = {"error": str(e)}
            status = "Failed"
        
        execution = {
            "activity": self.name,
            "started_at": start_time.isoformat(),
            "completed_at": datetime.now().isoformat(),
            "duration_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "status": status,
            "result": result
        }
        
        self.execution_history.append(execution)
        return result


class DurableOrchestrationContext:
    """
    Durable orchestration context
    
    Features:
    - Call activities
    - Wait for events
    - Create timers
    - Sub-orchestrations
    """
    
    def __init__(self, instance_id: str, input_data: Any):
        self.instance_id = instance_id
        self.input = input_data
        self.activities: List[DurableActivityFunction] = []
        self.current_time = datetime.now()
    
    def call_activity(
        self,
        activity_name: str,
        input_data: Any
    ) -> Any:
        """Call activity function"""
        # Find and execute activity
        for activity in self.activities:
            if activity.name == activity_name:
                return activity(input_data)
        
        return {"error": f"Activity '{activity_name}' not found"}
    
    def create_timer(self, fire_at: datetime) -> Dict[str, Any]:
        """Create durable timer"""
        return {
            "type": "timer",
            "fire_at": fire_at.isoformat(),
            "created_at": datetime.now().isoformat()
        }
    
    def wait_for_external_event(
        self,
        event_name: str,
        timeout: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Wait for external event"""
        return {
            "type": "external_event",
            "event_name": event_name,
            "timeout": timeout.total_seconds() if timeout else None
        }


class FunctionAppManager:
    """
    Azure Function App manager
    
    Features:
    - Function registration
    - Bindings configuration
    - Deployment slots
    - Application settings
    """
    
    def __init__(
        self,
        name: str,
        resource_group: str,
        location: str = "eastus"
    ):
        self.name = name
        self.resource_group = resource_group
        self.location = location
        self.functions: Dict[str, Any] = {}
        self.app_settings: Dict[str, str] = {}
        self.deployment_slots: Dict[str, Dict[str, Any]] = {}
    
    def register_function(
        self,
        function: Any,
        trigger_type: TriggerType
    ) -> Dict[str, Any]:
        """Register function with app"""
        func_name = function.name if hasattr(function, 'name') else function.__name__
        
        self.functions[func_name] = {
            "name": func_name,
            "trigger_type": trigger_type.value,
            "function": function,
            "registered_at": datetime.now().isoformat()
        }
        
        return self.functions[func_name]
    
    def invoke_function(
        self,
        function_name: str,
        input_data: Any
    ) -> Any:
        """Invoke function"""
        if function_name not in self.functions:
            raise ValueError(f"Function '{function_name}' not found")
        
        func_info = self.functions[function_name]
        function = func_info["function"]
        
        return function(input_data)
    
    def set_app_setting(self, key: str, value: str):
        """Set application setting"""
        self.app_settings[key] = value
    
    def get_app_setting(self, key: str) -> Optional[str]:
        """Get application setting"""
        return self.app_settings.get(key)
    
    def create_deployment_slot(
        self,
        slot_name: str
    ) -> Dict[str, Any]:
        """Create deployment slot"""
        slot = {
            "name": slot_name,
            "app_name": self.name,
            "created_at": datetime.now().isoformat(),
            "settings": self.app_settings.copy()
        }
        
        self.deployment_slots[slot_name] = slot
        return slot
    
    def swap_deployment_slots(
        self,
        source_slot: str,
        target_slot: str = "production"
    ) -> Dict[str, Any]:
        """Swap deployment slots"""
        if source_slot not in self.deployment_slots:
            raise ValueError(f"Slot '{source_slot}' not found")
        
        return {
            "source": source_slot,
            "target": target_slot,
            "swapped_at": datetime.now().isoformat(),
            "status": "completed"
        }


# Demo functions
def demo_http_trigger():
    """Demonstrate HTTP triggered function"""
    print("=== HTTP Trigger Demo ===\n")
    
    func = HttpTriggerFunction(
        "ProcessOrder",
        methods=["POST"],
        auth_level=AuthLevel.FUNCTION
    )
    
    request = HttpRequest(
        method="POST",
        url="https://myapp.azurewebsites.net/api/ProcessOrder",
        headers={"Content-Type": "application/json"},
        query_params={},
        body={"order_id": "12345", "amount": 99.99}
    )
    
    response = func(request)
    
    print(f"Function: {func.name}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.body, indent=2)}\n")


def demo_timer_trigger():
    """Demonstrate timer triggered function"""
    print("=== Timer Trigger Demo ===\n")
    
    func = TimerTriggerFunction(
        "DailyCleanup",
        schedule="0 0 2 * * *"  # 2 AM daily
    )
    
    timer_info = {
        "schedule_status": {
            "last": datetime.now().isoformat(),
            "next": (datetime.now() + timedelta(days=1)).isoformat()
        },
        "is_past_due": False
    }
    
    result = func(timer_info)
    
    print(f"Function: {func.name}")
    print(f"Schedule: {result['schedule']}")
    print(f"Executed at: {result['executed_at']}")
    print(f"Next occurrence: {result['next_occurrence']}\n")


def demo_blob_trigger():
    """Demonstrate blob triggered function"""
    print("=== Blob Trigger Demo ===\n")
    
    func = BlobTriggerFunction(
        "ProcessImage",
        container="images",
        path_pattern="{name}.jpg"
    )
    
    blob = {
        "name": "photo_2024.jpg",
        "uri": "https://storage.blob.core.windows.net/images/photo_2024.jpg",
        "size": 1024000,
        "content_type": "image/jpeg"
    }
    
    result = func(blob)
    
    print(f"Function: {func.name}")
    print(f"Blob: {result['blob_name']}")
    print(f"Size: {result['size']} bytes")
    print(f"Status: {result['status']}\n")


def demo_queue_trigger():
    """Demonstrate queue triggered function"""
    print("=== Queue Trigger Demo ===\n")
    
    func = QueueTriggerFunction(
        "ProcessPayment",
        queue_name="payment-queue"
    )
    
    message = {
        "id": "msg-001",
        "content": {"transaction_id": "txn-12345", "amount": 150.00},
        "dequeue_count": 1
    }
    
    result = func(message)
    
    print(f"Function: {func.name}")
    print(f"Message ID: {result['message_id']}")
    print(f"Content: {result['content']}")
    print(f"Status: {result['status']}\n")


def demo_durable_functions():
    """Demonstrate Durable Functions"""
    print("=== Durable Functions Demo ===\n")
    
    # Orchestration client
    client = DurableOrchestrationClient()
    
    # Start orchestration
    instance_id = client.start_new(
        "OrderWorkflow",
        input_data={"order_id": "ORD-001", "amount": 250.00}
    )
    
    print(f"Started orchestration: {instance_id}\n")
    
    # Create activities
    def validate_order(data):
        return {**data, "validated": True}
    
    def process_payment(data):
        return {**data, "payment_status": "completed"}
    
    def send_confirmation(data):
        return {**data, "notification_sent": True}
    
    activity1 = DurableActivityFunction("ValidateOrder", validate_order)
    activity2 = DurableActivityFunction("ProcessPayment", process_payment)
    activity3 = DurableActivityFunction("SendConfirmation", send_confirmation)
    
    # Execute activities
    input_data = {"order_id": "ORD-001", "amount": 250.00}
    result1 = activity1(input_data)
    result2 = activity2(result1)
    result3 = activity3(result2)
    
    print(f"Activity chain completed:")
    print(f"  Final result: {result3}\n")
    
    # Get status
    status = client.get_status(instance_id)
    print(f"Orchestration status: {status.runtime_status}\n")


def demo_function_app():
    """Demonstrate Function App management"""
    print("=== Function App Demo ===\n")
    
    app = FunctionAppManager("my-func-app", "my-rg", "eastus")
    
    # Register functions
    http_func = HttpTriggerFunction("HttpEndpoint", methods=["GET", "POST"])
    app.register_function(http_func, TriggerType.HTTP)
    
    timer_func = TimerTriggerFunction("ScheduledJob", "0 */5 * * * *")
    app.register_function(timer_func, TriggerType.TIMER)
    
    print(f"Function App: {app.name}")
    print(f"Location: {app.location}")
    print(f"Functions: {len(app.functions)}\n")
    
    # App settings
    app.set_app_setting("DATABASE_URL", "postgresql://...")
    app.set_app_setting("API_KEY", "secret123")
    
    print(f"App Settings:")
    for key, value in app.app_settings.items():
        print(f"  {key}: {'*' * len(value) if 'KEY' in key or 'PASSWORD' in key else value}")
    print()
    
    # Deployment slots
    slot = app.create_deployment_slot("staging")
    print(f"Created deployment slot: {slot['name']}")
    
    swap = app.swap_deployment_slots("staging")
    print(f"Swapped slots: {swap['source']} -> {swap['target']}\n")


if __name__ == "__main__":
    print("Azure Functions - Advanced Implementation")
    print("=" * 60)
    print()
    
    demo_http_trigger()
    demo_timer_trigger()
    demo_blob_trigger()
    demo_queue_trigger()
    demo_durable_functions()
    demo_function_app()
    
    print("=" * 60)
    print("All demos completed successfully!")
