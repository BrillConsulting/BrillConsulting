"""
Cloud Tasks - Asynchronous Task Queue Service
Author: BrillConsulting
Description: Comprehensive task queue management with HTTP and App Engine targets
"""

import json
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class TaskQueue:
    """Manage Cloud Tasks queues"""

    def __init__(self, project_id: str, location: str = 'us-central1'):
        """
        Initialize task queue manager

        Args:
            project_id: GCP project ID
            location: Queue location
        """
        self.project_id = project_id
        self.location = location
        self.queues = []

    def create_queue(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a task queue

        Args:
            config: Queue configuration

        Returns:
            Queue creation result
        """
        print(f"\n{'='*60}")
        print("Creating Task Queue")
        print(f"{'='*60}")

        queue_name = config.get('queue_name', 'my-queue')
        max_concurrent = config.get('max_concurrent_dispatches', 1000)
        max_rate = config.get('max_dispatches_per_second', 500.0)
        max_attempts = config.get('max_attempts', 5)

        code = f"""
from google.cloud import tasks_v2

client = tasks_v2.CloudTasksClient()

# Build the parent path
parent = f"projects/{self.project_id}/locations/{self.location}"

# Define the queue
queue = {{
    "name": f"{{parent}}/queues/{queue_name}",

    # Rate limits
    "rate_limits": {{
        "max_concurrent_dispatches": {max_concurrent},
        "max_dispatches_per_second": {max_rate},
    }},

    # Retry configuration
    "retry_config": {{
        "max_attempts": {max_attempts},
        "max_retry_duration": {{"seconds": 3600}},  # 1 hour
        "min_backoff": {{"seconds": 10}},
        "max_backoff": {{"seconds": 600}},
        "max_doublings": 5,
    }},

    # Task TTL
    "task_ttl": {{"seconds": 86400}},  # 24 hours
}}

# Create the queue
response = client.create_queue(parent=parent, queue=queue)
print(f"Created queue: {{response.name}}")
"""

        result = {
            'queue_name': queue_name,
            'max_concurrent_dispatches': max_concurrent,
            'max_dispatches_per_second': max_rate,
            'max_attempts': max_attempts,
            'full_name': f"projects/{self.project_id}/locations/{self.location}/queues/{queue_name}",
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.queues.append(result)

        print(f"✓ Queue created: {queue_name}")
        print(f"  Max concurrent: {max_concurrent}")
        print(f"  Max rate: {max_rate}/sec")
        print(f"  Max attempts: {max_attempts}")
        print(f"{'='*60}")

        return result

    def pause_queue(self, queue_name: str) -> Dict[str, Any]:
        """
        Pause a queue

        Args:
            queue_name: Queue name

        Returns:
            Pause operation result
        """
        print(f"\n{'='*60}")
        print("Pausing Queue")
        print(f"{'='*60}")

        code = f"""
from google.cloud import tasks_v2

client = tasks_v2.CloudTasksClient()

# Build queue path
name = f"projects/{self.project_id}/locations/{self.location}/queues/{queue_name}"

# Pause the queue
response = client.pause_queue(name=name)
print(f"Paused queue: {{response.name}}")
print(f"State: {{response.state.name}}")
"""

        result = {
            'queue_name': queue_name,
            'action': 'PAUSED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Queue paused: {queue_name}")
        print(f"{'='*60}")

        return result


class HttpTask:
    """Create and manage HTTP target tasks"""

    def __init__(self, project_id: str, location: str = 'us-central1'):
        """Initialize HTTP task manager"""
        self.project_id = project_id
        self.location = location
        self.tasks = []

    def create_http_task(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create HTTP task

        Args:
            config: Task configuration

        Returns:
            Task creation result
        """
        print(f"\n{'='*60}")
        print("Creating HTTP Task")
        print(f"{'='*60}")

        queue_name = config.get('queue_name', 'my-queue')
        task_name = config.get('task_name', None)  # Auto-generate if None
        url = config.get('url', 'https://example.com/api/process')
        http_method = config.get('http_method', 'POST')
        payload = config.get('payload', {})
        headers = config.get('headers', {'Content-Type': 'application/json'})
        schedule_delay_seconds = config.get('schedule_delay_seconds', 0)

        code = f"""
from google.cloud import tasks_v2
from datetime import datetime, timedelta

client = tasks_v2.CloudTasksClient()

# Build queue path
parent = f"projects/{self.project_id}/locations/{self.location}/queues/{queue_name}"

# Define the task
task = {{
    "http_request": {{
        "url": "{url}",
        "http_method": tasks_v2.HttpMethod.{http_method},
        "headers": {headers},
        "body": '{json.dumps(payload)}'.encode('utf-8'),

        # OIDC authentication (if using Cloud Run)
        # "oidc_token": {{
        #     "service_account_email": "task-runner@{self.project_id}.iam.gserviceaccount.com",
        # }},
    }},
}}

# Schedule for later (optional)
{'if ' + str(schedule_delay_seconds) + ' > 0:' if schedule_delay_seconds > 0 else '# No delay'}
    schedule_time = datetime.utcnow() + timedelta(seconds={schedule_delay_seconds})
    task["schedule_time"] = schedule_time

# Create the task
response = client.create_task(parent=parent, task=task)
print(f"Created task: {{response.name}}")
print(f"Scheduled for: {{response.schedule_time}}")
"""

        task_id = task_name or f"task-{datetime.now().timestamp()}"

        result = {
            'task_name': task_id,
            'queue_name': queue_name,
            'url': url,
            'http_method': http_method,
            'schedule_delay_seconds': schedule_delay_seconds,
            'full_name': f"projects/{self.project_id}/locations/{self.location}/queues/{queue_name}/tasks/{task_id}",
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.tasks.append(result)

        print(f"✓ HTTP task created: {task_id}")
        print(f"  Queue: {queue_name}")
        print(f"  URL: {url}")
        print(f"  Method: {http_method}")
        if schedule_delay_seconds > 0:
            print(f"  Delayed by: {schedule_delay_seconds}s")
        print(f"{'='*60}")

        return result

    def create_batch_tasks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create multiple tasks in batch

        Args:
            config: Batch configuration

        Returns:
            Batch creation result
        """
        print(f"\n{'='*60}")
        print("Creating Batch HTTP Tasks")
        print(f"{'='*60}")

        queue_name = config.get('queue_name', 'my-queue')
        url = config.get('url', 'https://example.com/api/process')
        items = config.get('items', [])
        batch_size = len(items)

        code = f"""
from google.cloud import tasks_v2
import json

client = tasks_v2.CloudTasksClient()

# Build queue path
parent = f"projects/{self.project_id}/locations/{self.location}/queues/{queue_name}"

# Items to process
items = {items}

# Create tasks for each item
created_tasks = []
for item in items:
    task = {{
        "http_request": {{
            "url": "{url}",
            "http_method": tasks_v2.HttpMethod.POST,
            "headers": {{"Content-Type": "application/json"}},
            "body": json.dumps(item).encode('utf-8'),
        }},
    }}

    response = client.create_task(parent=parent, task=task)
    created_tasks.append(response.name)
    print(f"Created task {{len(created_tasks)}}/{{len(items)}}: {{response.name}}")

print(f"\\nBatch complete: {{len(created_tasks)}} tasks created")
"""

        result = {
            'queue_name': queue_name,
            'batch_size': batch_size,
            'url': url,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Batch created: {batch_size} tasks")
        print(f"  Queue: {queue_name}")
        print(f"  Target: {url}")
        print(f"{'='*60}")

        return result


class AppEngineTask:
    """Create App Engine target tasks"""

    def __init__(self, project_id: str, location: str = 'us-central1'):
        """Initialize App Engine task manager"""
        self.project_id = project_id
        self.location = location

    def create_appengine_task(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create App Engine task

        Args:
            config: Task configuration

        Returns:
            Task creation result
        """
        print(f"\n{'='*60}")
        print("Creating App Engine Task")
        print(f"{'='*60}")

        queue_name = config.get('queue_name', 'my-queue')
        relative_uri = config.get('relative_uri', '/tasks/process')
        http_method = config.get('http_method', 'POST')
        payload = config.get('payload', {})

        code = f"""
from google.cloud import tasks_v2

client = tasks_v2.CloudTasksClient()

# Build queue path
parent = f"projects/{self.project_id}/locations/{self.location}/queues/{queue_name}"

# Define the App Engine task
task = {{
    "app_engine_http_request": {{
        "relative_uri": "{relative_uri}",
        "http_method": tasks_v2.HttpMethod.{http_method},
        "body": '{json.dumps(payload)}'.encode('utf-8'),

        "app_engine_routing": {{
            "service": "default",
            "version": "v1",
            "instance": "",  # Empty for any instance
        }},
    }},
}}

# Create the task
response = client.create_task(parent=parent, task=task)
print(f"Created App Engine task: {{response.name}}")
"""

        result = {
            'queue_name': queue_name,
            'relative_uri': relative_uri,
            'http_method': http_method,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ App Engine task created")
        print(f"  Queue: {queue_name}")
        print(f"  URI: {relative_uri}")
        print(f"  Method: {http_method}")
        print(f"{'='*60}")

        return result


class TaskMonitoring:
    """Monitor and manage tasks"""

    def __init__(self, project_id: str, location: str = 'us-central1'):
        """Initialize monitoring"""
        self.project_id = project_id
        self.location = location

    def list_tasks(self, queue_name: str) -> str:
        """
        List tasks in a queue

        Args:
            queue_name: Queue name

        Returns:
            Code to list tasks
        """
        code = f"""
from google.cloud import tasks_v2

client = tasks_v2.CloudTasksClient()

# Build queue path
parent = f"projects/{self.project_id}/locations/{self.location}/queues/{queue_name}"

# List all tasks
print("Tasks in queue '{queue_name}':")
print("=" * 60)

for task in client.list_tasks(parent=parent):
    print(f"Task: {{task.name}}")
    print(f"  Schedule time: {{task.schedule_time}}")
    print(f"  Dispatch count: {{task.dispatch_count}}")
    print(f"  Response count: {{task.response_count}}")
    if task.http_request:
        print(f"  Target: {{task.http_request.url}}")
    print("-" * 60)
"""

        print(f"\n✓ Task listing code generated for queue: {queue_name}")
        return code

    def delete_task(self, queue_name: str, task_name: str) -> Dict[str, Any]:
        """
        Delete a task

        Args:
            queue_name: Queue name
            task_name: Task name

        Returns:
            Delete operation result
        """
        print(f"\n{'='*60}")
        print("Deleting Task")
        print(f"{'='*60}")

        code = f"""
from google.cloud import tasks_v2

client = tasks_v2.CloudTasksClient()

# Build task path
name = f"projects/{self.project_id}/locations/{self.location}/queues/{queue_name}/tasks/{task_name}"

# Delete the task
client.delete_task(name=name)
print(f"Deleted task: {task_name}")
"""

        result = {
            'queue_name': queue_name,
            'task_name': task_name,
            'action': 'DELETED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Task deleted: {task_name}")
        print(f"  Queue: {queue_name}")
        print(f"{'='*60}")

        return result

    def purge_queue(self, queue_name: str) -> Dict[str, Any]:
        """
        Purge all tasks from a queue

        Args:
            queue_name: Queue name

        Returns:
            Purge operation result
        """
        print(f"\n{'='*60}")
        print("Purging Queue")
        print(f"{'='*60}")

        code = f"""
from google.cloud import tasks_v2

client = tasks_v2.CloudTasksClient()

# Build queue path
name = f"projects/{self.project_id}/locations/{self.location}/queues/{queue_name}"

# Purge all tasks
response = client.purge_queue(name=name)
print(f"Purged queue: {{response.name}}")
print("All tasks removed from queue")
"""

        result = {
            'queue_name': queue_name,
            'action': 'PURGED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Queue purged: {queue_name}")
        print(f"  All tasks removed")
        print(f"{'='*60}")

        return result


class CloudTasksManager:
    """Comprehensive Cloud Tasks management"""

    def __init__(self, project_id: str = 'my-project', location: str = 'us-central1'):
        """
        Initialize Cloud Tasks manager

        Args:
            project_id: GCP project ID
            location: Tasks location
        """
        self.project_id = project_id
        self.location = location
        self.queue_manager = TaskQueue(project_id, location)
        self.http_tasks = HttpTask(project_id, location)
        self.appengine_tasks = AppEngineTask(project_id, location)
        self.monitoring = TaskMonitoring(project_id, location)

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'location': self.location,
            'queues': len(self.queue_manager.queues),
            'tasks': len(self.http_tasks.tasks),
            'features': [
                'http_tasks',
                'appengine_tasks',
                'task_scheduling',
                'rate_limiting',
                'retry_configuration',
                'batch_creation'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Cloud Tasks capabilities"""
    print("=" * 60)
    print("Cloud Tasks Comprehensive Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'
    location = 'us-central1'

    # Initialize manager
    mgr = CloudTasksManager(project_id, location)

    # Create queue
    queue_result = mgr.queue_manager.create_queue({
        'queue_name': 'api-tasks',
        'max_concurrent_dispatches': 500,
        'max_dispatches_per_second': 100.0,
        'max_attempts': 5
    })

    # Create HTTP task
    task_result = mgr.http_tasks.create_http_task({
        'queue_name': 'api-tasks',
        'task_name': 'process-user-123',
        'url': 'https://api.example.com/process',
        'http_method': 'POST',
        'payload': {'user_id': 123, 'action': 'verify'},
        'schedule_delay_seconds': 60  # Delay 1 minute
    })

    # Create batch tasks
    batch_result = mgr.http_tasks.create_batch_tasks({
        'queue_name': 'api-tasks',
        'url': 'https://api.example.com/batch',
        'items': [
            {'id': 1, 'value': 'a'},
            {'id': 2, 'value': 'b'},
            {'id': 3, 'value': 'c'},
        ]
    })

    # Create App Engine task
    appengine_result = mgr.appengine_tasks.create_appengine_task({
        'queue_name': 'api-tasks',
        'relative_uri': '/tasks/cleanup',
        'http_method': 'POST',
        'payload': {'type': 'cleanup', 'days': 30}
    })

    # Monitoring
    list_code = mgr.monitoring.list_tasks('api-tasks')

    # Delete a task
    delete_result = mgr.monitoring.delete_task('api-tasks', 'process-user-123')

    # Pause queue
    pause_result = mgr.queue_manager.pause_queue('api-tasks')

    # Manager info
    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Cloud Tasks Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Location: {info['location']}")
    print(f"Queues: {info['queues']}")
    print(f"Tasks: {info['tasks']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")
    print("\nCloud Tasks Best Practices:")
    print("  1. Configure appropriate rate limits per queue")
    print("  2. Use task scheduling for delayed execution")
    print("  3. Implement idempotent task handlers")
    print("  4. Set appropriate retry configurations")
    print("  5. Monitor task execution and failure rates")
    print("  6. Use named tasks for deduplication")


if __name__ == "__main__":
    demo()
