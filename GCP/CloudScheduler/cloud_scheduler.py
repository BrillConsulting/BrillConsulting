"""
Cloud Scheduler - Managed Cron Job Service
Author: BrillConsulting
Description: Comprehensive cron job scheduling with HTTP, Pub/Sub, and App Engine targets
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class SchedulerJob:
    """Create and manage scheduled jobs"""

    def __init__(self, project_id: str, location: str = 'us-central1'):
        """
        Initialize scheduler job manager

        Args:
            project_id: GCP project ID
            location: Cloud Scheduler location
        """
        self.project_id = project_id
        self.location = location
        self.jobs = []

    def create_http_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create HTTP target scheduled job

        Args:
            config: Job configuration

        Returns:
            Job creation result
        """
        print(f"\n{'='*60}")
        print("Creating HTTP Scheduled Job")
        print(f"{'='*60}")

        job_name = config.get('job_name', 'my-http-job')
        schedule = config.get('schedule', '0 */6 * * *')  # Every 6 hours
        uri = config.get('uri', 'https://example.com/api/task')
        http_method = config.get('http_method', 'POST')
        body = config.get('body', {})
        headers = config.get('headers', {'Content-Type': 'application/json'})

        code = f"""
from google.cloud import scheduler_v1

client = scheduler_v1.CloudSchedulerClient()

# Build the parent path
parent = f"projects/{self.project_id}/locations/{self.location}"

# Define the job
job = {{
    "name": f"{{parent}}/jobs/{job_name}",
    "description": "HTTP scheduled job",
    "schedule": "{schedule}",
    "time_zone": "America/New_York",

    "http_target": {{
        "uri": "{uri}",
        "http_method": scheduler_v1.HttpMethod.{http_method},
        "headers": {headers},
        "body": '{json.dumps(body)}'.encode('utf-8'),
    }},

    "retry_config": {{
        "retry_count": 3,
        "max_retry_duration": {{"seconds": 300}},
        "min_backoff_duration": {{"seconds": 5}},
        "max_backoff_duration": {{"seconds": 60}},
        "max_doublings": 3,
    }},
}}

# Create the job
response = client.create_job(parent=parent, job=job)
print(f"Created job: {{response.name}}")
"""

        result = {
            'job_name': job_name,
            'schedule': schedule,
            'target_type': 'HTTP',
            'uri': uri,
            'http_method': http_method,
            'full_name': f"projects/{self.project_id}/locations/{self.location}/jobs/{job_name}",
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.jobs.append(result)

        print(f"✓ HTTP job created: {job_name}")
        print(f"  Schedule: {schedule}")
        print(f"  Target URI: {uri}")
        print(f"  Method: {http_method}")
        print(f"{'='*60}")

        return result

    def create_pubsub_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Pub/Sub target scheduled job

        Args:
            config: Job configuration

        Returns:
            Job creation result
        """
        print(f"\n{'='*60}")
        print("Creating Pub/Sub Scheduled Job")
        print(f"{'='*60}")

        job_name = config.get('job_name', 'my-pubsub-job')
        schedule = config.get('schedule', '*/15 * * * *')  # Every 15 minutes
        topic_name = config.get('topic_name', 'scheduled-tasks')
        message = config.get('message', {'task': 'process_data'})

        code = f"""
from google.cloud import scheduler_v1

client = scheduler_v1.CloudSchedulerClient()

# Build the parent path
parent = f"projects/{self.project_id}/locations/{self.location}"

# Define the job with Pub/Sub target
job = {{
    "name": f"{{parent}}/jobs/{job_name}",
    "description": "Pub/Sub scheduled job",
    "schedule": "{schedule}",
    "time_zone": "UTC",

    "pubsub_target": {{
        "topic_name": f"projects/{self.project_id}/topics/{topic_name}",
        "data": '{json.dumps(message)}'.encode('utf-8'),
        "attributes": {{
            "source": "cloud-scheduler",
            "job_name": "{job_name}",
        }},
    }},

    "retry_config": {{
        "retry_count": 5,
        "max_retry_duration": {{"seconds": 600}},
    }},
}}

# Create the job
response = client.create_job(parent=parent, job=job)
print(f"Created Pub/Sub job: {{response.name}}")
"""

        result = {
            'job_name': job_name,
            'schedule': schedule,
            'target_type': 'PUBSUB',
            'topic_name': topic_name,
            'full_name': f"projects/{self.project_id}/locations/{self.location}/jobs/{job_name}",
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.jobs.append(result)

        print(f"✓ Pub/Sub job created: {job_name}")
        print(f"  Schedule: {schedule}")
        print(f"  Topic: {topic_name}")
        print(f"{'='*60}")

        return result

    def create_app_engine_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create App Engine target scheduled job

        Args:
            config: Job configuration

        Returns:
            Job creation result
        """
        print(f"\n{'='*60}")
        print("Creating App Engine Scheduled Job")
        print(f"{'='*60}")

        job_name = config.get('job_name', 'my-appengine-job')
        schedule = config.get('schedule', '0 2 * * *')  # Daily at 2 AM
        relative_uri = config.get('relative_uri', '/tasks/cleanup')
        http_method = config.get('http_method', 'POST')

        code = f"""
from google.cloud import scheduler_v1

client = scheduler_v1.CloudSchedulerClient()

# Build the parent path
parent = f"projects/{self.project_id}/locations/{self.location}"

# Define the job with App Engine target
job = {{
    "name": f"{{parent}}/jobs/{job_name}",
    "description": "App Engine scheduled job",
    "schedule": "{schedule}",
    "time_zone": "America/Los_Angeles",

    "app_engine_http_target": {{
        "relative_uri": "{relative_uri}",
        "http_method": scheduler_v1.HttpMethod.{http_method},
        "app_engine_routing": {{
            "service": "default",
            "version": "v1",
        }},
    }},
}}

# Create the job
response = client.create_job(parent=parent, job=job)
print(f"Created App Engine job: {{response.name}}")
"""

        result = {
            'job_name': job_name,
            'schedule': schedule,
            'target_type': 'APP_ENGINE',
            'relative_uri': relative_uri,
            'full_name': f"projects/{self.project_id}/locations/{self.location}/jobs/{job_name}",
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.jobs.append(result)

        print(f"✓ App Engine job created: {job_name}")
        print(f"  Schedule: {schedule}")
        print(f"  URI: {relative_uri}")
        print(f"{'='*60}")

        return result


class SchedulerMonitoring:
    """Monitor and manage scheduled jobs"""

    def __init__(self, project_id: str, location: str = 'us-central1'):
        """Initialize monitoring"""
        self.project_id = project_id
        self.location = location

    def list_jobs(self) -> str:
        """
        List all scheduled jobs

        Returns:
            Code to list jobs
        """
        code = f"""
from google.cloud import scheduler_v1

client = scheduler_v1.CloudSchedulerClient()

# List all jobs
parent = f"projects/{self.project_id}/locations/{self.location}"

print("Scheduled Jobs:")
print("=" * 60)

for job in client.list_jobs(parent=parent):
    print(f"Name: {{job.name}}")
    print(f"  Schedule: {{job.schedule}}")
    print(f"  State: {{job.state.name}}")
    print(f"  Last attempt: {{job.status.last_attempt_time if job.status else 'N/A'}}")
    print(f"  Next run: {{job.schedule_time}}")
    print("-" * 60)
"""

        print("\n✓ Job listing code generated")
        return code

    def pause_job(self, job_name: str) -> Dict[str, Any]:
        """
        Pause a scheduled job

        Args:
            job_name: Job name

        Returns:
            Pause operation result
        """
        print(f"\n{'='*60}")
        print("Pausing Scheduled Job")
        print(f"{'='*60}")

        code = f"""
from google.cloud import scheduler_v1

client = scheduler_v1.CloudSchedulerClient()

# Build job path
name = f"projects/{self.project_id}/locations/{self.location}/jobs/{job_name}"

# Pause the job
response = client.pause_job(name=name)
print(f"Paused job: {{response.name}}")
print(f"State: {{response.state.name}}")
"""

        result = {
            'job_name': job_name,
            'action': 'PAUSED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Job paused: {job_name}")
        print(f"{'='*60}")

        return result

    def resume_job(self, job_name: str) -> Dict[str, Any]:
        """
        Resume a paused job

        Args:
            job_name: Job name

        Returns:
            Resume operation result
        """
        print(f"\n{'='*60}")
        print("Resuming Scheduled Job")
        print(f"{'='*60}")

        code = f"""
from google.cloud import scheduler_v1

client = scheduler_v1.CloudSchedulerClient()

# Build job path
name = f"projects/{self.project_id}/locations/{self.location}/jobs/{job_name}"

# Resume the job
response = client.resume_job(name=name)
print(f"Resumed job: {{response.name}}")
print(f"State: {{response.state.name}}")
"""

        result = {
            'job_name': job_name,
            'action': 'RESUMED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Job resumed: {job_name}")
        print(f"{'='*60}")

        return result

    def run_job_now(self, job_name: str) -> Dict[str, Any]:
        """
        Force run a job immediately

        Args:
            job_name: Job name

        Returns:
            Run operation result
        """
        print(f"\n{'='*60}")
        print("Running Job Immediately")
        print(f"{'='*60}")

        code = f"""
from google.cloud import scheduler_v1

client = scheduler_v1.CloudSchedulerClient()

# Build job path
name = f"projects/{self.project_id}/locations/{self.location}/jobs/{job_name}"

# Run the job now
response = client.run_job(name=name)
print(f"Job triggered: {{response.name}}")
print(f"Execution time: {{response.schedule_time}}")
"""

        result = {
            'job_name': job_name,
            'action': 'TRIGGERED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Job triggered: {job_name}")
        print(f"  Execution started immediately")
        print(f"{'='*60}")

        return result


class CronExpressions:
    """Helper for cron expression patterns"""

    @staticmethod
    def get_common_schedules() -> Dict[str, str]:
        """
        Get common cron schedules

        Returns:
            Dictionary of common schedules
        """
        schedules = {
            'every_minute': '* * * * *',
            'every_5_minutes': '*/5 * * * *',
            'every_15_minutes': '*/15 * * * *',
            'every_30_minutes': '*/30 * * * *',
            'hourly': '0 * * * *',
            'every_6_hours': '0 */6 * * *',
            'daily_midnight': '0 0 * * *',
            'daily_noon': '0 12 * * *',
            'daily_2am': '0 2 * * *',
            'weekly_monday': '0 0 * * 1',
            'weekly_sunday': '0 0 * * 0',
            'monthly_first': '0 0 1 * *',
            'monthly_last': '0 0 L * *',
            'yearly': '0 0 1 1 *',
        }

        print("\nCommon Cron Schedules:")
        print("=" * 60)
        for name, schedule in schedules.items():
            print(f"{name:20s} → {schedule}")
        print("=" * 60)

        return schedules

    @staticmethod
    def validate_schedule(schedule: str) -> bool:
        """
        Basic cron schedule validation

        Args:
            schedule: Cron expression

        Returns:
            True if valid format
        """
        parts = schedule.split()
        if len(parts) != 5:
            return False

        # Basic validation (not comprehensive)
        print(f"✓ Schedule format valid: {schedule}")
        return True


class CloudSchedulerManager:
    """Comprehensive Cloud Scheduler management"""

    def __init__(self, project_id: str = 'my-project', location: str = 'us-central1'):
        """
        Initialize Cloud Scheduler manager

        Args:
            project_id: GCP project ID
            location: Scheduler location
        """
        self.project_id = project_id
        self.location = location
        self.job_manager = SchedulerJob(project_id, location)
        self.monitoring = SchedulerMonitoring(project_id, location)
        self.cron = CronExpressions()

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'location': self.location,
            'jobs_created': len(self.job_manager.jobs),
            'features': [
                'http_targets',
                'pubsub_targets',
                'app_engine_targets',
                'retry_configuration',
                'job_monitoring',
                'cron_expressions'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Cloud Scheduler capabilities"""
    print("=" * 60)
    print("Cloud Scheduler Comprehensive Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'
    location = 'us-central1'

    # Initialize manager
    mgr = CloudSchedulerManager(project_id, location)

    # Show common schedules
    schedules = mgr.cron.get_common_schedules()

    # Create HTTP job
    http_job = mgr.job_manager.create_http_job({
        'job_name': 'api-health-check',
        'schedule': '*/5 * * * *',  # Every 5 minutes
        'uri': 'https://api.example.com/health',
        'http_method': 'GET',
        'headers': {'Authorization': 'Bearer token'}
    })

    # Create Pub/Sub job
    pubsub_job = mgr.job_manager.create_pubsub_job({
        'job_name': 'data-processing',
        'schedule': '0 */2 * * *',  # Every 2 hours
        'topic_name': 'data-process-trigger',
        'message': {'action': 'process', 'batch_size': 1000}
    })

    # Create App Engine job
    appengine_job = mgr.job_manager.create_app_engine_job({
        'job_name': 'daily-cleanup',
        'schedule': '0 3 * * *',  # Daily at 3 AM
        'relative_uri': '/tasks/cleanup',
        'http_method': 'POST'
    })

    # Monitoring operations
    list_code = mgr.monitoring.list_jobs()

    # Pause job
    pause_result = mgr.monitoring.pause_job('data-processing')

    # Resume job
    resume_result = mgr.monitoring.resume_job('data-processing')

    # Run job now
    run_result = mgr.monitoring.run_job_now('api-health-check')

    # Manager info
    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Cloud Scheduler Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Location: {info['location']}")
    print(f"Jobs created: {info['jobs_created']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")
    print("\nCloud Scheduler Best Practices:")
    print("  1. Use appropriate retry configurations")
    print("  2. Set time zones explicitly")
    print("  3. Monitor job execution status")
    print("  4. Use Pub/Sub for reliable job processing")
    print("  5. Implement idempotent job handlers")


if __name__ == "__main__":
    demo()
