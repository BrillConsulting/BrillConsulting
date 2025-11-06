"""
Dataflow - Stream and Batch Processing with Apache Beam
Author: BrillConsulting
Description: Comprehensive data processing pipelines for streaming and batch workloads
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class BatchPipeline:
    """Create and manage batch processing pipelines"""

    def __init__(self, project_id: str, region: str = 'us-central1'):
        """
        Initialize batch pipeline manager

        Args:
            project_id: GCP project ID
            region: Dataflow region
        """
        self.project_id = project_id
        self.region = region
        self.pipelines = []

    def create_batch_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create batch processing pipeline

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline creation result
        """
        print(f"\n{'='*60}")
        print("Creating Batch Pipeline")
        print(f"{'='*60}")

        job_name = config.get('job_name', 'batch-processing')
        input_path = config.get('input_path', 'gs://bucket/input/*')
        output_path = config.get('output_path', 'gs://bucket/output/')
        temp_location = config.get('temp_location', 'gs://bucket/temp/')

        code = f"""
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions

# Configure pipeline options
options = PipelineOptions()

# Google Cloud options
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = '{self.project_id}'
google_cloud_options.job_name = '{job_name}'
google_cloud_options.staging_location = '{temp_location}/staging'
google_cloud_options.temp_location = '{temp_location}'
google_cloud_options.region = '{self.region}'

# Standard options
standard_options = options.view_as(StandardOptions)
standard_options.runner = 'DataflowRunner'

# Define pipeline
with beam.Pipeline(options=options) as pipeline:
    # Read input data
    lines = pipeline | 'Read' >> beam.io.ReadFromText('{input_path}')

    # Transform data
    processed = (lines
        | 'Parse JSON' >> beam.Map(lambda x: json.loads(x))
        | 'Filter' >> beam.Filter(lambda x: x.get('active', False))
        | 'Transform' >> beam.Map(lambda x: {{
            'id': x['id'],
            'value': x['value'] * 2,
            'timestamp': x['timestamp']
        }})
        | 'Format' >> beam.Map(lambda x: json.dumps(x))
    )

    # Write output
    processed | 'Write' >> beam.io.WriteToText(
        '{output_path}/result',
        file_name_suffix='.json'
    )

print(f"Pipeline {{'{job_name}'}} submitted successfully")
"""

        result = {
            'job_name': job_name,
            'pipeline_type': 'BATCH',
            'input_path': input_path,
            'output_path': output_path,
            'temp_location': temp_location,
            'region': self.region,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.pipelines.append(result)

        print(f"✓ Batch pipeline created: {job_name}")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Region: {self.region}")
        print(f"{'='*60}")

        return result

    def create_etl_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create ETL (Extract, Transform, Load) pipeline

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline creation result
        """
        print(f"\n{'='*60}")
        print("Creating ETL Pipeline")
        print(f"{'='*60}")

        job_name = config.get('job_name', 'etl-pipeline')
        source_table = config.get('source_table', 'project.dataset.source')
        dest_table = config.get('dest_table', 'project.dataset.destination')

        code = f"""
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.gcp.bigquery import ReadFromBigQuery, WriteToBigQuery

# Pipeline options
options = PipelineOptions(
    project='{self.project_id}',
    region='{self.region}',
    runner='DataflowRunner',
    temp_location='gs://bucket/temp',
)

# Define transformation
class TransformRow(beam.DoFn):
    def process(self, element):
        # Extract
        user_id = element.get('user_id')
        events = element.get('events', [])

        # Transform
        total_events = len(events)
        active_events = sum(1 for e in events if e.get('active', False))

        # Load format
        yield {{
            'user_id': user_id,
            'total_events': total_events,
            'active_events': active_events,
            'activity_rate': active_events / total_events if total_events > 0 else 0,
            'processed_at': beam.utils.timestamp.Timestamp.now().to_utc_datetime().isoformat()
        }}

# Pipeline
with beam.Pipeline(options=options) as pipeline:
    (pipeline
        # Extract from BigQuery
        | 'Read from BigQuery' >> ReadFromBigQuery(
            query='SELECT * FROM `{source_table}` WHERE DATE(timestamp) = CURRENT_DATE()',
            use_standard_sql=True
        )
        # Transform
        | 'Transform Data' >> beam.ParDo(TransformRow())
        # Load to BigQuery
        | 'Write to BigQuery' >> WriteToBigQuery(
            '{dest_table}',
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
        )
    )

print("ETL pipeline submitted")
"""

        result = {
            'job_name': job_name,
            'pipeline_type': 'ETL',
            'source_table': source_table,
            'dest_table': dest_table,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ ETL pipeline created: {job_name}")
        print(f"  Source: {source_table}")
        print(f"  Destination: {dest_table}")
        print(f"{'='*60}")

        return result


class StreamingPipeline:
    """Create and manage streaming pipelines"""

    def __init__(self, project_id: str, region: str = 'us-central1'):
        """Initialize streaming pipeline manager"""
        self.project_id = project_id
        self.region = region
        self.pipelines = []

    def create_streaming_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create streaming pipeline with windowing

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline creation result
        """
        print(f"\n{'='*60}")
        print("Creating Streaming Pipeline")
        print(f"{'='*60}")

        job_name = config.get('job_name', 'streaming-pipeline')
        pubsub_topic = config.get('pubsub_topic', 'projects/project/topics/events')
        output_table = config.get('output_table', 'project.dataset.streaming_results')
        window_duration = config.get('window_duration_seconds', 60)

        code = f"""
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from apache_beam.transforms import window
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.io.gcp.bigquery import WriteToBigQuery
import json

# Pipeline options
options = PipelineOptions(
    project='{self.project_id}',
    region='{self.region}',
    streaming=True,
    runner='DataflowRunner',
)

# Enable streaming
standard_options = options.view_as(StandardOptions)
standard_options.streaming = True

# Transform function
class ParseAndAggregate(beam.DoFn):
    def process(self, element, window=beam.DoFn.WindowParam):
        data = json.loads(element.decode('utf-8'))

        yield {{
            'user_id': data.get('user_id'),
            'event_type': data.get('event_type'),
            'value': data.get('value', 0),
            'window_start': window.start.to_utc_datetime().isoformat(),
            'window_end': window.end.to_utc_datetime().isoformat(),
        }}

# Pipeline
with beam.Pipeline(options=options) as pipeline:
    (pipeline
        # Read from Pub/Sub
        | 'Read from Pub/Sub' >> ReadFromPubSub(topic='{pubsub_topic}')

        # Apply windowing (fixed windows of {window_duration} seconds)
        | 'Window' >> beam.WindowInto(
            window.FixedWindows({window_duration}),
            trigger=window.AfterWatermark(
                early=window.AfterProcessingTime(10),  # Early firing every 10 seconds
                late=window.AfterCount(1)  # Late data handling
            ),
            accumulation_mode=window.AccumulationMode.DISCARDING
        )

        # Parse and process
        | 'Parse' >> beam.ParDo(ParseAndAggregate())

        # Group by user_id and aggregate
        | 'Key by user' >> beam.Map(lambda x: (x['user_id'], x))
        | 'Group' >> beam.GroupByKey()
        | 'Aggregate' >> beam.Map(lambda x: {{
            'user_id': x[0],
            'event_count': len(x[1]),
            'total_value': sum(item['value'] for item in x[1]),
            'window_start': x[1][0]['window_start'],
            'window_end': x[1][0]['window_end'],
        }})

        # Write to BigQuery
        | 'Write to BigQuery' >> WriteToBigQuery(
            '{output_table}',
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
        )
    )

print("Streaming pipeline submitted")
"""

        result = {
            'job_name': job_name,
            'pipeline_type': 'STREAMING',
            'pubsub_topic': pubsub_topic,
            'output_table': output_table,
            'window_duration_seconds': window_duration,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.pipelines.append(result)

        print(f"✓ Streaming pipeline created: {job_name}")
        print(f"  Pub/Sub topic: {pubsub_topic}")
        print(f"  Output table: {output_table}")
        print(f"  Window: {window_duration}s")
        print(f"{'='*60}")

        return result


class PipelineTemplates:
    """Manage Dataflow templates"""

    def __init__(self, project_id: str):
        """Initialize template manager"""
        self.project_id = project_id

    def create_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create reusable pipeline template

        Args:
            config: Template configuration

        Returns:
            Template creation result
        """
        print(f"\n{'='*60}")
        print("Creating Pipeline Template")
        print(f"{'='*60}")

        template_name = config.get('template_name', 'my-template')
        template_path = config.get('template_path', 'gs://bucket/templates/my-template')

        code = f"""
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.io.gcp.bigquery import WriteToBigQuery

# Template options with runtime parameters
class MyOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument(
            '--input_path',
            type=str,
            help='Input file path'
        )
        parser.add_value_provider_argument(
            '--output_table',
            type=str,
            help='Output BigQuery table'
        )
        parser.add_value_provider_argument(
            '--filter_field',
            type=str,
            default='active',
            help='Field to filter on'
        )

# Configure options
options = MyOptions()
setup_options = options.view_as(SetupOptions)
setup_options.save_main_session = True

# Pipeline (parameterized)
with beam.Pipeline(options=options) as pipeline:
    (pipeline
        | 'Read' >> beam.io.ReadFromText(options.input_path)
        | 'Parse' >> beam.Map(lambda x: json.loads(x))
        | 'Filter' >> beam.Filter(
            lambda x: x.get(options.filter_field.get(), False)
        )
        | 'Write' >> WriteToBigQuery(
            options.output_table,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
        )
    )

# Create template
# Run with: --runner=DataflowRunner --template_location={template_path}
print(f"Template created at: {template_path}")
"""

        result = {
            'template_name': template_name,
            'template_path': template_path,
            'parameters': ['input_path', 'output_table', 'filter_field'],
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Template created: {template_name}")
        print(f"  Path: {template_path}")
        print(f"  Parameters: {', '.join(result['parameters'])}")
        print(f"{'='*60}")

        return result

    def run_template(self, config: Dict[str, Any]) -> str:
        """
        Run a template with parameters

        Args:
            config: Template execution configuration

        Returns:
            Template execution code
        """
        template_path = config.get('template_path', 'gs://bucket/templates/my-template')
        job_name = config.get('job_name', 'template-job')
        parameters = config.get('parameters', {})

        code = f"""
from google.cloud import dataflow_v1beta3

client = dataflow_v1beta3.TemplatesServiceClient()

# Launch template
request = dataflow_v1beta3.LaunchTemplateRequest(
    project_id='{self.project_id}',
    gcs_path='{template_path}',
    job_name='{job_name}',
    parameters={parameters},
    environment=dataflow_v1beta3.RuntimeEnvironment(
        max_workers=10,
        zone='us-central1-a',
        service_account_email='dataflow@{self.project_id}.iam.gserviceaccount.com',
    )
)

response = client.launch_template(request=request)
print(f"Template launched: {{response.job.name}}")
print(f"Job ID: {{response.job.id}}")
"""

        print(f"\n✓ Template execution code generated")
        print(f"  Template: {template_path}")
        return code


class PipelineMonitoring:
    """Monitor and manage Dataflow jobs"""

    def __init__(self, project_id: str, region: str = 'us-central1'):
        """Initialize monitoring"""
        self.project_id = project_id
        self.region = region

    def list_jobs(self) -> str:
        """
        List Dataflow jobs

        Returns:
            Code to list jobs
        """
        code = f"""
from google.cloud import dataflow_v1beta3

client = dataflow_v1beta3.JobsV1Beta3Client()

# List jobs
request = dataflow_v1beta3.ListJobsRequest(
    project_id='{self.project_id}',
    location='{self.region}'
)

print("Dataflow Jobs:")
print("=" * 60)

for job in client.list_jobs(request=request):
    print(f"Job: {{job.name}}")
    print(f"  ID: {{job.id}}")
    print(f"  Type: {{job.type_.name}}")
    print(f"  State: {{job.current_state.name}}")
    print(f"  Created: {{job.create_time}}")
    print("-" * 60)
"""

        print("\n✓ Job listing code generated")
        return code

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running job

        Args:
            job_id: Job ID

        Returns:
            Cancel operation result
        """
        print(f"\n{'='*60}")
        print("Cancelling Job")
        print(f"{'='*60}")

        code = f"""
from google.cloud import dataflow_v1beta3

client = dataflow_v1beta3.JobsV1Beta3Client()

# Cancel job
request = dataflow_v1beta3.UpdateJobRequest(
    project_id='{self.project_id}',
    location='{self.region}',
    job_id='{job_id}',
    job=dataflow_v1beta3.Job(
        id='{job_id}',
        requested_state=dataflow_v1beta3.JobState.JOB_STATE_CANCELLED
    )
)

response = client.update_job(request=request)
print(f"Job cancelled: {{response.name}}")
print(f"State: {{response.current_state.name}}")
"""

        result = {
            'job_id': job_id,
            'action': 'CANCELLED',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Job cancelled: {job_id}")
        print(f"{'='*60}")

        return result


class DataflowManager:
    """Comprehensive Dataflow management"""

    def __init__(self, project_id: str = 'my-project', region: str = 'us-central1'):
        """
        Initialize Dataflow manager

        Args:
            project_id: GCP project ID
            region: Dataflow region
        """
        self.project_id = project_id
        self.region = region
        self.batch = BatchPipeline(project_id, region)
        self.streaming = StreamingPipeline(project_id, region)
        self.templates = PipelineTemplates(project_id)
        self.monitoring = PipelineMonitoring(project_id, region)

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'region': self.region,
            'batch_pipelines': len(self.batch.pipelines),
            'streaming_pipelines': len(self.streaming.pipelines),
            'features': [
                'batch_processing',
                'streaming_processing',
                'etl_pipelines',
                'windowing',
                'templates',
                'monitoring'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Dataflow capabilities"""
    print("=" * 60)
    print("Dataflow Comprehensive Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'
    region = 'us-central1'

    # Initialize manager
    mgr = DataflowManager(project_id, region)

    # Create batch pipeline
    batch_result = mgr.batch.create_batch_pipeline({
        'job_name': 'daily-batch-processing',
        'input_path': 'gs://my-bucket/input/data-*.json',
        'output_path': 'gs://my-bucket/output/',
        'temp_location': 'gs://my-bucket/temp/'
    })

    # Create ETL pipeline
    etl_result = mgr.batch.create_etl_pipeline({
        'job_name': 'user-analytics-etl',
        'source_table': 'my-project.raw_data.user_events',
        'dest_table': 'my-project.analytics.user_metrics'
    })

    # Create streaming pipeline
    streaming_result = mgr.streaming.create_streaming_pipeline({
        'job_name': 'realtime-analytics',
        'pubsub_topic': 'projects/my-project/topics/events',
        'output_table': 'my-project.realtime.aggregated_metrics',
        'window_duration_seconds': 60
    })

    # Create template
    template_result = mgr.templates.create_template({
        'template_name': 'data-processor',
        'template_path': 'gs://my-bucket/templates/data-processor'
    })

    # Run template
    template_run_code = mgr.templates.run_template({
        'template_path': 'gs://my-bucket/templates/data-processor',
        'job_name': 'process-daily-data',
        'parameters': {
            'input_path': 'gs://my-bucket/daily/2025-11-06/*.json',
            'output_table': 'my-project.processed.daily_metrics'
        }
    })

    # Monitoring
    list_code = mgr.monitoring.list_jobs()

    # Manager info
    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Dataflow Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Region: {info['region']}")
    print(f"Batch pipelines: {info['batch_pipelines']}")
    print(f"Streaming pipelines: {info['streaming_pipelines']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")
    print("\nDataflow Best Practices:")
    print("  1. Use streaming pipelines for real-time processing")
    print("  2. Apply windowing for time-based aggregations")
    print("  3. Use templates for reusable pipelines")
    print("  4. Configure appropriate autoscaling")
    print("  5. Monitor pipeline metrics and errors")
    print("  6. Use side inputs for enrichment data")


if __name__ == "__main__":
    demo()
