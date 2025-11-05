"""
AWS Glue
========

Fully managed ETL service and data catalog for data lakes.

Author: Brill Consulting
"""

import boto3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GlueManager:
    """
    Advanced AWS Glue Management System

    Provides comprehensive Glue operations including:
    - Data Catalog (databases, tables, partitions)
    - Crawlers for automatic schema discovery
    - ETL Jobs (Spark and Python Shell)
    - Job runs and monitoring
    - Triggers and workflows
    - Schema registry
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize Glue Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.glue_client = session.client('glue', region_name=region)
            self.region = region
            logger.info(f"Glue Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing Glue Manager: {e}")
            raise

    # ==================== Data Catalog - Databases ====================

    def create_database(
        self,
        database_name: str,
        description: Optional[str] = None,
        location_uri: Optional[str] = None
    ) -> None:
        """
        Create database in Data Catalog.

        Args:
            database_name: Database name
            description: Database description
            location_uri: S3 location for database
        """
        try:
            logger.info(f"Creating database: {database_name}")

            database_input = {'Name': database_name}

            if description:
                database_input['Description'] = description

            if location_uri:
                database_input['LocationUri'] = location_uri

            self.glue_client.create_database(DatabaseInput=database_input)

            logger.info(f"✓ Database created: {database_name}")

        except ClientError as e:
            logger.error(f"Error creating database: {e}")
            raise

    def get_database(self, database_name: str) -> Dict[str, Any]:
        """Get database details."""
        try:
            response = self.glue_client.get_database(Name=database_name)

            db = response['Database']
            return {
                'name': db['Name'],
                'description': db.get('Description', ''),
                'location_uri': db.get('LocationUri', ''),
                'create_time': db['CreateTime'].isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting database: {e}")
            raise

    def list_databases(self) -> List[str]:
        """List all databases."""
        try:
            response = self.glue_client.get_databases()

            databases = [db['Name'] for db in response.get('DatabaseList', [])]
            logger.info(f"Found {len(databases)} database(s)")

            return databases

        except ClientError as e:
            logger.error(f"Error listing databases: {e}")
            raise

    def delete_database(self, database_name: str) -> None:
        """Delete database."""
        try:
            self.glue_client.delete_database(Name=database_name)
            logger.info(f"✓ Database deleted: {database_name}")

        except ClientError as e:
            logger.error(f"Error deleting database: {e}")
            raise

    # ==================== Data Catalog - Tables ====================

    def create_table(
        self,
        database_name: str,
        table_name: str,
        storage_location: str,
        columns: List[Dict[str, str]],
        partition_keys: Optional[List[Dict[str, str]]] = None,
        input_format: str = "org.apache.hadoop.mapred.TextInputFormat",
        output_format: str = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
        serde_info: Optional[Dict[str, Any]] = None,
        table_type: str = "EXTERNAL_TABLE"
    ) -> None:
        """
        Create table in Data Catalog.

        Args:
            database_name: Database name
            table_name: Table name
            storage_location: S3 location
            columns: List of {'Name': 'col1', 'Type': 'string', 'Comment': '...'}
            partition_keys: Partition columns
            input_format: Input format class
            output_format: Output format class
            serde_info: SerDe information
            table_type: 'EXTERNAL_TABLE' or 'VIRTUAL_VIEW'
        """
        try:
            logger.info(f"Creating table: {database_name}.{table_name}")

            storage_descriptor = {
                'Columns': columns,
                'Location': storage_location,
                'InputFormat': input_format,
                'OutputFormat': output_format,
                'SerdeInfo': serde_info or {
                    'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
                }
            }

            table_input = {
                'Name': table_name,
                'StorageDescriptor': storage_descriptor,
                'TableType': table_type
            }

            if partition_keys:
                table_input['PartitionKeys'] = partition_keys

            self.glue_client.create_table(
                DatabaseName=database_name,
                TableInput=table_input
            )

            logger.info(f"✓ Table created: {database_name}.{table_name}")

        except ClientError as e:
            logger.error(f"Error creating table: {e}")
            raise

    def get_table(self, database_name: str, table_name: str) -> Dict[str, Any]:
        """Get table details."""
        try:
            response = self.glue_client.get_table(
                DatabaseName=database_name,
                Name=table_name
            )

            table = response['Table']
            return {
                'name': table['Name'],
                'database': table['DatabaseName'],
                'location': table['StorageDescriptor']['Location'],
                'columns': table['StorageDescriptor']['Columns'],
                'partition_keys': table.get('PartitionKeys', []),
                'create_time': table['CreateTime'].isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting table: {e}")
            raise

    def list_tables(self, database_name: str) -> List[str]:
        """List tables in database."""
        try:
            response = self.glue_client.get_tables(DatabaseName=database_name)

            tables = [table['Name'] for table in response.get('TableList', [])]
            logger.info(f"Found {len(tables)} table(s) in {database_name}")

            return tables

        except ClientError as e:
            logger.error(f"Error listing tables: {e}")
            raise

    def delete_table(self, database_name: str, table_name: str) -> None:
        """Delete table."""
        try:
            self.glue_client.delete_table(
                DatabaseName=database_name,
                Name=table_name
            )
            logger.info(f"✓ Table deleted: {database_name}.{table_name}")

        except ClientError as e:
            logger.error(f"Error deleting table: {e}")
            raise

    # ==================== Crawlers ====================

    def create_crawler(
        self,
        crawler_name: str,
        role_arn: str,
        database_name: str,
        s3_targets: List[str],
        description: Optional[str] = None,
        schedule: Optional[str] = None
    ) -> None:
        """
        Create crawler for automatic schema discovery.

        Args:
            crawler_name: Crawler name
            role_arn: IAM role ARN
            database_name: Target database
            s3_targets: List of S3 paths to crawl
            description: Crawler description
            schedule: Cron expression (e.g., 'cron(0 12 * * ? *)')
        """
        try:
            logger.info(f"Creating crawler: {crawler_name}")

            params = {
                'Name': crawler_name,
                'Role': role_arn,
                'DatabaseName': database_name,
                'Targets': {
                    'S3Targets': [{'Path': path} for path in s3_targets]
                }
            }

            if description:
                params['Description'] = description

            if schedule:
                params['Schedule'] = schedule

            self.glue_client.create_crawler(**params)

            logger.info(f"✓ Crawler created: {crawler_name}")

        except ClientError as e:
            logger.error(f"Error creating crawler: {e}")
            raise

    def start_crawler(self, crawler_name: str) -> None:
        """Start crawler run."""
        try:
            self.glue_client.start_crawler(Name=crawler_name)
            logger.info(f"✓ Crawler started: {crawler_name}")

        except ClientError as e:
            logger.error(f"Error starting crawler: {e}")
            raise

    def stop_crawler(self, crawler_name: str) -> None:
        """Stop crawler run."""
        try:
            self.glue_client.stop_crawler(Name=crawler_name)
            logger.info(f"✓ Crawler stopped: {crawler_name}")

        except ClientError as e:
            logger.error(f"Error stopping crawler: {e}")
            raise

    def get_crawler(self, crawler_name: str) -> Dict[str, Any]:
        """Get crawler details."""
        try:
            response = self.glue_client.get_crawler(Name=crawler_name)

            crawler = response['Crawler']
            return {
                'name': crawler['Name'],
                'state': crawler['State'],
                'database_name': crawler['DatabaseName'],
                'last_crawl': crawler.get('LastCrawl', {})
            }

        except ClientError as e:
            logger.error(f"Error getting crawler: {e}")
            raise

    def delete_crawler(self, crawler_name: str) -> None:
        """Delete crawler."""
        try:
            self.glue_client.delete_crawler(Name=crawler_name)
            logger.info(f"✓ Crawler deleted: {crawler_name}")

        except ClientError as e:
            logger.error(f"Error deleting crawler: {e}")
            raise

    # ==================== ETL Jobs ====================

    def create_job(
        self,
        job_name: str,
        role_arn: str,
        script_location: str,
        glue_version: str = "4.0",
        max_capacity: Optional[float] = None,
        worker_type: Optional[str] = None,
        number_of_workers: Optional[int] = None,
        default_arguments: Optional[Dict[str, str]] = None,
        command_name: str = "glueetl",
        description: Optional[str] = None
    ) -> None:
        """
        Create ETL job.

        Args:
            job_name: Job name
            role_arn: IAM role ARN
            script_location: S3 path to script
            glue_version: Glue version (2.0, 3.0, 4.0)
            max_capacity: DPU capacity (deprecated, use worker_type)
            worker_type: 'Standard', 'G.1X', 'G.2X', 'G.025X'
            number_of_workers: Number of workers
            default_arguments: Job parameters
            command_name: 'glueetl', 'pythonshell', 'gluestreaming'
            description: Job description
        """
        try:
            logger.info(f"Creating job: {job_name}")

            params = {
                'Name': job_name,
                'Role': role_arn,
                'Command': {
                    'Name': command_name,
                    'ScriptLocation': script_location,
                    'PythonVersion': '3'
                },
                'GlueVersion': glue_version
            }

            if max_capacity:
                params['MaxCapacity'] = max_capacity
            elif worker_type and number_of_workers:
                params['WorkerType'] = worker_type
                params['NumberOfWorkers'] = number_of_workers

            if default_arguments:
                params['DefaultArguments'] = default_arguments

            if description:
                params['Description'] = description

            self.glue_client.create_job(**params)

            logger.info(f"✓ Job created: {job_name}")

        except ClientError as e:
            logger.error(f"Error creating job: {e}")
            raise

    def start_job_run(
        self,
        job_name: str,
        arguments: Optional[Dict[str, str]] = None
    ) -> str:
        """Start job run."""
        try:
            logger.info(f"Starting job: {job_name}")

            params = {'JobName': job_name}
            if arguments:
                params['Arguments'] = arguments

            response = self.glue_client.start_job_run(**params)

            job_run_id = response['JobRunId']
            logger.info(f"✓ Job started: {job_run_id}")

            return job_run_id

        except ClientError as e:
            logger.error(f"Error starting job: {e}")
            raise

    def get_job_run(self, job_name: str, job_run_id: str) -> Dict[str, Any]:
        """Get job run status."""
        try:
            response = self.glue_client.get_job_run(
                JobName=job_name,
                RunId=job_run_id
            )

            job_run = response['JobRun']
            return {
                'job_run_id': job_run['Id'],
                'job_run_state': job_run['JobRunState'],
                'started_on': job_run.get('StartedOn', datetime.now()).isoformat(),
                'execution_time': job_run.get('ExecutionTime', 0),
                'error_message': job_run.get('ErrorMessage', '')
            }

        except ClientError as e:
            logger.error(f"Error getting job run: {e}")
            raise

    def list_jobs(self) -> List[str]:
        """List all jobs."""
        try:
            response = self.glue_client.list_jobs()

            jobs = response.get('JobNames', [])
            logger.info(f"Found {len(jobs)} job(s)")

            return jobs

        except ClientError as e:
            logger.error(f"Error listing jobs: {e}")
            raise

    def delete_job(self, job_name: str) -> None:
        """Delete job."""
        try:
            self.glue_client.delete_job(JobName=job_name)
            logger.info(f"✓ Job deleted: {job_name}")

        except ClientError as e:
            logger.error(f"Error deleting job: {e}")
            raise

    # ==================== Monitoring ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get Glue summary."""
        try:
            databases = self.list_databases()
            jobs = self.list_jobs()

            return {
                'region': self.region,
                'databases': len(databases),
                'jobs': len(jobs),
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of Glue Manager capabilities."""
    print("AWS Glue Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    print("\n1️⃣  Data Catalog - Databases:")
    print("""
    glue = GlueManager(region='us-east-1')

    # Create database
    glue.create_database(
        database_name='sales_db',
        description='Sales data lake',
        location_uri='s3://my-data-lake/sales/'
    )

    # List databases
    databases = glue.list_databases()
    """)

    print("\n2️⃣  Data Catalog - Tables:")
    print("""
    # Create table
    glue.create_table(
        database_name='sales_db',
        table_name='orders',
        storage_location='s3://my-data-lake/sales/orders/',
        columns=[
            {'Name': 'order_id', 'Type': 'bigint'},
            {'Name': 'customer_id', 'Type': 'bigint'},
            {'Name': 'amount', 'Type': 'decimal(10,2)'},
            {'Name': 'status', 'Type': 'string'}
        ],
        partition_keys=[
            {'Name': 'year', 'Type': 'int'},
            {'Name': 'month', 'Type': 'int'}
        ]
    )
    """)

    print("\n3️⃣  Crawlers (Automatic Schema Discovery):")
    print("""
    # Create crawler
    glue.create_crawler(
        crawler_name='sales-crawler',
        role_arn='arn:aws:iam::123456789012:role/GlueServiceRole',
        database_name='sales_db',
        s3_targets=['s3://my-data-lake/sales/'],
        schedule='cron(0 12 * * ? *)'  # Daily at noon
    )

    # Start crawler
    glue.start_crawler('sales-crawler')

    # Check status
    status = glue.get_crawler('sales-crawler')
    """)

    print("\n4️⃣  ETL Jobs:")
    print("""
    # Create ETL job
    glue.create_job(
        job_name='transform-sales-data',
        role_arn='arn:aws:iam::123456789012:role/GlueServiceRole',
        script_location='s3://my-scripts/transform.py',
        glue_version='4.0',
        worker_type='G.1X',
        number_of_workers=5,
        default_arguments={
            '--TempDir': 's3://my-temp-dir/',
            '--job-bookmark-option': 'job-bookmark-enable'
        }
    )

    # Start job
    job_run_id = glue.start_job_run(
        job_name='transform-sales-data',
        arguments={'--input': 's3://input/', '--output': 's3://output/'}
    )

    # Monitor job
    status = glue.get_job_run('transform-sales-data', job_run_id)
    print(f"Job state: {status['job_run_state']}")
    """)

    print("\n5️⃣  Typical ETL Pipeline:")
    print("""
    # 1. Crawl source data
    glue.start_crawler('source-crawler')

    # 2. Wait for crawler to complete
    # (Poll crawler status)

    # 3. Run transformation job
    job_run_id = glue.start_job_run('transform-job')

    # 4. Monitor job execution
    # (Poll job run status)

    # 5. Data is ready for Athena queries
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
