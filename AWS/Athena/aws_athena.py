"""
AWS Athena
==========

Serverless SQL queries on data in Amazon S3 using standard SQL.

Author: Brill Consulting
"""

import boto3
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AthenaManager:
    """
    Advanced AWS Athena Management System

    Provides comprehensive Athena operations including:
    - Query execution and management
    - Database and table operations
    - Workgroup management
    - Named queries
    - Result retrieval and processing
    - Query cost tracking
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """Initialize Athena Manager."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.athena_client = session.client('athena', region_name=region)
            self.region = region
            logger.info(f"Athena Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing Athena Manager: {e}")
            raise

    # ==================== Query Execution ====================

    def start_query_execution(
        self,
        query_string: str,
        output_location: str,
        database: Optional[str] = None,
        workgroup: str = "primary",
        encryption_option: Optional[str] = None,
        kms_key: Optional[str] = None
    ) -> str:
        """
        Start query execution.

        Args:
            query_string: SQL query
            output_location: S3 path for results (s3://bucket/path/)
            database: Database name
            workgroup: Workgroup name
            encryption_option: 'SSE_S3', 'SSE_KMS', 'CSE_KMS'
            kms_key: KMS key ARN for encryption

        Returns:
            Query execution ID
        """
        try:
            logger.info(f"Starting query execution")

            params = {
                'QueryString': query_string,
                'ResultConfiguration': {
                    'OutputLocation': output_location
                },
                'WorkGroup': workgroup
            }

            if database:
                params['QueryExecutionContext'] = {'Database': database}

            if encryption_option:
                params['ResultConfiguration']['EncryptionConfiguration'] = {
                    'EncryptionOption': encryption_option
                }
                if kms_key:
                    params['ResultConfiguration']['EncryptionConfiguration']['KmsKey'] = kms_key

            response = self.athena_client.start_query_execution(**params)

            execution_id = response['QueryExecutionId']
            logger.info(f"✓ Query started: {execution_id}")

            return execution_id

        except ClientError as e:
            logger.error(f"Error starting query: {e}")
            raise

    def get_query_execution(self, execution_id: str) -> Dict[str, Any]:
        """Get query execution status and details."""
        try:
            response = self.athena_client.get_query_execution(
                QueryExecutionId=execution_id
            )

            execution = response['QueryExecution']
            status = execution['Status']

            result = {
                'execution_id': execution_id,
                'query': execution['Query'],
                'state': status['State'],
                'database': execution.get('QueryExecutionContext', {}).get('Database'),
                'workgroup': execution.get('WorkGroup'),
                'submission_time': execution['Status']['SubmissionDateTime'].isoformat()
            }

            if 'CompletionDateTime' in status:
                result['completion_time'] = status['CompletionDateTime'].isoformat()

            if status['State'] == 'SUCCEEDED':
                stats = execution.get('Statistics', {})
                result['data_scanned_bytes'] = stats.get('DataScannedInBytes', 0)
                result['execution_time_ms'] = stats.get('EngineExecutionTimeInMillis', 0)
                result['output_location'] = execution['ResultConfiguration']['OutputLocation']

            if status['State'] == 'FAILED':
                result['error_message'] = status.get('StateChangeReason', '')

            return result

        except ClientError as e:
            logger.error(f"Error getting query execution: {e}")
            raise

    def wait_for_query_completion(
        self,
        execution_id: str,
        max_wait_seconds: int = 300,
        poll_interval: int = 2
    ) -> Dict[str, Any]:
        """Wait for query to complete."""
        try:
            logger.info(f"Waiting for query completion: {execution_id}")

            start_time = time.time()
            while time.time() - start_time < max_wait_seconds:
                execution = self.get_query_execution(execution_id)

                if execution['state'] in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    logger.info(f"✓ Query {execution['state'].lower()}: {execution_id}")
                    return execution

                time.sleep(poll_interval)

            raise TimeoutError(f"Query did not complete within {max_wait_seconds} seconds")

        except ClientError as e:
            logger.error(f"Error waiting for query: {e}")
            raise

    def stop_query_execution(self, execution_id: str) -> None:
        """Stop query execution."""
        try:
            self.athena_client.stop_query_execution(QueryExecutionId=execution_id)
            logger.info(f"✓ Query stopped: {execution_id}")

        except ClientError as e:
            logger.error(f"Error stopping query: {e}")
            raise

    def get_query_results(
        self,
        execution_id: str,
        max_results: int = 1000
    ) -> Dict[str, Any]:
        """Get query results."""
        try:
            response = self.athena_client.get_query_results(
                QueryExecutionId=execution_id,
                MaxResults=max_results
            )

            result_set = response['ResultSet']

            # Extract column names
            columns = [
                col['Name']
                for col in result_set['ResultSetMetadata']['ColumnInfo']
            ]

            # Extract rows (skip header row)
            rows = []
            for row in result_set['Rows'][1:]:  # Skip header
                row_data = [
                    col.get('VarCharValue', '')
                    for col in row['Data']
                ]
                rows.append(dict(zip(columns, row_data)))

            logger.info(f"Retrieved {len(rows)} row(s)")
            return {
                'columns': columns,
                'rows': rows,
                'row_count': len(rows)
            }

        except ClientError as e:
            logger.error(f"Error getting query results: {e}")
            raise

    def execute_query_and_wait(
        self,
        query_string: str,
        output_location: str,
        database: Optional[str] = None,
        workgroup: str = "primary"
    ) -> Dict[str, Any]:
        """Execute query and wait for results (convenience method)."""
        execution_id = self.start_query_execution(
            query_string=query_string,
            output_location=output_location,
            database=database,
            workgroup=workgroup
        )

        execution = self.wait_for_query_completion(execution_id)

        if execution['state'] == 'SUCCEEDED':
            results = self.get_query_results(execution_id)
            return {
                'execution': execution,
                'results': results
            }
        else:
            return {
                'execution': execution,
                'error': execution.get('error_message', 'Query failed')
            }

    # ==================== Databases and Tables ====================

    def create_database(
        self,
        database_name: str,
        output_location: str,
        comment: Optional[str] = None
    ) -> str:
        """Create database."""
        try:
            query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
            if comment:
                query += f" COMMENT '{comment}'"

            execution_id = self.start_query_execution(
                query_string=query,
                output_location=output_location
            )

            self.wait_for_query_completion(execution_id)
            logger.info(f"✓ Database created: {database_name}")

            return execution_id

        except ClientError as e:
            logger.error(f"Error creating database: {e}")
            raise

    def drop_database(
        self,
        database_name: str,
        output_location: str,
        cascade: bool = False
    ) -> str:
        """Drop database."""
        try:
            query = f"DROP DATABASE IF EXISTS {database_name}"
            if cascade:
                query += " CASCADE"

            execution_id = self.start_query_execution(
                query_string=query,
                output_location=output_location
            )

            self.wait_for_query_completion(execution_id)
            logger.info(f"✓ Database dropped: {database_name}")

            return execution_id

        except ClientError as e:
            logger.error(f"Error dropping database: {e}")
            raise

    def list_databases(self, output_location: str) -> List[str]:
        """List databases."""
        try:
            result = self.execute_query_and_wait(
                query_string="SHOW DATABASES",
                output_location=output_location
            )

            if 'results' in result:
                return [row['database_name'] for row in result['results']['rows']]
            return []

        except ClientError as e:
            logger.error(f"Error listing databases: {e}")
            raise

    # ==================== Named Queries ====================

    def create_named_query(
        self,
        name: str,
        query_string: str,
        database: str,
        description: Optional[str] = None,
        workgroup: str = "primary"
    ) -> str:
        """Create named query."""
        try:
            logger.info(f"Creating named query: {name}")

            params = {
                'Name': name,
                'QueryString': query_string,
                'Database': database,
                'WorkGroup': workgroup
            }

            if description:
                params['Description'] = description

            response = self.athena_client.create_named_query(**params)

            logger.info(f"✓ Named query created: {name}")
            return response['NamedQueryId']

        except ClientError as e:
            logger.error(f"Error creating named query: {e}")
            raise

    def get_named_query(self, named_query_id: str) -> Dict[str, Any]:
        """Get named query."""
        try:
            response = self.athena_client.get_named_query(NamedQueryId=named_query_id)

            query = response['NamedQuery']
            return {
                'named_query_id': query['NamedQueryId'],
                'name': query['Name'],
                'query_string': query['QueryString'],
                'database': query['Database'],
                'description': query.get('Description', ''),
                'workgroup': query.get('WorkGroup')
            }

        except ClientError as e:
            logger.error(f"Error getting named query: {e}")
            raise

    def list_named_queries(self, workgroup: Optional[str] = None) -> List[str]:
        """List named queries."""
        try:
            params = {}
            if workgroup:
                params['WorkGroup'] = workgroup

            response = self.athena_client.list_named_queries(**params)

            query_ids = response.get('NamedQueryIds', [])
            logger.info(f"Found {len(query_ids)} named quer(ies)")

            return query_ids

        except ClientError as e:
            logger.error(f"Error listing named queries: {e}")
            raise

    def delete_named_query(self, named_query_id: str) -> None:
        """Delete named query."""
        try:
            self.athena_client.delete_named_query(NamedQueryId=named_query_id)
            logger.info(f"✓ Named query deleted: {named_query_id}")

        except ClientError as e:
            logger.error(f"Error deleting named query: {e}")
            raise

    # ==================== Workgroups ====================

    def create_workgroup(
        self,
        workgroup_name: str,
        output_location: str,
        enforce_workgroup_configuration: bool = True,
        bytes_scanned_cutoff_per_query: Optional[int] = None,
        description: Optional[str] = None
    ) -> None:
        """Create workgroup."""
        try:
            logger.info(f"Creating workgroup: {workgroup_name}")

            config = {
                'ResultConfigurationUpdates': {
                    'OutputLocation': output_location
                },
                'EnforceWorkGroupConfiguration': enforce_workgroup_configuration
            }

            if bytes_scanned_cutoff_per_query:
                config['BytesScannedCutoffPerQuery'] = bytes_scanned_cutoff_per_query

            params = {
                'Name': workgroup_name,
                'Configuration': config
            }

            if description:
                params['Description'] = description

            self.athena_client.create_work_group(**params)

            logger.info(f"✓ Workgroup created: {workgroup_name}")

        except ClientError as e:
            logger.error(f"Error creating workgroup: {e}")
            raise

    def delete_workgroup(self, workgroup_name: str, recursive: bool = False) -> None:
        """Delete workgroup."""
        try:
            self.athena_client.delete_work_group(
                WorkGroup=workgroup_name,
                RecursiveDeleteOption=recursive
            )
            logger.info(f"✓ Workgroup deleted: {workgroup_name}")

        except ClientError as e:
            logger.error(f"Error deleting workgroup: {e}")
            raise

    # ==================== Monitoring ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get Athena summary."""
        try:
            return {
                'region': self.region,
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """Demonstration of Athena Manager capabilities."""
    print("AWS Athena Manager - Advanced Demo")
    print("=" * 70)

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    print("\n1️⃣  Database Operations:")
    print("""
    athena = AthenaManager(region='us-east-1')
    output_location = 's3://my-athena-results/'

    # Create database
    athena.create_database(
        database_name='sales_db',
        output_location=output_location,
        comment='Sales data warehouse'
    )

    # List databases
    databases = athena.list_databases(output_location)
    """)

    print("\n2️⃣  Query Execution:")
    print("""
    # Execute query
    execution_id = athena.start_query_execution(
        query_string='SELECT * FROM sales_db.orders WHERE amount > 100',
        output_location=output_location,
        database='sales_db'
    )

    # Wait for completion
    execution = athena.wait_for_query_completion(execution_id)

    # Get results
    results = athena.get_query_results(execution_id)
    for row in results['rows']:
        print(row)
    """)

    print("\n3️⃣  Convenience Method:")
    print("""
    # Execute and wait in one call
    result = athena.execute_query_and_wait(
        query_string='SELECT COUNT(*) as total FROM orders',
        output_location=output_location,
        database='sales_db'
    )

    total = result['results']['rows'][0]['total']
    print(f"Total orders: {total}")
    """)

    print("\n4️⃣  Named Queries:")
    print("""
    # Save frequently used query
    query_id = athena.create_named_query(
        name='high-value-orders',
        query_string='SELECT * FROM orders WHERE amount > 1000',
        database='sales_db',
        description='Orders over $1000'
    )

    # Retrieve and execute later
    query = athena.get_named_query(query_id)
    athena.execute_query_and_wait(
        query_string=query['query_string'],
        output_location=output_location,
        database=query['database']
    )
    """)

    print("\n5️⃣  Workgroups (Cost Control):")
    print("""
    # Create workgroup with cost limits
    athena.create_workgroup(
        workgroup_name='analytics-team',
        output_location='s3://analytics-results/',
        bytes_scanned_cutoff_per_query=10 * 1024**3,  # 10 GB limit
        description='Analytics team workgroup'
    )
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
