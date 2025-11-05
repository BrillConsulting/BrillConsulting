"""
Apache Airflow Workflow Orchestration
Author: BrillConsulting
Description: DAG creation and workflow management for data pipelines
"""

import json
from typing import Dict, List, Any
from datetime import datetime, timedelta


class AirflowDAGManager:
    """Airflow DAG and workflow management"""

    def __init__(self, airflow_url: str = 'http://localhost:8080'):
        self.airflow_url = airflow_url
        self.dags = []

    def create_dag(self, dag_config: Dict[str, Any]) -> str:
        """Create Airflow DAG"""
        dag_id = dag_config.get('dag_id', 'data_pipeline')
        schedule = dag_config.get('schedule', '@daily')
        tasks = dag_config.get('tasks', [])

        dag_code = f'''"""
{dag_id} DAG
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {{
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}}

with DAG(
    dag_id='{dag_id}',
    default_args=default_args,
    schedule_interval='{schedule}',
    catchup=False,
    tags=['production', 'data-pipeline'],
) as dag:

    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data_func,
    )

    transform = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data_func,
    )

    load = PythonOperator(
        task_id='load_data',
        python_callable=load_data_func,
    )

    extract >> transform >> load
'''

        self.dags.append({'dag_id': dag_id, 'schedule': schedule, 'tasks': len(tasks)})
        print(f"✓ DAG created: {dag_id}")
        print(f"  Schedule: {schedule}, Tasks: {len(tasks)}")
        return dag_code

    def get_dag_runs(self, dag_id: str) -> List[Dict[str, Any]]:
        """Get DAG run history"""
        runs = [
            {'run_id': f'{dag_id}_run_1', 'state': 'success', 'execution_date': datetime.now().isoformat()},
            {'run_id': f'{dag_id}_run_2', 'state': 'running', 'execution_date': datetime.now().isoformat()}
        ]
        print(f"✓ Retrieved {len(runs)} DAG runs for {dag_id}")
        return runs


def demo():
    """Demonstrate Airflow DAG management"""
    print("=" * 60)
    print("Apache Airflow Workflow Orchestration Demo")
    print("=" * 60)

    mgr = AirflowDAGManager()

    print("\n1. Creating DAG...")
    dag_code = mgr.create_dag({
        'dag_id': 'etl_pipeline',
        'schedule': '@daily',
        'tasks': ['extract', 'transform', 'load']
    })
    print(dag_code[:200] + "...")

    print("\n2. Getting DAG runs...")
    mgr.get_dag_runs('etl_pipeline')

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
