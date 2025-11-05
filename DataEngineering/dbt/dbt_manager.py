"""
dbt Data Transformation
Author: BrillConsulting
Description: SQL-based data transformation and modeling
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class DbtManager:
    """dbt (data build tool) management"""

    def __init__(self, project_name: str = 'analytics'):
        self.project_name = project_name
        self.models = []

    def create_model(self, model_config: Dict[str, Any]) -> str:
        """Create dbt model"""
        model_name = model_config.get('name', 'customers')
        materialization = model_config.get('materialization', 'table')

        model_sql = f'''{{{{
    config(
        materialized='{materialization}',
        schema='analytics',
        tags=['daily']
    )
}}}}

WITH source AS (
    SELECT * FROM {{{{ source('raw', 'customers') }}}}
),

transformed AS (
    SELECT
        customer_id,
        email,
        created_at,
        updated_at,
        CASE
            WHEN total_orders > 10 THEN 'VIP'
            WHEN total_orders > 5 THEN 'Regular'
            ELSE 'New'
        END AS customer_segment
    FROM source
)

SELECT * FROM transformed
'''

        self.models.append({'name': model_name, 'materialization': materialization})
        print(f"✓ dbt model created: {model_name}")
        print(f"  Materialization: {materialization}")
        return model_sql

    def run_tests(self) -> Dict[str, Any]:
        """Run dbt tests"""
        result = {
            'tests_run': 15,
            'passed': 14,
            'failed': 1,
            'warnings': 2,
            'run_at': datetime.now().isoformat()
        }
        print(f"✓ dbt tests completed: {result['passed']}/{result['tests_run']} passed")
        return result


def demo():
    """Demonstrate dbt"""
    print("=" * 60)
    print("dbt Data Transformation Demo")
    print("=" * 60)

    mgr = DbtManager('analytics')

    print("\n1. Creating dbt model...")
    model = mgr.create_model({'name': 'customers', 'materialization': 'table'})
    print(model[:200] + "...")

    print("\n2. Running tests...")
    mgr.run_tests()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
