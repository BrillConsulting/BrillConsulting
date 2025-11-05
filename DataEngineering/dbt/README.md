# ⚙️ dbt Data Transformation

**SQL-based data transformation and modeling**

## Overview
dbt implementation for analytics engineering with SQL-based transformations, testing, and documentation.

## Key Features
- Model creation (table, view, incremental)
- DAG-based dependencies
- Built-in testing framework
- Documentation generation
- Macros and packages
- Source freshness checks

## Quick Start
```python
from dbt_manager import DbtManager

mgr = DbtManager('analytics')
model = mgr.create_model({'name': 'customers', 'materialization': 'table'})
results = mgr.run_tests()
```

## Technologies
- dbt Core
- SQL transformations
- Jinja2 templating

**Author:** Brill Consulting | clientbrill@gmail.com
