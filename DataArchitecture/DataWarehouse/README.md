# Data Warehouse Design

Star schema and dimensional modeling for analytics warehouses.

## Features

- Fact and dimension table creation
- Slowly Changing Dimensions (SCD Type 2)
- Star schema design patterns
- Aggregate table generation
- Query optimization
- Surrogate key management

## Usage

```python
from data_warehouse import DataWarehouse

dw = DataWarehouse()

# Create dimensions
dw.create_dimension_table("dim_customer", customer_data)

# Create fact table
dw.create_fact_table("fact_sales", sales_data,
                     measures=["amount"],
                     dimensions=["customer_id"])

# SCD Type 2
dw.implement_scd_type2("dim_customer", updates, "customer_id")
```

## Demo

```bash
python data_warehouse.py
```
