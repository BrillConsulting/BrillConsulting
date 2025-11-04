# Data Lake Architecture

Scalable data lake with multi-zone architecture for raw and processed data.

## Features

- Multi-zone architecture (raw/curated/refined)
- Object storage with Parquet format
- Date-based partitioning
- Data cataloging
- Schema-on-read support
- Transformation pipelines

## Usage

```python
from data_lake import DataLake

lake = DataLake(base_path="./datalake")

# Ingest raw data
lake.ingest_raw(data, source="api", dataset="events")

# Curate data
curated = lake.curate_data("api", "events", {"deduplicate": True})

# Create refined view
refined = lake.create_refined_view("metrics", query_logic={})
```

## Demo

```bash
python data_lake.py
```
