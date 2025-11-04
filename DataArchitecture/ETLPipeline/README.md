# ETL Pipeline Framework

Extract, Transform, Load pipeline for data integration and processing.

## Features

- Multi-source data extraction
- Data transformation and cleaning
- Target system loading
- Data validation
- Error handling and logging
- Incremental loads support

## Usage

```python
from etl_pipeline import ETLPipeline

config = {"sources": ["database", "api"], "target": "warehouse"}
pipeline = ETLPipeline(config)
result = pipeline.run(["database", "api"], "warehouse")
```

## Demo

```bash
python etl_pipeline.py
```
