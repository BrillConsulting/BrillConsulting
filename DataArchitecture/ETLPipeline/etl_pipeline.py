"""
ETL Pipeline Framework
======================

Extract, Transform, Load pipeline for data integration:
- Data extraction from multiple sources
- Data transformation and cleaning
- Data loading to target systems
- Error handling and logging
- Incremental loads
- Schedule management

Author: Brill Consulting
"""

import json
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


class ETLPipeline:
    """ETL Pipeline framework."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize ETL pipeline."""
        self.config = config
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs = []

    def log(self, message: str, level: str = "INFO"):
        """Log pipeline message."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)
        print(f"[{level}] {message}")

    def extract(self, source: str) -> pd.DataFrame:
        """Extract data from source."""
        self.log(f"Extracting data from {source}")

        # Simulate extraction
        if source == "database":
            data = pd.DataFrame({
                "id": range(1, 101),
                "name": [f"User_{i}" for i in range(1, 101)],
                "amount": [100 + i * 10 for i in range(1, 101)],
                "date": pd.date_range("2024-01-01", periods=100, freq="D")
            })
        elif source == "api":
            data = pd.DataFrame({
                "id": range(101, 201),
                "category": [f"Cat_{i%5}" for i in range(100)],
                "value": [50 + i * 5 for i in range(100)]
            })
        else:
            data = pd.DataFrame()

        self.log(f"Extracted {len(data)} rows from {source}")
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        self.log("Transforming data")

        # Remove duplicates
        initial_rows = len(data)
        data = data.drop_duplicates()
        self.log(f"Removed {initial_rows - len(data)} duplicates")

        # Handle missing values
        data = data.fillna(0)

        # Add metadata
        data["etl_run_id"] = self.run_id
        data["etl_timestamp"] = datetime.now()

        self.log(f"Transformation complete: {len(data)} rows")
        return data

    def load(self, data: pd.DataFrame, target: str) -> bool:
        """Load data to target."""
        self.log(f"Loading {len(data)} rows to {target}")

        try:
            # Simulate loading
            if target == "warehouse":
                # Would use: data.to_sql(...)
                self.log(f"Loaded to warehouse: {len(data)} rows")
            elif target == "lake":
                # Would use: data.to_parquet(...)
                self.log(f"Loaded to data lake: {len(data)} rows")

            return True

        except Exception as e:
            self.log(f"Load failed: {str(e)}", level="ERROR")
            return False

    def validate(self, data: pd.DataFrame) -> Dict:
        """Validate data quality."""
        self.log("Validating data quality")

        validation = {
            "row_count": len(data),
            "null_count": data.isnull().sum().sum(),
            "duplicate_count": data.duplicated().sum(),
            "passed": True
        }

        # Check rules
        if validation["null_count"] > 0:
            self.log(f"Warning: {validation['null_count']} null values found", level="WARN")

        if validation["duplicate_count"] > 0:
            self.log(f"Warning: {validation['duplicate_count']} duplicates found", level="WARN")
            validation["passed"] = False

        return validation

    def run(self, sources: List[str], target: str) -> Dict:
        """Run complete ETL pipeline."""
        self.log("="*50)
        self.log(f"Starting ETL Pipeline - Run {self.run_id}")
        self.log("="*50)

        all_data = []

        # Extract from all sources
        for source in sources:
            data = self.extract(source)
            all_data.append(data)

        # Combine data
        combined_data = pd.concat(all_data, ignore_index=True)
        self.log(f"Combined data: {len(combined_data)} rows")

        # Validate
        validation = self.validate(combined_data)

        # Transform
        transformed_data = self.transform(combined_data)

        # Load
        success = self.load(transformed_data, target)

        self.log("="*50)
        self.log("ETL Pipeline Complete")
        self.log("="*50)

        return {
            "run_id": self.run_id,
            "success": success,
            "rows_processed": len(transformed_data),
            "validation": validation,
            "logs": self.logs
        }


def demo():
    """Demo ETL pipeline."""
    print("ETL Pipeline Demo")
    print("="*50)

    config = {
        "sources": ["database", "api"],
        "target": "warehouse"
    }

    pipeline = ETLPipeline(config)
    result = pipeline.run(["database", "api"], "warehouse")

    print("\nPipeline Result:")
    print(json.dumps({
        "run_id": result["run_id"],
        "success": result["success"],
        "rows_processed": result["rows_processed"],
        "validation": result["validation"]
    }, indent=2))


if __name__ == '__main__':
    demo()
