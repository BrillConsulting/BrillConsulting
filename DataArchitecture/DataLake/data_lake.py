"""
Data Lake Architecture
=======================

Scalable data lake for raw and processed data:
- Object storage management
- Data partitioning
- Schema-on-read
- Data cataloging
- Zone management (raw/curated/refined)

Author: Brill Consulting
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json


class DataLake:
    """Data lake management system."""

    def __init__(self, base_path: str = "./datalake"):
        """Initialize data lake."""
        self.base_path = Path(base_path)
        self.zones = {
            "raw": self.base_path / "raw",
            "curated": self.base_path / "curated",
            "refined": self.base_path / "refined"
        }

        # Create zone directories
        for zone_path in self.zones.values():
            zone_path.mkdir(parents=True, exist_ok=True)

        self.catalog = {}

    def ingest_raw(self, data: pd.DataFrame, source: str, dataset: str) -> Dict:
        """Ingest data into raw zone."""
        print(f"Ingesting raw data: {source}/{dataset}")

        # Create partitions by date
        ingestion_date = datetime.now().strftime("%Y-%m-%d")
        partition_path = self.zones["raw"] / source / dataset / f"date={ingestion_date}"
        partition_path.mkdir(parents=True, exist_ok=True)

        # Save as parquet
        filename = f"data_{datetime.now().strftime('%H%M%S')}.parquet"
        filepath = partition_path / filename

        data.to_parquet(filepath, index=False)

        # Update catalog
        catalog_entry = {
            "zone": "raw",
            "source": source,
            "dataset": dataset,
            "partition": ingestion_date,
            "filepath": str(filepath),
            "row_count": len(data),
            "ingestion_timestamp": datetime.now().isoformat()
        }

        catalog_key = f"{source}.{dataset}.{ingestion_date}"
        self.catalog[catalog_key] = catalog_entry

        print(f"✓ Ingested {len(data)} rows to {filepath}")
        return catalog_entry

    def curate_data(self, source: str, dataset: str, transformations: Dict = None) -> pd.DataFrame:
        """Move data from raw to curated zone with transformations."""
        print(f"Curating data: {source}/{dataset}")

        # Find raw data
        raw_pattern = f"{source}.{dataset}.*"
        raw_entries = [k for k in self.catalog.keys() if k.startswith(f"{source}.{dataset}")]

        if not raw_entries:
            print(f"No raw data found for {source}/{dataset}")
            return pd.DataFrame()

        # Load raw data
        all_data = []
        for entry_key in raw_entries:
            entry = self.catalog[entry_key]
            data = pd.read_parquet(entry["filepath"])
            all_data.append(data)

        combined_data = pd.concat(all_data, ignore_index=True)

        # Apply transformations
        if transformations:
            if "deduplicate" in transformations:
                combined_data = combined_data.drop_duplicates()

            if "filter" in transformations:
                # Apply filter logic
                pass

        # Save to curated zone
        curated_path = self.zones["curated"] / source / dataset
        curated_path.mkdir(parents=True, exist_ok=True)

        output_file = curated_path / f"{dataset}_curated.parquet"
        combined_data.to_parquet(output_file, index=False)

        print(f"✓ Curated {len(combined_data)} rows to {output_file}")
        return combined_data

    def create_refined_view(self, name: str, query_logic: Dict) -> pd.DataFrame:
        """Create refined data view."""
        print(f"Creating refined view: {name}")

        # Simulate joining curated data
        # In production, would use SQL/Spark queries

        refined_data = pd.DataFrame({
            "id": range(1, 11),
            "metric": [100 + i * 10 for i in range(10)],
            "category": [f"Cat_{i%3}" for i in range(10)]
        })

        # Save to refined zone
        refined_path = self.zones["refined"] / name
        refined_path.mkdir(parents=True, exist_ok=True)

        output_file = refined_path / f"{name}.parquet"
        refined_data.to_parquet(output_file, index=False)

        print(f"✓ Created refined view: {len(refined_data)} rows")
        return refined_data

    def get_catalog(self) -> Dict:
        """Get data catalog."""
        return self.catalog

    def query_zone(self, zone: str, source: str = None, dataset: str = None) -> List[Dict]:
        """Query data catalog by zone."""
        results = []

        for key, entry in self.catalog.items():
            if entry["zone"] != zone:
                continue

            if source and entry["source"] != source:
                continue

            if dataset and entry["dataset"] != dataset:
                continue

            results.append(entry)

        return results


def demo():
    """Demo data lake."""
    print("Data Lake Demo")
    print("="*50)

    lake = DataLake()

    # 1. Ingest raw data
    print("\n1. Ingesting Raw Data")
    print("-"*50)

    # Source 1: API data
    api_data = pd.DataFrame({
        "user_id": range(1, 101),
        "action": ["click"] * 50 + ["purchase"] * 50,
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="H")
    })

    lake.ingest_raw(api_data, source="api", dataset="user_events")

    # Source 2: Database data
    db_data = pd.DataFrame({
        "product_id": range(1, 51),
        "name": [f"Product_{i}" for i in range(1, 51)],
        "price": [10 + i * 5 for i in range(50)]
    })

    lake.ingest_raw(db_data, source="database", dataset="products")

    # 2. View catalog
    print("\n2. Data Catalog")
    print("-"*50)
    catalog = lake.get_catalog()
    print(f"Total datasets: {len(catalog)}")
    for key, entry in catalog.items():
        print(f"  {key}: {entry['row_count']} rows in {entry['zone']} zone")

    # 3. Curate data
    print("\n3. Curating Data")
    print("-"*50)

    curated = lake.curate_data("api", "user_events", {"deduplicate": True})

    # 4. Create refined view
    print("\n4. Creating Refined View")
    print("-"*50)

    refined = lake.create_refined_view("user_metrics", {})

    # 5. Query catalog
    print("\n5. Querying Catalog")
    print("-"*50)

    raw_datasets = lake.query_zone("raw")
    print(f"Raw zone datasets: {len(raw_datasets)}")

    print("\n✓ Data Lake Demo Complete!")


if __name__ == '__main__':
    demo()
