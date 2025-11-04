"""
Data Warehouse Design
======================

Star schema and dimensional modeling for analytics:
- Fact and dimension tables
- Slowly Changing Dimensions (SCD)
- Star schema design
- Aggregate tables
- Query optimization

Author: Brill Consulting
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List


class DataWarehouse:
    """Data warehouse with star schema."""

    def __init__(self):
        """Initialize data warehouse."""
        self.fact_tables = {}
        self.dimension_tables = {}

    def create_dimension_table(self, name: str, data: pd.DataFrame,
                               surrogate_key: str = "id") -> pd.DataFrame:
        """Create dimension table with surrogate key."""
        print(f"Creating dimension table: {name}")

        # Add surrogate key if not exists
        if surrogate_key not in data.columns:
            data[surrogate_key] = range(1, len(data) + 1)

        # Add metadata
        data["valid_from"] = datetime.now()
        data["valid_to"] = pd.to_datetime("2099-12-31")
        data["is_current"] = True

        self.dimension_tables[name] = data
        print(f"✓ Created {name}: {len(data)} rows")
        return data

    def create_fact_table(self, name: str, data: pd.DataFrame,
                         measures: List[str], dimensions: List[str]) -> pd.DataFrame:
        """Create fact table."""
        print(f"Creating fact table: {name}")

        # Validate columns
        required_cols = measures + dimensions
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            print(f"Warning: Missing columns {missing}")

        # Add timestamp
        data["fact_timestamp"] = datetime.now()

        self.fact_tables[name] = {
            "data": data,
            "measures": measures,
            "dimensions": dimensions
        }

        print(f"✓ Created {name}: {len(data)} rows")
        return data

    def implement_scd_type2(self, dim_name: str, updates: pd.DataFrame,
                           natural_key: str) -> pd.DataFrame:
        """Implement Slowly Changing Dimension Type 2."""
        print(f"Implementing SCD Type 2 for {dim_name}")

        if dim_name not in self.dimension_tables:
            print(f"Error: Dimension {dim_name} not found")
            return pd.DataFrame()

        current_dim = self.dimension_tables[dim_name]

        # Find changed records
        changed_records = []
        new_records = []

        for idx, update_row in updates.iterrows():
            key_value = update_row[natural_key]

            # Find existing record
            existing = current_dim[
                (current_dim[natural_key] == key_value) &
                (current_dim["is_current"] == True)
            ]

            if len(existing) > 0:
                # Check if changed
                has_changed = False
                for col in updates.columns:
                    if col != natural_key and col in existing.columns:
                        if update_row[col] != existing.iloc[0][col]:
                            has_changed = True
                            break

                if has_changed:
                    changed_records.append(existing.index[0])
                    new_records.append(update_row)
            else:
                # New record
                new_records.append(update_row)

        # Expire old records
        if changed_records:
            current_dim.loc[changed_records, "valid_to"] = datetime.now()
            current_dim.loc[changed_records, "is_current"] = False

        # Add new records
        if new_records:
            new_df = pd.DataFrame(new_records)
            new_df["valid_from"] = datetime.now()
            new_df["valid_to"] = pd.to_datetime("2099-12-31")
            new_df["is_current"] = True
            new_df["id"] = range(len(current_dim) + 1, len(current_dim) + len(new_df) + 1)

            self.dimension_tables[dim_name] = pd.concat(
                [current_dim, new_df], ignore_index=True
            )

        print(f"✓ SCD Type 2: Expired {len(changed_records)}, Added {len(new_records)}")
        return self.dimension_tables[dim_name]

    def query_fact(self, fact_name: str, dimensions: Dict = None) -> pd.DataFrame:
        """Query fact table with dimension filters."""
        if fact_name not in self.fact_tables:
            print(f"Error: Fact table {fact_name} not found")
            return pd.DataFrame()

        fact_data = self.fact_tables[fact_name]["data"]

        # Apply dimension filters
        if dimensions:
            for dim_col, dim_value in dimensions.items():
                fact_data = fact_data[fact_data[dim_col] == dim_value]

        return fact_data

    def create_aggregate_table(self, fact_name: str, group_by: List[str],
                               aggregations: Dict[str, str]) -> pd.DataFrame:
        """Create aggregate table."""
        print(f"Creating aggregate table from {fact_name}")

        if fact_name not in self.fact_tables:
            return pd.DataFrame()

        fact_data = self.fact_tables[fact_name]["data"]

        # Group and aggregate
        agg_data = fact_data.groupby(group_by).agg(aggregations).reset_index()

        print(f"✓ Created aggregate: {len(agg_data)} rows")
        return agg_data


def demo():
    """Demo data warehouse."""
    print("Data Warehouse Demo")
    print("="*50)

    dw = DataWarehouse()

    # 1. Create dimensions
    print("\n1. Creating Dimension Tables")
    print("-"*50)

    # Customer dimension
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "city": ["NYC", "LA", "Chicago"],
        "segment": ["Premium", "Standard", "Premium"]
    })
    dw.create_dimension_table("dim_customer", customers, "customer_id")

    # Product dimension
    products = pd.DataFrame({
        "product_id": [101, 102, 103],
        "product_name": ["Widget A", "Widget B", "Widget C"],
        "category": ["Electronics", "Electronics", "Home"]
    })
    dw.create_dimension_table("dim_product", products, "product_id")

    # 2. Create fact table
    print("\n2. Creating Fact Table")
    print("-"*50)

    sales = pd.DataFrame({
        "customer_id": [1, 2, 1, 3, 2],
        "product_id": [101, 102, 103, 101, 102],
        "quantity": [2, 1, 3, 1, 2],
        "amount": [200, 150, 300, 100, 300]
    })

    dw.create_fact_table("fact_sales", sales,
                        measures=["quantity", "amount"],
                        dimensions=["customer_id", "product_id"])

    # 3. SCD Type 2
    print("\n3. Slowly Changing Dimension Type 2")
    print("-"*50)

    updates = pd.DataFrame({
        "customer_id": [1, 4],
        "name": ["Alice", "Dave"],
        "city": ["Boston", "Seattle"],  # Alice moved
        "segment": ["Premium", "Standard"]
    })

    dw.implement_scd_type2("dim_customer", updates, "customer_id")

    # 4. Query fact table
    print("\n4. Querying Fact Table")
    print("-"*50)

    result = dw.query_fact("fact_sales", {"customer_id": 1})
    print(f"Sales for customer 1: {len(result)} transactions")
    print(f"Total amount: ${result['amount'].sum()}")

    # 5. Create aggregate
    print("\n5. Creating Aggregate Table")
    print("-"*50)

    agg = dw.create_aggregate_table(
        "fact_sales",
        group_by=["product_id"],
        aggregations={"quantity": "sum", "amount": "sum"}
    )
    print(agg)

    print("\n✓ Data Warehouse Demo Complete!")


if __name__ == '__main__':
    demo()
