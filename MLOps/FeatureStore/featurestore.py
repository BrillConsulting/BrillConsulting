"""
Feature Store
=============

Centralized feature management and serving system:
- Feature registration and versioning
- Online feature serving (low-latency)
- Offline feature storage (batch)
- Point-in-time correctness
- Feature transformation pipeline
- Feature validation and monitoring
- Integration with training/inference

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle


@dataclass
class Feature:
    """Feature definition."""
    name: str
    dtype: str
    description: str = ""
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    transform: Optional[str] = None  # Transformation function name
    dependencies: List[str] = field(default_factory=list)


@dataclass
class FeatureGroup:
    """Group of related features."""
    name: str
    features: List[Feature]
    entity: str  # Entity key (e.g., "user_id", "product_id")
    description: str = ""
    version: str = "1.0"


class FeatureStore:
    """
    Centralized feature store for ML.

    Supports:
    - Feature registration and discovery
    - Online serving (Redis-like in-memory)
    - Offline storage (file-based)
    - Point-in-time correctness
    """

    def __init__(self, storage_path: str = "./feature_store"):
        """Initialize feature store."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        self.online_path = self.storage_path / "online"
        self.offline_path = self.storage_path / "offline"
        self.metadata_path = self.storage_path / "metadata"

        for path in [self.online_path, self.offline_path, self.metadata_path]:
            path.mkdir(exist_ok=True)

        # In-memory cache (simulates Redis)
        self.online_store: Dict[str, Dict] = {}

        # Feature registry
        self.feature_groups: Dict[str, FeatureGroup] = {}

        # Load existing metadata
        self._load_metadata()

    def register_feature_group(self, feature_group: FeatureGroup):
        """Register a feature group."""
        self.feature_groups[feature_group.name] = feature_group

        # Save metadata
        self._save_feature_group_metadata(feature_group)

        print(f"✓ Registered feature group: {feature_group.name}")
        print(f"  Entity: {feature_group.entity}")
        print(f"  Features: {len(feature_group.features)}")

    def materialize_features(
        self,
        feature_group_name: str,
        data: pd.DataFrame,
        to_online: bool = True,
        to_offline: bool = True
    ):
        """
        Materialize features from data.

        Args:
            feature_group_name: Name of feature group
            data: DataFrame with features
            to_online: Store in online store
            to_offline: Store in offline store
        """
        if feature_group_name not in self.feature_groups:
            raise ValueError(f"Feature group {feature_group_name} not registered")

        feature_group = self.feature_groups[feature_group_name]
        entity_key = feature_group.entity

        if entity_key not in data.columns:
            raise ValueError(f"Entity key {entity_key} not in data")

        # Store online (in-memory, keyed by entity)
        if to_online:
            for _, row in data.iterrows():
                entity_id = str(row[entity_key])

                if entity_id not in self.online_store:
                    self.online_store[entity_id] = {}

                # Store features
                for feature in feature_group.features:
                    if feature.name in data.columns:
                        self.online_store[entity_id][f"{feature_group_name}.{feature.name}"] = {
                            "value": row[feature.name],
                            "timestamp": datetime.now().isoformat()
                        }

        # Store offline (file-based, with timestamp)
        if to_offline:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            offline_file = self.offline_path / f"{feature_group_name}_{timestamp}.parquet"

            # Add ingestion timestamp
            data["_ingestion_time"] = datetime.now()

            # Save as parquet (in production, use actual parquet library)
            data.to_pickle(offline_file)

            print(f"✓ Materialized {len(data)} rows to offline store")

        if to_online:
            print(f"✓ Materialized features to online store")

    def get_online_features(
        self,
        entity_ids: List[str],
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Get features from online store (low-latency).

        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names (format: "group.feature")
        """
        records = []

        for entity_id in entity_ids:
            record = {"entity_id": entity_id}

            if entity_id in self.online_store:
                for feature_name in feature_names:
                    if feature_name in self.online_store[entity_id]:
                        record[feature_name] = self.online_store[entity_id][feature_name]["value"]
                    else:
                        record[feature_name] = None
            else:
                # Entity not found
                for feature_name in feature_names:
                    record[feature_name] = None

            records.append(record)

        return pd.DataFrame(records)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_names: List[str],
        point_in_time_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical features with point-in-time correctness.

        Args:
            entity_df: DataFrame with entity IDs and timestamps
            feature_names: List of feature names
            point_in_time_col: Column with timestamps for point-in-time join
        """
        # Extract feature groups needed
        feature_groups_needed = set()
        for fname in feature_names:
            if "." in fname:
                group_name = fname.split(".")[0]
                feature_groups_needed.add(group_name)

        # Load offline data for each feature group
        result_df = entity_df.copy()

        for group_name in feature_groups_needed:
            # Find latest offline file for this group
            offline_files = sorted(self.offline_path.glob(f"{group_name}_*.parquet"))

            if not offline_files:
                print(f"⚠ No offline data for {group_name}")
                continue

            # Load latest file
            latest_file = offline_files[-1]
            feature_data = pd.read_pickle(latest_file)

            # Get entity key
            if group_name not in self.feature_groups:
                continue

            entity_key = self.feature_groups[group_name].entity

            # Point-in-time join if timestamp provided
            if point_in_time_col and point_in_time_col in entity_df.columns:
                # Filter features to those before point-in-time
                # (Simplified - in production, use proper temporal join)
                feature_data = feature_data[
                    feature_data["_ingestion_time"] <= entity_df[point_in_time_col].max()
                ]

            # Join features
            feature_cols = [f.name for f in self.feature_groups[group_name].features
                          if f"{group_name}.{f.name}" in feature_names]

            if feature_cols and entity_key in feature_data.columns:
                # Rename columns to include group prefix
                rename_dict = {col: f"{group_name}.{col}" for col in feature_cols}
                feature_data = feature_data[[entity_key] + feature_cols].rename(columns=rename_dict)

                # Merge
                result_df = result_df.merge(feature_data, on=entity_key, how="left")

        return result_df

    def create_feature_view(
        self,
        name: str,
        feature_groups: List[str],
        features: Optional[List[str]] = None
    ) -> Dict:
        """
        Create a feature view (saved query).

        Args:
            name: View name
            feature_groups: List of feature group names
            features: Specific features to include (None = all)
        """
        view = {
            "name": name,
            "feature_groups": feature_groups,
            "features": features,
            "created_at": datetime.now().isoformat()
        }

        # Save view
        view_path = self.metadata_path / f"view_{name}.json"
        with open(view_path, 'w') as f:
            json.dump(view, f, indent=2)

        print(f"✓ Created feature view: {name}")
        return view

    def apply_transformation(
        self,
        feature_group_name: str,
        feature_name: str,
        transform_func: callable,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply transformation to features.

        Args:
            feature_group_name: Feature group name
            feature_name: Feature to transform
            transform_func: Transformation function
            data: Input data
        """
        if feature_name in data.columns:
            data[f"{feature_group_name}.{feature_name}_transformed"] = data[feature_name].apply(transform_func)

        return data

    def validate_features(
        self,
        feature_group_name: str,
        data: pd.DataFrame
    ) -> Dict:
        """
        Validate features against schema.

        Args:
            feature_group_name: Feature group name
            data: Data to validate
        """
        if feature_group_name not in self.feature_groups:
            return {"valid": False, "error": "Feature group not found"}

        feature_group = self.feature_groups[feature_group_name]
        errors = []

        # Check required features
        for feature in feature_group.features:
            if feature.name not in data.columns:
                errors.append(f"Missing feature: {feature.name}")
            else:
                # Check dtype
                expected_dtype = feature.dtype
                actual_dtype = str(data[feature.name].dtype)

                if expected_dtype != actual_dtype:
                    # Try basic compatibility check
                    if not (expected_dtype.startswith("float") and actual_dtype.startswith("float")):
                        if not (expected_dtype.startswith("int") and actual_dtype.startswith("int")):
                            errors.append(
                                f"Type mismatch for {feature.name}: "
                                f"expected {expected_dtype}, got {actual_dtype}"
                            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "checked_features": len(feature_group.features)
        }

    def get_feature_statistics(self, feature_group_name: str) -> Dict:
        """Get statistics for a feature group."""
        # Load offline data
        offline_files = sorted(self.offline_path.glob(f"{feature_group_name}_*.parquet"))

        if not offline_files:
            return {"error": "No data available"}

        latest_file = offline_files[-1]
        data = pd.read_pickle(latest_file)

        stats = {
            "feature_group": feature_group_name,
            "num_rows": len(data),
            "features": {}
        }

        if feature_group_name in self.feature_groups:
            for feature in self.feature_groups[feature_group_name].features:
                if feature.name in data.columns:
                    col_data = data[feature.name]

                    if pd.api.types.is_numeric_dtype(col_data):
                        stats["features"][feature.name] = {
                            "mean": float(col_data.mean()),
                            "std": float(col_data.std()),
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                            "null_count": int(col_data.isnull().sum())
                        }
                    else:
                        stats["features"][feature.name] = {
                            "unique_values": int(col_data.nunique()),
                            "null_count": int(col_data.isnull().sum()),
                            "most_common": str(col_data.mode()[0]) if len(col_data.mode()) > 0 else None
                        }

        return stats

    def _save_feature_group_metadata(self, feature_group: FeatureGroup):
        """Save feature group metadata."""
        metadata = {
            "name": feature_group.name,
            "entity": feature_group.entity,
            "description": feature_group.description,
            "version": feature_group.version,
            "features": [
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "description": f.description,
                    "version": f.version,
                    "created_at": f.created_at.isoformat()
                }
                for f in feature_group.features
            ]
        }

        metadata_file = self.metadata_path / f"{feature_group.name}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self):
        """Load existing metadata."""
        for metadata_file in self.metadata_path.glob("*.json"):
            if metadata_file.stem.startswith("view_"):
                continue  # Skip views

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Reconstruct feature group
            features = [
                Feature(
                    name=f["name"],
                    dtype=f["dtype"],
                    description=f.get("description", ""),
                    version=f.get("version", "1.0")
                )
                for f in metadata["features"]
            ]

            feature_group = FeatureGroup(
                name=metadata["name"],
                entity=metadata["entity"],
                features=features,
                description=metadata.get("description", ""),
                version=metadata.get("version", "1.0")
            )

            self.feature_groups[feature_group.name] = feature_group


def demo():
    """Demo feature store."""
    print("Feature Store Demo")
    print("="*70 + "\n")

    # 1. Initialize feature store
    print("1. Initializing Feature Store")
    print("-"*70)
    store = FeatureStore("./demo_feature_store")
    print("✓ Feature store initialized\n")

    # 2. Register feature groups
    print("2. Registering Feature Groups")
    print("-"*70)

    user_features = FeatureGroup(
        name="user_features",
        entity="user_id",
        description="User demographic and behavioral features",
        features=[
            Feature("age", "int64", "User age"),
            Feature("gender", "object", "User gender"),
            Feature("total_purchases", "int64", "Total purchase count"),
            Feature("avg_order_value", "float64", "Average order value"),
            Feature("days_since_signup", "int64", "Days since registration")
        ]
    )

    store.register_feature_group(user_features)
    print()

    # 3. Materialize features
    print("3. Materializing Features")
    print("-"*70)

    # Create sample data
    user_data = pd.DataFrame({
        "user_id": [f"user_{i}" for i in range(1000)],
        "age": np.random.randint(18, 70, 1000),
        "gender": np.random.choice(["M", "F", "Other"], 1000),
        "total_purchases": np.random.randint(0, 100, 1000),
        "avg_order_value": np.random.uniform(10, 500, 1000),
        "days_since_signup": np.random.randint(1, 365, 1000)
    })

    store.materialize_features(
        "user_features",
        user_data,
        to_online=True,
        to_offline=True
    )
    print()

    # 4. Online feature serving
    print("4. Online Feature Serving (Low-Latency)")
    print("-"*70)

    entity_ids = ["user_1", "user_2", "user_3"]
    feature_names = [
        "user_features.age",
        "user_features.total_purchases",
        "user_features.avg_order_value"
    ]

    online_features = store.get_online_features(entity_ids, feature_names)
    print(online_features)
    print()

    # 5. Historical features
    print("5. Historical Features (Training)")
    print("-"*70)

    entity_df = pd.DataFrame({
        "user_id": ["user_1", "user_5", "user_10"]
    })

    historical_features = store.get_historical_features(
        entity_df,
        feature_names
    )
    print(historical_features)
    print()

    # 6. Feature validation
    print("6. Feature Validation")
    print("-"*70)

    validation_result = store.validate_features("user_features", user_data)
    print(f"Valid: {validation_result['valid']}")
    print(f"Checked features: {validation_result['checked_features']}")
    if validation_result['errors']:
        print(f"Errors: {validation_result['errors']}")
    print()

    # 7. Feature statistics
    print("7. Feature Statistics")
    print("-"*70)

    stats = store.get_feature_statistics("user_features")
    print(f"Feature group: {stats['feature_group']}")
    print(f"Rows: {stats['num_rows']}")
    print("\nSample statistics:")
    for feat_name, feat_stats in list(stats['features'].items())[:2]:
        print(f"  {feat_name}: {feat_stats}")
    print()

    # 8. Feature view
    print("8. Creating Feature View")
    print("-"*70)

    view = store.create_feature_view(
        name="user_model_features",
        feature_groups=["user_features"],
        features=["age", "total_purchases", "avg_order_value"]
    )
    print()

    print("="*70)
    print("✓ Feature Store Demo Complete!")


if __name__ == '__main__':
    demo()
