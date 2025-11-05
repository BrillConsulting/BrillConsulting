"""
Data Catalog System
===================

Centralized metadata management and data discovery:
- Asset registration and management
- Business glossary
- Technical metadata
- Search and discovery
- Tagging and classification
- Impact analysis
- Data profiling integration

Author: Brill Consulting
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Set
import json


class DataCatalog:
    """Enterprise data catalog system."""

    def __init__(self):
        """Initialize data catalog."""
        self.assets = {}
        self.glossary = {}
        self.tags = {}
        self.relationships = {}
        self.search_index = {}

    def register_asset(self, asset_id: str, asset_info: Dict) -> Dict:
        """Register data asset in catalog."""
        print(f"Registering asset: {asset_id}")

        asset = {
            "asset_id": asset_id,
            "name": asset_info.get("name", asset_id),
            "type": asset_info.get("type", "table"),
            "description": asset_info.get("description", ""),
            "owner": asset_info.get("owner", "unknown"),
            "source_system": asset_info.get("source_system", ""),
            "schema": asset_info.get("schema", {}),
            "tags": asset_info.get("tags", []),
            "classification": asset_info.get("classification", "internal"),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": asset_info.get("metadata", {})
        }

        self.assets[asset_id] = asset

        # Update search index
        self._index_asset(asset)

        # Add tags
        for tag in asset["tags"]:
            if tag not in self.tags:
                self.tags[tag] = []
            self.tags[tag].append(asset_id)

        print(f"✓ Registered asset: {asset_id}")
        return asset

    def update_asset(self, asset_id: str, updates: Dict) -> Dict:
        """Update asset metadata."""
        if asset_id not in self.assets:
            print(f"Error: Asset {asset_id} not found")
            return {}

        print(f"Updating asset: {asset_id}")

        asset = self.assets[asset_id]

        # Update fields
        for key, value in updates.items():
            if key != "asset_id" and key != "created_at":
                asset[key] = value

        asset["updated_at"] = datetime.now().isoformat()

        # Re-index
        self._index_asset(asset)

        print(f"✓ Updated asset: {asset_id}")
        return asset

    def add_glossary_term(self, term_id: str, term_info: Dict) -> Dict:
        """Add business glossary term."""
        print(f"Adding glossary term: {term_id}")

        term = {
            "term_id": term_id,
            "name": term_info.get("name", term_id),
            "definition": term_info.get("definition", ""),
            "business_owner": term_info.get("business_owner", ""),
            "technical_owner": term_info.get("technical_owner", ""),
            "synonyms": term_info.get("synonyms", []),
            "related_terms": term_info.get("related_terms", []),
            "mapped_assets": term_info.get("mapped_assets", []),
            "created_at": datetime.now().isoformat()
        }

        self.glossary[term_id] = term
        print(f"✓ Added glossary term: {term_id}")
        return term

    def link_term_to_asset(self, term_id: str, asset_id: str, column: Optional[str] = None):
        """Link glossary term to data asset."""
        if term_id not in self.glossary:
            print(f"Error: Term {term_id} not found")
            return

        if asset_id not in self.assets:
            print(f"Error: Asset {asset_id} not found")
            return

        link = {
            "asset_id": asset_id,
            "column": column,
            "linked_at": datetime.now().isoformat()
        }

        if "mapped_assets" not in self.glossary[term_id]:
            self.glossary[term_id]["mapped_assets"] = []

        self.glossary[term_id]["mapped_assets"].append(link)
        print(f"✓ Linked {term_id} to {asset_id}" + (f".{column}" if column else ""))

    def add_relationship(self, source_asset: str, target_asset: str,
                        relationship_type: str, metadata: Dict = None):
        """Add relationship between assets."""
        print(f"Adding relationship: {source_asset} -> {target_asset}")

        rel_id = f"{source_asset}_{target_asset}_{relationship_type}"

        relationship = {
            "source": source_asset,
            "target": target_asset,
            "type": relationship_type,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }

        self.relationships[rel_id] = relationship
        print(f"✓ Added relationship: {relationship_type}")

    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search catalog."""
        print(f"Searching for: '{query}'")

        results = []
        query_lower = query.lower()

        for asset_id, asset in self.assets.items():
            # Search in name, description, tags
            matches = (
                query_lower in asset["name"].lower() or
                query_lower in asset["description"].lower() or
                any(query_lower in tag.lower() for tag in asset["tags"])
            )

            if matches:
                score = self._calculate_relevance(asset, query_lower)

                # Apply filters
                if filters:
                    if not self._apply_filters(asset, filters):
                        continue

                results.append({
                    "asset": asset,
                    "relevance_score": score
                })

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        print(f"✓ Found {len(results)} results")
        return results

    def get_asset_lineage(self, asset_id: str, direction: str = "both") -> Dict:
        """Get asset lineage (upstream/downstream)."""
        if asset_id not in self.assets:
            return {}

        lineage = {
            "asset_id": asset_id,
            "upstream": [],
            "downstream": []
        }

        for rel_id, rel in self.relationships.items():
            if direction in ["upstream", "both"] and rel["target"] == asset_id:
                lineage["upstream"].append({
                    "asset_id": rel["source"],
                    "relationship": rel["type"],
                    "metadata": rel["metadata"]
                })

            if direction in ["downstream", "both"] and rel["source"] == asset_id:
                lineage["downstream"].append({
                    "asset_id": rel["target"],
                    "relationship": rel["type"],
                    "metadata": rel["metadata"]
                })

        return lineage

    def impact_analysis(self, asset_id: str, max_depth: int = 3) -> Dict:
        """Analyze downstream impact of changes to asset."""
        print(f"Analyzing impact for: {asset_id}")

        impacted = set()
        to_process = [(asset_id, 0)]
        processed = set()

        while to_process:
            current_id, depth = to_process.pop(0)

            if current_id in processed or depth >= max_depth:
                continue

            processed.add(current_id)

            # Find downstream dependencies
            for rel_id, rel in self.relationships.items():
                if rel["source"] == current_id:
                    target = rel["target"]
                    impacted.add(target)
                    to_process.append((target, depth + 1))

        result = {
            "source_asset": asset_id,
            "impacted_assets": list(impacted),
            "impact_count": len(impacted),
            "max_depth_analyzed": max_depth
        }

        print(f"✓ Impact analysis: {len(impacted)} assets impacted")
        return result

    def get_assets_by_tag(self, tag: str) -> List[Dict]:
        """Get all assets with specific tag."""
        if tag not in self.tags:
            return []

        return [self.assets[asset_id] for asset_id in self.tags[tag]
                if asset_id in self.assets]

    def get_data_quality_score(self, asset_id: str) -> Dict:
        """Calculate data quality score for asset."""
        if asset_id not in self.assets:
            return {}

        asset = self.assets[asset_id]

        # Calculate completeness
        required_fields = ["name", "description", "owner", "schema"]
        completed_fields = sum(1 for field in required_fields if asset.get(field))
        completeness = (completed_fields / len(required_fields)) * 100

        # Calculate documentation score
        doc_score = min(len(asset.get("description", "")) / 100, 1.0) * 100

        # Has business glossary links
        has_glossary = any(
            asset_id in term.get("mapped_assets", [])
            for term in self.glossary.values()
        )

        overall_score = (completeness * 0.5 + doc_score * 0.3 + (100 if has_glossary else 0) * 0.2)

        return {
            "asset_id": asset_id,
            "overall_score": round(overall_score, 2),
            "completeness": round(completeness, 2),
            "documentation": round(doc_score, 2),
            "has_glossary_links": has_glossary
        }

    def generate_catalog_report(self) -> Dict:
        """Generate catalog statistics report."""
        print("\nGenerating catalog report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_assets": len(self.assets),
            "total_glossary_terms": len(self.glossary),
            "total_relationships": len(self.relationships),
            "total_tags": len(self.tags),
            "assets_by_type": {},
            "assets_by_classification": {},
            "top_tags": []
        }

        # Count by type
        for asset in self.assets.values():
            asset_type = asset.get("type", "unknown")
            report["assets_by_type"][asset_type] = \
                report["assets_by_type"].get(asset_type, 0) + 1

            # Count by classification
            classification = asset.get("classification", "unknown")
            report["assets_by_classification"][classification] = \
                report["assets_by_classification"].get(classification, 0) + 1

        # Top tags
        tag_counts = [(tag, len(assets)) for tag, assets in self.tags.items()]
        tag_counts.sort(key=lambda x: x[1], reverse=True)
        report["top_tags"] = tag_counts[:10]

        print("✓ Report generated")
        return report

    def _index_asset(self, asset: Dict):
        """Index asset for search."""
        asset_id = asset["asset_id"]

        # Create search keywords
        keywords = set()
        keywords.add(asset["name"].lower())
        keywords.update(asset.get("tags", []))

        # Add description words
        desc_words = asset.get("description", "").lower().split()
        keywords.update(desc_words)

        self.search_index[asset_id] = list(keywords)

    def _calculate_relevance(self, asset: Dict, query: str) -> float:
        """Calculate search relevance score."""
        score = 0.0

        # Exact name match
        if query == asset["name"].lower():
            score += 10.0

        # Name contains query
        if query in asset["name"].lower():
            score += 5.0

        # Description contains query
        if query in asset.get("description", "").lower():
            score += 2.0

        # Tag match
        if any(query in tag.lower() for tag in asset.get("tags", [])):
            score += 3.0

        return score

    def _apply_filters(self, asset: Dict, filters: Dict) -> bool:
        """Apply search filters."""
        for key, value in filters.items():
            if key == "type" and asset.get("type") != value:
                return False
            if key == "owner" and asset.get("owner") != value:
                return False
            if key == "classification" and asset.get("classification") != value:
                return False

        return True


def demo():
    """Demo data catalog."""
    print("Data Catalog Demo")
    print("="*60)

    catalog = DataCatalog()

    # 1. Register assets
    print("\n1. Registering Data Assets")
    print("-"*60)

    catalog.register_asset("customers_table", {
        "name": "Customers",
        "type": "table",
        "description": "Customer master data table",
        "owner": "data_team",
        "source_system": "CRM",
        "schema": {
            "customer_id": "integer",
            "name": "string",
            "email": "string",
            "created_at": "timestamp"
        },
        "tags": ["customer", "master-data", "pii"],
        "classification": "confidential"
    })

    catalog.register_asset("orders_table", {
        "name": "Orders",
        "type": "table",
        "description": "Customer orders and transactions",
        "owner": "data_team",
        "source_system": "ERP",
        "schema": {
            "order_id": "integer",
            "customer_id": "integer",
            "amount": "decimal",
            "order_date": "timestamp"
        },
        "tags": ["orders", "transactions"],
        "classification": "internal"
    })

    catalog.register_asset("customer_360_view", {
        "name": "Customer 360 View",
        "type": "view",
        "description": "Unified customer view with orders",
        "owner": "analytics_team",
        "source_system": "Data Warehouse",
        "tags": ["customer", "analytics", "360"],
        "classification": "internal"
    })

    # 2. Add business glossary
    print("\n2. Building Business Glossary")
    print("-"*60)

    catalog.add_glossary_term("customer", {
        "name": "Customer",
        "definition": "Individual or organization purchasing products/services",
        "business_owner": "sales_dept",
        "technical_owner": "data_team",
        "synonyms": ["client", "account"],
        "related_terms": ["prospect", "lead"]
    })

    catalog.link_term_to_asset("customer", "customers_table")
    catalog.link_term_to_asset("customer", "orders_table", "customer_id")

    # 3. Add relationships
    print("\n3. Adding Asset Relationships")
    print("-"*60)

    catalog.add_relationship("customers_table", "customer_360_view",
                           "feeds", {"join_key": "customer_id"})
    catalog.add_relationship("orders_table", "customer_360_view",
                           "feeds", {"join_key": "customer_id"})

    # 4. Search
    print("\n4. Searching Catalog")
    print("-"*60)

    results = catalog.search("customer", filters={"type": "table"})
    for result in results:
        asset = result["asset"]
        score = result["relevance_score"]
        print(f"  - {asset['name']} (score: {score})")

    # 5. Lineage
    print("\n5. Asset Lineage")
    print("-"*60)

    lineage = catalog.get_asset_lineage("customer_360_view")
    print(f"Upstream dependencies: {len(lineage['upstream'])}")
    for dep in lineage['upstream']:
        print(f"  - {dep['asset_id']} ({dep['relationship']})")

    # 6. Impact analysis
    print("\n6. Impact Analysis")
    print("-"*60)

    impact = catalog.impact_analysis("customers_table")
    print(f"Impacted assets if customers_table changes: {impact['impact_count']}")
    for asset_id in impact['impacted_assets']:
        print(f"  - {asset_id}")

    # 7. Data quality scores
    print("\n7. Data Quality Scores")
    print("-"*60)

    for asset_id in ["customers_table", "orders_table", "customer_360_view"]:
        score = catalog.get_data_quality_score(asset_id)
        print(f"{asset_id}: {score['overall_score']}%")

    # 8. Catalog report
    print("\n8. Catalog Statistics")
    print("-"*60)

    report = catalog.generate_catalog_report()
    print(f"Total assets: {report['total_assets']}")
    print(f"Total glossary terms: {report['total_glossary_terms']}")
    print(f"Total relationships: {report['total_relationships']}")
    print(f"Assets by type: {report['assets_by_type']}")
    print(f"Top tags: {dict(report['top_tags'][:5])}")

    print("\n✓ Data Catalog Demo Complete!")


if __name__ == '__main__':
    demo()
