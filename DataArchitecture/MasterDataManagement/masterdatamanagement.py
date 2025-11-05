"""
Master Data Management Framework
================================

Comprehensive MDM solution with entity resolution and golden records:
- Entity resolution and matching
- Golden record creation and maintenance
- Survivorship rules
- Data merging and deduplication
- Hierarchy management
- Data quality scoring

Author: Brill Consulting
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib


class Entity:
    """Represents a master data entity."""

    def __init__(self, entity_id: str, entity_type: str, attributes: Dict,
                 source: str):
        """Initialize entity."""
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.attributes = attributes
        self.source = source
        self.created_at = datetime.now().isoformat()
        self.quality_score = 0.0
        self.match_candidates = []

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "attributes": self.attributes,
            "source": self.source,
            "created_at": self.created_at,
            "quality_score": self.quality_score
        }


class GoldenRecord:
    """Represents a golden record (master record)."""

    def __init__(self, record_id: str, entity_type: str, attributes: Dict,
                 source_entities: List[str]):
        """Initialize golden record."""
        self.record_id = record_id
        self.entity_type = entity_type
        self.attributes = attributes
        self.source_entities = source_entities
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.confidence_score = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "entity_type": entity_type,
            "attributes": self.attributes,
            "source_entities": self.source_entities,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confidence_score": self.confidence_score
        }


class MasterDataManagement:
    """Master Data Management system."""

    def __init__(self):
        """Initialize MDM system."""
        self.entities = {}
        self.golden_records = {}
        self.match_rules = {}
        self.survivorship_rules = {}
        self.entity_relationships = {}
        self.merge_history = []

    def register_entity(self, entity_id: str, entity_type: str,
                       attributes: Dict, source: str) -> Entity:
        """Register an entity in the MDM system."""
        print(f"Registering entity: {entity_id}")

        entity = Entity(entity_id, entity_type, attributes, source)

        # Calculate quality score
        entity.quality_score = self._calculate_quality_score(attributes)

        self.entities[entity_id] = entity

        print(f"✓ Registered entity: {entity_id}")
        print(f"  Type: {entity_type}")
        print(f"  Source: {source}")
        print(f"  Quality: {entity.quality_score:.2f}")

        return entity

    def _calculate_quality_score(self, attributes: Dict) -> float:
        """Calculate data quality score for attributes."""
        total_fields = len(attributes)
        if total_fields == 0:
            return 0.0

        filled_fields = sum(1 for v in attributes.values() if v is not None and v != "")
        completeness = filled_fields / total_fields

        # Simple quality score based on completeness
        return completeness * 100

    def define_match_rule(self, rule_id: str, entity_type: str,
                         rule: Dict) -> Dict:
        """Define matching rule for entity resolution."""
        print(f"Defining match rule: {rule_id}")

        match_rule = {
            "rule_id": rule_id,
            "entity_type": entity_type,
            "fields": rule.get("fields", []),
            "threshold": rule.get("threshold", 0.8),
            "method": rule.get("method", "fuzzy"),
            "created_at": datetime.now().isoformat()
        }

        self.match_rules[rule_id] = match_rule

        print(f"✓ Match rule defined: {rule_id}")
        print(f"  Entity type: {entity_type}")
        print(f"  Fields: {len(match_rule['fields'])}")
        print(f"  Threshold: {match_rule['threshold']}")

        return match_rule

    def find_matches(self, entity_id: str, rule_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """Find matching entities."""
        print(f"Finding matches for entity: {entity_id}")

        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not found")

        entity = self.entities[entity_id]
        matches = []

        # Get matching rule
        if rule_id:
            if rule_id not in self.match_rules:
                raise ValueError(f"Match rule {rule_id} not found")
            rule = self.match_rules[rule_id]
        else:
            # Use default rule for entity type
            rule = self._get_default_match_rule(entity.entity_type)

        # Compare with other entities of same type
        for other_id, other_entity in self.entities.items():
            if other_id == entity_id:
                continue

            if other_entity.entity_type != entity.entity_type:
                continue

            # Calculate match score
            score = self._calculate_match_score(
                entity.attributes,
                other_entity.attributes,
                rule
            )

            if score >= rule["threshold"]:
                matches.append((other_id, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)

        print(f"✓ Found {len(matches)} matches")
        if matches:
            print(f"  Top match: {matches[0][0]} (score: {matches[0][1]:.3f})")

        return matches

    def _get_default_match_rule(self, entity_type: str) -> Dict:
        """Get default match rule for entity type."""
        return {
            "fields": ["name", "email"],
            "threshold": 0.8,
            "method": "fuzzy"
        }

    def _calculate_match_score(self, attrs1: Dict, attrs2: Dict,
                               rule: Dict) -> float:
        """Calculate match score between two entities."""
        fields = rule["fields"]
        method = rule["method"]

        if not fields:
            return 0.0

        scores = []
        for field in fields:
            val1 = attrs1.get(field, "")
            val2 = attrs2.get(field, "")

            if method == "exact":
                score = 1.0 if val1 == val2 else 0.0
            elif method == "fuzzy":
                score = self._fuzzy_match(str(val1), str(val2))
            else:
                score = 0.0

            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _fuzzy_match(self, str1: str, str2: str) -> float:
        """Calculate fuzzy match score (simplified Levenshtein)."""
        if str1 == str2:
            return 1.0

        str1, str2 = str1.lower(), str2.lower()

        if str1 == str2:
            return 0.95

        # Simple character overlap
        set1, set2 = set(str1), set(str2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def merge_entities(self, entity_ids: List[str],
                      survivorship_rule: Optional[str] = None) -> GoldenRecord:
        """Merge multiple entities into a golden record."""
        print(f"Merging {len(entity_ids)} entities...")

        # Validate entities
        entities = []
        for eid in entity_ids:
            if eid not in self.entities:
                raise ValueError(f"Entity {eid} not found")
            entities.append(self.entities[eid])

        # Check all same type
        entity_types = set(e.entity_type for e in entities)
        if len(entity_types) > 1:
            raise ValueError("Cannot merge entities of different types")

        entity_type = entities[0].entity_type

        # Apply survivorship rules to create golden record attributes
        if survivorship_rule and survivorship_rule in self.survivorship_rules:
            rule = self.survivorship_rules[survivorship_rule]
        else:
            rule = self._get_default_survivorship_rule()

        merged_attributes = self._apply_survivorship_rules(entities, rule)

        # Create golden record
        record_id = self._generate_golden_record_id(entity_ids)

        golden_record = GoldenRecord(
            record_id=record_id,
            entity_type=entity_type,
            attributes=merged_attributes,
            source_entities=entity_ids
        )

        # Calculate confidence score
        golden_record.confidence_score = self._calculate_merge_confidence(entities)

        self.golden_records[record_id] = golden_record

        # Record merge history
        self.merge_history.append({
            "record_id": record_id,
            "merged_entities": entity_ids,
            "timestamp": datetime.now().isoformat()
        })

        print(f"✓ Entities merged into golden record: {record_id}")
        print(f"  Confidence: {golden_record.confidence_score:.2f}")

        return golden_record

    def _generate_golden_record_id(self, entity_ids: List[str]) -> str:
        """Generate unique golden record ID."""
        content = ":".join(sorted(entity_ids))
        return f"GR_{hashlib.md5(content.encode()).hexdigest()[:12]}"

    def _get_default_survivorship_rule(self) -> Dict:
        """Get default survivorship rule."""
        return {
            "strategy": "most_complete",
            "source_priority": []
        }

    def _apply_survivorship_rules(self, entities: List[Entity],
                                  rule: Dict) -> Dict:
        """Apply survivorship rules to select best attributes."""
        strategy = rule["strategy"]
        merged_attrs = {}

        # Get all unique attribute keys
        all_keys = set()
        for entity in entities:
            all_keys.update(entity.attributes.keys())

        for key in all_keys:
            values = []
            for entity in entities:
                if key in entity.attributes and entity.attributes[key]:
                    values.append({
                        "value": entity.attributes[key],
                        "source": entity.source,
                        "quality": entity.quality_score
                    })

            if not values:
                merged_attrs[key] = None
                continue

            if strategy == "most_complete":
                # Select from entity with highest quality score
                best = max(values, key=lambda x: x["quality"])
                merged_attrs[key] = best["value"]

            elif strategy == "most_recent":
                # Would use timestamp if available
                merged_attrs[key] = values[-1]["value"]

            elif strategy == "source_priority":
                # Use source priority
                source_priority = rule.get("source_priority", [])
                for source in source_priority:
                    for v in values:
                        if v["source"] == source:
                            merged_attrs[key] = v["value"]
                            break
                    if key in merged_attrs:
                        break
                if key not in merged_attrs:
                    merged_attrs[key] = values[0]["value"]

            else:
                merged_attrs[key] = values[0]["value"]

        return merged_attrs

    def _calculate_merge_confidence(self, entities: List[Entity]) -> float:
        """Calculate confidence score for merged record."""
        if not entities:
            return 0.0

        # Average quality scores
        avg_quality = sum(e.quality_score for e in entities) / len(entities)

        # Bonus for multiple sources
        unique_sources = len(set(e.source for e in entities))
        source_bonus = min(unique_sources * 5, 20)

        return min(avg_quality + source_bonus, 100.0)

    def define_survivorship_rule(self, rule_id: str, rule: Dict) -> Dict:
        """Define survivorship rule for merging."""
        print(f"Defining survivorship rule: {rule_id}")

        survivorship_rule = {
            "rule_id": rule_id,
            "strategy": rule.get("strategy", "most_complete"),
            "source_priority": rule.get("source_priority", []),
            "field_rules": rule.get("field_rules", {}),
            "created_at": datetime.now().isoformat()
        }

        self.survivorship_rules[rule_id] = survivorship_rule

        print(f"✓ Survivorship rule defined: {rule_id}")
        print(f"  Strategy: {survivorship_rule['strategy']}")

        return survivorship_rule

    def update_golden_record(self, record_id: str, updates: Dict) -> GoldenRecord:
        """Update a golden record."""
        print(f"Updating golden record: {record_id}")

        if record_id not in self.golden_records:
            raise ValueError(f"Golden record {record_id} not found")

        record = self.golden_records[record_id]

        # Update attributes
        record.attributes.update(updates)
        record.updated_at = datetime.now().isoformat()

        print(f"✓ Golden record updated")
        print(f"  Updated fields: {len(updates)}")

        return record

    def create_relationship(self, entity1_id: str, entity2_id: str,
                          relationship_type: str, metadata: Optional[Dict] = None) -> Dict:
        """Create relationship between entities."""
        print(f"Creating relationship: {entity1_id} -> {entity2_id}")

        relationship = {
            "entity1": entity1_id,
            "entity2": entity2_id,
            "type": relationship_type,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }

        rel_id = f"{entity1_id}:{relationship_type}:{entity2_id}"
        self.entity_relationships[rel_id] = relationship

        print(f"✓ Relationship created: {relationship_type}")

        return relationship

    def get_entity_lineage(self, entity_id: str) -> Dict:
        """Get lineage information for an entity."""
        lineage = {
            "entity_id": entity_id,
            "golden_records": [],
            "relationships": []
        }

        # Find golden records containing this entity
        for record_id, record in self.golden_records.items():
            if entity_id in record.source_entities:
                lineage["golden_records"].append(record_id)

        # Find relationships
        for rel_id, rel in self.entity_relationships.items():
            if rel["entity1"] == entity_id or rel["entity2"] == entity_id:
                lineage["relationships"].append(rel)

        return lineage

    def deduplicate(self, entity_type: str, auto_merge: bool = False) -> Dict:
        """Find and optionally merge duplicate entities."""
        print(f"Deduplicating entities of type: {entity_type}")

        # Get all entities of this type
        type_entities = {
            eid: e for eid, e in self.entities.items()
            if e.entity_type == entity_type
        }

        duplicate_groups = []
        processed = set()

        for entity_id, entity in type_entities.items():
            if entity_id in processed:
                continue

            # Find matches
            matches = self.find_matches(entity_id)

            if matches:
                # Create duplicate group
                group = [entity_id] + [m[0] for m in matches]
                duplicate_groups.append({
                    "entities": group,
                    "match_scores": [m[1] for m in matches]
                })

                processed.update(group)

        result = {
            "entity_type": entity_type,
            "duplicate_groups": len(duplicate_groups),
            "total_duplicates": sum(len(g["entities"]) - 1 for g in duplicate_groups),
            "groups": duplicate_groups
        }

        # Auto-merge if requested
        if auto_merge:
            merged_records = []
            for group in duplicate_groups:
                record = self.merge_entities(group["entities"])
                merged_records.append(record.record_id)

            result["merged_records"] = merged_records

        print(f"✓ Deduplication complete")
        print(f"  Duplicate groups: {result['duplicate_groups']}")
        print(f"  Total duplicates: {result['total_duplicates']}")

        return result

    def generate_mdm_report(self) -> Dict:
        """Generate comprehensive MDM report."""
        print("\nGenerating MDM Report...")
        print("="*50)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_entities": len(self.entities),
                "total_golden_records": len(self.golden_records),
                "total_relationships": len(self.entity_relationships),
                "match_rules": len(self.match_rules),
                "survivorship_rules": len(self.survivorship_rules)
            },
            "entity_types": {},
            "quality_stats": {}
        }

        # Count by entity type
        for entity in self.entities.values():
            entity_type = entity.entity_type
            if entity_type not in report["entity_types"]:
                report["entity_types"][entity_type] = 0
            report["entity_types"][entity_type] += 1

        # Quality statistics
        if self.entities:
            quality_scores = [e.quality_score for e in self.entities.values()]
            report["quality_stats"] = {
                "avg_quality": sum(quality_scores) / len(quality_scores),
                "min_quality": min(quality_scores),
                "max_quality": max(quality_scores)
            }

        print(f"Total Entities: {report['summary']['total_entities']}")
        print(f"Total Golden Records: {report['summary']['total_golden_records']}")
        print(f"Total Relationships: {report['summary']['total_relationships']}")
        if quality_scores:
            print(f"Avg Quality Score: {report['quality_stats']['avg_quality']:.2f}")

        return report


def demo():
    """Demo Master Data Management."""
    print("Master Data Management Demo")
    print("="*50)

    mdm = MasterDataManagement()

    # 1. Register entities from different sources
    print("\n1. Registering Entities")
    print("-"*50)

    mdm.register_entity(
        "customer_001_crm",
        "customer",
        {
            "name": "John Smith",
            "email": "john.smith@example.com",
            "phone": "555-0100",
            "address": "123 Main St"
        },
        source="CRM"
    )

    mdm.register_entity(
        "customer_001_erp",
        "customer",
        {
            "name": "Jon Smith",  # Slight variation
            "email": "john.smith@example.com",
            "phone": "555-0100",
            "company": "Acme Corp"  # Additional field
        },
        source="ERP"
    )

    mdm.register_entity(
        "customer_002_web",
        "customer",
        {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "555-0200"
        },
        source="Website"
    )

    # 2. Define match rules
    print("\n2. Defining Match Rules")
    print("-"*50)

    mdm.define_match_rule(
        "customer_email_match",
        "customer",
        {
            "fields": ["email", "phone"],
            "threshold": 0.85,
            "method": "fuzzy"
        }
    )

    # 3. Find matches
    print("\n3. Finding Duplicate Entities")
    print("-"*50)

    matches = mdm.find_matches("customer_001_crm", "customer_email_match")

    # 4. Define survivorship rules
    print("\n4. Defining Survivorship Rules")
    print("-"*50)

    mdm.define_survivorship_rule(
        "customer_merge_rule",
        {
            "strategy": "most_complete",
            "source_priority": ["CRM", "ERP", "Website"]
        }
    )

    # 5. Merge duplicates
    print("\n5. Merging Duplicate Entities")
    print("-"*50)

    golden_record = mdm.merge_entities(
        ["customer_001_crm", "customer_001_erp"],
        survivorship_rule="customer_merge_rule"
    )

    print(f"\nGolden Record Attributes:")
    for key, value in golden_record.attributes.items():
        print(f"  {key}: {value}")

    # 6. Update golden record
    print("\n6. Updating Golden Record")
    print("-"*50)

    mdm.update_golden_record(
        golden_record.record_id,
        {"verified": True, "last_contact": "2024-01-15"}
    )

    # 7. Create relationships
    print("\n7. Creating Entity Relationships")
    print("-"*50)

    mdm.register_entity(
        "account_001",
        "account",
        {
            "account_number": "ACC-12345",
            "type": "premium",
            "balance": 50000
        },
        source="ERP"
    )

    mdm.create_relationship(
        golden_record.record_id,
        "account_001",
        "owns",
        metadata={"since": "2020-01-01"}
    )

    # 8. Get entity lineage
    print("\n8. Getting Entity Lineage")
    print("-"*50)

    lineage = mdm.get_entity_lineage("customer_001_crm")
    print(f"Entity is part of {len(lineage['golden_records'])} golden record(s)")
    print(f"Entity has {len(lineage['relationships'])} relationship(s)")

    # 9. Deduplicate
    print("\n9. Deduplicating All Customers")
    print("-"*50)

    # Add more test data
    mdm.register_entity(
        "customer_003_web",
        "customer",
        {"name": "Jane Doe", "email": "jane.doe@example.com"},
        source="Website"
    )

    dedup_result = mdm.deduplicate("customer", auto_merge=False)

    # 10. Generate report
    print("\n10. MDM Report")
    print("-"*50)

    report = mdm.generate_mdm_report()

    print(f"\nEntity Types:")
    for entity_type, count in report["entity_types"].items():
        print(f"  {entity_type}: {count}")

    print("\n✓ Master Data Management Demo Complete!")


if __name__ == '__main__':
    demo()
