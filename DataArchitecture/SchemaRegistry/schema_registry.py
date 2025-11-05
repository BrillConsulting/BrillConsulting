"""
Schema Registry Framework
=========================

Centralized schema management with versioning and compatibility:
- Schema registration and versioning
- Compatibility checking (backward, forward, full)
- Schema evolution tracking
- Format support (Avro, JSON Schema, Protobuf)
- Schema validation
- Subject management

Author: Brill Consulting
"""

from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import json
import hashlib


class CompatibilityMode(Enum):
    """Schema compatibility modes."""
    BACKWARD = "backward"
    FORWARD = "forward"
    FULL = "full"
    NONE = "none"


class SchemaFormat(Enum):
    """Supported schema formats."""
    AVRO = "avro"
    JSON = "json"
    PROTOBUF = "protobuf"


class Schema:
    """Represents a schema."""

    def __init__(self, schema_id: str, subject: str, version: int,
                 schema_format: str, schema_definition: Dict):
        """Initialize schema."""
        self.schema_id = schema_id
        self.subject = subject
        self.version = version
        self.schema_format = schema_format
        self.schema_definition = schema_definition
        self.registered_at = datetime.now().isoformat()
        self.fingerprint = self._calculate_fingerprint()

    def _calculate_fingerprint(self) -> str:
        """Calculate schema fingerprint."""
        schema_str = json.dumps(self.schema_definition, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "schema_id": self.schema_id,
            "subject": self.subject,
            "version": self.version,
            "schema_format": self.schema_format,
            "schema_definition": self.schema_definition,
            "registered_at": self.registered_at,
            "fingerprint": self.fingerprint
        }


class SchemaRegistry:
    """Schema registry with versioning and compatibility checking."""

    def __init__(self):
        """Initialize schema registry."""
        self.schemas = {}
        self.subjects = {}
        self.compatibility_modes = {}
        self.default_compatibility = CompatibilityMode.BACKWARD
        self.schema_by_fingerprint = {}

    def register_schema(self, subject: str, schema_definition: Dict,
                       schema_format: str = "json") -> Schema:
        """Register a new schema version."""
        print(f"Registering schema for subject: {subject}")

        # Initialize subject if needed
        if subject not in self.subjects:
            self.subjects[subject] = {
                "subject": subject,
                "versions": [],
                "latest_version": 0,
                "compatibility": self.default_compatibility.value
            }

        # Get next version number
        next_version = self.subjects[subject]["latest_version"] + 1

        # Check compatibility
        if next_version > 1:
            compatibility_mode = self._get_compatibility_mode(subject)
            is_compatible, reason = self.check_compatibility(
                subject,
                schema_definition,
                compatibility_mode
            )

            if not is_compatible:
                raise ValueError(f"Schema not compatible: {reason}")

        # Generate schema ID
        schema_id = self._generate_schema_id(subject, next_version)

        # Create schema
        schema = Schema(
            schema_id=schema_id,
            subject=subject,
            version=next_version,
            schema_format=schema_format,
            schema_definition=schema_definition
        )

        # Store schema
        self.schemas[schema_id] = schema
        self.schema_by_fingerprint[schema.fingerprint] = schema
        self.subjects[subject]["versions"].append(schema_id)
        self.subjects[subject]["latest_version"] = next_version

        print(f"✓ Schema registered: {schema_id}")
        print(f"  Subject: {subject}")
        print(f"  Version: {next_version}")
        print(f"  Format: {schema_format}")

        return schema

    def _generate_schema_id(self, subject: str, version: int) -> str:
        """Generate unique schema ID."""
        content = f"{subject}:v{version}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def get_schema(self, schema_id: str) -> Optional[Schema]:
        """Get schema by ID."""
        return self.schemas.get(schema_id)

    def get_schema_by_subject(self, subject: str,
                             version: Optional[int] = None) -> Optional[Schema]:
        """Get schema for subject (latest or specific version)."""
        if subject not in self.subjects:
            return None

        subject_data = self.subjects[subject]

        if version is None:
            # Get latest version
            version = subject_data["latest_version"]

        if version < 1 or version > len(subject_data["versions"]):
            return None

        schema_id = subject_data["versions"][version - 1]
        return self.schemas.get(schema_id)

    def get_all_versions(self, subject: str) -> List[Schema]:
        """Get all schema versions for a subject."""
        if subject not in self.subjects:
            return []

        schemas = []
        for schema_id in self.subjects[subject]["versions"]:
            schema = self.schemas.get(schema_id)
            if schema:
                schemas.append(schema)

        return schemas

    def set_compatibility_mode(self, subject: str, mode: str) -> Dict:
        """Set compatibility mode for a subject."""
        print(f"Setting compatibility mode for {subject}: {mode}")

        if subject not in self.subjects:
            raise ValueError(f"Subject {subject} not found")

        # Validate mode
        try:
            CompatibilityMode(mode)
        except ValueError:
            raise ValueError(f"Invalid compatibility mode: {mode}")

        self.subjects[subject]["compatibility"] = mode

        print(f"✓ Compatibility mode set: {mode}")

        return {
            "subject": subject,
            "compatibility": mode
        }

    def _get_compatibility_mode(self, subject: str) -> CompatibilityMode:
        """Get compatibility mode for subject."""
        if subject in self.subjects:
            mode_str = self.subjects[subject].get("compatibility",
                                                  self.default_compatibility.value)
            return CompatibilityMode(mode_str)
        return self.default_compatibility

    def check_compatibility(self, subject: str, new_schema: Dict,
                          compatibility_mode: CompatibilityMode) -> tuple:
        """Check if new schema is compatible with existing schemas."""
        print(f"Checking compatibility: {subject} ({compatibility_mode.value})")

        if subject not in self.subjects:
            return True, "New subject, no compatibility check needed"

        latest_schema = self.get_schema_by_subject(subject)
        if not latest_schema:
            return True, "No previous schema to compare"

        # Perform compatibility check based on mode
        if compatibility_mode == CompatibilityMode.NONE:
            return True, "Compatibility checking disabled"

        elif compatibility_mode == CompatibilityMode.BACKWARD:
            return self._check_backward_compatibility(
                latest_schema.schema_definition,
                new_schema
            )

        elif compatibility_mode == CompatibilityMode.FORWARD:
            return self._check_forward_compatibility(
                latest_schema.schema_definition,
                new_schema
            )

        elif compatibility_mode == CompatibilityMode.FULL:
            backward, backward_reason = self._check_backward_compatibility(
                latest_schema.schema_definition,
                new_schema
            )
            forward, forward_reason = self._check_forward_compatibility(
                latest_schema.schema_definition,
                new_schema
            )

            if backward and forward:
                return True, "Schema is fully compatible"
            else:
                reasons = []
                if not backward:
                    reasons.append(f"Backward: {backward_reason}")
                if not forward:
                    reasons.append(f"Forward: {forward_reason}")
                return False, "; ".join(reasons)

        return True, "Unknown compatibility mode"

    def _check_backward_compatibility(self, old_schema: Dict,
                                     new_schema: Dict) -> tuple:
        """Check backward compatibility (new readers can read old data)."""
        # Simplified backward compatibility check
        old_fields = {f["name"]: f for f in old_schema.get("fields", [])}
        new_fields = {f["name"]: f for f in new_schema.get("fields", [])}

        # Check for removed fields without defaults
        for field_name, field in old_fields.items():
            if field_name not in new_fields:
                # Field was removed - backward compatible if old field had default
                if "default" not in field:
                    return False, f"Field '{field_name}' removed without default"

        # Check for type changes
        for field_name in old_fields:
            if field_name in new_fields:
                old_type = old_fields[field_name].get("type")
                new_type = new_fields[field_name].get("type")
                if old_type != new_type:
                    return False, f"Field '{field_name}' type changed from {old_type} to {new_type}"

        return True, "Schema is backward compatible"

    def _check_forward_compatibility(self, old_schema: Dict,
                                    new_schema: Dict) -> tuple:
        """Check forward compatibility (old readers can read new data)."""
        # Simplified forward compatibility check
        old_fields = {f["name"]: f for f in old_schema.get("fields", [])}
        new_fields = {f["name"]: f for f in new_schema.get("fields", [])}

        # Check for added fields without defaults
        for field_name, field in new_fields.items():
            if field_name not in old_fields:
                # New field added - forward compatible if has default
                if "default" not in field:
                    return False, f"New field '{field_name}' added without default"

        # Check for type changes
        for field_name in old_fields:
            if field_name in new_fields:
                old_type = old_fields[field_name].get("type")
                new_type = new_fields[field_name].get("type")
                if old_type != new_type:
                    return False, f"Field '{field_name}' type changed"

        return True, "Schema is forward compatible"

    def validate_data(self, subject: str, data: Dict,
                     version: Optional[int] = None) -> Dict:
        """Validate data against schema."""
        print(f"Validating data against schema: {subject}")

        schema = self.get_schema_by_subject(subject, version)
        if not schema:
            return {
                "valid": False,
                "errors": ["Schema not found"]
            }

        errors = []
        schema_fields = {f["name"]: f for f in
                        schema.schema_definition.get("fields", [])}

        # Check required fields
        for field_name, field in schema_fields.items():
            if field_name not in data:
                if "default" not in field:
                    errors.append(f"Required field '{field_name}' missing")

        # Check field types
        for field_name, value in data.items():
            if field_name in schema_fields:
                expected_type = schema_fields[field_name].get("type")
                actual_type = type(value).__name__

                # Simplified type checking
                type_map = {
                    "string": "str",
                    "int": "int",
                    "float": "float",
                    "boolean": "bool"
                }

                if type_map.get(expected_type) != actual_type:
                    errors.append(
                        f"Field '{field_name}' type mismatch: "
                        f"expected {expected_type}, got {actual_type}"
                    )

        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "schema_version": schema.version
        }

        if result["valid"]:
            print("✓ Data is valid")
        else:
            print(f"✗ Validation failed: {len(errors)} error(s)")

        return result

    def delete_subject(self, subject: str) -> Dict:
        """Delete a subject and all its schemas."""
        print(f"Deleting subject: {subject}")

        if subject not in self.subjects:
            raise ValueError(f"Subject {subject} not found")

        # Remove all schemas for this subject
        schema_ids = self.subjects[subject]["versions"]
        for schema_id in schema_ids:
            if schema_id in self.schemas:
                schema = self.schemas[schema_id]
                del self.schema_by_fingerprint[schema.fingerprint]
                del self.schemas[schema_id]

        # Remove subject
        del self.subjects[subject]

        print(f"✓ Subject deleted: {subject}")
        print(f"  Removed {len(schema_ids)} schema version(s)")

        return {
            "subject": subject,
            "deleted_versions": len(schema_ids)
        }

    def get_subjects(self) -> List[str]:
        """Get all registered subjects."""
        return list(self.subjects.keys())

    def get_schema_evolution(self, subject: str) -> List[Dict]:
        """Get schema evolution history for a subject."""
        if subject not in self.subjects:
            return []

        evolution = []
        schemas = self.get_all_versions(subject)

        for i, schema in enumerate(schemas):
            step = {
                "version": schema.version,
                "registered_at": schema.registered_at,
                "fingerprint": schema.fingerprint,
                "field_count": len(schema.schema_definition.get("fields", []))
            }

            # Compare with previous version
            if i > 0:
                prev_schema = schemas[i - 1]
                changes = self._compare_schemas(
                    prev_schema.schema_definition,
                    schema.schema_definition
                )
                step["changes"] = changes

            evolution.append(step)

        return evolution

    def _compare_schemas(self, schema1: Dict, schema2: Dict) -> Dict:
        """Compare two schemas and return differences."""
        fields1 = {f["name"]: f for f in schema1.get("fields", [])}
        fields2 = {f["name"]: f for f in schema2.get("fields", [])}

        added = [name for name in fields2 if name not in fields1]
        removed = [name for name in fields1 if name not in fields2]
        modified = []

        for name in fields1:
            if name in fields2:
                if fields1[name].get("type") != fields2[name].get("type"):
                    modified.append(name)

        return {
            "added_fields": added,
            "removed_fields": removed,
            "modified_fields": modified
        }

    def generate_registry_report(self) -> Dict:
        """Generate comprehensive registry report."""
        print("\nGenerating Schema Registry Report...")
        print("="*50)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_subjects": len(self.subjects),
                "total_schemas": len(self.schemas),
                "total_formats": {}
            },
            "subjects": []
        }

        # Count schemas by format
        for schema in self.schemas.values():
            fmt = schema.schema_format
            if fmt not in report["summary"]["total_formats"]:
                report["summary"]["total_formats"][fmt] = 0
            report["summary"]["total_formats"][fmt] += 1

        # Subject details
        for subject_name, subject_data in self.subjects.items():
            report["subjects"].append({
                "name": subject_name,
                "versions": len(subject_data["versions"]),
                "latest_version": subject_data["latest_version"],
                "compatibility": subject_data["compatibility"]
            })

        print(f"Total Subjects: {report['summary']['total_subjects']}")
        print(f"Total Schemas: {report['summary']['total_schemas']}")

        return report


def demo():
    """Demo Schema Registry."""
    print("Schema Registry Demo")
    print("="*50)

    registry = SchemaRegistry()

    # 1. Register initial schema
    print("\n1. Registering Initial Schema")
    print("-"*50)

    user_schema_v1 = registry.register_schema(
        "user-events",
        {
            "type": "record",
            "name": "UserEvent",
            "fields": [
                {"name": "user_id", "type": "int"},
                {"name": "event_type", "type": "string"},
                {"name": "timestamp", "type": "string"}
            ]
        },
        schema_format="avro"
    )

    # 2. Set compatibility mode
    print("\n2. Setting Compatibility Mode")
    print("-"*50)

    registry.set_compatibility_mode("user-events", "backward")

    # 3. Register backward-compatible schema (add field with default)
    print("\n3. Registering Backward-Compatible Schema")
    print("-"*50)

    user_schema_v2 = registry.register_schema(
        "user-events",
        {
            "type": "record",
            "name": "UserEvent",
            "fields": [
                {"name": "user_id", "type": "int"},
                {"name": "event_type", "type": "string"},
                {"name": "timestamp", "type": "string"},
                {"name": "metadata", "type": "string", "default": ""}
            ]
        }
    )

    # 4. Try incompatible schema (should fail)
    print("\n4. Testing Incompatible Schema")
    print("-"*50)

    try:
        registry.register_schema(
            "user-events",
            {
                "type": "record",
                "name": "UserEvent",
                "fields": [
                    {"name": "user_id", "type": "int"},
                    # Removed event_type without default - incompatible!
                    {"name": "timestamp", "type": "string"}
                ]
            }
        )
        print("✗ Should have failed compatibility check!")
    except ValueError as e:
        print(f"✓ Correctly rejected: {str(e)[:60]}...")

    # 5. Get schema versions
    print("\n5. Getting Schema Versions")
    print("-"*50)

    versions = registry.get_all_versions("user-events")
    print(f"Total versions: {len(versions)}")
    for schema in versions:
        print(f"  v{schema.version}: {len(schema.schema_definition.get('fields', []))} fields")

    # 6. Validate data
    print("\n6. Validating Data Against Schema")
    print("-"*50)

    valid_data = {
        "user_id": 123,
        "event_type": "login",
        "timestamp": "2024-01-15T10:30:00Z"
    }

    validation = registry.validate_data("user-events", valid_data, version=1)
    print(f"Validation result: {'VALID' if validation['valid'] else 'INVALID'}")

    invalid_data = {
        "user_id": "not_an_int",  # Wrong type
        "event_type": "login"
        # Missing timestamp
    }

    validation = registry.validate_data("user-events", invalid_data)
    if not validation["valid"]:
        print(f"\nExpected validation errors:")
        for error in validation["errors"]:
            print(f"  - {error}")

    # 7. Register another subject
    print("\n7. Registering Another Subject")
    print("-"*50)

    registry.register_schema(
        "product-events",
        {
            "type": "record",
            "name": "ProductEvent",
            "fields": [
                {"name": "product_id", "type": "string"},
                {"name": "action", "type": "string"},
                {"name": "price", "type": "float", "default": 0.0}
            ]
        }
    )

    # 8. Get all subjects
    print("\n8. Listing All Subjects")
    print("-"*50)

    subjects = registry.get_subjects()
    print(f"Registered subjects: {len(subjects)}")
    for subject in subjects:
        print(f"  - {subject}")

    # 9. Get schema evolution
    print("\n9. Schema Evolution History")
    print("-"*50)

    evolution = registry.get_schema_evolution("user-events")
    print(f"Evolution history for user-events:")
    for step in evolution:
        print(f"  v{step['version']}: {step['field_count']} fields")
        if "changes" in step:
            changes = step["changes"]
            if changes["added_fields"]:
                print(f"    Added: {', '.join(changes['added_fields'])}")

    # 10. Test different compatibility modes
    print("\n10. Testing Full Compatibility Mode")
    print("-"*50)

    registry.set_compatibility_mode("product-events", "full")

    # This should work (has default)
    registry.register_schema(
        "product-events",
        {
            "type": "record",
            "name": "ProductEvent",
            "fields": [
                {"name": "product_id", "type": "string"},
                {"name": "action", "type": "string"},
                {"name": "price", "type": "float", "default": 0.0},
                {"name": "category", "type": "string", "default": "general"}
            ]
        }
    )

    # 11. Generate report
    print("\n11. Registry Report")
    print("-"*50)

    report = registry.generate_registry_report()

    print(f"\nSubject Details:")
    for subj in report["subjects"]:
        print(f"  {subj['name']}: v{subj['latest_version']} "
              f"({subj['compatibility']} compatibility)")

    print("\n✓ Schema Registry Demo Complete!")


if __name__ == '__main__':
    demo()
