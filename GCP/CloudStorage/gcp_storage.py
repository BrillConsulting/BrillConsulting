"""
Google Cloud Storage
====================

Cloud storage and data management:
- Bucket management
- Object upload/download
- Lifecycle policies
- Access control
- Signed URLs

Author: Brill Consulting
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json


class GCPCloudStorage:
    """GCP Cloud Storage management."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.buckets = {}

    def create_bucket(self, bucket_name: str, location: str = "US",
                     storage_class: str = "STANDARD") -> Dict:
        """Create storage bucket."""
        print(f"\n🪣 Creating bucket: {bucket_name}")
        print(f"   Location: {location}")
        print(f"   Storage class: {storage_class}")

        bucket = {
            "name": bucket_name,
            "location": location,
            "storageClass": storage_class,
            "objects": [],
            "versioning": {"enabled": False},
            "lifecycle": [],
            "created_at": datetime.now().isoformat()
        }

        self.buckets[bucket_name] = bucket
        print(f"✓ Bucket created: gs://{bucket_name}")

        return bucket

    def upload_object(self, bucket_name: str, object_name: str, data: bytes,
                     metadata: Optional[Dict] = None) -> Dict:
        """Upload object to bucket."""
        if bucket_name not in self.buckets:
            return {"error": f"Bucket {bucket_name} not found"}

        print(f"\n⬆️  Uploading: gs://{bucket_name}/{object_name}")

        obj = {
            "name": object_name,
            "bucket": bucket_name,
            "size": len(data),
            "contentType": "application/octet-stream",
            "metadata": metadata or {},
            "uploaded_at": datetime.now().isoformat(),
            "generation": "1",
            "etag": f"etag_{datetime.now().timestamp()}"
        }

        self.buckets[bucket_name]["objects"].append(obj)
        print(f"✓ Object uploaded ({obj['size']} bytes)")

        return obj

    def list_objects(self, bucket_name: str, prefix: str = "") -> List[Dict]:
        """List objects in bucket."""
        if bucket_name not in self.buckets:
            return []

        objects = self.buckets[bucket_name]["objects"]

        if prefix:
            objects = [obj for obj in objects if obj["name"].startswith(prefix)]

        print(f"\n📋 Listing objects in gs://{bucket_name}")
        print(f"   Prefix: {prefix if prefix else '(all)'}")
        print(f"   Found: {len(objects)} objects")

        return objects

    def delete_object(self, bucket_name: str, object_name: str) -> Dict:
        """Delete object from bucket."""
        if bucket_name not in self.buckets:
            return {"error": f"Bucket {bucket_name} not found"}

        bucket = self.buckets[bucket_name]
        bucket["objects"] = [obj for obj in bucket["objects"] if obj["name"] != object_name]

        print(f"🗑️  Deleted: gs://{bucket_name}/{object_name}")

        return {"status": "deleted"}

    def enable_versioning(self, bucket_name: str) -> Dict:
        """Enable versioning for bucket."""
        if bucket_name not in self.buckets:
            return {"error": f"Bucket {bucket_name} not found"}

        self.buckets[bucket_name]["versioning"]["enabled"] = True
        print(f"✓ Versioning enabled for: {bucket_name}")

        return {"versioning": "enabled"}

    def set_lifecycle_policy(self, bucket_name: str, rules: List[Dict]) -> Dict:
        """Set lifecycle policy."""
        if bucket_name not in self.buckets:
            return {"error": f"Bucket {bucket_name} not found"}

        print(f"\n🔄 Setting lifecycle policy for: {bucket_name}")

        self.buckets[bucket_name]["lifecycle"] = rules

        for i, rule in enumerate(rules, 1):
            action = rule.get("action", {}).get("type")
            condition = rule.get("condition", {})
            print(f"   Rule {i}: {action}")
            if "age" in condition:
                print(f"     Age: {condition['age']} days")

        print(f"✓ Lifecycle policy set ({len(rules)} rules)")

        return {"rules": len(rules)}

    def generate_signed_url(self, bucket_name: str, object_name: str,
                          expiration_minutes: int = 60) -> str:
        """Generate signed URL for object."""
        expiration = datetime.now() + timedelta(minutes=expiration_minutes)

        signed_url = f"https://storage.googleapis.com/{bucket_name}/{object_name}?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=...&X-Goog-Expires={expiration_minutes*60}"

        print(f"\n🔐 Generated signed URL:")
        print(f"   Object: gs://{bucket_name}/{object_name}")
        print(f"   Expires: {expiration.isoformat()}")

        return signed_url

    def copy_object(self, source_bucket: str, source_object: str,
                   dest_bucket: str, dest_object: str) -> Dict:
        """Copy object between buckets."""
        print(f"\n📋 Copying object:")
        print(f"   From: gs://{source_bucket}/{source_object}")
        print(f"   To: gs://{dest_bucket}/{dest_object}")

        # Find source object
        if source_bucket not in self.buckets:
            return {"error": f"Source bucket {source_bucket} not found"}

        source_obj = next(
            (obj for obj in self.buckets[source_bucket]["objects"] if obj["name"] == source_object),
            None
        )

        if not source_obj:
            return {"error": f"Source object {source_object} not found"}

        # Create copy in destination
        copied_obj = source_obj.copy()
        copied_obj["bucket"] = dest_bucket
        copied_obj["name"] = dest_object

        if dest_bucket in self.buckets:
            self.buckets[dest_bucket]["objects"].append(copied_obj)

        print(f"✓ Object copied successfully")

        return copied_obj

    def get_bucket_stats(self, bucket_name: str) -> Dict:
        """Get bucket statistics."""
        if bucket_name not in self.buckets:
            return {"error": f"Bucket {bucket_name} not found"}

        bucket = self.buckets[bucket_name]
        total_size = sum(obj["size"] for obj in bucket["objects"])

        return {
            "bucket": bucket_name,
            "object_count": len(bucket["objects"]),
            "total_size_bytes": total_size,
            "storage_class": bucket["storageClass"],
            "versioning_enabled": bucket["versioning"]["enabled"],
            "lifecycle_rules": len(bucket["lifecycle"])
        }


def demo():
    """Demo GCP Cloud Storage."""
    print("Google Cloud Storage Demo")
    print("=" * 60)

    storage = GCPCloudStorage("my-gcp-project")

    # 1. Create buckets
    print("\n1. Create Storage Buckets")
    print("-" * 60)

    storage.create_bucket("my-data-bucket", "US", "STANDARD")
    storage.create_bucket("my-archive-bucket", "US", "COLDLINE")

    # 2. Upload objects
    print("\n2. Upload Objects")
    print("-" * 60)

    storage.upload_object("my-data-bucket", "data/file1.txt", b"content here", {"author": "Alice"})
    storage.upload_object("my-data-bucket", "data/file2.txt", b"more content", {"author": "Bob"})
    storage.upload_object("my-data-bucket", "images/photo.jpg", b"image data" * 1000)

    # 3. List objects
    print("\n3. List Objects")
    print("-" * 60)

    all_objects = storage.list_objects("my-data-bucket")
    data_objects = storage.list_objects("my-data-bucket", prefix="data/")

    # 4. Versioning
    print("\n4. Enable Versioning")
    print("-" * 60)

    storage.enable_versioning("my-data-bucket")

    # 5. Lifecycle policy
    print("\n5. Lifecycle Policy")
    print("-" * 60)

    lifecycle_rules = [
        {
            "action": {"type": "Delete"},
            "condition": {"age": 30}
        },
        {
            "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
            "condition": {"age": 90}
        }
    ]

    storage.set_lifecycle_policy("my-data-bucket", lifecycle_rules)

    # 6. Signed URL
    print("\n6. Generate Signed URL")
    print("-" * 60)

    signed_url = storage.generate_signed_url("my-data-bucket", "data/file1.txt", 60)

    # 7. Copy object
    print("\n7. Copy Object")
    print("-" * 60)

    storage.copy_object("my-data-bucket", "data/file1.txt",
                       "my-archive-bucket", "archive/file1.txt")

    # 8. Bucket statistics
    print("\n8. Bucket Statistics")
    print("-" * 60)

    for bucket_name in ["my-data-bucket", "my-archive-bucket"]:
        stats = storage.get_bucket_stats(bucket_name)
        print(f"\n  {bucket_name}:")
        print(f"    Objects: {stats['object_count']}")
        print(f"    Total size: {stats['total_size_bytes']} bytes")
        print(f"    Storage class: {stats['storage_class']}")
        print(f"    Versioning: {'Enabled' if stats['versioning_enabled'] else 'Disabled'}")

    print("\n✓ GCP Cloud Storage Demo Complete!")


if __name__ == '__main__':
    demo()
