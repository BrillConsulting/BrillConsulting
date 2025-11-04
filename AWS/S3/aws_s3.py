"""
AWS S3
======

S3 bucket and object management.

Author: Brill Consulting
"""

from typing import List, Dict
from datetime import datetime


class AWSS3:
    """AWS S3 management."""

    def __init__(self):
        self.buckets = {}

    def create_bucket(self, bucket_name: str, region: str = "us-east-1") -> Dict:
        """Create S3 bucket."""
        print(f"\n🪣 Creating bucket: {bucket_name}")

        bucket = {
            "name": bucket_name,
            "region": region,
            "objects": [],
            "versioning": False,
            "created_at": datetime.now().isoformat()
        }

        self.buckets[bucket_name] = bucket
        print(f"✓ Bucket created: s3://{bucket_name}")

        return bucket

    def upload_object(self, bucket_name: str, key: str, data: bytes) -> Dict:
        """Upload object to S3."""
        if bucket_name not in self.buckets:
            return {"error": f"Bucket {bucket_name} not found"}

        print(f"\n⬆️  Uploading: s3://{bucket_name}/{key}")

        obj = {
            "key": key,
            "size": len(data),
            "etag": f"etag-{datetime.now().timestamp()}",
            "uploaded_at": datetime.now().isoformat()
        }

        self.buckets[bucket_name]["objects"].append(obj)
        print(f"✓ Object uploaded ({obj['size']} bytes)")

        return obj

    def list_objects(self, bucket_name: str) -> List[Dict]:
        """List objects in bucket."""
        if bucket_name not in self.buckets:
            return []

        objects = self.buckets[bucket_name]["objects"]
        print(f"\n📋 Listing objects in s3://{bucket_name}: {len(objects)} objects")

        return objects

    def enable_versioning(self, bucket_name: str) -> Dict:
        """Enable bucket versioning."""
        if bucket_name not in self.buckets:
            return {"error": f"Bucket {bucket_name} not found"}

        self.buckets[bucket_name]["versioning"] = True
        print(f"✓ Versioning enabled for {bucket_name}")

        return {"versioning": "enabled"}

    def get_summary(self) -> Dict:
        """Get S3 summary."""
        total_objects = sum(len(b["objects"]) for b in self.buckets.values())

        return {
            "buckets": len(self.buckets),
            "total_objects": total_objects
        }


def demo():
    """Demo AWS S3."""
    print("AWS S3 Demo")
    print("=" * 60)

    s3 = AWSS3()

    s3.create_bucket("my-data-bucket")
    s3.upload_object("my-data-bucket", "data/file1.txt", b"content here")
    s3.upload_object("my-data-bucket", "images/photo.jpg", b"image data" * 1000)

    s3.list_objects("my-data-bucket")
    s3.enable_versioning("my-data-bucket")

    print("\n📊 Summary:")
    summary = s3.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n✓ AWS S3 Demo Complete!")


if __name__ == '__main__':
    demo()
