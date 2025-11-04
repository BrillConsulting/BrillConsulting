"""
Azure Data Services
===================

Azure data services integration:
- Azure SQL Database
- Cosmos DB (NoSQL)
- Azure Blob Storage
- Azure Data Lake
- Table Storage

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json


class AzureSQLDatabase:
    """Azure SQL Database client."""

    def __init__(self, server_name: str, database_name: str, admin_user: str):
        self.server_name = server_name
        self.database_name = database_name
        self.admin_user = admin_user
        self.connection_string = f"Server={server_name}.database.windows.net;Database={database_name}"
        self.tables = {}

    def create_table(self, table_name: str, columns: Dict[str, str]):
        """Create table."""
        print(f"\n📊 Creating table: {table_name}")

        self.tables[table_name] = {
            "columns": columns,
            "rows": [],
            "created_at": datetime.now().isoformat()
        }

        column_defs = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
        sql = f"CREATE TABLE {table_name} ({column_defs})"

        print(f"   SQL: {sql}")
        print(f"✓ Table created with {len(columns)} columns")

        return {"status": "success", "table": table_name}

    def insert_data(self, table_name: str, data: Dict) -> Dict:
        """Insert data into table."""
        if table_name not in self.tables:
            return {"error": f"Table {table_name} not found"}

        self.tables[table_name]["rows"].append(data)
        print(f"✓ Inserted row into {table_name}")

        return {"status": "success", "row_count": len(self.tables[table_name]["rows"])}

    def query(self, sql: str) -> List[Dict]:
        """Execute SQL query."""
        print(f"\n🔍 Executing query: {sql}")

        # Simplified query execution
        if "SELECT" in sql.upper():
            table_name = sql.split("FROM")[1].strip().split()[0] if "FROM" in sql else None
            if table_name and table_name in self.tables:
                results = self.tables[table_name]["rows"]
                print(f"✓ Query returned {len(results)} rows")
                return results

        return []

    def get_table_info(self, table_name: str) -> Optional[Dict]:
        """Get table information."""
        return self.tables.get(table_name)


class AzureCosmosDB:
    """Azure Cosmos DB client."""

    def __init__(self, account_name: str, database_name: str):
        self.account_name = account_name
        self.database_name = database_name
        self.endpoint = f"https://{account_name}.documents.azure.com:443/"
        self.containers = {}

    def create_container(self, container_name: str, partition_key: str):
        """Create Cosmos DB container."""
        print(f"\n🗄️  Creating container: {container_name}")

        self.containers[container_name] = {
            "partition_key": partition_key,
            "documents": [],
            "created_at": datetime.now().isoformat()
        }

        print(f"✓ Container created with partition key: {partition_key}")
        return {"status": "success", "container": container_name}

    def create_item(self, container_name: str, item: Dict) -> Dict:
        """Create document in container."""
        if container_name not in self.containers:
            return {"error": f"Container {container_name} not found"}

        item_with_id = {
            "id": item.get("id", f"doc_{datetime.now().timestamp()}"),
            **item,
            "_ts": datetime.now().timestamp()
        }

        self.containers[container_name]["documents"].append(item_with_id)
        print(f"✓ Document created in {container_name}")

        return {"status": "success", "id": item_with_id["id"]}

    def query_items(self, container_name: str, query: str) -> List[Dict]:
        """Query documents."""
        print(f"\n🔎 Querying container: {container_name}")
        print(f"   Query: {query}")

        if container_name not in self.containers:
            return []

        documents = self.containers[container_name]["documents"]
        print(f"✓ Found {len(documents)} documents")

        return documents

    def get_container_info(self, container_name: str) -> Optional[Dict]:
        """Get container information."""
        container = self.containers.get(container_name)
        if container:
            return {
                "name": container_name,
                "partition_key": container["partition_key"],
                "document_count": len(container["documents"]),
                "created_at": container["created_at"]
            }
        return None


class AzureBlobStorage:
    """Azure Blob Storage client."""

    def __init__(self, account_name: str):
        self.account_name = account_name
        self.endpoint = f"https://{account_name}.blob.core.windows.net/"
        self.containers = {}

    def create_container(self, container_name: str, access_level: str = "private") -> Dict:
        """Create blob container."""
        print(f"\n📦 Creating container: {container_name}")

        self.containers[container_name] = {
            "access_level": access_level,
            "blobs": [],
            "created_at": datetime.now().isoformat()
        }

        print(f"✓ Container created with access level: {access_level}")
        return {"status": "success", "container": container_name}

    def upload_blob(self, container_name: str, blob_name: str, data: Any, metadata: Optional[Dict] = None) -> Dict:
        """Upload blob."""
        if container_name not in self.containers:
            return {"error": f"Container {container_name} not found"}

        blob = {
            "name": blob_name,
            "size": len(str(data)),
            "content_type": "application/octet-stream",
            "metadata": metadata or {},
            "url": f"{self.endpoint}{container_name}/{blob_name}",
            "uploaded_at": datetime.now().isoformat()
        }

        self.containers[container_name]["blobs"].append(blob)
        print(f"✓ Uploaded blob: {blob_name} ({blob['size']} bytes)")

        return {"status": "success", "url": blob["url"]}

    def list_blobs(self, container_name: str) -> List[Dict]:
        """List blobs in container."""
        if container_name not in self.containers:
            return []

        blobs = self.containers[container_name]["blobs"]
        print(f"📋 Listed {len(blobs)} blobs in {container_name}")

        return blobs

    def download_blob(self, container_name: str, blob_name: str) -> Optional[Dict]:
        """Download blob."""
        if container_name not in self.containers:
            return None

        for blob in self.containers[container_name]["blobs"]:
            if blob["name"] == blob_name:
                print(f"⬇️  Downloaded blob: {blob_name}")
                return blob

        return None


class AzureDataLake:
    """Azure Data Lake Storage Gen2."""

    def __init__(self, account_name: str):
        self.account_name = account_name
        self.endpoint = f"https://{account_name}.dfs.core.windows.net/"
        self.filesystems = {}

    def create_filesystem(self, filesystem_name: str) -> Dict:
        """Create filesystem (container)."""
        print(f"\n🌊 Creating filesystem: {filesystem_name}")

        self.filesystems[filesystem_name] = {
            "directories": {},
            "files": [],
            "created_at": datetime.now().isoformat()
        }

        print(f"✓ Filesystem created")
        return {"status": "success", "filesystem": filesystem_name}

    def create_directory(self, filesystem_name: str, directory_path: str) -> Dict:
        """Create directory."""
        if filesystem_name not in self.filesystems:
            return {"error": f"Filesystem {filesystem_name} not found"}

        self.filesystems[filesystem_name]["directories"][directory_path] = {
            "files": [],
            "created_at": datetime.now().isoformat()
        }

        print(f"✓ Created directory: {directory_path}")
        return {"status": "success", "path": directory_path}

    def upload_file(self, filesystem_name: str, file_path: str, data: Any) -> Dict:
        """Upload file to data lake."""
        if filesystem_name not in self.filesystems:
            return {"error": f"Filesystem {filesystem_name} not found"}

        file_info = {
            "path": file_path,
            "size": len(str(data)),
            "url": f"{self.endpoint}{filesystem_name}/{file_path}",
            "uploaded_at": datetime.now().isoformat()
        }

        self.filesystems[filesystem_name]["files"].append(file_info)
        print(f"✓ Uploaded file: {file_path}")

        return {"status": "success", "url": file_info["url"]}

    def list_paths(self, filesystem_name: str, path: str = "/") -> List[str]:
        """List paths in filesystem."""
        if filesystem_name not in self.filesystems:
            return []

        filesystem = self.filesystems[filesystem_name]
        paths = list(filesystem["directories"].keys()) + [f["path"] for f in filesystem["files"]]

        print(f"📂 Listed {len(paths)} paths in {filesystem_name}")
        return paths


def demo():
    """Demo Azure Data Services."""
    print("Azure Data Services Demo")
    print("=" * 60)

    # 1. Azure SQL Database
    print("\n1. Azure SQL Database")
    print("-" * 60)

    sql_db = AzureSQLDatabase("srv-demo", "db-customers", "admin")

    sql_db.create_table("customers", {
        "id": "INT PRIMARY KEY",
        "name": "NVARCHAR(100)",
        "email": "NVARCHAR(100)",
        "created_at": "DATETIME"
    })

    sql_db.insert_data("customers", {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com",
        "created_at": datetime.now().isoformat()
    })

    results = sql_db.query("SELECT * FROM customers")
    print(f"Query results: {len(results)} rows")

    # 2. Cosmos DB
    print("\n2. Azure Cosmos DB")
    print("-" * 60)

    cosmos = AzureCosmosDB("cosmos-demo", "products-db")

    cosmos.create_container("products", "/category")

    cosmos.create_item("products", {
        "id": "prod_001",
        "name": "Laptop",
        "category": "Electronics",
        "price": 999.99
    })

    cosmos.create_item("products", {
        "id": "prod_002",
        "name": "Desk Chair",
        "category": "Furniture",
        "price": 299.99
    })

    docs = cosmos.query_items("products", "SELECT * FROM c WHERE c.category = 'Electronics'")

    # 3. Blob Storage
    print("\n3. Azure Blob Storage")
    print("-" * 60)

    blob_storage = AzureBlobStorage("stdemodemo123")

    blob_storage.create_container("images", "blob")
    blob_storage.create_container("documents", "private")

    blob_storage.upload_blob("images", "photo1.jpg", b"image_data_here", {
        "photographer": "Alice",
        "location": "Seattle"
    })

    blobs = blob_storage.list_blobs("images")
    for blob in blobs:
        print(f"  • {blob['name']} - {blob['size']} bytes")

    # 4. Data Lake Storage
    print("\n4. Azure Data Lake Storage Gen2")
    print("-" * 60)

    data_lake = AzureDataLake("datalake-demo")

    data_lake.create_filesystem("raw-data")
    data_lake.create_directory("raw-data", "/logs/2024")
    data_lake.create_directory("raw-data", "/events/2024")

    data_lake.upload_file("raw-data", "/logs/2024/app.log", "log data content")
    data_lake.upload_file("raw-data", "/events/2024/events.json", '{"event": "click"}')

    paths = data_lake.list_paths("raw-data")
    print(f"Data Lake paths:")
    for path in paths:
        print(f"  • {path}")

    # Summary
    print("\n5. Data Services Summary")
    print("-" * 60)
    print(f"  SQL Database: {sql_db.database_name}")
    print(f"    Tables: {len(sql_db.tables)}")
    print(f"\n  Cosmos DB: {cosmos.database_name}")
    print(f"    Containers: {len(cosmos.containers)}")
    print(f"\n  Blob Storage: {blob_storage.account_name}")
    print(f"    Containers: {len(blob_storage.containers)}")
    print(f"\n  Data Lake: {data_lake.account_name}")
    print(f"    Filesystems: {len(data_lake.filesystems)}")

    print("\n✓ Azure Data Services Demo Complete!")


if __name__ == '__main__':
    demo()
