# Azure Data Services

Azure data services integration and management.

## Features

- Azure SQL Database operations
- Cosmos DB (NoSQL) document operations
- Azure Blob Storage management
- Azure Data Lake Storage Gen2
- Table Storage operations

## Usage

```python
from azure_data_services import AzureSQLDatabase, AzureCosmosDB, AzureBlobStorage

# SQL Database
sql_db = AzureSQLDatabase("srv-demo", "db-customers", "admin")
sql_db.create_table("customers", {"id": "INT PRIMARY KEY", "name": "NVARCHAR(100)"})
sql_db.insert_data("customers", {"id": 1, "name": "John Doe"})

# Cosmos DB
cosmos = AzureCosmosDB("cosmos-demo", "products-db")
cosmos.create_container("products", "/category")
cosmos.create_item("products", {"id": "prod_001", "name": "Laptop"})

# Blob Storage
blob = AzureBlobStorage("stdemodemo123")
blob.create_container("images")
blob.upload_blob("images", "photo.jpg", b"data")
```

## Demo

```bash
python azure_data_services.py
```
