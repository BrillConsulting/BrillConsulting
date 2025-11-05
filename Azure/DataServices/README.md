# Azure Data Services Integration

Comprehensive implementation of Azure data services including SQL Database, Cosmos DB, Blob Storage, Data Lake, and Table Storage.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a complete Python implementation for Azure Data Services, featuring SQL Database operations, NoSQL with Cosmos DB, object storage with Blob Storage, big data with Data Lake, and Table Storage for structured data. Built for enterprise data management and analytics workloads.

## Features

### SQL Database Capabilities
- **Database Management**: Create, configure, and manage databases
- **Table Operations**: DDL and DML operations
- **Query Execution**: Complex SQL queries
- **Transaction Management**: ACID compliance
- **Connection Pooling**: Efficient connection management
- **Stored Procedures**: Custom database logic
- **Performance Tuning**: Indexing and optimization

### Cosmos DB Features
- **Multi-Model Support**: Document, key-value, graph, column-family
- **Global Distribution**: Multi-region replication
- **Container Management**: Partition key configuration
- **CRUD Operations**: Create, read, update, delete documents
- **Query API**: SQL-like query language
- **Change Feed**: Real-time data streaming
- **Conflict Resolution**: Multi-master writes

### Blob Storage Features
- **Container Management**: Create and configure containers
- **Blob Operations**: Upload, download, delete blobs
- **Blob Types**: Block, append, and page blobs
- **Access Tiers**: Hot, cool, and archive storage
- **Metadata Management**: Custom blob metadata
- **Shared Access Signatures**: Secure temporary access
- **Lifecycle Management**: Automated tier transitions

### Data Lake Features
- **Hierarchical Namespace**: File system operations
- **ACL Management**: Fine-grained access control
- **Directory Operations**: Create, list, delete directories
- **Big Data Analytics**: Integration with analytics tools
- **Performance**: Optimized for large-scale analytics

### Table Storage Features
- **Entity Operations**: Insert, update, delete entities
- **Batch Operations**: Transactional entity groups
- **Query Filters**: Complex query conditions
- **Schema-less**: Flexible entity structure

## Architecture

```
DataServices/
├── azure_data_services.py     # Main implementation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/DataServices

# Install dependencies
pip install -r requirements.txt
```

## Configuration

```python
from azure_data_services import (
    AzureSQLDatabase,
    AzureCosmosDB,
    AzureBlobStorage
)

# SQL Database
sql_db = AzureSQLDatabase(
    server="myserver.database.windows.net",
    database="mydb",
    username="admin",
    password="password"
)

# Cosmos DB
cosmos = AzureCosmosDB(
    endpoint="https://myaccount.documents.azure.com:443/",
    key="your-key",
    database="mydb"
)

# Blob Storage
blob = AzureBlobStorage(
    account_name="mystorageaccount",
    account_key="your-key"
)
```

## Usage Examples

### SQL Database Operations

```python
# Create table
sql_db.create_table(
    "customers",
    schema={
        "id": "INT PRIMARY KEY",
        "name": "NVARCHAR(100)",
        "email": "NVARCHAR(255)",
        "created_at": "DATETIME"
    }
)

# Insert data
sql_db.insert_data("customers", {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2024-01-01"
})

# Query data
results = sql_db.query("SELECT * FROM customers WHERE name LIKE '%John%'")
for row in results:
    print(row)

# Update data
sql_db.update_data(
    "customers",
    {"email": "newemail@example.com"},
    "id = 1"
)
```

### Cosmos DB Operations

```python
# Create container
cosmos.create_container(
    "products",
    partition_key="/category"
)

# Create item
cosmos.create_item("products", {
    "id": "prod_001",
    "name": "Laptop",
    "category": "Electronics",
    "price": 999.99,
    "in_stock": True
})

# Read item
item = cosmos.read_item("products", "prod_001", "Electronics")
print(item)

# Query items
query = "SELECT * FROM c WHERE c.price < 1000"
results = cosmos.query_items("products", query)

# Update item
cosmos.update_item("products", "prod_001", {
    "price": 899.99
}, "Electronics")
```

### Blob Storage Operations

```python
# Create container
blob.create_container("images")

# Upload blob
with open("photo.jpg", "rb") as data:
    blob.upload_blob("images", "photo.jpg", data.read())

# Download blob
data = blob.download_blob("images", "photo.jpg")
with open("downloaded_photo.jpg", "wb") as f:
    f.write(data)

# List blobs
blobs = blob.list_blobs("images")
for blob_item in blobs:
    print(blob_item['name'])

# Delete blob
blob.delete_blob("images", "photo.jpg")
```

### Data Lake Operations

```python
from azure_data_services import AzureDataLake

datalake = AzureDataLake(
    account_name="mydatalake",
    account_key="your-key"
)

# Create directory
datalake.create_directory("raw-data/2024/01")

# Upload file
with open("data.csv", "rb") as data:
    datalake.upload_file("raw-data/2024/01/data.csv", data.read())

# List files
files = datalake.list_files("raw-data/2024/01")
```

### Table Storage Operations

```python
from azure_data_services import AzureTableStorage

table = AzureTableStorage(
    account_name="mystorage",
    account_key="your-key"
)

# Create table
table.create_table("logs")

# Insert entity
table.insert_entity("logs", {
    "PartitionKey": "2024-01",
    "RowKey": "001",
    "message": "Application started",
    "level": "INFO"
})

# Query entities
entities = table.query_entities(
    "logs",
    "PartitionKey eq '2024-01' and level eq 'ERROR'"
)
```

## Running Demos

```bash
# Run all demo functions
python azure_data_services.py
```

## API Reference

### AzureSQLDatabase

**`create_table(table_name, schema)`** - Create new table

**`insert_data(table_name, data)`** - Insert row

**`query(sql_query, parameters)`** - Execute query

**`update_data(table_name, data, where_clause)`** - Update rows

### AzureCosmosDB

**`create_container(container_name, partition_key)`** - Create container

**`create_item(container, item)`** - Create document

**`read_item(container, item_id, partition_key)`** - Read document

**`query_items(container, query)`** - Query documents

### AzureBlobStorage

**`create_container(container_name)`** - Create container

**`upload_blob(container, blob_name, data)`** - Upload blob

**`download_blob(container, blob_name)`** - Download blob

**`list_blobs(container)`** - List blobs

## Best Practices

### 1. Use Connection Pooling
```python
sql_db = AzureSQLDatabase(
    server="myserver.database.windows.net",
    database="mydb",
    pool_size=10
)
```

### 2. Partition Key Design (Cosmos DB)
```python
# Good partition key: evenly distributed
cosmos.create_container("orders", partition_key="/customerId")

# Avoid: hot partitions
# Don't use "/status" if most orders have same status
```

### 3. Use Access Tiers (Blob Storage)
```python
# Set access tier for cost optimization
blob.set_blob_tier("archive-container", "old-data.zip", "Archive")
```

### 4. Implement Retry Logic
```python
from azure.core.exceptions import ServiceRequestError

try:
    cosmos.create_item(container, item)
except ServiceRequestError:
    # Implement exponential backoff
    time.sleep(1)
    cosmos.create_item(container, item)
```

### 5. Use Batch Operations
```python
# Batch insert for better performance
entities = [entity1, entity2, entity3]
table.batch_insert_entities("logs", entities)
```

## Use Cases

### 1. E-Commerce Application
```python
# SQL for transactional data
sql_db.insert_data("orders", order_data)

# Cosmos DB for product catalog
cosmos.create_item("products", product_data)

# Blob Storage for product images
blob.upload_blob("product-images", f"{product_id}.jpg", image_data)
```

### 2. IoT Data Pipeline
```python
# Table Storage for device telemetry
table.insert_entity("telemetry", {
    "PartitionKey": device_id,
    "RowKey": timestamp,
    "temperature": 25.5
})

# Data Lake for raw data
datalake.upload_file(f"raw/{date}/data.json", json_data)
```

### 3. Data Warehouse
```python
# SQL Database for structured data
sql_db.bulk_insert("fact_sales", sales_data)

# Blob Storage for staging
blob.upload_blob("staging", "daily_export.csv", csv_data)
```

## Performance Optimization

### 1. Use Bulk Operations
```python
# Bulk insert for SQL
sql_db.bulk_insert("customers", customer_list)
```

### 2. Enable Caching
```python
# Cache frequently accessed data
cache = {}
if key not in cache:
    cache[key] = cosmos.read_item(container, key, partition)
```

### 3. Use Async Operations
```python
# Async blob upload for better throughput
await blob.upload_blob_async(container, name, data)
```

## Security Considerations

1. **Connection Strings**: Use Azure Key Vault
2. **Firewall Rules**: Restrict database access
3. **Encryption**: Enable at-rest encryption
4. **RBAC**: Use role-based access control
5. **Audit Logging**: Track all data access

## Troubleshooting

**Issue**: Connection timeout
**Solution**: Check firewall rules and network connectivity

**Issue**: High latency
**Solution**: Use appropriate partition keys and indexes

**Issue**: Storage quota exceeded
**Solution**: Implement lifecycle policies

## Deployment

### Azure Setup
```bash
# Create SQL Database
az sql server create --name myserver --resource-group rg --location eastus
az sql db create --server myserver --name mydb --resource-group rg

# Create Cosmos DB
az cosmosdb create --name myaccount --resource-group rg

# Create Storage Account
az storage account create --name mystorage --resource-group rg
```

## Monitoring

### Key Metrics
- Query performance
- Storage usage
- Request units (Cosmos DB)
- Connection pool utilization
- Error rates

## Dependencies

```
Python >= 3.8
azure-cosmos >= 4.0.0
azure-storage-blob >= 12.0.0
pyodbc >= 4.0.0
```

## Support

For questions or support:
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure Synapse](../AzureSynapse/)
- [Cosmos DB](../CosmosDB/)
- [Data Lake](../DataServices/)

---

**Built with Azure Data Services** | **Brill Consulting © 2024**
