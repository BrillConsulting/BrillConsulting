# Azure Cosmos DB Integration

Advanced implementation of Azure Cosmos DB with multi-model NoSQL database capabilities, global distribution, and comprehensive data operations.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a comprehensive Python implementation for Azure Cosmos DB, featuring multi-model database operations, global distribution, consistency level management, change feed processing, and stored procedures. Built for globally distributed applications requiring low-latency data access, automatic scaling, and multiple consistency models.

## Features

### Core Capabilities
- **Multi-Model Support**: Document, key-value, graph, and column-family APIs
- **Global Distribution**: Multi-region writes and automatic failover
- **Consistency Levels**: Five tunable consistency models
- **Automatic Indexing**: Schema-agnostic automatic indexing
- **Partitioning**: Horizontal scaling with logical and physical partitions
- **Change Feed**: Real-time change data capture
- **Stored Procedures**: Server-side logic execution
- **Triggers**: Pre and post-operation triggers

### Advanced Features
- **Multi-Region Replication**: Geo-redundant data distribution
- **Request Units (RUs)**: Predictable performance and cost
- **Time-to-Live (TTL)**: Automatic document expiration
- **Conflict Resolution**: Multi-master write conflict policies
- **Point-in-Time Restore**: Continuous backup and restore
- **Analytical Store**: Real-time analytics without ETL
- **Serverless Mode**: Pay-per-operation pricing model
- **Bulk Operations**: Efficient batch processing

## Architecture

```
CosmosDB/
├── cosmos_d_b.py              # Main implementation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

### Key Components

1. **CosmosDBManager**: Main service interface
   - Database and container management
   - CRUD operations
   - Query execution

2. **Document Operations**:
   - Create, read, update, delete documents
   - Upsert and replace operations
   - Batch operations

3. **Query Engine**:
   - SQL API queries
   - Cross-partition queries
   - Parameterized queries

4. **Change Feed**:
   - Real-time change notifications
   - Change feed processor
   - Event-driven architectures

5. **Stored Procedures & Triggers**:
   - Server-side JavaScript execution
   - Transactional operations
   - Pre/post operation hooks

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/CosmosDB

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your Azure Cosmos DB credentials:

```python
from cosmos_d_b import CosmosDBManager

manager = CosmosDBManager(
    endpoint="https://your-account.documents.azure.com:443/",
    key="your-primary-key",
    database_name="your-database",
    container_name="your-container"
)
```

### Environment Variables (Recommended)

```bash
export COSMOS_DB_ENDPOINT="https://your-account.documents.azure.com:443/"
export COSMOS_DB_KEY="your-primary-key"
export COSMOS_DB_DATABASE="your-database"
export COSMOS_DB_CONTAINER="your-container"
```

## Usage Examples

### 1. Database and Container Management

```python
from cosmos_d_b import CosmosDBManager

manager = CosmosDBManager(
    endpoint="https://your-account.documents.azure.com:443/",
    key="your-primary-key"
)

# Create database
database = manager.create_database(
    database_id="ProductsDB",
    throughput=400  # Request Units per second
)

# Create container with partition key
container = manager.create_container(
    database_id="ProductsDB",
    container_id="Products",
    partition_key_path="/category",
    throughput=400
)

print(f"Database: {database['id']}")
print(f"Container: {container['id']}")

# List all databases
databases = manager.list_databases()
for db in databases:
    print(f"Database: {db['id']}")
```

### 2. Document CRUD Operations

```python
# Create a document
product = {
    "id": "product-001",
    "name": "Laptop",
    "category": "Electronics",
    "price": 999.99,
    "stock": 50,
    "tags": ["computers", "hardware"]
}

created = manager.create_document(
    database_id="ProductsDB",
    container_id="Products",
    document=product,
    partition_key="Electronics"
)

print(f"Created document: {created['id']}")

# Read a document
retrieved = manager.read_document(
    database_id="ProductsDB",
    container_id="Products",
    document_id="product-001",
    partition_key="Electronics"
)

print(f"Product: {retrieved['name']}, Price: ${retrieved['price']}")

# Update a document
retrieved['price'] = 899.99
retrieved['stock'] = 45

updated = manager.update_document(
    database_id="ProductsDB",
    container_id="Products",
    document=retrieved,
    partition_key="Electronics"
)

print(f"Updated price: ${updated['price']}")

# Delete a document
manager.delete_document(
    database_id="ProductsDB",
    container_id="Products",
    document_id="product-001",
    partition_key="Electronics"
)

print("Document deleted")
```

### 3. Upsert Operations

```python
# Upsert (create if not exists, update if exists)
product = {
    "id": "product-002",
    "name": "Mouse",
    "category": "Electronics",
    "price": 29.99,
    "stock": 100
}

upserted = manager.upsert_document(
    database_id="ProductsDB",
    container_id="Products",
    document=product,
    partition_key="Electronics"
)

print(f"Upserted document: {upserted['id']}")
```

### 4. Query Documents

```python
# Simple query
query = "SELECT * FROM c WHERE c.category = 'Electronics'"

results = manager.query_documents(
    database_id="ProductsDB",
    container_id="Products",
    query=query,
    enable_cross_partition_query=True
)

for item in results:
    print(f"{item['name']}: ${item['price']}")

# Parameterized query
query = "SELECT * FROM c WHERE c.price < @maxPrice ORDER BY c.price DESC"
parameters = [{"name": "@maxPrice", "value": 1000}]

results = manager.query_documents(
    database_id="ProductsDB",
    container_id="Products",
    query=query,
    parameters=parameters
)

# Complex query with aggregation
query = """
    SELECT c.category,
           COUNT(1) as count,
           AVG(c.price) as avg_price,
           SUM(c.stock) as total_stock
    FROM c
    GROUP BY c.category
"""

results = manager.query_documents(
    database_id="ProductsDB",
    container_id="Products",
    query=query
)

for item in results:
    print(f"{item['category']}: {item['count']} items, Avg: ${item['avg_price']:.2f}")
```

### 5. Bulk Operations

```python
# Bulk create documents
products = [
    {
        "id": f"product-{i:03d}",
        "name": f"Product {i}",
        "category": "Electronics" if i % 2 == 0 else "Clothing",
        "price": 50 + (i * 10),
        "stock": 100 - i
    }
    for i in range(1, 101)
]

results = manager.bulk_create_documents(
    database_id="ProductsDB",
    container_id="Products",
    documents=products
)

print(f"Created {len(results)} documents")

# Bulk update with batch operations
updates = []
for i in range(1, 11):
    updates.append({
        "id": f"product-{i:03d}",
        "price": 100 + (i * 5),
        "category": "Electronics"
    })

updated_count = manager.bulk_update_documents(
    database_id="ProductsDB",
    container_id="Products",
    documents=updates
)

print(f"Updated {updated_count} documents")
```

### 6. Change Feed Processing

```python
# Monitor changes in real-time
def process_changes(changes):
    """Process change feed items"""
    for change in changes:
        print(f"Changed document: {change['id']}")
        print(f"Operation: {change.get('_lsn')}")  # Log sequence number

# Start change feed processor
manager.start_change_feed_processor(
    database_id="ProductsDB",
    container_id="Products",
    processor_name="ProductChangeProcessor",
    callback=process_changes,
    start_from_beginning=False
)

print("Change feed processor started")

# Get change feed from specific point
continuation_token = None
while True:
    changes, continuation_token = manager.get_change_feed(
        database_id="ProductsDB",
        container_id="Products",
        continuation_token=continuation_token,
        max_item_count=100
    )

    if not changes:
        break

    for change in changes:
        print(f"Processing change: {change['id']}")
```

### 7. Stored Procedures

```python
# Create stored procedure
stored_proc_body = """
function bulkUpdatePrice(categoryName, priceIncrease) {
    var collection = getContext().getCollection();
    var response = getContext().getResponse();

    var query = "SELECT * FROM c WHERE c.category = '" + categoryName + "'";
    var accept = collection.queryDocuments(
        collection.getSelfLink(),
        query,
        {},
        function (err, items, options) {
            if (err) throw err;

            var count = 0;
            items.forEach(function(item) {
                item.price += priceIncrease;
                collection.replaceDocument(
                    item._self,
                    item,
                    function(err, updated) {
                        if (err) throw err;
                        count++;
                    }
                );
            });

            response.setBody(count);
        }
    );

    if (!accept) throw new Error("Query not accepted");
}
"""

sproc = manager.create_stored_procedure(
    database_id="ProductsDB",
    container_id="Products",
    sproc_id="bulkUpdatePrice",
    sproc_body=stored_proc_body
)

# Execute stored procedure
result = manager.execute_stored_procedure(
    database_id="ProductsDB",
    container_id="Products",
    sproc_id="bulkUpdatePrice",
    params=["Electronics", 10],
    partition_key="Electronics"
)

print(f"Updated {result} documents")
```

## Running Demos

```bash
# Run the implementation
python cosmos_d_b.py
```

Demo output includes:
- Database and container creation
- CRUD operations
- Query examples
- Bulk operations
- Change feed processing

## Consistency Levels

Azure Cosmos DB offers five consistency levels:

### 1. Strong
```python
# Guaranteed linearizability
manager.set_consistency_level("Strong")
# Highest consistency, highest latency
```

### 2. Bounded Staleness
```python
# Reads lag behind writes by at most K versions or T interval
manager.set_consistency_level("BoundedStaleness", max_staleness=100000, max_interval=300)
# Balanced consistency and performance
```

### 3. Session (Default)
```python
# Consistent within a client session
manager.set_consistency_level("Session")
# Best for single-user scenarios
```

### 4. Consistent Prefix
```python
# Reads never see out-of-order writes
manager.set_consistency_level("ConsistentPrefix")
# Good for social media feeds
```

### 5. Eventual
```python
# Weakest consistency, best performance
manager.set_consistency_level("Eventual")
# Lowest latency, highest throughput
```

## API Reference

### CosmosDBManager

#### Database Methods

**`create_database(database_id, throughput)`**
- Creates a new database
- **Parameters**: database_id (str), throughput (int, optional)
- **Returns**: `Dict[str, Any]`

**`delete_database(database_id)`**
- Deletes a database
- **Parameters**: database_id (str)
- **Returns**: `None`

**`list_databases()`**
- Lists all databases
- **Returns**: `List[Dict[str, Any]]`

#### Container Methods

**`create_container(database_id, container_id, partition_key_path, throughput)`**
- Creates a new container
- **Parameters**: database_id (str), container_id (str), partition_key_path (str), throughput (int)
- **Returns**: `Dict[str, Any]`

**`delete_container(database_id, container_id)`**
- Deletes a container
- **Returns**: `None`

**`list_containers(database_id)`**
- Lists all containers in a database
- **Returns**: `List[Dict[str, Any]]`

#### Document Methods

**`create_document(database_id, container_id, document, partition_key)`**
- Creates a new document
- **Parameters**: database_id (str), container_id (str), document (Dict), partition_key (str)
- **Returns**: `Dict[str, Any]`

**`read_document(database_id, container_id, document_id, partition_key)`**
- Reads a document by ID
- **Returns**: `Dict[str, Any]`

**`update_document(database_id, container_id, document, partition_key)`**
- Updates a document
- **Returns**: `Dict[str, Any]`

**`delete_document(database_id, container_id, document_id, partition_key)`**
- Deletes a document
- **Returns**: `None`

**`upsert_document(database_id, container_id, document, partition_key)`**
- Creates or updates a document
- **Returns**: `Dict[str, Any]`

#### Query Methods

**`query_documents(database_id, container_id, query, parameters, enable_cross_partition_query)`**
- Executes a SQL query
- **Returns**: `List[Dict[str, Any]]`

**`query_documents_with_continuation(database_id, container_id, query, continuation_token, max_item_count)`**
- Executes query with pagination
- **Returns**: `Tuple[List[Dict], str]`

#### Bulk Methods

**`bulk_create_documents(database_id, container_id, documents)`**
- Creates multiple documents efficiently
- **Returns**: `List[Dict[str, Any]]`

**`bulk_update_documents(database_id, container_id, documents)`**
- Updates multiple documents
- **Returns**: `int`

**`bulk_delete_documents(database_id, container_id, document_ids, partition_keys)`**
- Deletes multiple documents
- **Returns**: `int`

#### Change Feed Methods

**`get_change_feed(database_id, container_id, continuation_token, max_item_count)`**
- Gets change feed items
- **Returns**: `Tuple[List[Dict], str]`

**`start_change_feed_processor(database_id, container_id, processor_name, callback)`**
- Starts change feed processor
- **Returns**: `None`

#### Stored Procedure Methods

**`create_stored_procedure(database_id, container_id, sproc_id, sproc_body)`**
- Creates a stored procedure
- **Returns**: `Dict[str, Any]`

**`execute_stored_procedure(database_id, container_id, sproc_id, params, partition_key)`**
- Executes a stored procedure
- **Returns**: `Any`

## Best Practices

### 1. Choose Appropriate Partition Key
```python
# Good: High cardinality, evenly distributed
partition_key = "/userId"  # Many unique values

# Bad: Low cardinality, hot partitions
partition_key = "/country"  # Few unique values
```

### 2. Optimize Request Units
```python
# Start with autoscale
container = manager.create_container(
    database_id="ProductsDB",
    container_id="Products",
    partition_key_path="/category",
    autoscale_max_throughput=4000  # Scales 400-4000 RU/s
)
```

### 3. Use Appropriate Consistency Level
```python
# Session consistency for most scenarios
manager.set_consistency_level("Session")

# Strong only when absolutely necessary
manager.set_consistency_level("Strong")
```

### 4. Implement Efficient Queries
```python
# Good: Query within partition
query = "SELECT * FROM c WHERE c.category = 'Electronics' AND c.price < 1000"

# Bad: Full collection scan
query = "SELECT * FROM c WHERE c.price < 1000"  # Cross-partition query
```

### 5. Use Bulk Operations for Large Datasets
```python
# Efficient bulk operations
results = manager.bulk_create_documents(
    database_id="ProductsDB",
    container_id="Products",
    documents=large_document_list
)
```

### 6. Implement TTL for Temporary Data
```python
# Set time-to-live at container level
container = manager.create_container(
    database_id="SessionsDB",
    container_id="UserSessions",
    partition_key_path="/userId",
    default_ttl=3600  # Documents expire after 1 hour
)

# Or set TTL per document
document = {
    "id": "session-001",
    "userId": "user123",
    "data": "...",
    "ttl": 1800  # Expires after 30 minutes
}
```

### 7. Monitor and Optimize RU Consumption
```python
# Check RU consumption
response = manager.query_documents(
    database_id="ProductsDB",
    container_id="Products",
    query="SELECT * FROM c WHERE c.price > 500"
)

ru_charge = response.headers['x-ms-request-charge']
print(f"Query consumed {ru_charge} RUs")
```

## Use Cases

### 1. E-commerce Product Catalog
Global product catalog with low-latency reads across multiple regions and automatic scaling.

### 2. IoT Data Storage
Store and query high-volume IoT telemetry data with time-series queries and TTL-based retention.

### 3. User Profile Management
Globally distributed user profiles with session consistency for personalized experiences.

### 4. Real-time Analytics
Process change feeds for real-time analytics, notifications, and data synchronization.

### 5. Content Management
Store and serve content with geo-replication for optimal performance worldwide.

### 6. Gaming Leaderboards
Low-latency reads and writes for real-time game state and player rankings.

## Troubleshooting

### Common Issues

**Issue**: High RU consumption
**Solution**: Optimize queries, use appropriate indexing, implement caching, choose better partition key

**Issue**: Hot partitions
**Solution**: Choose partition key with higher cardinality, redistribute data, use synthetic partition keys

**Issue**: Throttling (429 errors)
**Solution**: Increase provisioned throughput, implement retry logic with exponential backoff, use autoscale

**Issue**: Large document size
**Solution**: Split documents, use references, implement document sharding

**Issue**: Slow queries
**Solution**: Add composite indexes, avoid cross-partition queries, use pagination

**Issue**: Consistency conflicts
**Solution**: Implement conflict resolution policies, use appropriate consistency level

## Deployment

### Azure CLI Deployment
```bash
# Create Cosmos DB account
az cosmosdb create \
    --name my-cosmos-account \
    --resource-group my-resource-group \
    --kind GlobalDocumentDB \
    --default-consistency-level Session \
    --locations regionName=eastus failoverPriority=0 isZoneRedundant=False \
    --enable-automatic-failover true

# Create database
az cosmosdb sql database create \
    --account-name my-cosmos-account \
    --resource-group my-resource-group \
    --name ProductsDB \
    --throughput 400

# Create container
az cosmosdb sql container create \
    --account-name my-cosmos-account \
    --resource-group my-resource-group \
    --database-name ProductsDB \
    --name Products \
    --partition-key-path "/category" \
    --throughput 400
```

### Infrastructure as Code
```python
# Terraform example
resource "azurerm_cosmosdb_account" "db" {
  name                = "my-cosmos-account"
  resource_group_name = azurerm_resource_group.example.name
  location            = azurerm_resource_group.example.location
  offer_type         = "Standard"
  kind               = "GlobalDocumentDB"

  consistency_policy {
    consistency_level = "Session"
  }

  geo_location {
    location          = "eastus"
    failover_priority = 0
  }
}
```

### Container Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY cosmos_d_b.py .
CMD ["python", "cosmos_d_b.py"]
```

## Monitoring

### Key Metrics
- Request Units consumed
- Request rate and throttling
- Document count and storage
- Replication lag (multi-region)
- Query performance
- Indexing progress

### Azure Monitor Integration
```bash
# Enable diagnostic logs
az cosmosdb update \
    --name my-cosmos-account \
    --resource-group my-resource-group \
    --enable-analytical-storage true

# Configure alerts
az monitor metrics alert create \
    --name HighRUConsumption \
    --resource-group my-resource-group \
    --scopes /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.DocumentDB/databaseAccounts/{account} \
    --condition "avg Percentage CPU > 80" \
    --window-size 5m \
    --evaluation-frequency 1m
```

### Query Metrics
```python
# Enable query metrics
results = manager.query_documents(
    database_id="ProductsDB",
    container_id="Products",
    query="SELECT * FROM c WHERE c.price > 500",
    populate_query_metrics=True
)

metrics = results.query_metrics
print(f"Retrieved document count: {metrics['retrievedDocumentCount']}")
print(f"Total query execution time: {metrics['totalQueryExecutionTime']}")
print(f"Request charge: {metrics['requestCharge']}")
```

## Dependencies

```
Python >= 3.8
azure-core >= 1.26.0
azure-cosmos >= 4.3.0
typing
datetime
json
```

See `requirements.txt` for complete list.

## Version History

- **v1.0.0**: Initial release with basic CRUD operations
- **v1.1.0**: Added bulk operations and change feed
- **v1.2.0**: Enhanced query capabilities and stored procedures
- **v2.0.0**: Added multi-region support and analytical store

## Contributing

Contributions are welcome! Please submit pull requests or open issues on GitHub.

## License

This project is part of the Brill Consulting portfolio.

## Support

For questions or support:
- **Email**: clientbrill@gmail.com
- **LinkedIn**: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure Table Storage](../TableStorage/)
- [Azure SQL Database](../SQLDatabase/)
- [Azure Cache for Redis](../Redis/)

---

**Built with Azure Cosmos DB** | **Brill Consulting © 2024**
