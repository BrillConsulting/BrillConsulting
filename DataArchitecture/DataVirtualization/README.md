# Data Virtualization
Federated query engine with data abstraction and intelligent caching

## Overview

A production-grade data virtualization framework that provides unified access to multiple heterogeneous data sources without physical data movement. Features federated query execution, intelligent caching, query optimization, and virtual views for creating a logical data abstraction layer across your data landscape.

## Features

### Core Capabilities
- **Multi-Source Registration**: Connect to multiple heterogeneous data sources
- **Virtual Views**: Create unified views spanning multiple sources
- **Federated Queries**: Execute queries across distributed data sources
- **Query Caching**: Intelligent result caching with TTL management
- **Query Optimization**: Automatic query optimization with pushdown predicates
- **Data Federation**: Join data from multiple sources seamlessly

### Advanced Features
- **Materialized Views**: Cache frequently accessed views for performance
- **View Refresh**: Update materialized views on demand
- **Query Plan Analysis**: Explain and optimize query execution plans
- **Performance Metrics**: Track query statistics and cache hit rates
- **Source Health Monitoring**: Monitor data source availability
- **Access Optimization**: Automatic recommendations for performance improvement

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/DataArchitecture.git
cd DataArchitecture/DataVirtualization

# Install dependencies
pip install pandas

# Run the implementation
python datavirtualization.py
```

## Usage Examples

### Register Data Sources

```python
from datavirtualization import DataVirtualization

# Initialize virtualization system
dv = DataVirtualization()

# Register PostgreSQL database
dv.register_data_source(
    source_id="postgres_db",
    name="PostgreSQL Database",
    source_type="relational",
    connection_info={"host": "localhost", "port": 5432, "database": "sales"},
    tables={
        "customers": {"columns": ["id", "name", "email"]},
        "orders": {"columns": ["id", "customer_id", "amount", "date"]}
    }
)

# Register MongoDB
dv.register_data_source(
    source_id="mongodb",
    name="MongoDB",
    source_type="document",
    connection_info={"host": "localhost", "port": 27017, "database": "analytics"},
    tables={
        "events": {"columns": ["_id", "user_id", "event_type", "timestamp"]},
        "sessions": {"columns": ["_id", "user_id", "duration"]}
    }
)
```

### Create Virtual Views

```python
# Create unified customer view
customer_view = dv.create_virtual_view(
    view_id="customer_360",
    name="Customer 360 View",
    query="SELECT c.*, o.total_orders FROM customers c JOIN orders o ON c.id = o.customer_id",
    sources=["postgres_db"],
    metadata={"description": "Comprehensive customer view"}
)

# Create cross-source analytics view
unified_view = dv.create_virtual_view(
    view_id="unified_analytics",
    name="Unified Analytics View",
    query="SELECT * FROM events JOIN sessions ON events.user_id = sessions.user_id",
    sources=["mongodb", "s3_data"],
    metadata={"description": "Unified view across MongoDB and S3"}
)
```

### Execute Federated Queries

```python
# Execute query with caching
result = dv.execute_query(
    query="SELECT * FROM customer_360 WHERE total_orders > 5",
    enable_cache=True,
    enable_optimization=True
)

print(f"Execution time: {result['execution_time_ms']:.2f}ms")
print(f"Cached: {result['cached']}")
print(f"Rows: {result['result']['row_count']}")

# Second execution (cached)
result2 = dv.execute_query(
    query="SELECT * FROM customer_360 WHERE total_orders > 5",
    enable_cache=True
)

print(f"Second execution: {result2['execution_time_ms']:.2f}ms (cached: {result2['cached']})")
```

### Query Plan Analysis

```python
# Analyze query execution plan
plan = dv.analyze_query_plan(
    query="SELECT * FROM PostgreSQL JOIN MongoDB WHERE customer_id = user_id"
)

print("Query Plan:")
print(f"  Estimated cost: {plan['estimated_cost']}")
print(f"  Sources accessed: {plan['sources_accessed']}")

print("\nExecution steps:")
for i, step in enumerate(plan["steps"], 1):
    print(f"  {i}. {step['step']} (cost: {step['cost']})")
```

### Materialize Views

```python
# Materialize frequently accessed view
materialization = dv.materialize_view("customer_360")

print(f"Materialized view: {materialization['view_id']}")
print(f"Rows: {materialization['row_count']}")

# Refresh materialized view
refreshed = dv.refresh_materialized_view("customer_360")
print(f"Refreshed at: {refreshed['materialized_at']}")
```

### Create Data Federation

```python
# Federate multiple sources with join rules
federation = dv.create_data_federation(
    federation_id="analytics_federation",
    name="Analytics Federation",
    source_ids=["postgres_db", "mongodb", "s3_data"],
    join_rules={
        "postgres_db.customers.id": "mongodb.events.user_id",
        "postgres_db.customers.id": "s3_data.user_profiles.user_id"
    }
)

print(f"Federation created: {federation['federation_id']}")
print(f"Sources: {len(federation['sources'])}")
```

### Query Statistics

```python
# Get query performance statistics
stats = dv.get_query_statistics()

print(f"Total queries: {stats['total_queries']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
print(f"Avg execution time: {stats['avg_execution_time_ms']:.2f}ms")
print(f"Min execution time: {stats['min_execution_time_ms']:.2f}ms")
print(f"Max execution time: {stats['max_execution_time_ms']:.2f}ms")
```

### Optimize Data Access

```python
# Get optimization recommendations
optimization = dv.optimize_data_access("customer_360")

print(f"Current access count: {optimization['current_access_count']}")
print(f"Materialized: {optimization['materialized']}")

print("\nRecommendations:")
for rec in optimization["recommendations"]:
    print(f"  - {rec['type']}: {rec['reason']}")
```

### Cache Management

```python
# Clear query cache
cache_result = dv.clear_cache()
print(f"Cleared {cache_result['cleared_entries']} cache entries")
```

### Source Health Check

```python
# Check health of all data sources
health = dv.get_source_health()

print("Data Source Health:")
for source in health:
    print(f"  {source['name']}: {source['status']}")
    print(f"    Type: {source['type']}")
    print(f"    Tables: {source['tables']}")
```

## Demo Instructions

Run the included demonstration:

```bash
python datavirtualization.py
```

The demo showcases:
1. Registering multiple heterogeneous data sources
2. Creating virtual views across sources
3. Executing federated queries with caching
4. Query plan analysis and optimization
5. Materializing views for performance
6. Creating data federations
7. Query statistics and monitoring
8. Access optimization recommendations
9. Cache management
10. Source health checking
11. Comprehensive virtualization reporting

## Key Concepts

### Data Virtualization

Access data without physical movement:
- **Logical Layer**: Unified view of distributed data
- **Query Federation**: Join across multiple sources
- **On-Demand Access**: Query data where it lives
- **No ETL**: Eliminate traditional data movement

### Virtual Views

Logical data abstractions:
- Defined by queries, not physical storage
- Span multiple data sources
- Updated in real-time
- Reduced data duplication

### Query Optimization

Intelligent query processing:
- **Predicate Pushdown**: Filter at source
- **Join Optimization**: Minimize data transfer
- **Projection Pushdown**: Select only needed columns
- **Cost-Based**: Choose optimal execution plan

### Caching Strategy

Performance optimization:
- **Result Caching**: Cache query results with TTL
- **Materialized Views**: Persist frequently accessed data
- **Cache Invalidation**: Automatic expiration
- **Hit Rate Monitoring**: Track cache effectiveness

## Architecture

```
┌──────────────────────────────────────────────┐
│      Data Virtualization Layer               │
│                                              │
│  ┌────────────────────────────────────┐     │
│  │        Virtual Views               │     │
│  │  - customer_360                    │     │
│  │  - unified_analytics               │     │
│  └─────────────┬──────────────────────┘     │
│                │                             │
│                ▼                             │
│  ┌────────────────────────────────────┐     │
│  │    Federated Query Engine          │     │
│  │  - Query Optimization              │     │
│  │  - Pushdown Predicates             │     │
│  │  - Caching                         │     │
│  └─────────────┬──────────────────────┘     │
│                │                             │
│        ┌───────┴────────┬──────────┐        │
│        ▼                ▼          ▼        │
│  ┌──────────┐    ┌──────────┐  ┌────────┐  │
│  │PostgreSQL│    │ MongoDB  │  │   S3   │  │
│  └──────────┘    └──────────┘  └────────┘  │
└──────────────────────────────────────────────┘
```

## Use Cases

- **Data Integration**: Unified access to siloed data sources
- **Real-Time Analytics**: Query fresh data without ETL delays
- **Data Lake Access**: Query data directly in object storage
- **Hybrid Cloud**: Access on-premise and cloud data seamlessly
- **Legacy Integration**: Modern interface to legacy systems
- **Cost Optimization**: Reduce data duplication and storage
- **Agile Development**: Rapid data access without pipelines

## Performance Considerations

- Enable caching for frequently accessed queries
- Materialize views with high access frequency
- Use pushdown optimization for large datasets
- Monitor cache hit rates regularly
- Balance freshness vs. performance needs
- Implement proper indexing at sources
- Consider network latency between sources

## Best Practices

- Design efficient virtual views
- Use appropriate caching strategies
- Monitor query performance continuously
- Implement security at virtual layer
- Document view definitions clearly
- Test federation performance
- Plan for source failures
- Regular cache maintenance

## Integration

Integrate with:
- BI tools (Tableau, Power BI) via ODBC/JDBC
- Data catalogs for metadata
- Query engines (Presto, Trino)
- Security frameworks for access control
- Monitoring tools for performance tracking

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [linkedin.com/in/brillconsulting](https://linkedin.com/in/brillconsulting)
- Specialization: Data Architecture & Engineering Solutions
