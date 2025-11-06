# Firestore - NoSQL Document Database

Comprehensive Cloud Firestore implementation for scalable NoSQL document storage with real-time synchronization.

## Features

### Document Management
- **CRUD Operations**: Create, read, update, delete documents
- **Batch Operations**: Atomic multi-document operations
- **Nested Data**: Support for complex nested structures
- **Document References**: Cross-collection references

### Advanced Queries
- **Simple Queries**: Single-field filtering
- **Compound Queries**: Multiple where clauses with composite indexes
- **Range Queries**: Greater than, less than operations
- **Pagination**: Cursor-based pagination with start_after()
- **Ordering**: Sort results by multiple fields

### Transactions
- **Read-Write Transactions**: Atomic multi-document updates
- **Field Increments**: Atomic counter operations with firestore.Increment
- **Array Operations**: ArrayUnion and ArrayRemove for array fields
- **Conditional Updates**: Update only if conditions met

### Indexing
- **Automatic Indexing**: Single-field indexes created automatically
- **Composite Indexes**: Multi-field indexes for complex queries
- **Index Management**: Create and manage custom indexes
- **Query Performance**: Optimized queries with proper indexes

### Real-Time Features
- **Collection Listeners**: Real-time updates on collection changes
- **Document Listeners**: Real-time updates on document changes
- **Query Filters**: Real-time updates with filtered queries
- **Snapshot Callbacks**: on_snapshot event handlers

## Usage Example

```python
from firestore import FirestoreManager

# Initialize manager
mgr = FirestoreManager(project_id='my-gcp-project')

# Create document
doc = mgr.collection.create_document('users', 'user123', {
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30,
    'verified': True
})

# Batch write
mgr.collection.batch_write([
    {'id': 'user1', 'name': 'Alice'},
    {'id': 'user2', 'name': 'Bob'}
])

# Compound query
results = mgr.query.compound_query([
    {'field': 'age', 'operator': '>=', 'value': 18},
    {'field': 'verified', 'operator': '==', 'value': True}
])

# Pagination
page1 = mgr.query.pagination_query(page_size=10)

# Atomic increment
mgr.transaction.increment_field('users', 'user123', 'login_count', 1)

# Array operations
mgr.transaction.array_operations('users', 'user123')

# Create composite index
index = mgr.index.create_composite_index({
    'collection_id': 'users',
    'fields': [
        {'field_path': 'age', 'order': 'ASCENDING'},
        {'field_path': 'created_at', 'order': 'DESCENDING'}
    ]
})

# Real-time listener
mgr.realtime.create_collection_listener('users', {
    'field': 'verified',
    'operator': '==',
    'value': True
})
```

## Best Practices

1. **Use composite indexes** for complex queries
2. **Implement pagination** for large result sets
3. **Use transactions** for atomic multi-document updates
4. **Leverage real-time listeners** for live data
5. **Structure data** to minimize reads
6. **Use subcollections** for hierarchical data

## Requirements

```
google-cloud-firestore
```

## Author

BrillConsulting - Enterprise Cloud Solutions
