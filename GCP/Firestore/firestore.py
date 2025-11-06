"""
Cloud Firestore NoSQL Database
Author: BrillConsulting
Description: Advanced Firestore with queries, transactions, and real-time updates
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import time


class FirestoreCollection:
    """Firestore collection management"""

    def __init__(self, project_id: str, collection_name: str):
        """
        Initialize collection manager

        Args:
            project_id: GCP project ID
            collection_name: Collection name
        """
        self.project_id = project_id
        self.collection_name = collection_name
        self.documents = []

    def create_document(self, document_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create document in collection

        Args:
            document_id: Document ID
            data: Document data

        Returns:
            Document creation result
        """
        print(f"\n{'='*60}")
        print("Creating Firestore Document")
        print(f"{'='*60}")

        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Create document
doc_ref = db.collection('{self.collection_name}').document('{document_id}')
doc_ref.set({data})

print(f"Document created: {{doc_ref.id}}")
"""

        result = {
            'collection': self.collection_name,
            'document_id': document_id,
            'data': data,
            'path': f"{self.collection_name}/{document_id}",
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.documents.append(result)

        print(f"✓ Document created: {document_id}")
        print(f"  Collection: {self.collection_name}")
        print(f"  Fields: {list(data.keys())}")
        print(f"{'='*60}")

        return result

    def batch_write(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch write multiple documents

        Args:
            documents: List of documents to write

        Returns:
            Batch write result
        """
        print(f"\n{'='*60}")
        print("Batch Writing Documents")
        print(f"{'='*60}")

        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Create batch
batch = db.batch()

# Add documents to batch
documents = {documents}

for doc in documents:
    doc_ref = db.collection('{self.collection_name}').document(doc['id'])
    batch.set(doc_ref, doc['data'])

# Commit batch
batch.commit()
print(f"Batch committed: {{len(documents)}} documents")
"""

        result = {
            'collection': self.collection_name,
            'documents_count': len(documents),
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Batch write: {len(documents)} documents")
        print(f"{'='*60}")

        return result

    def update_document(self, document_id: str, updates: Dict[str, Any]) -> str:
        """
        Update document fields

        Args:
            document_id: Document ID
            updates: Fields to update

        Returns:
            Update code
        """
        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Update document
doc_ref = db.collection('{self.collection_name}').document('{document_id}')
doc_ref.update({updates})

# Or use set with merge
doc_ref.set({updates}, merge=True)

print("Document updated")
"""

        print(f"\n✓ Document update configured: {document_id}")
        return code


class FirestoreQuery:
    """Advanced Firestore queries"""

    def __init__(self, project_id: str, collection_name: str):
        """Initialize query builder"""
        self.project_id = project_id
        self.collection_name = collection_name

    def simple_query(self, field: str, operator: str, value: Any) -> str:
        """
        Simple field query

        Args:
            field: Field name
            operator: Comparison operator
            value: Field value

        Returns:
            Query code
        """
        print(f"\n{'='*60}")
        print("Firestore Simple Query")
        print(f"{'='*60}")

        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Query documents
docs = db.collection('{self.collection_name}') \\
    .where('{field}', '{operator}', {repr(value)}) \\
    .stream()

for doc in docs:
    print(f"{{doc.id}}: {{doc.to_dict()}}")
"""

        print(f"✓ Query: {field} {operator} {value}")
        print(f"{'='*60}")

        return code

    def compound_query(self, filters: List[Dict[str, Any]]) -> str:
        """
        Compound query with multiple filters

        Args:
            filters: List of filter conditions

        Returns:
            Compound query code
        """
        print(f"\n{'='*60}")
        print("Firestore Compound Query")
        print(f"{'='*60}")

        filter_lines = []
        for f in filters:
            filter_lines.append(f"    .where('{f['field']}', '{f['operator']}', {repr(f['value'])})")

        filters_code = " \\\n".join(filter_lines)

        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Compound query
docs = db.collection('{self.collection_name}') \\
{filters_code} \\
    .stream()

for doc in docs:
    print(f"{{doc.id}}: {{doc.to_dict()}}")
"""

        print(f"✓ Compound query with {len(filters)} filters")
        print(f"{'='*60}")

        return code

    def range_query(self, field: str, start_value: Any, end_value: Any) -> str:
        """
        Range query

        Args:
            field: Field name
            start_value: Range start
            end_value: Range end

        Returns:
            Range query code
        """
        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Range query
docs = db.collection('{self.collection_name}') \\
    .where('{field}', '>=', {repr(start_value)}) \\
    .where('{field}', '<=', {repr(end_value)}) \\
    .stream()

for doc in docs:
    print(f"{{doc.id}}: {{doc.to_dict()}}")
"""

        print(f"\n✓ Range query: {field} between {start_value} and {end_value}")
        return code

    def order_and_limit(self, order_by: str, limit: int, direction: str = 'ASCENDING') -> str:
        """
        Query with ordering and limit

        Args:
            order_by: Field to order by
            limit: Result limit
            direction: Sort direction

        Returns:
            Query code
        """
        direction_code = 'firestore.Query.DESCENDING' if direction == 'DESCENDING' else 'firestore.Query.ASCENDING'

        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Query with ordering and limit
docs = db.collection('{self.collection_name}') \\
    .order_by('{order_by}', direction={direction_code}) \\
    .limit({limit}) \\
    .stream()

for doc in docs:
    print(f"{{doc.id}}: {{doc.to_dict()}}")
"""

        print(f"\n✓ Ordered query: {order_by} {direction}, limit {limit}")
        return code

    def pagination_query(self, page_size: int) -> str:
        """
        Paginated query

        Args:
            page_size: Documents per page

        Returns:
            Pagination code
        """
        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Paginated query
collection = db.collection('{self.collection_name}')

# First page
docs = collection.limit({page_size}).stream()
last_doc = None

for doc in docs:
    print(f"{{doc.id}}: {{doc.to_dict()}}")
    last_doc = doc

# Next page
if last_doc:
    next_docs = collection \\
        .start_after(last_doc) \\
        .limit({page_size}) \\
        .stream()

    for doc in next_docs:
        print(f"{{doc.id}}: {{doc.to_dict()}}")
"""

        print(f"\n✓ Pagination query: {page_size} per page")
        return code


class FirestoreTransaction:
    """Firestore transactions and atomic operations"""

    def __init__(self, project_id: str):
        """Initialize transaction manager"""
        self.project_id = project_id

    def execute_transaction(self, operations: List[Dict[str, Any]]) -> str:
        """
        Execute atomic transaction

        Args:
            operations: List of operations

        Returns:
            Transaction code
        """
        print(f"\n{'='*60}")
        print("Firestore Transaction")
        print(f"{'='*60}")

        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Define transaction
@firestore.transactional
def update_in_transaction(transaction, doc_refs):
    # Read phase
    snapshots = []
    for doc_ref in doc_refs:
        snapshot = doc_ref.get(transaction=transaction)
        snapshots.append(snapshot)

    # Validation and computation
    for snapshot in snapshots:
        if snapshot.exists:
            current_value = snapshot.get('count')
            # Business logic here

    # Write phase
    for doc_ref in doc_refs:
        transaction.update(doc_ref, {{'count': firestore.Increment(1)}})

# Execute transaction
transaction = db.transaction()
doc_refs = [
    db.collection('counters').document('counter1'),
    db.collection('counters').document('counter2')
]

update_in_transaction(transaction, doc_refs)
print("Transaction completed")
"""

        print(f"✓ Transaction with {len(operations)} operations")
        print(f"{'='*60}")

        return code

    def increment_field(self, collection: str, document_id: str, field: str, amount: int = 1) -> str:
        """
        Atomically increment field

        Args:
            collection: Collection name
            document_id: Document ID
            field: Field to increment
            amount: Increment amount

        Returns:
            Increment code
        """
        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Atomic increment
doc_ref = db.collection('{collection}').document('{document_id}')
doc_ref.update({{
    '{field}': firestore.Increment({amount})
}})

print("Field incremented atomically")
"""

        print(f"\n✓ Atomic increment: {field} +{amount}")
        return code

    def array_operations(self, collection: str, document_id: str) -> str:
        """
        Array union and remove operations

        Args:
            collection: Collection name
            document_id: Document ID

        Returns:
            Array operations code
        """
        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

doc_ref = db.collection('{collection}').document('{document_id}')

# Array union (add if not exists)
doc_ref.update({{
    'tags': firestore.ArrayUnion(['new_tag', 'another_tag'])
}})

# Array remove
doc_ref.update({{
    'tags': firestore.ArrayRemove(['old_tag'])
}})

print("Array operations completed")
"""

        print(f"\n✓ Array operations configured")
        return code


class FirestoreIndex:
    """Firestore index management"""

    def __init__(self, project_id: str):
        """Initialize index manager"""
        self.project_id = project_id

    def create_composite_index(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create composite index

        Args:
            config: Index configuration

        Returns:
            Index details
        """
        print(f"\n{'='*60}")
        print("Creating Composite Index")
        print(f"{'='*60}")

        collection = config.get('collection')
        fields = config.get('fields', [])

        # Generate index JSON
        index_json = {
            "indexes": [
                {
                    "collectionGroup": collection,
                    "queryScope": "COLLECTION",
                    "fields": [
                        {"fieldPath": field['name'], "order": field.get('order', 'ASCENDING')}
                        for field in fields
                    ]
                }
            ]
        }

        code = f"""
# Add to firestore.indexes.json
{json.dumps(index_json, indent=2)}

# Deploy with Firebase CLI
# firebase deploy --only firestore:indexes

# Or use gcloud
# gcloud firestore indexes composite create \\
#   --collection-group={collection} \\
#   --query-scope=COLLECTION \\
#   --field-config field-path={fields[0]['name']},order=ASCENDING
"""

        result = {
            'collection': collection,
            'fields': fields,
            'index_config': index_json,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Composite index created: {collection}")
        print(f"  Fields: {[f['name'] for f in fields]}")
        print(f"{'='*60}")

        return result


class FirestoreRealtime:
    """Real-time listeners and subscriptions"""

    def __init__(self, project_id: str):
        """Initialize realtime manager"""
        self.project_id = project_id

    def create_document_listener(self, collection: str, document_id: str) -> str:
        """
        Listen to document changes

        Args:
            collection: Collection name
            document_id: Document ID

        Returns:
            Listener code
        """
        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Listen to document
def on_snapshot(doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        print(f"Received document snapshot: {{doc.id}}")
        print(f"Data: {{doc.to_dict()}}")

doc_ref = db.collection('{collection}').document('{document_id}')
doc_watch = doc_ref.on_snapshot(on_snapshot)

# Keep listening...
# To stop: doc_watch.unsubscribe()
"""

        print(f"\n✓ Document listener: {collection}/{document_id}")
        return code

    def create_collection_listener(self, collection: str, query_filter: Optional[Dict[str, Any]] = None) -> str:
        """
        Listen to collection changes

        Args:
            collection: Collection name
            query_filter: Optional query filter

        Returns:
            Collection listener code
        """
        filter_code = ""
        if query_filter:
            filter_code = f".where('{query_filter['field']}', '{query_filter['operator']}', {repr(query_filter['value'])})"

        code = f"""
from google.cloud import firestore

db = firestore.Client(project='{self.project_id}')

# Listen to collection
def on_snapshot(col_snapshot, changes, read_time):
    print(f"Received {{len(changes)}} changes")

    for change in changes:
        if change.type.name == 'ADDED':
            print(f"New document: {{change.document.id}}")
        elif change.type.name == 'MODIFIED':
            print(f"Modified document: {{change.document.id}}")
        elif change.type.name == 'REMOVED':
            print(f"Removed document: {{change.document.id}}")

col_query = db.collection('{collection}'){filter_code}
col_watch = col_query.on_snapshot(on_snapshot)

# Keep listening...
# To stop: col_watch.unsubscribe()
"""

        print(f"\n✓ Collection listener: {collection}")
        return code


class FirestoreManager:
    """Comprehensive Firestore management"""

    def __init__(self, project_id: str = 'my-project'):
        """Initialize Firestore manager"""
        self.project_id = project_id
        self.collections = {}
        self.documents_created = 0

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'collections': len(self.collections),
            'documents_created': self.documents_created,
            'features': [
                'documents',
                'queries',
                'transactions',
                'indexes',
                'realtime_listeners',
                'batch_operations'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Firestore capabilities"""
    print("=" * 60)
    print("Cloud Firestore NoSQL Database Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'

    # Create documents
    collection = FirestoreCollection(project_id, 'users')

    doc = collection.create_document(
        document_id='user123',
        data={
            'name': 'John Doe',
            'email': 'john@example.com',
            'age': 30,
            'tags': ['premium', 'verified']
        }
    )

    batch_docs = [
        {'id': f'user{i}', 'data': {'name': f'User {i}', 'score': i * 10}}
        for i in range(5)
    ]
    batch_result = collection.batch_write(batch_docs)

    update_code = collection.update_document('user123', {'age': 31})

    # Queries
    query_mgr = FirestoreQuery(project_id, 'users')

    simple_q = query_mgr.simple_query('age', '>=', 18)

    compound_q = query_mgr.compound_query([
        {'field': 'age', 'operator': '>=', 'value': 18},
        {'field': 'verified', 'operator': '==', 'value': True}
    ])

    range_q = query_mgr.range_query('score', 0, 100)
    ordered_q = query_mgr.order_and_limit('score', 10, 'DESCENDING')
    pagination_q = query_mgr.pagination_query(20)

    # Transactions
    tx_mgr = FirestoreTransaction(project_id)
    tx_code = tx_mgr.execute_transaction([
        {'type': 'update', 'collection': 'counters', 'document': 'counter1'}
    ])

    increment_code = tx_mgr.increment_field('counters', 'page_views', 'count', 1)
    array_code = tx_mgr.array_operations('users', 'user123')

    # Indexes
    index_mgr = FirestoreIndex(project_id)
    index = index_mgr.create_composite_index({
        'collection': 'users',
        'fields': [
            {'name': 'age', 'order': 'ASCENDING'},
            {'name': 'score', 'order': 'DESCENDING'}
        ]
    })

    # Real-time listeners
    realtime_mgr = FirestoreRealtime(project_id)
    doc_listener = realtime_mgr.create_document_listener('users', 'user123')
    col_listener = realtime_mgr.create_collection_listener('users', {
        'field': 'verified',
        'operator': '==',
        'value': True
    })

    # Manager info
    mgr = FirestoreManager(project_id)
    mgr.collections['users'] = collection
    mgr.documents_created = 1 + len(batch_docs)

    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Firestore Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Collections: {info['collections']}")
    print(f"Documents created: {info['documents_created']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    demo()
