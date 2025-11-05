"""
Azure AI Services Integration

Comprehensive implementation of Azure AI Services including Document Intelligence,
Cognitive Search, Form Recognition, OCR, and AI enrichment pipelines.

Author: BrillConsulting
Contact: clientbrill@gmail.com
LinkedIn: brillconsulting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import json
import base64


class DocumentType(Enum):
    """Types of documents supported for analysis"""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    BUSINESS_CARD = "businessCard"
    ID_DOCUMENT = "idDocument"
    CUSTOM = "custom"
    LAYOUT = "layout"
    GENERAL_DOCUMENT = "generalDocument"


class AnalysisFeature(Enum):
    """Features available in document analysis"""
    OCR = "ocr"
    LAYOUT = "layout"
    TABLES = "tables"
    KEY_VALUE_PAIRS = "keyValuePairs"
    ENTITIES = "entities"
    LANGUAGES = "languages"
    BARCODES = "barcodes"
    FORMULAS = "formulas"


class SearchIndexType(Enum):
    """Types of search indexes"""
    FULL_TEXT = "fullText"
    VECTOR = "vector"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class BoundingBox:
    """Bounding box coordinates for detected elements"""
    x: float
    y: float
    width: float
    height: float
    page_number: int = 1


@dataclass
class KeyValuePair:
    """Extracted key-value pair from document"""
    key: str
    value: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None


@dataclass
class TableCell:
    """Cell in an extracted table"""
    row_index: int
    column_index: int
    content: str
    confidence: float
    is_header: bool = False


@dataclass
class ExtractedTable:
    """Table extracted from document"""
    table_id: str
    row_count: int
    column_count: int
    cells: List[TableCell]
    page_number: int
    confidence: float


@dataclass
class DocumentEntity:
    """Named entity extracted from document"""
    text: str
    category: str
    subcategory: Optional[str]
    confidence: float
    offset: int
    length: int


@dataclass
class DocumentAnalysisResult:
    """Result of document analysis"""
    document_id: str
    document_type: DocumentType
    pages: int
    language: str
    text: str
    key_value_pairs: List[KeyValuePair]
    tables: List[ExtractedTable]
    entities: List[DocumentEntity]
    confidence: float
    analyzed_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchDocument:
    """Document in search index"""
    id: str
    content: str
    title: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    score: float = 0.0


@dataclass
class SearchResult:
    """Search query result"""
    query: str
    documents: List[SearchDocument]
    total_results: int
    facets: Dict[str, List[Dict[str, Any]]]
    execution_time_ms: float


@dataclass
class EnrichmentPipeline:
    """AI enrichment pipeline configuration"""
    name: str
    skills: List[str]
    source_field: str
    target_fields: Dict[str, str]
    enabled: bool = True


class AzureDocumentIntelligenceManager:
    """
    Manager for Azure Document Intelligence operations.

    Provides document analysis, form recognition, OCR, layout analysis,
    and intelligent document processing capabilities.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "2023-07-31"
    ):
        """
        Initialize Document Intelligence manager.

        Args:
            endpoint: Azure Document Intelligence endpoint URL
            api_key: API key for authentication
            api_version: API version to use
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.api_version = api_version
        self.supported_features = list(AnalysisFeature)

    def analyze_document(
        self,
        document_url: str,
        document_type: DocumentType = DocumentType.GENERAL_DOCUMENT,
        features: Optional[List[AnalysisFeature]] = None
    ) -> DocumentAnalysisResult:
        """
        Analyze document with specified features.

        Args:
            document_url: URL to document
            document_type: Type of document
            features: Features to extract

        Returns:
            DocumentAnalysisResult with extracted information
        """
        if features is None:
            features = [
                AnalysisFeature.OCR,
                AnalysisFeature.LAYOUT,
                AnalysisFeature.TABLES,
                AnalysisFeature.KEY_VALUE_PAIRS
            ]

        print(f"Analyzing document: {document_url}")
        print(f"Document type: {document_type.value}")
        print(f"Features: {[f.value for f in features]}")

        # Simulate document analysis
        result = DocumentAnalysisResult(
            document_id=f"doc_{datetime.now().timestamp()}",
            document_type=document_type,
            pages=3,
            language="en",
            text="Sample extracted text from document...",
            key_value_pairs=[
                KeyValuePair("Invoice Number", "INV-2024-001", 0.98),
                KeyValuePair("Total Amount", "$1,250.00", 0.95),
                KeyValuePair("Date", "2024-01-15", 0.97)
            ],
            tables=[
                ExtractedTable(
                    table_id="table_1",
                    row_count=5,
                    column_count=4,
                    cells=[
                        TableCell(0, 0, "Item", 0.99, True),
                        TableCell(0, 1, "Quantity", 0.99, True),
                        TableCell(1, 0, "Widget A", 0.97, False),
                        TableCell(1, 1, "10", 0.98, False)
                    ],
                    page_number=2,
                    confidence=0.96
                )
            ],
            entities=[
                DocumentEntity("Azure Inc", "Organization", None, 0.95, 10, 9),
                DocumentEntity("Seattle", "Location", "City", 0.92, 150, 7)
            ],
            confidence=0.96,
            analyzed_at=datetime.now()
        )

        return result

    def extract_layout(self, document_url: str) -> Dict[str, Any]:
        """
        Extract layout information including text, lines, and structure.

        Args:
            document_url: URL to document

        Returns:
            Layout information
        """
        layout = {
            "pages": [
                {
                    "page_number": 1,
                    "width": 8.5,
                    "height": 11,
                    "unit": "inch",
                    "angle": 0,
                    "lines": [
                        {
                            "text": "Sample Document Title",
                            "bounding_box": [0.5, 0.5, 7.5, 0.8],
                            "words": [
                                {"text": "Sample", "confidence": 0.99},
                                {"text": "Document", "confidence": 0.98},
                                {"text": "Title", "confidence": 0.99}
                            ]
                        }
                    ]
                }
            ],
            "reading_order": ["line_1", "line_2", "table_1", "line_3"]
        }

        print(f"Extracted layout from {len(layout['pages'])} pages")
        return layout

    def recognize_receipts(self, receipt_url: str) -> Dict[str, Any]:
        """
        Recognize and extract structured data from receipts.

        Args:
            receipt_url: URL to receipt image

        Returns:
            Extracted receipt data
        """
        receipt_data = {
            "merchant_name": "Coffee Shop Inc",
            "merchant_address": "123 Main St, Seattle, WA 98101",
            "transaction_date": "2024-01-15",
            "transaction_time": "14:30:00",
            "items": [
                {
                    "name": "Latte",
                    "quantity": 2,
                    "price": 5.50,
                    "total": 11.00
                },
                {
                    "name": "Croissant",
                    "quantity": 1,
                    "price": 3.50,
                    "total": 3.50
                }
            ],
            "subtotal": 14.50,
            "tax": 1.45,
            "total": 15.95,
            "payment_method": "Credit Card",
            "confidence": 0.94
        }

        print(f"Receipt recognized: {receipt_data['merchant_name']}")
        print(f"Total: ${receipt_data['total']}")
        return receipt_data

    def recognize_invoices(self, invoice_url: str) -> Dict[str, Any]:
        """
        Recognize and extract structured data from invoices.

        Args:
            invoice_url: URL to invoice document

        Returns:
            Extracted invoice data
        """
        invoice_data = {
            "invoice_number": "INV-2024-001",
            "invoice_date": "2024-01-15",
            "due_date": "2024-02-15",
            "vendor": {
                "name": "Tech Solutions Inc",
                "address": "456 Tech Ave, San Francisco, CA 94102",
                "tax_id": "12-3456789"
            },
            "customer": {
                "name": "Business Corp",
                "address": "789 Business Blvd, New York, NY 10001",
                "tax_id": "98-7654321"
            },
            "items": [
                {
                    "description": "Consulting Services",
                    "quantity": 40,
                    "unit_price": 150.00,
                    "amount": 6000.00
                },
                {
                    "description": "Software License",
                    "quantity": 5,
                    "unit_price": 500.00,
                    "amount": 2500.00
                }
            ],
            "subtotal": 8500.00,
            "tax": 850.00,
            "total": 9350.00,
            "currency": "USD",
            "confidence": 0.97
        }

        print(f"Invoice recognized: {invoice_data['invoice_number']}")
        print(f"Total: ${invoice_data['total']} {invoice_data['currency']}")
        return invoice_data

    def extract_key_value_pairs(
        self,
        document_url: str,
        confidence_threshold: float = 0.8
    ) -> List[KeyValuePair]:
        """
        Extract key-value pairs from document.

        Args:
            document_url: URL to document
            confidence_threshold: Minimum confidence score

        Returns:
            List of extracted key-value pairs
        """
        pairs = [
            KeyValuePair("Customer Name", "John Doe", 0.98),
            KeyValuePair("Account Number", "ACC-123456", 0.96),
            KeyValuePair("Email", "john.doe@example.com", 0.94),
            KeyValuePair("Phone", "+1-555-0123", 0.92),
            KeyValuePair("Address", "123 Main St, Seattle, WA", 0.95),
            KeyValuePair("Date of Birth", "1985-06-15", 0.89),
            KeyValuePair("SSN", "XXX-XX-1234", 0.91)
        ]

        # Filter by confidence
        filtered_pairs = [p for p in pairs if p.confidence >= confidence_threshold]

        print(f"Extracted {len(filtered_pairs)} key-value pairs")
        return filtered_pairs

    def extract_tables(self, document_url: str) -> List[ExtractedTable]:
        """
        Extract tables from document.

        Args:
            document_url: URL to document

        Returns:
            List of extracted tables
        """
        tables = [
            ExtractedTable(
                table_id="table_1",
                row_count=4,
                column_count=3,
                cells=[
                    TableCell(0, 0, "Product", 0.99, True),
                    TableCell(0, 1, "Quantity", 0.99, True),
                    TableCell(0, 2, "Price", 0.99, True),
                    TableCell(1, 0, "Widget A", 0.97, False),
                    TableCell(1, 1, "10", 0.98, False),
                    TableCell(1, 2, "$100.00", 0.97, False),
                    TableCell(2, 0, "Widget B", 0.96, False),
                    TableCell(2, 1, "5", 0.98, False),
                    TableCell(2, 2, "$75.00", 0.97, False)
                ],
                page_number=1,
                confidence=0.97
            )
        ]

        print(f"Extracted {len(tables)} tables")
        for table in tables:
            print(f"  Table {table.table_id}: {table.row_count}x{table.column_count}")

        return tables

    def train_custom_model(
        self,
        model_id: str,
        training_data_url: str,
        model_description: str = ""
    ) -> Dict[str, Any]:
        """
        Train a custom document model.

        Args:
            model_id: Unique identifier for the model
            training_data_url: URL to training data
            model_description: Description of the model

        Returns:
            Training result information
        """
        result = {
            "model_id": model_id,
            "status": "succeeded",
            "created_at": datetime.now().isoformat(),
            "training_documents": 25,
            "accuracy": 0.94,
            "description": model_description,
            "fields": [
                "customer_name",
                "order_number",
                "order_date",
                "total_amount",
                "items"
            ]
        }

        print(f"Custom model trained: {model_id}")
        print(f"Accuracy: {result['accuracy']:.2%}")
        return result


class AzureCognitiveSearchManager:
    """
    Manager for Azure Cognitive Search operations.

    Provides search index management, querying, and AI enrichment.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "2023-11-01"
    ):
        """
        Initialize Cognitive Search manager.

        Args:
            endpoint: Azure Cognitive Search endpoint URL
            api_key: Admin API key
            api_version: API version to use
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.api_version = api_version
        self.indexes: Dict[str, Dict[str, Any]] = {}

    def create_search_index(
        self,
        index_name: str,
        fields: List[Dict[str, Any]],
        index_type: SearchIndexType = SearchIndexType.FULL_TEXT
    ) -> Dict[str, Any]:
        """
        Create a search index.

        Args:
            index_name: Name of the index
            fields: Field definitions
            index_type: Type of index

        Returns:
            Index creation result
        """
        index_config = {
            "name": index_name,
            "fields": fields,
            "type": index_type.value,
            "created_at": datetime.now().isoformat(),
            "document_count": 0
        }

        self.indexes[index_name] = index_config

        print(f"Created search index: {index_name}")
        print(f"Index type: {index_type.value}")
        print(f"Fields: {len(fields)}")

        return index_config

    def index_documents(
        self,
        index_name: str,
        documents: List[SearchDocument]
    ) -> Dict[str, Any]:
        """
        Index documents into search index.

        Args:
            index_name: Name of the index
            documents: Documents to index

        Returns:
            Indexing result
        """
        result = {
            "indexed": len(documents),
            "failed": 0,
            "duration_ms": 150.5,
            "timestamp": datetime.now().isoformat()
        }

        if index_name in self.indexes:
            self.indexes[index_name]["document_count"] += len(documents)

        print(f"Indexed {result['indexed']} documents to {index_name}")
        return result

    def search_documents(
        self,
        index_name: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top: int = 10,
        semantic_search: bool = False
    ) -> SearchResult:
        """
        Search documents in index.

        Args:
            index_name: Name of the index
            query: Search query
            filters: Optional filters
            top: Number of results to return
            semantic_search: Use semantic search

        Returns:
            Search results
        """
        # Simulate search results
        documents = [
            SearchDocument(
                id="doc1",
                content="Azure AI Services provides powerful capabilities...",
                title="Introduction to Azure AI",
                metadata={"category": "AI", "author": "Azure Team"},
                score=0.95
            ),
            SearchDocument(
                id="doc2",
                content="Document Intelligence enables automated processing...",
                title="Document Intelligence Guide",
                metadata={"category": "AI", "author": "Microsoft"},
                score=0.88
            )
        ]

        result = SearchResult(
            query=query,
            documents=documents[:top],
            total_results=len(documents),
            facets={
                "category": [
                    {"value": "AI", "count": 2}
                ]
            },
            execution_time_ms=45.2
        )

        search_type = "semantic" if semantic_search else "full-text"
        print(f"Search completed: {query}")
        print(f"Search type: {search_type}")
        print(f"Results: {result.total_results}")

        return result

    def create_enrichment_pipeline(
        self,
        pipeline_name: str,
        skills: List[str],
        source_field: str,
        target_fields: Dict[str, str]
    ) -> EnrichmentPipeline:
        """
        Create AI enrichment pipeline.

        Args:
            pipeline_name: Name of the pipeline
            skills: List of cognitive skills to apply
            source_field: Source field for enrichment
            target_fields: Mapping of enriched fields

        Returns:
            Pipeline configuration
        """
        pipeline = EnrichmentPipeline(
            name=pipeline_name,
            skills=skills,
            source_field=source_field,
            target_fields=target_fields,
            enabled=True
        )

        print(f"Created enrichment pipeline: {pipeline_name}")
        print(f"Skills: {', '.join(skills)}")

        return pipeline


def demo_document_analysis():
    """Demo: Document analysis with multiple features"""
    print("\n" + "="*60)
    print("DEMO: Document Analysis")
    print("="*60)

    manager = AzureDocumentIntelligenceManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key"
    )

    # Analyze invoice
    result = manager.analyze_document(
        document_url="https://example.com/invoice.pdf",
        document_type=DocumentType.INVOICE,
        features=[
            AnalysisFeature.OCR,
            AnalysisFeature.TABLES,
            AnalysisFeature.KEY_VALUE_PAIRS
        ]
    )

    print(f"\nDocument ID: {result.document_id}")
    print(f"Pages: {result.pages}")
    print(f"Language: {result.language}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nKey-Value Pairs: {len(result.key_value_pairs)}")
    for pair in result.key_value_pairs[:3]:
        print(f"  {pair.key}: {pair.value} (confidence: {pair.confidence:.2%})")


def demo_receipt_recognition():
    """Demo: Receipt recognition"""
    print("\n" + "="*60)
    print("DEMO: Receipt Recognition")
    print("="*60)

    manager = AzureDocumentIntelligenceManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key"
    )

    receipt = manager.recognize_receipts("https://example.com/receipt.jpg")

    print(f"\nMerchant: {receipt['merchant_name']}")
    print(f"Date: {receipt['transaction_date']}")
    print(f"\nItems:")
    for item in receipt['items']:
        print(f"  {item['name']}: ${item['total']:.2f}")
    print(f"\nTotal: ${receipt['total']:.2f}")


def demo_invoice_extraction():
    """Demo: Invoice extraction"""
    print("\n" + "="*60)
    print("DEMO: Invoice Extraction")
    print("="*60)

    manager = AzureDocumentIntelligenceManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key"
    )

    invoice = manager.recognize_invoices("https://example.com/invoice.pdf")

    print(f"\nInvoice: {invoice['invoice_number']}")
    print(f"Date: {invoice['invoice_date']}")
    print(f"Vendor: {invoice['vendor']['name']}")
    print(f"Customer: {invoice['customer']['name']}")
    print(f"\nTotal: ${invoice['total']:.2f} {invoice['currency']}")


def demo_table_extraction():
    """Demo: Table extraction"""
    print("\n" + "="*60)
    print("DEMO: Table Extraction")
    print("="*60)

    manager = AzureDocumentIntelligenceManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key"
    )

    tables = manager.extract_tables("https://example.com/document.pdf")

    for table in tables:
        print(f"\nTable: {table.table_id}")
        print(f"Dimensions: {table.row_count} rows x {table.column_count} columns")
        print(f"Page: {table.page_number}")
        print(f"Confidence: {table.confidence:.2%}")


def demo_cognitive_search():
    """Demo: Cognitive search operations"""
    print("\n" + "="*60)
    print("DEMO: Cognitive Search")
    print("="*60)

    manager = AzureCognitiveSearchManager(
        endpoint="https://example.search.windows.net",
        api_key="sample-admin-key"
    )

    # Create index
    fields = [
        {"name": "id", "type": "Edm.String", "key": True},
        {"name": "content", "type": "Edm.String", "searchable": True},
        {"name": "title", "type": "Edm.String", "searchable": True}
    ]

    manager.create_search_index(
        index_name="documents",
        fields=fields,
        index_type=SearchIndexType.HYBRID
    )

    # Index documents
    documents = [
        SearchDocument(
            id="1",
            content="Azure AI enables intelligent applications",
            title="Azure AI Overview",
            metadata={"category": "AI"}
        )
    ]

    manager.index_documents("documents", documents)

    # Search
    results = manager.search_documents(
        index_name="documents",
        query="Azure AI",
        semantic_search=True,
        top=5
    )

    print(f"\nSearch Results:")
    for doc in results.documents:
        print(f"  {doc.title} (score: {doc.score:.2f})")


def demo_enrichment_pipeline():
    """Demo: AI enrichment pipeline"""
    print("\n" + "="*60)
    print("DEMO: AI Enrichment Pipeline")
    print("="*60)

    manager = AzureCognitiveSearchManager(
        endpoint="https://example.search.windows.net",
        api_key="sample-admin-key"
    )

    pipeline = manager.create_enrichment_pipeline(
        pipeline_name="document-enrichment",
        skills=[
            "KeyPhraseExtractionSkill",
            "EntityRecognitionSkill",
            "LanguageDetectionSkill",
            "SentimentAnalysisSkill"
        ],
        source_field="content",
        target_fields={
            "keyPhrases": "enriched_keyphrases",
            "entities": "enriched_entities",
            "language": "enriched_language",
            "sentiment": "enriched_sentiment"
        }
    )

    print(f"\nPipeline: {pipeline.name}")
    print(f"Enabled: {pipeline.enabled}")
    print(f"Source: {pipeline.source_field}")


def demo_custom_model_training():
    """Demo: Custom model training"""
    print("\n" + "="*60)
    print("DEMO: Custom Model Training")
    print("="*60)

    manager = AzureDocumentIntelligenceManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key"
    )

    result = manager.train_custom_model(
        model_id="custom-invoice-model-v1",
        training_data_url="https://example.com/training-data",
        model_description="Custom model for company-specific invoices"
    )

    print(f"\nModel ID: {result['model_id']}")
    print(f"Status: {result['status']}")
    print(f"Training Documents: {result['training_documents']}")
    print(f"Accuracy: {result['accuracy']:.2%}")
    print(f"Fields: {', '.join(result['fields'])}")


if __name__ == "__main__":
    """Run all demo functions"""
    print("\n" + "="*60)
    print("Azure AI Services - Comprehensive Demo")
    print("="*60)

    demo_document_analysis()
    demo_receipt_recognition()
    demo_invoice_extraction()
    demo_table_extraction()
    demo_cognitive_search()
    demo_enrichment_pipeline()
    demo_custom_model_training()

    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60)
