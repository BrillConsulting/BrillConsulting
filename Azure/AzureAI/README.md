# Azure AI Services Integration

Advanced implementation of Azure AI Services with Document Intelligence, Cognitive Search, OCR, and AI enrichment pipelines.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a comprehensive Python implementation for Azure AI Services, featuring Document Intelligence for automated document processing, Cognitive Search for intelligent information retrieval, and AI enrichment pipelines for content analysis. Built for enterprise applications requiring scalable document processing with Azure's security and compliance features.

## Features

### Core Capabilities
- **Document Intelligence**: Automated document analysis and processing
- **Form Recognition**: Extract structured data from forms and receipts
- **Invoice Processing**: Automated invoice data extraction
- **OCR**: Optical character recognition with high accuracy
- **Layout Analysis**: Understand document structure and reading order
- **Table Extraction**: Extract tables with cell-level accuracy
- **Key-Value Pair Extraction**: Identify and extract field-value relationships
- **Custom Model Training**: Train models for domain-specific documents

### Cognitive Search Features
- **Full-Text Search**: Traditional keyword-based search
- **Vector Search**: Semantic search using embeddings
- **Semantic Search**: AI-powered understanding of query intent
- **Hybrid Search**: Combined keyword and semantic search
- **AI Enrichment**: Extract insights using cognitive skills
- **Faceted Navigation**: Organize results by categories
- **Index Management**: Create and manage search indexes

### Advanced Features
- **Multi-Page Analysis**: Process documents with multiple pages
- **Confidence Scoring**: Reliability metrics for extracted data
- **Batch Processing**: Efficient processing of multiple documents
- **Custom Fields**: Define domain-specific extraction fields
- **Entity Recognition**: Identify named entities in documents

## Architecture

```
AzureAI/
├── azureai.py                 # Main implementation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

### Key Components

1. **AzureDocumentIntelligenceManager**: Document processing interface
   - Document analysis with multiple features
   - Receipt and invoice recognition
   - Custom model training
   - Layout and table extraction

2. **AzureCognitiveSearchManager**: Search operations handler
   - Search index creation and management
   - Document indexing
   - Query execution
   - AI enrichment pipelines

3. **Data Classes**:
   - `DocumentAnalysisResult`: Complete analysis results
   - `KeyValuePair`: Extracted field-value pairs
   - `ExtractedTable`: Table structure and data
   - `SearchResult`: Search query results
   - `EnrichmentPipeline`: AI enrichment configuration

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/AzureAI

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your Azure AI Services credentials:

```python
from azureai import AzureDocumentIntelligenceManager, AzureCognitiveSearchManager

# Document Intelligence
doc_manager = AzureDocumentIntelligenceManager(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    api_key="your-api-key",
    api_version="2023-07-31"
)

# Cognitive Search
search_manager = AzureCognitiveSearchManager(
    endpoint="https://your-search.search.windows.net",
    api_key="your-admin-key",
    api_version="2023-11-01"
)
```

### Environment Variables (Recommended)

```bash
export AZURE_DOC_INTELLIGENCE_ENDPOINT="https://your-resource.cognitiveservices.azure.com"
export AZURE_DOC_INTELLIGENCE_KEY="your-api-key"
export AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net"
export AZURE_SEARCH_ADMIN_KEY="your-admin-key"
```

## Usage Examples

### Document Analysis

```python
from azureai import AzureDocumentIntelligenceManager, DocumentType, AnalysisFeature

manager = AzureDocumentIntelligenceManager(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    api_key="your-api-key"
)

# Analyze invoice with multiple features
result = manager.analyze_document(
    document_url="https://example.com/invoice.pdf",
    document_type=DocumentType.INVOICE,
    features=[
        AnalysisFeature.OCR,
        AnalysisFeature.TABLES,
        AnalysisFeature.KEY_VALUE_PAIRS,
        AnalysisFeature.ENTITIES
    ]
)

print(f"Document ID: {result.document_id}")
print(f"Pages: {result.pages}")
print(f"Confidence: {result.confidence:.2%}")

# Access extracted data
for pair in result.key_value_pairs:
    print(f"{pair.key}: {pair.value} (confidence: {pair.confidence:.2%})")
```

### Receipt Recognition

```python
receipt = manager.recognize_receipts("https://example.com/receipt.jpg")

print(f"Merchant: {receipt['merchant_name']}")
print(f"Total: ${receipt['total']:.2f}")

for item in receipt['items']:
    print(f"{item['name']}: ${item['total']:.2f}")
```

### Invoice Extraction

```python
invoice = manager.recognize_invoices("https://example.com/invoice.pdf")

print(f"Invoice: {invoice['invoice_number']}")
print(f"Date: {invoice['invoice_date']}")
print(f"Total: ${invoice['total']:.2f} {invoice['currency']}")

# Access vendor and customer info
print(f"Vendor: {invoice['vendor']['name']}")
print(f"Customer: {invoice['customer']['name']}")
```

### Table Extraction

```python
tables = manager.extract_tables("https://example.com/document.pdf")

for table in tables:
    print(f"Table {table.table_id}: {table.row_count}x{table.column_count}")
    print(f"Confidence: {table.confidence:.2%}")

    # Access individual cells
    for cell in table.cells:
        print(f"Cell [{cell.row_index},{cell.column_index}]: {cell.content}")
```

### Key-Value Pair Extraction

```python
pairs = manager.extract_key_value_pairs(
    document_url="https://example.com/form.pdf",
    confidence_threshold=0.8
)

for pair in pairs:
    print(f"{pair.key}: {pair.value}")
```

### Cognitive Search - Index Creation

```python
from azureai import AzureCognitiveSearchManager, SearchIndexType

search_manager = AzureCognitiveSearchManager(
    endpoint="https://your-search.search.windows.net",
    api_key="your-admin-key"
)

# Define index fields
fields = [
    {"name": "id", "type": "Edm.String", "key": True},
    {"name": "content", "type": "Edm.String", "searchable": True},
    {"name": "title", "type": "Edm.String", "searchable": True},
    {"name": "category", "type": "Edm.String", "filterable": True}
]

# Create hybrid search index
index = search_manager.create_search_index(
    index_name="documents",
    fields=fields,
    index_type=SearchIndexType.HYBRID
)
```

### Document Indexing

```python
from azureai import SearchDocument

documents = [
    SearchDocument(
        id="1",
        content="Azure AI Services provides powerful document processing...",
        title="Azure AI Overview",
        metadata={"category": "AI", "author": "Microsoft"}
    ),
    SearchDocument(
        id="2",
        content="Document Intelligence enables automated analysis...",
        title="Document Intelligence Guide",
        metadata={"category": "AI", "author": "Azure Team"}
    )
]

result = search_manager.index_documents("documents", documents)
print(f"Indexed {result['indexed']} documents")
```

### Search Operations

```python
# Semantic search
results = search_manager.search_documents(
    index_name="documents",
    query="document processing with AI",
    semantic_search=True,
    top=10
)

print(f"Found {results.total_results} results")
for doc in results.documents:
    print(f"{doc.title} (score: {doc.score:.2f})")
    print(f"Content: {doc.content[:100]}...")
```

## AI Enrichment Pipelines

```python
pipeline = search_manager.create_enrichment_pipeline(
    pipeline_name="document-enrichment",
    skills=[
        "KeyPhraseExtractionSkill",
        "EntityRecognitionSkill",
        "LanguageDetectionSkill",
        "SentimentAnalysisSkill",
        "ImageAnalysisSkill",
        "OCRSkill"
    ],
    source_field="content",
    target_fields={
        "keyPhrases": "enriched_keyphrases",
        "entities": "enriched_entities",
        "language": "enriched_language",
        "sentiment": "enriched_sentiment"
    }
)

print(f"Pipeline created: {pipeline.name}")
```

## Custom Model Training

```python
result = manager.train_custom_model(
    model_id="custom-invoice-model-v1",
    training_data_url="https://example.com/training-data",
    model_description="Custom model for company-specific invoices"
)

print(f"Model ID: {result['model_id']}")
print(f"Status: {result['status']}")
print(f"Accuracy: {result['accuracy']:.2%}")
print(f"Fields: {', '.join(result['fields'])}")
```

## Running Demos

```bash
# Run all demo functions
python azureai.py
```

Demo output includes:
- Document analysis with multiple features
- Receipt recognition
- Invoice extraction
- Table extraction
- Cognitive search operations
- AI enrichment pipeline setup
- Custom model training

## API Reference

### AzureDocumentIntelligenceManager

#### Methods

**`analyze_document(document_url, document_type, features)`**
- Analyzes document with specified features
- Returns: `DocumentAnalysisResult`

**`recognize_receipts(receipt_url)`**
- Extracts structured data from receipts
- Returns: `Dict[str, Any]`

**`recognize_invoices(invoice_url)`**
- Extracts structured data from invoices
- Returns: `Dict[str, Any]`

**`extract_tables(document_url)`**
- Extracts tables from document
- Returns: `List[ExtractedTable]`

**`extract_key_value_pairs(document_url, confidence_threshold)`**
- Extracts field-value pairs
- Returns: `List[KeyValuePair]`

**`train_custom_model(model_id, training_data_url, model_description)`**
- Trains custom document model
- Returns: `Dict[str, Any]`

### AzureCognitiveSearchManager

#### Methods

**`create_search_index(index_name, fields, index_type)`**
- Creates a search index
- Returns: `Dict[str, Any]`

**`index_documents(index_name, documents)`**
- Indexes documents into search index
- Returns: `Dict[str, Any]`

**`search_documents(index_name, query, filters, top, semantic_search)`**
- Searches documents in index
- Returns: `SearchResult`

**`create_enrichment_pipeline(pipeline_name, skills, source_field, target_fields)`**
- Creates AI enrichment pipeline
- Returns: `EnrichmentPipeline`

## Best Practices

### 1. Document Processing
Always specify the correct document type for optimal results:
```python
result = manager.analyze_document(
    document_url=url,
    document_type=DocumentType.INVOICE,  # Be specific
    features=[AnalysisFeature.KEY_VALUE_PAIRS]
)
```

### 2. Confidence Thresholds
Filter results by confidence score:
```python
high_confidence_pairs = [
    pair for pair in pairs
    if pair.confidence >= 0.9
]
```

### 3. Batch Processing
Process multiple documents efficiently:
```python
urls = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = [manager.analyze_document(url) for url in urls]
```

### 4. Error Handling
```python
try:
    result = manager.analyze_document(document_url)
except Exception as e:
    print(f"Error processing document: {e}")
    # Implement retry logic or fallback
```

### 5. Search Optimization
Use appropriate search types for your use case:
```python
# Use semantic search for natural language queries
results = search_manager.search_documents(
    query="how to process invoices",
    semantic_search=True  # Better understanding
)

# Use full-text for exact keyword matching
results = search_manager.search_documents(
    query="invoice-2024-001",
    semantic_search=False  # Exact matching
)
```

## Use Cases

### 1. Automated Invoice Processing
```python
# Extract and validate invoice data
invoice = manager.recognize_invoices(invoice_url)
if invoice['total'] > 10000:
    # Trigger approval workflow
    pass
```

### 2. Receipt Digitization
```python
# Convert paper receipts to structured data
receipt = manager.recognize_receipts(receipt_image_url)
# Store in database for expense tracking
```

### 3. Document Search System
```python
# Build intelligent document search
results = search_manager.search_documents(
    index_name="company-docs",
    query=user_query,
    semantic_search=True
)
```

### 4. Compliance Document Analysis
```python
# Extract entities and key information
result = manager.analyze_document(
    document_url=compliance_doc,
    features=[
        AnalysisFeature.ENTITIES,
        AnalysisFeature.KEY_VALUE_PAIRS
    ]
)
```

## Performance Optimization

### 1. Feature Selection
Only request needed features:
```python
# Instead of all features, be selective
result = manager.analyze_document(
    url,
    features=[AnalysisFeature.KEY_VALUE_PAIRS]  # Only what you need
)
```

### 2. Caching
Cache frequently accessed results:
```python
cache = {}
if document_id not in cache:
    cache[document_id] = manager.analyze_document(url)
```

### 3. Parallel Processing
Process multiple documents in parallel:
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(manager.analyze_document, urls))
```

## Security Considerations

1. **API Key Protection**: Never commit API keys to version control
2. **Document Privacy**: Ensure documents don't contain PII before processing
3. **Access Control**: Use Azure RBAC for resource access
4. **Audit Logging**: Log all document processing operations
5. **Data Retention**: Follow compliance requirements for document storage

## Troubleshooting

### Common Issues

**Issue**: Document analysis fails
**Solution**: Verify document URL is accessible and format is supported

**Issue**: Low confidence scores
**Solution**: Use higher quality scans or train custom model

**Issue**: Table extraction incomplete
**Solution**: Ensure table has clear borders and structure

**Issue**: Search returns no results
**Solution**: Verify index exists and documents are indexed

## Deployment

### Azure Deployment

```bash
# Create Document Intelligence resource
az cognitiveservices account create \
    --name doc-intelligence \
    --resource-group rg-ai \
    --kind FormRecognizer \
    --sku S0 \
    --location eastus

# Create Cognitive Search service
az search service create \
    --name search-service \
    --resource-group rg-ai \
    --sku standard \
    --location eastus
```

### Container Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY azureai.py .
CMD ["python", "azureai.py"]
```

## Monitoring

### Key Metrics
- Document processing success rate
- Average processing time
- Confidence score distribution
- Search query latency
- Index size and document count
- API usage and throttling

### Azure Monitor Integration

```python
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=...'
))
```

## Dependencies

```
Python >= 3.8
dataclasses
typing
json
datetime
```

See `requirements.txt` for complete list.

## Version History

- **v1.0.0**: Initial release with basic features
- **v2.0.0**: Added Cognitive Search integration
- **v2.1.0**: Custom model training support
- **v2.2.0**: Enhanced enrichment pipelines

## Contributing

Contributions are welcome! Please submit pull requests or open issues on GitHub.

## License

This project is part of the Brill Consulting portfolio.

## Support

For questions or support:
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure OpenAI](../AzureOpenAI/)
- [Cognitive Services](../CognitiveServices/)
- [Azure Machine Learning](../MachineLearning/)

---

**Built with Azure AI Services** | **Brill Consulting © 2024**
