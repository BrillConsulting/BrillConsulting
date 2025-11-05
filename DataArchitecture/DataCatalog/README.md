# Data Catalog System

Enterprise-grade data catalog for centralized metadata management and data discovery.

## Overview

A comprehensive data catalog system that enables organizations to discover, understand, and trust their data assets through centralized metadata management, business glossary, and impact analysis.

## Features

### Asset Management
- **Asset Registration**: Register data assets with comprehensive metadata
- **Schema Management**: Track and manage data schemas
- **Ownership Tracking**: Define and track data ownership
- **Classification**: Classify data by sensitivity levels
- **Tagging System**: Organize assets with flexible tagging

### Business Glossary
- **Term Management**: Define business terms and definitions
- **Synonym Mapping**: Map business synonyms to technical assets
- **Lineage Linking**: Link glossary terms to data assets
- **Stewardship**: Track business and technical ownership

### Search & Discovery
- **Intelligent Search**: Search across names, descriptions, and tags
- **Relevance Scoring**: AI-powered relevance ranking
- **Filtering**: Filter by type, owner, classification
- **Asset Relationships**: Discover related assets

### Impact Analysis
- **Dependency Tracking**: Track upstream/downstream dependencies
- **Change Impact**: Analyze impact of schema/data changes
- **Lineage Visualization**: Visualize data lineage
- **Quality Scoring**: Calculate metadata quality scores

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from datacatalog import DataCatalog

# Initialize catalog
catalog = DataCatalog()

# Register data asset
catalog.register_asset("customers_table", {
    "name": "Customers",
    "type": "table",
    "description": "Customer master data",
    "owner": "data_team",
    "schema": {"customer_id": "integer", "name": "string"},
    "tags": ["customer", "master-data"],
    "classification": "confidential"
})

# Search catalog
results = catalog.search("customer")

# Impact analysis
impact = catalog.impact_analysis("customers_table")
```

## Demo

```bash
python datacatalog.py
```

## Key Concepts

- **Asset Types**: Table, View, Dataset, Report, API
- **Classifications**: Public, Internal, Confidential, Restricted
- **Relationships**: Feeds, Derives, References, Transforms

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
