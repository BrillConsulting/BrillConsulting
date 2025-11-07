# Advanced Named Entity Linking System v2.0

**Author:** BrillConsulting | **Version:** 2.0 - Wikipedia & Wikidata
**Features:** Entity Recognition + Knowledge Base Linking

## Overview

Link named entities to knowledge bases (Wikipedia, Wikidata) for entity disambiguation and knowledge graph construction. Combines spaCy NER with entity linking.

## Features

- **Entity Recognition:** Detect persons, organizations, locations, etc.
- **KB Linking:** Link to Wikipedia/Wikidata entries
- **Disambiguation:** Resolve entity ambiguity
- **Multi-language:** Support for 50+ languages
- **Knowledge Graph:** Build entity relationships
- **Custom KBs:** Link to domain-specific knowledge bases

## Installation

```bash
pip install spacy
python -m spacy download en_core_web_lg
```

## Quick Start

```python
from named_entity_linking import EntityLinker

linker = EntityLinker()

entities = linker.link_entities(
    "Apple Inc. was founded by Steve Jobs in Cupertino"
)

for ent in entities:
    print(f"{ent['text']} ({ent['label']})")
    if 'wikipedia_url' in ent:
        print(f"  → {ent['wikipedia_url']}")
```

## Use Cases

### 1. Knowledge Graph Construction

```python
text = """
Elon Musk is the CEO of Tesla and SpaceX. 
He was born in South Africa.
"""

entities = linker.link_entities(text)

# Build graph
for ent in entities:
    if 'kb_id' in ent:
        add_to_knowledge_graph(ent)
```

### 2. Entity Disambiguation

```python
# "Apple" could mean company or fruit
text1 = "Apple reported record profits"
text2 = "I ate an apple for lunch"

e1 = linker.link_entities(text1)
e2 = linker.link_entities(text2)

# e1 links to Apple Inc.
# e2 doesn't link (common noun)
```

### 3. Document Enrichment

```python
articles = load_articles()

for article in articles:
    entities = linker.link_entities(article['text'])
    article['entities'] = entities
    article['topics'] = [e['kb_id'] for e in entities if 'kb_id' in e]
```

### 4. Semantic Search

```python
# Find articles about specific entities
query_entity = "Q95"  # Wikidata ID for "Google"

articles_about_google = [
    article for article in articles
    if any(e.get('kb_id') == query_entity for e in article['entities'])
]
```

## Command Line

```bash
# Basic linking
python named_entity_linking.py \
    --text "Microsoft was founded by Bill Gates"

# From file
python named_entity_linking.py \
    --file article.txt
```

## Entity Types

- **PERSON:** Individual people
- **ORG:** Companies, agencies, institutions
- **GPE:** Countries, cities, states
- **LOC:** Mountains, bodies of water
- **PRODUCT:** Vehicles, foods, items
- **EVENT:** Named events
- **WORK_OF_ART:** Titles of books, songs
- **LAW:** Named laws
- **LANGUAGE:** Named languages

## Best Practices

### 1. Disambiguation
```python
# Use context for disambiguation
context = get_surrounding_text(entity, text)
linked = linker.link_with_context(entity, context)
```

### 2. Confidence Filtering
```python
entities = linker.link_entities(text)
high_confidence = [e for e in entities if e.get('confidence', 0) > 0.7]
```

### 3. Batch Processing
```python
documents = load_documents()
all_entities = []

for doc in documents:
    entities = linker.link_entities(doc)
    all_entities.extend(entities)
```

---

**BrillConsulting** - Advanced NLP Solutions
