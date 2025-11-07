# Advanced Named Entity Recognition (NER) System v2.0

**Author:** BrillConsulting | **Version:** 2.0 - spaCy + Custom Models
**Entities:** 18 types | **Languages:** 50+

## Overview

Extract and classify named entities from text using state-of-the-art spaCy models. Recognizes persons, organizations, locations, dates, monetary values, and 13+ other entity types with high accuracy.

## Features

- **18 Entity Types:** PERSON, ORG, GPE, LOC, DATE, TIME, MONEY, PERCENT, and more
- **Multi-language:** English, Spanish, French, German, Chinese, etc.
- **Visualization:** Interactive entity highlighting
- **Statistics:** Entity frequency analysis
- **Custom Models:** Train on domain-specific data
- **Batch Processing:** Process thousands of documents

## Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| PERSON | People names | "Steve Jobs", "Marie Curie" |
| ORG | Organizations | "Apple Inc.", "United Nations" |
| GPE | Geopolitical entities | "United States", "Paris" |
| LOC | Non-GPE locations | "Mount Everest", "Pacific Ocean" |
| DATE | Dates | "January 1, 2020", "last week" |
| TIME | Times | "3:00 PM", "morning" |
| MONEY | Monetary values | "$100", "€50" |
| PERCENT | Percentages | "25%", "half" |
| PRODUCT | Products | "iPhone", "Windows 10" |
| EVENT | Named events | "World War II", "Olympics" |
| LANGUAGE | Languages | "English", "Spanish" |
| LAW | Laws | "Bill of Rights" |
| WORK_OF_ART | Titles | "Mona Lisa", "Harry Potter" |
| FAC | Facilities | "Golden Gate Bridge" |
| NORP | Nationalities | "American", "Buddhist" |
| QUANTITY | Measurements | "10 miles", "3 kg" |
| ORDINAL | Ordinal numbers | "first", "3rd" |
| CARDINAL | Cardinal numbers | "two", "100" |

## Installation

```bash
pip install spacy matplotlib
python -m spacy download en_core_web_sm  # Small model
python -m spacy download en_core_web_lg  # Large model (better accuracy)
```

## Quick Start

```python
from ner_system import NERSystem

# Initialize
ner = NERSystem(model_name='en_core_web_sm')

# Extract entities
text = "Apple Inc. was founded by Steve Jobs in Cupertino on April 1, 1976."
entities = ner.extract_entities(text)

for ent in entities:
    print(f"{ent['text']}: {ent['label']}")
```

**Output:**
```
Apple Inc.: ORG
Steve Jobs: PERSON
Cupertino: GPE
April 1, 1976: DATE
```

## Usage Examples

### 1. Document Analysis

```python
ner = NERSystem()

document = """
Microsoft was founded by Bill Gates and Paul Allen in 1975.
The company is headquartered in Redmond, Washington.
"""

analysis = ner.analyze_document(document)

print(f"Total entities: {analysis['total_entities']}")
print(f"Entity types: {analysis['entity_counts']}")

# Output:
# Total entities: 5
# Entity types: {'ORG': 1, 'PERSON': 2, 'DATE': 1, 'GPE': 1}
```

### 2. Resume Parsing

```python
resume = load_resume()
entities = ner.extract_entities(resume)

# Extract relevant information
person_names = [e['text'] for e in entities if e['label'] == 'PERSON']
organizations = [e['text'] for e in entities if e['label'] == 'ORG']
locations = [e['text'] for e in entities if e['label'] == 'GPE']
dates = [e['text'] for e in entities if e['label'] == 'DATE']

candidate_info = {
    'name': person_names[0] if person_names else None,
    'companies': organizations,
    'locations': locations,
    'timeline': dates
}
```

### 3. News Article Processing

```python
articles = load_news_articles()

for article in articles:
    analysis = ner.analyze_document(article['text'])

    # Tag article
    article['people'] = [e for e in analysis['entities_by_type'].get('PERSON', [])]
    article['organizations'] = [e for e in analysis['entities_by_type'].get('ORG', [])]
    article['locations'] = [e for e in analysis['entities_by_type'].get('GPE', [])]
```

### 4. Contract Analysis

```python
contract = load_contract()
entities = ner.extract_entities(contract)

# Extract key information
parties = [e['text'] for e in entities if e['label'] in ['PERSON', 'ORG']]
dates = [e['text'] for e in entities if e['label'] == 'DATE']
amounts = [e['text'] for e in entities if e['label'] == 'MONEY']

print(f"Parties: {parties}")
print(f"Important dates: {dates}")
print(f"Monetary amounts: {amounts}")
```

### 5. Social Media Monitoring

```python
tweets = get_tweets(hashtag='#tech')

all_mentions = {}
for tweet in tweets:
    entities = ner.extract_entities(tweet['text'])

    for ent in entities:
        if ent['label'] in ['PERSON', 'ORG']:
            name = ent['text']
            all_mentions[name] = all_mentions.get(name, 0) + 1

# Top mentioned entities
top_entities = sorted(all_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
```

## Command Line

```bash
# Basic extraction
python ner_system.py --text "Apple Inc. was founded by Steve Jobs"

# From file with visualization
python ner_system.py --file article.txt --output entities.png

# With specific model
python ner_system.py \
    --file document.txt \
    --model en_core_web_lg
```

## Visualization

### HTML Highlighting
```python
html = ner.visualize_entities(text, save_path='entities.html')
# Opens in browser with color-coded entities
```

### Distribution Plot
```python
ner.plot_entity_distribution(text, save_path='distribution.png')
# Bar chart of entity type frequencies
```

## Model Comparison

| Model | Size | Speed | Accuracy | Memory |
|-------|------|-------|----------|--------|
| en_core_web_sm | 13MB | Fast | 85% | 50MB |
| en_core_web_md | 43MB | Medium | 91% | 200MB |
| en_core_web_lg | 741MB | Slow | 94% | 1GB |
| en_core_web_trf | 438MB | Very Slow | 96% | 2GB |

*trf = Transformer-based (highest accuracy)*

## Multi-language Support

```python
# Spanish
ner_es = NERSystem('es_core_news_sm')
entities_es = ner_es.extract_entities("Microsoft fue fundada por Bill Gates")

# German
ner_de = NERSystem('de_core_news_sm')
entities_de = ner_de.extract_entities("Microsoft wurde von Bill Gates gegründet")

# French
ner_fr = NERSystem('fr_core_news_sm')
entities_fr = ner_fr.extract_entities("Microsoft a été fondée par Bill Gates")
```

## Batch Processing

```python
documents = load_documents()  # List of 10,000 documents

all_entities = []
for idx, doc in enumerate(documents):
    entities = ner.extract_entities(doc)
    all_entities.extend(entities)

    if (idx + 1) % 1000 == 0:
        print(f"Processed {idx + 1} documents")

# Analyze all entities
entity_counts = {}
for ent in all_entities:
    key = (ent['text'], ent['label'])
    entity_counts[key] = entity_counts.get(key, 0) + 1

# Most common entities
top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20]
```

## Custom Entity Rules

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load('en_core_web_sm')

# Add custom entity patterns
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "PRODUCT", "pattern": "iPhone"},
    {"label": "PRODUCT", "pattern": "MacBook Pro"},
    {"label": "ORG", "pattern": [{"LOWER": "openai"}]}
]
ruler.add_patterns(patterns)

# Use custom model
doc = nlp("I bought an iPhone from OpenAI")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

## Best Practices

### 1. Choose Right Model
```python
# For speed: sm model
ner = NERSystem('en_core_web_sm')

# For accuracy: lg or trf model
ner = NERSystem('en_core_web_lg')
```

### 2. Filter by Confidence
```python
entities = ner.extract_entities(text)
high_confidence = [e for e in entities if e.get('confidence', 1.0) > 0.8]
```

### 3. Deduplicate Entities
```python
unique_entities = {}
for ent in entities:
    key = (ent['text'].lower(), ent['label'])
    if key not in unique_entities:
        unique_entities[key] = ent

entities = list(unique_entities.values())
```

### 4. Context-aware Extraction
```python
# Keep track of entity positions for context
for ent in entities:
    start = ent['start']
    end = ent['end']
    context_before = text[max(0, start-50):start]
    context_after = text[end:min(len(text), end+50)]
```

## Performance Optimization

### For Large Documents
```python
# Process in chunks
chunk_size = 10000
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

all_entities = []
for chunk in chunks:
    entities = ner.extract_entities(chunk)
    all_entities.extend(entities)
```

### For Real-time Processing
```python
# Use small model + caching
ner = NERSystem('en_core_web_sm')

entity_cache = {}
def get_entities_cached(text):
    if text not in entity_cache:
        entity_cache[text] = ner.extract_entities(text)
    return entity_cache[text]
```

## Troubleshooting

**Problem:** Missing entities
**Solution:** Use larger model (lg or trf), add custom patterns

**Problem:** Wrong entity types
**Solution:** Train custom model on domain data, adjust confidence threshold

**Problem:** Slow processing
**Solution:** Use sm model, process in batches, enable GPU

**Problem:** Memory issues
**Solution:** Use sm model, process documents one at a time, clear cache

## API Reference

```python
class NERSystem:
    def __init__(self, model_name='en_core_web_sm')

    def extract_entities(self, text: str) -> List[Dict]

    def analyze_document(self, text: str) -> Dict

    def visualize_entities(self, text: str, save_path=None) -> str

    def plot_entity_distribution(self, text: str, save_path=None)
```

## Use Cases Summary

1. **Information Extraction:** Extract structured data from unstructured text
2. **Resume Parsing:** Identify candidates, skills, experience
3. **News Analysis:** Track mentions of people, companies, locations
4. **Contract Analysis:** Extract parties, dates, amounts
5. **Social Media Monitoring:** Track entity mentions and trends
6. **Document Classification:** Use entities as features
7. **Knowledge Graph Construction:** Build entity relationships
8. **Search Enhancement:** Improve search with entity tagging

## Version History

- **v2.0:** spaCy 3.x, 18 entity types, visualization, multi-language
- **v1.0:** Basic NER with spaCy 2.x

---

**BrillConsulting** - Advanced NLP Solutions
