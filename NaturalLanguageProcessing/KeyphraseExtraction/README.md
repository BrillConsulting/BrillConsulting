# Advanced Keyphrase Extraction System v2.0

**Author:** BrillConsulting | **Version:** 2.0 - RAKE, YAKE, KeyBERT
**Methods:** RAKE (statistical), YAKE (unsupervised), KeyBERT (BERT-based)

## Overview

Extract important keyphrases and keywords from documents using multiple state-of-the-art methods. No training required - works out of the box on any text.

## Features

- **Multiple Methods:** RAKE, YAKE, KeyBERT
- **No Training Needed:** Unsupervised extraction
- **Multi-word Phrases:** Extract n-grams (unigrams to 5-grams)
- **Scoring:** Relevance scores for each keyphrase
- **Domain Agnostic:** Works on any domain
- **Fast Processing:** Extract from 1000s of documents

## Installation

```bash
# RAKE
pip install rake-nltk

# YAKE
pip install yake

# KeyBERT (best quality)
pip install keybert sentence-transformers
```

## Methods Comparison

| Method | Speed | Quality | Training | Best For |
|--------|-------|---------|----------|----------|
| RAKE | ⚡⚡⚡ | ⭐⭐⭐ | None | Quick extraction |
| YAKE | ⚡⚡ | ⭐⭐⭐⭐ | None | Balanced |
| KeyBERT | ⚡ | ⭐⭐⭐⭐⭐ | None | Highest quality |

## Quick Start

```python
from keyphrase_extraction import KeyphraseExtractor

# Using KeyBERT (best quality)
extractor = KeyphraseExtractor(method='keybert')

keyphrases = extractor.extract(document, top_n=10)

for kp in keyphrases:
    print(f"{kp['keyphrase']}: {kp['score']:.3f}")
```

## Usage Examples

### 1. Extract from News Article

```python
extractor = KeyphraseExtractor(method='keybert')

article = """
Artificial intelligence is transforming healthcare...
"""

keyphrases = extractor.extract(article, top_n=10)

# Output:
# artificial intelligence: 0.758
# healthcare transformation: 0.692
# machine learning: 0.654
```

### 2. Compare Methods

```python
methods = ['rake', 'yake', 'keybert']
results = {}

for method in methods:
    extractor = KeyphraseExtractor(method=method)
    results[method] = extractor.extract(text, top_n=5)

# Compare outputs
for method, keyphrases in results.items():
    print(f"\n{method.upper()}:")
    for kp in keyphrases:
        print(f"  - {kp['keyphrase']}")
```

### 3. Batch Processing

```python
documents = load_documents()  # List of texts
extractor = KeyphraseExtractor(method='yake')

all_keyphrases = []
for doc in documents:
    keyphrases = extractor.extract(doc, top_n=5)
    all_keyphrases.append(keyphrases)
```

### 4. Extract with Custom N-grams

```python
# KeyBERT with specific ngram range
extractor = KeyphraseExtractor(method='keybert')

keyphrases = extractor.extract(
    text,
    top_n=15,
    keyphrase_ngram_range=(1, 3),  # unigrams to trigrams
    stop_words='english'
)
```

## Command Line

```bash
# Basic extraction
python keyphrase_extraction.py \
    --file document.txt \
    --method keybert \
    --top-n 10

# From text
python keyphrase_extraction.py \
    --text "Your text here..." \
    --method rake

# Compare methods
for method in rake yake keybert; do
    echo "=== $method ==="
    python keyphrase_extraction.py --file doc.txt --method $method --top-n 5
done
```

## Use Cases

### 1. SEO Optimization
```python
# Extract keywords for meta tags
extractor = KeyphraseExtractor('keybert')
keyphrases = extractor.extract(webpage_content, top_n=10)
meta_keywords = ', '.join([kp['keyphrase'] for kp in keyphrases[:5]])
```

### 2. Document Indexing
```python
# Index documents by keyphrases
for doc in document_collection:
    keyphrases = extractor.extract(doc['content'], top_n=10)
    doc['keywords'] = [kp['keyphrase'] for kp in keyphrases]
    index_document(doc)
```

### 3. Content Summarization
```python
# Get main topics
keyphrases = extractor.extract(long_article, top_n=20)
main_topics = [kp['keyphrase'] for kp in keyphrases if kp['score'] > 0.6]
```

### 4. Research Paper Analysis
```python
# Extract key concepts from papers
papers = load_research_papers()
for paper in papers:
    abstract_kp = extractor.extract(paper['abstract'], top_n=10)
    paper['key_concepts'] = [kp['keyphrase'] for kp in abstract_kp]
```

### 5. Tag Generation
```python
# Auto-generate tags for blog posts
blog_posts = load_blog_posts()
for post in blog_posts:
    keyphrases = extractor.extract(post['content'], top_n=8)
    post['tags'] = [kp['keyphrase'] for kp in keyphrases[:5]]
```

## Method Details

### RAKE (Rapid Automatic Keyword Extraction)
- **Algorithm:** Statistical, uses word co-occurrence
- **Pros:** Very fast, language independent
- **Cons:** May extract less relevant phrases
- **Best for:** Large-scale processing

### YAKE (Yet Another Keyword Extractor)
- **Algorithm:** Unsupervised, statistical features
- **Pros:** Good balance of speed/quality, no training
- **Cons:** May need parameter tuning
- **Best for:** General purpose extraction

### KeyBERT
- **Algorithm:** BERT embeddings + cosine similarity
- **Pros:** Highest quality, semantic understanding
- **Cons:** Slower, needs more memory
- **Best for:** When quality matters most

## Best Practices

### 1. Choose Right Method
```python
# For speed: RAKE
extractor = KeyphraseExtractor('rake')

# For balance: YAKE
extractor = KeyphraseExtractor('yake')

# For quality: KeyBERT
extractor = KeyphraseExtractor('keybert')
```

### 2. Adjust Top-N Based on Document Length
```python
doc_length = len(text.split())

if doc_length < 100:
    top_n = 5
elif doc_length < 500:
    top_n = 10
else:
    top_n = 20
    
keyphrases = extractor.extract(text, top_n=top_n)
```

### 3. Filter by Score
```python
keyphrases = extractor.extract(text, top_n=20)
high_quality = [kp for kp in keyphrases if kp['score'] > 0.5]
```

### 4. Combine Methods (Ensemble)
```python
def ensemble_extract(text, top_n=10):
    methods = ['rake', 'yake', 'keybert']
    all_keyphrases = {}
    
    for method in methods:
        extractor = KeyphraseExtractor(method)
        kps = extractor.extract(text, top_n=top_n)
        for kp in kps:
            phrase = kp['keyphrase']
            if phrase in all_keyphrases:
                all_keyphrases[phrase] += kp['score']
            else:
                all_keyphrases[phrase] = kp['score']
    
    # Sort by combined score
    return sorted(all_keyphrases.items(), key=lambda x: x[1], reverse=True)[:top_n]
```

## Performance Optimization

### Large Documents
```python
# Process in chunks
chunk_size = 1000
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

all_keyphrases = []
for chunk in chunks:
    kps = extractor.extract(chunk, top_n=5)
    all_keyphrases.extend(kps)

# Deduplicate and re-score
```

### Batch Processing
```python
from concurrent.futures import ThreadPoolExecutor

def process_document(doc):
    return extractor.extract(doc, top_n=10)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_document, documents))
```

## Troubleshooting

**Problem:** Too many irrelevant keyphrases
**Solution:** Increase top_n threshold, use KeyBERT, filter by score

**Problem:** Missing important phrases
**Solution:** Increase top_n, adjust ngram_range, try different method

**Problem:** Slow processing
**Solution:** Use RAKE for speed, reduce top_n, batch process

---

**BrillConsulting** - Advanced NLP Solutions
