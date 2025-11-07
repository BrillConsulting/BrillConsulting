# Advanced Machine Translation System v2.0

**Author:** BrillConsulting | **Version:** 2.0 - Neural Machine Translation
**Models:** MarianMT, M2M100 | **Languages:** 100+

## Overview

Production-ready neural machine translation system supporting 100+ language pairs with state-of-the-art MarianMT and M2M100 models. Features include batch processing, automatic language detection, GPU acceleration, and custom model fine-tuning capabilities.

## Features

### Core Capabilities
- **100+ Language Pairs:** en-es, en-fr, en-de, en-zh, fr-de, etc.
- **Neural MT:** MarianMT (fast, accurate) and M2M100 (multilingual)
- **Batch Processing:** Translate multiple texts efficiently
- **GPU Acceleration:** Automatic CUDA utilization
- **Quality Scoring:** Confidence scores for translations
- **Custom Models:** Support for fine-tuned domain-specific models

### Supported Language Pairs
- **Popular:** en↔es, en↔fr, en↔de, en↔zh, en↔ja, en↔ru, en↔ar
- **European:** fr↔de, de↔es, it↔fr, pt↔es
- **Asian:** zh↔ja, ko↔ja, zh↔ko
- **And 80+ more pairs**

## Installation

```bash
pip install transformers torch sentencepiece sacremoses
```

## Quick Start

### Basic Translation

```python
from machinetranslation import MachineTranslator

# Initialize for English to Spanish
translator = MachineTranslator(source_lang='en', target_lang='es')

# Translate
result = translator.translate("Hello, how are you?")
print(result['translation'])  # "Hola, ¿cómo estás?"
```

### Command Line

```bash
# Single translation
python machinetranslation.py \
    --text "Hello world" \
    --source en \
    --target es

# From file
python machinetranslation.py \
    --file texts.txt \
    --source en \
    --target fr

# Multiple targets
for lang in es fr de it; do
    python machinetranslation.py --text "Hello" --source en --target $lang
done
```

## Supported Models

| Model Family | Languages | Quality | Speed | Size |
|--------------|-----------|---------|-------|------|
| MarianMT | 100+ pairs | ⭐⭐⭐⭐ | Fast | 300MB |
| M2M100-418M | 100→100 | ⭐⭐⭐⭐ | Medium | 1.6GB |
| M2M100-1.2B | 100→100 | ⭐⭐⭐⭐⭐ | Slow | 4.8GB |

## Usage Examples

### 1. Basic Translation

```python
translator = MachineTranslator(source_lang='en', target_lang='es')
result = translator.translate("Machine learning is fascinating")
print(result['translation'])
```

### 2. Batch Translation

```python
texts = [
    "Hello world",
    "How are you?",
    "Thank you very much"
]

results = translator.translate_batch(texts)
for r in results:
    print(f"{r['source_text']} → {r['translation']}")
```

### 3. Multiple Language Pairs

```python
# Translate to multiple languages
source_text = "Welcome to our platform"

for target in ['es', 'fr', 'de', 'it']:
    translator = MachineTranslator('en', target)
    result = translator.translate(source_text)
    print(f"[{target}] {result['translation']}")
```

### 4. Long Text Translation

```python
long_text = """
Your long document here spanning multiple paragraphs...
"""

# Automatically chunks if needed
result = translator.translate(long_text, max_length=512)
```

## Language Codes

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| en | English | es | Spanish | fr | French |
| de | German | it | Italian | pt | Portuguese |
| zh | Chinese | ja | Japanese | ko | Korean |
| ar | Arabic | ru | Russian | hi | Hindi |

*Full list: ISO 639-1 codes*

## Performance Benchmarks

### WMT14 En-De (BLEU Score)

| Model | BLEU | Speed (sent/sec) |
|-------|------|------------------|
| MarianMT | 28.4 | 45 |
| M2M100-418M | 29.2 | 25 |
| M2M100-1.2B | 31.8 | 12 |

*On NVIDIA V100 GPU*

## Use Cases

### 1. Website Localization
```python
# Translate website content
content = {
    'title': 'Welcome',
    'subtitle': 'Learn more about us',
    'body': '...'
}

translator = MachineTranslator('en', 'es')
translated = {k: translator.translate(v)['translation'] for k, v in content.items()}
```

### 2. Customer Support
```python
# Real-time translation for support
customer_message = "I need help with my order"
translator_en_to_es = MachineTranslator('en', 'es')
translated_msg = translator_en_to_es.translate(customer_message)

# Response translation
response = "We'll help you right away"
translator_es_to_en = MachineTranslator('es', 'en')
```

### 3. Document Translation
```bash
python machinetranslation.py \
    --file legal_document.txt \
    --source en \
    --target es \
    --output translated_document.txt
```

### 4. E-commerce Product Descriptions
```python
products = load_products()
translator = MachineTranslator('en', 'es')

for product in products:
    product['description_es'] = translator.translate(
        product['description_en']
    )['translation']
```

## Best Practices

### 1. Choose Right Model
- **MarianMT:** Best for single language pair, fastest
- **M2M100:** Best for multiple languages, slightly slower

### 2. Batch for Efficiency
```python
# Good - batch processing
results = translator.translate_batch(texts)

# Avoid - one at a time
for text in texts:
    result = translator.translate(text)  # Slower
```

### 3. Handle Long Texts
```python
# Automatically handles chunking
result = translator.translate(long_text, max_length=512)
```

### 4. Cache Translations
```python
translation_cache = {}

def get_translation(text, src, tgt):
    key = (text, src, tgt)
    if key not in translation_cache:
        translator = MachineTranslator(src, tgt)
        translation_cache[key] = translator.translate(text)
    return translation_cache[key]
```

## API Reference

```python
class MachineTranslator:
    def __init__(self, source_lang: str, target_lang: str)
    
    def translate(self, text: str, max_length: int = 512) -> Dict
    
    def translate_batch(self, texts: List[str]) -> List[Dict]
```

## Troubleshooting

**Problem:** Unsupported language pair
**Solution:** Use M2M100 model for rare pairs

**Problem:** Poor translation quality
**Solution:** Try M2M100-1.2B model, check input text quality

**Problem:** Out of memory
**Solution:** Reduce max_length, use smaller model, process in smaller batches

## Version History
- **v2.0:** MarianMT + M2M100, 100+ languages, batch processing
- **v1.0:** Basic Google Translate API wrapper

---

**BrillConsulting** - Advanced NLP Solutions
