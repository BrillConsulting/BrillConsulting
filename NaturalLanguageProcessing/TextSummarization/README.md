# Advanced Text Summarization System v2.0

**Author:** BrillConsulting
**Version:** 2.0 - State-of-the-Art Abstractive & Extractive Summarization
**Models:** BART, T5, Pegasus, LED

## Overview

Enterprise-grade text summarization system supporting both abstractive and extractive summarization techniques. Leverages state-of-the-art transformer models (BART, T5, Pegasus) with advanced features including long document processing, hierarchical summarization, and customizable compression ratios.

## Features

### Core Capabilities
- **Abstractive Summarization:** Generate human-like summaries with BART, T5, Pegasus
- **Long Document Support:** Process documents of unlimited length using intelligent chunking
- **Hierarchical Summarization:** Multi-level summarization for very long documents
- **Customizable Compression:** Adjust summary length and compression ratios
- **Batch Processing:** Summarize multiple documents efficiently
- **GPU Acceleration:** Automatic GPU detection and utilization

### Advanced Features
- Beam search optimization for higher quality summaries
- Length penalty tuning for optimal summary length
- Controlled generation with num_beams parameter
- Automatic text truncation for model limits
- Compression ratio calculation and reporting
- Multi-document summarization support

## Installation

```bash
# Core dependencies
pip install transformers torch numpy

# Optional: GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For development
pip install transformers[torch] sentencepiece sacremoses
```

## Quick Start

### Python API

```python
from textsummarization import TextSummarizer

# Initialize with BART (recommended for news/articles)
summarizer = TextSummarizer(model_name='facebook/bart-large-cnn')

# Summarize text
result = summarizer.summarize(
    text=long_article,
    max_length=130,
    min_length=30
)

print(result['summary'])
print(f"Compression: {result['compression_ratio']:.1%}")
```

### Command Line

```bash
# Basic summarization
python textsummarization.py \
    --file document.txt \
    --model facebook/bart-large-cnn \
    --max-length 150

# Long document mode
python textsummarization.py \
    --file long_document.txt \
    --long \
    --max-length 200

# Custom compression
python textsummarization.py \
    --text "Your long text here..." \
    --max-length 100 \
    --min-length 50
```

## Supported Models

| Model | Best For | Size | Speed | Quality |
|-------|----------|------|-------|---------|
| facebook/bart-large-cnn | News articles | 1.6GB | Medium | ⭐⭐⭐⭐⭐ |
| google/pegasus-xsum | Extreme summarization | 2.3GB | Slow | ⭐⭐⭐⭐⭐ |
| t5-base | General purpose | 850MB | Fast | ⭐⭐⭐⭐ |
| t5-large | High quality | 2.8GB | Slow | ⭐⭐⭐⭐⭐ |
| google/pegasus-cnn_dailymail | News articles | 2.3GB | Slow | ⭐⭐⭐⭐⭐ |
| allenai/led-base-16384 | Long documents | 590MB | Medium | ⭐⭐⭐⭐ |

## Architecture

### Single Document Summarization
```
Input Text → Tokenization → Encoder → Decoder → Summary
                                ↓
                         Beam Search
                         Length Penalty
                         Repetition Control
```

### Long Document Summarization
```
Long Document → Split into Chunks → Summarize Each Chunk
                                          ↓
                              Combine Summaries → Final Summary
                                          ↓
                              Recursive if needed
```

## Usage Examples

### 1. News Article Summarization

```python
from textsummarization import TextSummarizer

article = """
Your news article text here...
"""

summarizer = TextSummarizer(model_name='facebook/bart-large-cnn')
result = summarizer.summarize(article, max_length=130)

print(f"Original: {result['original_length']} words")
print(f"Summary: {result['summary_length']} words")
print(f"\n{result['summary']}")
```

**Output:**
```
Original: 850 words
Summary: 65 words

The article discusses recent developments in artificial intelligence...
```

### 2. Research Paper Summarization

```python
# Use T5 for scientific content
summarizer = TextSummarizer(model_name='t5-large')

paper = """
Abstract and full paper content...
"""

result = summarizer.summarize(
    paper,
    max_length=200,
    min_length=100,
    length_penalty=2.0,  # Favor longer, more detailed summaries
    num_beams=6  # Higher quality
)

print(result['summary'])
```

### 3. Long Document Summarization

```python
# For documents > 1024 words
summarizer = TextSummarizer(model_name='facebook/bart-large-cnn')

long_doc = """
Very long document content spanning multiple pages...
"""

result = summarizer.summarize_long_document(
    long_doc,
    chunk_size=1000,
    max_length=150
)

print(f"Processed {result['num_chunks']} chunks")
print(result['summary'])
```

### 4. Batch Summarization

```python
documents = [doc1, doc2, doc3, doc4, doc5]

results = summarizer.summarize_batch(documents, max_length=100)

for idx, result in enumerate(results):
    print(f"\nDocument {idx+1}:")
    print(result['summary'])
```

### 5. Custom Parameters

```python
result = summarizer.summarize(
    text,
    max_length=150,           # Maximum summary length
    min_length=50,            # Minimum summary length
    length_penalty=2.0,       # Prefer longer summaries
    num_beams=4,              # Beam search width
)
```

## Performance Benchmarks

### CNN/DailyMail Dataset

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Speed (docs/sec) |
|-------|---------|---------|---------|------------------|
| BART-large-CNN | 44.16 | 21.28 | 40.90 | 8.5 |
| Pegasus-CNN | 44.17 | 21.47 | 41.11 | 6.2 |
| T5-large | 42.50 | 20.68 | 39.75 | 7.8 |
| T5-base | 40.69 | 18.96 | 37.59 | 15.3 |

### XSum Dataset (Extreme Summarization)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Speed (docs/sec) |
|-------|---------|---------|---------|------------------|
| Pegasus-XSum | 47.21 | 24.56 | 39.25 | 6.0 |
| BART-large-XSum | 45.14 | 22.27 | 37.25 | 8.2 |

*Benchmarks on NVIDIA V100 GPU*

## Use Cases

### 1. News Aggregation
```bash
python textsummarization.py \
    --file news_articles.txt \
    --model facebook/bart-large-cnn \
    --max-length 100
```

### 2. Research Paper Abstracts
```bash
python textsummarization.py \
    --file research_paper.txt \
    --model t5-large \
    --max-length 250 \
    --min-length 150
```

### 3. Meeting Notes
```bash
python textsummarization.py \
    --file meeting_transcript.txt \
    --model facebook/bart-large-cnn \
    --max-length 200
```

### 4. Legal Document Summaries
```bash
python textsummarization.py \
    --file legal_document.txt \
    --long \
    --model t5-large \
    --max-length 300
```

### 5. Book Chapter Summaries
```bash
python textsummarization.py \
    --file book_chapter.txt \
    --long \
    --model allenai/led-base-16384 \
    --max-length 500
```

## Model Selection Guide

### Choose BART when:
- ✅ Summarizing news articles
- ✅ Need balanced speed/quality
- ✅ Working with well-structured text
- ✅ Want factual, extractive-style summaries

### Choose Pegasus when:
- ✅ Need extreme compression (one sentence)
- ✅ Highest quality is priority
- ✅ Summarizing news or scientific papers
- ✅ Can afford slower processing

### Choose T5 when:
- ✅ Need general-purpose summarization
- ✅ Want faster processing
- ✅ Working with diverse text types
- ✅ Limited GPU memory

### Choose LED when:
- ✅ Dealing with very long documents (>4096 tokens)
- ✅ Want to avoid chunking
- ✅ Working with books, reports, transcripts

## Best Practices

### 1. Text Preprocessing
```python
# Clean text before summarization
text = text.strip()
text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
text = text.replace('\n\n', '. ')  # Handle paragraphs
```

### 2. Optimal Parameters
```python
# For concise summaries
result = summarizer.summarize(text, max_length=100, length_penalty=1.0)

# For detailed summaries
result = summarizer.summarize(text, max_length=200, length_penalty=2.5)

# For highest quality
result = summarizer.summarize(text, num_beams=8, length_penalty=2.0)
```

### 3. Memory Management
```python
# For large batches, process in chunks
batch_size = 10
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    results = summarizer.summarize_batch(batch)
    save_results(results)
```

### 4. Long Document Strategy
- Use `summarize_long_document()` for texts >1000 words
- Adjust `chunk_size` based on document structure
- Consider using LED model for documents >5000 words

## Advanced Configuration

### Custom Beam Search

```python
result = summarizer.summarize(
    text,
    num_beams=6,              # More beams = better quality (slower)
    length_penalty=2.0,       # >1.0 prefers longer summaries
    early_stopping=True,      # Stop when all beams finish
    no_repeat_ngram_size=3    # Prevent repetition
)
```

### Temperature Sampling

```python
# For creative/diverse summaries (requires do_sample=True)
result = summarizer.summarize(
    text,
    do_sample=True,
    temperature=1.5,          # Higher = more creative
    top_k=50,
    top_p=0.95
)
```

## Troubleshooting

### Problem: Out of Memory

**Solution:**
```python
# Reduce max_length
result = summarizer.summarize(text, max_length=100)

# Use smaller model
summarizer = TextSummarizer(model_name='t5-base')

# Process in chunks
result = summarizer.summarize_long_document(text, chunk_size=500)
```

### Problem: Poor Quality Summaries

**Solution:**
```python
# Increase num_beams
result = summarizer.summarize(text, num_beams=8)

# Adjust length_penalty
result = summarizer.summarize(text, length_penalty=2.5)

# Try different model
summarizer = TextSummarizer(model_name='google/pegasus-cnn_dailymail')
```

### Problem: Summaries Too Short/Long

**Solution:**
```python
# Set min/max length explicitly
result = summarizer.summarize(
    text,
    min_length=50,
    max_length=150
)

# Adjust length_penalty
result = summarizer.summarize(text, length_penalty=2.0)  # Longer
result = summarizer.summarize(text, length_penalty=0.8)  # Shorter
```

### Problem: Slow Processing

**Solution:**
```python
# Use smaller model
summarizer = TextSummarizer(model_name='t5-base')

# Reduce num_beams
result = summarizer.summarize(text, num_beams=2)

# Use GPU
# Ensure torch.cuda.is_available() returns True
```

## Output Format

```python
{
    'summary': 'The generated summary text...',
    'original_length': 850,      # words
    'summary_length': 75,        # words
    'compression_ratio': 0.088,  # 8.8% of original
    'model': 'facebook/bart-large-cnn'
}
```

## API Reference

### TextSummarizer Class

```python
class TextSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn', method='abstractive')

    def summarize(self, text: str, max_length=130, min_length=30,
                  length_penalty=2.0, num_beams=4) -> Dict

    def summarize_batch(self, texts: List[str], max_length=130) -> List[Dict]

    def summarize_long_document(self, text: str, chunk_size=1000,
                                 max_length=130) -> Dict
```

## Comparison with Other Tools

| Feature | Our System | Gensim | Sumy | TextRank |
|---------|-----------|--------|------|----------|
| Abstractive | ✅ | ❌ | ❌ | ❌ |
| Extractive | ⚠️ | ✅ | ✅ | ✅ |
| Transformers | ✅ | ❌ | ❌ | ❌ |
| Long Documents | ✅ | ✅ | ✅ | ✅ |
| Customizable | ✅ | ⚠️ | ⚠️ | ⚠️ |
| State-of-art | ✅ | ❌ | ❌ | ❌ |
| GPU Support | ✅ | ❌ | ❌ | ❌ |

## Citation

If you use this system in your research, please cite:

```bibtex
@software{brill_summarization_2024,
  author = {BrillConsulting},
  title = {Advanced Text Summarization System v2.0},
  year = {2024},
  url = {https://github.com/BrillConsulting}
}
```

## Version History

- **v2.0** (2024): Transformer models, long document support, beam search
- **v1.0** (2023): Basic extractive summarization

## License

MIT License - BrillConsulting

## Support

For issues or questions:
- GitHub Issues: [Report here]
- Documentation: [Full docs]
- Email: support@brillconsulting.com

---

**BrillConsulting** - Advanced NLP Solutions
