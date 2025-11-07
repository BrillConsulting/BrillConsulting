# Advanced Text Generation System v2.0

**Author:** BrillConsulting | **Version:** 2.0 - GPT-2, GPT-Neo, GPT-J
**Models:** GPT-2, GPT-Neo (1.3B, 2.7B), GPT-J (6B)

## Overview

State-of-the-art text generation system powered by GPT-2 and GPT-Neo transformers. Supports creative writing, code generation, dialogue, and controlled text generation with temperature, top-k, and nucleus sampling.

## Features

- **Multiple Models:** GPT-2 (124M-1.5B), GPT-Neo (1.3B-2.7B), GPT-J (6B)
- **Controlled Generation:** Temperature, top-k, top-p sampling
- **Creative Modes:** Story generation, poetry, dialogue
- **Code Generation:** Python, JavaScript, SQL code completion
- **Batch Generation:** Multiple sequences from single prompt
- **GPU Acceleration:** CUDA support for faster generation

## Installation

```bash
pip install transformers torch
```

## Quick Start

```python
from text_generation_v2 import TextGenerator

# Initialize
generator = TextGenerator(model_name='gpt2')

# Generate
result = generator.generate(
    prompt="Once upon a time",
    max_length=100,
    temperature=1.0
)

print(result['generations'][0])
```

## Models

| Model | Parameters | Memory | Speed | Quality |
|-------|------------|--------|-------|---------|
| gpt2 | 124M | 500MB | Fast | ⭐⭐⭐ |
| gpt2-medium | 355M | 1.4GB | Medium | ⭐⭐⭐⭐ |
| gpt2-large | 774M | 3GB | Slow | ⭐⭐⭐⭐ |
| EleutherAI/gpt-neo-1.3B | 1.3B | 5GB | Slow | ⭐⭐⭐⭐⭐ |
| EleutherAI/gpt-j-6B | 6B | 24GB | Very Slow | ⭐⭐⭐⭐⭐ |

## Usage Examples

### 1. Creative Writing

```python
generator = TextGenerator('gpt2-medium')

result = generator.generate(
    "In a world where time travel exists,",
    max_length=200,
    temperature=1.2,  # More creative
    num_return_sequences=3
)

for idx, text in enumerate(result['generations']):
    print(f"\nVersion {idx+1}:")
    print(text)
```

### 2. Code Generation

```python
generator = TextGenerator('EleutherAI/gpt-neo-1.3B')

code_result = generator.generate(
    "def fibonacci(n):\n    ",
    max_length=150,
    temperature=0.7,  # Less random for code
    top_p=0.95
)

print(code_result['generations'][0])
```

### 3. Dialogue Generation

```python
prompt = """
Human: What is machine learning?
AI: Machine learning is
"""

result = generator.generate(
    prompt,
    max_length=100,
    temperature=0.9
)
```

### 4. Poetry Generation

```python
result = generator.generate_story(
    "Roses are red, violets are blue,",
    max_length=150
)
```

## Parameters

### Temperature
- **0.1-0.5:** Focused, deterministic
- **0.7-1.0:** Balanced creativity
- **1.2-2.0:** Highly creative, diverse

### Top-K
- **10:** Very focused
- **50:** Balanced (default)
- **100+:** More diverse

### Top-P (Nucleus Sampling)
- **0.9:** Focused
- **0.95:** Balanced (default)
- **0.99:** Diverse

## Command Line

```bash
# Basic generation
python text_generation_v2.py --prompt "The future of AI is"

# Creative story
python text_generation_v2.py \
    --prompt "In the year 2050" \
    --model gpt2-medium \
    --max-length 300 \
    --temperature 1.3

# Code generation
python text_generation_v2.py \
    --prompt "def quicksort(arr):" \
    --model EleutherAI/gpt-neo-1.3B \
    --temperature 0.7

# Multiple variations
python text_generation_v2.py \
    --prompt "Once upon a time" \
    --num-sequences 5 \
    --temperature 1.2
```

## Use Cases

1. **Content Creation:** Blog posts, articles, marketing copy
2. **Creative Writing:** Stories, poetry, scripts
3. **Code Completion:** Python, JavaScript, SQL
4. **Chatbots:** Dialogue and conversation
5. **Brainstorming:** Ideas and concepts
6. **Email Drafts:** Professional correspondence

## Best Practices

### For Creative Writing
```python
result = generator.generate(
    prompt,
    temperature=1.2,
    top_k=50,
    top_p=0.95,
    num_return_sequences=5
)
```

### For Code
```python
result = generator.generate(
    prompt,
    temperature=0.7,
    top_p=0.9,
    max_length=200
)
```

### For Factual Text
```python
result = generator.generate(
    prompt,
    temperature=0.5,
    top_k=20,
    num_beams=5
)
```

## Troubleshooting

**Problem:** Repetitive text
**Solution:** Increase temperature, reduce top_k

**Problem:** Nonsensical output
**Solution:** Decrease temperature, increase top_k

**Problem:** Out of memory
**Solution:** Use smaller model (gpt2 instead of gpt2-large)

---

**BrillConsulting** - Advanced NLP Solutions
