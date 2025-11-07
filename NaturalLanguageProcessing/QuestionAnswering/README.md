# Advanced Question Answering System v2.0

**Author:** BrillConsulting
**Version:** 2.0 - Extractive & Generative QA with Transformers

## Overview

State-of-the-art question answering system supporting extractive QA (BERT, RoBERTa) and generative QA (T5, BART). Includes multi-document search and evaluation capabilities.

## Features

- **Extractive QA:** Extract answers from context (DistilBERT, BERT, RoBERTa)
- **Generative QA:** Generate answers (T5, BART)
- **Multi-Document:** Search across multiple documents
- **Top-K Answers:** Return multiple candidate answers
- **Evaluation:** Automated evaluation on QA datasets
- **GPU Support:** Accelerated inference

## Installation

```bash
pip install transformers torch numpy pandas
```

## Quick Start

```python
from questionanswering import QuestionAnsweringSystem

# Initialize
qa = QuestionAnsweringSystem()

# Answer question
result = qa.answer_question(
    question="What is BrillConsulting?",
    context="BrillConsulting is an AI company..."
)

print(result['answer'])
print(result['confidence'])
```

## Command Line

```bash
# Simple QA
python questionanswering.py \
    --question "Who founded Microsoft?" \
    --context "Microsoft was founded by Bill Gates and Paul Allen in 1975."

# From file
python questionanswering.py \
    --question "What is the capital?" \
    --context-file document.txt

# Top-K answers
python questionanswering.py \
    --question "What are the benefits?" \
    --context-file doc.txt \
    --top-k 3

# Evaluate on dataset
python questionanswering.py \
    --qa-file squad_dataset.json \
    --evaluate
```

## Models

| Model | Type | Size | Accuracy |
|-------|------|------|----------|
| distilbert-base-cased-distilled-squad | Extractive | 260MB | 86% |
| bert-large-uncased-whole-word-masking-finetuned-squad | Extractive | 1.3GB | 93% |
| roberta-base-squad2 | Extractive | 500MB | 89% |
| google/flan-t5-base | Generative | 850MB | 85% |

## Use Cases

- Customer support automation
- Document search & retrieval
- FAQ systems
- Educational tutoring
- Research assistance
- Legal document analysis

---

**BrillConsulting** - Advanced NLP Solutions
