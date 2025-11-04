# LLM Evaluation Toolkit

Comprehensive evaluation and benchmarking for language models.

## Features

- **Automatic Metrics**: BLEU, ROUGE, exact match, word overlap
- **Quality Assessment**: Length, diversity, readability checks
- **Issue Detection**: Repetition, incompleteness, length issues
- **Benchmarking**: Systematic testing on datasets
- **Model Comparison**: Compare multiple models side-by-side
- **Latency Tracking**: Performance monitoring

## Usage

```python
from llm_evaluation import LLMEvaluator

evaluator = LLMEvaluator()

# Evaluate single response
metrics = evaluator.evaluate_response(generated, reference)

# Quality assessment
quality = evaluator.evaluate_quality(text)

# Detect issues
issues = evaluator.detect_issues(text)

# Benchmark
test_cases = [{"input": "...", "expected_output": "..."}]
results = evaluator.benchmark(test_cases)

# Compare models
comparison = evaluator.compare_models(models, test_cases)
```

## Demo

```bash
python llm_evaluation.py
```

Provides comprehensive metrics for LLM outputs.
