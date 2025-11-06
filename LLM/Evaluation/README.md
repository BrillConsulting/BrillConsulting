# LLM Evaluation Toolkit

**Production-ready evaluation and benchmarking system for Large Language Models**

A comprehensive, enterprise-grade toolkit for evaluating, monitoring, and improving LLM performance across multiple dimensions including quality metrics, bias detection, A/B testing, and real-time performance monitoring.

## Features

### Core Evaluation Metrics
- **BLEU Scores**: Complete BLEU-1, BLEU-2, BLEU-3, BLEU-4 implementation with n-gram precision
- **ROUGE Metrics**: ROUGE-1, ROUGE-2, ROUGE-L for summarization quality
- **METEOR**: Alignment-based metric with synonym matching
- **Character/Word Error Rates**: Edit distance-based evaluation (CER/WER)
- **Perplexity Calculation**: Language model probability assessment
- **Exact Match & Word Overlap**: Basic similarity metrics

### Bias & Safety Detection
- **Multi-dimensional Bias Detection**:
  - Gender bias (male/female term imbalance)
  - Age bias (young/old stereotyping)
  - Racial bias indicators
  - Stereotype detection
- **Toxicity Analysis**: Profanity, aggressive language, and threat detection
- **Configurable Bias Patterns**: Extensible pattern matching system

### A/B Testing Framework
- **Statistical Significance Testing**: T-tests and p-value calculation
- **Power Analysis**: Sample size calculations for desired statistical power
- **Multi-variant Support**: Compare multiple model configurations
- **Automated Winner Selection**: Confidence-based decision making

### Human Evaluation
- **Structured Evaluation Forms**: 5-point scales for relevance, fluency, coherence, factuality
- **Inter-Annotator Agreement**: Calculate consensus between human evaluators
- **Tag-based Analysis**: Categorize and analyze evaluation patterns
- **Aggregate Statistics**: Mean, median, standard deviation across evaluators

### Performance Monitoring
- **Real-time Metrics**: Latency tracking (mean, p50, p95, p99)
- **Quality Trends**: Automatic trend detection (improving/stable/degrading)
- **Alert System**: Configurable thresholds with automatic alerting
- **Historical Analysis**: Time-windowed performance analysis
- **Model Comparison**: Side-by-side performance benchmarking

### Quality Assessment
- **Readability Scoring**: Flesch Reading Ease calculation
- **Lexical Diversity**: Vocabulary richness metrics
- **Structural Analysis**: Sentence length, word count, composition
- **Issue Detection**: Repetition, incompleteness, length problems

### Reporting & Export
- **Comprehensive Reports**: Detailed evaluation reports with all metrics
- **Multiple Export Formats**: JSON and CSV export
- **Aggregated Statistics**: Min, max, mean, median, standard deviation
- **Visual Summaries**: Human-readable formatted reports

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Evaluation

```python
from llm_evaluation import LLMEvaluator

evaluator = LLMEvaluator()

# Evaluate a single response
result = evaluator.evaluate_response(
    generated="Machine learning is a subset of AI.",
    reference="Machine learning is a subset of artificial intelligence.",
    model_id="gpt-4"
)

print(f"BLEU Score: {result.metrics['bleu']:.4f}")
print(f"ROUGE-L: {result.metrics['rouge_l']:.4f}")
print(f"Evaluation Level: {result.level}")
```

### Bias Detection

```python
# Detect bias in text
text = "The male engineer was assertive while the female nurse was nurturing."
bias_scores = evaluator.detect_bias(text)

print(f"Gender Bias: {bias_scores['gender_bias']:.4f}")
print(f"Overall Bias: {bias_scores['overall_bias']:.4f}")
```

### A/B Testing

```python
from llm_evaluation import ABTestVariant

# Create test variants
variants = [
    ABTestVariant(
        name="baseline",
        model_id="gpt-3.5-turbo",
        prompt_template="Answer: {input}",
        parameters={"temperature": 0.7}
    ),
    ABTestVariant(
        name="optimized",
        model_id="gpt-4",
        prompt_template="Detailed answer: {input}",
        parameters={"temperature": 0.5}
    )
]

# Create and run A/B test
test_id = evaluator.create_ab_test("model_comparison", variants)

def generate_function(variant, input_text):
    # Your model inference code here
    return f"Response from {variant.model_id}"

results = evaluator.run_ab_test(test_id, test_cases, generate_function)

if results['statistical_significance']['is_significant']:
    print(f"Winner: {results['winner']}")
    print(f"Confidence: {results['confidence']:.1f}%")
```

### Human Evaluation

```python
from llm_evaluation import HumanEvaluation
from datetime import datetime

# Add human evaluation
evaluation = HumanEvaluation(
    evaluator_id="annotator_001",
    timestamp=datetime.now().isoformat(),
    text_id="response_123",
    relevance=5,
    fluency=4,
    coherence=5,
    factuality=4,
    overall=4,
    comments="Clear and accurate response",
    tags=["helpful", "concise"]
)

evaluator.add_human_evaluation(evaluation)

# Calculate inter-annotator agreement
agreement = evaluator.calculate_inter_annotator_agreement("response_123")
print(f"Agreement Level: {agreement['agreement_level']}")
```

### Performance Monitoring

```python
# Monitor model performance over time
report = evaluator.monitor_performance("gpt-4", time_window_hours=24)

print(f"Mean Latency: {report['latency']['mean']:.4f}s")
print(f"P95 Latency: {report['latency']['p95']:.4f}s")
print(f"Quality Trend: {report['quality']['bleu_trend']}")

# Check for alerts
if report['alerts']:
    for alert in report['alerts']:
        print(f"[{alert['level']}] {alert['metric']}: {alert['value']:.4f}")
```

### Generate Reports

```python
# Generate comprehensive report
report = evaluator.generate_report("gpt-4", output_path="evaluation_report.txt")
print(report)

# Export results to JSON
evaluator.export_results("results.json", format="json")

# Export to CSV
evaluator.export_results("results.csv", format="csv")
```

### Perplexity Calculation

```python
# Calculate perplexity from token probabilities
token_probs = [0.8, 0.7, 0.9, 0.6, 0.85]  # From your LLM
perplexity = evaluator.calculate_perplexity("Sample text", token_probs)
print(f"Perplexity: {perplexity:.2f}")
```

## Configuration

```python
config = {
    "max_ngram": 4,
    "rouge_types": ["rouge-1", "rouge-2", "rouge-l"],
    "enable_advanced_metrics": True,
    "bias_detection_enabled": True,
    "performance_monitoring_enabled": True
}

evaluator = LLMEvaluator(config=config)

# Set custom alert thresholds
evaluator.set_alert_threshold("latency_p95", 1.5)
evaluator.set_alert_threshold("bleu_threshold", 0.4)
evaluator.set_alert_threshold("bias_score_max", 0.6)
```

## Advanced Usage

### Custom Bias Patterns

Extend the bias detection system with custom patterns:

```python
evaluator.bias_patterns[BiasCategory.CUSTOM] = {
    "terms": ["your", "custom", "terms"],
    "stereotypes": ["stereotype1", "stereotype2"]
}
```

### Benchmarking Multiple Models

```python
test_cases = [
    {"input": "What is Python?", "expected_output": "Python is a programming language."},
    {"input": "Define AI", "expected_output": "AI is artificial intelligence."},
    # ... more test cases
]

models = {
    "gpt-3.5": lambda x: your_model_a_inference(x),
    "gpt-4": lambda x: your_model_b_inference(x),
}

comparison = evaluator.compare_models(models, test_cases)
print(f"Best Model: {comparison['best_model']}")
print(f"Rankings by BLEU: {comparison['rankings']['avg_bleu']}")
```

### Statistical Power Analysis

```python
# Calculate required sample size for A/B test
effect_size = 0.3  # Cohen's d
required_n = evaluator.analyze_ab_test_power(
    effect_size=effect_size,
    alpha=0.05,
    power=0.8
)
print(f"Required samples per variant: {required_n}")
```

## API Reference

### Core Classes

- **`LLMEvaluator`**: Main evaluation class
- **`EvaluationResult`**: Structured result with metrics, quality scores, bias analysis
- **`HumanEvaluation`**: Human evaluation record
- **`ABTestVariant`**: A/B test variant configuration
- **`BiasCategory`**: Enum for bias types
- **`EvaluationLevel`**: Quality level classification

### Key Methods

- `evaluate_response()`: Comprehensive response evaluation
- `detect_bias()`: Multi-dimensional bias detection
- `detect_toxicity()`: Toxic content analysis
- `calculate_perplexity()`: Perplexity from probabilities
- `create_ab_test()`: Initialize A/B test
- `run_ab_test()`: Execute A/B test with statistical analysis
- `add_human_evaluation()`: Record human assessment
- `calculate_inter_annotator_agreement()`: Agreement metrics
- `monitor_performance()`: Performance monitoring report
- `generate_report()`: Comprehensive evaluation report
- `export_results()`: Export data to JSON/CSV

## Metrics Interpretation

### BLEU Score (0-1)
- **> 0.7**: Excellent match
- **0.5-0.7**: Good quality
- **0.3-0.5**: Acceptable
- **< 0.3**: Poor quality

### Bias Score (0-1)
- **< 0.3**: Low bias
- **0.3-0.6**: Medium bias
- **> 0.6**: High bias (requires review)

### Readability (Flesch Reading Ease)
- **> 80**: Very easy to read
- **60-80**: Easy
- **50-60**: Moderate
- **< 50**: Difficult

## Demo

Run the comprehensive demo to see all features:

```bash
python llm_evaluation.py
```

The demo showcases:
1. Comprehensive response evaluation with all metrics
2. Bias detection across multiple dimensions
3. Toxicity analysis
4. Perplexity calculation
5. Human evaluation framework
6. A/B testing setup
7. Performance monitoring
8. Report generation
9. Data export
10. Quality assessment

## Production Deployment

### Best Practices

1. **Set Appropriate Thresholds**: Configure alert thresholds based on your use case
2. **Monitor Continuously**: Use performance monitoring for production models
3. **Regular Bias Audits**: Run bias detection on sample outputs regularly
4. **Human Evaluation**: Supplement automated metrics with human assessment
5. **A/B Testing**: Always validate changes with statistical significance
6. **Export Results**: Maintain evaluation history for compliance and analysis

### Performance Considerations

- Evaluation operations are CPU-bound; consider parallel processing for large batches
- Keep history size limited (default: 1000 recent evaluations)
- Use time-windowed queries for performance monitoring
- Export and archive old results periodically

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional language-specific metrics
- More bias categories and patterns
- Advanced statistical tests
- Integration with popular LLM frameworks
- Visualization dashboards

## License

Copyright (c) 2024 Brill Consulting. All rights reserved.

## Support

For questions, issues, or feature requests, please contact Brill Consulting.

---

**Version**: 2.0.0 (Production-Ready)
**Last Updated**: 2024
**Author**: Brill Consulting
