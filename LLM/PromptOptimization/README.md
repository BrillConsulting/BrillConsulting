# PromptOptimization

Production-ready system for optimizing LLM prompts using advanced algorithms and multi-objective optimization techniques.

## Overview

PromptOptimization is a comprehensive framework for automatically improving prompt quality through multiple optimization strategies including genetic algorithms, gradient-based methods, and multi-objective Pareto optimization. The system evaluates prompts across multiple dimensions (accuracy, latency, coherence, relevance, diversity) and tracks performance over time.

## Features

### Core Optimization Strategies

- **Genetic Algorithm Optimization**: Evolutionary approach using mutation, crossover, and selection
- **Gradient-Based Optimization**: Feedback-driven refinement targeting specific metrics
- **Hybrid Optimization**: Combined genetic exploration with gradient-based refinement
- **Multi-Objective Optimization**: Pareto front optimization balancing multiple objectives

### Evaluation Framework

- **Multi-Metric Evaluation**: Accuracy, latency, token count, coherence, relevance, diversity
- **Weighted Scoring**: Configurable composite scores for overall performance
- **Pareto Dominance**: Non-dominated solution identification for multi-objective problems
- **Custom Evaluation Functions**: Support for domain-specific evaluation logic

### Performance Tracking

- **Real-Time Monitoring**: Track metrics across all iterations
- **Best Prompt Tracking**: Maintain top-performing prompts
- **Convergence Detection**: Identify when optimization plateaus
- **Result Persistence**: Save optimization history and results

### Prompt Evolution

- **Multiple Mutation Operators**: Add, remove, replace, reorder, expand
- **Crossover Mechanisms**: Two-point crossover for prompt recombination
- **Elite Selection**: Preserve top performers across generations
- **Tournament Selection**: Probabilistic parent selection

## Architecture

### Class Structure

```
PromptOptimizationSystem
├── PromptEvaluator
│   ├── Custom evaluation functions
│   └── Default heuristic evaluation
├── GeneticPromptOptimizer
│   ├── Population initialization
│   ├── Evolution operators
│   └── Selection mechanisms
├── GradientPromptOptimizer
│   ├── Gradient approximation
│   └── Step-by-step refinement
├── MultiObjectiveOptimizer
│   ├── Pareto front management
│   └── Non-dominated solution tracking
└── PerformanceTracker
    ├── Metrics logging
    ├── Statistical analysis
    └── Result persistence
```

### Data Models

- **PromptCandidate**: Represents a prompt with metadata (generation, parent, mutation type)
- **EvaluationResult**: Stores all evaluation metrics with timestamp
- **Weighted Scoring**: Customizable metric weights for composite scores

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- numpy
- Standard library: json, logging, hashlib, pathlib, dataclasses

## Usage

### Basic Usage

```python
from promptoptimization import PromptOptimizationSystem

# Initialize system
system = PromptOptimizationSystem()

# Optimize a prompt
seed_prompt = "Explain the concept to the user."
optimized_prompt, result = system.optimize(
    seed_prompt=seed_prompt,
    strategy='genetic',
    max_iterations=50
)

print(f"Optimized: {optimized_prompt}")
print(f"Score: {result.weighted_score:.4f}")
```

### Advanced Usage with Custom Evaluation

```python
def custom_evaluator(prompt: str, test_cases: list) -> dict:
    """Custom evaluation logic"""
    # Your evaluation implementation
    return {
        'accuracy': 0.95,
        'latency': 0.3,
        'token_count': len(prompt.split()),
        'coherence': 0.9,
        'relevance': 0.85,
        'diversity': 0.7
    }

system = PromptOptimizationSystem(evaluation_fn=custom_evaluator)
optimized_prompt, result = system.optimize(
    seed_prompt="Your prompt here",
    strategy='hybrid',
    max_iterations=100,
    test_cases=[{'input': 'test', 'expected': 'output'}]
)
```

### Strategy Comparison

```python
# Compare different strategies
strategies = ['genetic', 'gradient', 'hybrid', 'multi_objective']
results = {}

for strategy in strategies:
    optimized, result = system.optimize(
        seed_prompt=seed_prompt,
        strategy=strategy,
        max_iterations=50
    )
    results[strategy] = result.weighted_score

# Get best strategy
best_strategy = max(results, key=results.get)
print(f"Best strategy: {best_strategy}")
```

### Multi-Objective Optimization

```python
# Optimize for multiple objectives
optimized, result = system.optimize(
    seed_prompt=seed_prompt,
    strategy='multi_objective',
    max_iterations=100
)

# Access Pareto front
pareto_front = system.multi_objective.pareto_front
print(f"Found {len(pareto_front)} non-dominated solutions")

# Get best compromise
best_candidate, best_result = system.multi_objective.get_best_compromise()
```

## Optimization Strategies

### Genetic Algorithm

**Best for**: Broad exploration of prompt space, discovering novel formulations

**Parameters**:
- `population_size`: Number of prompts in each generation (default: 20)
- `mutation_rate`: Probability of mutation (default: 0.3)
- `crossover_rate`: Probability of crossover (default: 0.7)
- `elite_size`: Number of top prompts preserved (default: 2)

**Mutation Operators**:
1. Add Word: Insert descriptive adjectives or adverbs
2. Remove Word: Eliminate redundant words
3. Replace Word: Substitute with synonyms
4. Reorder: Shuffle sentence order
5. Expand: Add instructional phrases

### Gradient-Based

**Best for**: Fine-tuning prompts toward specific target metrics

**Parameters**:
- `learning_rate`: Step size for optimization (default: 0.1)
- `target_metrics`: Desired metric values

**Approach**:
- Calculate gaps between current and target metrics
- Apply modifications to address largest gaps
- Iteratively refine until convergence

### Hybrid

**Best for**: Balanced exploration and exploitation

**Process**:
1. Phase 1 (60%): Genetic exploration to find promising regions
2. Phase 2 (40%): Gradient refinement to polish best candidates

### Multi-Objective

**Best for**: Complex scenarios requiring trade-offs between competing objectives

**Features**:
- Maintains Pareto front of non-dominated solutions
- Identifies optimal trade-offs
- Archives dominated solutions for analysis

## Evaluation Metrics

### Primary Metrics

| Metric | Range | Description | Weight |
|--------|-------|-------------|--------|
| Accuracy | 0-1 | Correctness of responses | 0.35 |
| Latency | 0+ | Response time in seconds | 0.15 |
| Token Count | 0+ | Prompt length in tokens | 0.15 |
| Coherence | 0-1 | Logical flow and structure | 0.20 |
| Relevance | 0-1 | Alignment with requirements | 0.10 |
| Diversity | 0-1 | Vocabulary richness | 0.05 |

### Composite Score

```
weighted_score = 0.35×accuracy + 0.15×(1-latency/10) +
                 0.15×(1-tokens/1000) + 0.20×coherence +
                 0.10×relevance + 0.05×diversity
```

## Performance Tracking

### Logging

All optimizations are tracked with:
- Iteration number and timestamp
- Prompt content and ID
- All evaluation metrics
- Strategy and generation metadata

### Convergence Detection

The system automatically detects convergence when the score variation within a 10-iteration window falls below 0.01.

### Results Persistence

Results are saved in timestamped directories:
```
optimization_results/
├── run_name_20250101_120000/
│   ├── metrics_history.json
│   ├── best_prompts.json
│   └── pareto_front.json
```

## API Reference

### PromptOptimizationSystem

Main orchestration class for all optimization strategies.

#### Methods

**`optimize(seed_prompt, strategy, max_iterations, target_metrics, test_cases)`**
- Optimize a prompt using specified strategy
- Returns: `(optimized_prompt, evaluation_result)`

**`get_report()`**
- Generate comprehensive optimization report
- Returns: Dict with tracker, evaluator, and multi-objective stats

**`save_results(run_name)`**
- Save all optimization results to disk
- Creates timestamped directory with JSON files

### PromptEvaluator

Evaluates prompts using multiple metrics.

#### Methods

**`evaluate(candidate, test_cases)`**
- Evaluate a prompt candidate
- Returns: `EvaluationResult`

**`get_statistics()`**
- Get statistical summary of all evaluations
- Returns: Dict with averages, best scores, improvement

### GeneticPromptOptimizer

Genetic algorithm implementation for prompt evolution.

#### Methods

**`initialize_population(seed_prompt)`**
- Create initial population from seed
- Returns: List of `PromptCandidate`

**`evolve(population, fitness_scores)`**
- Evolve population to next generation
- Returns: New population list

### GradientPromptOptimizer

Gradient-based optimization using feedback.

#### Methods

**`optimize_step(prompt, evaluation_result, target_metrics)`**
- Perform one optimization step
- Returns: Improved prompt string

### MultiObjectiveOptimizer

Multi-objective optimization using Pareto fronts.

#### Methods

**`update_pareto_front(candidate, result)`**
- Update Pareto front with new candidate
- Returns: Boolean (True if non-dominated)

**`get_best_compromise()`**
- Get best compromise solution from Pareto front
- Returns: `(PromptCandidate, EvaluationResult)`

**`get_statistics()`**
- Get statistics about optimization
- Returns: Dict with front size, best metrics

## Examples

### Example 1: Basic Optimization

```python
from promptoptimization import PromptOptimizationSystem

system = PromptOptimizationSystem()

# Simple optimization
optimized, result = system.optimize(
    seed_prompt="Write a summary.",
    strategy='genetic',
    max_iterations=30
)

print(f"Original: Write a summary.")
print(f"Optimized: {optimized}")
print(f"Improvement: {result.weighted_score:.4f}")
```

### Example 2: Target-Driven Optimization

```python
# Optimize toward specific targets
target_metrics = {
    'accuracy': 0.98,
    'coherence': 0.95,
    'relevance': 0.90
}

optimized, result = system.optimize(
    seed_prompt="Analyze the data.",
    strategy='gradient',
    max_iterations=50,
    target_metrics=target_metrics
)
```

### Example 3: Complete Workflow

```python
# Initialize with custom evaluation
def evaluate_with_llm(prompt, test_cases):
    # Call your LLM API
    responses = [call_llm(prompt, tc) for tc in test_cases]
    return calculate_metrics(responses)

system = PromptOptimizationSystem(
    evaluation_fn=evaluate_with_llm,
    save_dir='./my_optimization_results'
)

# Run optimization
optimized, result = system.optimize(
    seed_prompt="Your initial prompt",
    strategy='hybrid',
    max_iterations=100,
    test_cases=load_test_cases()
)

# Generate report
report = system.get_report()
print(f"Total iterations: {report['tracker_summary']['total_iterations']}")
print(f"Best score: {report['tracker_summary']['best_score']:.4f}")
print(f"Improvement: {report['tracker_summary']['improvement']:.4f}")

# Save results
system.save_results('production_run')
```

### Example 4: Batch Optimization

```python
# Optimize multiple prompts
prompts = [
    "Explain quantum computing.",
    "Summarize the article.",
    "Generate test cases."
]

results = {}
for prompt in prompts:
    optimized, result = system.optimize(
        seed_prompt=prompt,
        strategy='hybrid',
        max_iterations=50
    )
    results[prompt] = (optimized, result.weighted_score)

# Compare results
for original, (optimized, score) in results.items():
    print(f"\nOriginal: {original}")
    print(f"Optimized: {optimized}")
    print(f"Score: {score:.4f}")
```

## Performance Considerations

### Computational Complexity

- **Genetic**: O(population_size × iterations × evaluation_cost)
- **Gradient**: O(iterations × evaluation_cost)
- **Hybrid**: O((0.6 × pop_size + 0.4) × iterations × eval_cost)
- **Multi-objective**: O(population_size × iterations × evaluation_cost + pareto_updates)

### Memory Usage

- Each prompt candidate: ~1-10 KB (depends on prompt length)
- Evaluation history: ~1 KB per evaluation
- Pareto front: ~10-100 prompts typically

### Optimization Tips

1. **Start with genetic** for broad exploration (20-30 iterations)
2. **Use gradient** when you have clear target metrics
3. **Choose hybrid** for best overall results
4. **Multi-objective** for complex trade-offs (accuracy vs. latency)
5. **Increase population** for more diverse exploration
6. **Higher mutation rate** encourages exploration
7. **Lower mutation rate** refines existing solutions

## Testing

Run the demo:

```bash
python promptoptimization.py
```

Expected output:
- Initialization messages
- Progress logs for each strategy
- Optimized prompts with scores
- Performance report
- Saved results location

## Configuration

### Genetic Algorithm Configuration

```python
genetic_optimizer = GeneticPromptOptimizer(
    population_size=30,     # More prompts = better exploration
    mutation_rate=0.4,      # Higher = more variation
    crossover_rate=0.8,     # Higher = more recombination
    elite_size=3            # Preserve top 3 prompts
)
```

### Gradient Optimizer Configuration

```python
gradient_optimizer = GradientPromptOptimizer(
    learning_rate=0.15  # Higher = faster convergence (but less stable)
)
```

### Performance Tracker Configuration

```python
tracker = PerformanceTracker(
    save_dir='./custom_results'
)
```

## Troubleshooting

### Issue: Slow convergence

**Solution**:
- Increase population size
- Adjust mutation rate
- Use hybrid strategy
- Reduce max_iterations for faster testing

### Issue: Poor optimization results

**Solution**:
- Check evaluation function accuracy
- Verify test cases are representative
- Try different strategies
- Increase iterations
- Adjust metric weights

### Issue: Memory issues with large populations

**Solution**:
- Reduce population_size
- Limit tracking history
- Use gradient instead of genetic
- Process in batches

## Best Practices

1. **Always validate** with real test cases after optimization
2. **Save results regularly** to avoid losing progress
3. **Monitor convergence** to avoid unnecessary iterations
4. **Use custom evaluators** for domain-specific optimization
5. **Compare strategies** to find best fit for your use case
6. **Track multiple runs** to ensure reproducibility
7. **Analyze Pareto front** for multi-objective problems

## Limitations

- Evaluation quality depends on test cases
- Computational cost scales with population and iterations
- May require domain expertise for custom evaluation
- No guarantee of global optimum
- Results vary due to stochastic nature of algorithms

## Future Enhancements

- Neural prompt optimization using embeddings
- Active learning for efficient evaluation
- Transfer learning across similar tasks
- Automated hyperparameter tuning
- Real-time visualization dashboard
- Distributed optimization for large-scale problems
- Integration with popular LLM APIs

## Contributing

Contributions welcome! Areas of interest:
- New mutation operators
- Alternative selection strategies
- Better evaluation metrics
- Performance optimizations
- Documentation improvements

## License

Copyright (c) 2025 BrillConsulting. All rights reserved.

## Support

For questions or issues:
- Review the examples in this documentation
- Check the troubleshooting section
- Examine the code documentation
- Test with the included demo

## Version History

### v1.0.0 (2025-01-06)
- Initial production release
- Genetic, gradient, hybrid, and multi-objective optimization
- Comprehensive evaluation framework
- Performance tracking and persistence
- Complete documentation and examples

## Acknowledgments

Built with inspiration from:
- Genetic algorithms in evolutionary computation
- Gradient-based optimization in deep learning
- Multi-objective optimization theory
- Natural language processing research

---

**BrillConsulting** - Professional LLM Solutions
