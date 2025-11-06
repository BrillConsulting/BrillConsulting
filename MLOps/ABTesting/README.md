# A/B Testing Framework

Production-ready A/B testing framework with advanced statistical methods and multi-armed bandit algorithms for optimizing experiments and making data-driven decisions.

## Features

### Multi-Armed Bandit Algorithms
- **Epsilon-Greedy**: Balance exploration and exploitation with configurable epsilon
- **Upper Confidence Bound (UCB)**: Optimistic exploration with confidence bounds
- **Thompson Sampling**: Bayesian approach using Beta distributions

### Statistical Tests
- **Z-Test**: For testing proportions (conversion rates)
- **T-Test**: For continuous metrics (revenue, engagement time)
- **Chi-Square Test**: For comparing multiple variants simultaneously
- **Bayesian A/B Testing**: Probability-based approach with expected loss calculation

### Advanced Features
- Traffic allocation and splitting strategies
- Sample size calculation with power analysis
- Confidence intervals for effect sizes
- Winner selection with statistical validation
- Sequential testing with early stopping
- Experiment tracking and reporting
- Lift calculation and significance testing

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from abtesting import ABTest

# Create experiment
test = ABTest(
    name="homepage_redesign",
    variants=["control", "variant_a", "variant_b"],
    metric_type="binary"  # or "continuous" for revenue
)

# Assign variant to user
variant = test.assign_variant(
    strategy="epsilon_greedy",  # or "ucb", "thompson_sampling"
    epsilon=0.1
)

# Track interactions
test.record_impression(variant)
test.record_conversion(variant, revenue=49.99)  # optional revenue

# Run statistical tests
z_result = test.z_test("control", "variant_a")
bayes_result = test.bayesian_test("control", "variant_a")

# Select winner
winner = test.select_winner(min_impressions=1000)
print(f"Winner: {winner['winner']}")
print(f"Recommendation: {winner['recommendation']}")

# Generate report
report = test.get_report()
test.save_experiment("./experiments/my_test.json")
```

## Usage Examples

### 1. Basic A/B Test with Random Assignment

```python
test = ABTest("button_color", ["blue", "green"], metric_type="binary")

# Simulate traffic
for user_id in range(1000):
    variant = test.assign_variant(strategy="random")
    test.record_impression(variant)

    # Simulate conversion (blue=8%, green=10%)
    if should_convert(variant):
        test.record_conversion(variant)

# Check results
result = test.z_test("blue", "green")
print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
print(f"Lift: {result['lift']:.2f}%")
```

### 2. Multi-Armed Bandit with Epsilon-Greedy

```python
test = ABTest("pricing_model", ["price_19", "price_29", "price_39"])

for user_id in range(5000):
    # Epsilon-greedy balances exploration/exploitation
    variant = test.assign_variant(strategy="epsilon_greedy", epsilon=0.1)
    test.record_impression(variant)

    if makes_purchase(user_id, variant):
        test.record_conversion(variant, revenue=get_price(variant))

# The algorithm automatically focuses on better-performing variants
```

### 3. Thompson Sampling for Faster Convergence

```python
test = ABTest("email_subject", ["subject_a", "subject_b", "subject_c"])

for email_sent in range(10000):
    # Thompson Sampling adapts quickly to best variant
    variant = test.assign_variant(strategy="thompson_sampling")
    test.record_impression(variant)

    if email_opened(variant):
        test.record_conversion(variant)
```

### 4. Sample Size Calculation

```python
# How many users do we need?
required_n = test.calculate_sample_size(
    baseline_rate=0.10,           # Current 10% conversion
    min_detectable_effect=0.20,   # Want to detect 20% lift
    alpha=0.05,                   # 5% significance level
    power=0.80                    # 80% power
)

print(f"Need {required_n} users per variant")
# Output: Need 3842 users per variant
```

### 5. Bayesian A/B Testing

```python
# Bayesian approach gives probability-based insights
result = test.bayesian_test("control", "variant_a", n_simulations=10000)

print(f"P(Variant A > Control): {result['prob_a_better']:.2%}")
print(f"Expected loss if choose A: {result['expected_loss_a']:.6f}")

# Make decision based on probability threshold
if result['prob_a_better'] > 0.95:
    print("Deploy Variant A with high confidence!")
```

### 6. Winner Selection with Validation

```python
# Automatically select winner with statistical validation
winner_result = test.select_winner(
    min_impressions=1000,
    confidence_level=0.95
)

if winner_result['winner']:
    print(f"Winner: {winner_result['winner']}")
    print(f"Metric: {winner_result['metric']} = {winner_result['value']:.4f}")
    print(f"Won {winner_result['significant_wins']} out of {winner_result['total_comparisons']} comparisons")
    print(f"Action: {winner_result['recommendation']}")
else:
    print("No clear winner yet. Continue experiment.")
```

## Choosing the Right Strategy

### Random Assignment
- Use for traditional A/B tests
- Equal traffic distribution
- Simple and unbiased
- Best when you have plenty of traffic

### Epsilon-Greedy
- Good balance of exploration/exploitation
- Configurable epsilon (0.1 = 10% exploration)
- Works well in most scenarios
- Easy to understand and explain

### Upper Confidence Bound (UCB)
- Optimistic exploration strategy
- Automatically balances exploration/exploitation
- Great for maximizing overall performance
- Best with limited traffic

### Thompson Sampling
- Bayesian approach
- Fastest convergence to best variant
- Naturally handles uncertainty
- Best for rapid optimization

## Statistical Tests Guide

### When to Use Each Test

**Z-Test (Proportions)**
- Binary outcomes (conversions, clicks)
- Large samples (n ≥ 30)
- Testing conversion rate differences
- Most common A/B test scenario

**T-Test (Continuous)**
- Continuous metrics (revenue, time on site)
- Any sample size
- Testing mean differences
- Revenue or engagement optimization

**Chi-Square Test**
- Comparing 3+ variants simultaneously
- Categorical outcomes
- Independence testing
- Multi-variant experiments

**Bayesian Test**
- When you want probability statements
- Continuous monitoring
- Small sample sizes
- Need to quantify uncertainty

## Metrics and Reporting

```python
# Get comprehensive report
report = test.get_report()

# Report includes:
# - Variant performance metrics
# - Statistical test results
# - Winner analysis
# - Experiment duration
# - Sample sizes

# Save for analysis
test.save_experiment("./experiments/results.json")
```

## Best Practices

1. **Determine Sample Size First**
   - Calculate required sample size before starting
   - Ensure adequate power (typically 80%)
   - Plan for 1-2 weeks minimum runtime

2. **Set Success Criteria**
   - Define primary metric before starting
   - Set significance threshold (typically α=0.05)
   - Document decision criteria

3. **Avoid Peeking**
   - Don't stop test early based on results
   - Use sequential testing if needed
   - Wait for required sample size

4. **Consider Practical Significance**
   - Statistical significance ≠ practical significance
   - Evaluate effect size and business impact
   - Use confidence intervals

5. **Run Multiple Tests Carefully**
   - Control for multiple comparisons
   - Use Bonferroni correction if needed
   - Focus on primary metric

## Technical Details

### Confidence Intervals
The framework calculates confidence intervals using:
- Normal approximation for proportions
- Pooled standard error estimation
- Configurable confidence levels (default 95%)

### Power Analysis
Sample size calculation based on:
- Effect size to detect
- Significance level (α)
- Statistical power (1-β)
- Baseline conversion rate

### Sequential Testing
- Continuously monitors experiments
- Implements alpha spending functions
- Allows early stopping with validity

## Example Output

```
A/B Testing Framework Demo
======================================================================

1. Creating Experiment
----------------------------------------------------------------------
✓ Created experiment: homepage_redesign
  Variants: ['control', 'variant_a', 'variant_b']

2. Running Experiment (Simulated)
----------------------------------------------------------------------
✓ Simulated 5000 users
  Strategy: Epsilon-Greedy (ε=0.1)

3. Variant Performance
----------------------------------------------------------------------
control:
  Impressions: 1543
  Conversions: 162
  Conversion Rate: 0.1050

variant_a:
  Impressions: 2834
  Conversions: 356
  Conversion Rate: 0.1256

variant_b:
  Impressions: 623
  Conversions: 71
  Conversion Rate: 0.1140

4. Statistical Significance Tests
----------------------------------------------------------------------
Z-Test (Control vs Variant A):
  p-value: 0.0234
  Statistically significant: True
  Lift: 19.62%

Bayesian Test (Control vs Variant A):
  P(A > B): 0.9847
  P(B > A): 0.0153
  Expected loss if choose A: 0.000234

6. Winner Selection
----------------------------------------------------------------------
Winner: variant_a
Best performer: variant_a
Metric: conversion_rate = 0.1256
Significant wins: 2/2
Recommendation: Deploy winner

✓ A/B Testing Demo Complete!
```

## Requirements

- Python 3.7+
- numpy
- scipy
- typing
- dataclasses (Python 3.7+)

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## License

Professional implementation for portfolio demonstration.

## References

- Multi-Armed Bandit Algorithms
- Sequential Probability Ratio Test (SPRT)
- Bayesian A/B Testing Methods
- Statistical Power Analysis
