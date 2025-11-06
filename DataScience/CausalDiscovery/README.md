# Causal Discovery Toolkit

A comprehensive toolkit for discovering causal relationships and building causal graphs from observational data using advanced statistical and algorithmic methods.

## Description

The Causal Discovery Toolkit provides powerful methods for inferring causal structures from data. It implements constraint-based, score-based, and hybrid algorithms to discover directed acyclic graphs (DAGs) representing causal relationships, enabling users to move beyond correlation to understand causation.

## Key Features

- **Constraint-Based Algorithms**
  - PC (Peter-Clark) algorithm
  - FCI (Fast Causal Inference)
  - Conditional independence testing
  - Skeleton discovery
  - Edge orientation with v-structures

- **Conditional Independence Tests**
  - Partial correlation tests
  - Chi-square test for discrete variables
  - Kernel-based independence tests
  - Fisher's Z-transform

- **Causal Graph Construction**
  - DAG (Directed Acyclic Graph) building
  - CPDAG (Completed Partially DAG) construction
  - Markov equivalence class identification
  - Edge orientation rules

- **Markov Blanket Discovery**
  - Identifying direct causes and effects
  - Finding minimal conditioning sets
  - Parent-child relationship detection
  - Spouse (common effect) identification

- **D-Separation Testing**
  - Path blocking analysis
  - Conditional independence verification
  - Backdoor criterion checking
  - Front-door criterion checking

- **Intervention Simulation**
  - Do-calculus implementation
  - Counterfactual reasoning
  - Treatment effect estimation
  - Causal effect identification

- **Visualization**
  - Causal graph visualization with NetworkX
  - DAG plotting with node positions
  - V-structure highlighting
  - Edge strength visualization

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **NetworkX** - Graph data structures and algorithms
- **SciPy** - Statistical tests
- **scikit-learn** - Machine learning utilities
- **Matplotlib** - Visualization
- **pgmpy** - Probabilistic graphical models (optional)

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/CausalDiscovery

# Install required packages
pip install numpy pandas networkx scipy scikit-learn matplotlib
```

## Usage Examples

### PC Algorithm for Causal Discovery

```python
from causal_discovery import CausalDiscoveryToolkit
import numpy as np
import pandas as pd

# Generate data with known causal structure
# True structure: X -> Y -> Z, X -> Z
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples)
Y = 0.8 * X + np.random.randn(n_samples) * 0.5
Z = 0.6 * Y + 0.4 * X + np.random.randn(n_samples) * 0.3

data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

# Initialize toolkit
causal = CausalDiscoveryToolkit(alpha=0.05, random_state=42)

# Discover causal structure using PC algorithm
pc_result = causal.pc_algorithm(
    data,
    alpha=0.05,
    method='pearson'
)

print(f"PC Algorithm Results:")
print(f"  Discovered edges: {pc_result['edges']}")
print(f"  Skeleton: {pc_result['skeleton']}")
print(f"  V-structures: {pc_result['v_structures']}")
print(f"  Separating sets: {pc_result['separating_sets']}")

# Visualize causal graph
fig = pc_result['graph_plot']
fig.savefig('causal_graph.png', dpi=300, bbox_inches='tight')
```

### Conditional Independence Testing

```python
# Test conditional independence: X ⊥ Z | Y
ci_test = causal.conditional_independence_test(
    data,
    x='X',
    y='Z',
    conditioning_set=['Y'],
    method='partial_correlation'
)

print(f"Conditional Independence Test: X ⊥ Z | Y")
print(f"  Test statistic: {ci_test['statistic']:.4f}")
print(f"  P-value: {ci_test['p_value']:.4f}")
print(f"  Independent: {ci_test['independent']}")
print(f"  Correlation: {ci_test['correlation']:.4f}")
print(f"  Partial correlation: {ci_test['partial_correlation']:.4f}")

# Test without conditioning
unconditional_test = causal.conditional_independence_test(
    data,
    x='X',
    y='Z',
    conditioning_set=[],
    method='partial_correlation'
)

print(f"\nUnconditional Test: X ⊥ Z")
print(f"  P-value: {unconditional_test['p_value']:.4f}")
print(f"  Independent: {unconditional_test['independent']}")
```

### Markov Blanket Discovery

```python
# Discover Markov blanket of a variable
mb_result = causal.discover_markov_blanket(
    data,
    target_var='Y',
    alpha=0.05
)

print(f"Markov Blanket of Y:")
print(f"  Variables in blanket: {mb_result['markov_blanket']}")
print(f"  Parents: {mb_result['parents']}")
print(f"  Children: {mb_result['children']}")
print(f"  Spouses (common effects): {mb_result['spouses']}")
print(f"  Blanket size: {mb_result['blanket_size']}")
```

### D-Separation Testing

```python
# Build causal graph
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([('X', 'Y'), ('Y', 'Z'), ('X', 'Z')])

# Test d-separation
d_sep_result = causal.d_separation(
    graph=G,
    x='X',
    y='Z',
    conditioning_set=['Y']
)

print(f"D-Separation Test: X ⊥_d Z | Y")
print(f"  D-separated: {d_sep_result['d_separated']}")
print(f"  Active paths: {d_sep_result['active_paths']}")
print(f"  Blocked paths: {d_sep_result['blocked_paths']}")
print(f"  Conditioning blocks all paths: {d_sep_result['all_blocked']}")
```

### Intervention Simulation (Do-Calculus)

```python
# Simulate intervention: do(X = x0)
intervention_result = causal.simulate_intervention(
    data,
    causal_graph=G,
    intervention_var='X',
    intervention_value=1.0,
    outcome_var='Z'
)

print(f"Intervention Simulation: do(X = 1.0)")
print(f"  Observational E[Z]: {intervention_result['observational_mean']:.3f}")
print(f"  Interventional E[Z | do(X=1.0)]: {intervention_result['interventional_mean']:.3f}")
print(f"  Causal effect: {intervention_result['causal_effect']:.3f}")
print(f"  95% CI: [{intervention_result['ci_lower']:.3f}, {intervention_result['ci_upper']:.3f}]")

# Visualize intervention effect
fig = intervention_result['effect_plot']
fig.savefig('intervention_effect.png', dpi=300, bbox_inches='tight')
```

### Treatment Effect Estimation

```python
# Estimate average treatment effect (ATE)
ate_result = causal.estimate_treatment_effect(
    data,
    treatment='X',
    outcome='Z',
    causal_graph=G,
    adjustment_set=['Y'],
    method='regression'
)

print(f"Average Treatment Effect (ATE):")
print(f"  ATE: {ate_result['ate']:.3f}")
print(f"  Standard error: {ate_result['se']:.3f}")
print(f"  95% CI: [{ate_result['ci_lower']:.3f}, {ate_result['ci_upper']:.3f}]")
print(f"  P-value: {ate_result['p_value']:.4f}")
print(f"  Adjustment set used: {ate_result['adjustment_set']}")
```

### Backdoor Criterion Checking

```python
# Check backdoor criterion for causal effect identification
backdoor_result = causal.check_backdoor_criterion(
    graph=G,
    treatment='X',
    outcome='Z',
    adjustment_set=['Y']
)

print(f"Backdoor Criterion Check:")
print(f"  Criterion satisfied: {backdoor_result['satisfied']}")
print(f"  Backdoor paths: {backdoor_result['backdoor_paths']}")
print(f"  All paths blocked: {backdoor_result['all_blocked']}")
print(f"  Valid adjustment set: {backdoor_result['valid_adjustment']}")

# Find minimal adjustment set
minimal_set = causal.find_minimal_adjustment_set(
    graph=G,
    treatment='X',
    outcome='Z'
)

print(f"\nMinimal Adjustment Set:")
print(f"  Variables: {minimal_set['minimal_set']}")
print(f"  Set size: {len(minimal_set['minimal_set'])}")
```

### Causal Graph Validation

```python
# Validate discovered causal graph against data
validation = causal.validate_causal_graph(
    data,
    causal_graph=G,
    n_tests=100
)

print(f"Causal Graph Validation:")
print(f"  Tests passed: {validation['tests_passed']}/{validation['total_tests']}")
print(f"  Pass rate: {validation['pass_rate']:.1f}%")
print(f"  Graph consistent with data: {validation['is_valid']}")
print(f"  Failed tests: {validation['failed_tests']}")

# Calculate graph quality metrics
quality = causal.graph_quality_metrics(
    true_graph=G,  # If ground truth is known
    discovered_graph=pc_result['graph']
)

print(f"\nGraph Quality Metrics:")
print(f"  Precision: {quality['precision']:.3f}")
print(f"  Recall: {quality['recall']:.3f}")
print(f"  F1-score: {quality['f1_score']:.3f}")
print(f"  Structural Hamming Distance: {quality['shd']}")
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python causal_discovery.py
```

The demo will:
1. Generate synthetic data with known causal structure
2. Apply PC algorithm to discover causal relationships
3. Perform conditional independence tests
4. Discover Markov blankets
5. Test d-separation properties
6. Simulate interventions and estimate causal effects
7. Check backdoor criterion and find adjustment sets
8. Validate discovered causal graphs
9. Generate visualizations (causal graphs, intervention effects)
10. Display comprehensive causal analysis results

## Output Examples

**Console Output:**
```
Causal Discovery Toolkit Demo
======================================================================

Generating data with known causal structure...
True causal structure:
  X → Y → Z
  X → Z
  W → X
  W → Z (confounding path)

Sample size: 1000
Variables: X, Y, Z, W

1. PC Algorithm - Skeleton Discovery
----------------------------------------------------------------------
Discovering skeleton with α = 0.05...
Initial complete graph edges: 6
After independence testing: 5 edges

Skeleton edges:
  X - Y, Y - Z, X - Z, W - X, W - Z

Conditional independence tests performed: 15
Separating sets found: 1
  Z ⊥ W | {X, Y}

2. PC Algorithm - Edge Orientation
----------------------------------------------------------------------
Detecting v-structures...
V-structures found: 1
  X → Y ← W (collision at Y)

Applying orientation rules...
  Rule 1 (avoid cycles): 3 edges oriented
  Rule 2 (avoid new v-structures): 1 edge oriented
  Rule 3 (avoid cycles): 0 edges oriented

Final DAG edges:
  W → X → Y → Z
  W → Z
  X → Z

3. Conditional Independence Tests
----------------------------------------------------------------------
Test: X ⊥ Z | Y
  Partial correlation: -0.023
  P-value: 0.234
  Result: Independent (cannot reject)

Test: X ⊥ Z | ∅
  Correlation: 0.756
  P-value: < 0.001
  Result: Dependent (reject independence)

Test: X ⊥ W | ∅
  Correlation: 0.623
  P-value: < 0.001
  Result: Dependent (X and W are associated)

4. Markov Blanket Discovery
----------------------------------------------------------------------
Target variable: Y

Markov Blanket of Y: {X, Z}
  Parents: {X}
  Children: {Z}
  Spouses: ∅

Blanket discovery method: Forward-backward selection
Tests performed: 12

5. D-Separation Analysis
----------------------------------------------------------------------
Query: X ⊥_d Z | Y

Active paths (without conditioning): 2
  X → Y → Z
  X → Z

After conditioning on Y:
  Blocked paths: 1 (X → Y → Z)
  Active paths: 1 (X → Z)

Result: NOT d-separated (direct edge X → Z remains active)

6. Intervention Simulation
----------------------------------------------------------------------
Intervention: do(X = 2.0)
Outcome: Z

Observational distribution:
  E[Z]: 1.234
  Std[Z]: 2.345

Interventional distribution do(X = 2.0):
  E[Z | do(X=2.0)]: 2.567
  Std[Z | do(X=2.0)]: 1.234

Causal effect: 1.333
95% Confidence Interval: [1.156, 1.510]

Interpretation: Increasing X by 1 unit causes Z to increase by 1.333 units

7. Treatment Effect Estimation
----------------------------------------------------------------------
Treatment: X
Outcome: Z
Adjustment set: {W} (blocks backdoor path)

Average Treatment Effect (ATE):
  ATE: 1.289
  Standard error: 0.045
  95% CI: [1.201, 1.377]
  P-value: < 0.001

CATE by subgroup (W quartiles):
  Q1 (W < -0.67): ATE = 1.234
  Q2 (-0.67 ≤ W < 0.00): ATE = 1.278
  Q3 (0.00 ≤ W < 0.67): ATE = 1.301
  Q4 (W ≥ 0.67): ATE = 1.345

8. Backdoor Criterion
----------------------------------------------------------------------
Treatment: X
Outcome: Z

Backdoor paths from X to Z:
  X ← W → Z

Checking adjustment set {W}:
  Blocks path X ← W → Z: Yes
  Contains descendant of X: No
  Contains node on causal path: No

Backdoor criterion satisfied: Yes
{W} is a valid adjustment set

Minimal adjustment set: {W}

9. Causal Graph Validation
----------------------------------------------------------------------
Testing d-separation implications...
Total tests: 100
Tests passed: 94
Pass rate: 94.0%

Failed tests:
  X ⊥ Y | W (expected independent, found dependent)
  Note: May indicate unmeasured confounding

Graph quality (vs. true graph):
  Precision: 1.000 (no false positives)
  Recall: 0.833 (1 missing edge)
  F1-score: 0.909
  Structural Hamming Distance: 2

Discovered graph is consistent with data!
```

**Generated Visualizations:**
- `causal_graph.png` - Discovered causal DAG
- `skeleton_evolution.png` - Skeleton discovery process
- `v_structures.png` - Detected colliders
- `intervention_effect.png` - Intervention simulation results
- `treatment_effect_distribution.png` - ATE distribution
- `graph_comparison.png` - True vs. discovered graph

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `causal_discovery.py`.
