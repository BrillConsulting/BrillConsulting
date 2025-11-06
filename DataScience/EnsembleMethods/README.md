# Ensemble Methods Toolkit

A comprehensive toolkit implementing advanced ensemble learning techniques for combining multiple models to achieve superior predictive performance.

## Description

The Ensemble Methods Toolkit provides a unified interface for building powerful ensemble models. It implements various ensemble strategies including bagging, boosting, stacking, voting, and blending to leverage the strengths of multiple base learners for improved accuracy, robustness, and generalization.

## Key Features

- **Bagging Methods**
  - Bootstrap aggregating
  - Random Forest implementation
  - Extra Trees ensemble
  - Out-of-bag error estimation
  - Feature importance aggregation

- **Boosting Algorithms**
  - AdaBoost (Adaptive Boosting)
  - Gradient Boosting
  - XGBoost integration
  - LightGBM integration
  - Sample weight adaptation

- **Stacking**
  - Multi-level stacking
  - Cross-validation-based predictions
  - Meta-learner training
  - Feature concatenation strategies

- **Voting Ensembles**
  - Hard voting (majority voting)
  - Soft voting (probability averaging)
  - Weighted voting
  - Dynamic weight optimization

- **Blending**
  - Hold-out set-based blending
  - Weight optimization
  - Custom blending functions

- **Analysis Tools**
  - Diversity metrics (Q-statistic, correlation)
  - Feature importance aggregation
  - Model contribution analysis
  - Ensemble pruning

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms and ensemble methods
- **XGBoost** - Gradient boosting framework (optional)
- **LightGBM** - Gradient boosting framework (optional)
- **Matplotlib/Seaborn** - Visualization

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/EnsembleMethods

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn

# Optional: Install advanced boosting libraries
pip install xgboost lightgbm
```

## Usage Examples

### Bagging Ensemble

```python
from ensemble_methods import EnsembleMethodsToolkit
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize toolkit
ensemble = EnsembleMethodsToolkit(random_state=42)

# Create bagging ensemble
base_estimator = DecisionTreeClassifier(max_depth=10)
bagging_result = ensemble.bagging(
    X_train, y_train,
    X_test, y_test,
    base_estimator=base_estimator,
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8
)

print(f"Bagging Results:")
print(f"  Accuracy: {bagging_result['accuracy']:.3f}")
print(f"  OOB Score: {bagging_result['oob_score']:.3f}")
print(f"  Number of estimators: {bagging_result['n_estimators']}")
print(f"  Diversity (Q-statistic): {bagging_result['diversity_q']:.3f}")
```

### AdaBoost (Adaptive Boosting)

```python
# Apply AdaBoost
adaboost_result = ensemble.adaboost(
    X_train, y_train,
    X_test, y_test,
    n_estimators=100,
    learning_rate=1.0,
    algorithm='SAMME.R'
)

print(f"AdaBoost Results:")
print(f"  Accuracy: {adaboost_result['accuracy']:.3f}")
print(f"  Training error: {adaboost_result['train_error']:.3f}")
print(f"  Number of estimators: {adaboost_result['n_estimators']}")
print(f"  Estimator weights: {adaboost_result['estimator_weights'][:5]}")
print(f"  Estimator errors: {adaboost_result['estimator_errors'][:5]}")

# Visualize learning curve
fig = adaboost_result['learning_curve']
fig.savefig('adaboost_learning_curve.png', dpi=300, bbox_inches='tight')
```

### Gradient Boosting

```python
# Apply Gradient Boosting
gb_result = ensemble.gradient_boosting(
    X_train, y_train,
    X_test, y_test,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8
)

print(f"Gradient Boosting Results:")
print(f"  Accuracy: {gb_result['accuracy']:.3f}")
print(f"  Feature importances: {gb_result['feature_importances'][:5]}")
print(f"  Training deviance: {gb_result['train_deviance'][-1]:.4f}")
print(f"  Test deviance: {gb_result['test_deviance'][-1]:.4f}")

# Visualize feature importance
fig = gb_result['feature_importance_plot']
fig.savefig('gb_feature_importance.png', dpi=300, bbox_inches='tight')
```

### Stacking Ensemble

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Define base models
base_models = [
    ('dt', DecisionTreeClassifier(max_depth=10)),
    ('nb', GaussianNB()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Define meta-learner
meta_learner = LogisticRegression()

# Create stacking ensemble
stacking_result = ensemble.stacking(
    X_train, y_train,
    X_test, y_test,
    base_models=base_models,
    meta_learner=meta_learner,
    cv=5,
    use_probas=True
)

print(f"Stacking Results:")
print(f"  Accuracy: {stacking_result['accuracy']:.3f}")
print(f"  Base model performances:")
for name, score in stacking_result['base_model_scores'].items():
    print(f"    {name}: {score:.3f}")
print(f"  Meta-learner accuracy: {stacking_result['meta_learner_score']:.3f}")
print(f"  Improvement over best base: {stacking_result['improvement']:.3f}")
```

### Voting Ensemble

```python
# Create voting ensemble with hard voting
voting_result = ensemble.voting(
    X_train, y_train,
    X_test, y_test,
    estimators=base_models,
    voting='hard'
)

print(f"Hard Voting Results:")
print(f"  Accuracy: {voting_result['accuracy']:.3f}")
print(f"  Individual model accuracies:")
for name, acc in voting_result['individual_accuracies'].items():
    print(f"    {name}: {acc:.3f}")

# Create voting ensemble with soft voting
soft_voting_result = ensemble.voting(
    X_train, y_train,
    X_test, y_test,
    estimators=base_models,
    voting='soft',
    weights=[2, 1, 1]  # Give more weight to decision tree
)

print(f"\nSoft Voting Results:")
print(f"  Accuracy: {soft_voting_result['accuracy']:.3f}")
print(f"  Weights: {soft_voting_result['weights']}")
```

### Blending

```python
# Create blending ensemble
blending_result = ensemble.blending(
    X_train, y_train,
    X_test, y_test,
    base_models=[m[1] for m in base_models],
    meta_learner=LogisticRegression(),
    blend_size=0.2
)

print(f"Blending Results:")
print(f"  Accuracy: {blending_result['accuracy']:.3f}")
print(f"  Blend set size: {blending_result['blend_size']}")
print(f"  Optimal weights: {blending_result['optimal_weights']}")
print(f"  Base model contributions: {blending_result['contributions']}")
```

### Ensemble Diversity Analysis

```python
# Analyze ensemble diversity
diversity = ensemble.analyze_diversity(
    X_test, y_test,
    models=[bagging_result['model'], adaboost_result['model'],
            stacking_result['model']]
)

print(f"Ensemble Diversity Analysis:")
print(f"  Average Q-statistic: {diversity['avg_q_statistic']:.3f}")
print(f"  Average correlation: {diversity['avg_correlation']:.3f}")
print(f"  Disagreement measure: {diversity['disagreement']:.3f}")
print(f"  Double-fault measure: {diversity['double_fault']:.3f}")
print(f"\nPairwise diversity:")
for pair, q_stat in diversity['pairwise_q'].items():
    print(f"    {pair}: {q_stat:.3f}")
```

### Feature Importance Aggregation

```python
# Aggregate feature importances across ensemble members
importance_result = ensemble.aggregate_feature_importance(
    models=[bagging_result['model'], gb_result['model']],
    feature_names=[f'feature_{i}' for i in range(X.shape[1])],
    method='mean'
)

print(f"Aggregated Feature Importance:")
print(f"Top 5 features:")
for feat, imp in list(importance_result['importances'].items())[:5]:
    print(f"  {feat}: {imp:.4f}")

# Visualize aggregated importance
fig = importance_result['plot']
fig.savefig('aggregated_importance.png', dpi=300, bbox_inches='tight')
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python ensemble_methods.py
```

The demo will:
1. Generate synthetic classification dataset
2. Train multiple base models (Decision Trees, Random Forest, SVM, etc.)
3. Apply all ensemble methods (Bagging, AdaBoost, Gradient Boosting, Stacking, Voting, Blending)
4. Analyze ensemble diversity and model contributions
5. Compare performance across all methods
6. Generate visualizations (learning curves, feature importance, diversity metrics)
7. Display comprehensive performance comparison

## Output Examples

**Console Output:**
```
Ensemble Methods Toolkit Demo
======================================================================

Generating classification dataset...
Total samples: 1000
Features: 20 (15 informative, 5 redundant)
Classes: 2 (balanced)
Train/Test split: 700/300

Training Base Models...
----------------------------------------------------------------------
Decision Tree: Accuracy = 0.823
Random Forest: Accuracy = 0.867
SVM: Accuracy = 0.856
Naive Bayes: Accuracy = 0.789
KNN: Accuracy = 0.834

1. Bagging Ensemble
----------------------------------------------------------------------
Base estimator: Decision Tree (max_depth=10)
Number of estimators: 100
Max samples: 80%
Max features: 80%

Results:
  Accuracy: 0.887
  Precision: 0.881
  Recall: 0.893
  F1-score: 0.887
  OOB Score: 0.879
  Diversity (Q-statistic): 0.234

Improvement over base: +6.4%

2. AdaBoost
----------------------------------------------------------------------
Number of estimators: 100
Learning rate: 1.0
Algorithm: SAMME.R

Results:
  Accuracy: 0.901
  Training error: 0.043
  Final estimator weight: 3.456
  Convergence: Achieved at iteration 87

Weak learner contributions:
  Top 5 estimators: [12, 45, 67, 23, 89]
  Cumulative weight: 67.8%

3. Gradient Boosting
----------------------------------------------------------------------
Number of estimators: 100
Learning rate: 0.1
Max depth: 3
Subsample: 0.8

Results:
  Accuracy: 0.913
  Precision: 0.908
  Recall: 0.918
  F1-score: 0.913
  Training deviance: 0.234
  Test deviance: 0.267

Top 5 Important Features:
  1. feature_7: 0.1234
  2. feature_3: 0.0987
  3. feature_12: 0.0856
  4. feature_1: 0.0745
  5. feature_15: 0.0678

4. Stacking Ensemble
----------------------------------------------------------------------
Base models: Decision Tree, Naive Bayes, KNN
Meta-learner: Logistic Regression
Cross-validation folds: 5

Base model CV scores:
  Decision Tree: 0.821 (±0.023)
  Naive Bayes: 0.786 (±0.031)
  KNN: 0.831 (±0.019)

Stacking results:
  Accuracy: 0.907
  Improvement over best base: +7.6%
  Meta-learner contribution: +2.3%

5. Voting Ensemble (Hard)
----------------------------------------------------------------------
Estimators: 3
Voting strategy: Majority voting

Individual accuracies:
  Decision Tree: 0.823
  Naive Bayes: 0.789
  KNN: 0.834

Ensemble accuracy: 0.873
Improvement: +3.9%

6. Voting Ensemble (Soft)
----------------------------------------------------------------------
Voting strategy: Weighted probability averaging
Weights: [2, 1, 1] (Decision Tree emphasized)

Ensemble accuracy: 0.889
Improvement over hard voting: +1.6%

7. Blending
----------------------------------------------------------------------
Blend set size: 20% of training data
Meta-learner: Logistic Regression

Optimal weights found:
  Decision Tree: 0.412
  Naive Bayes: 0.123
  KNN: 0.465

Blending accuracy: 0.896
Model contributions:
  Decision Tree: 41.2%
  KNN: 46.5%
  Naive Bayes: 12.3%

Diversity Analysis
----------------------------------------------------------------------
Number of models compared: 7

Diversity metrics:
  Average Q-statistic: 0.234 (good diversity)
  Average correlation: 0.678
  Disagreement measure: 0.187
  Double-fault measure: 0.045

Pairwise diversity (Q-statistic):
  Bagging vs AdaBoost: 0.189
  Bagging vs Gradient Boosting: 0.234
  AdaBoost vs Gradient Boosting: 0.156
  Stacking vs Voting: 0.267

Performance Summary
----------------------------------------------------------------------
Method                    Accuracy    Precision   Recall      F1-Score
----------------------------------------------------------------------
Decision Tree (base)      0.823       0.816       0.831       0.823
Bagging                   0.887       0.881       0.893       0.887
AdaBoost                  0.901       0.896       0.907       0.901
Gradient Boosting         0.913       0.908       0.918       0.913
Stacking                  0.907       0.902       0.912       0.907
Voting (Hard)             0.873       0.867       0.879       0.873
Voting (Soft)             0.889       0.884       0.894       0.889
Blending                  0.896       0.891       0.901       0.896
----------------------------------------------------------------------

Best Method: Gradient Boosting (0.913 accuracy)
Average Improvement: +8.7% over best base model
```

**Generated Visualizations:**
- `ensemble_comparison.png` - Performance comparison across all methods
- `adaboost_learning_curve.png` - AdaBoost iteration vs. error
- `gb_feature_importance.png` - Gradient Boosting feature importance
- `diversity_heatmap.png` - Pairwise diversity between models
- `aggregated_importance.png` - Aggregated feature importance
- `voting_comparison.png` - Hard vs. soft voting comparison

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `ensemble_methods.py`.
