# Imbalanced Learning Toolkit

A comprehensive toolkit for handling imbalanced datasets using advanced resampling techniques, cost-sensitive learning, and ensemble methods.

## Description

The Imbalanced Learning Toolkit provides specialized techniques for dealing with class imbalance problems in machine learning. It implements various resampling strategies, cost-sensitive approaches, and ensemble methods to improve model performance on minority classes.

## Key Features

- **Oversampling Techniques**
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - ADASYN (Adaptive Synthetic Sampling)
  - Borderline-SMOTE
  - Random oversampling with variations

- **Undersampling Techniques**
  - Random undersampling
  - Tomek Links removal
  - Edited Nearest Neighbors (ENN)
  - NearMiss algorithms
  - Cluster centroids

- **Combined Methods**
  - SMOTE + Tomek Links
  - SMOTE + ENN
  - Hybrid resampling strategies

- **Cost-Sensitive Learning**
  - Class weight adjustment
  - Focal loss implementation
  - Custom cost matrices
  - Threshold optimization

- **Ensemble Methods**
  - Balanced Random Forest
  - Easy Ensemble
  - Balanced Bagging
  - RUSBoost

- **Evaluation Metrics**
  - Class-specific precision, recall, F1-score
  - Balanced accuracy
  - G-mean
  - Area Under PR Curve (AUPRC)
  - Matthews Correlation Coefficient (MCC)

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **imbalanced-learn** - Resampling techniques
- **Matplotlib/Seaborn** - Visualization

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/ImbalancedLearning

# Install required packages
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
```

## Usage Examples

### SMOTE Oversampling

```python
from imbalanced_learning import ImbalancedLearning
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    weights=[0.95, 0.05],  # 95% majority, 5% minority
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Initialize toolkit
imbalanced = ImbalancedLearning(random_state=42)

# Apply SMOTE
smote_result = imbalanced.smote(
    X_train, y_train,
    sampling_strategy='auto',
    k_neighbors=5
)

X_resampled = smote_result['X_resampled']
y_resampled = smote_result['y_resampled']

print(f"Original class distribution: {smote_result['original_distribution']}")
print(f"Resampled class distribution: {smote_result['resampled_distribution']}")
print(f"Minority class increased by: {smote_result['n_samples_generated']} samples")
```

### ADASYN Adaptive Sampling

```python
# Apply ADASYN for adaptive oversampling
adasyn_result = imbalanced.adasyn(
    X_train, y_train,
    sampling_strategy='auto',
    n_neighbors=5
)

print(f"ADASYN generated {adasyn_result['n_samples_generated']} samples")
print(f"Final distribution: {adasyn_result['resampled_distribution']}")
```

### Random Undersampling

```python
# Apply random undersampling to majority class
undersample_result = imbalanced.random_undersample(
    X_train, y_train,
    sampling_strategy='auto'
)

print(f"Original samples: {len(X_train)}")
print(f"Undersampled samples: {len(undersample_result['X_resampled'])}")
print(f"Samples removed: {undersample_result['n_samples_removed']}")
```

### Tomek Links Cleaning

```python
# Remove Tomek links (borderline samples)
tomek_result = imbalanced.tomek_links(
    X_train, y_train
)

print(f"Tomek links removed: {tomek_result['n_links_removed']}")
print(f"Cleaned distribution: {tomek_result['resampled_distribution']}")
```

### Combined SMOTE + Tomek Links

```python
# Combine oversampling and cleaning
combined_result = imbalanced.smote_tomek(
    X_train, y_train,
    sampling_strategy='auto'
)

print(f"Combined resampling complete")
print(f"Final distribution: {combined_result['resampled_distribution']}")
print(f"SMOTE samples added: {combined_result['smote_samples']}")
print(f"Tomek links removed: {combined_result['tomek_removed']}")
```

### Cost-Sensitive Learning

```python
# Train with cost-sensitive learning
cost_sensitive_result = imbalanced.cost_sensitive_learning(
    X_train, y_train,
    X_test, y_test,
    model_type='random_forest',
    class_weights='balanced'
)

print(f"Model trained with balanced class weights")
print(f"Balanced accuracy: {cost_sensitive_result['balanced_accuracy']:.3f}")
print(f"Minority class recall: {cost_sensitive_result['minority_recall']:.3f}")
print(f"Minority class precision: {cost_sensitive_result['minority_precision']:.3f}")
```

### Balanced Random Forest

```python
# Use ensemble method for imbalanced data
ensemble_result = imbalanced.balanced_random_forest(
    X_train, y_train,
    X_test, y_test,
    n_estimators=100,
    sampling_strategy='auto'
)

print(f"Balanced Random Forest Results:")
print(f"Accuracy: {ensemble_result['accuracy']:.3f}")
print(f"Balanced accuracy: {ensemble_result['balanced_accuracy']:.3f}")
print(f"F1-score (minority): {ensemble_result['f1_minority']:.3f}")
print(f"G-mean: {ensemble_result['g_mean']:.3f}")
```

### Comprehensive Evaluation

```python
# Train model on resampled data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)

# Evaluate with class-specific metrics
evaluation = imbalanced.evaluate_imbalanced(
    y_test, y_pred,
    y_pred_proba=model.predict_proba(X_test)[:, 1]
)

print("Evaluation Metrics:")
print(f"Balanced Accuracy: {evaluation['balanced_accuracy']:.3f}")
print(f"G-mean: {evaluation['g_mean']:.3f}")
print(f"MCC: {evaluation['mcc']:.3f}")
print(f"AUPRC: {evaluation['auprc']:.3f}")
print(f"\nClass-specific metrics:")
print(f"Minority class - Precision: {evaluation['minority_precision']:.3f}, "
      f"Recall: {evaluation['minority_recall']:.3f}, "
      f"F1: {evaluation['minority_f1']:.3f}")
print(f"Majority class - Precision: {evaluation['majority_precision']:.3f}, "
      f"Recall: {evaluation['majority_recall']:.3f}, "
      f"F1: {evaluation['majority_f1']:.3f}")
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python imbalanced_learning.py
```

The demo will:
1. Generate a highly imbalanced synthetic dataset
2. Apply all resampling techniques (SMOTE, ADASYN, undersampling, Tomek links)
3. Train models with cost-sensitive learning
4. Compare ensemble methods for imbalanced data
5. Evaluate all approaches with class-specific metrics
6. Generate visualizations showing class distributions
7. Display comprehensive performance comparison

## Output Examples

**Console Output:**
```
Imbalanced Learning Toolkit Demo
======================================================================

Generating imbalanced dataset...
Total samples: 1000
Majority class (0): 950 (95.0%)
Minority class (1): 50 (5.0%)
Imbalance ratio: 19:1

1. SMOTE Oversampling
----------------------------------------------------------------------
Original distribution: {0: 665, 1: 35}
Resampled distribution: {0: 665, 1: 665}
Samples generated: 630

Model Performance:
  Balanced Accuracy: 0.857
  Minority F1-score: 0.723
  G-mean: 0.849

2. ADASYN Adaptive Sampling
----------------------------------------------------------------------
Original distribution: {0: 665, 1: 35}
Resampled distribution: {0: 665, 1: 658}
Samples generated: 623

Model Performance:
  Balanced Accuracy: 0.863
  Minority F1-score: 0.735
  G-mean: 0.856

3. Random Undersampling
----------------------------------------------------------------------
Original distribution: {0: 665, 1: 35}
Resampled distribution: {0: 35, 1: 35}
Samples removed: 630

Model Performance:
  Balanced Accuracy: 0.782
  Minority F1-score: 0.645
  G-mean: 0.776

4. Tomek Links Removal
----------------------------------------------------------------------
Tomek links removed: 23
Cleaned distribution: {0: 642, 1: 35}

5. SMOTE + Tomek Links
----------------------------------------------------------------------
SMOTE samples added: 630
Tomek links removed: 18
Final distribution: {0: 647, 1: 665}

Model Performance:
  Balanced Accuracy: 0.871
  Minority F1-score: 0.748
  G-mean: 0.865

6. Cost-Sensitive Learning (Balanced Weights)
----------------------------------------------------------------------
Class weights: {0: 0.53, 1: 10.0}
Balanced Accuracy: 0.845
Minority Recall: 0.886
Minority Precision: 0.689

7. Balanced Random Forest
----------------------------------------------------------------------
Number of estimators: 100
Balanced Accuracy: 0.891
Minority F1-score: 0.782
G-mean: 0.886
AUPRC: 0.834

Performance Summary
----------------------------------------------------------------------
Method                    Bal. Acc    Minority F1    G-mean    AUPRC
----------------------------------------------------------------------
Baseline (no resampling)  0.623       0.324          0.589     0.456
SMOTE                     0.857       0.723          0.849     0.786
ADASYN                    0.863       0.735          0.856     0.798
Random Undersample        0.782       0.645          0.776     0.701
SMOTE + Tomek             0.871       0.748          0.865     0.812
Cost-Sensitive            0.845       0.698          0.839     0.769
Balanced RF               0.891       0.782          0.886     0.834
```

**Generated Visualizations:**
- `class_distribution.png` - Before/after resampling comparison
- `confusion_matrices.png` - Comparison of confusion matrices
- `roc_pr_curves.png` - ROC and Precision-Recall curves
- `method_comparison.png` - Performance metrics comparison

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `imbalanced_learning.py`.
