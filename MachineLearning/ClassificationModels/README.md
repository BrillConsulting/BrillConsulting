# 🎯 Classification Models

Multi-algorithm classification system with 7 algorithms, automatic model comparison, and comprehensive evaluation metrics.

## 🌟 Algorithms

1. **Logistic Regression** - Linear binary/multi-class classification
2. **SVM** - Support Vector Machines with RBF kernel
3. **Decision Tree** - Tree-based classification
4. **Random Forest** - Ensemble of decision trees
5. **Gradient Boosting** - Boosted tree ensemble
6. **K-Nearest Neighbors** - Instance-based learning
7. **Naive Bayes** - Probabilistic classifier

## 🚀 Quick Start

```bash
python classifiers.py --data data.csv --target label --output confusion.png
```

## 📊 Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / Actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve (binary classification)
- **Confusion Matrix**: Visual performance breakdown

## 📝 Example

```python
from classifiers import ClassificationAnalyzer

analyzer = ClassificationAnalyzer()
X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y)
analyzer.train_all_models(X_train, y_train, X_test, y_test)
comparison = analyzer.compare_models()
print(comparison)
```

## 🎨 Use Cases

- Spam detection
- Customer churn prediction
- Medical diagnosis
- Fraud detection
- Image classification
- Sentiment analysis

---

**Author**: BrillConsulting
