# Text Classification System v2.0

**Author:** BrillConsulting
**Version:** 2.0 - Advanced Implementation with Transformers & Deep Learning

## Overview

Enterprise-grade text classification system combining traditional machine learning, deep learning with PyTorch, and state-of-the-art Transformer models (BERT, RoBERTa, DistilBERT) for high-performance text classification tasks.

## Features

### Multiple Model Types
- **Traditional ML:** Naive Bayes, Logistic Regression, Linear SVM, Random Forest, Gradient Boosting
- **Deep Learning:** Custom PyTorch neural networks with batch normalization and dropout
- **Transformers:** BERT, RoBERTa, DistilBERT via HuggingFace
- **Ensemble:** Voting classifier combining best models

### Advanced Techniques
- N-gram features (unigrams, bigrams, trigrams)
- Hyperparameter tuning with cross-validation
- Class balancing for imbalanced datasets
- Comprehensive preprocessing pipeline
- GPU acceleration support
- Label encoding for multi-class problems

### Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- ROC AUC (for binary classification)
- Confusion matrices
- Detailed classification reports
- Visual model comparisons

## Installation

```bash
pip install numpy pandas torch scikit-learn transformers datasets
pip install matplotlib seaborn nltk joblib
python -m nltk.downloader punkt stopwords wordnet
```

## Usage

### Basic Usage

```python
from text_classifier import AdvancedTextClassifier

# Initialize
classifier = AdvancedTextClassifier(
    vectorizer_type='tfidf',
    max_features=5000,
    use_gpu=True
)

# Prepare data
X_train, X_test, y_train, y_test = classifier.prepare_data(texts, labels)

# Train all models
classifier.train_all_models(
    X_train, y_train, X_test, y_test,
    include_deep_learning=True,
    include_transformer=False
)

# Compare models
comparison = classifier.compare_models()
print(comparison)

# Predict
predictions = classifier.predict(['New text to classify'], model_name='Ensemble')
```

### Command Line

```bash
# Basic classification with traditional ML
python text_classifier.py \
    --data dataset.csv \
    --text-col text \
    --label-col label \
    --vectorizer tfidf

# With deep learning
python text_classifier.py \
    --data dataset.csv \
    --deep-learning \
    --output results.png \
    --save-model best_model.pkl

# With transformers
python text_classifier.py \
    --data dataset.csv \
    --transformer \
    --max-features 10000
```

## Model Performance

Typical performance on standard datasets:

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| Naive Bayes | 82-85% | 0.81-0.84 | Fast |
| Logistic Regression | 85-88% | 0.84-0.87 | Fast |
| Linear SVM | 86-89% | 0.85-0.88 | Medium |
| Random Forest | 83-87% | 0.82-0.86 | Medium |
| Gradient Boosting | 85-89% | 0.84-0.88 | Slow |
| Ensemble | 87-91% | 0.86-0.90 | Medium |
| Deep Neural Network | 88-92% | 0.87-0.91 | Slow |
| Transformer (BERT) | 91-95% | 0.90-0.94 | Very Slow |

## Architecture

### Traditional ML Pipeline
1. Text preprocessing (lowercase, remove URLs, tokenization)
2. Stopword removal and lemmatization
3. TF-IDF or Count vectorization with n-grams
4. Train multiple classifiers with optimized hyperparameters
5. Ensemble the best models

### Deep Learning Architecture
- Input Layer: TF-IDF features
- Hidden Layer 1: 512 neurons + ReLU + BatchNorm + Dropout(0.5)
- Hidden Layer 2: 256 neurons + ReLU + BatchNorm + Dropout(0.5)
- Hidden Layer 3: 128 neurons + ReLU + BatchNorm + Dropout(0.5)
- Output Layer: num_classes neurons
- Optimizer: Adam with weight decay
- Loss: CrossEntropyLoss

### Transformer Pipeline
1. Minimal preprocessing (preserve context)
2. Transformer tokenization (WordPiece/BPE)
3. Fine-tune pre-trained model (DistilBERT, BERT, RoBERTa)
4. Automatic hyperparameter optimization

## Input Data Format

CSV file with at least two columns:

```csv
text,label
"This is a positive review",positive
"This product is terrible",negative
"Amazing experience!",positive
```

## Examples

### Binary Sentiment Classification
```bash
python text_classifier.py \
    --data movie_reviews.csv \
    --text-col review \
    --label-col sentiment \
    --deep-learning \
    --output sentiment_results.png
```

### Multi-class Topic Classification
```bash
python text_classifier.py \
    --data news_articles.csv \
    --text-col article \
    --label-col category \
    --vectorizer tfidf \
    --max-features 10000
```

### Spam Detection
```bash
python text_classifier.py \
    --data emails.csv \
    --text-col message \
    --label-col is_spam \
    --save-model spam_detector.pkl
```

## Use Cases

- Sentiment analysis (positive/negative/neutral)
- Spam detection
- Topic classification
- Product review analysis
- Customer feedback categorization
- News article classification
- Intent recognition
- Language detection

## Visualization

The system generates comprehensive visualizations:

1. **Model Comparison:** Bar charts comparing accuracy and F1 scores
2. **Precision vs Recall:** Scatter plot showing trade-offs
3. **Metrics Heatmap:** All metrics in one view
4. **Confusion Matrices:** Per-model confusion matrices

## Performance Optimization

### For Speed:
- Use traditional ML models (Logistic Regression, Linear SVM)
- Reduce `max_features` to 1000-2000
- Use Count vectorizer instead of TF-IDF

### For Accuracy:
- Use Transformer models with `--transformer`
- Increase `max_features` to 10000+
- Enable deep learning with `--deep-learning`
- Use ensemble methods

### For Large Datasets:
- Use GPU acceleration (automatically detected)
- Batch processing in DataLoader
- Use DistilBERT instead of full BERT

## Best Practices

1. **Data Quality:** Clean and balance your dataset
2. **Feature Engineering:** Experiment with max_features and n-grams
3. **Model Selection:** Start with Logistic Regression, then try ensemble
4. **Validation:** Use cross-validation for small datasets
5. **GPU Usage:** Enable GPU for deep learning and transformers
6. **Preprocessing:** Keep preprocessing consistent between training and inference

## Troubleshooting

**Low accuracy:**
- Check class balance
- Increase max_features
- Try different preprocessing
- Use ensemble or deep learning

**Out of memory:**
- Reduce batch_size
- Use DistilBERT instead of BERT
- Reduce max_features
- Process in smaller batches

**Slow training:**
- Use GPU acceleration
- Reduce epochs for deep learning
- Use traditional ML first
- Sample your dataset

## Version History

- **v2.0:** Added Transformers, Deep Learning, Ensemble methods, GPU support, comprehensive metrics
- **v1.0:** Basic text classification with traditional ML

---

**BrillConsulting** - Advanced NLP Solutions
