# Text Mining Toolkit

Advanced NLP and text analysis methods for text preprocessing, sentiment analysis, topic modeling, and text classification.

## Overview

The Text Mining Toolkit provides comprehensive methods for analyzing unstructured text data. It implements text preprocessing, TF-IDF vectorization, sentiment analysis, topic modeling (LDA), keyword extraction, and text classification.

## Key Features

- **Text Preprocessing**: Tokenization, cleaning, stopword removal, normalization
- **TF-IDF Vectorization**: Convert text to numerical features
- **Sentiment Analysis**: Lexicon-based sentiment classification
- **Topic Modeling (LDA)**: Discover latent topics in documents
- **Document Similarity**: Cosine similarity between text documents
- **N-gram Extraction**: Extract bigrams, trigrams, and higher-order n-grams
- **Keyword Extraction**: Identify important terms using TF-IDF
- **Text Classification**: Naive Bayes classifier for document categorization
- **Visualization**: Word frequency charts and text analytics

## Technologies Used

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Sparse matrix operations and clustering
- **Matplotlib & Seaborn**: Visualization

## Installation

```bash
cd TextMining/
pip install numpy pandas scipy matplotlib seaborn
```

## Usage Examples

### Text Preprocessing

```python
from text_mining import TextMining

tm = TextMining()

# Preprocess text
tokens = tm.preprocess_text(
    text,
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True
)
print(tokens)
```

### TF-IDF Vectorization

```python
# Compute TF-IDF
documents = [tm.preprocess_text(doc) for doc in raw_documents]
tfidf_result = tm.compute_tf_idf(documents)

print(f"Vocabulary size: {len(tfidf_result['vocabulary'])}")
print(f"TF-IDF matrix shape: {tfidf_result['tfidf_matrix'].shape}")
```

### Sentiment Analysis

```python
# Analyze sentiment
result = tm.sentiment_analysis(text, method='lexicon')

print(f"Sentiment: {result['sentiment']}")
print(f"Score: {result['score']:.3f}")
print(f"Positive words: {result['positive_words']}")
print(f"Negative words: {result['negative_words']}")
```

### Topic Modeling

```python
# Discover topics using LDA
lda_result = tm.lda_topic_modeling(documents, n_topics=5, n_iterations=100)

print(f"Number of topics: {lda_result['n_topics']}")
for topic, words in lda_result['top_words_per_topic'].items():
    print(f"{topic}: {', '.join(words[:5])}")
```

### Text Classification

```python
# Train classifier
clf_result = tm.text_classification(
    train_docs=train_documents,
    train_labels=train_labels,
    test_docs=test_documents
)

print(f"Predictions: {clf_result['predictions']}")
```

## Demo

```bash
python text_mining.py
```

The demo includes:
- Text preprocessing and tokenization
- TF-IDF vectorization
- Sentiment analysis on multiple texts
- Document similarity computation
- N-gram extraction
- Keyword extraction
- Topic modeling (LDA)
- Text classification
- Word frequency visualization

## Output Examples

- `text_mining_wordcloud.png`: Top 30 words by frequency
- Console output with sentiment scores, topics, and classification results

## Key Concepts

**TF-IDF**: Term Frequency-Inverse Document Frequency weights terms by importance

**Sentiment Analysis**: Classify text as positive, negative, or neutral

**Topic Modeling**: Discover hidden thematic structure in documents

**N-grams**: Contiguous sequences of n words for capturing phrases

## Applications

- Social media monitoring and analysis
- Customer review analysis
- Document classification and organization
- Content recommendation
- Market research and opinion mining
- Chatbot training data analysis

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
