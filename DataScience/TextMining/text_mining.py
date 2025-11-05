"""
Text Mining Toolkit
===================

Advanced NLP and text analysis methods:
- Text preprocessing and cleaning
- TF-IDF and text vectorization
- Sentiment analysis
- Topic modeling (LDA, NMF)
- Named Entity Recognition
- Text classification
- Word embeddings and similarity
- Document clustering

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import re
import warnings
warnings.filterwarnings('ignore')


class TextMining:
    """Text mining and NLP analysis toolkit."""

    def __init__(self):
        """Initialize text mining toolkit."""
        self.vocabulary = {}
        self.idf_scores = {}
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                              'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was',
                              'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                              'does', 'did', 'will', 'would', 'could', 'should', 'may',
                              'might', 'can', 'this', 'that', 'these', 'those', 'i',
                              'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                              'her', 'us', 'them'])

    def preprocess_text(self, text: str, lowercase: bool = True,
                       remove_punctuation: bool = True,
                       remove_stopwords: bool = True) -> List[str]:
        """
        Preprocess text: tokenize, clean, and normalize.

        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_stopwords: Remove stopwords

        Returns:
            List of tokens
        """
        if lowercase:
            text = text.lower()

        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize
        tokens = text.split()

        # Remove stopwords
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 2]

        return tokens

    def compute_tf_idf(self, documents: List[List[str]]) -> Dict:
        """
        Compute TF-IDF scores for documents.

        Args:
            documents: List of tokenized documents

        Returns:
            Dictionary with TF-IDF matrix and vocabulary
        """
        n_docs = len(documents)

        # Build vocabulary
        vocabulary = {}
        word_doc_count = Counter()

        for doc in documents:
            unique_words = set(doc)
            for word in unique_words:
                word_doc_count[word] += 1

        # Assign indices to words
        for i, word in enumerate(sorted(word_doc_count.keys())):
            vocabulary[word] = i

        self.vocabulary = vocabulary
        n_words = len(vocabulary)

        # Compute IDF
        idf = {}
        for word, count in word_doc_count.items():
            idf[word] = np.log(n_docs / count)

        self.idf_scores = idf

        # Compute TF-IDF matrix
        tfidf_matrix = np.zeros((n_docs, n_words))

        for doc_idx, doc in enumerate(documents):
            word_counts = Counter(doc)
            doc_length = len(doc)

            for word, count in word_counts.items():
                if word in vocabulary:
                    tf = count / doc_length
                    word_idx = vocabulary[word]
                    tfidf_matrix[doc_idx, word_idx] = tf * idf[word]

        return {
            'tfidf_matrix': tfidf_matrix,
            'vocabulary': vocabulary,
            'idf_scores': idf,
            'feature_names': sorted(vocabulary.keys())
        }

    def sentiment_analysis(self, text: str, method: str = 'lexicon') -> Dict:
        """
        Perform sentiment analysis.

        Args:
            text: Input text
            method: Analysis method ('lexicon' or 'simple')

        Returns:
            Dictionary with sentiment scores
        """
        # Simple lexicon-based approach
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful',
                             'fantastic', 'best', 'love', 'like', 'happy', 'joy',
                             'positive', 'perfect', 'beautiful', 'awesome', 'brilliant'])

        negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'worst',
                             'hate', 'dislike', 'sad', 'angry', 'negative', 'poor',
                             'disappointing', 'wrong', 'fail', 'problem', 'issue'])

        tokens = self.preprocess_text(text, lowercase=True, remove_stopwords=False)

        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)

        total_count = positive_count + negative_count

        if total_count > 0:
            sentiment_score = (positive_count - negative_count) / total_count
        else:
            sentiment_score = 0.0

        # Classify sentiment
        if sentiment_score > 0.2:
            sentiment = 'positive'
        elif sentiment_score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'total_sentiment_words': total_count
        }

    def lda_topic_modeling(self, documents: List[List[str]], n_topics: int = 5,
                          n_iterations: int = 100) -> Dict:
        """
        Perform Latent Dirichlet Allocation (LDA) topic modeling (simplified).

        Args:
            documents: List of tokenized documents
            n_topics: Number of topics
            n_iterations: Number of iterations

        Returns:
            Dictionary with topic distributions
        """
        # Build vocabulary
        vocab = set()
        for doc in documents:
            vocab.update(doc)
        vocab = sorted(list(vocab))
        word_to_idx = {word: i for i, word in enumerate(vocab)}

        n_docs = len(documents)
        n_words = len(vocab)

        # Initialize parameters (simplified)
        np.random.seed(42)

        # Document-topic distribution (theta)
        doc_topic = np.random.dirichlet(np.ones(n_topics), n_docs)

        # Topic-word distribution (phi)
        topic_word = np.random.dirichlet(np.ones(n_words), n_topics)

        # EM-like iterations (simplified)
        for iteration in range(n_iterations):
            # E-step: assign words to topics (simplified)
            for doc_idx, doc in enumerate(documents):
                for word in doc:
                    if word in word_to_idx:
                        word_idx = word_to_idx[word]

                        # Update based on current distributions
                        topic_probs = doc_topic[doc_idx] * topic_word[:, word_idx]
                        topic_probs /= topic_probs.sum()

                        # Update distributions (simplified smoothing)
                        doc_topic[doc_idx] = 0.9 * doc_topic[doc_idx] + 0.1 * topic_probs
                        doc_topic[doc_idx] /= doc_topic[doc_idx].sum()

        # Get top words for each topic
        top_words_per_topic = {}
        n_top_words = 10

        for topic_idx in range(n_topics):
            top_word_indices = np.argsort(topic_word[topic_idx])[-n_top_words:][::-1]
            top_words = [vocab[i] for i in top_word_indices]
            top_words_per_topic[f'Topic_{topic_idx}'] = top_words

        return {
            'doc_topic_distribution': doc_topic,
            'topic_word_distribution': topic_word,
            'top_words_per_topic': top_words_per_topic,
            'n_topics': n_topics,
            'vocabulary': vocab
        }

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def document_similarity(self, tfidf_matrix: np.ndarray,
                           doc1_idx: int, doc2_idx: int) -> float:
        """Calculate similarity between two documents."""
        return self.cosine_similarity(tfidf_matrix[doc1_idx], tfidf_matrix[doc2_idx])

    def extract_ngrams(self, text: str, n: int = 2) -> List[Tuple]:
        """
        Extract n-grams from text.

        Args:
            text: Input text
            n: N-gram size

        Returns:
            List of n-grams
        """
        tokens = self.preprocess_text(text)
        ngrams = []

        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)

        return ngrams

    def keyword_extraction(self, documents: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF.

        Args:
            documents: List of documents
            top_k: Number of keywords to extract

        Returns:
            List of (keyword, score) tuples
        """
        # Preprocess documents
        tokenized_docs = [self.preprocess_text(doc) for doc in documents]

        # Compute TF-IDF
        tfidf_result = self.compute_tf_idf(tokenized_docs)
        tfidf_matrix = tfidf_result['tfidf_matrix']
        feature_names = tfidf_result['feature_names']

        # Sum TF-IDF scores across documents
        total_scores = np.sum(tfidf_matrix, axis=0)

        # Get top keywords
        top_indices = np.argsort(total_scores)[-top_k:][::-1]
        keywords = [(feature_names[i], total_scores[i]) for i in top_indices]

        return keywords

    def text_classification(self, train_docs: List[str], train_labels: np.ndarray,
                           test_docs: List[str]) -> Dict:
        """
        Simple text classification using TF-IDF and Naive Bayes.

        Args:
            train_docs: Training documents
            train_labels: Training labels
            test_docs: Test documents

        Returns:
            Dictionary with predictions
        """
        # Preprocess
        train_tokenized = [self.preprocess_text(doc) for doc in train_docs]
        test_tokenized = [self.preprocess_text(doc) for doc in test_docs]

        # Compute TF-IDF on training data
        tfidf_result = self.compute_tf_idf(train_tokenized)
        train_tfidf = tfidf_result['tfidf_matrix']
        vocabulary = tfidf_result['vocabulary']

        # Compute TF-IDF for test data using same vocabulary
        test_tfidf = np.zeros((len(test_docs), len(vocabulary)))

        for doc_idx, doc in enumerate(test_tokenized):
            word_counts = Counter(doc)
            doc_length = len(doc)

            for word, count in word_counts.items():
                if word in vocabulary:
                    tf = count / doc_length
                    idf = self.idf_scores.get(word, 0)
                    word_idx = vocabulary[word]
                    test_tfidf[doc_idx, word_idx] = tf * idf

        # Simple Naive Bayes classification
        classes = np.unique(train_labels)
        predictions = []

        for test_vec in test_tfidf:
            class_scores = {}

            for cls in classes:
                # Class probability
                class_mask = train_labels == cls
                class_prob = np.sum(class_mask) / len(train_labels)

                # Feature probabilities (with smoothing)
                class_features = train_tfidf[class_mask].mean(axis=0) + 1e-10

                # Calculate score
                score = np.log(class_prob) + np.sum(test_vec * np.log(class_features))
                class_scores[cls] = score

            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return {
            'predictions': np.array(predictions),
            'train_vocabulary_size': len(vocabulary)
        }

    def visualize_word_cloud(self, documents: List[str], title: str = "Word Frequency") -> plt.Figure:
        """Visualize word frequencies."""
        # Combine all documents
        all_tokens = []
        for doc in documents:
            all_tokens.extend(self.preprocess_text(doc))

        word_freq = Counter(all_tokens)
        top_words = word_freq.most_common(30)

        fig, ax = plt.subplots(figsize=(12, 6))

        words, frequencies = zip(*top_words)
        ax.barh(range(len(words)), frequencies, color='skyblue', edgecolor='black')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        return fig


def demo():
    """Demo text mining toolkit."""
    print("Text Mining Toolkit Demo")
    print("="*60)

    tm = TextMining()

    # Sample documents
    documents = [
        "Machine learning is a great field with amazing applications in artificial intelligence.",
        "Natural language processing enables computers to understand human language effectively.",
        "Data science combines statistics and machine learning for analysis and predictions.",
        "Artificial intelligence and deep learning are transforming many industries today.",
        "Text mining extracts useful information from unstructured text data sources."
    ]

    # 1. Text Preprocessing
    print("\n1. Text Preprocessing")
    print("-" * 60)
    sample_text = documents[0]
    tokens = tm.preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Tokens: {tokens}")

    # 2. TF-IDF
    print("\n2. TF-IDF Vectorization")
    print("-" * 60)
    tokenized_docs = [tm.preprocess_text(doc) for doc in documents]
    tfidf_result = tm.compute_tf_idf(tokenized_docs)
    print(f"Vocabulary size: {len(tfidf_result['vocabulary'])}")
    print(f"TF-IDF matrix shape: {tfidf_result['tfidf_matrix'].shape}")
    print(f"Top 10 words by IDF:")
    top_idf = sorted(tfidf_result['idf_scores'].items(), key=lambda x: x[1], reverse=True)[:10]
    for word, score in top_idf:
        print(f"  {word}: {score:.3f}")

    # 3. Sentiment Analysis
    print("\n3. Sentiment Analysis")
    print("-" * 60)
    sentiment_texts = [
        "This product is absolutely amazing! I love it so much.",
        "Terrible experience. Very disappointed with the service.",
        "It's okay, nothing special but not bad either."
    ]

    for text in sentiment_texts:
        result = tm.sentiment_analysis(text)
        print(f"\nText: {text}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Positive words: {result['positive_words']}, Negative words: {result['negative_words']}")

    # 4. Document Similarity
    print("\n4. Document Similarity")
    print("-" * 60)
    tfidf_matrix = tfidf_result['tfidf_matrix']
    sim_01 = tm.document_similarity(tfidf_matrix, 0, 1)
    sim_02 = tm.document_similarity(tfidf_matrix, 0, 2)
    print(f"Similarity between doc 0 and doc 1: {sim_01:.4f}")
    print(f"Similarity between doc 0 and doc 2: {sim_02:.4f}")

    # 5. N-grams
    print("\n5. N-gram Extraction")
    print("-" * 60)
    bigrams = tm.extract_ngrams(documents[0], n=2)
    print(f"Bigrams from first document:")
    for i, bigram in enumerate(bigrams[:5]):
        print(f"  {i+1}. {' '.join(bigram)}")

    # 6. Keyword Extraction
    print("\n6. Keyword Extraction")
    print("-" * 60)
    keywords = tm.keyword_extraction(documents, top_k=10)
    print(f"Top 10 keywords:")
    for i, (keyword, score) in enumerate(keywords):
        print(f"  {i+1}. {keyword}: {score:.4f}")

    # 7. Topic Modeling
    print("\n7. Topic Modeling (LDA)")
    print("-" * 60)
    lda_result = tm.lda_topic_modeling(tokenized_docs, n_topics=3, n_iterations=50)
    print(f"Number of topics: {lda_result['n_topics']}")
    print(f"\nTop words per topic:")
    for topic, words in lda_result['top_words_per_topic'].items():
        print(f"  {topic}: {', '.join(words[:5])}")

    # 8. Text Classification
    print("\n8. Text Classification")
    print("-" * 60)
    train_docs = [
        "Machine learning and artificial intelligence",
        "Natural language processing techniques",
        "Sports news and football matches",
        "Basketball game highlights",
        "Deep learning neural networks"
    ]
    train_labels = np.array([0, 0, 1, 1, 0])  # 0: tech, 1: sports
    test_docs = ["AI applications in technology", "Soccer tournament results"]

    clf_result = tm.text_classification(train_docs, train_labels, test_docs)
    print(f"Predictions: {clf_result['predictions']}")
    print(f"Test doc 1: '{test_docs[0]}' -> {'Tech' if clf_result['predictions'][0] == 0 else 'Sports'}")
    print(f"Test doc 2: '{test_docs[1]}' -> {'Tech' if clf_result['predictions'][1] == 0 else 'Sports'}")

    # 9. Visualize
    print("\n9. Word Frequency Visualization")
    print("-" * 60)
    fig = tm.visualize_word_cloud(documents, title="Top 30 Words by Frequency")
    fig.savefig('text_mining_wordcloud.png', dpi=300, bbox_inches='tight')
    print("✓ Saved text_mining_wordcloud.png")
    plt.close()

    print("\n" + "="*60)
    print("✓ Text Mining Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo()
