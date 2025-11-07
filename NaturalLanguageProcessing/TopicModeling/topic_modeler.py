"""
Advanced Topic Modeling System v2.0
Author: BrillConsulting
Description: Discover topics in documents using LDA, NMF, and BERTopic

Supports traditional methods (LDA, NMF) and modern transformer-based approaches (BERTopic)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies
try:
    import pyLDAvis
    import pyLDAvis.lda_model
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False
    print("Warning: pyLDAvis not available. Install with: pip install pyldavis")

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("Warning: BERTopic not available. Install with: pip install bertopic")

try:
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Warning: gensim not available for coherence calculation")


class TopicModeler:
    """
    Advanced Topic Modeling System

    Supports:
    - LDA (Latent Dirichlet Allocation)
    - NMF (Non-negative Matrix Factorization)
    - BERTopic (Transformer-based topic modeling)
    """

    def __init__(self, n_topics=5, method='lda', random_state=42):
        """
        Initialize topic modeler

        Args:
            n_topics: Number of topics to discover
            method: 'lda', 'nmf', or 'bertopic'
            random_state: Random seed for reproducibility
        """
        self.n_topics = n_topics
        self.method = method
        self.random_state = random_state

        # Components
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.doc_term_matrix = None
        self.documents = []

        # BERTopic
        self.bertopic_model = None
        self.topics = None
        self.probs = None

        print(f"✓ TopicModeler initialized (method={method}, n_topics={n_topics})")

    def fit(self, documents: List[str], max_features=1000):
        """
        Fit topic model on documents

        Args:
            documents: List of text documents
            max_features: Maximum vocabulary size
        """
        self.documents = documents
        print(f"\n🔧 Training {self.method.upper()} with {self.n_topics} topics on {len(documents)} documents...")

        if self.method == 'bertopic':
            return self._fit_bertopic(documents)
        else:
            return self._fit_traditional(documents, max_features)

    def _fit_traditional(self, documents: List[str], max_features=1000):
        """Fit LDA or NMF model"""
        # Vectorize
        if self.method == 'lda':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
        else:  # NMF
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )

        self.doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Fit model
        if self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=20,
                learning_method='online'
            )
        else:  # NMF
            self.model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=200,
                init='nndsvda'
            )

        self.model.fit(self.doc_term_matrix)
        print("✅ Model trained!")

        return self

    def _fit_bertopic(self, documents: List[str]):
        """Fit BERTopic model"""
        if not BERTOPIC_AVAILABLE:
            raise ImportError("BERTopic not available")

        self.bertopic_model = BERTopic(
            nr_topics=self.n_topics if self.n_topics else 'auto',
            calculate_probabilities=True
        )

        self.topics, self.probs = self.bertopic_model.fit_transform(documents)
        print("✅ BERTopic model trained!")

        return self

    def fit_transform(self, documents: List[str], max_features=1000) -> Dict:
        """
        Fit model and return comprehensive results

        Returns:
            Dict with topics, document-topic distributions, and metadata
        """
        self.fit(documents, max_features)

        result = {
            'topics': {},
            'document_topics': [],
            'method': self.method,
            'n_topics': self.n_topics,
            'n_documents': len(documents)
        }

        # Get topics
        if self.method == 'bertopic':
            topic_info = self.bertopic_model.get_topic_info()
            for topic_id in topic_info['Topic']:
                if topic_id != -1:  # Exclude outliers
                    words = [word for word, _ in self.bertopic_model.get_topic(topic_id)[:10]]
                    result['topics'][topic_id] = words
        else:
            topics = self.get_top_words(n_words=10)
            for idx, words in enumerate(topics):
                result['topics'][idx] = words

        # Get document-topic distributions
        if self.method == 'bertopic':
            for doc_id, topic_id in enumerate(self.topics):
                if self.probs is not None:
                    result['document_topics'].append({topic_id: float(self.probs[doc_id][topic_id])})
                else:
                    result['document_topics'].append({topic_id: 1.0})
        else:
            doc_topics = self.transform(documents)
            for doc_topic_dist in doc_topics:
                topic_dict = {i: float(prob) for i, prob in enumerate(doc_topic_dist)}
                result['document_topics'].append(topic_dict)

        return result

    def get_top_words(self, n_words=10) -> List[List[str]]:
        """
        Get top words for each topic

        Args:
            n_words: Number of top words per topic

        Returns:
            List of word lists for each topic
        """
        if self.method == 'bertopic':
            topics = []
            topic_info = self.bertopic_model.get_topic_info()
            for topic_id in topic_info['Topic']:
                if topic_id != -1:
                    words = [word for word, _ in self.bertopic_model.get_topic(topic_id)[:n_words]]
                    topics.append(words)
            return topics
        else:
            topics = []
            for topic_idx, topic in enumerate(self.model.components_):
                top_indices = topic.argsort()[-n_words:][::-1]
                top_words = [self.feature_names[i] for i in top_indices]
                topics.append(top_words)
            return topics

    def print_topics(self, n_words=10):
        """Print top words for each topic"""
        topics = self.get_top_words(n_words)

        print(f"\n📚 Discovered {len(topics)} Topics:")
        print("=" * 80)

        for idx, words in enumerate(topics):
            print(f"\nTopic {idx}: {', '.join(words)}")

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Get topic distributions for documents

        Args:
            documents: List of text documents

        Returns:
            Document-topic matrix
        """
        if self.method == 'bertopic':
            topics, probs = self.bertopic_model.transform(documents)
            return probs
        else:
            doc_term_matrix = self.vectorizer.transform(documents)
            return self.model.transform(doc_term_matrix)

    def calculate_coherence(self, documents: List[str], coherence_type='c_v') -> float:
        """
        Calculate topic coherence score

        Args:
            documents: Training documents
            coherence_type: 'c_v', 'u_mass', etc.

        Returns:
            Coherence score (higher is better)
        """
        if not GENSIM_AVAILABLE:
            print("Warning: gensim not available for coherence calculation")
            return 0.0

        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]

        # Create dictionary and corpus
        dictionary = Dictionary(tokenized_docs)

        # Get topics as list of words
        topics = self.get_top_words(n_words=10)

        # Calculate coherence
        coherence_model = CoherenceModel(
            topics=topics,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence=coherence_type
        )

        coherence_score = coherence_model.get_coherence()
        return coherence_score

    def plot_topics(self, save_path=None):
        """Visualize topic word clouds"""
        topics = self.get_top_words(10)

        n_cols = min(3, len(topics))
        n_rows = (len(topics) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

        if len(topics) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (ax, words) in enumerate(zip(axes, topics)):
            y_pos = np.arange(len(words))
            ax.barh(y_pos, range(len(words), 0, -1), color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'Topic {idx}')

        # Hide empty subplots
        for idx in range(len(topics), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Topics plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_topic_distribution(self, documents: List[str], save_path=None):
        """Plot document-topic distribution"""
        doc_topics = self.transform(documents)

        # Get dominant topic for each document
        if self.method == 'bertopic':
            dominant_topics = [self.bertopic_model.topics_[i] for i in range(len(documents))]
        else:
            dominant_topics = np.argmax(doc_topics, axis=1)

        # Count documents per topic
        from collections import Counter
        topic_counts = Counter(dominant_topics)

        plt.figure(figsize=(10, 6))
        topics_list = sorted(topic_counts.keys())
        counts = [topic_counts[t] for t in topics_list]

        plt.bar(topics_list, counts, color='steelblue')
        plt.xlabel('Topic ID')
        plt.ylabel('Number of Documents')
        plt.title('Document Distribution Across Topics')
        plt.xticks(topics_list)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Distribution plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_topics_interactive(self, save_path='topics_interactive.html'):
        """Create interactive LDA visualization (pyLDAvis)"""
        if not PYLDAVIS_AVAILABLE:
            print("Warning: pyLDAvis not available")
            return

        if self.method != 'lda':
            print("Interactive visualization only available for LDA")
            return

        # Prepare visualization
        vis = pyLDAvis.lda_model.prepare(
            self.model,
            self.doc_term_matrix,
            self.vectorizer,
            mds='tsne'
        )

        # Save to HTML
        pyLDAvis.save_html(vis, save_path)
        print(f"📊 Interactive visualization saved to {save_path}")

    def build_topic_hierarchy(self, n_super_topics=3) -> Dict[int, List[int]]:
        """
        Build hierarchical topic structure

        Args:
            n_super_topics: Number of super-topics

        Returns:
            Dict mapping super-topic ID to list of topic IDs
        """
        if self.method == 'bertopic' and BERTOPIC_AVAILABLE:
            # Use BERTopic's hierarchical topics
            hierarchical_topics = self.bertopic_model.hierarchical_topics(self.documents)
            return hierarchical_topics
        else:
            # Simple clustering of topics based on similarity
            from sklearn.cluster import KMeans

            # Get topic-word distributions
            topic_word_dist = self.model.components_

            # Cluster topics
            kmeans = KMeans(n_clusters=n_super_topics, random_state=self.random_state)
            super_topics = kmeans.fit_predict(topic_word_dist)

            # Group topics by super-topic
            hierarchy = {}
            for topic_id, super_topic_id in enumerate(super_topics):
                if super_topic_id not in hierarchy:
                    hierarchy[super_topic_id] = []
                hierarchy[super_topic_id].append(topic_id)

            return hierarchy


def find_optimal_topics(documents: List[str], topic_range=range(2, 21), method='lda') -> List[float]:
    """
    Find optimal number of topics using coherence scores

    Args:
        documents: List of documents
        topic_range: Range of topic numbers to try
        method: Topic modeling method

    Returns:
        List of coherence scores
    """
    coherence_scores = []

    for n_topics in topic_range:
        print(f"Trying {n_topics} topics...")

        modeler = TopicModeler(n_topics=n_topics, method=method)
        modeler.fit(documents)

        coherence = modeler.calculate_coherence(documents)
        coherence_scores.append(coherence)

        print(f"  Coherence: {coherence:.4f}")

    return coherence_scores


def main():
    parser = argparse.ArgumentParser(description='Advanced Topic Modeling')
    parser.add_argument('--data', type=str, required=True,
                       help='CSV file with text column')
    parser.add_argument('--text-col', type=str, default='text',
                       help='Name of text column')
    parser.add_argument('--n-topics', type=int, default=5,
                       help='Number of topics')
    parser.add_argument('--method', type=str, default='lda',
                       choices=['lda', 'nmf', 'bertopic'],
                       help='Topic modeling method')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--output', type=str,
                       help='Output plot path')
    parser.add_argument('--coherence', action='store_true',
                       help='Calculate coherence score')

    args = parser.parse_args()

    # Load data
    print(f"📂 Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    documents = df[args.text_col].tolist()

    print(f"📊 Dataset: {len(documents)} documents\n")

    # Initialize and fit
    modeler = TopicModeler(n_topics=args.n_topics, method=args.method)
    result = modeler.fit_transform(documents)

    # Print topics
    modeler.print_topics(n_words=10)

    # Coherence
    if args.coherence:
        print("\n📈 Calculating coherence score...")
        coherence = modeler.calculate_coherence(documents)
        print(f"Coherence Score: {coherence:.4f}")

    # Visualize
    if args.visualize:
        if args.output:
            modeler.plot_topics(save_path=args.output)
        else:
            modeler.plot_topics()

        # Interactive visualization for LDA
        if args.method == 'lda' and PYLDAVIS_AVAILABLE:
            modeler.visualize_topics_interactive('topics_interactive.html')


if __name__ == "__main__":
    # Check if running with arguments
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Demo mode
        print("=" * 80)
        print("Advanced Topic Modeling System v2.0 - Demo")
        print("=" * 80)

        # Sample documents
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing enables computers to understand human language",
            "Computer vision allows machines to interpret visual information",
            "Reinforcement learning trains agents through rewards and penalties",
            "Data science combines statistics, programming, and domain expertise",
            "Big data analytics processes large datasets for insights",
            "Cloud computing provides on-demand computing resources",
            "Cybersecurity protects systems from digital attacks",
            "Blockchain is a distributed ledger technology",
        ]

        print(f"\n📊 Sample Dataset: {len(documents)} documents\n")

        # LDA
        print("\n" + "=" * 80)
        print("Method: LDA (Latent Dirichlet Allocation)")
        print("=" * 80)

        modeler = TopicModeler(n_topics=3, method='lda')
        result = modeler.fit_transform(documents)
        modeler.print_topics(n_words=5)

        print("\n✓ Demo completed!")
