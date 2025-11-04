"""
Topic Modeling System
Author: BrillConsulting
Description: Discover topics in documents using LDA and NMF
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.lda_model
import argparse
from typing import List


class TopicModeler:
    """Topic modeling with LDA and NMF"""

    def __init__(self, n_topics=5, method='lda'):
        """
        Initialize topic modeler

        Args:
            n_topics: Number of topics to discover
            method: 'lda' or 'nmf'
        """
        self.n_topics = n_topics
        self.method = method
        self.vectorizer = None
        self.model = None
        self.feature_names = None

    def fit(self, documents: List[str], max_features=1000):
        """
        Fit topic model on documents

        Args:
            documents: List of text documents
            max_features: Maximum vocabulary size
        """
        print(f"🔧 Training {self.method.upper()} with {self.n_topics} topics...\n")

        # Vectorize
        if self.method == 'lda':
            self.vectorizer = CountVectorizer(max_features=max_features,
                                             stop_words='english')
        else:  # NMF
            self.vectorizer = TfidfVectorizer(max_features=max_features,
                                             stop_words='english')

        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Fit model
        if self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20
            )
        else:  # NMF
            self.model = NMF(
                n_components=self.n_topics,
                random_state=42,
                max_iter=200
            )

        self.model.fit(doc_term_matrix)

        print("✅ Model trained!\n")

    def get_top_words(self, n_words=10) -> List[List[str]]:
        """Get top words for each topic"""
        topics = []

        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_indices]
            topics.append(top_words)

        return topics

    def print_topics(self, n_words=10):
        """Print top words for each topic"""
        topics = self.get_top_words(n_words)

        print(f"📚 Discovered {self.n_topics} Topics:\n")

        for idx, words in enumerate(topics):
            print(f"Topic {idx + 1}: {', '.join(words)}")

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Get topic distributions for documents

        Args:
            documents: List of text documents

        Returns:
            Document-topic matrix
        """
        doc_term_matrix = self.vectorizer.transform(documents)
        return self.model.transform(doc_term_matrix)

    def plot_topics(self, save_path=None):
        """Visualize topics"""
        topics = self.get_top_words(10)

        fig, axes = plt.subplots(1, self.n_topics, figsize=(5*self.n_topics, 4))

        if self.n_topics == 1:
            axes = [axes]

        for idx, (ax, words) in enumerate(zip(axes, topics)):
            ax.barh(range(len(words)), range(len(words), 0, -1))
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.set_title(f'Topic {idx + 1}')
            ax.set_xlabel('Importance')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Topics plot saved to {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Topic Modeling')
    parser.add_argument('--data', type=str, required=True,
                       help='CSV file with text column')
    parser.add_argument('--text-col', type=str, default='text',
                       help='Name of text column')
    parser.add_argument('--n-topics', type=int, default=5,
                       help='Number of topics')
    parser.add_argument('--method', type=str, default='lda',
                       choices=['lda', 'nmf'],
                       help='Topic modeling method')
    parser.add_argument('--output', type=str,
                       help='Output plot path')

    args = parser.parse_args()

    # Load data
    print(f"📂 Loading data from {args.data}...\n")
    df = pd.read_csv(args.data)
    documents = df[args.text_col].tolist()

    print(f"📊 Dataset: {len(documents)} documents\n")

    # Initialize and fit
    modeler = TopicModeler(n_topics=args.n_topics, method=args.method)
    modeler.fit(documents)

    # Print topics
    modeler.print_topics(n_words=10)

    # Plot
    if args.output:
        modeler.plot_topics(save_path=args.output)


if __name__ == "__main__":
    main()
