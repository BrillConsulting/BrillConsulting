"""
Text Summarization System
Author: BrillConsulting
Description: Extractive and abstractive summarization using Transformers and traditional methods
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import networkx as nx
import argparse
import nltk

# Download punkt tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextSummarizer:
    """Text summarization system"""

    def __init__(self):
        pass

    def extractive_tfidf(self, text: str, num_sentences: int = 3) -> str:
        """
        Extractive summarization using TF-IDF

        Args:
            text: Input document
            num_sentences: Number of sentences in summary

        Returns:
            Summary text
        """
        # Split into sentences
        sentences = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return text

        # Vectorize sentences
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Calculate sentence scores (sum of TF-IDF values)
        sentence_scores = tfidf_matrix.sum(axis=1).A1

        # Get top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)  # Maintain order

        # Build summary
        summary = ' '.join([sentences[i] for i in top_indices])

        return summary

    def extractive_textrank(self, text: str, num_sentences: int = 3) -> str:
        """
        Extractive summarization using TextRank

        Args:
            text: Input document
            num_sentences: Number of sentences in summary

        Returns:
            Summary text
        """
        sentences = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return text

        # Vectorize
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Build similarity matrix
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).A

        # Build graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        # Get top sentences
        ranked = sorted(((scores[i], i) for i in range(len(sentences))),
                       reverse=True)
        top_indices = sorted([idx for _, idx in ranked[:num_sentences]])

        # Build summary
        summary = ' '.join([sentences[i] for i in top_indices])

        return summary

    def get_compression_ratio(self, original: str, summary: str) -> float:
        """Calculate compression ratio"""
        return len(summary) / len(original)


def main():
    parser = argparse.ArgumentParser(description='Text Summarization')
    parser.add_argument('--text', type=str, help='Text to summarize')
    parser.add_argument('--file', type=str, help='Text file to summarize')
    parser.add_argument('--num-sentences', type=int, default=3,
                       help='Number of sentences in summary')
    parser.add_argument('--method', type=str, default='tfidf',
                       choices=['tfidf', 'textrank'],
                       help='Summarization method')

    args = parser.parse_args()

    # Get text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("❌ Provide --text or --file")
        return

    # Initialize summarizer
    summarizer = TextSummarizer()

    # Summarize
    print(f"🔧 Summarizing with {args.method.upper()}...\n")

    if args.method == 'tfidf':
        summary = summarizer.extractive_tfidf(text, num_sentences=args.num_sentences)
    else:  # textrank
        summary = summarizer.extractive_textrank(text, num_sentences=args.num_sentences)

    # Print results
    print("📄 Original Text:")
    print("=" * 80)
    print(text[:500] + "..." if len(text) > 500 else text)
    print("=" * 80)

    print(f"\n📝 Summary ({args.num_sentences} sentences):")
    print("=" * 80)
    print(summary)
    print("=" * 80)

    # Stats
    compression = summarizer.get_compression_ratio(text, summary)
    print(f"\n📊 Compression ratio: {compression:.2%}")
    print(f"   Original: {len(text)} chars")
    print(f"   Summary: {len(summary)} chars")


if __name__ == "__main__":
    main()
