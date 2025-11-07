"""
Advanced Keyphrase Extraction System v2.0
Author: BrillConsulting
Description: Multi-algorithm keyphrase extraction for SEO, indexing, and content analysis

Supports RAKE, YAKE, KeyBERT, and statistical methods
"""

import re
import string
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# RAKE implementation
from collections import defaultdict
import operator

# Advanced methods (optional)
try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    print("Warning: yake not available. Install with: pip install yake")

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("Warning: KeyBERT not available. Install with: pip install keybert")


class KeyphraseExtractor:
    """
    Advanced Keyphrase Extraction System

    Supports multiple algorithms:
    - RAKE (Rapid Automatic Keyword Extraction)
    - YAKE (Yet Another Keyword Extractor)
    - KeyBERT (BERT-based extraction)
    - TF-IDF based extraction
    """

    def __init__(self, method='rake', language='en'):
        """
        Initialize keyphrase extractor

        Args:
            method: 'rake', 'yake', 'keybert', 'tfidf'
            language: Language code (default: 'en')
        """
        self.method = method
        self.language = language

        # RAKE
        self.rake_stopwords = self._load_stopwords()

        # YAKE
        self.yake_extractor = None
        if method == 'yake' and YAKE_AVAILABLE:
            self.yake_extractor = yake.KeywordExtractor(lan=language, n=3, top=20)

        # KeyBERT
        self.keybert_model = None
        if method == 'keybert' and KEYBERT_AVAILABLE:
            self.keybert_model = KeyBERT()

        print(f"✓ KeyphraseExtractor initialized (method={method})")

    def extract(self, text: str, top_n=10, min_length=2, max_length=4) -> List[Tuple[str, float]]:
        """
        Extract keyphrases from text

        Args:
            text: Input text
            top_n: Number of keyphrases to return
            min_length: Minimum phrase length (words)
            max_length: Maximum phrase length (words)

        Returns:
            List of (keyphrase, score) tuples, sorted by score
        """
        if self.method == 'rake':
            return self._extract_rake(text, top_n, max_length)
        elif self.method == 'yake':
            return self._extract_yake(text, top_n)
        elif self.method == 'keybert':
            return self._extract_keybert(text, top_n)
        elif self.method == 'tfidf':
            return self._extract_tfidf(text, top_n, min_length, max_length)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _extract_rake(self, text: str, top_n=10, max_length=4) -> List[Tuple[str, float]]:
        """
        RAKE (Rapid Automatic Keyword Extraction) algorithm

        Steps:
        1. Split text into candidate keyphrases using stopwords
        2. Calculate word scores based on word frequency and degree
        3. Calculate phrase scores as sum of word scores
        """
        # Tokenize
        sentences = self._split_sentences(text)

        # Generate candidate phrases
        phrase_list = []
        for sentence in sentences:
            words = self._tokenize(sentence.lower())
            phrase = []

            for word in words:
                if word in self.rake_stopwords or word in string.punctuation:
                    if phrase:
                        phrase_list.append(' '.join(phrase))
                        phrase = []
                else:
                    phrase.append(word)

            if phrase:
                phrase_list.append(' '.join(phrase))

        # Filter by length
        phrase_list = [p for p in phrase_list if 1 <= len(p.split()) <= max_length]

        # Calculate word scores
        word_freq = Counter()
        word_degree = Counter()

        for phrase in phrase_list:
            words = phrase.split()
            for word in words:
                word_freq[word] += 1
                word_degree[word] += len(words)

        word_scores = {word: word_degree[word] / word_freq[word] for word in word_freq}

        # Calculate phrase scores
        phrase_scores = {}
        for phrase in phrase_list:
            words = phrase.split()
            score = sum(word_scores.get(word, 0) for word in words)
            phrase_scores[phrase] = score

        # Sort and return top N
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_phrases[:top_n]

    def _extract_yake(self, text: str, top_n=10) -> List[Tuple[str, float]]:
        """YAKE algorithm"""
        if not self.yake_extractor:
            raise ImportError("YAKE not available")

        keywords = self.yake_extractor.extract_keywords(text)

        # YAKE returns (keyword, score) where lower is better
        # Convert to (keyword, 1/score) for consistency
        return [(kw, 1 / (score + 1e-10)) for kw, score in keywords[:top_n]]

    def _extract_keybert(self, text: str, top_n=10) -> List[Tuple[str, float]]:
        """KeyBERT algorithm (BERT-based)"""
        if not self.keybert_model:
            raise ImportError("KeyBERT not available")

        keywords = self.keybert_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=top_n
        )

        return keywords

    def _extract_tfidf(self, text: str, top_n=10, min_length=2, max_length=4) -> List[Tuple[str, float]]:
        """
        TF-IDF based extraction (simplified single-document version)
        Uses term frequency and inverse phrase frequency
        """
        # Generate n-grams
        phrases = []
        sentences = self._split_sentences(text)

        for sentence in sentences:
            words = [w for w in self._tokenize(sentence.lower())
                     if w not in self.rake_stopwords and w not in string.punctuation]

            # Generate n-grams
            for n in range(min_length, max_length + 1):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i + n])
                    phrases.append(phrase)

        # Calculate TF (term frequency)
        phrase_freq = Counter(phrases)
        max_freq = max(phrase_freq.values()) if phrase_freq else 1

        # Calculate normalized TF scores
        phrase_scores = {phrase: freq / max_freq for phrase, freq in phrase_freq.items()}

        # Sort and return top N
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_phrases[:top_n]

    def extract_from_documents(self, documents: List[str], top_n=10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract keyphrases from multiple documents

        Args:
            documents: List of document texts
            top_n: Number of keyphrases per document

        Returns:
            Dict mapping document index to keyphrases
        """
        results = {}
        for idx, doc in enumerate(documents):
            results[idx] = self.extract(doc, top_n=top_n)

        return results

    def get_common_keyphrases(self, documents: List[str], top_n=10) -> List[Tuple[str, int]]:
        """
        Find common keyphrases across multiple documents

        Returns:
            List of (keyphrase, count) tuples
        """
        all_keyphrases = []

        for doc in documents:
            keyphrases = self.extract(doc, top_n=top_n)
            all_keyphrases.extend([kp for kp, _ in keyphrases])

        # Count occurrences
        keyphrase_counts = Counter(all_keyphrases)

        return keyphrase_counts.most_common(top_n)

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords for English"""
        # Common English stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by',
            'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'were', 'will', 'with', 'the', 'this',
            'can', 'have', 'should', 'would', 'could', 'may', 'might', 'must',
            'i', 'you', 'we', 'they', 'my', 'your', 'his', 'her', 'our', 'their',
            'am', 'been', 'being', 'do', 'does', 'did', 'done', 'having'
        }
        return stopwords

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r'[.!?\n]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple word tokenizer
        words = re.findall(r'\b\w+\b', text.lower())
        return words


class MultiMethodKeyphraseExtractor:
    """
    Ensemble keyphrase extractor using multiple methods
    """

    def __init__(self, methods=['rake', 'tfidf']):
        """
        Initialize with multiple methods

        Args:
            methods: List of method names
        """
        self.extractors = {}

        for method in methods:
            try:
                self.extractors[method] = KeyphraseExtractor(method=method)
            except Exception as e:
                print(f"Warning: Could not initialize {method}: {e}")

        print(f"✓ MultiMethodKeyphraseExtractor initialized with {len(self.extractors)} methods")

    def extract(self, text: str, top_n=10) -> List[Tuple[str, float]]:
        """
        Extract keyphrases using multiple methods and combine results

        Args:
            text: Input text
            top_n: Number of keyphrases to return

        Returns:
            List of (keyphrase, score) tuples
        """
        all_keyphrases = defaultdict(float)

        # Extract with each method
        for method, extractor in self.extractors.items():
            try:
                keyphrases = extractor.extract(text, top_n=top_n * 2)

                for phrase, score in keyphrases:
                    # Normalize scores to [0, 1]
                    max_score = max(s for _, s in keyphrases) if keyphrases else 1
                    normalized_score = score / max_score if max_score > 0 else 0

                    all_keyphrases[phrase] += normalized_score

            except Exception as e:
                print(f"Warning: {method} extraction failed: {e}")

        # Average scores across methods
        num_methods = len(self.extractors)
        averaged_keyphrases = {phrase: score / num_methods for phrase, score in all_keyphrases.items()}

        # Sort and return top N
        sorted_keyphrases = sorted(averaged_keyphrases.items(), key=lambda x: x[1], reverse=True)
        return sorted_keyphrases[:top_n]


def demo_keyphrase_extraction():
    """Demonstrate keyphrase extraction"""
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on the development
    of algorithms and statistical models. Deep learning is a type of machine learning based
    on artificial neural networks. Natural language processing enables computers to understand
    human language. Computer vision allows machines to interpret visual information from the world.
    Data science combines domain expertise, programming skills, and knowledge of mathematics
    and statistics to extract meaningful insights from data. Big data analytics involves
    examining large datasets to uncover hidden patterns and correlations.
    """

    print("=" * 80)
    print("Advanced Keyphrase Extraction System v2.0")
    print("Author: BrillConsulting")
    print("=" * 80)
    print(f"\n📄 Sample Text ({len(sample_text.split())} words)")

    # Method 1: RAKE
    print("\n" + "=" * 80)
    print("Method 1: RAKE (Rapid Automatic Keyword Extraction)")
    print("=" * 80)

    extractor_rake = KeyphraseExtractor(method='rake')
    keyphrases_rake = extractor_rake.extract(sample_text, top_n=10)

    print("\n🔑 Extracted Keyphrases:")
    for i, (phrase, score) in enumerate(keyphrases_rake, 1):
        print(f"{i:2d}. {phrase:40s} (score: {score:.3f})")

    # Method 2: TF-IDF
    print("\n" + "=" * 80)
    print("Method 2: TF-IDF Based Extraction")
    print("=" * 80)

    extractor_tfidf = KeyphraseExtractor(method='tfidf')
    keyphrases_tfidf = extractor_tfidf.extract(sample_text, top_n=10)

    print("\n🔑 Extracted Keyphrases:")
    for i, (phrase, score) in enumerate(keyphrases_tfidf, 1):
        print(f"{i:2d}. {phrase:40s} (score: {score:.3f})")

    # Method 3: YAKE (if available)
    if YAKE_AVAILABLE:
        print("\n" + "=" * 80)
        print("Method 3: YAKE (Yet Another Keyword Extractor)")
        print("=" * 80)

        extractor_yake = KeyphraseExtractor(method='yake')
        keyphrases_yake = extractor_yake.extract(sample_text, top_n=10)

        print("\n🔑 Extracted Keyphrases:")
        for i, (phrase, score) in enumerate(keyphrases_yake, 1):
            print(f"{i:2d}. {phrase:40s} (score: {score:.3f})")

    # Method 4: Multi-method ensemble
    print("\n" + "=" * 80)
    print("Method 4: Multi-Method Ensemble")
    print("=" * 80)

    methods = ['rake', 'tfidf']
    if YAKE_AVAILABLE:
        methods.append('yake')

    extractor_multi = MultiMethodKeyphraseExtractor(methods=methods)
    keyphrases_multi = extractor_multi.extract(sample_text, top_n=10)

    print("\n🔑 Extracted Keyphrases (Ensemble):")
    for i, (phrase, score) in enumerate(keyphrases_multi, 1):
        print(f"{i:2d}. {phrase:40s} (score: {score:.3f})")

    print("\n" + "=" * 80)
    print("✓ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo_keyphrase_extraction()
