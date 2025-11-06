"""
ContextCompression - Production-Ready Context Window Optimization
Author: BrillConsulting
Version: 2.0.0
Description: Advanced context compression for LLM applications with multiple strategies,
            token optimization, and relevance scoring.
"""

import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import Counter
import numpy as np
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Enumeration of available compression strategies."""
    SEMANTIC = "semantic"
    EXTRACTIVE = "extractive"
    HYBRID = "hybrid"
    TOKEN_OPTIMIZATION = "token_optimization"


class CompressionError(Exception):
    """Base exception for compression-related errors."""
    pass


class InvalidInputError(CompressionError):
    """Raised when input validation fails."""
    pass


class CompressionFailedError(CompressionError):
    """Raised when compression process fails."""
    pass


@dataclass
class CompressionResult:
    """Result object containing compressed text and metadata."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy_used: str
    relevance_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    @property
    def token_savings(self) -> int:
        """Calculate tokens saved through compression."""
        return self.original_tokens - self.compressed_tokens

    @property
    def compression_percentage(self) -> float:
        """Calculate compression percentage."""
        return (1 - self.compression_ratio) * 100


@dataclass
class ContextWindow:
    """Represents a sliding context window with metadata."""
    text: str
    start_idx: int
    end_idx: int
    relevance_score: float = 0.0
    token_count: int = 0

    def __len__(self) -> int:
        return len(self.text)


class BaseCompressor(ABC):
    """Abstract base class for compression strategies."""

    @abstractmethod
    def compress(self, text: str, target_ratio: float = 0.5) -> str:
        """
        Compress the input text.

        Args:
            text: Input text to compress
            target_ratio: Target compression ratio (0.0 to 1.0)

        Returns:
            Compressed text
        """
        pass

    @abstractmethod
    def get_relevance_scores(self, text: str, query: Optional[str] = None) -> List[float]:
        """
        Calculate relevance scores for text segments.

        Args:
            text: Input text
            query: Optional query for relevance scoring

        Returns:
            List of relevance scores
        """
        pass


class TokenOptimizer:
    """Optimizes token usage through various techniques."""

    def __init__(self):
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'\s*([.,!?;:])\s*')

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimation: ~1.3 tokens per word on average for English
        words = len(text.split())
        chars = len(text)
        return int(words * 1.3 + chars * 0.02)

    def optimize_whitespace(self, text: str) -> str:
        """Remove excessive whitespace."""
        return self.whitespace_pattern.sub(' ', text).strip()

    def optimize_punctuation(self, text: str) -> str:
        """Normalize punctuation spacing."""
        return self.punctuation_pattern.sub(r'\1 ', text)

    def remove_redundant_phrases(self, text: str, min_length: int = 4) -> str:
        """
        Remove redundant repeated phrases.

        Args:
            text: Input text
            min_length: Minimum phrase length to consider

        Returns:
            Text with redundant phrases removed
        """
        words = text.split()
        seen_phrases = set()
        result = []

        i = 0
        while i < len(words):
            found_redundancy = False
            for length in range(min_length, max(i, min_length) + 1):
                if i + length <= len(words):
                    phrase = ' '.join(words[i:i+length])
                    if phrase.lower() in seen_phrases:
                        i += length
                        found_redundancy = True
                        break
                    seen_phrases.add(phrase.lower())

            if not found_redundancy:
                result.append(words[i])
                i += 1

        return ' '.join(result)

    def optimize(self, text: str, aggressive: bool = False) -> str:
        """
        Apply all optimization techniques.

        Args:
            text: Input text
            aggressive: If True, apply aggressive optimizations

        Returns:
            Optimized text
        """
        text = self.optimize_whitespace(text)
        text = self.optimize_punctuation(text)

        if aggressive:
            text = self.remove_redundant_phrases(text)

        return text


class SemanticCompressor(BaseCompressor):
    """Compresses text based on semantic importance."""

    def __init__(self):
        self.optimizer = TokenOptimizer()
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                            'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had'])

    def _calculate_sentence_importance(self, sentence: str, word_freq: Counter) -> float:
        """
        Calculate importance score for a sentence based on word frequency.

        Args:
            sentence: Input sentence
            word_freq: Word frequency counter

        Returns:
            Importance score
        """
        words = [w.lower() for w in re.findall(r'\w+', sentence)]
        if not words:
            return 0.0

        # Calculate score based on word frequency and non-stopword ratio
        total_score = sum(word_freq.get(w, 0) for w in words if w not in self.stopwords)
        non_stopword_ratio = len([w for w in words if w not in self.stopwords]) / len(words)

        return total_score * non_stopword_ratio

    def get_relevance_scores(self, text: str, query: Optional[str] = None) -> List[float]:
        """Calculate relevance scores for each sentence."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        # Calculate word frequencies
        all_words = [w.lower() for w in re.findall(r'\w+', text)]
        word_freq = Counter(all_words)

        # Calculate query-specific relevance if query provided
        query_words = set()
        if query:
            query_words = set(w.lower() for w in re.findall(r'\w+', query))

        scores = []
        for sentence in sentences:
            base_score = self._calculate_sentence_importance(sentence, word_freq)

            if query_words:
                sentence_words = set(w.lower() for w in re.findall(r'\w+', sentence))
                query_overlap = len(query_words & sentence_words) / max(len(query_words), 1)
                base_score *= (1 + query_overlap)

            scores.append(base_score)

        # Normalize scores
        max_score = max(scores) if scores else 1.0
        return [s / max_score if max_score > 0 else 0.0 for s in scores]

    def compress(self, text: str, target_ratio: float = 0.5) -> str:
        """
        Compress text by selecting semantically important sentences.

        Args:
            text: Input text
            target_ratio: Target compression ratio (0.0 to 1.0)

        Returns:
            Compressed text
        """
        if not text.strip():
            return ""

        sentences = re.split(r'([.!?]+)', text)
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])

        full_sentences = [s.strip() for s in full_sentences if s.strip()]

        if not full_sentences:
            return text

        # Get relevance scores
        scores = self.get_relevance_scores(text)

        # Calculate how many sentences to keep
        num_to_keep = max(1, int(len(full_sentences) * target_ratio))

        # Sort by score and keep top sentences
        sentence_scores = list(zip(full_sentences, scores, range(len(full_sentences))))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Keep top sentences and restore original order
        kept_sentences = sentence_scores[:num_to_keep]
        kept_sentences.sort(key=lambda x: x[2])

        result = ' '.join([s[0] for s in kept_sentences])
        return self.optimizer.optimize(result)


class ExtractiveSummarizer(BaseCompressor):
    """Extracts key information using extractive summarization."""

    def __init__(self):
        self.optimizer = TokenOptimizer()

    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text."""
        # Simple noun phrase extraction
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        phrase_freq = Counter(words)
        return [phrase for phrase, _ in phrase_freq.most_common(max_phrases)]

    def get_relevance_scores(self, text: str, query: Optional[str] = None) -> List[float]:
        """Calculate relevance scores based on key phrase presence."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        key_phrases = self._extract_key_phrases(text)

        scores = []
        for sentence in sentences:
            score = sum(1 for phrase in key_phrases if phrase in sentence)
            scores.append(float(score))

        # Normalize
        max_score = max(scores) if scores else 1.0
        return [s / max_score if max_score > 0 else 0.0 for s in scores]

    def compress(self, text: str, target_ratio: float = 0.5) -> str:
        """
        Compress text using extractive summarization.

        Args:
            text: Input text
            target_ratio: Target compression ratio

        Returns:
            Compressed text
        """
        if not text.strip():
            return ""

        sentences = re.split(r'([.!?]+)', text)
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])

        full_sentences = [s.strip() for s in full_sentences if s.strip()]

        if not full_sentences:
            return text

        scores = self.get_relevance_scores(text)
        num_to_keep = max(1, int(len(full_sentences) * target_ratio))

        sentence_scores = list(zip(full_sentences, scores, range(len(full_sentences))))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        kept_sentences = sentence_scores[:num_to_keep]
        kept_sentences.sort(key=lambda x: x[2])

        result = ' '.join([s[0] for s in kept_sentences])
        return self.optimizer.optimize(result)


class HybridCompressor(BaseCompressor):
    """Combines multiple compression strategies."""

    def __init__(self):
        self.semantic_compressor = SemanticCompressor()
        self.extractive_summarizer = ExtractiveSummarizer()
        self.optimizer = TokenOptimizer()

    def get_relevance_scores(self, text: str, query: Optional[str] = None) -> List[float]:
        """Calculate relevance scores using combined strategies."""
        semantic_scores = self.semantic_compressor.get_relevance_scores(text, query)
        extractive_scores = self.extractive_summarizer.get_relevance_scores(text, query)

        if not semantic_scores or not extractive_scores:
            return semantic_scores or extractive_scores

        # Combine scores with weighted average
        combined = [(s + e) / 2 for s, e in zip(semantic_scores, extractive_scores)]
        return combined

    def compress(self, text: str, target_ratio: float = 0.5) -> str:
        """
        Compress text using hybrid approach.

        Args:
            text: Input text
            target_ratio: Target compression ratio

        Returns:
            Compressed text
        """
        if not text.strip():
            return ""

        # First pass: semantic compression
        intermediate = self.semantic_compressor.compress(text, target_ratio * 1.2)

        # Second pass: extractive refinement
        result = self.extractive_summarizer.compress(intermediate, target_ratio / (target_ratio * 1.2))

        # Final pass: token optimization
        return self.optimizer.optimize(result, aggressive=True)


class ContextCompressionManager:
    """
    Production-ready context compression manager with multiple strategies.

    This class provides comprehensive context compression capabilities including:
    - Multiple compression strategies (semantic, extractive, hybrid)
    - Token optimization
    - Relevance scoring
    - Context windowing
    - Async support
    - Comprehensive error handling

    Example:
        >>> manager = ContextCompressionManager()
        >>> result = manager.compress_context(
        ...     text="Long document text here...",
        ...     strategy=CompressionStrategy.HYBRID,
        ...     target_ratio=0.5
        ... )
        >>> print(f"Compressed from {result.original_tokens} to {result.compressed_tokens} tokens")
    """

    def __init__(self, default_strategy: CompressionStrategy = CompressionStrategy.HYBRID):
        """
        Initialize the compression manager.

        Args:
            default_strategy: Default compression strategy to use
        """
        self.default_strategy = default_strategy
        self.token_optimizer = TokenOptimizer()
        self.compressors: Dict[CompressionStrategy, BaseCompressor] = {
            CompressionStrategy.SEMANTIC: SemanticCompressor(),
            CompressionStrategy.EXTRACTIVE: ExtractiveSummarizer(),
            CompressionStrategy.HYBRID: HybridCompressor(),
        }

        logger.info(f"ContextCompressionManager initialized with strategy: {default_strategy.value}")

    def _validate_input(self, text: str, target_ratio: float) -> None:
        """
        Validate input parameters.

        Args:
            text: Input text to validate
            target_ratio: Target compression ratio to validate

        Raises:
            InvalidInputError: If validation fails
        """
        if not isinstance(text, str):
            raise InvalidInputError(f"Text must be a string, got {type(text)}")

        if not text.strip():
            raise InvalidInputError("Input text cannot be empty")

        if not 0.0 < target_ratio <= 1.0:
            raise InvalidInputError(f"Target ratio must be between 0.0 and 1.0, got {target_ratio}")

    def compress_context(
        self,
        text: str,
        strategy: Optional[CompressionStrategy] = None,
        target_ratio: float = 0.5,
        query: Optional[str] = None
    ) -> CompressionResult:
        """
        Compress context using specified strategy.

        Args:
            text: Input text to compress
            strategy: Compression strategy to use (defaults to manager's default)
            target_ratio: Target compression ratio (0.0 to 1.0, where 0.5 means 50% of original)
            query: Optional query for relevance-based compression

        Returns:
            CompressionResult object containing compressed text and metadata

        Raises:
            InvalidInputError: If input validation fails
            CompressionFailedError: If compression process fails

        Example:
            >>> result = manager.compress_context(
            ...     text="Very long text...",
            ...     strategy=CompressionStrategy.SEMANTIC,
            ...     target_ratio=0.6
            ... )
        """
        start_time = datetime.now()

        try:
            # Validate input
            self._validate_input(text, target_ratio)

            # Select strategy
            strategy = strategy or self.default_strategy
            compressor = self.compressors.get(strategy)

            if not compressor:
                raise CompressionFailedError(f"Unknown compression strategy: {strategy}")

            # Calculate original tokens
            original_tokens = self.token_optimizer.estimate_tokens(text)

            # Get relevance scores
            relevance_scores = compressor.get_relevance_scores(text, query)

            # Perform compression
            compressed_text = compressor.compress(text, target_ratio)

            # Calculate compressed tokens
            compressed_tokens = self.token_optimizer.estimate_tokens(compressed_text)

            # Calculate actual compression ratio
            actual_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0.0

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            result = CompressionResult(
                original_text=text,
                compressed_text=compressed_text,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=actual_ratio,
                strategy_used=strategy.value,
                relevance_scores=relevance_scores,
                metadata={
                    'target_ratio': target_ratio,
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                },
                processing_time_ms=processing_time
            )

            logger.info(
                f"Compression complete: {original_tokens} -> {compressed_tokens} tokens "
                f"({result.compression_percentage:.1f}% reduction) in {processing_time:.2f}ms"
            )

            return result

        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            raise CompressionFailedError(f"Compression failed: {str(e)}") from e

    async def compress_context_async(
        self,
        text: str,
        strategy: Optional[CompressionStrategy] = None,
        target_ratio: float = 0.5,
        query: Optional[str] = None
    ) -> CompressionResult:
        """
        Asynchronously compress context.

        Args:
            text: Input text to compress
            strategy: Compression strategy to use
            target_ratio: Target compression ratio
            query: Optional query for relevance-based compression

        Returns:
            CompressionResult object

        Example:
            >>> result = await manager.compress_context_async(text="Long text...")
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.compress_context(text, strategy, target_ratio, query)
        )

    def create_context_windows(
        self,
        text: str,
        window_size: int = 512,
        overlap: int = 128,
        min_relevance: float = 0.0,
        query: Optional[str] = None
    ) -> List[ContextWindow]:
        """
        Create sliding context windows with relevance scoring.

        Args:
            text: Input text to window
            window_size: Size of each window in tokens (approximate)
            overlap: Number of overlapping tokens between windows
            min_relevance: Minimum relevance score to include window
            query: Optional query for relevance scoring

        Returns:
            List of ContextWindow objects

        Raises:
            InvalidInputError: If parameters are invalid

        Example:
            >>> windows = manager.create_context_windows(
            ...     text="Long document...",
            ...     window_size=256,
            ...     overlap=64
            ... )
        """
        try:
            if not text.strip():
                raise InvalidInputError("Input text cannot be empty")

            if window_size <= 0:
                raise InvalidInputError("Window size must be positive")

            if overlap >= window_size:
                raise InvalidInputError("Overlap must be less than window size")

            # Split into words for rough token estimation
            words = text.split()
            words_per_token = 0.75  # Rough estimate
            words_per_window = int(window_size * words_per_token)
            words_overlap = int(overlap * words_per_token)

            step = words_per_window - words_overlap
            windows = []

            compressor = self.compressors[CompressionStrategy.SEMANTIC]

            for i in range(0, len(words), step):
                window_words = words[i:i + words_per_window]
                window_text = ' '.join(window_words)

                if not window_text.strip():
                    continue

                # Calculate relevance score for this window
                scores = compressor.get_relevance_scores(window_text, query)
                avg_score = np.mean(scores) if scores else 0.0

                if avg_score >= min_relevance:
                    window = ContextWindow(
                        text=window_text,
                        start_idx=i,
                        end_idx=min(i + words_per_window, len(words)),
                        relevance_score=float(avg_score),
                        token_count=self.token_optimizer.estimate_tokens(window_text)
                    )
                    windows.append(window)

            logger.info(f"Created {len(windows)} context windows from {len(words)} words")
            return windows

        except InvalidInputError:
            raise
        except Exception as e:
            logger.error(f"Window creation failed: {str(e)}")
            raise CompressionFailedError(f"Window creation failed: {str(e)}") from e

    def optimize_tokens(self, text: str, aggressive: bool = False) -> str:
        """
        Optimize token usage without semantic compression.

        Args:
            text: Input text to optimize
            aggressive: If True, apply aggressive optimizations

        Returns:
            Optimized text

        Example:
            >>> optimized = manager.optimize_tokens("Text   with    extra  spaces")
        """
        try:
            if not text.strip():
                raise InvalidInputError("Input text cannot be empty")

            return self.token_optimizer.optimize(text, aggressive)

        except Exception as e:
            logger.error(f"Token optimization failed: {str(e)}")
            raise CompressionFailedError(f"Token optimization failed: {str(e)}") from e

    def batch_compress(
        self,
        texts: List[str],
        strategy: Optional[CompressionStrategy] = None,
        target_ratio: float = 0.5
    ) -> List[CompressionResult]:
        """
        Compress multiple texts in batch.

        Args:
            texts: List of texts to compress
            strategy: Compression strategy to use
            target_ratio: Target compression ratio

        Returns:
            List of CompressionResult objects

        Example:
            >>> results = manager.batch_compress(
            ...     texts=["Text 1", "Text 2", "Text 3"],
            ...     target_ratio=0.5
            ... )
        """
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.compress_context(text, strategy, target_ratio)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to compress text {i}: {str(e)}")
                # Continue with other texts

        logger.info(f"Batch compression complete: {len(results)}/{len(texts)} successful")
        return results

    async def batch_compress_async(
        self,
        texts: List[str],
        strategy: Optional[CompressionStrategy] = None,
        target_ratio: float = 0.5
    ) -> List[CompressionResult]:
        """
        Asynchronously compress multiple texts in batch.

        Args:
            texts: List of texts to compress
            strategy: Compression strategy to use
            target_ratio: Target compression ratio

        Returns:
            List of CompressionResult objects

        Example:
            >>> results = await manager.batch_compress_async(
            ...     texts=["Text 1", "Text 2", "Text 3"]
            ... )
        """
        tasks = [
            self.compress_context_async(text, strategy, target_ratio)
            for text in texts
        ]

        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to compress text: {str(e)}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get manager statistics and information.

        Returns:
            Dictionary containing manager statistics
        """
        return {
            'version': '2.0.0',
            'default_strategy': self.default_strategy.value,
            'available_strategies': [s.value for s in CompressionStrategy],
            'initialized': True,
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """
    Demonstration of ContextCompression capabilities.

    This function showcases various features including:
    - Different compression strategies
    - Token optimization
    - Context windowing
    - Async compression
    - Batch processing
    """
    print("=" * 80)
    print("ContextCompression v2.0.0 - Production Demo")
    print("=" * 80)
    print()

    # Sample text for demonstration
    sample_text = """
    Artificial intelligence has revolutionized the way we interact with technology.
    Machine learning models have become increasingly sophisticated over the years.
    Natural language processing enables computers to understand human language.
    Deep learning networks can process vast amounts of data efficiently.
    The field of AI continues to grow and evolve rapidly.
    Researchers are constantly developing new techniques and methodologies.
    AI applications span across healthcare, finance, education, and many other domains.
    The future of artificial intelligence holds tremendous potential.
    Ethical considerations in AI development are becoming increasingly important.
    AI systems must be designed with transparency and fairness in mind.
    """

    # Initialize manager
    print("1. Initializing ContextCompressionManager...")
    manager = ContextCompressionManager()
    print(f"   Manager statistics: {manager.get_statistics()}")
    print()

    # Demo 1: Semantic Compression
    print("2. Semantic Compression (50% target)...")
    result = manager.compress_context(
        text=sample_text,
        strategy=CompressionStrategy.SEMANTIC,
        target_ratio=0.5
    )
    print(f"   Original tokens: {result.original_tokens}")
    print(f"   Compressed tokens: {result.compressed_tokens}")
    print(f"   Compression ratio: {result.compression_ratio:.2%}")
    print(f"   Token savings: {result.token_savings}")
    print(f"   Processing time: {result.processing_time_ms:.2f}ms")
    print(f"   Compressed text preview: {result.compressed_text[:100]}...")
    print()

    # Demo 2: Extractive Compression
    print("3. Extractive Compression (40% target)...")
    result = manager.compress_context(
        text=sample_text,
        strategy=CompressionStrategy.EXTRACTIVE,
        target_ratio=0.4
    )
    print(f"   Compression percentage: {result.compression_percentage:.1f}%")
    print(f"   Strategy used: {result.strategy_used}")
    print()

    # Demo 3: Hybrid Compression
    print("4. Hybrid Compression with query (30% target)...")
    result = manager.compress_context(
        text=sample_text,
        strategy=CompressionStrategy.HYBRID,
        target_ratio=0.3,
        query="machine learning and ethics"
    )
    print(f"   Original: {result.original_tokens} tokens")
    print(f"   Compressed: {result.compressed_tokens} tokens")
    print(f"   Savings: {result.token_savings} tokens ({result.compression_percentage:.1f}%)")
    print()

    # Demo 4: Context Windowing
    print("5. Context Windowing...")
    windows = manager.create_context_windows(
        text=sample_text,
        window_size=50,
        overlap=10,
        min_relevance=0.3
    )
    print(f"   Created {len(windows)} context windows")
    for i, window in enumerate(windows[:3], 1):
        print(f"   Window {i}: {window.token_count} tokens, "
              f"relevance: {window.relevance_score:.2f}")
    print()

    # Demo 5: Token Optimization
    print("6. Token Optimization...")
    messy_text = "This   is   a    text   with    excessive    whitespace  ."
    optimized = manager.optimize_tokens(messy_text, aggressive=True)
    print(f"   Original: '{messy_text}'")
    print(f"   Optimized: '{optimized}'")
    print()

    # Demo 6: Batch Compression
    print("7. Batch Compression...")
    texts = [
        "First document about artificial intelligence and machine learning.",
        "Second document discussing natural language processing applications.",
        "Third document covering deep learning and neural networks."
    ]
    results = manager.batch_compress(texts, target_ratio=0.5)
    print(f"   Compressed {len(results)} documents")
    total_saved = sum(r.token_savings for r in results)
    print(f"   Total tokens saved: {total_saved}")
    print()

    # Demo 7: Async Compression
    print("8. Async Compression Demo...")
    async def async_demo():
        result = await manager.compress_context_async(
            text=sample_text,
            target_ratio=0.5
        )
        return result

    result = asyncio.run(async_demo())
    print(f"   Async compression completed: {result.compressed_tokens} tokens")
    print()

    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Run demonstration
    demo()
