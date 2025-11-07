"""
Advanced Language Modeling System v2.0
Author: BrillConsulting
Description: N-gram and neural language models for perplexity calculation and text generation

Supports N-gram models, RNN-based models, and transformer-based models (GPT-2)
"""

import re
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# Neural language models (optional)
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")


class NgramLanguageModel:
    """
    N-gram Language Model

    Estimates probability of sequences using n-grams with smoothing
    """

    def __init__(self, n=3, smoothing='laplace', alpha=1.0):
        """
        Initialize N-gram language model

        Args:
            n: N-gram size (1=unigram, 2=bigram, 3=trigram, etc.)
            smoothing: 'laplace' or 'none'
            alpha: Smoothing parameter
        """
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha

        # N-gram counts
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()

        print(f"✓ NgramLanguageModel initialized (n={n}, smoothing={smoothing})")

    def train(self, texts: List[str]):
        """
        Train language model on corpus

        Args:
            texts: List of training texts
        """
        print(f"Training on {len(texts)} documents...")

        for text in texts:
            tokens = self._tokenize(text)
            self.vocabulary.update(tokens)

            # Add start/end tokens
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

            # Count n-grams
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = ngram[:-1]

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        print(f"✓ Vocabulary size: {len(self.vocabulary)}")
        print(f"✓ Number of {self.n}-grams: {len(self.ngram_counts)}")

    def probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Calculate probability of n-gram

        Args:
            ngram: N-gram tuple

        Returns:
            Probability P(w_n | w_1...w_{n-1})
        """
        if len(ngram) != self.n:
            raise ValueError(f"Expected {self.n}-gram, got {len(ngram)}-gram")

        context = ngram[:-1]

        if self.smoothing == 'laplace':
            # Laplace (add-alpha) smoothing
            numerator = self.ngram_counts[ngram] + self.alpha
            denominator = self.context_counts[context] + self.alpha * len(self.vocabulary)

            return numerator / denominator if denominator > 0 else 0.0

        elif self.smoothing == 'none':
            # No smoothing (may result in zero probabilities)
            context_count = self.context_counts[context]
            return self.ngram_counts[ngram] / context_count if context_count > 0 else 0.0

        else:
            raise ValueError(f"Unknown smoothing: {self.smoothing}")

    def sentence_probability(self, text: str) -> float:
        """
        Calculate probability of a sentence

        Args:
            text: Input sentence

        Returns:
            Log probability
        """
        tokens = self._tokenize(text)
        tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

        log_prob = 0.0
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            prob = self.probability(ngram)

            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += -float('inf')

        return log_prob

    def perplexity(self, texts: List[str]) -> float:
        """
        Calculate perplexity on test set

        Perplexity = exp(-1/N * sum(log P(w_i)))

        Lower perplexity = better model

        Args:
            texts: List of test texts

        Returns:
            Perplexity score
        """
        total_log_prob = 0.0
        total_words = 0

        for text in texts:
            tokens = self._tokenize(text)
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                prob = self.probability(ngram)

                if prob > 0:
                    total_log_prob += math.log(prob)
                else:
                    total_log_prob += math.log(1e-10)  # Small constant to avoid -inf

                total_words += 1

        avg_log_prob = total_log_prob / total_words if total_words > 0 else 0
        perplexity = math.exp(-avg_log_prob)

        return perplexity

    def generate(self, prefix: str = '', max_length=20) -> str:
        """
        Generate text using the language model

        Args:
            prefix: Starting text
            max_length: Maximum number of words to generate

        Returns:
            Generated text
        """
        tokens = self._tokenize(prefix) if prefix else ['<s>'] * (self.n - 1)

        for _ in range(max_length):
            # Get context
            context = tuple(tokens[-(self.n - 1):])

            # Find all possible next words
            candidates = []
            for ngram, count in self.ngram_counts.items():
                if ngram[:-1] == context:
                    candidates.append((ngram[-1], count))

            if not candidates:
                break

            # Sample next word based on counts
            words, counts = zip(*candidates)
            total = sum(counts)
            probs = [c / total for c in counts]

            next_word = np.random.choice(words, p=probs)

            if next_word == '</s>':
                break

            tokens.append(next_word)

        # Remove start tokens
        result_tokens = [t for t in tokens if t not in ['<s>', '</s>']]
        return ' '.join(result_tokens)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple word tokenizer
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words


class TransformerLanguageModel:
    """
    Transformer-based Language Model using GPT-2

    For perplexity calculation and text generation
    """

    def __init__(self, model_name='gpt2'):
        """
        Initialize transformer language model

        Args:
            model_name: Hugging Face model name (e.g., 'gpt2', 'gpt2-medium')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available")

        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"✓ TransformerLanguageModel initialized (model={model_name}, device={self.device})")

    def perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text

        Args:
            text: Input text

        Returns:
            Perplexity score
        """
        # Tokenize
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.device)

        # Calculate loss
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        # Perplexity = exp(loss)
        perplexity = math.exp(loss)

        return perplexity

    def generate(self, prefix: str = '', max_length=50, temperature=1.0, top_k=50) -> str:
        """
        Generate text using GPT-2

        Args:
            prefix: Starting text
            max_length: Maximum number of tokens
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling

        Returns:
            Generated text
        """
        # Encode prefix
        if prefix:
            input_ids = self.tokenizer.encode(prefix, return_tensors='pt').to(self.device)
        else:
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_text

    def next_word_probability(self, prefix: str, top_k=10) -> List[Tuple[str, float]]:
        """
        Get most probable next words

        Args:
            prefix: Context text
            top_k: Number of top predictions

        Returns:
            List of (word, probability) tuples
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prefix, return_tensors='pt').to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = outputs.logits[0, -1, :]

        # Apply softmax
        probs = torch.softmax(predictions, dim=0)

        # Get top k
        top_probs, top_indices = torch.topk(probs, top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            word = self.tokenizer.decode([idx.item()])
            results.append((word, prob.item()))

        return results


def demo_language_modeling():
    """Demonstrate language modeling"""
    # Sample corpus
    training_texts = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat ate the rat",
        "the dog ate the bone",
        "cats and dogs are pets",
        "the quick brown fox jumps over the lazy dog",
        "the cat and the dog are friends"
    ]

    test_texts = [
        "the cat sat on the rug",
        "the dog ate the food"
    ]

    print("=" * 80)
    print("Advanced Language Modeling System v2.0")
    print("Author: BrillConsulting")
    print("=" * 80)

    # Method 1: N-gram Language Model
    print("\n" + "=" * 80)
    print("Method 1: Trigram Language Model")
    print("=" * 80)

    trigram_model = NgramLanguageModel(n=3, smoothing='laplace')
    trigram_model.train(training_texts)

    print("\n📊 Test Set Perplexity:")
    ppl = trigram_model.perplexity(test_texts)
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  (Lower is better)")

    print("\n🎲 Text Generation:")
    for i in range(3):
        generated = trigram_model.generate(prefix='the cat', max_length=10)
        print(f"  {i+1}. {generated}")

    # Method 2: Bigram Language Model
    print("\n" + "=" * 80)
    print("Method 2: Bigram Language Model")
    print("=" * 80)

    bigram_model = NgramLanguageModel(n=2, smoothing='laplace')
    bigram_model.train(training_texts)

    print("\n📊 Test Set Perplexity:")
    ppl = bigram_model.perplexity(test_texts)
    print(f"  Perplexity: {ppl:.2f}")

    # Method 3: Transformer Language Model (GPT-2) if available
    if TRANSFORMERS_AVAILABLE:
        print("\n" + "=" * 80)
        print("Method 3: Transformer Language Model (GPT-2)")
        print("=" * 80)

        gpt2_model = TransformerLanguageModel(model_name='gpt2')

        print("\n📊 Perplexity on sample text:")
        sample_text = "The quick brown fox jumps over the lazy dog."
        ppl = gpt2_model.perplexity(sample_text)
        print(f"  Text: '{sample_text}'")
        print(f"  Perplexity: {ppl:.2f}")

        print("\n🔮 Next Word Prediction:")
        prefix = "The cat sat on the"
        predictions = gpt2_model.next_word_probability(prefix, top_k=5)
        print(f"  Context: '{prefix}'")
        print(f"  Top 5 predictions:")
        for word, prob in predictions:
            print(f"    {word:15s} (p={prob:.4f})")

        print("\n🎲 Text Generation:")
        generated = gpt2_model.generate(
            prefix="Once upon a time",
            max_length=30,
            temperature=0.8
        )
        print(f"  {generated}")

    print("\n" + "=" * 80)
    print("✓ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo_language_modeling()
