"""
Text Generation System
Author: BrillConsulting
Description: Generate text using N-gram language models and Markov chains
"""

import random
from collections import defaultdict, Counter
import argparse
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextGenerator:
    """N-gram based text generation"""

    def __init__(self, n=2):
        """
        Initialize text generator

        Args:
            n: N-gram size (2=bigram, 3=trigram)
        """
        self.n = n
        self.ngrams = defaultdict(Counter)

    def train(self, text: str):
        """
        Train on text corpus

        Args:
            text: Training text
        """
        # Tokenize
        tokens = word_tokenize(text.lower())

        # Build n-grams
        for i in range(len(tokens) - self.n):
            # N-1 words as context
            context = tuple(tokens[i:i + self.n - 1])
            # Next word
            next_word = tokens[i + self.n - 1]

            self.ngrams[context][next_word] += 1

        print(f"✅ Trained on {len(tokens)} tokens")
        print(f"   Vocabulary size: {len(set(tokens))}")
        print(f"   {self.n}-gram contexts: {len(self.ngrams)}\n")

    def generate(self, seed: str = None, length: int = 50) -> str:
        """
        Generate text

        Args:
            seed: Starting words (None for random)
            length: Number of words to generate

        Returns:
            Generated text
        """
        if not self.ngrams:
            return "Model not trained!"

        # Get starting context
        if seed:
            context = tuple(word_tokenize(seed.lower())[:self.n - 1])
            if context not in self.ngrams:
                # Fallback to random
                context = random.choice(list(self.ngrams.keys()))
        else:
            context = random.choice(list(self.ngrams.keys()))

        # Generate words
        generated = list(context)

        for _ in range(length):
            if context not in self.ngrams:
                break

            # Get possible next words and their counts
            possible_words = self.ngrams[context]

            # Weighted random choice
            words = list(possible_words.keys())
            weights = list(possible_words.values())

            next_word = random.choices(words, weights=weights)[0]

            generated.append(next_word)

            # Update context
            context = tuple(generated[-(self.n - 1):])

        return ' '.join(generated)


class MarkovChainGenerator:
    """Simple Markov chain text generator"""

    def __init__(self):
        self.chain = defaultdict(list)

    def train(self, text: str):
        """Train Markov chain on text"""
        words = word_tokenize(text)

        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            self.chain[current_word].append(next_word)

        print(f"✅ Trained Markov chain on {len(words)} words\n")

    def generate(self, seed: str = None, length: int = 50) -> str:
        """Generate text using Markov chain"""
        if not self.chain:
            return "Model not trained!"

        # Starting word
        if seed:
            current = seed.split()[0]
            if current not in self.chain:
                current = random.choice(list(self.chain.keys()))
        else:
            current = random.choice(list(self.chain.keys()))

        generated = [current]

        for _ in range(length - 1):
            if current not in self.chain or not self.chain[current]:
                break

            next_word = random.choice(self.chain[current])
            generated.append(next_word)
            current = next_word

        return ' '.join(generated)


def main():
    parser = argparse.ArgumentParser(description='Text Generation')
    parser.add_argument('--train-file', type=str, required=True,
                       help='Text file for training')
    parser.add_argument('--method', type=str, default='ngram',
                       choices=['ngram', 'markov'],
                       help='Generation method')
    parser.add_argument('--n', type=int, default=2,
                       help='N-gram size (for ngram method)')
    parser.add_argument('--seed', type=str,
                       help='Starting words for generation')
    parser.add_argument('--length', type=int, default=50,
                       help='Number of words to generate')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to generate')

    args = parser.parse_args()

    # Load training text
    print(f"📂 Loading training data from {args.train_file}...\n")
    with open(args.train_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Initialize generator
    if args.method == 'ngram':
        generator = TextGenerator(n=args.n)
        print(f"🔧 Training {args.n}-gram model...\n")
    else:
        generator = MarkovChainGenerator()
        print("🔧 Training Markov chain...\n")

    generator.train(text)

    # Generate samples
    print(f"✨ Generating {args.num_samples} text sample(s):\n")
    print("=" * 80)

    for i in range(args.num_samples):
        generated = generator.generate(seed=args.seed, length=args.length)

        print(f"\nSample {i + 1}:")
        print(generated)
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
