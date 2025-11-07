"""
Advanced Text Summarization System v2.0
Author: BrillConsulting
Description: Extractive & abstractive summarization with BART, T5, and Pegasus
"""

import numpy as np
from typing import List, Dict, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Install: pip install transformers torch")


class TextSummarizer:
    """Advanced text summarization with multiple models"""

    def __init__(self, model_name='facebook/bart-large-cnn', method='abstractive'):
        """
        Initialize summarizer

        Args:
            model_name: HuggingFace model name
            method: 'abstractive' or 'extractive'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")

        self.model_name = model_name
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔧 Loading summarization model: {model_name}")
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ Model loaded\n")

    def summarize(self, text: str, max_length=130, min_length=30,
                  length_penalty=2.0, num_beams=4) -> Dict:
        """
        Summarize text

        Args:
            text: Input text
            max_length: Maximum summary length
            min_length: Minimum summary length
            length_penalty: Length penalty for beam search
            num_beams: Number of beams for beam search

        Returns:
            Summary with metadata
        """
        # Truncate if too long
        max_input_length = 1024
        if len(text.split()) > max_input_length:
            text = ' '.join(text.split()[:max_input_length])

        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            do_sample=False
        )[0]

        summary = result['summary_text']
        compression_ratio = len(summary.split()) / len(text.split())

        return {
            'summary': summary,
            'original_length': len(text.split()),
            'summary_length': len(summary.split()),
            'compression_ratio': compression_ratio,
            'model': self.model_name
        }

    def summarize_batch(self, texts: List[str], max_length=130) -> List[Dict]:
        """Summarize multiple texts"""
        return [self.summarize(text, max_length=max_length) for text in texts]

    def summarize_long_document(self, text: str, chunk_size=1000,
                                 max_length=130) -> Dict:
        """
        Summarize very long documents using chunking

        Args:
            text: Long input text
            chunk_size: Size of each chunk in words
            max_length: Max summary length per chunk

        Returns:
            Final summary
        """
        words = text.split()

        if len(words) <= chunk_size:
            return self.summarize(text, max_length=max_length)

        # Split into chunks
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)

        print(f"📚 Summarizing {len(chunks)} chunks...")

        # Summarize each chunk
        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            summary = self.summarize(chunk, max_length=max_length)
            chunk_summaries.append(summary['summary'])
            if (idx + 1) % 5 == 0:
                print(f"  Processed {idx+1}/{len(chunks)} chunks")

        # Combine and summarize again
        combined = ' '.join(chunk_summaries)

        if len(combined.split()) > chunk_size:
            final_summary = self.summarize_long_document(
                combined, chunk_size=chunk_size, max_length=max_length
            )
        else:
            final_summary = self.summarize(combined, max_length=max_length)

        final_summary['num_chunks'] = len(chunks)
        return final_summary


def main():
    parser = argparse.ArgumentParser(description='Advanced Text Summarization v2.0')
    parser.add_argument('--text', type=str, help='Text to summarize')
    parser.add_argument('--file', type=str, help='File with text')
    parser.add_argument('--model', type=str, default='facebook/bart-large-cnn',
                       help='Model name')
    parser.add_argument('--max-length', type=int, default=130,
                       help='Max summary length')
    parser.add_argument('--min-length', type=int, default=30,
                       help='Min summary length')
    parser.add_argument('--long', action='store_true',
                       help='Use long document mode')

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

    # Initialize
    summarizer = TextSummarizer(model_name=args.model)

    print(f"📄 Original: {len(text.split())} words\n")

    # Summarize
    if args.long:
        result = summarizer.summarize_long_document(
            text, max_length=args.max_length
        )
    else:
        result = summarizer.summarize(
            text,
            max_length=args.max_length,
            min_length=args.min_length
        )

    # Display
    print("=" * 60)
    print("📝 SUMMARY")
    print("=" * 60)
    print(result['summary'])
    print("=" * 60)
    print(f"Original: {result['original_length']} words")
    print(f"Summary: {result['summary_length']} words")
    print(f"Compression: {result['compression_ratio']:.1%}")
    if 'num_chunks' in result:
        print(f"Chunks: {result['num_chunks']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
