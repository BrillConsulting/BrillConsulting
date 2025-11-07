"""
Advanced Sentiment Analysis System v2.0
Author: BrillConsulting
Description: Multi-method sentiment analysis with Transformers, VADER, TextBlob, and aspect-based analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# VADER for rule-based sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️ VADER not available. Install with: pip install vaderSentiment")

# TextBlob for pattern-based sentiment
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️ TextBlob not available. Install with: pip install textblob")

# Transformers for deep learning sentiment
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")


class SentimentAnalyzer:
    """
    Advanced sentiment analysis with multiple methods
    """

    def __init__(self, method='transformer', model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        """
        Initialize sentiment analyzer

        Args:
            method: 'transformer', 'vader', 'textblob', or 'ensemble'
            model_name: HuggingFace model name for transformer method
        """
        self.method = method
        self.model_name = model_name
        self.results_cache = {}

        # Initialize selected method
        if method == 'vader':
            if VADER_AVAILABLE:
                self.vader = SentimentIntensityAnalyzer()
            else:
                raise ImportError("VADER not available")

        elif method == 'textblob':
            if not TEXTBLOB_AVAILABLE:
                raise ImportError("TextBlob not available")

        elif method == 'transformer':
            if TRANSFORMERS_AVAILABLE:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"🔧 Loading transformer model: {model_name}")
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                print("✅ Model loaded successfully\n")
            else:
                raise ImportError("Transformers not available")

        elif method == 'ensemble':
            # Initialize all available methods
            if VADER_AVAILABLE:
                self.vader = SentimentIntensityAnalyzer()
            if TRANSFORMERS_AVAILABLE:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )

    def analyze_vader(self, text: str) -> Dict:
        """
        VADER sentiment analysis (rule-based, good for social media)

        Returns:
            Dict with compound, positive, negative, neutral scores
        """
        scores = self.vader.polarity_scores(text)

        # Classify based on compound score
        if scores['compound'] >= 0.05:
            sentiment = 'POSITIVE'
        elif scores['compound'] <= -0.05:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'

        return {
            'sentiment': sentiment,
            'confidence': abs(scores['compound']),
            'scores': scores,
            'method': 'VADER'
        }

    def analyze_textblob(self, text: str) -> Dict:
        """
        TextBlob sentiment analysis (pattern-based)

        Returns:
            Dict with polarity and subjectivity
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Classify based on polarity
        if polarity > 0.1:
            sentiment = 'POSITIVE'
        elif polarity < -0.1:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'

        return {
            'sentiment': sentiment,
            'confidence': abs(polarity),
            'polarity': polarity,
            'subjectivity': subjectivity,
            'method': 'TextBlob'
        }

    def analyze_transformer(self, text: str) -> Dict:
        """
        Transformer-based sentiment analysis (state-of-the-art)

        Returns:
            Dict with sentiment and confidence
        """
        # Truncate long texts
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]

        result = self.sentiment_pipeline(text)[0]

        return {
            'sentiment': result['label'].upper(),
            'confidence': result['score'],
            'method': 'Transformer'
        }

    def analyze_ensemble(self, text: str) -> Dict:
        """
        Ensemble of multiple methods (voting)

        Returns:
            Combined sentiment analysis
        """
        results = []

        # VADER
        if VADER_AVAILABLE:
            results.append(self.analyze_vader(text))

        # TextBlob
        if TEXTBLOB_AVAILABLE:
            results.append(self.analyze_textblob(text))

        # Transformer
        if TRANSFORMERS_AVAILABLE:
            results.append(self.analyze_transformer(text))

        # Voting
        sentiments = [r['sentiment'] for r in results]
        sentiment_counts = Counter(sentiments)
        final_sentiment = sentiment_counts.most_common(1)[0][0]

        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in results])

        return {
            'sentiment': final_sentiment,
            'confidence': avg_confidence,
            'individual_results': results,
            'method': 'Ensemble'
        }

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment using selected method

        Args:
            text: Input text

        Returns:
            Sentiment analysis result
        """
        if self.method == 'vader':
            return self.analyze_vader(text)
        elif self.method == 'textblob':
            return self.analyze_textblob(text)
        elif self.method == 'transformer':
            return self.analyze_transformer(text)
        elif self.method == 'ensemble':
            return self.analyze_ensemble(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def analyze_batch(self, texts: List[str], show_progress=True) -> List[Dict]:
        """
        Analyze multiple texts

        Args:
            texts: List of texts
            show_progress: Show progress bar

        Returns:
            List of sentiment results
        """
        results = []

        for idx, text in enumerate(texts):
            if show_progress and (idx + 1) % 100 == 0:
                print(f"📊 Processed {idx + 1}/{len(texts)} texts")

            result = self.analyze(text)
            result['text'] = text
            results.append(result)

        return results

    def analyze_aspects(self, text: str, aspects: List[str]) -> Dict:
        """
        Aspect-based sentiment analysis

        Args:
            text: Input text
            aspects: List of aspects to analyze (e.g., ['food', 'service', 'price'])

        Returns:
            Sentiment for each aspect
        """
        results = {}

        for aspect in aspects:
            # Find sentences containing aspect
            sentences = text.split('.')
            aspect_sentences = [s for s in sentences if aspect.lower() in s.lower()]

            if aspect_sentences:
                # Analyze sentiment of aspect sentences
                aspect_text = '. '.join(aspect_sentences)
                sentiment = self.analyze(aspect_text)
                results[aspect] = sentiment
            else:
                results[aspect] = {
                    'sentiment': 'NOT_MENTIONED',
                    'confidence': 0.0
                }

        return results

    def get_emotions(self, text: str) -> Dict:
        """
        Emotion detection using transformer model

        Args:
            text: Input text

        Returns:
            Detected emotions with scores
        """
        if not TRANSFORMERS_AVAILABLE:
            return {'error': 'Transformers not available'}

        try:
            emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )

            emotions = emotion_pipeline(text[:512])[0]
            return {e['label']: e['score'] for e in emotions}

        except Exception as e:
            return {'error': str(e)}

    def compare_methods(self, texts: List[str]) -> pd.DataFrame:
        """
        Compare all available methods on a dataset

        Args:
            texts: List of texts

        Returns:
            DataFrame with comparison
        """
        results = []

        for text in texts[:min(100, len(texts))]:  # Limit for speed
            row = {'text': text[:50] + '...'}

            if VADER_AVAILABLE:
                vader_result = self.analyze_vader(text)
                row['VADER'] = vader_result['sentiment']
                row['VADER_conf'] = vader_result['confidence']

            if TEXTBLOB_AVAILABLE:
                tb_result = self.analyze_textblob(text)
                row['TextBlob'] = tb_result['sentiment']
                row['TextBlob_conf'] = tb_result['confidence']

            if TRANSFORMERS_AVAILABLE:
                trans_result = self.analyze_transformer(text)
                row['Transformer'] = trans_result['sentiment']
                row['Transformer_conf'] = trans_result['confidence']

            results.append(row)

        return pd.DataFrame(results)

    def visualize_results(self, results: List[Dict], save_path=None):
        """
        Visualize sentiment analysis results

        Args:
            results: List of sentiment results
            save_path: Path to save plot
        """
        sentiments = [r['sentiment'] for r in results]
        confidences = [r['confidence'] for r in results]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sentiment distribution
        sentiment_counts = Counter(sentiments)
        axes[0, 0].bar(sentiment_counts.keys(), sentiment_counts.values(),
                      color=['green', 'red', 'gray'])
        axes[0, 0].set_title('Sentiment Distribution')
        axes[0, 0].set_ylabel('Count')

        # Confidence distribution
        axes[0, 1].hist(confidences, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')

        # Sentiment by confidence
        df = pd.DataFrame({'sentiment': sentiments, 'confidence': confidences})
        df.boxplot(by='sentiment', ax=axes[1, 0])
        axes[1, 0].set_title('Confidence by Sentiment')
        axes[1, 0].set_xlabel('Sentiment')
        axes[1, 0].set_ylabel('Confidence')

        # Sentiment percentage pie chart
        axes[1, 1].pie(sentiment_counts.values(), labels=sentiment_counts.keys(),
                      autopct='%1.1f%%', colors=['green', 'red', 'gray'])
        axes[1, 1].set_title('Sentiment Percentage')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")

        plt.show()

    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Get statistics from sentiment results

        Args:
            results: List of sentiment results

        Returns:
            Statistics dictionary
        """
        sentiments = [r['sentiment'] for r in results]
        confidences = [r['confidence'] for r in results]

        sentiment_counts = Counter(sentiments)

        return {
            'total': len(results),
            'positive': sentiment_counts.get('POSITIVE', 0),
            'negative': sentiment_counts.get('NEGATIVE', 0),
            'neutral': sentiment_counts.get('NEUTRAL', 0),
            'positive_pct': sentiment_counts.get('POSITIVE', 0) / len(results) * 100,
            'negative_pct': sentiment_counts.get('NEGATIVE', 0) / len(results) * 100,
            'neutral_pct': sentiment_counts.get('NEUTRAL', 0) / len(results) * 100,
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'method': self.method
        }


def main():
    parser = argparse.ArgumentParser(description='Advanced Sentiment Analysis v2.0')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File with texts (one per line)')
    parser.add_argument('--csv', type=str, help='CSV file with text column')
    parser.add_argument('--text-col', type=str, default='text',
                       help='Column name for text in CSV')
    parser.add_argument('--method', type=str, default='transformer',
                       choices=['vader', 'textblob', 'transformer', 'ensemble'],
                       help='Sentiment analysis method')
    parser.add_argument('--model', type=str,
                       default='distilbert-base-uncased-finetuned-sst-2-english',
                       help='Transformer model name')
    parser.add_argument('--aspects', type=str, nargs='+',
                       help='Aspects for aspect-based sentiment')
    parser.add_argument('--emotions', action='store_true',
                       help='Detect emotions')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all methods')
    parser.add_argument('--output', type=str,
                       help='Output visualization path')

    args = parser.parse_args()

    # Get texts
    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.csv:
        df = pd.read_csv(args.csv)
        texts = df[args.text_col].tolist()
    else:
        print("❌ Provide --text, --file, or --csv")
        return

    print(f"📊 Analyzing {len(texts)} text(s) with method: {args.method}\n")

    # Initialize analyzer
    analyzer = SentimentAnalyzer(method=args.method, model_name=args.model)

    # Compare methods
    if args.compare:
        print("🔍 Comparing all methods...\n")
        comparison = analyzer.compare_methods(texts)
        print(comparison)
        return

    # Analyze
    if len(texts) == 1:
        # Single text analysis
        result = analyzer.analyze(texts[0])

        print("=" * 60)
        print(f"📝 Text: {texts[0][:100]}...")
        print("=" * 60)
        print(f"😊 Sentiment: {result['sentiment']}")
        print(f"📊 Confidence: {result['confidence']:.4f}")
        print(f"🔧 Method: {result['method']}")

        # Aspect-based
        if args.aspects:
            print(f"\n🔍 Aspect-Based Sentiment:")
            aspect_results = analyzer.analyze_aspects(texts[0], args.aspects)
            for aspect, asp_result in aspect_results.items():
                print(f"  {aspect}: {asp_result['sentiment']} (conf: {asp_result['confidence']:.4f})")

        # Emotions
        if args.emotions:
            print(f"\n😊 Emotion Detection:")
            emotions = analyzer.get_emotions(texts[0])
            if 'error' not in emotions:
                for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {emotion}: {score:.4f}")

    else:
        # Batch analysis
        results = analyzer.analyze_batch(texts)

        # Statistics
        stats = analyzer.get_statistics(results)

        print("\n" + "=" * 60)
        print("📊 SENTIMENT ANALYSIS STATISTICS")
        print("=" * 60)
        print(f"Total texts: {stats['total']}")
        print(f"Positive: {stats['positive']} ({stats['positive_pct']:.1f}%)")
        print(f"Negative: {stats['negative']} ({stats['negative_pct']:.1f}%)")
        print(f"Neutral: {stats['neutral']} ({stats['neutral_pct']:.1f}%)")
        print(f"Avg confidence: {stats['avg_confidence']:.4f}")
        print(f"Method: {stats['method']}")
        print("=" * 60)

        # Visualize
        if args.output:
            analyzer.visualize_results(results, save_path=args.output)

        # Show sample results
        print("\n📋 Sample Results:")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. {result['text'][:80]}...")
            print(f"   Sentiment: {result['sentiment']} (confidence: {result['confidence']:.4f})")


if __name__ == "__main__":
    main()
