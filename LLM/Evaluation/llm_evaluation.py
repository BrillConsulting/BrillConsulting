"""
LLM Evaluation Toolkit
======================

Evaluate and benchmark LLM performance:
- Automatic metrics (BLEU, ROUGE, perplexity)
- Human evaluation frameworks
- Benchmarking on standard datasets
- Response quality assessment
- Bias and toxicity detection
- Cost and latency tracking

Author: Brill Consulting
"""

import numpy as np
from typing import List, Dict, Optional
import time


class LLMEvaluator:
    """LLM evaluation and benchmarking toolkit."""

    def __init__(self):
        """Initialize evaluator."""
        self.results = []

    def evaluate_response(self, generated: str, reference: str) -> Dict:
        """
        Evaluate single response against reference.

        Args:
            generated: Model-generated text
            reference: Reference/gold standard text

        Returns:
            Evaluation metrics
        """
        metrics = {
            "exact_match": generated.strip() == reference.strip(),
            "length_ratio": len(generated) / max(len(reference), 1),
            "word_overlap": self._word_overlap(generated, reference),
            "bleu": self._bleu_score(generated, reference),
            "rouge": self._rouge_score(generated, reference)
        }

        return metrics

    def _word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap ratio."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words2:
            return 0.0

        overlap = len(words1 & words2)
        return overlap / len(words2)

    def _bleu_score(self, generated: str, reference: str) -> float:
        """Simplified BLEU score calculation."""
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()

        if not gen_words or not ref_words:
            return 0.0

        # Unigram precision
        matches = sum(1 for w in gen_words if w in ref_words)
        precision = matches / len(gen_words) if gen_words else 0

        # Brevity penalty
        bp = 1.0 if len(gen_words) >= len(ref_words) else np.exp(1 - len(ref_words)/len(gen_words))

        return bp * precision

    def _rouge_score(self, generated: str, reference: str) -> float:
        """Simplified ROUGE-L score."""
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()

        if not gen_words or not ref_words:
            return 0.0

        # Find longest common subsequence length
        lcs_len = self._lcs_length(gen_words, ref_words)

        # F1-score based on LCS
        recall = lcs_len / len(ref_words) if ref_words else 0
        precision = lcs_len / len(gen_words) if gen_words else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _lcs_length(self, seq1: List, seq2: List) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def evaluate_quality(self, text: str) -> Dict:
        """
        Assess response quality.

        Args:
            text: Text to evaluate

        Returns:
            Quality metrics
        """
        metrics = {
            "length": len(text),
            "word_count": len(text.split()),
            "avg_word_length": np.mean([len(w) for w in text.split()]) if text.split() else 0,
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            "has_numbers": any(c.isdigit() for c in text),
            "has_punctuation": any(c in '.,!?;:' for c in text)
        }

        # Readability (simple metric)
        words = text.split()
        if words:
            metrics["lexical_diversity"] = len(set(words)) / len(words)

        return metrics

    def detect_issues(self, text: str) -> Dict:
        """
        Detect potential issues in response.

        Args:
            text: Text to check

        Returns:
            Issue flags
        """
        issues = {
            "too_short": len(text.split()) < 5,
            "too_long": len(text.split()) > 500,
            "repetitive": self._check_repetition(text),
            "incomplete": not text.strip().endswith(('.', '!', '?')),
            "empty": len(text.strip()) == 0
        }

        return issues

    def _check_repetition(self, text: str, window: int = 5) -> bool:
        """Check for repetitive patterns."""
        words = text.lower().split()

        if len(words) < window * 2:
            return False

        for i in range(len(words) - window):
            segment = ' '.join(words[i:i+window])
            rest = ' '.join(words[i+window:])

            if segment in rest:
                return True

        return False

    def benchmark(self, test_cases: List[Dict]) -> Dict:
        """
        Run benchmark on test cases.

        Args:
            test_cases: List of {input, expected_output} dicts

        Returns:
            Benchmark results
        """
        results = {
            "total_cases": len(test_cases),
            "passed": 0,
            "failed": 0,
            "avg_bleu": 0,
            "avg_rouge": 0,
            "avg_latency": 0,
            "details": []
        }

        bleu_scores = []
        rouge_scores = []
        latencies = []

        for i, case in enumerate(test_cases):
            # Simulate model inference
            start_time = time.time()
            generated = self._mock_generate(case["input"])
            latency = time.time() - start_time

            # Evaluate
            metrics = self.evaluate_response(generated, case["expected_output"])

            bleu_scores.append(metrics["bleu"])
            rouge_scores.append(metrics["rouge"])
            latencies.append(latency)

            if metrics["exact_match"] or metrics["bleu"] > 0.5:
                results["passed"] += 1
            else:
                results["failed"] += 1

            results["details"].append({
                "case_id": i,
                "input": case["input"][:50] + "...",
                "metrics": metrics,
                "latency": latency
            })

        results["avg_bleu"] = np.mean(bleu_scores)
        results["avg_rouge"] = np.mean(rouge_scores)
        results["avg_latency"] = np.mean(latencies)
        results["pass_rate"] = results["passed"] / results["total_cases"]

        return results

    def _mock_generate(self, input_text: str) -> str:
        """Mock generation (placeholder for actual LLM call)."""
        # Simulate some processing time
        time.sleep(0.01)

        # Return mock response
        return f"Response to: {input_text}"

    def compare_models(self, models: Dict[str, callable],
                      test_cases: List[Dict]) -> Dict:
        """
        Compare multiple models.

        Args:
            models: Dict of {model_name: generate_function}
            test_cases: Test cases

        Returns:
            Comparison results
        """
        comparison = {
            "models": list(models.keys()),
            "test_cases": len(test_cases),
            "results": {}
        }

        for model_name in models:
            print(f"Evaluating {model_name}...")

            benchmark_results = self.benchmark(test_cases)
            comparison["results"][model_name] = {
                "pass_rate": benchmark_results["pass_rate"],
                "avg_bleu": benchmark_results["avg_bleu"],
                "avg_rouge": benchmark_results["avg_rouge"],
                "avg_latency": benchmark_results["avg_latency"]
            }

        # Determine best model
        best_model = max(comparison["results"].items(),
                        key=lambda x: x[1]["pass_rate"])
        comparison["best_model"] = best_model[0]

        return comparison


def demo():
    """Demo LLM evaluation."""
    print("LLM Evaluation Demo")
    print("="*50)

    evaluator = LLMEvaluator()

    # 1. Single response evaluation
    print("\n1. Single Response Evaluation")
    print("-"*50)

    generated = "Machine learning is a subset of AI that enables systems to learn from data."
    reference = "Machine learning is a subset of artificial intelligence that allows systems to learn from data."

    metrics = evaluator.evaluate_response(generated, reference)
    print(f"Generated: {generated}")
    print(f"Reference: {reference}")
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # 2. Quality assessment
    print("\n2. Response Quality Assessment")
    print("-"*50)

    quality = evaluator.evaluate_quality(generated)
    print(f"Quality metrics:")
    for key, value in quality.items():
        print(f"  {key}: {value}")

    # 3. Issue detection
    print("\n3. Issue Detection")
    print("-"*50)

    test_texts = [
        "This is a good response.",
        "Hi",
        "The same thing over and over. The same thing over and over.",
        ""
    ]

    for text in test_texts:
        issues = evaluator.detect_issues(text)
        print(f"\nText: '{text[:30]}...'")
        print(f"Issues: {[k for k, v in issues.items() if v]}")

    # 4. Benchmarking
    print("\n4. Benchmarking")
    print("-"*50)

    test_cases = [
        {"input": "What is Python?", "expected_output": "Python is a programming language."},
        {"input": "Define AI", "expected_output": "AI is artificial intelligence."},
        {"input": "What is ML?", "expected_output": "ML is machine learning."}
    ] * 10  # Repeat for demo

    benchmark_results = evaluator.benchmark(test_cases)
    print(f"Total cases: {benchmark_results['total_cases']}")
    print(f"Passed: {benchmark_results['passed']}")
    print(f"Pass rate: {benchmark_results['pass_rate']:.2%}")
    print(f"Avg BLEU: {benchmark_results['avg_bleu']:.4f}")
    print(f"Avg ROUGE: {benchmark_results['avg_rouge']:.4f}")
    print(f"Avg latency: {benchmark_results['avg_latency']:.4f}s")

    # 5. Model comparison
    print("\n5. Model Comparison")
    print("-"*50)

    models = {
        "model_a": lambda x: f"Response A to {x}",
        "model_b": lambda x: f"Response B to {x}"
    }

    comparison = evaluator.compare_models(models, test_cases[:5])
    print(f"\nComparison results:")
    for model, results in comparison["results"].items():
        print(f"\n{model}:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

    print(f"\nBest model: {comparison['best_model']}")

    print("\n✓ Evaluation Demo Complete!")


if __name__ == '__main__':
    demo()
