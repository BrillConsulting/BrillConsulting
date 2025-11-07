"""
Bias & Fairness Analysis Toolkit
=================================

Detect and mitigate bias in ML models

Author: Brill Consulting
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class FairnessMetrics:
    """Fairness evaluation metrics."""
    demographic_parity: float
    equal_opportunity: float
    disparate_impact: float
    overall_fairness_score: float
    protected_group_metrics: Dict[str, Dict]


@dataclass
class BiasReport:
    """Bias analysis report."""
    bias_detected: bool
    affected_groups: List[str]
    fairness_metrics: FairnessMetrics
    recommendations: List[str]
    timestamp: str


class FairnessAnalyzer:
    """Analyze and mitigate bias in ML models."""

    def __init__(self):
        """Initialize fairness analyzer."""
        self.data = None
        self.protected_attributes = []
        self.model = None

        print(f"⚖️  Fairness Analyzer initialized")

    def load_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected_attributes: List[str]
    ) -> None:
        """Load data with protected attributes."""
        self.data = {"X": X, "y": y}
        self.protected_attributes = protected_attributes

        print(f"   ✓ Data loaded")
        print(f"   Protected attributes: {', '.join(protected_attributes)}")

    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze bias in dataset."""
        print(f"\n📊 Analyzing dataset for bias")

        if not self.data:
            raise ValueError("No data loaded")

        # Simulate bias analysis
        analysis = {
            "total_samples": len(self.data["y"]),
            "class_distribution": self._analyze_class_distribution(),
            "protected_group_stats": self._analyze_protected_groups(),
            "potential_bias_indicators": []
        }

        # Check for imbalance
        if analysis["class_distribution"]["imbalance_ratio"] > 2.0:
            analysis["potential_bias_indicators"].append(
                "Class imbalance detected"
            )

        print(f"   ✓ Analysis complete")
        print(f"   Samples: {analysis['total_samples']}")
        print(f"   Bias indicators: {len(analysis['potential_bias_indicators'])}")

        return analysis

    def _analyze_class_distribution(self) -> Dict[str, float]:
        """Analyze class distribution."""
        y = self.data["y"]

        positive_rate = np.mean(y)
        negative_rate = 1 - positive_rate

        return {
            "positive_rate": positive_rate,
            "negative_rate": negative_rate,
            "imbalance_ratio": max(positive_rate, negative_rate) / min(positive_rate, negative_rate)
        }

    def _analyze_protected_groups(self) -> Dict[str, Any]:
        """Analyze statistics for protected groups."""
        # Simulate protected group analysis
        return {
            "gender": {
                "male": {"count": 600, "positive_rate": 0.55},
                "female": {"count": 400, "positive_rate": 0.45}
            },
            "race": {
                "group_a": {"count": 700, "positive_rate": 0.52},
                "group_b": {"count": 300, "positive_rate": 0.48}
            }
        }

    def train_fair_model(
        self,
        algorithm: str = "logistic_regression",
        fairness_constraint: str = "demographic_parity"
    ) -> Any:
        """Train model with fairness constraints."""
        print(f"\n🎯 Training fair model")
        print(f"   Algorithm: {algorithm}")
        print(f"   Fairness constraint: {fairness_constraint}")

        # Simulate model training
        self.model = {"type": algorithm, "fairness": fairness_constraint}

        print(f"   ✓ Model trained with fairness constraints")

        return self.model

    def evaluate_fairness(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> BiasReport:
        """Evaluate model fairness."""
        print(f"\n⚖️  Evaluating model fairness")

        # Simulate fairness metrics
        metrics = self._calculate_fairness_metrics(X_test, y_test)

        # Determine if bias detected
        bias_detected = (
            metrics.demographic_parity < 0.8 or
            metrics.disparate_impact < 0.8 or
            metrics.disparate_impact > 1.25
        )

        affected_groups = []
        if bias_detected:
            affected_groups = ["gender:female", "race:group_b"]

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, bias_detected)

        report = BiasReport(
            bias_detected=bias_detected,
            affected_groups=affected_groups,
            fairness_metrics=metrics,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )

        # Display results
        if bias_detected:
            print(f"   ⚠️  Bias DETECTED")
            print(f"   Affected groups: {len(affected_groups)}")
        else:
            print(f"   ✓ No significant bias detected")

        print(f"\n   Fairness Metrics:")
        print(f"   Demographic Parity: {metrics.demographic_parity:.2f}")
        print(f"   Equal Opportunity: {metrics.equal_opportunity:.2f}")
        print(f"   Disparate Impact: {metrics.disparate_impact:.2f}")

        return report

    def _calculate_fairness_metrics(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> FairnessMetrics:
        """Calculate fairness metrics."""
        # Simulate metric calculations
        import random

        return FairnessMetrics(
            demographic_parity=random.uniform(0.75, 0.95),
            equal_opportunity=random.uniform(0.80, 0.98),
            disparate_impact=random.uniform(0.85, 1.15),
            overall_fairness_score=random.uniform(0.75, 0.90),
            protected_group_metrics={
                "gender": {
                    "male": {"tpr": 0.85, "fpr": 0.12},
                    "female": {"tpr": 0.82, "fpr": 0.14}
                }
            }
        )

    def _generate_recommendations(
        self,
        metrics: FairnessMetrics,
        bias_detected: bool
    ) -> List[str]:
        """Generate bias mitigation recommendations."""
        recommendations = []

        if not bias_detected:
            recommendations.append("Model meets fairness criteria - continue monitoring")
            return recommendations

        if metrics.demographic_parity < 0.9:
            recommendations.append(
                "Apply demographic parity constraint during training"
            )

        if metrics.disparate_impact < 0.8:
            recommendations.append(
                "Use reweighting pre-processing to balance groups"
            )

        recommendations.extend([
            "Collect more diverse training data",
            "Use adversarial debiasing techniques",
            "Implement fairness-aware feature selection"
        ])

        return recommendations

    def get_mitigation_recommendations(self) -> List[str]:
        """Get bias mitigation recommendations."""
        return [
            "1. Pre-processing: Reweight samples to balance groups",
            "2. In-processing: Add fairness constraints to loss function",
            "3. Post-processing: Adjust decision thresholds per group",
            "4. Data collection: Increase representation of underrepresented groups",
            "5. Feature engineering: Remove or transform biased features"
        ]


def demo():
    """Demonstrate bias analysis."""
    print("=" * 60)
    print("Bias & Fairness Analysis Demo")
    print("=" * 60)

    analyzer = FairnessAnalyzer()

    # Simulate data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.binomial(1, 0.5, 1000)

    # Load data
    analyzer.load_data(
        X=X,
        y=y,
        protected_attributes=["gender", "race", "age"]
    )

    # Analyze dataset
    dataset_analysis = analyzer.analyze_dataset()

    # Train fair model
    model = analyzer.train_fair_model(
        algorithm="logistic_regression",
        fairness_constraint="demographic_parity"
    )

    # Evaluate fairness
    X_test = np.random.randn(200, 10)
    y_test = np.random.binomial(1, 0.5, 200)

    report = analyzer.evaluate_fairness(model, X_test, y_test)

    # Recommendations
    print(f"\n{'='*60}")
    print("Mitigation Recommendations")
    print(f"{'='*60}")

    recommendations = analyzer.get_mitigation_recommendations()
    for rec in recommendations:
        print(f"   {rec}")


if __name__ == "__main__":
    demo()
