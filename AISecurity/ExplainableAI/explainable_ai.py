"""
Explainable AI Dashboards
==========================

Model interpretability with SHAP, Captum, and interactive dashboards

Author: Brill Consulting
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class Explanation:
    """Model explanation result."""
    instance_id: str
    feature_values: Dict[str, float]
    feature_importance: Dict[str, float]
    prediction: float
    base_value: float
    timestamp: str


@dataclass
class GlobalExplanation:
    """Global model explanation."""
    feature_importance: Dict[str, float]
    top_features: List[str]
    model_type: str
    timestamp: str


class SHAPExplainer:
    """SHAP-based model explainer."""

    def __init__(self, model: Any = None):
        """Initialize SHAP explainer."""
        self.model = model
        self.explanations: List[Explanation] = []

        print(f"🔍 SHAP Explainer initialized")

    def explain(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Explanation:
        """Explain single prediction."""
        print(f"\n💡 Explaining prediction")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(instance))]

        # Simulate SHAP value calculation
        # In production: shap.TreeExplainer(model).shap_values(instance)
        shap_values = np.random.randn(len(instance)) * 0.5

        # Create feature importance dict
        feature_importance = {
            name: float(shap_val)
            for name, shap_val in zip(feature_names, shap_values)
        }

        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Create explanation
        explanation = Explanation(
            instance_id=f"exp_{len(self.explanations) + 1}",
            feature_values={
                name: float(val)
                for name, val in zip(feature_names, instance)
            },
            feature_importance=feature_importance,
            prediction=float(np.random.rand()),  # Simulated prediction
            base_value=0.5,
            timestamp=datetime.now().isoformat()
        )

        self.explanations.append(explanation)

        # Display top features
        print(f"   Top 3 important features:")
        for name, importance in sorted_features[:3]:
            sign = "+" if importance > 0 else ""
            print(f"      {name}: {sign}{importance:.3f}")

        return explanation

    def plot_force_plot(self, explanation: Explanation) -> None:
        """Visualize force plot."""
        print(f"\n📊 Force Plot")
        print(f"   Base value: {explanation.base_value:.3f}")
        print(f"   Prediction: {explanation.prediction:.3f}")

        # Show top positive and negative contributors
        sorted_imp = sorted(
            explanation.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"\n   Positive contributors:")
        for name, imp in sorted_imp[:3]:
            if imp > 0:
                print(f"      {name}: +{imp:.3f}")

        print(f"\n   Negative contributors:")
        for name, imp in reversed(sorted_imp[-3:]):
            if imp < 0:
                print(f"      {name}: {imp:.3f}")

    def plot_waterfall(self, explanation: Explanation) -> None:
        """Visualize waterfall chart."""
        print(f"\n📈 Waterfall Chart")

        cumulative = explanation.base_value

        print(f"   Base: {cumulative:.3f}")

        # Sort by absolute importance
        sorted_features = sorted(
            explanation.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for name, importance in sorted_features[:5]:
            cumulative += importance
            sign = "+" if importance > 0 else ""
            print(f"   {name}: {sign}{importance:.3f} → {cumulative:.3f}")

        print(f"   Final: {explanation.prediction:.3f}")

    def get_global_importance(
        self,
        feature_names: List[str]
    ) -> GlobalExplanation:
        """Get global feature importance."""
        print(f"\n🌍 Global Feature Importance")

        if not self.explanations:
            raise ValueError("No explanations available")

        # Aggregate SHAP values across all explanations
        aggregated_importance = {}

        for feat in feature_names:
            # Average absolute SHAP values
            values = [
                abs(exp.feature_importance.get(feat, 0))
                for exp in self.explanations
            ]
            aggregated_importance[feat] = np.mean(values)

        # Get top features
        top_features = sorted(
            aggregated_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        global_exp = GlobalExplanation(
            feature_importance=aggregated_importance,
            top_features=[name for name, _ in top_features],
            model_type="tree_based",
            timestamp=datetime.now().isoformat()
        )

        print(f"   Top 5 features globally:")
        for name, importance in top_features[:5]:
            print(f"      {name}: {importance:.3f}")

        return global_exp

    def generate_report(
        self,
        explanations: List[Explanation],
        output_path: str = "report.html"
    ) -> str:
        """Generate HTML explanation report."""
        print(f"\n📄 Generating explanation report")
        print(f"   Explanations: {len(explanations)}")
        print(f"   Output: {output_path}")

        # Simulate report generation
        print(f"   ✓ Report generated successfully")

        return output_path


class CaptumExplainer:
    """Captum-based explainer for PyTorch models."""

    def __init__(self, model: Any = None, method: str = "integrated_gradients"):
        """Initialize Captum explainer."""
        self.model = model
        self.method = method

        print(f"🔬 Captum Explainer initialized")
        print(f"   Method: {method}")

    def attribute(self, input_tensor: np.ndarray) -> np.ndarray:
        """Calculate feature attributions."""
        print(f"\n⚡ Computing attributions")
        print(f"   Method: {self.method}")

        # Simulate attribution calculation
        # In production: IntegratedGradients(model).attribute(input_tensor)
        attributions = np.random.randn(*input_tensor.shape) * 0.3

        print(f"   ✓ Attributions computed")

        return attributions

    def visualize_attributions(
        self,
        attributions: np.ndarray,
        input_tensor: np.ndarray
    ) -> None:
        """Visualize feature attributions."""
        print(f"\n📊 Attribution Visualization")

        # For demo: show top attributions
        flat_attrs = attributions.flatten()
        top_indices = np.argsort(np.abs(flat_attrs))[-5:]

        print(f"   Top 5 attributions:")
        for idx in reversed(top_indices):
            print(f"      Index {idx}: {flat_attrs[idx]:.3f}")


def demo():
    """Demonstrate explainable AI."""
    print("=" * 60)
    print("Explainable AI Dashboards Demo")
    print("=" * 60)

    # SHAP Explainer
    print(f"\n{'='*60}")
    print("SHAP Explanations")
    print(f"{'='*60}")

    explainer = SHAPExplainer()

    # Explain instances
    feature_names = ["age", "income", "credit_score", "debt_ratio", "employment_years"]
    instances = [
        np.array([35, 50000, 720, 0.3, 5]),
        np.array([25, 30000, 650, 0.5, 2]),
        np.array([45, 80000, 780, 0.2, 15])
    ]

    for i, instance in enumerate(instances, 1):
        print(f"\n--- Instance {i} ---")
        explanation = explainer.explain(instance, feature_names)

        # Visualizations
        explainer.plot_force_plot(explanation)
        explainer.plot_waterfall(explanation)

    # Global importance
    print(f"\n{'='*60}")
    print("Global Analysis")
    print(f"{'='*60}")

    global_exp = explainer.get_global_importance(feature_names)

    # Generate report
    print(f"\n{'='*60}")
    print("Report Generation")
    print(f"{'='*60}")

    report_path = explainer.generate_report(
        explanations=explainer.explanations,
        output_path="explanations_report.html"
    )

    # Captum for Deep Learning
    print(f"\n{'='*60}")
    print("Captum Deep Learning Explainer")
    print(f"{'='*60}")

    captum_explainer = CaptumExplainer(method="integrated_gradients")

    # Simulate tensor input
    input_tensor = np.random.randn(1, 3, 224, 224)  # Image-like input

    attributions = captum_explainer.attribute(input_tensor)
    captum_explainer.visualize_attributions(attributions, input_tensor)


if __name__ == "__main__":
    demo()
