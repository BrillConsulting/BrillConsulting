"""
Model Interpretation Implementation
Author: BrillConsulting
Description: Advanced neural network interpretability techniques - SHAP, LIME, GradCAM, and attention visualization
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class GradCAMInterpreter:
    """Gradient-weighted Class Activation Mapping"""

    def __init__(self):
        """Initialize GradCAM interpreter"""
        self.activation_maps = []

    def compute_gradcam(self, feature_maps: np.ndarray,
                       gradients: np.ndarray) -> np.ndarray:
        """
        Compute GradCAM heatmap

        Args:
            feature_maps: Feature maps from target layer
            gradients: Gradients of target class w.r.t. feature maps

        Returns:
            GradCAM heatmap
        """
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted combination of feature maps
        cam = np.zeros(feature_maps.shape[1:3])
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]

        # ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def gradcam_plusplus(self, feature_maps: np.ndarray,
                        gradients: np.ndarray) -> np.ndarray:
        """
        GradCAM++ - improved version

        Args:
            feature_maps: Feature maps from target layer
            gradients: Gradients of target class w.r.t. feature maps

        Returns:
            GradCAM++ heatmap
        """
        # Calculate alpha weights
        grad_squared = gradients ** 2
        grad_cubed = grad_squared * gradients

        alpha = grad_squared / (2 * grad_squared + np.sum(grad_cubed, axis=(1, 2), keepdims=True) + 1e-10)

        # Weight gradients
        weights = np.sum(alpha * np.maximum(gradients, 0), axis=(1, 2))

        # Weighted combination
        cam = np.zeros(feature_maps.shape[1:3])
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]

        cam = np.maximum(cam, 0)

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


class SHAPInterpreter:
    """SHAP (SHapley Additive exPlanations) interpreter"""

    def __init__(self, num_features: int):
        """
        Initialize SHAP interpreter

        Args:
            num_features: Number of input features
        """
        self.num_features = num_features
        self.shapley_values = []

    def compute_shapley_values(self, instance: np.ndarray,
                              baseline: np.ndarray,
                              num_samples: int = 100) -> np.ndarray:
        """
        Compute approximate Shapley values

        Args:
            instance: Input instance
            baseline: Baseline/reference instance
            num_samples: Number of samples for approximation

        Returns:
            Shapley values for each feature
        """
        shapley_values = np.zeros(self.num_features)

        for _ in range(num_samples):
            # Random permutation
            perm = np.random.permutation(self.num_features)

            # Build coalition
            coalition = baseline.copy()
            prev_pred = self._predict(coalition)

            for feature_idx in perm:
                coalition[feature_idx] = instance[feature_idx]
                curr_pred = self._predict(coalition)

                # Marginal contribution
                shapley_values[feature_idx] += (curr_pred - prev_pred)
                prev_pred = curr_pred

        shapley_values /= num_samples

        return shapley_values

    def _predict(self, x: np.ndarray) -> float:
        """
        Simulated model prediction

        Args:
            x: Input features

        Returns:
            Prediction score
        """
        # Simulate prediction (linear + non-linear)
        return np.sum(x * np.random.rand(len(x))) + 0.1 * np.sum(x ** 2)

    def kernel_shap(self, instance: np.ndarray,
                   num_samples: int = 100) -> Dict[str, Any]:
        """
        Kernel SHAP approximation

        Args:
            instance: Input instance
            num_samples: Number of samples

        Returns:
            SHAP explanation
        """
        # Generate samples
        samples = np.random.binomial(1, 0.5, (num_samples, self.num_features))

        # Calculate weights
        num_present = np.sum(samples, axis=1)
        weights = (self.num_features - 1) / (num_present * (self.num_features - num_present) + 1e-10)

        # Get predictions
        baseline = np.zeros(self.num_features)
        baseline_pred = self._predict(baseline)

        predictions = []
        for sample in samples:
            masked_instance = instance * sample
            predictions.append(self._predict(masked_instance) - baseline_pred)

        predictions = np.array(predictions)

        # Solve weighted linear regression
        shapley_values = np.linalg.lstsq(samples, predictions, rcond=None)[0]

        return {
            'shapley_values': shapley_values,
            'feature_importance': np.abs(shapley_values),
            'base_value': baseline_pred,
            'prediction': self._predict(instance)
        }


class LIMEInterpreter:
    """LIME (Local Interpretable Model-agnostic Explanations) interpreter"""

    def __init__(self, num_features: int):
        """
        Initialize LIME interpreter

        Args:
            num_features: Number of input features
        """
        self.num_features = num_features

    def explain_instance(self, instance: np.ndarray,
                        num_samples: int = 1000,
                        num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for instance

        Args:
            instance: Input instance to explain
            num_samples: Number of perturbed samples
            num_features: Number of top features to return

        Returns:
            LIME explanation
        """
        # Generate perturbed samples
        samples = instance + np.random.normal(0, 0.3, (num_samples, self.num_features))

        # Get predictions
        predictions = np.array([self._predict(s) for s in samples])

        # Calculate distances
        distances = np.linalg.norm(samples - instance, axis=1)
        kernel_width = np.sqrt(self.num_features) * 0.75
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        # Fit linear model
        from numpy.linalg import lstsq
        weighted_samples = samples * weights[:, np.newaxis]
        weighted_predictions = predictions * weights

        coefficients = lstsq(weighted_samples, weighted_predictions, rcond=None)[0]

        # Get top features
        top_indices = np.argsort(np.abs(coefficients))[-num_features:][::-1]

        explanation = {
            'coefficients': coefficients,
            'top_features': top_indices.tolist(),
            'top_values': coefficients[top_indices].tolist(),
            'prediction': self._predict(instance),
            'r2_score': self._calculate_r2(predictions, np.dot(samples, coefficients), weights)
        }

        return explanation

    def _predict(self, x: np.ndarray) -> float:
        """Simulated model prediction"""
        return np.sum(x * np.sin(np.arange(len(x)))) + np.sum(x ** 2) * 0.05

    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted R² score"""
        ss_res = np.sum(weights * (y_true - y_pred) ** 2)
        ss_tot = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))


class AttentionVisualizer:
    """Attention mechanism visualization"""

    def __init__(self):
        """Initialize attention visualizer"""
        self.attention_maps = []

    def visualize_self_attention(self, query: np.ndarray,
                                key: np.ndarray,
                                value: np.ndarray) -> Dict[str, Any]:
        """
        Visualize self-attention weights

        Args:
            query: Query matrix
            key: Key matrix
            value: Value matrix

        Returns:
            Attention visualization data
        """
        # Compute attention scores
        d_k = query.shape[-1]
        scores = np.matmul(query, key.T) / np.sqrt(d_k)

        # Softmax
        attention_weights = self._softmax(scores)

        # Apply attention
        output = np.matmul(attention_weights, value)

        visualization = {
            'attention_weights': attention_weights,
            'attention_entropy': self._calculate_entropy(attention_weights),
            'max_attention': np.max(attention_weights, axis=1),
            'attention_concentration': np.std(attention_weights, axis=1)
        }

        return visualization

    def multi_head_attention_analysis(self, num_heads: int,
                                     seq_length: int,
                                     embed_dim: int) -> Dict[str, Any]:
        """
        Analyze multi-head attention patterns

        Args:
            num_heads: Number of attention heads
            seq_length: Sequence length
            embed_dim: Embedding dimension

        Returns:
            Multi-head attention analysis
        """
        head_dim = embed_dim // num_heads
        heads_analysis = []

        for head in range(num_heads):
            # Simulate attention weights
            query = np.random.randn(seq_length, head_dim)
            key = np.random.randn(seq_length, head_dim)
            value = np.random.randn(seq_length, head_dim)

            attention_viz = self.visualize_self_attention(query, key, value)
            attention_viz['head_id'] = head

            heads_analysis.append(attention_viz)

        # Calculate head diversity
        all_weights = np.array([h['attention_weights'] for h in heads_analysis])
        head_diversity = np.mean(np.std(all_weights, axis=0))

        return {
            'num_heads': num_heads,
            'heads_analysis': heads_analysis,
            'head_diversity': head_diversity,
            'average_entropy': np.mean([h['attention_entropy'] for h in heads_analysis])
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        return -np.sum(probabilities * np.log(probabilities + 1e-10))


class ModelInterpretationManager:
    """Main model interpretation manager"""

    def __init__(self):
        """Initialize model interpretation manager"""
        self.gradcam = GradCAMInterpreter()
        self.interpretations = []

    def comprehensive_interpretation(self, input_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive model interpretation

        Args:
            input_config: Input configuration

        Returns:
            Interpretation results
        """
        print(f"\n{'='*60}")
        print("Comprehensive Model Interpretation")
        print(f"{'='*60}")

        num_features = input_config.get('num_features', 20)
        results = {}

        # 1. GradCAM
        print("\n1. Computing GradCAM visualization...")
        feature_maps = np.random.randn(64, 7, 7)
        gradients = np.random.randn(64, 7, 7)
        gradcam_map = self.gradcam.compute_gradcam(feature_maps, gradients)
        print(f"   GradCAM shape: {gradcam_map.shape}")
        print(f"   Max activation: {gradcam_map.max():.4f}")
        results['gradcam'] = {'shape': gradcam_map.shape, 'max': float(gradcam_map.max())}

        # 2. SHAP
        print("\n2. Computing SHAP values...")
        shap = SHAPInterpreter(num_features)
        instance = np.random.randn(num_features)
        baseline = np.zeros(num_features)
        shap_values = shap.compute_shapley_values(instance, baseline, num_samples=50)
        print(f"   Top features: {np.argsort(np.abs(shap_values))[-5:][::-1].tolist()}")
        results['shap'] = {'values': shap_values.tolist(), 'top_features': np.argsort(np.abs(shap_values))[-5:][::-1].tolist()}

        # 3. LIME
        print("\n3. Computing LIME explanation...")
        lime = LIMEInterpreter(num_features)
        lime_explanation = lime.explain_instance(instance, num_samples=500)
        print(f"   R² score: {lime_explanation['r2_score']:.4f}")
        print(f"   Top features: {lime_explanation['top_features'][:5]}")
        results['lime'] = lime_explanation

        # 4. Attention
        print("\n4. Analyzing attention patterns...")
        attention_viz = AttentionVisualizer()
        attention_analysis = attention_viz.multi_head_attention_analysis(
            num_heads=8, seq_length=10, embed_dim=512
        )
        print(f"   Heads: {attention_analysis['num_heads']}")
        print(f"   Head diversity: {attention_analysis['head_diversity']:.4f}")
        results['attention'] = {
            'num_heads': attention_analysis['num_heads'],
            'diversity': float(attention_analysis['head_diversity'])
        }

        print(f"\n{'='*60}")
        print("Interpretation completed!")
        print(f"{'='*60}")

        self.interpretations.append(results)

        return results

    def get_interpretation_code(self) -> str:
        """Generate interpretation implementation code"""

        code = """
import torch
import torch.nn as nn
import numpy as np

# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute GradCAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)
        heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()

# SHAP for neural networks
import shap

def explain_with_shap(model, X_test):
    explainer = shap.DeepExplainer(model, X_test[:100])
    shap_values = explainer.shap_values(X_test[:10])
    shap.summary_plot(shap_values, X_test[:10])

# LIME for neural networks
from lime import lime_tabular

def explain_with_lime(model, X_train, instance):
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        mode='classification',
        training_labels=None
    )

    exp = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=10
    )

    return exp

# Attention visualization
def visualize_attention(attention_weights, tokens):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 10))
    sns.heatmap(attention_weights, xticklabels=tokens,
                yticklabels=tokens, cmap='viridis')
    plt.title('Attention Weights')
    plt.show()
"""

        return code

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'interpretations': len(self.interpretations),
            'methods': ['GradCAM', 'SHAP', 'LIME', 'Attention'],
            'framework': 'Model Interpretation',
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate model interpretation"""

    print("="*60)
    print("Model Interpretation Demo")
    print("="*60)

    manager = ModelInterpretationManager()

    # Run comprehensive interpretation
    print("\n1. Running comprehensive interpretation...")
    results = manager.comprehensive_interpretation({
        'num_features': 30
    })

    # Show implementation code
    print("\n2. PyTorch implementation code:")
    code = manager.get_interpretation_code()
    print(code[:500] + "...\n")

    # Manager info
    print("\n3. Manager summary:")
    info = manager.get_manager_info()
    print(f"  Interpretations: {info['interpretations']}")
    print(f"  Methods: {', '.join(info['methods'])}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    demo()
