"""
Advanced Model Interpretability System v2.0
Author: BrillConsulting
Description: Production-ready model interpretability with SHAP, LIME, feature importance, and permutation importance
Version: 2.0 - Enhanced with multiple explanation methods, global/local interpretability, and comprehensive visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, List, Tuple, Optional, Any, Union
import argparse
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Try to import LIME
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ModelExplainer:
    """
    Advanced model interpretability system

    Features:
    - SHAP (SHapley Additive exPlanations) - Model-agnostic explanations
    - LIME (Local Interpretable Model-agnostic Explanations) - Local explanations
    - Feature Importance - Built-in model importance
    - Permutation Importance - Model-agnostic importance
    - Partial Dependence Plots - Feature effect visualization
    - Global & Local explanations
    - Multiple visualization types
    """

    def __init__(self, model: Any, X_train: np.ndarray, feature_names: Optional[List[str]] = None,
                 task_type: str = 'classification'):
        """
        Initialize model explainer

        Args:
            model: Trained sklearn model
            X_train: Training data used to fit the model
            feature_names: List of feature names
            task_type: 'classification' or 'regression'
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'Feature {i}' for i in range(X_train.shape[1])]
        self.task_type = task_type
        self.n_features = X_train.shape[1]

        # Explainers
        self.shap_explainer = None
        self.lime_explainer = None

        # Results
        self.feature_importance_results = None
        self.permutation_importance_results = None
        self.shap_values = None

    def _initialize_shap(self):
        """Initialize SHAP explainer"""
        if not SHAP_AVAILABLE:
            print("⚠️  SHAP not available. Install with: pip install shap")
            return False

        if self.shap_explainer is None:
            print("🔧 Initializing SHAP explainer...")

            # Choose appropriate explainer based on model type
            if hasattr(self.model, 'tree_') or 'Tree' in str(type(self.model)):
                # Tree-based models
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, 'coef_'):
                # Linear models
                self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
            else:
                # General models (kernel explainer - slower)
                print("   Using KernelExplainer (may be slow)...")
                self.shap_explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X_train, 100))

        return True

    def _initialize_lime(self):
        """Initialize LIME explainer"""
        if not LIME_AVAILABLE:
            print("⚠️  LIME not available. Install with: pip install lime")
            return False

        if self.lime_explainer is None:
            print("🔧 Initializing LIME explainer...")

            mode = 'classification' if self.task_type == 'classification' else 'regression'

            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                mode=mode,
                verbose=False
            )

        return True

    def get_feature_importance(self, top_k: int = 20) -> Dict:
        """
        Get built-in feature importance from tree-based models

        Returns:
            Dictionary with feature importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("⚠️  Model does not have feature_importances_ attribute")
            return {}

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]

        self.feature_importance_results = {
            'importances': importances,
            'indices': indices,
            'names': [self.feature_names[i] for i in indices],
            'top_k': top_k
        }

        return self.feature_importance_results

    def get_permutation_importance(self, X_test: np.ndarray, y_test: np.ndarray,
                                   n_repeats: int = 10, top_k: int = 20) -> Dict:
        """
        Calculate permutation importance (model-agnostic)

        Args:
            X_test: Test features
            y_test: Test labels
            n_repeats: Number of times to permute each feature
            top_k: Number of top features to return
        """
        print(f"🔧 Computing permutation importance (n_repeats={n_repeats})...")

        perm_importance = permutation_importance(
            self.model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        importances_mean = perm_importance.importances_mean
        importances_std = perm_importance.importances_std
        indices = np.argsort(importances_mean)[::-1][:top_k]

        self.permutation_importance_results = {
            'importances_mean': importances_mean,
            'importances_std': importances_std,
            'indices': indices,
            'names': [self.feature_names[i] for i in indices],
            'top_k': top_k
        }

        return self.permutation_importance_results

    def explain_with_shap(self, X_explain: np.ndarray, max_display: int = 20) -> Dict:
        """
        Generate SHAP explanations

        Args:
            X_explain: Data to explain (usually X_test)
            max_display: Maximum number of features to display

        Returns:
            Dictionary with SHAP values and metadata
        """
        if not self._initialize_shap():
            return {}

        print(f"🔧 Computing SHAP values for {X_explain.shape[0]} samples...")

        # Compute SHAP values
        try:
            shap_values = self.shap_explainer.shap_values(X_explain)
        except Exception as e:
            print(f"❌ Error computing SHAP values: {e}")
            return {}

        # For classification, shap_values might be a list (one per class)
        # Take the positive class (index 1) or first class
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        self.shap_values = {
            'values': shap_values,
            'X': X_explain,
            'feature_names': self.feature_names,
            'max_display': max_display
        }

        return self.shap_values

    def explain_instance_with_lime(self, instance: np.ndarray, instance_idx: int = 0,
                                    num_features: int = 10) -> Dict:
        """
        Explain a single instance using LIME

        Args:
            instance: Single instance to explain (1D array)
            instance_idx: Instance index (for labeling)
            num_features: Number of top features to show

        Returns:
            Dictionary with LIME explanation
        """
        if not self._initialize_lime():
            return {}

        print(f"🔧 Generating LIME explanation for instance {instance_idx}...")

        # Reshape if needed
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        try:
            if self.task_type == 'classification':
                explanation = self.lime_explainer.explain_instance(
                    instance[0],
                    self.model.predict_proba,
                    num_features=num_features
                )
            else:
                explanation = self.lime_explainer.explain_instance(
                    instance[0],
                    self.model.predict,
                    num_features=num_features
                )

            lime_result = {
                'explanation': explanation,
                'instance_idx': instance_idx,
                'num_features': num_features
            }

            return lime_result

        except Exception as e:
            print(f"❌ Error generating LIME explanation: {e}")
            return {}

    def plot_feature_importance(self, save_path: Optional[str] = None):
        """Plot built-in feature importance"""
        if self.feature_importance_results is None:
            print("⚠️  Run get_feature_importance() first")
            return

        importances = self.feature_importance_results['importances']
        indices = self.feature_importance_results['indices']
        names = self.feature_importance_results['names']

        plt.figure(figsize=(10, max(6, len(indices) * 0.3)))
        plt.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.7)
        plt.yticks(range(len(indices)), names)
        plt.xlabel('Importance')
        plt.title('Feature Importance (Built-in)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance plot saved to {save_path}")

        plt.show()

    def plot_permutation_importance(self, save_path: Optional[str] = None):
        """Plot permutation importance with error bars"""
        if self.permutation_importance_results is None:
            print("⚠️  Run get_permutation_importance() first")
            return

        importances_mean = self.permutation_importance_results['importances_mean']
        importances_std = self.permutation_importance_results['importances_std']
        indices = self.permutation_importance_results['indices']
        names = self.permutation_importance_results['names']

        plt.figure(figsize=(10, max(6, len(indices) * 0.3)))
        plt.barh(range(len(indices)), importances_mean[indices],
                xerr=importances_std[indices], color='coral', alpha=0.7)
        plt.yticks(range(len(indices)), names)
        plt.xlabel('Importance (with std)')
        plt.title('Permutation Importance (Model-Agnostic)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Permutation importance plot saved to {save_path}")

        plt.show()

    def plot_shap_summary(self, save_path: Optional[str] = None):
        """Plot SHAP summary plot (global explanation)"""
        if self.shap_values is None or not SHAP_AVAILABLE:
            print("⚠️  Run explain_with_shap() first")
            return

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values['values'],
            self.shap_values['X'],
            feature_names=self.feature_names,
            max_display=self.shap_values['max_display'],
            show=False
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 SHAP summary plot saved to {save_path}")

        plt.show()

    def plot_shap_bar(self, save_path: Optional[str] = None):
        """Plot SHAP bar plot (mean absolute SHAP values)"""
        if self.shap_values is None or not SHAP_AVAILABLE:
            print("⚠️  Run explain_with_shap() first")
            return

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values['values'],
            self.shap_values['X'],
            feature_names=self.feature_names,
            max_display=self.shap_values['max_display'],
            plot_type='bar',
            show=False
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 SHAP bar plot saved to {save_path}")

        plt.show()

    def plot_shap_waterfall(self, instance_idx: int = 0, save_path: Optional[str] = None):
        """Plot SHAP waterfall plot for a single instance"""
        if self.shap_values is None or not SHAP_AVAILABLE:
            print("⚠️  Run explain_with_shap() first")
            return

        # Create explanation object for waterfall plot
        explanation = shap.Explanation(
            values=self.shap_values['values'][instance_idx],
            base_values=self.shap_explainer.expected_value if hasattr(self.shap_explainer, 'expected_value') else 0,
            data=self.shap_values['X'][instance_idx],
            feature_names=self.feature_names
        )

        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 SHAP waterfall plot saved to {save_path}")

        plt.show()

    def plot_lime_explanation(self, lime_result: Dict, save_path: Optional[str] = None):
        """Plot LIME explanation"""
        if not lime_result or 'explanation' not in lime_result:
            print("⚠️  Provide valid LIME result from explain_instance_with_lime()")
            return

        explanation = lime_result['explanation']

        plt.figure(figsize=(10, 6))
        explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation - Instance {lime_result['instance_idx']}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 LIME explanation saved to {save_path}")

        plt.show()

    def plot_partial_dependence(self, features: List[int], X_data: np.ndarray,
                                 save_path: Optional[str] = None):
        """
        Plot partial dependence for selected features

        Args:
            features: List of feature indices to plot
            X_data: Data to compute PDP on (usually X_train or X_test)
        """
        print(f"🔧 Computing partial dependence for {len(features)} features...")

        fig, ax = plt.subplots(figsize=(14, 4 * ((len(features) + 2) // 3)))

        try:
            display = PartialDependenceDisplay.from_estimator(
                self.model,
                X_data,
                features=features,
                feature_names=self.feature_names,
                ax=ax,
                n_jobs=-1
            )

            plt.suptitle('Partial Dependence Plots', fontsize=16, y=1.00)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Partial dependence plots saved to {save_path}")

            plt.show()

        except Exception as e:
            print(f"❌ Error creating partial dependence plots: {e}")

    def compare_importance_methods(self, save_path: Optional[str] = None):
        """Compare different importance methods side by side"""
        methods = {}

        if self.feature_importance_results is not None:
            methods['Feature Importance'] = (
                self.feature_importance_results['importances'],
                self.feature_importance_results['indices'][:10]
            )

        if self.permutation_importance_results is not None:
            methods['Permutation Importance'] = (
                self.permutation_importance_results['importances_mean'],
                self.permutation_importance_results['indices'][:10]
            )

        if not methods:
            print("⚠️  No importance methods available. Run get_feature_importance() or get_permutation_importance() first")
            return

        fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
        if len(methods) == 1:
            axes = [axes]

        for idx, (method_name, (importances, indices)) in enumerate(methods.items()):
            names = [self.feature_names[i] for i in indices]
            axes[idx].barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.7)
            axes[idx].set_yticks(range(len(indices)))
            axes[idx].set_yticklabels(names)
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(method_name)
            axes[idx].invert_yaxis()
            axes[idx].grid(True, alpha=0.3, axis='x')

        plt.suptitle('Importance Method Comparison', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Importance comparison saved to {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Model Interpretability v2.0')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pkl)')
    parser.add_argument('--data-train', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--data-test', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'],
                        help='Task type')
    parser.add_argument('--top-k', type=int, default=20, help='Number of top features to show')

    args = parser.parse_args()

    # Load model
    print(f"📂 Loading model from {args.model}...")
    model = joblib.load(args.model)

    # Load data
    print(f"📂 Loading training data from {args.data_train}...")
    df_train = pd.read_csv(args.data_train)
    X_train = df_train.drop(columns=[args.target]).values
    y_train = df_train[args.target].values
    feature_names = df_train.drop(columns=[args.target]).columns.tolist()

    print(f"📂 Loading test data from {args.data_test}...")
    df_test = pd.read_csv(args.data_test)
    X_test = df_test.drop(columns=[args.target]).values
    y_test = df_test[args.target].values

    print(f"📊 Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"📊 Features: {X_train.shape[1]}\n")

    # Initialize explainer
    explainer = ModelExplainer(model, X_train, feature_names=feature_names, task_type=args.task)

    # Feature importance (if available)
    print("=" * 80)
    print("1️⃣  Feature Importance (Built-in)")
    print("=" * 80)
    try:
        explainer.get_feature_importance(top_k=args.top_k)
        explainer.plot_feature_importance()
    except:
        print("⚠️  Feature importance not available for this model\n")

    # Permutation importance
    print("=" * 80)
    print("2️⃣  Permutation Importance (Model-Agnostic)")
    print("=" * 80)
    explainer.get_permutation_importance(X_test, y_test, n_repeats=10, top_k=args.top_k)
    explainer.plot_permutation_importance()

    # SHAP
    if SHAP_AVAILABLE:
        print("\n" + "=" * 80)
        print("3️⃣  SHAP Explanations")
        print("=" * 80)
        explainer.explain_with_shap(X_test[:100], max_display=args.top_k)  # Use first 100 samples
        explainer.plot_shap_summary()
        explainer.plot_shap_bar()
        explainer.plot_shap_waterfall(instance_idx=0)

    # LIME
    if LIME_AVAILABLE:
        print("\n" + "=" * 80)
        print("4️⃣  LIME Explanation (Single Instance)")
        print("=" * 80)
        lime_result = explainer.explain_instance_with_lime(X_test[0], instance_idx=0, num_features=10)
        if lime_result:
            explainer.plot_lime_explanation(lime_result)

    # Partial dependence
    print("\n" + "=" * 80)
    print("5️⃣  Partial Dependence Plots")
    print("=" * 80)
    # Plot for top 3 most important features
    if explainer.permutation_importance_results:
        top_features = explainer.permutation_importance_results['indices'][:3].tolist()
        explainer.plot_partial_dependence(top_features, X_test[:500])

    # Comparison
    print("\n" + "=" * 80)
    print("6️⃣  Importance Method Comparison")
    print("=" * 80)
    explainer.compare_importance_methods()

    print("\n✅ Model interpretability analysis completed successfully!")


if __name__ == "__main__":
    main()
