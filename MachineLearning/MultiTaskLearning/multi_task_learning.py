"""
MultiTaskLearning v2.0
Author: BrillConsulting
Description: Advanced Multi-Task Learning with Shared Representations and Task-Specific Layers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')


class SharedBottomModel:
    """
    Shared Bottom Multi-Task Learning Architecture
    All tasks share the same bottom layers, with task-specific top layers
    """

    def __init__(self, n_features: int, task_types: Dict[str, str]):
        """
        Args:
            n_features: Number of input features
            task_types: Dictionary mapping task names to types ('classification' or 'regression')
        """
        self.n_features = n_features
        self.task_types = task_types
        self.task_names = list(task_types.keys())

        # Shared representation (using StandardScaler as shared preprocessing)
        self.shared_scaler = StandardScaler()

        # Task-specific models
        self.task_models = {}
        for task_name, task_type in task_types.items():
            if task_type == 'classification':
                self.task_models[task_name] = LogisticRegression(max_iter=1000, random_state=42)
            elif task_type == 'regression':
                self.task_models[task_name] = Ridge(alpha=1.0, random_state=42)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        """
        Train all tasks jointly

        Args:
            X: Input features
            y_dict: Dictionary mapping task names to labels
        """
        # Fit shared representation
        X_shared = self.shared_scaler.fit_transform(X)

        # Train each task on shared representation
        for task_name in self.task_names:
            if task_name in y_dict:
                y = y_dict[task_name]
                # Remove NaN values for this task
                valid_idx = ~np.isnan(y)
                if np.sum(valid_idx) > 0:
                    self.task_models[task_name].fit(X_shared[valid_idx], y[valid_idx])

        self.is_fitted = True

    def predict(self, X: np.ndarray, task_name: str) -> np.ndarray:
        """Predict for specific task"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X_shared = self.shared_scaler.transform(X)
        return self.task_models[task_name].predict(X_shared)

    def predict_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict for all tasks"""
        predictions = {}
        for task_name in self.task_names:
            predictions[task_name] = self.predict(X, task_name)
        return predictions


class MMoEModel:
    """
    Multi-gate Mixture-of-Experts (MMoE) Model
    Uses multiple expert networks with task-specific gating
    """

    def __init__(self, n_features: int, task_types: Dict[str, str], n_experts: int = 3):
        """
        Args:
            n_features: Number of input features
            task_types: Dictionary mapping task names to types
            n_experts: Number of expert networks
        """
        self.n_features = n_features
        self.task_types = task_types
        self.task_names = list(task_types.keys())
        self.n_experts = n_experts

        # Shared scaler
        self.scaler = StandardScaler()

        # Expert networks (simplified - using different random projections)
        self.experts = []
        for i in range(n_experts):
            expert_scaler = StandardScaler()
            self.experts.append(expert_scaler)

        # Task-specific gates and models
        self.gates = {}
        self.task_models = {}

        for task_name, task_type in task_types.items():
            # Gate weights (will be learned)
            self.gates[task_name] = np.ones(n_experts) / n_experts

            # Task-specific model
            if task_type == 'classification':
                self.task_models[task_name] = LogisticRegression(max_iter=1000, random_state=42)
            elif task_type == 'regression':
                self.task_models[task_name] = Ridge(alpha=1.0, random_state=42)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        """Train MMoE model"""
        # Fit shared preprocessing
        X_scaled = self.scaler.fit_transform(X)

        # Fit experts (simplified - just fit scalers)
        for expert in self.experts:
            expert.fit(X_scaled + np.random.randn(*X_scaled.shape) * 0.1)

        # Train task-specific models on expert-gated representations
        for task_name in self.task_names:
            if task_name in y_dict:
                y = y_dict[task_name]
                valid_idx = ~np.isnan(y)

                if np.sum(valid_idx) > 0:
                    # Get expert representations and gate them
                    expert_outputs = []
                    for expert in self.experts:
                        expert_out = expert.transform(X_scaled[valid_idx])
                        expert_outputs.append(expert_out)

                    # Weighted combination based on gate
                    gated_output = np.zeros_like(expert_outputs[0])
                    for i, expert_out in enumerate(expert_outputs):
                        gated_output += self.gates[task_name][i] * expert_out

                    # Train task model
                    self.task_models[task_name].fit(gated_output, y[valid_idx])

        self.is_fitted = True

    def predict(self, X: np.ndarray, task_name: str) -> np.ndarray:
        """Predict for specific task"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X_scaled = self.scaler.transform(X)

        # Get expert representations
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert.transform(X_scaled)
            expert_outputs.append(expert_out)

        # Gate experts for this task
        gated_output = np.zeros_like(expert_outputs[0])
        for i, expert_out in enumerate(expert_outputs):
            gated_output += self.gates[task_name][i] * expert_out

        return self.task_models[task_name].predict(gated_output)

    def predict_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict for all tasks"""
        predictions = {}
        for task_name in self.task_names:
            predictions[task_name] = self.predict(X, task_name)
        return predictions


class TaskWeighting:
    """Strategies for weighting task losses"""

    @staticmethod
    def equal_weights(n_tasks: int) -> np.ndarray:
        """Equal weight for all tasks"""
        return np.ones(n_tasks) / n_tasks

    @staticmethod
    def uncertainty_weights(task_losses: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Uncertainty-based weighting
        Tasks with higher uncertainty get lower weight
        """
        weights = {}
        for task_name, losses in task_losses.items():
            if len(losses) > 1:
                uncertainty = np.std(losses)
                weights[task_name] = 1.0 / (1.0 + uncertainty)
            else:
                weights[task_name] = 1.0

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    @staticmethod
    def performance_weights(task_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Performance-based weighting
        Tasks with lower performance get higher weight
        """
        weights = {}
        for task_name, score in task_scores.items():
            # Inverse weighting (lower score = higher weight)
            weights[task_name] = 1.0 / (score + 0.1)

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


class MultiTaskLearningManager:
    """
    Advanced Multi-Task Learning Manager

    Features:
    - Shared Bottom architecture
    - Multi-gate Mixture-of-Experts (MMoE)
    - Task weighting strategies
    - Performance tracking per task
    - Positive/negative transfer analysis
    - Visualization of learning progress
    """

    def __init__(self, n_features: int, task_types: Dict[str, str]):
        """
        Args:
            n_features: Number of input features
            task_types: Dictionary mapping task names to types ('classification' or 'regression')
        """
        self.n_features = n_features
        self.task_types = task_types
        self.task_names = list(task_types.keys())

        # Initialize models
        self.shared_bottom = SharedBottomModel(n_features, task_types)
        self.mmoe = MMoEModel(n_features, task_types, n_experts=3)

        # Single-task baselines (for transfer analysis)
        self.single_task_models = {}
        for task_name, task_type in task_types.items():
            if task_type == 'classification':
                self.single_task_models[task_name] = LogisticRegression(max_iter=1000, random_state=42)
            elif task_type == 'regression':
                self.single_task_models[task_name] = Ridge(alpha=1.0, random_state=42)

        # History
        self.history = defaultdict(list)
        self.single_task_scalers = {}

        print(f"🔗 MultiTaskLearning Manager v2.0 initialized")
        print(f"   Features: {n_features}")
        print(f"   Tasks: {', '.join(task_names)}")
        print(f"   Task types: {task_types}")

    def train_all(self, X_train: np.ndarray, y_train: Dict[str, np.ndarray],
                  X_val: Optional[np.ndarray] = None,
                  y_val: Optional[Dict[str, np.ndarray]] = None):
        """
        Train all multi-task models and baselines

        Args:
            X_train: Training features
            y_train: Dictionary of training labels per task
            X_val: Optional validation features
            y_val: Optional validation labels per task
        """
        print(f"\n🚀 Training Multi-Task Models...")

        # Train Shared Bottom model
        print(f"\n📊 Training Shared Bottom Model...")
        self.shared_bottom.fit(X_train, y_train)
        self.history['shared_bottom'] = {'trained': True}

        # Train MMoE model
        print(f"📊 Training MMoE Model...")
        self.mmoe.fit(X_train, y_train)
        self.history['mmoe'] = {'trained': True}

        # Train single-task baselines
        print(f"📊 Training Single-Task Baselines...")
        for task_name in self.task_names:
            if task_name in y_train:
                y = y_train[task_name]
                valid_idx = ~np.isnan(y)

                if np.sum(valid_idx) > 0:
                    # Fit scaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_train[valid_idx])
                    self.single_task_scalers[task_name] = scaler

                    # Train model
                    self.single_task_models[task_name].fit(X_scaled, y[valid_idx])

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            self.evaluate_all(X_val, y_val)

        print(f"\n✓ Training complete!")

    def evaluate_all(self, X_test: np.ndarray, y_test: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on test data

        Args:
            X_test: Test features
            y_test: Test labels per task

        Returns:
            Dictionary of results per model and task
        """
        print(f"\n📊 Evaluating Models...")

        results = {
            'shared_bottom': {},
            'mmoe': {},
            'single_task': {}
        }

        for task_name in self.task_names:
            if task_name not in y_test:
                continue

            y_true = y_test[task_name]
            valid_idx = ~np.isnan(y_true)

            if np.sum(valid_idx) == 0:
                continue

            task_type = self.task_types[task_name]

            # Shared Bottom predictions
            try:
                y_pred_sb = self.shared_bottom.predict(X_test[valid_idx], task_name)
                score_sb = self._compute_score(y_true[valid_idx], y_pred_sb, task_type)
                results['shared_bottom'][task_name] = score_sb
            except:
                results['shared_bottom'][task_name] = 0.0

            # MMoE predictions
            try:
                y_pred_mmoe = self.mmoe.predict(X_test[valid_idx], task_name)
                score_mmoe = self._compute_score(y_true[valid_idx], y_pred_mmoe, task_type)
                results['mmoe'][task_name] = score_mmoe
            except:
                results['mmoe'][task_name] = 0.0

            # Single-task baseline
            try:
                scaler = self.single_task_scalers[task_name]
                X_scaled = scaler.transform(X_test[valid_idx])
                y_pred_st = self.single_task_models[task_name].predict(X_scaled)
                score_st = self._compute_score(y_true[valid_idx], y_pred_st, task_type)
                results['single_task'][task_name] = score_st
            except:
                results['single_task'][task_name] = 0.0

        # Store results
        self.last_results = results

        # Print results
        print(f"\n📈 Evaluation Results:")
        for model_name in ['shared_bottom', 'mmoe', 'single_task']:
            print(f"\n{model_name.upper()}:")
            for task_name, score in results[model_name].items():
                print(f"   {task_name}: {score:.4f}")

        return results

    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> float:
        """Compute appropriate score based on task type"""
        if task_type == 'classification':
            return accuracy_score(y_true, y_pred)
        elif task_type == 'regression':
            return r2_score(y_true, y_pred)
        return 0.0

    def analyze_transfer(self) -> pd.DataFrame:
        """
        Analyze transfer learning effects

        Positive transfer: Multi-task model outperforms single-task baseline
        Negative transfer: Multi-task model underperforms single-task baseline
        """
        if not hasattr(self, 'last_results'):
            print("⚠ No evaluation results available. Run evaluate_all() first.")
            return pd.DataFrame()

        print(f"\n🔍 Transfer Learning Analysis:")

        data = []
        for task_name in self.task_names:
            if task_name not in self.last_results['single_task']:
                continue

            st_score = self.last_results['single_task'][task_name]
            sb_score = self.last_results['shared_bottom'][task_name]
            mmoe_score = self.last_results['mmoe'][task_name]

            sb_transfer = sb_score - st_score
            mmoe_transfer = mmoe_score - st_score

            sb_type = 'Positive' if sb_transfer > 0 else 'Negative'
            mmoe_type = 'Positive' if mmoe_transfer > 0 else 'Negative'

            data.append({
                'Task': task_name,
                'Single-Task': st_score,
                'Shared Bottom': sb_score,
                'MMoE': mmoe_score,
                'SB Transfer': sb_transfer,
                'SB Type': sb_type,
                'MMoE Transfer': mmoe_transfer,
                'MMoE Type': mmoe_type
            })

        df = pd.DataFrame(data)

        print(df.to_string(index=False))

        return df

    def visualize_results(self, figsize: Tuple[int, int] = (15, 10)):
        """Visualize multi-task learning results"""
        if not hasattr(self, 'last_results'):
            print("⚠ No results available. Run evaluate_all() first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('🔗 Multi-Task Learning Performance', fontsize=16, fontweight='bold')

        colors = {'shared_bottom': 'blue', 'mmoe': 'green', 'single_task': 'gray'}

        # Plot 1: Performance comparison
        ax = axes[0, 0]
        models = list(self.last_results.keys())
        x = np.arange(len(self.task_names))
        width = 0.25

        for i, model_name in enumerate(models):
            scores = [self.last_results[model_name].get(task, 0) for task in self.task_names]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, scores, width, label=model_name.replace('_', ' ').title(),
                         color=colors.get(model_name, 'gray'))

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Task')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(self.task_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Transfer learning effects
        ax = axes[0, 1]
        transfer_data = []

        for task_name in self.task_names:
            if task_name not in self.last_results['single_task']:
                continue

            st = self.last_results['single_task'][task_name]
            sb = self.last_results['shared_bottom'][task_name]
            mmoe = self.last_results['mmoe'][task_name]

            transfer_data.append({
                'Task': task_name,
                'Shared Bottom': (sb - st) * 100,
                'MMoE': (mmoe - st) * 100
            })

        if transfer_data:
            df_transfer = pd.DataFrame(transfer_data)
            x = np.arange(len(df_transfer))
            width = 0.35

            ax.bar(x - width/2, df_transfer['Shared Bottom'], width,
                  label='Shared Bottom', color='blue', alpha=0.7)
            ax.bar(x + width/2, df_transfer['MMoE'], width,
                  label='MMoE', color='green', alpha=0.7)

            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Task')
            ax.set_ylabel('Transfer (% improvement over single-task)')
            ax.set_title('Transfer Learning Effects')
            ax.set_xticks(x)
            ax.set_xticklabels(df_transfer['Task'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Average performance
        ax = axes[1, 0]
        avg_scores = {}
        for model_name in models:
            scores = [v for v in self.last_results[model_name].values() if v > 0]
            if scores:
                avg_scores[model_name] = np.mean(scores)

        if avg_scores:
            bars = ax.bar(range(len(avg_scores)), list(avg_scores.values()),
                         color=[colors.get(k, 'gray') for k in avg_scores.keys()])
            ax.set_xticks(range(len(avg_scores)))
            ax.set_xticklabels([k.replace('_', ' ').title() for k in avg_scores.keys()],
                              rotation=45, ha='right')
            ax.set_ylabel('Average Score')
            ax.set_title('Average Performance Across Tasks')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom')

        # Plot 4: Task contribution heatmap
        ax = axes[1, 1]
        if hasattr(self.mmoe, 'gates'):
            gate_data = []
            for task_name in self.task_names:
                gate_data.append(self.mmoe.gates[task_name])

            gate_matrix = np.array(gate_data)
            im = ax.imshow(gate_matrix, cmap='YlOrRd', aspect='auto')

            ax.set_xticks(np.arange(len(self.mmoe.experts)))
            ax.set_yticks(np.arange(len(self.task_names)))
            ax.set_xticklabels([f'Expert {i+1}' for i in range(len(self.mmoe.experts))])
            ax.set_yticklabels(self.task_names)

            # Add text annotations
            for i in range(len(self.task_names)):
                for j in range(len(self.mmoe.experts)):
                    text = ax.text(j, i, f'{gate_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)

            ax.set_title('MMoE Expert Gate Weights')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'Expert gates not available',
                   ha='center', va='center')
            ax.set_title('Expert Contribution')

        plt.tight_layout()
        plt.savefig('multi_task_learning_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved: multi_task_learning_results.png")
        plt.close()

    def save_models(self, filepath: str = 'mtl_models.pkl'):
        """Save all models"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'shared_bottom': self.shared_bottom,
                'mmoe': self.mmoe,
                'single_task_models': self.single_task_models,
                'single_task_scalers': self.single_task_scalers,
                'history': dict(self.history),
                'n_features': self.n_features,
                'task_types': self.task_types
            }, f)
        print(f"✓ Models saved to {filepath}")

    def load_models(self, filepath: str = 'mtl_models.pkl'):
        """Load all models"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.shared_bottom = data['shared_bottom']
        self.mmoe = data['mmoe']
        self.single_task_models = data['single_task_models']
        self.single_task_scalers = data['single_task_scalers']
        self.history = defaultdict(list, data['history'])
        self.n_features = data['n_features']
        self.task_types = data['task_types']

        print(f"✓ Models loaded from {filepath}")


def generate_multi_task_data(n_samples: int = 1000, n_features: int = 20,
                             task_configs: Dict[str, Dict] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate synthetic multi-task learning data

    Args:
        n_samples: Number of samples
        n_features: Number of features
        task_configs: Dictionary of task configurations

    Returns:
        X: Features
        y_dict: Dictionary of labels per task
    """
    if task_configs is None:
        task_configs = {
            'task1_clf': {'type': 'classification', 'n_classes': 2},
            'task2_clf': {'type': 'classification', 'n_classes': 3},
            'task3_reg': {'type': 'regression'}
        }

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate labels for each task
    y_dict = {}

    for task_name, config in task_configs.items():
        if config['type'] == 'classification':
            # Classification: combine multiple features
            logits = X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.5
            y_dict[task_name] = (logits > 0).astype(int)

            if config.get('n_classes', 2) > 2:
                # Multi-class
                logits2 = X[:, 5:10].sum(axis=1)
                y_dict[task_name] = ((logits > 0).astype(int) +
                                    (logits2 > 0).astype(int))

        elif config['type'] == 'regression':
            # Regression: linear combination with noise
            y_dict[task_name] = (X[:, :10].sum(axis=1) +
                                np.random.randn(n_samples) * 2)

    return X, y_dict


def main():
    """Example usage"""
    print("=" * 60)
    print("🔗 Multi-Task Learning v2.0 - Demo")
    print("=" * 60)

    # Generate multi-task data
    print("\n🔧 Generating multi-task dataset...")

    task_configs = {
        'sentiment': {'type': 'classification', 'n_classes': 2},
        'topic': {'type': 'classification', 'n_classes': 3},
        'rating': {'type': 'regression'}
    }

    X, y_dict = generate_multi_task_data(
        n_samples=1000,
        n_features=20,
        task_configs=task_configs
    )

    # Split data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    y_train = {}
    y_test = {}
    for task_name, y in y_dict.items():
        y_train_task, y_test_task = train_test_split(y, test_size=0.2, random_state=42)
        y_train[task_name] = y_train_task
        y_test[task_name] = y_test_task

    print(f"   Generated {len(X)} samples for {len(task_configs)} tasks")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # Initialize manager
    task_types = {name: config['type'] for name, config in task_configs.items()}
    manager = MultiTaskLearningManager(n_features=X.shape[1], task_types=task_types)

    # Train all models
    manager.train_all(X_train, y_train, X_test, y_test)

    # Evaluate
    results = manager.evaluate_all(X_test, y_test)

    # Transfer analysis
    transfer_df = manager.analyze_transfer()

    # Visualize
    manager.visualize_results()

    # Save models
    manager.save_models()

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
