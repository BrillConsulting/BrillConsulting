"""
MetaLearning v2.0
Author: BrillConsulting
Description: Advanced Meta-Learning with MAML, Prototypical Networks, and Few-Shot Learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')


class Task:
    """Represents a single few-shot learning task"""

    def __init__(self, support_X: np.ndarray, support_y: np.ndarray,
                 query_X: np.ndarray, query_y: np.ndarray):
        self.support_X = support_X
        self.support_y = support_y
        self.query_X = query_X
        self.query_y = query_y

        self.n_way = len(np.unique(support_y))
        self.k_shot = len(support_y) // self.n_way

    def __repr__(self):
        return f"Task({self.n_way}-way, {self.k_shot}-shot, {len(self.query_y)} queries)"


class TaskGenerator:
    """Generate few-shot learning tasks from data"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.class_to_indices = {c: np.where(y == c)[0] for c in self.classes}

    def sample_task(self, n_way: int = 5, k_shot: int = 5,
                   query_per_class: int = 10) -> Task:
        """
        Sample a single N-way K-shot task

        Args:
            n_way: Number of classes in task
            k_shot: Number of examples per class in support set
            query_per_class: Number of query examples per class
        """
        # Randomly select N classes
        selected_classes = np.random.choice(self.classes, size=n_way, replace=False)

        support_X = []
        support_y = []
        query_X = []
        query_y = []

        # For each class, sample K support and Q query examples
        for new_label, original_class in enumerate(selected_classes):
            indices = self.class_to_indices[original_class]

            # Sample K+Q examples
            sampled = np.random.choice(indices, size=k_shot + query_per_class, replace=False)

            support_indices = sampled[:k_shot]
            query_indices = sampled[k_shot:]

            support_X.append(self.X[support_indices])
            support_y.extend([new_label] * k_shot)

            query_X.append(self.X[query_indices])
            query_y.extend([new_label] * query_per_class)

        support_X = np.vstack(support_X)
        support_y = np.array(support_y)
        query_X = np.vstack(query_X)
        query_y = np.array(query_y)

        return Task(support_X, support_y, query_X, query_y)

    def sample_batch_tasks(self, batch_size: int = 8, **task_params) -> List[Task]:
        """Sample a batch of tasks"""
        return [self.sample_task(**task_params) for _ in range(batch_size)]


class PrototypicalNetwork:
    """
    Prototypical Networks for Few-Shot Learning
    Learns to compute class prototypes and classify based on distance
    """

    def __init__(self, distance_metric: str = 'euclidean'):
        self.distance_metric = distance_metric
        self.scaler = StandardScaler()
        self.prototypes = {}

    def compute_prototypes(self, support_X: np.ndarray, support_y: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute class prototypes (mean of support examples)"""
        prototypes = {}
        for class_label in np.unique(support_y):
            class_examples = support_X[support_y == class_label]
            prototypes[class_label] = np.mean(class_examples, axis=0)
        return prototypes

    def distance(self, x: np.ndarray, prototype: np.ndarray) -> float:
        """Compute distance between example and prototype"""
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(x - prototype)
        elif self.distance_metric == 'cosine':
            return 1 - np.dot(x, prototype) / (np.linalg.norm(x) * np.linalg.norm(prototype) + 1e-8)
        else:
            return np.linalg.norm(x - prototype)

    def predict_proba(self, X: np.ndarray, prototypes: Dict[int, np.ndarray]) -> np.ndarray:
        """Predict class probabilities based on distances to prototypes"""
        n_samples = len(X)
        n_classes = len(prototypes)
        distances = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            for class_idx, (class_label, prototype) in enumerate(prototypes.items()):
                distances[i, class_idx] = self.distance(x, prototype)

        # Convert distances to probabilities using softmax
        # Negate distances (closer = higher score)
        scores = -distances
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray, prototypes: Dict[int, np.ndarray]) -> np.ndarray:
        """Predict classes"""
        probs = self.predict_proba(X, prototypes)
        class_labels = list(prototypes.keys())
        predictions = np.array([class_labels[i] for i in np.argmax(probs, axis=1)])
        return predictions

    def evaluate_task(self, task: Task) -> float:
        """Evaluate on a single task"""
        # Scale features
        self.scaler.fit(task.support_X)
        support_X_scaled = self.scaler.transform(task.support_X)
        query_X_scaled = self.scaler.transform(task.query_X)

        # Compute prototypes
        prototypes = self.compute_prototypes(support_X_scaled, task.support_y)

        # Predict
        predictions = self.predict(query_X_scaled, prototypes)

        # Accuracy
        return accuracy_score(task.query_y, predictions)


class MAML:
    """
    Model-Agnostic Meta-Learning (simplified version)
    Uses logistic regression as base model
    """

    def __init__(self, n_features: int, inner_lr: float = 0.01,
                 inner_steps: int = 5):
        self.n_features = n_features
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        # Meta-parameters (simplified - using sklearn model)
        self.meta_model = LogisticRegression(max_iter=100, random_state=42)
        self.scaler = StandardScaler()

    def inner_loop(self, task: Task, model: LogisticRegression) -> LogisticRegression:
        """
        Inner loop: Adapt model to task using support set

        Args:
            task: Few-shot task
            model: Initial model (with meta-parameters)

        Returns:
            Adapted model
        """
        # Clone model
        adapted_model = LogisticRegression(
            max_iter=self.inner_steps,
            warm_start=True,
            random_state=42
        )

        # Initialize with meta-parameters if available
        if hasattr(model, 'coef_'):
            adapted_model.coef_ = model.coef_.copy()
            adapted_model.intercept_ = model.intercept_.copy()
            adapted_model.classes_ = model.classes_

        # Train on support set
        try:
            adapted_model.fit(task.support_X, task.support_y)
        except:
            # If not enough classes, initialize fresh
            adapted_model = LogisticRegression(max_iter=self.inner_steps, random_state=42)
            adapted_model.fit(task.support_X, task.support_y)

        return adapted_model

    def meta_train(self, tasks: List[Task], outer_lr: float = 0.01) -> Dict[str, float]:
        """
        Meta-training: Update meta-parameters across tasks

        Args:
            tasks: Batch of training tasks
            outer_lr: Outer loop learning rate

        Returns:
            Training metrics
        """
        task_losses = []
        task_accuracies = []

        # Collect gradients from all tasks (simplified)
        all_support_X = []
        all_support_y = []

        for task in tasks:
            # Inner loop: adapt to task
            adapted_model = self.inner_loop(task, self.meta_model)

            # Evaluate on query set
            try:
                predictions = adapted_model.predict(task.query_X)
                acc = accuracy_score(task.query_y, predictions)
                task_accuracies.append(acc)

                # Accumulate support data for meta-update
                all_support_X.append(task.support_X)
                all_support_y.append(task.support_y)
            except:
                pass

        # Outer loop: update meta-parameters
        if all_support_X:
            all_support_X = np.vstack(all_support_X)
            all_support_y = np.hstack(all_support_y)

            # Update meta-model
            self.meta_model.partial_fit(all_support_X, all_support_y,
                                       classes=np.unique(all_support_y))

        return {
            'mean_accuracy': np.mean(task_accuracies) if task_accuracies else 0,
            'task_count': len(task_accuracies)
        }

    def evaluate_task(self, task: Task) -> float:
        """Evaluate on a single task"""
        # Scale features
        self.scaler.fit(task.support_X)
        task.support_X = self.scaler.transform(task.support_X)
        task.query_X = self.scaler.transform(task.query_X)

        # Adapt to task
        adapted_model = self.inner_loop(task, self.meta_model)

        # Evaluate
        try:
            predictions = adapted_model.predict(task.query_X)
            return accuracy_score(task.query_y, predictions)
        except:
            return 0.0


class MatchingNetwork:
    """
    Matching Networks - Uses attention over support set
    Simplified version using KNN with cosine similarity
    """

    def __init__(self, k_neighbors: int = 3):
        self.k_neighbors = k_neighbors
        self.scaler = StandardScaler()

    def evaluate_task(self, task: Task) -> float:
        """Evaluate on a single task"""
        # Scale features
        self.scaler.fit(task.support_X)
        support_X_scaled = self.scaler.transform(task.support_X)
        query_X_scaled = self.scaler.transform(task.query_X)

        # Use KNN with cosine similarity
        k = min(self.k_neighbors, len(task.support_y))
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(support_X_scaled, task.support_y)

        # Predict
        predictions = knn.predict(query_X_scaled)

        # Accuracy
        return accuracy_score(task.query_y, predictions)


class MetaLearningManager:
    """
    Advanced Meta-Learning Manager

    Features:
    - Prototypical Networks
    - MAML (Model-Agnostic Meta-Learning)
    - Matching Networks
    - N-way K-shot task generation
    - Meta-training and evaluation
    - Few-shot learning benchmarks
    """

    def __init__(self, n_features: int = 20):
        self.n_features = n_features

        # Initialize meta-learners
        self.proto_net = PrototypicalNetwork(distance_metric='euclidean')
        self.maml = MAML(n_features=n_features)
        self.matching_net = MatchingNetwork(k_neighbors=3)

        # History
        self.history = defaultdict(list)

        print(f"🧠 MetaLearning Manager v2.0 initialized")
        print(f"   Features: {n_features}")
        print(f"   Algorithms: Prototypical Networks, MAML, Matching Networks")

    def meta_train(self, X_train: np.ndarray, y_train: np.ndarray,
                   n_episodes: int = 100, n_way: int = 5, k_shot: int = 5,
                   query_per_class: int = 10, tasks_per_episode: int = 8):
        """
        Meta-training across multiple tasks

        Args:
            X_train: Training features
            y_train: Training labels
            n_episodes: Number of meta-training episodes
            n_way: Number of classes per task
            k_shot: Number of support examples per class
            query_per_class: Number of query examples per class
            tasks_per_episode: Number of tasks per episode
        """
        print(f"\n🎓 Meta-Training")
        print(f"   Episodes: {n_episodes}")
        print(f"   Task config: {n_way}-way {k_shot}-shot")
        print(f"   Tasks per episode: {tasks_per_episode}")

        # Create task generator
        task_gen = TaskGenerator(X_train, y_train)

        # Meta-training loop
        for episode in range(n_episodes):
            # Sample batch of tasks
            tasks = task_gen.sample_batch_tasks(
                batch_size=tasks_per_episode,
                n_way=n_way,
                k_shot=k_shot,
                query_per_class=query_per_class
            )

            # Train MAML
            maml_metrics = self.maml.meta_train(tasks)
            self.history['maml_train_acc'].append(maml_metrics['mean_accuracy'])

            # Evaluate other methods
            proto_accs = [self.proto_net.evaluate_task(task) for task in tasks]
            matching_accs = [self.matching_net.evaluate_task(task) for task in tasks]

            self.history['proto_train_acc'].append(np.mean(proto_accs))
            self.history['matching_train_acc'].append(np.mean(matching_accs))

            if (episode + 1) % 20 == 0:
                print(f"   Episode {episode + 1}:")
                print(f"      MAML: {maml_metrics['mean_accuracy']:.4f}")
                print(f"      ProtoNet: {np.mean(proto_accs):.4f}")
                print(f"      Matching: {np.mean(matching_accs):.4f}")

        print(f"✓ Meta-training complete!")

    def meta_test(self, X_test: np.ndarray, y_test: np.ndarray,
                  n_episodes: int = 100, n_way: int = 5, k_shot: int = 5,
                  query_per_class: int = 10) -> Dict[str, Any]:
        """
        Meta-testing on unseen tasks

        Args:
            X_test: Test features
            y_test: Test labels
            n_episodes: Number of test episodes
            n_way: Number of classes per task
            k_shot: Number of support examples per class
            query_per_class: Number of query examples per class

        Returns:
            Test results
        """
        print(f"\n📊 Meta-Testing")
        print(f"   Episodes: {n_episodes}")
        print(f"   Task config: {n_way}-way {k_shot}-shot")

        # Create task generator
        task_gen = TaskGenerator(X_test, y_test)

        # Test on multiple episodes
        maml_accs = []
        proto_accs = []
        matching_accs = []

        for episode in range(n_episodes):
            # Sample test task
            task = task_gen.sample_task(
                n_way=n_way,
                k_shot=k_shot,
                query_per_class=query_per_class
            )

            # Evaluate all methods
            maml_acc = self.maml.evaluate_task(task)
            proto_acc = self.proto_net.evaluate_task(task)
            matching_acc = self.matching_net.evaluate_task(task)

            maml_accs.append(maml_acc)
            proto_accs.append(proto_acc)
            matching_accs.append(matching_acc)

            if (episode + 1) % 20 == 0:
                print(f"   Episode {episode + 1}: MAML={np.mean(maml_accs[-20:]):.4f}, "
                     f"Proto={np.mean(proto_accs[-20:]):.4f}, "
                     f"Match={np.mean(matching_accs[-20:]):.4f}")

        results = {
            'maml': {
                'mean_accuracy': np.mean(maml_accs),
                'std_accuracy': np.std(maml_accs),
                'ci_95': 1.96 * np.std(maml_accs) / np.sqrt(len(maml_accs))
            },
            'protonet': {
                'mean_accuracy': np.mean(proto_accs),
                'std_accuracy': np.std(proto_accs),
                'ci_95': 1.96 * np.std(proto_accs) / np.sqrt(len(proto_accs))
            },
            'matching': {
                'mean_accuracy': np.mean(matching_accs),
                'std_accuracy': np.std(matching_accs),
                'ci_95': 1.96 * np.std(matching_accs) / np.sqrt(len(matching_accs))
            }
        }

        print(f"\n✓ Meta-testing complete!")
        return results

    def visualize_results(self, figsize: Tuple[int, int] = (15, 10)):
        """Visualize meta-learning results"""
        if not self.history:
            print("⚠ No history available. Run meta-training first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('🧠 Meta-Learning Performance', fontsize=16, fontweight='bold')

        colors = {'maml': 'blue', 'proto': 'green', 'matching': 'red'}

        # Plot 1: Training accuracy over episodes
        ax = axes[0, 0]
        if 'maml_train_acc' in self.history:
            ax.plot(self.history['maml_train_acc'], label='MAML',
                   color=colors['maml'], linewidth=2)
        if 'proto_train_acc' in self.history:
            ax.plot(self.history['proto_train_acc'], label='ProtoNet',
                   color=colors['proto'], linewidth=2)
        if 'matching_train_acc' in self.history:
            ax.plot(self.history['matching_train_acc'], label='Matching',
                   color=colors['matching'], linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Accuracy')
        ax.set_title('Meta-Training Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Smoothed training curves
        ax = axes[0, 1]
        window = 10
        for key, label, color in [('maml_train_acc', 'MAML', 'maml'),
                                   ('proto_train_acc', 'ProtoNet', 'proto'),
                                   ('matching_train_acc', 'Matching', 'matching')]:
            if key in self.history and len(self.history[key]) > window:
                smoothed = pd.Series(self.history[key]).rolling(window=window, min_periods=1).mean()
                ax.plot(smoothed, label=label, color=colors[color], linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Accuracy (smoothed)')
        ax.set_title('Smoothed Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Algorithm comparison (if test results available)
        ax = axes[1, 0]
        if hasattr(self, 'test_results'):
            algorithms = []
            accuracies = []
            errors = []

            for name, results in self.test_results.items():
                algorithms.append(name.upper())
                accuracies.append(results['mean_accuracy'])
                errors.append(results['ci_95'])

            bars = ax.bar(algorithms, accuracies, yerr=errors, capsize=5,
                         color=[colors.get(name.lower(), 'gray') for name in algorithms])
            ax.set_ylabel('Test Accuracy')
            ax.set_title('Meta-Test Performance Comparison')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Run meta_test() to see results',
                   ha='center', va='center')
            ax.set_title('Test Performance')

        # Plot 4: Convergence speed
        ax = axes[1, 1]
        for key, label, color in [('maml_train_acc', 'MAML', 'maml'),
                                   ('proto_train_acc', 'ProtoNet', 'proto'),
                                   ('matching_train_acc', 'Matching', 'matching')]:
            if key in self.history:
                cumulative = pd.Series(self.history[key]).expanding().mean()
                ax.plot(cumulative, label=label, color=colors[color], linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Mean Accuracy')
        ax.set_title('Learning Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('meta_learning_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved: meta_learning_results.png")
        plt.close()

    def save_models(self, filepath: str = 'meta_models.pkl'):
        """Save meta-learned models"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'proto_net': self.proto_net,
                'maml': self.maml,
                'matching_net': self.matching_net,
                'history': dict(self.history),
                'n_features': self.n_features
            }, f)
        print(f"✓ Models saved to {filepath}")

    def load_models(self, filepath: str = 'meta_models.pkl'):
        """Load meta-learned models"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.proto_net = data['proto_net']
        self.maml = data['maml']
        self.matching_net = data['matching_net']
        self.history = defaultdict(list, data['history'])
        self.n_features = data['n_features']

        print(f"✓ Models loaded from {filepath}")

    def get_summary(self) -> pd.DataFrame:
        """Get summary of meta-learning performance"""
        if not hasattr(self, 'test_results'):
            print("⚠ No test results available. Run meta_test() first.")
            return pd.DataFrame()

        data = []
        for algorithm, results in self.test_results.items():
            data.append({
                'Algorithm': algorithm.upper(),
                'Mean Accuracy': results['mean_accuracy'],
                'Std Accuracy': results['std_accuracy'],
                'CI 95%': results['ci_95']
            })

        return pd.DataFrame(data)


def generate_few_shot_data(n_classes: int = 20, n_samples_per_class: int = 100,
                           n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for few-shot learning"""
    X = []
    y = []

    for class_id in range(n_classes):
        # Each class has different mean
        mean = np.random.randn(n_features) * 2
        class_X = np.random.randn(n_samples_per_class, n_features) + mean
        X.append(class_X)
        y.extend([class_id] * n_samples_per_class)

    return np.vstack(X), np.array(y)


def main():
    """Example usage"""
    print("=" * 60)
    print("🧠 Meta-Learning v2.0 - Demo")
    print("=" * 60)

    # Generate few-shot learning dataset
    print("\n🔧 Generating few-shot dataset...")
    n_train_classes = 15
    n_test_classes = 5
    total_classes = n_train_classes + n_test_classes

    X, y = generate_few_shot_data(n_classes=total_classes, n_samples_per_class=100)

    # Split into meta-train and meta-test classes
    train_classes = np.arange(n_train_classes)
    test_classes = np.arange(n_train_classes, total_classes)

    X_train = X[y < n_train_classes]
    y_train = y[y < n_train_classes]
    X_test = X[y >= n_train_classes]
    y_test = y[y >= n_train_classes] - n_train_classes  # Re-label

    print(f"   Meta-train: {len(X_train)} samples, {n_train_classes} classes")
    print(f"   Meta-test: {len(X_test)} samples, {n_test_classes} classes")

    # Initialize manager
    manager = MetaLearningManager(n_features=X.shape[1])

    # Meta-training
    manager.meta_train(
        X_train, y_train,
        n_episodes=100,
        n_way=5,
        k_shot=5,
        query_per_class=10,
        tasks_per_episode=8
    )

    # Meta-testing
    test_results = manager.meta_test(
        X_test, y_test,
        n_episodes=100,
        n_way=5,
        k_shot=5,
        query_per_class=10
    )

    # Store results
    manager.test_results = test_results

    # Print results
    print(f"\n📊 Meta-Test Results:")
    for algorithm, results in test_results.items():
        print(f"\n{algorithm.upper()}:")
        print(f"   Accuracy: {results['mean_accuracy']:.4f} ± {results['ci_95']:.4f}")
        print(f"   Std: {results['std_accuracy']:.4f}")

    # Visualize
    manager.visualize_results()

    # Summary
    print(f"\n📋 Summary:")
    print(manager.get_summary().to_string(index=False))

    # Save models
    manager.save_models()

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
