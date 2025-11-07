"""
OnlineLearning v2.0
Author: BrillConsulting
Description: Advanced Online Learning with Incremental Algorithms and Drift Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


class DriftDetector:
    """Base class for concept drift detection"""

    def __init__(self):
        self.drift_detected = False
        self.warning_detected = False

    def update(self, prediction: int, true_label: int) -> bool:
        """Update detector with new prediction"""
        raise NotImplementedError


class DDM(DriftDetector):
    """
    Drift Detection Method (DDM)
    Monitors error rate and standard deviation
    """

    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0):
        super().__init__()
        self.warning_level = warning_level
        self.drift_level = drift_level

        self.error_count = 0
        self.sample_count = 0
        self.p_min = float('inf')
        self.s_min = float('inf')

    def update(self, prediction: int, true_label: int) -> bool:
        """Update DDM with new prediction"""
        self.sample_count += 1
        if prediction != true_label:
            self.error_count += 1

        # Calculate error rate and std
        p = self.error_count / self.sample_count
        s = np.sqrt(p * (1 - p) / self.sample_count)

        # Update minimums
        if p + s < self.p_min + self.s_min:
            self.p_min = p
            self.s_min = s

        # Detect drift
        self.drift_detected = False
        self.warning_detected = False

        if p + s > self.p_min + self.drift_level * self.s_min:
            self.drift_detected = True
            # Reset
            self.error_count = 0
            self.sample_count = 0
            self.p_min = float('inf')
            self.s_min = float('inf')
        elif p + s > self.p_min + self.warning_level * self.s_min:
            self.warning_detected = True

        return self.drift_detected

    def reset(self):
        """Reset detector"""
        self.error_count = 0
        self.sample_count = 0
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.drift_detected = False
        self.warning_detected = False


class EDDM(DriftDetector):
    """
    Early Drift Detection Method (EDDM)
    Monitors distances between errors
    """

    def __init__(self, warning_level: float = 0.95, drift_level: float = 0.90):
        super().__init__()
        self.warning_level = warning_level
        self.drift_level = drift_level

        self.last_error_position = 0
        self.current_position = 0
        self.distances = []
        self.mean_max = 0
        self.std_max = 0

    def update(self, prediction: int, true_label: int) -> bool:
        """Update EDDM with new prediction"""
        self.current_position += 1

        if prediction != true_label:
            # Calculate distance from last error
            if self.last_error_position != 0:
                distance = self.current_position - self.last_error_position
                self.distances.append(distance)

                if len(self.distances) > 30:  # Minimum window
                    mean = np.mean(self.distances)
                    std = np.std(self.distances)

                    # Update max
                    if mean + 2 * std > self.mean_max + 2 * self.std_max:
                        self.mean_max = mean
                        self.std_max = std

                    # Detect drift
                    if len(self.distances) > 30 and self.mean_max > 0:
                        ratio = (mean + 2 * std) / (self.mean_max + 2 * self.std_max)

                        self.drift_detected = False
                        self.warning_detected = False

                        if ratio < self.drift_level:
                            self.drift_detected = True
                            self.reset()
                        elif ratio < self.warning_level:
                            self.warning_detected = True

                        return self.drift_detected

            self.last_error_position = self.current_position

        return False

    def reset(self):
        """Reset detector"""
        self.last_error_position = 0
        self.current_position = 0
        self.distances = []
        self.mean_max = 0
        self.std_max = 0
        self.drift_detected = False
        self.warning_detected = False


class ADWIN(DriftDetector):
    """
    Adaptive Windowing (ADWIN)
    Automatically adjusts window size based on change detection
    """

    def __init__(self, delta: float = 0.002):
        super().__init__()
        self.delta = delta
        self.window = deque()
        self.total = 0
        self.variance = 0
        self.width = 0

    def update(self, prediction: int, true_label: int) -> bool:
        """Update ADWIN with new prediction"""
        # Add new element (1 if correct, 0 if error)
        is_correct = int(prediction == true_label)
        self.window.append(is_correct)
        self.width += 1

        # Update statistics
        if self.width > 1:
            # Simplified variance calculation
            mean = sum(self.window) / len(self.window)
            self.variance = sum((x - mean) ** 2 for x in self.window) / len(self.window)

        # Check for drift (simplified version)
        self.drift_detected = False

        if len(self.window) > 30:
            # Split window and compare
            mid = len(self.window) // 2
            mean1 = sum(list(self.window)[:mid]) / mid
            mean2 = sum(list(self.window)[mid:]) / (len(self.window) - mid)

            # Simplified drift detection
            epsilon = np.sqrt(2 * np.log(2 / self.delta) / mid)

            if abs(mean1 - mean2) > epsilon:
                self.drift_detected = True
                # Remove old half
                for _ in range(mid):
                    self.window.popleft()
                self.width = len(self.window)

        # Limit window size
        if len(self.window) > 1000:
            self.window.popleft()
            self.width = len(self.window)

        return self.drift_detected

    def reset(self):
        """Reset detector"""
        self.window.clear()
        self.total = 0
        self.variance = 0
        self.width = 0
        self.drift_detected = False
        self.warning_detected = False


class StreamingMetrics:
    """Track metrics over a data stream"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.true_labels = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

    def update(self, prediction: int, true_label: int):
        """Add new prediction"""
        self.predictions.append(prediction)
        self.true_labels.append(true_label)
        self.timestamps.append(datetime.now())

    def get_accuracy(self) -> float:
        """Get current accuracy"""
        if len(self.predictions) == 0:
            return 0.0
        return accuracy_score(list(self.true_labels), list(self.predictions))

    def get_error_rate(self) -> float:
        """Get current error rate"""
        return 1.0 - self.get_accuracy()

    def get_f1(self) -> float:
        """Get F1 score"""
        if len(self.predictions) < 2:
            return 0.0
        try:
            return f1_score(list(self.true_labels), list(self.predictions), average='weighted', zero_division=0)
        except:
            return 0.0


class OnlineLearningManager:
    """
    Advanced Online Learning Manager

    Features:
    - Incremental learning with SGD and Passive-Aggressive classifiers
    - Concept drift detection (DDM, EDDM, ADWIN)
    - Streaming performance metrics
    - Prequential (test-then-train) evaluation
    - Model adaptation and retraining
    - Visualization of learning progress
    """

    def __init__(self, n_features: int = 20, n_classes: int = 2,
                 drift_detection: str = 'adwin'):
        self.n_features = n_features
        self.n_classes = n_classes

        # Initialize models
        self.sgd_model = SGDClassifier(max_iter=1, warm_start=True, random_state=42)
        self.pa_model = PassiveAggressiveClassifier(max_iter=1, warm_start=True, random_state=42)

        # Scaler for normalization
        self.scaler = StandardScaler()
        self.scaler_initialized = False

        # Drift detectors
        self.drift_detection_type = drift_detection
        self.drift_detectors = {
            'sgd': self._create_drift_detector(drift_detection),
            'pa': self._create_drift_detector(drift_detection)
        }

        # Metrics trackers
        self.metrics = {
            'sgd': StreamingMetrics(window_size=100),
            'pa': StreamingMetrics(window_size=100)
        }

        # History
        self.history = {
            'sgd': {'accuracy': [], 'errors': [], 'drift_points': []},
            'pa': {'accuracy': [], 'errors': [], 'drift_points': []}
        }

        self.sample_count = 0

        print(f"📊 OnlineLearning Manager v2.0 initialized")
        print(f"   Features: {n_features}, Classes: {n_classes}")
        print(f"   Drift detection: {drift_detection.upper()}")

    def _create_drift_detector(self, detector_type: str) -> DriftDetector:
        """Create drift detector"""
        if detector_type == 'ddm':
            return DDM()
        elif detector_type == 'eddm':
            return EDDM()
        elif detector_type == 'adwin':
            return ADWIN()
        else:
            return ADWIN()

    def partial_fit(self, X: np.ndarray, y: np.ndarray, model_name: str = 'both'):
        """
        Incremental training on new data

        Args:
            X: Feature matrix
            y: Labels
            model_name: 'sgd', 'pa', or 'both'
        """
        # Initialize scaler if needed
        if not self.scaler_initialized:
            self.scaler.fit(X)
            self.scaler_initialized = True

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Train models
        if model_name in ['sgd', 'both']:
            self.sgd_model.partial_fit(X_scaled, y, classes=np.arange(self.n_classes))

        if model_name in ['pa', 'both']:
            self.pa_model.partial_fit(X_scaled, y, classes=np.arange(self.n_classes))

        self.sample_count += len(X)

    def prequential_evaluation(self, X_stream: np.ndarray, y_stream: np.ndarray,
                               initial_train_size: int = 100) -> Dict[str, Any]:
        """
        Prequential (test-then-train) evaluation on streaming data

        Args:
            X_stream: Stream of features
            y_stream: Stream of labels
            initial_train_size: Number of samples for initial training
        """
        print(f"\n🔄 Prequential Evaluation")
        print(f"   Stream size: {len(X_stream)}, Initial training: {initial_train_size}")

        # Initial training
        print(f"   Training on first {initial_train_size} samples...")
        self.partial_fit(X_stream[:initial_train_size], y_stream[:initial_train_size])

        # Stream processing
        total_correct = {'sgd': 0, 'pa': 0}
        total_processed = 0

        for i in range(initial_train_size, len(X_stream)):
            X_sample = X_stream[i:i+1]
            y_true = y_stream[i]

            # Scale
            X_scaled = self.scaler.transform(X_sample)

            # Test (predict before training)
            for model_name, model in [('sgd', self.sgd_model), ('pa', self.pa_model)]:
                try:
                    y_pred = model.predict(X_scaled)[0]

                    # Update metrics
                    self.metrics[model_name].update(y_pred, y_true)
                    is_correct = (y_pred == y_true)
                    if is_correct:
                        total_correct[model_name] += 1

                    # Check for drift
                    drift_detected = self.drift_detectors[model_name].update(y_pred, y_true)

                    if drift_detected:
                        print(f"   ⚠ Drift detected at sample {i} ({model_name.upper()})")
                        self.history[model_name]['drift_points'].append(i)

                        # Reset model (optional)
                        if model_name == 'sgd':
                            self.sgd_model = SGDClassifier(max_iter=1, warm_start=True, random_state=42)
                        else:
                            self.pa_model = PassiveAggressiveClassifier(max_iter=1, warm_start=True, random_state=42)

                        # Retrain on recent window
                        window_size = 50
                        start_idx = max(initial_train_size, i - window_size)
                        self.partial_fit(X_stream[start_idx:i], y_stream[start_idx:i], model_name)

                except Exception as e:
                    pass

            # Train (update model with true label)
            self.partial_fit(X_sample, np.array([y_true]))

            # Track accuracy
            total_processed += 1
            if (i - initial_train_size + 1) % 100 == 0:
                for model_name in ['sgd', 'pa']:
                    acc = self.metrics[model_name].get_accuracy()
                    self.history[model_name]['accuracy'].append(acc)
                    self.history[model_name]['errors'].append(1 - acc)

                sgd_acc = total_correct['sgd'] / total_processed
                pa_acc = total_correct['pa'] / total_processed
                print(f"   Sample {i}: SGD acc={sgd_acc:.4f}, PA acc={pa_acc:.4f}")

        # Final results
        results = {}
        for model_name in ['sgd', 'pa']:
            final_acc = total_correct[model_name] / total_processed
            results[model_name] = {
                'accuracy': final_acc,
                'error_rate': 1 - final_acc,
                'drift_count': len(self.history[model_name]['drift_points']),
                'samples_processed': total_processed
            }

        print(f"\n✓ Prequential evaluation complete!")
        return results

    def batch_train_and_stream(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_stream: np.ndarray, y_stream: np.ndarray) -> Dict[str, Any]:
        """
        Train on batch data, then evaluate on stream

        Args:
            X_train: Initial training features
            y_train: Initial training labels
            X_stream: Streaming features
            y_stream: Streaming labels
        """
        print(f"\n🚀 Batch Training + Stream Evaluation")
        print(f"   Batch size: {len(X_train)}, Stream size: {len(X_stream)}")

        # Batch training
        print(f"   Batch training...")
        self.partial_fit(X_train, y_train)

        # Stream evaluation
        print(f"   Stream evaluation...")
        results = self.prequential_evaluation(X_stream, y_stream, initial_train_size=0)

        return results

    def visualize_stream_performance(self, figsize: Tuple[int, int] = (15, 10)):
        """Visualize learning performance over stream"""
        if not self.history['sgd']['accuracy']:
            print("⚠ No history available. Run prequential evaluation first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('📊 Online Learning Performance', fontsize=16, fontweight='bold')

        colors = {'sgd': 'blue', 'pa': 'green'}

        # Plot 1: Accuracy over time
        ax = axes[0, 0]
        for model_name in ['sgd', 'pa']:
            if self.history[model_name]['accuracy']:
                x = np.arange(len(self.history[model_name]['accuracy'])) * 100
                ax.plot(x, self.history[model_name]['accuracy'],
                       label=model_name.upper(), color=colors[model_name], linewidth=2)

                # Mark drift points
                for drift_point in self.history[model_name]['drift_points']:
                    ax.axvline(drift_point, color=colors[model_name], alpha=0.3, linestyle='--')

        ax.set_xlabel('Samples')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Over Stream')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Error rate
        ax = axes[0, 1]
        for model_name in ['sgd', 'pa']:
            if self.history[model_name]['errors']:
                x = np.arange(len(self.history[model_name]['errors'])) * 100
                ax.plot(x, self.history[model_name]['errors'],
                       label=model_name.upper(), color=colors[model_name], linewidth=2)

        ax.set_xlabel('Samples')
        ax.set_ylabel('Error Rate')
        ax.set_title('Error Rate Over Stream')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Drift detection timeline
        ax = axes[1, 0]
        drift_data = []
        for model_name in ['sgd', 'pa']:
            for drift_point in self.history[model_name]['drift_points']:
                drift_data.append({'Model': model_name.upper(), 'Sample': drift_point})

        if drift_data:
            df_drift = pd.DataFrame(drift_data)
            for model_name in ['sgd', 'pa']:
                model_drifts = df_drift[df_drift['Model'] == model_name.upper()]
                if not model_drifts.empty:
                    ax.scatter(model_drifts['Sample'], [model_name.upper()] * len(model_drifts),
                             s=100, c=colors[model_name], marker='|', linewidths=3,
                             label=f"{model_name.upper()} ({len(model_drifts)} drifts)")

            ax.set_xlabel('Sample Number')
            ax.set_title(f'Concept Drift Detection ({self.drift_detection_type.upper()})')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No drifts detected', ha='center', va='center')
            ax.set_title('Concept Drift Detection')

        ax.grid(True, alpha=0.3)

        # Plot 4: Cumulative accuracy comparison
        ax = axes[1, 1]
        for model_name in ['sgd', 'pa']:
            if self.history[model_name]['accuracy']:
                cumulative = pd.Series(self.history[model_name]['accuracy']).expanding().mean()
                x = np.arange(len(cumulative)) * 100
                ax.plot(x, cumulative, label=model_name.upper(),
                       color=colors[model_name], linewidth=2)

        ax.set_xlabel('Samples')
        ax.set_ylabel('Cumulative Mean Accuracy')
        ax.set_title('Learning Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('online_learning_performance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Performance visualization saved: online_learning_performance.png")
        plt.close()

    def save_models(self, filepath: str = 'online_models.pkl'):
        """Save trained models"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'sgd_model': self.sgd_model,
                'pa_model': self.pa_model,
                'scaler': self.scaler,
                'history': self.history,
                'n_features': self.n_features,
                'n_classes': self.n_classes
            }, f)
        print(f"✓ Models saved to {filepath}")

    def load_models(self, filepath: str = 'online_models.pkl'):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.sgd_model = data['sgd_model']
        self.pa_model = data['pa_model']
        self.scaler = data['scaler']
        self.history = data['history']
        self.n_features = data['n_features']
        self.n_classes = data['n_classes']
        self.scaler_initialized = True

        print(f"✓ Models loaded from {filepath}")

    def get_summary(self) -> pd.DataFrame:
        """Get summary of performance"""
        if not self.history['sgd']['accuracy']:
            print("⚠ No history available")
            return pd.DataFrame()

        data = []
        for model_name in ['sgd', 'pa']:
            if self.history[model_name]['accuracy']:
                data.append({
                    'Model': model_name.upper(),
                    'Final Accuracy': self.history[model_name]['accuracy'][-1],
                    'Mean Accuracy': np.mean(self.history[model_name]['accuracy']),
                    'Drift Count': len(self.history[model_name]['drift_points']),
                    'Samples': self.sample_count
                })

        return pd.DataFrame(data)


def generate_stream_with_drift(n_samples: int = 1000, n_features: int = 20,
                               drift_positions: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data stream with concept drift

    Args:
        n_samples: Number of samples
        n_features: Number of features
        drift_positions: List of sample indices where drift occurs
    """
    if drift_positions is None:
        drift_positions = [n_samples // 3, 2 * n_samples // 3]

    X = []
    y = []

    current_drift = 0
    for i in range(n_samples):
        # Check if we should introduce drift
        if current_drift < len(drift_positions) and i >= drift_positions[current_drift]:
            current_drift += 1

        # Generate data with different distributions based on drift
        if current_drift % 2 == 0:
            # Original distribution
            x = np.random.randn(n_features)
            label = int(x[0] + x[1] > 0)
        else:
            # Drifted distribution
            x = np.random.randn(n_features) + 1
            label = int(x[2] + x[3] > 1)

        X.append(x)
        y.append(label)

    return np.array(X), np.array(y)


def main():
    """Example usage"""
    print("=" * 60)
    print("📊 Online Learning v2.0 - Demo")
    print("=" * 60)

    # Generate streaming data with concept drift
    print("\n🔧 Generating synthetic data stream...")
    n_samples = 2000
    n_features = 20
    drift_positions = [600, 1200, 1600]

    X_stream, y_stream = generate_stream_with_drift(
        n_samples=n_samples,
        n_features=n_features,
        drift_positions=drift_positions
    )

    print(f"   Generated {n_samples} samples with drifts at positions: {drift_positions}")

    # Test different drift detectors
    for detector_type in ['ddm', 'eddm', 'adwin']:
        print(f"\n{'='*60}")
        print(f"Testing {detector_type.upper()} drift detector")
        print(f"{'='*60}")

        manager = OnlineLearningManager(
            n_features=n_features,
            n_classes=2,
            drift_detection=detector_type
        )

        # Prequential evaluation
        results = manager.prequential_evaluation(X_stream, y_stream, initial_train_size=100)

        # Print results
        print(f"\n📊 Results with {detector_type.upper()}:")
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Error rate: {metrics['error_rate']:.4f}")
            print(f"   Drift detections: {metrics['drift_count']}")

        # Visualize
        manager.visualize_stream_performance()

        # Summary
        print(f"\n📋 Summary:")
        print(manager.get_summary().to_string(index=False))

        # Save model
        manager.save_models(f'online_models_{detector_type}.pkl')

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
