"""
Poison Detection
================

Detection and mitigation of data poisoning and backdoor attacks

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np


class AttackType(Enum):
    """Poisoning attack types."""
    LABEL_FLIPPING = "label_flipping"
    FEATURE_POISONING = "feature_poisoning"
    BACKDOOR = "backdoor"
    CLEAN_LABEL = "clean_label"
    FEDERATED_POISON = "federated_poison"


class DetectionMethod(Enum):
    """Detection methods."""
    ANOMALY = "anomaly"
    ACTIVATION_CLUSTERING = "activation_clustering"
    SPECTRAL = "spectral"
    NEURAL_CLEANSE = "neural_cleanse"


@dataclass
class PoisonDetectionResult:
    """Result of poison detection."""
    poisoned_indices: List[int]
    confidence: float
    detection_method: str
    contamination_estimate: float
    timestamp: str


@dataclass
class BackdoorReport:
    """Backdoor detection report."""
    has_backdoor: bool
    confidence: float
    target_class: Optional[int]
    trigger_description: str
    detection_method: str
    mitigation_recommendation: str
    timestamp: str


class PoisonDetector:
    """Detect poisoned samples in training data."""

    def __init__(
        self,
        detection_method: str = "anomaly",
        threshold: float = 0.95
    ):
        """Initialize poison detector."""
        self.detection_method = DetectionMethod(detection_method)
        self.threshold = threshold

        print(f"🔍 Poison Detector initialized")
        print(f"   Method: {detection_method}")
        print(f"   Threshold: {threshold}")

    def scan_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        contamination_rate: float = 0.05
    ) -> PoisonDetectionResult:
        """Scan dataset for poisoned samples."""
        print(f"\n🔎 Scanning dataset")
        print(f"   Samples: {len(X)}")
        print(f"   Expected contamination: {contamination_rate:.1%}")

        if self.detection_method == DetectionMethod.ANOMALY:
            poisoned_indices = self._anomaly_detection(X, y, contamination_rate)
        elif self.detection_method == DetectionMethod.SPECTRAL:
            poisoned_indices = self._spectral_detection(X, y)
        else:
            poisoned_indices = self._anomaly_detection(X, y, contamination_rate)

        confidence = self._calculate_confidence(len(poisoned_indices), len(X))

        result = PoisonDetectionResult(
            poisoned_indices=poisoned_indices,
            confidence=confidence,
            detection_method=self.detection_method.value,
            contamination_estimate=len(poisoned_indices) / len(X),
            timestamp=datetime.now().isoformat()
        )

        print(f"   Poisoned samples detected: {len(poisoned_indices)}")
        print(f"   Contamination rate: {result.contamination_estimate:.2%}")
        print(f"   Confidence: {confidence:.2%}")

        return result

    def _anomaly_detection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        contamination: float
    ) -> List[int]:
        """Anomaly-based poison detection."""
        print(f"   Using Isolation Forest...")

        # Simulate Isolation Forest
        # In production: from sklearn.ensemble import IsolationForest

        # Flatten data for analysis
        X_flat = X.reshape(len(X), -1)

        # Simulate anomaly scores
        anomaly_scores = np.random.rand(len(X))

        # Add some clear anomalies
        num_anomalies = int(len(X) * contamination)
        anomaly_indices = np.random.choice(len(X), num_anomalies, replace=False)
        anomaly_scores[anomaly_indices] = np.random.uniform(0.8, 1.0, num_anomalies)

        # Threshold for detection
        threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)

        poisoned_indices = np.where(anomaly_scores > threshold)[0].tolist()

        return poisoned_indices

    def _spectral_detection(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Spectral signature-based detection."""
        print(f"   Using spectral analysis...")

        # Simulate spectral analysis
        # In production: actual SVD and spectral clustering

        # Compute covariance
        X_flat = X.reshape(len(X), -1)
        cov = np.cov(X_flat.T)

        # Simulate eigenvalue analysis
        # Poisoned samples often affect top eigenvalues

        # Detect outliers (simulated)
        outlier_scores = np.random.rand(len(X))
        threshold = 0.9

        poisoned_indices = np.where(outlier_scores > threshold)[0].tolist()

        return poisoned_indices

    def _calculate_confidence(self, detected: int, total: int) -> float:
        """Calculate detection confidence."""
        # Confidence based on detection ratio and method
        detection_ratio = detected / total if total > 0 else 0

        # Higher confidence for moderate detection rates
        if 0.01 <= detection_ratio <= 0.15:
            return 0.9
        elif detection_ratio < 0.01:
            return 0.95  # Very few detections, high confidence if any
        else:
            return 0.7  # Many detections, lower confidence

class BackdoorDetector:
    """Detect backdoors in trained models."""

    def __init__(self, method: str = "activation_clustering"):
        """Initialize backdoor detector."""
        self.method = DetectionMethod(method)
        print(f"🎯 Backdoor Detector initialized")
        print(f"   Method: {method}")

    def detect(
        self,
        model: Any,
        clean_data: np.ndarray,
        num_classes: int = 10
    ) -> BackdoorReport:
        """Detect backdoors in model."""
        print(f"\n🔍 Detecting backdoors")
        print(f"   Classes: {num_classes}")

        if self.method == DetectionMethod.ACTIVATION_CLUSTERING:
            has_backdoor, target_class = self._activation_clustering(
                model, clean_data, num_classes
            )
        elif self.method == DetectionMethod.NEURAL_CLEANSE:
            has_backdoor, target_class = self._neural_cleanse(
                model, num_classes
            )
        else:
            has_backdoor, target_class = self._activation_clustering(
                model, clean_data, num_classes
            )

        report = BackdoorReport(
            has_backdoor=has_backdoor,
            confidence=0.85 if has_backdoor else 0.95,
            target_class=target_class if has_backdoor else None,
            trigger_description="Small patch in bottom-right corner" if has_backdoor else "None",
            detection_method=self.method.value,
            mitigation_recommendation="Fine-pruning or trigger inversion" if has_backdoor else "None",
            timestamp=datetime.now().isoformat()
        )

        if report.has_backdoor:
            print(f"   ⚠️  BACKDOOR DETECTED")
            print(f"   Target class: {target_class}")
            print(f"   Confidence: {report.confidence:.2%}")
        else:
            print(f"   ✓ No backdoor detected")

        return report

    def _activation_clustering(
        self,
        model: Any,
        clean_data: np.ndarray,
        num_classes: int
    ) -> Tuple[bool, Optional[int]]:
        """Activation clustering-based detection."""
        print(f"   Analyzing activation patterns...")

        # Simulate activation analysis
        # In production: extract activations from penultimate layer

        # For each class, cluster activations
        class_clusters = {}

        for class_idx in range(num_classes):
            # Simulate activation vectors
            activations = np.random.randn(50, 128)

            # Cluster (simulated)
            # In production: use K-means or DBSCAN

            # Compute silhouette score
            silhouette = np.random.uniform(0.4, 0.8)

            class_clusters[class_idx] = silhouette

        # Backdoor class typically has lower silhouette (outlier cluster)
        min_silhouette_class = min(class_clusters.items(), key=lambda x: x[1])

        # Threshold for backdoor detection
        if min_silhouette_class[1] < 0.5:
            has_backdoor = True
            target_class = min_silhouette_class[0]
        else:
            has_backdoor = False
            target_class = None

        return has_backdoor, target_class

    def _neural_cleanse(
        self,
        model: Any,
        num_classes: int
    ) -> Tuple[bool, Optional[int]]:
        """Neural Cleanse trigger synthesis."""
        print(f"   Synthesizing potential triggers...")

        # For each class, synthesize minimal trigger
        trigger_sizes = {}

        for class_idx in range(num_classes):
            # Optimize trigger to cause misclassification to class_idx
            # Minimize: ||trigger|| subject to: model(x + trigger) = class_idx

            # Simulate trigger size
            trigger_size = np.random.uniform(0.1, 0.5)
            trigger_sizes[class_idx] = trigger_size

        # Backdoor class has anomalously small trigger
        min_trigger_class = min(trigger_sizes.items(), key=lambda x: x[1])
        trigger_values = list(trigger_sizes.values())
        trigger_mean = np.mean(trigger_values)
        trigger_std = np.std(trigger_values)

        # Anomaly detection on trigger sizes
        if min_trigger_class[1] < trigger_mean - 2 * trigger_std:
            has_backdoor = True
            target_class = min_trigger_class[0]
        else:
            has_backdoor = False
            target_class = None

        return has_backdoor, target_class


class TriggerInverter:
    """Invert triggers from backdoored models."""

    def __init__(self):
        """Initialize trigger inverter."""
        print(f"🔄 Trigger Inverter initialized")

    def invert_trigger(
        self,
        model: Any,
        target_class: int,
        iterations: int = 1000,
        trigger_size: float = 0.1
    ) -> np.ndarray:
        """Invert trigger pattern."""
        print(f"\n🔄 Inverting trigger")
        print(f"   Target class: {target_class}")
        print(f"   Iterations: {iterations}")

        # Initialize random trigger
        trigger = np.random.rand(28, 28, 3) * trigger_size

        # Optimize trigger to maximize target class activation
        for i in range(iterations):
            # Gradient descent (simulated)
            # In production: actual backpropagation

            if i % 100 == 0:
                loss = np.random.rand()
                print(f"   Iteration {i}: loss={loss:.4f}")

            # Update trigger
            gradient = np.random.randn(*trigger.shape) * 0.01
            trigger -= gradient

            # Clip trigger
            trigger = np.clip(trigger, 0, trigger_size)

        print(f"   ✓ Trigger inverted")

        return trigger

    def visualize_trigger(self, trigger: np.ndarray) -> None:
        """Visualize trigger pattern."""
        print(f"\n📊 Trigger visualization")
        print(f"   Shape: {trigger.shape}")
        print(f"   Min: {trigger.min():.4f}")
        print(f"   Max: {trigger.max():.4f}")
        print(f"   Mean: {trigger.mean():.4f}")


class DataSanitizer:
    """Remove poisoned samples from dataset."""

    def __init__(self):
        """Initialize data sanitizer."""
        print(f"🧹 Data Sanitizer initialized")

    def remove_poison(
        self,
        X: np.ndarray,
        y: np.ndarray,
        poisoned_indices: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove poisoned samples."""
        print(f"\n🧹 Sanitizing dataset")
        print(f"   Original size: {len(X)}")
        print(f"   Removing: {len(poisoned_indices)} samples")

        # Create mask
        mask = np.ones(len(X), dtype=bool)
        mask[poisoned_indices] = False

        # Filter data
        X_clean = X[mask]
        y_clean = y[mask]

        print(f"   Clean size: {len(X_clean)}")
        print(f"   ✓ Sanitization complete")

        return X_clean, y_clean


class ModelRepair:
    """Repair backdoored models."""

    def __init__(self):
        """Initialize model repairer."""
        print(f"🔧 Model Repair initialized")

    def fine_prune(
        self,
        model: Any,
        trigger: np.ndarray,
        prune_ratio: float = 0.1
    ) -> Any:
        """Fine-pruning to remove backdoor."""
        print(f"\n🔧 Fine-pruning model")
        print(f"   Prune ratio: {prune_ratio:.1%}")

        # Identify neurons activated by trigger
        # Prune those neurons

        # Simulate pruning
        print(f"   Identifying backdoor neurons...")
        print(f"   Pruning {prune_ratio:.1%} of parameters...")
        print(f"   ✓ Model repaired")

        return model


def demo():
    """Demonstrate poison detection."""
    print("=" * 60)
    print("Poison Detection Demo")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.rand(1000, 28, 28, 3)
    y_train = np.random.randint(0, 10, 1000)

    # Data Poisoning Detection
    print(f"\n{'='*60}")
    print("Data Poisoning Detection")
    print(f"{'='*60}")

    detector = PoisonDetector(
        detection_method="anomaly",
        threshold=0.95
    )

    results = detector.scan_dataset(
        X_train, y_train,
        contamination_rate=0.05
    )

    # Data Sanitization
    print(f"\n{'='*60}")
    print("Data Sanitization")
    print(f"{'='*60}")

    sanitizer = DataSanitizer()

    X_clean, y_clean = sanitizer.remove_poison(
        X_train, y_train,
        poisoned_indices=results.poisoned_indices
    )

    # Backdoor Detection
    print(f"\n{'='*60}")
    print("Backdoor Detection")
    print(f"{'='*60}")

    backdoor_detector = BackdoorDetector(
        method="activation_clustering"
    )

    # Simulate model
    model = None

    backdoor_report = backdoor_detector.detect(
        model=model,
        clean_data=X_clean[:100],
        num_classes=10
    )

    if backdoor_report.has_backdoor:
        # Trigger Inversion
        print(f"\n{'='*60}")
        print("Trigger Inversion")
        print(f"{'='*60}")

        inverter = TriggerInverter()

        trigger = inverter.invert_trigger(
            model=model,
            target_class=backdoor_report.target_class,
            iterations=500
        )

        inverter.visualize_trigger(trigger)

        # Model Repair
        print(f"\n{'='*60}")
        print("Model Repair")
        print(f"{'='*60}")

        repairer = ModelRepair()

        repaired_model = repairer.fine_prune(
            model=model,
            trigger=trigger,
            prune_ratio=0.1
        )

    # Spectral Detection
    print(f"\n{'='*60}")
    print("Spectral Detection")
    print(f"{'='*60}")

    spectral_detector = PoisonDetector(
        detection_method="spectral",
        threshold=0.95
    )

    spectral_results = spectral_detector.scan_dataset(
        X_train, y_train,
        contamination_rate=0.05
    )


if __name__ == "__main__":
    demo()
