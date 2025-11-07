"""
Adversarial Robustness
======================

Defense against adversarial attacks on ML models

Author: Brill Consulting
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np


class AttackType(Enum):
    """Adversarial attack types."""
    FGSM = "fgsm"
    PGD = "pgd"
    CW = "carlini_wagner"
    DEEPFOOL = "deepfool"
    JSMA = "jsma"
    PATCH = "patch"
    UNKNOWN = "unknown"


class DefenseMechanism(Enum):
    """Defense strategies."""
    ADVERSARIAL_TRAINING = "adversarial_training"
    GRADIENT_MASKING = "gradient_masking"
    DEFENSIVE_DISTILLATION = "defensive_distillation"
    INPUT_TRANSFORMATION = "input_transformation"
    RANDOMIZED_SMOOTHING = "randomized_smoothing"
    ENSEMBLE = "ensemble"


@dataclass
class DetectionResult:
    """Adversarial detection result."""
    is_adversarial: bool
    confidence: float
    attack_type: AttackType
    perturbation_magnitude: float
    detection_method: str
    timestamp: str


@dataclass
class RobustnessReport:
    """Model robustness evaluation."""
    clean_accuracy: float
    robust_accuracy: float
    attack_success_rate: float
    mean_perturbation: float
    vulnerable_samples: int
    robustness_score: float
    attack_breakdown: Dict[str, float]
    timestamp: str


class AdversarialDetector:
    """Detect adversarial examples in real-time."""

    def __init__(
        self,
        model: Any = None,
        sensitivity: float = 0.8,
        detection_methods: Optional[List[str]] = None
    ):
        """Initialize adversarial detector."""
        self.model = model
        self.sensitivity = sensitivity
        self.detection_methods = detection_methods or ["statistical", "neural"]
        self.detection_count = 0
        self.adversarial_count = 0

        print(f"🛡️  Adversarial Detector initialized")
        print(f"   Sensitivity: {sensitivity}")
        print(f"   Methods: {', '.join(self.detection_methods)}")

    def detect(self, input_sample: np.ndarray) -> DetectionResult:
        """Detect if input is adversarial."""
        self.detection_count += 1

        print(f"\n🔍 Analyzing input #{self.detection_count}")

        # Multiple detection methods
        scores = {}

        if "statistical" in self.detection_methods:
            scores["statistical"] = self._statistical_detection(input_sample)

        if "neural" in self.detection_methods:
            scores["neural"] = self._neural_detection(input_sample)

        if "ensemble" in self.detection_methods:
            scores["ensemble"] = self._ensemble_detection(input_sample)

        # Aggregate scores
        avg_score = np.mean(list(scores.values()))
        is_adversarial = avg_score > self.sensitivity

        if is_adversarial:
            self.adversarial_count += 1
            attack_type = self._identify_attack(input_sample)
            perturbation = self._estimate_perturbation(input_sample)
        else:
            attack_type = AttackType.UNKNOWN
            perturbation = 0.0

        result = DetectionResult(
            is_adversarial=is_adversarial,
            confidence=float(avg_score),
            attack_type=attack_type,
            perturbation_magnitude=perturbation,
            detection_method=", ".join(self.detection_methods),
            timestamp=datetime.now().isoformat()
        )

        if result.is_adversarial:
            print(f"   ⚠️  ADVERSARIAL INPUT DETECTED")
            print(f"   Attack type: {attack_type.value}")
            print(f"   Confidence: {avg_score:.2%}")
            print(f"   Perturbation: {perturbation:.4f}")
        else:
            print(f"   ✅ Input appears clean")

        return result

    def _statistical_detection(self, sample: np.ndarray) -> float:
        """Statistical anomaly detection."""
        # Simulate statistical tests
        # In production: check feature statistics, KL divergence, etc.

        # Check for extreme values
        extreme_ratio = np.mean(np.abs(sample) > 0.95)

        # Check for unusual patterns
        gradient_magnitude = np.mean(np.abs(np.gradient(sample.flatten())))

        # Combine metrics
        anomaly_score = (extreme_ratio + min(gradient_magnitude / 0.5, 1.0)) / 2

        return float(anomaly_score)

    def _neural_detection(self, sample: np.ndarray) -> float:
        """Neural network-based detection."""
        # Simulate detector network
        # In production: use trained adversarial detector

        # Simulate prediction confidence analysis
        if self.model is not None:
            # Check prediction stability
            noise = np.random.randn(*sample.shape) * 0.01
            perturbed = sample + noise

            # Simulated confidence
            base_confidence = np.random.uniform(0.6, 0.95)
            perturbed_confidence = base_confidence + np.random.uniform(-0.3, 0.1)

            # Large confidence drop indicates adversarial
            confidence_drop = abs(base_confidence - perturbed_confidence)
            return float(min(confidence_drop * 3, 1.0))

        return 0.3

    def _ensemble_detection(self, sample: np.ndarray) -> float:
        """Ensemble-based detection."""
        # Simulate multiple model predictions
        # In production: use ensemble of models

        predictions = []
        for _ in range(5):
            # Simulate model prediction
            pred = np.random.randint(0, 10)
            predictions.append(pred)

        # Check prediction consistency
        unique_preds = len(set(predictions))
        inconsistency_score = unique_preds / 5.0

        return float(inconsistency_score)

    def _identify_attack(self, sample: np.ndarray) -> AttackType:
        """Identify type of adversarial attack."""
        # Analyze perturbation characteristics
        gradient_norm = np.linalg.norm(np.gradient(sample.flatten()))

        if gradient_norm < 0.1:
            return AttackType.CW  # Smooth perturbations
        elif gradient_norm > 0.5:
            return AttackType.FGSM  # Sharp perturbations
        else:
            return AttackType.PGD  # Iterative attacks

    def _estimate_perturbation(self, sample: np.ndarray) -> float:
        """Estimate perturbation magnitude."""
        # Simulate perturbation estimation
        # In production: compare with original or reference distribution

        return float(np.random.uniform(0.05, 0.3))

    def sanitize(self, adversarial_input: np.ndarray) -> np.ndarray:
        """Remove adversarial perturbations."""
        print(f"\n🧹 Sanitizing adversarial input")

        # Apply multiple sanitization techniques
        sanitized = adversarial_input.copy()

        # 1. Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        if len(sanitized.shape) > 1:
            sanitized = gaussian_filter(sanitized, sigma=0.5)

        # 2. Quantization (bit-depth reduction)
        sanitized = np.round(sanitized * 255) / 255

        # 3. JPEG-like compression simulation
        # In production: actual JPEG compression
        sanitized = np.round(sanitized * 128) / 128

        perturbation_removed = np.linalg.norm(adversarial_input - sanitized)
        print(f"   Perturbation removed: {perturbation_removed:.4f}")
        print(f"   ✓ Input sanitized")

        return sanitized


class AdversarialTrainer:
    """Train models with adversarial examples."""

    def __init__(
        self,
        model: Any = None,
        attack_method: str = "pgd",
        epsilon: float = 0.3,
        alpha: float = 0.01,
        iterations: int = 40
    ):
        """Initialize adversarial trainer."""
        self.model = model
        self.attack_method = attack_method
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations

        print(f"🎯 Adversarial Trainer initialized")
        print(f"   Attack method: {attack_method}")
        print(f"   Epsilon: {epsilon}")
        print(f"   Iterations: {iterations}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32
    ) -> Any:
        """Train model with adversarial examples."""
        print(f"\n🏋️  Training robust model")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            print(f"\n   Epoch {epoch}/{epochs}")

            # Generate adversarial examples
            X_adv = self._generate_adversarial_batch(X_train, y_train)

            # Mix clean and adversarial examples
            X_mixed = np.concatenate([X_train, X_adv])
            y_mixed = np.concatenate([y_train, y_train])

            # Simulate training
            # In production: actual model.fit()
            train_loss = 1.0 / epoch
            train_acc = 0.5 + (epoch / epochs) * 0.4

            print(f"      Loss: {train_loss:.4f}")
            print(f"      Accuracy: {train_acc:.2%}")

        print(f"\n   ✓ Adversarial training complete")
        return self.model

    def _generate_adversarial_batch(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Generate adversarial examples."""
        if self.attack_method == "fgsm":
            return self._fgsm_attack(X, y)
        elif self.attack_method == "pgd":
            return self._pgd_attack(X, y)
        else:
            return self._fgsm_attack(X, y)

    def _fgsm_attack(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fast Gradient Sign Method attack."""
        # Simulate FGSM
        # In production: compute gradient and add perturbation
        perturbation = np.random.randn(*X.shape) * self.epsilon
        perturbation = np.sign(perturbation) * self.epsilon

        X_adv = X + perturbation
        X_adv = np.clip(X_adv, 0, 1)

        return X_adv

    def _pgd_attack(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Projected Gradient Descent attack."""
        # Simulate PGD
        X_adv = X.copy()

        for _ in range(self.iterations):
            # Gradient step
            perturbation = np.random.randn(*X.shape) * self.alpha
            X_adv = X_adv + np.sign(perturbation) * self.alpha

            # Project back to epsilon ball
            perturbation = X_adv - X
            perturbation = np.clip(perturbation, -self.epsilon, self.epsilon)
            X_adv = X + perturbation

            # Clip to valid range
            X_adv = np.clip(X_adv, 0, 1)

        return X_adv


class RobustnessEvaluator:
    """Evaluate model robustness against attacks."""

    def __init__(self, model: Any = None):
        """Initialize robustness evaluator."""
        self.model = model
        print(f"📊 Robustness Evaluator initialized")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        attacks: Optional[List[str]] = None,
        epsilon: float = 0.3
    ) -> RobustnessReport:
        """Evaluate model robustness."""
        print(f"\n🔬 Evaluating model robustness")

        attacks = attacks or ["fgsm", "pgd", "cw"]
        print(f"   Testing attacks: {', '.join(attacks)}")
        print(f"   Epsilon: {epsilon}")

        # Clean accuracy
        clean_acc = self._evaluate_clean(X_test, y_test)

        # Test each attack
        attack_results = {}
        all_adversarial = []

        for attack_name in attacks:
            print(f"\n   Testing {attack_name.upper()} attack...")

            X_adv = self._generate_attack(X_test, y_test, attack_name, epsilon)
            adv_acc = self._evaluate_adversarial(X_adv, y_test)

            attack_results[attack_name] = adv_acc
            all_adversarial.extend(X_adv)

            print(f"      Robust accuracy: {adv_acc:.2%}")

        # Calculate metrics
        robust_acc = np.mean(list(attack_results.values()))
        asr = 1.0 - robust_acc
        mean_perturbation = epsilon  # Simplified

        vulnerable_count = int(len(X_test) * asr)
        robustness_score = (clean_acc + robust_acc) / 2

        report = RobustnessReport(
            clean_accuracy=clean_acc,
            robust_accuracy=robust_acc,
            attack_success_rate=asr,
            mean_perturbation=mean_perturbation,
            vulnerable_samples=vulnerable_count,
            robustness_score=robustness_score,
            attack_breakdown=attack_results,
            timestamp=datetime.now().isoformat()
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"Robustness Report")
        print(f"{'='*60}")
        print(f"Clean accuracy: {clean_acc:.2%}")
        print(f"Robust accuracy: {robust_acc:.2%}")
        print(f"Attack success rate: {asr:.2%}")
        print(f"Robustness score: {robustness_score:.2%}")
        print(f"Vulnerable samples: {vulnerable_count}/{len(X_test)}")

        return report

    def _evaluate_clean(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate on clean data."""
        # Simulate model evaluation
        return float(np.random.uniform(0.90, 0.95))

    def _evaluate_adversarial(self, X_adv: np.ndarray, y: np.ndarray) -> float:
        """Evaluate on adversarial data."""
        # Simulate adversarial evaluation
        return float(np.random.uniform(0.40, 0.70))

    def _generate_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_name: str,
        epsilon: float
    ) -> np.ndarray:
        """Generate adversarial attack."""
        if attack_name == "fgsm":
            perturbation = np.random.randn(*X.shape)
            perturbation = np.sign(perturbation) * epsilon
        elif attack_name == "pgd":
            perturbation = np.random.randn(*X.shape)
            perturbation = np.clip(perturbation, -epsilon, epsilon)
        else:  # C&W
            perturbation = np.random.randn(*X.shape) * epsilon * 0.5

        X_adv = X + perturbation
        X_adv = np.clip(X_adv, 0, 1)

        return X_adv

    def comprehensive_test(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        attack_budgets: List[float] = None
    ) -> Dict[str, RobustnessReport]:
        """Test across multiple epsilon values."""
        print(f"\n🎯 Comprehensive robustness testing")

        attack_budgets = attack_budgets or [0.1, 0.2, 0.3]
        reports = {}

        for epsilon in attack_budgets:
            print(f"\n{'='*60}")
            print(f"Testing with epsilon = {epsilon}")
            print(f"{'='*60}")

            report = self.evaluate(X_test, y_test, epsilon=epsilon)
            reports[f"epsilon_{epsilon}"] = report

        return reports


def demo():
    """Demonstrate adversarial robustness."""
    print("=" * 60)
    print("Adversarial Robustness Demo")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.rand(100, 28, 28)
    y_train = np.random.randint(0, 10, 100)
    X_test = np.random.rand(20, 28, 28)
    y_test = np.random.randint(0, 10, 20)

    # Adversarial Training
    print(f"\n{'='*60}")
    print("Adversarial Training")
    print(f"{'='*60}")

    trainer = AdversarialTrainer(
        attack_method="pgd",
        epsilon=0.3,
        iterations=40
    )

    robust_model = trainer.train(X_train, y_train, epochs=5)

    # Detection
    print(f"\n{'='*60}")
    print("Adversarial Detection")
    print(f"{'='*60}")

    detector = AdversarialDetector(
        model=robust_model,
        sensitivity=0.8,
        detection_methods=["statistical", "neural", "ensemble"]
    )

    # Test clean samples
    print(f"\nTesting clean samples:")
    for i in range(3):
        result = detector.detect(X_test[i])

    # Generate adversarial samples
    print(f"\nGenerating adversarial samples:")
    X_adv = trainer._fgsm_attack(X_test[:3], y_test[:3])

    for i, adv_sample in enumerate(X_adv):
        result = detector.detect(adv_sample)

        if result.is_adversarial:
            # Sanitize
            cleaned = detector.sanitize(adv_sample)

    # Robustness Evaluation
    print(f"\n{'='*60}")
    print("Robustness Evaluation")
    print(f"{'='*60}")

    evaluator = RobustnessEvaluator(model=robust_model)

    report = evaluator.evaluate(
        X_test, y_test,
        attacks=["fgsm", "pgd", "cw"],
        epsilon=0.3
    )

    # Comprehensive testing
    print(f"\n{'='*60}")
    print("Comprehensive Testing")
    print(f"{'='*60}")

    reports = evaluator.comprehensive_test(
        X_test, y_test,
        attack_budgets=[0.1, 0.2, 0.3]
    )


if __name__ == "__main__":
    demo()
