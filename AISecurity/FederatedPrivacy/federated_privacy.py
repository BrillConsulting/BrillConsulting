"""
Federated Privacy
=================

Privacy-preserving federated learning with differential privacy

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np
import math


class PrivacyMechanism(Enum):
    """Privacy mechanisms."""
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"


class CompositionType(Enum):
    """Composition theorems."""
    BASIC = "basic"
    ADVANCED = "advanced"
    RENYI = "renyi"


@dataclass
class PrivacyParams:
    """Differential privacy parameters."""
    epsilon: float
    delta: float
    sensitivity: float


@dataclass
class PrivacyBudgetState:
    """Current privacy budget state."""
    total_epsilon: float
    spent_epsilon: float
    remaining_epsilon: float
    delta: float
    query_count: int
    is_exhausted: bool


@dataclass
class AuditResult:
    """Privacy audit result."""
    claimed_epsilon: float
    empirical_epsilon: float
    is_violated: bool
    confidence: float
    timestamp: str


class DifferentialPrivacy:
    """Differential privacy mechanism."""

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        mechanism: str = "gaussian"
    ):
        """Initialize differential privacy."""
        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = PrivacyMechanism(mechanism)

        print(f"🔒 Differential Privacy initialized")
        print(f"   Epsilon: {epsilon}")
        print(f"   Delta: {delta}")
        print(f"   Mechanism: {mechanism}")

    def add_noise(
        self,
        value: float,
        sensitivity: float = 1.0
    ) -> float:
        """Add calibrated noise to value."""
        if self.mechanism == PrivacyMechanism.LAPLACE:
            return self._laplace_noise(value, sensitivity)
        elif self.mechanism == PrivacyMechanism.GAUSSIAN:
            return self._gaussian_noise(value, sensitivity)
        else:
            return value

    def _laplace_noise(self, value: float, sensitivity: float) -> float:
        """Laplace mechanism."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise

    def _gaussian_noise(self, value: float, sensitivity: float) -> float:
        """Gaussian mechanism."""
        # Calculate sigma for (epsilon, delta)-DP
        sigma = self._calculate_gaussian_sigma(sensitivity)
        noise = np.random.normal(0, sigma)
        return value + noise

    def _calculate_gaussian_sigma(self, sensitivity: float) -> float:
        """Calculate sigma for Gaussian mechanism."""
        # From (epsilon, delta)-DP to sigma
        # sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        return sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

    def privatize_gradient(
        self,
        gradient: np.ndarray,
        clip_norm: float = 1.0
    ) -> np.ndarray:
        """Add privacy to gradient."""
        # Clip gradient
        clipped = self._clip_gradient(gradient, clip_norm)

        # Add noise
        sensitivity = clip_norm
        sigma = self._calculate_gaussian_sigma(sensitivity)
        noise = np.random.normal(0, sigma, size=gradient.shape)

        return clipped + noise

    def _clip_gradient(
        self,
        gradient: np.ndarray,
        max_norm: float
    ) -> np.ndarray:
        """Clip gradient to bound sensitivity."""
        norm = np.linalg.norm(gradient)
        if norm > max_norm:
            return gradient * (max_norm / norm)
        return gradient


class PrivacyBudget:
    """Track and manage privacy budget."""

    def __init__(
        self,
        total_epsilon: float = 10.0,
        delta: float = 1e-5,
        composition: str = "advanced"
    ):
        """Initialize privacy budget."""
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.composition = CompositionType(composition)
        self.spent_epsilon = 0.0
        self.query_count = 0
        self.query_history: List[float] = []

        print(f"💰 Privacy Budget initialized")
        print(f"   Total epsilon: {total_epsilon}")
        print(f"   Delta: {delta}")
        print(f"   Composition: {composition}")

    def can_spend(self, epsilon: float) -> bool:
        """Check if budget allows spending epsilon."""
        composed_epsilon = self._compose_epsilon(epsilon)
        return self.spent_epsilon + composed_epsilon <= self.total_epsilon

    def spend(self, epsilon: float) -> bool:
        """Spend privacy budget."""
        if not self.can_spend(epsilon):
            print(f"   ⚠️  Insufficient privacy budget")
            return False

        composed_epsilon = self._compose_epsilon(epsilon)
        self.spent_epsilon += composed_epsilon
        self.query_count += 1
        self.query_history.append(epsilon)

        print(f"   Privacy spent: ε={epsilon:.2f}")
        print(f"   Remaining: ε={self.remaining_epsilon:.2f}")

        return True

    def _compose_epsilon(self, epsilon: float) -> float:
        """Calculate composed epsilon."""
        if self.composition == CompositionType.BASIC:
            # Basic composition: sum of epsilons
            return epsilon

        elif self.composition == CompositionType.ADVANCED:
            # Advanced composition
            k = self.query_count + 1
            term1 = math.sqrt(2 * k * math.log(1 / self.delta)) * epsilon
            term2 = k * epsilon * (math.exp(epsilon) - 1)
            return term1 + term2

        else:  # Renyi DP
            # Simplified Renyi composition
            return epsilon * 1.1  # Conservative bound

    @property
    def remaining_epsilon(self) -> float:
        """Get remaining privacy budget."""
        return self.total_epsilon - self.spent_epsilon

    @property
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.remaining_epsilon <= 0

    def get_state(self) -> PrivacyBudgetState:
        """Get current budget state."""
        return PrivacyBudgetState(
            total_epsilon=self.total_epsilon,
            spent_epsilon=self.spent_epsilon,
            remaining_epsilon=self.remaining_epsilon,
            delta=self.delta,
            query_count=self.query_count,
            is_exhausted=self.is_exhausted
        )


class SecureAggregator:
    """Secure aggregation for federated learning."""

    def __init__(
        self,
        num_clients: int = 100,
        threshold: float = 0.8,
        encryption: str = "paillier"
    ):
        """Initialize secure aggregator."""
        self.num_clients = num_clients
        self.threshold = threshold
        self.encryption = encryption
        self.min_clients = int(num_clients * threshold)

        print(f"🔐 Secure Aggregator initialized")
        print(f"   Clients: {num_clients}")
        print(f"   Threshold: {threshold:.0%}")
        print(f"   Min clients: {self.min_clients}")
        print(f"   Encryption: {encryption}")

    def encrypt(
        self,
        gradient: np.ndarray,
        client_id: int
    ) -> Dict[str, Any]:
        """Encrypt client gradient."""
        # Simulate encryption
        # In production: use actual Paillier or CKKS encryption

        encrypted = {
            "client_id": client_id,
            "encrypted_data": gradient,  # In reality, this would be encrypted
            "timestamp": datetime.now().isoformat()
        }

        return encrypted

    def aggregate(
        self,
        encrypted_gradients: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Aggregate encrypted gradients."""
        print(f"\n🔒 Secure aggregation")
        print(f"   Participants: {len(encrypted_gradients)}/{self.num_clients}")

        if len(encrypted_gradients) < self.min_clients:
            raise ValueError(
                f"Insufficient clients: {len(encrypted_gradients)} < {self.min_clients}"
            )

        # Homomorphic addition (simulated)
        # In production: actual homomorphic operations
        gradients = [eg["encrypted_data"] for eg in encrypted_gradients]
        aggregated = np.mean(gradients, axis=0)

        print(f"   ✓ Aggregation complete")

        return aggregated

    def decrypt(self, aggregated: np.ndarray) -> np.ndarray:
        """Decrypt aggregated result."""
        # Simulate decryption
        # In production: actual decryption
        return aggregated


class FederatedTrainer:
    """Privacy-preserving federated learning trainer."""

    def __init__(
        self,
        privacy_mechanism: Optional[DifferentialPrivacy] = None,
        secure_aggregation: bool = True,
        learning_rate: float = 0.01
    ):
        """Initialize federated trainer."""
        self.privacy = privacy_mechanism
        self.secure_aggregation = secure_aggregation
        self.learning_rate = learning_rate
        self.round_count = 0

        print(f"🌐 Federated Trainer initialized")
        print(f"   Privacy: {'enabled' if privacy_mechanism else 'disabled'}")
        print(f"   Secure aggregation: {secure_aggregation}")
        print(f"   Learning rate: {learning_rate}")

    def train(
        self,
        clients: List[Any],
        rounds: int = 50,
        local_epochs: int = 5,
        client_fraction: float = 0.1
    ) -> Any:
        """Train federated model."""
        print(f"\n🏋️  Starting federated training")
        print(f"   Total clients: {len(clients)}")
        print(f"   Rounds: {rounds}")
        print(f"   Local epochs: {local_epochs}")

        # Initialize secure aggregator if needed
        if self.secure_aggregation:
            aggregator = SecureAggregator(
                num_clients=len(clients),
                threshold=client_fraction
            )

        # Training loop
        for round_num in range(1, rounds + 1):
            self.round_count = round_num

            print(f"\n{'='*60}")
            print(f"Round {round_num}/{rounds}")
            print(f"{'='*60}")

            # Select clients
            num_selected = max(1, int(len(clients) * client_fraction))
            selected_clients = np.random.choice(clients, num_selected, replace=False)

            print(f"   Selected clients: {num_selected}")

            # Collect gradients
            gradients = []

            for i, client in enumerate(selected_clients):
                # Simulate local training
                client_gradient = np.random.randn(100)  # Simulated

                # Apply privacy if enabled
                if self.privacy:
                    client_gradient = self.privacy.privatize_gradient(
                        client_gradient,
                        clip_norm=1.0
                    )

                gradients.append(client_gradient)

            # Aggregate
            if self.secure_aggregation:
                # Encrypt and aggregate
                encrypted = [
                    aggregator.encrypt(grad, i)
                    for i, grad in enumerate(gradients)
                ]
                aggregated = aggregator.aggregate(encrypted)
                final_gradient = aggregator.decrypt(aggregated)
            else:
                # Simple averaging
                final_gradient = np.mean(gradients, axis=0)

            # Simulate model update
            # In production: apply gradient to model
            loss = 1.0 / round_num
            accuracy = 0.5 + (round_num / rounds) * 0.4

            print(f"   Loss: {loss:.4f}")
            print(f"   Accuracy: {accuracy:.2%}")

        print(f"\n✓ Federated training complete")

        return None  # Would return trained model

    def train_round(
        self,
        selected_clients: List[Any],
        global_model: Any
    ) -> np.ndarray:
        """Execute single training round."""
        gradients = []

        for client in selected_clients:
            # Local training
            gradient = self._local_train(client, global_model)

            # Apply privacy
            if self.privacy:
                gradient = self.privacy.privatize_gradient(gradient)

            gradients.append(gradient)

        # Aggregate
        return np.mean(gradients, axis=0)

    def _local_train(self, client: Any, model: Any) -> np.ndarray:
        """Local training on client."""
        # Simulate local training
        return np.random.randn(100)


class PrivacyAuditor:
    """Audit actual privacy leakage."""

    def __init__(self):
        """Initialize privacy auditor."""
        print(f"🔍 Privacy Auditor initialized")

    def audit(
        self,
        model: Any,
        training_data: np.ndarray,
        epsilon_claimed: float,
        num_samples: int = 1000
    ) -> AuditResult:
        """Audit privacy guarantees."""
        print(f"\n📊 Auditing privacy")
        print(f"   Claimed epsilon: {epsilon_claimed}")
        print(f"   Test samples: {num_samples}")

        # Membership inference attack
        empirical_epsilon = self._membership_inference(
            model, training_data, num_samples
        )

        is_violated = empirical_epsilon > epsilon_claimed * 1.2

        result = AuditResult(
            claimed_epsilon=epsilon_claimed,
            empirical_epsilon=empirical_epsilon,
            is_violated=is_violated,
            confidence=0.95,
            timestamp=datetime.now().isoformat()
        )

        print(f"\n   Results:")
        print(f"   Claimed: ε={epsilon_claimed:.2f}")
        print(f"   Empirical: ε={empirical_epsilon:.2f}")

        if is_violated:
            print(f"   ⚠️  Privacy violation detected!")
        else:
            print(f"   ✓ Privacy guarantees hold")

        return result

    def _membership_inference(
        self,
        model: Any,
        data: np.ndarray,
        num_samples: int
    ) -> float:
        """Perform membership inference attack."""
        # Simulate membership inference
        # In production: actual attack implementation

        # Simulate empirical epsilon (usually higher than claimed)
        empirical = np.random.uniform(0.8, 1.5) * 1.2

        return empirical


def demo():
    """Demonstrate federated privacy."""
    print("=" * 60)
    print("Federated Privacy Demo")
    print("=" * 60)

    # Differential Privacy
    print(f"\n{'='*60}")
    print("Differential Privacy")
    print(f"{'='*60}")

    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, mechanism="gaussian")

    # Privatize value
    true_value = 100.0
    private_value = dp.add_noise(true_value, sensitivity=1.0)
    print(f"\n   True value: {true_value}")
    print(f"   Private value: {private_value:.2f}")

    # Privatize gradient
    gradient = np.random.randn(10)
    private_gradient = dp.privatize_gradient(gradient, clip_norm=1.0)
    print(f"\n   Gradient norm: {np.linalg.norm(gradient):.4f}")
    print(f"   Private gradient norm: {np.linalg.norm(private_gradient):.4f}")

    # Privacy Budget
    print(f"\n{'='*60}")
    print("Privacy Budget Management")
    print(f"{'='*60}")

    budget = PrivacyBudget(
        total_epsilon=10.0,
        delta=1e-5,
        composition="advanced"
    )

    # Make queries
    for i in range(5):
        print(f"\n   Query {i+1}:")
        if budget.can_spend(epsilon=1.0):
            budget.spend(epsilon=1.0)
        else:
            print(f"      Budget exhausted!")
            break

    state = budget.get_state()
    print(f"\n   Final budget state:")
    print(f"   Total epsilon: {state.total_epsilon}")
    print(f"   Spent: {state.spent_epsilon:.2f}")
    print(f"   Remaining: {state.remaining_epsilon:.2f}")
    print(f"   Queries: {state.query_count}")

    # Secure Aggregation
    print(f"\n{'='*60}")
    print("Secure Aggregation")
    print(f"{'='*60}")

    aggregator = SecureAggregator(
        num_clients=100,
        threshold=0.8,
        encryption="paillier"
    )

    # Simulate client gradients
    num_clients = 85
    gradients = [np.random.randn(50) for _ in range(num_clients)]

    # Encrypt
    encrypted_gradients = [
        aggregator.encrypt(grad, i)
        for i, grad in enumerate(gradients)
    ]

    # Aggregate
    aggregated = aggregator.aggregate(encrypted_gradients)
    final = aggregator.decrypt(aggregated)

    print(f"\n   Aggregated gradient shape: {final.shape}")

    # Federated Training
    print(f"\n{'='*60}")
    print("Federated Training")
    print(f"{'='*60}")

    # Create privacy mechanism
    privacy = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

    # Initialize trainer
    trainer = FederatedTrainer(
        privacy_mechanism=privacy,
        secure_aggregation=True,
        learning_rate=0.01
    )

    # Simulate clients
    clients = [f"client_{i}" for i in range(100)]

    # Train
    model = trainer.train(
        clients=clients,
        rounds=10,
        local_epochs=5,
        client_fraction=0.1
    )

    # Privacy Auditing
    print(f"\n{'='*60}")
    print("Privacy Auditing")
    print(f"{'='*60}")

    auditor = PrivacyAuditor()

    # Simulate training data
    training_data = np.random.randn(1000, 28, 28)

    audit_result = auditor.audit(
        model=model,
        training_data=training_data,
        epsilon_claimed=1.0,
        num_samples=1000
    )


if __name__ == "__main__":
    demo()
