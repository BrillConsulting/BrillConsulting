"""
Federated Edge Learning
=======================

Privacy-preserving federated learning on edge

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ClientUpdate:
    """Client update package."""
    client_id: str
    gradients: np.ndarray
    num_samples: int
    loss: float


class FederatedServer:
    """Federated learning server."""

    def __init__(
        self,
        model: Any,
        aggregation_method: str = "fedavg",
        min_clients: int = 5,
        privacy: bool = False
    ):
        self.model = model
        self.aggregation_method = aggregation_method
        self.min_clients = min_clients
        self.privacy = privacy
        self.round_num = 0

        print(f"🌐 Federated Server initialized")
        print(f"   Aggregation: {aggregation_method}")
        print(f"   Min clients: {min_clients}")
        print(f"   Privacy: {privacy}")

    def select_clients(self, fraction: float = 0.1) -> List[str]:
        """Select clients for round."""
        # Simulate client selection
        num_clients = max(self.min_clients, int(100 * fraction))
        clients = [f"client_{i:03d}" for i in range(num_clients)]

        print(f"\n📱 Selected {len(clients)} clients")
        return clients

    def broadcast_model(self, clients: List[str]) -> None:
        """Send model to clients."""
        print(f"📤 Broadcasting model to {len(clients)} clients")

    def collect_updates(
        self,
        clients: List[str],
        timeout: int = 300
    ) -> List[ClientUpdate]:
        """Collect updates from clients."""
        print(f"📥 Collecting updates (timeout={timeout}s)")

        updates = []
        for client_id in clients:
            # Simulate client update
            update = ClientUpdate(
                client_id=client_id,
                gradients=np.random.randn(1000),
                num_samples=np.random.randint(50, 200),
                loss=np.random.uniform(0.5, 2.0)
            )
            updates.append(update)

        print(f"   Received {len(updates)}/{len(clients)} updates")
        return updates

    def aggregate_updates(self, updates: List[ClientUpdate]) -> None:
        """Aggregate client updates."""
        print(f"\n🔄 Aggregating updates")

        if self.aggregation_method == "fedavg":
            self._fedavg(updates)
        elif self.aggregation_method == "fedprox":
            self._fedprox(updates)
        else:
            self._fedavg(updates)

        self.round_num += 1
        print(f"   ✓ Aggregation complete")

    def _fedavg(self, updates: List[ClientUpdate]) -> None:
        """FedAvg aggregation."""
        total_samples = sum(u.num_samples for u in updates)

        # Weighted average
        aggregated_gradients = np.zeros_like(updates[0].gradients)

        for update in updates:
            weight = update.num_samples / total_samples
            aggregated_gradients += weight * update.gradients

        print(f"   Method: FedAvg (weighted average)")

    def _fedprox(self, updates: List[ClientUpdate]) -> None:
        """FedProx aggregation."""
        print(f"   Method: FedProx (with proximal term)")
        # Similar to FedAvg but with proximal regularization
        self._fedavg(updates)

    def evaluate(self, test_data: Optional[np.ndarray] = None) -> float:
        """Evaluate global model."""
        # Simulate evaluation
        accuracy = 0.85 + (self.round_num / 100) * 0.10
        return min(accuracy, 0.95)


class FederatedClient:
    """Federated learning client."""

    def __init__(
        self,
        client_id: str,
        server_url: str,
        local_epochs: int = 5
    ):
        self.client_id = client_id
        self.server_url = server_url
        self.local_epochs = local_epochs
        self.model = None

        print(f"📱 Federated Client: {client_id}")
        print(f"   Server: {server_url}")
        print(f"   Local epochs: {local_epochs}")

    def train_local(
        self,
        local_data: np.ndarray,
        privacy_budget: Optional[float] = None
    ) -> None:
        """Train on local data."""
        print(f"\n🏋️  Local training")
        print(f"   Samples: {len(local_data)}")
        print(f"   Epochs: {self.local_epochs}")

        if privacy_budget:
            print(f"   Privacy budget: ε={privacy_budget}")

        # Simulate training
        for epoch in range(1, self.local_epochs + 1):
            loss = 1.5 / epoch
            if epoch == self.local_epochs:
                print(f"   Final loss: {loss:.4f}")

        print(f"   ✓ Local training complete")

    def send_update(self) -> None:
        """Send update to server."""
        print(f"📤 Sending update to server")


class DPMechanism:
    """Differential privacy mechanism."""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta

        print(f"🔒 DP Mechanism: ε={epsilon}, δ={delta}")

    def privatize(
        self,
        gradients: np.ndarray,
        clip_norm: float = 1.0
    ) -> np.ndarray:
        """Add differential privacy noise."""
        print(f"\n🔒 Adding DP noise")
        print(f"   Clip norm: {clip_norm}")

        # Clip gradients
        norm = np.linalg.norm(gradients)
        if norm > clip_norm:
            gradients = gradients * (clip_norm / norm)

        # Add Gaussian noise
        noise_scale = clip_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, noise_scale, gradients.shape)

        private_gradients = gradients + noise

        print(f"   ✓ DP noise added")
        return private_gradients


class GradientCompressor:
    """Compress gradients for communication efficiency."""

    def __init__(self, method: str = "topk"):
        self.method = method
        print(f"📦 Gradient Compressor: {method}")

    def compress(
        self,
        gradients: np.ndarray,
        compression_ratio: float = 0.1,
        bits: int = 8
    ) -> np.ndarray:
        """Compress gradients."""
        print(f"\n📦 Compressing gradients")

        if self.method == "topk":
            # Keep only top k% gradients
            k = int(len(gradients) * compression_ratio)
            indices = np.argsort(np.abs(gradients))[-k:]
            compressed = np.zeros_like(gradients)
            compressed[indices] = gradients[indices]

            print(f"   Method: Top-K ({compression_ratio:.0%})")
            print(f"   Size reduction: {1/compression_ratio:.1f}x")

        elif self.method == "quantize":
            # Quantize to fewer bits
            min_val, max_val = gradients.min(), gradients.max()
            scale = (max_val - min_val) / (2**bits - 1)
            quantized = np.round((gradients - min_val) / scale)
            compressed = quantized * scale + min_val

            print(f"   Method: Quantization ({bits}-bit)")
            print(f"   Size reduction: {32/bits:.1f}x")
        else:
            compressed = gradients

        print(f"   ✓ Compression complete")
        return compressed


def demo():
    """Demonstrate federated edge learning."""
    print("=" * 60)
    print("Federated Edge Learning Demo")
    print("=" * 60)

    # Federated server
    print(f"\n{'='*60}")
    print("Federated Server")
    print(f"{'='*60}")

    server = FederatedServer(
        model=None,
        aggregation_method="fedavg",
        min_clients=10,
        privacy=True
    )

    # Training rounds
    for round_num in range(1, 6):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/5")
        print(f"{'='*60}")

        # Select clients
        clients = server.select_clients(fraction=0.1)

        # Broadcast model
        server.broadcast_model(clients)

        # Collect updates
        updates = server.collect_updates(clients, timeout=300)

        # Aggregate
        server.aggregate_updates(updates)

        # Evaluate
        acc = server.evaluate()
        print(f"\n   Global accuracy: {acc:.2%}")

    # Edge client
    print(f"\n{'='*60}")
    print("Edge Client")
    print(f"{'='*60}")

    client = FederatedClient(
        client_id="device_001",
        server_url="https://fl-server.example.com",
        local_epochs=5
    )

    local_data = np.random.rand(1000, 28, 28)

    client.train_local(local_data, privacy_budget=0.1)
    client.send_update()

    # Differential Privacy
    print(f"\n{'='*60}")
    print("Differential Privacy")
    print(f"{'='*60}")

    dp = DPMechanism(epsilon=1.0, delta=1e-5)

    gradients = np.random.randn(1000)
    private_gradients = dp.privatize(gradients, clip_norm=1.0)

    # Gradient Compression
    print(f"\n{'='*60}")
    print("Gradient Compression")
    print(f"{'='*60}")

    compressor = GradientCompressor(method="topk")
    compressed = compressor.compress(gradients, compression_ratio=0.1)

    compressor2 = GradientCompressor(method="quantize")
    quantized = compressor2.compress(gradients, bits=8)


if __name__ == "__main__":
    demo()
