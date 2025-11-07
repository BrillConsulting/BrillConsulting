"""
GPU Scheduling & Scaling
=========================

Kubernetes GPU scheduling with NVIDIA GPU Operator

Author: Brill Consulting
"""

from typing import List, Dict
from dataclasses import dataclass
from enum import Enum


class GPUType(Enum):
    """GPU types."""
    A100 = "a100"
    V100 = "v100"
    T4 = "t4"


@dataclass
class GPUNode:
    """Kubernetes node with GPUs."""
    name: str
    gpu_type: GPUType
    gpu_count: int
    available_gpus: int


class GPUScheduler:
    """Kubernetes GPU scheduler."""

    def __init__(self):
        """Initialize scheduler."""
        self.nodes: List[GPUNode] = []
        self.pods_scheduled = 0

        print(f"⚙️  GPU Scheduler initialized")

    def add_node(self, name: str, gpu_type: str, gpu_count: int) -> None:
        """Add GPU node to cluster."""
        node = GPUNode(
            name=name,
            gpu_type=GPUType(gpu_type),
            gpu_count=gpu_count,
            available_gpus=gpu_count
        )
        self.nodes.append(node)

        print(f"   ✓ Added node: {name} ({gpu_type} x{gpu_count})")

    def schedule_pod(self, required_gpus: int = 1) -> Dict:
        """Schedule pod on available node."""
        for node in self.nodes:
            if node.available_gpus >= required_gpus:
                node.available_gpus -= required_gpus
                self.pods_scheduled += 1

                print(f"   📦 Scheduled pod on {node.name}")
                print(f"      GPUs: {required_gpus}x {node.gpu_type.value}")

                return {
                    "status": "scheduled",
                    "node": node.name,
                    "gpus_allocated": required_gpus
                }

        return {"status": "pending", "reason": "insufficient_gpus"}

    def scale_node_pool(self, target_nodes: int) -> None:
        """Scale GPU node pool."""
        print(f"\n📈 Scaling node pool to {target_nodes} nodes")
        # Simulate scaling
        print(f"   ✓ Scaled successfully")


def demo():
    """Demonstrate GPU scheduling."""
    print("=" * 60)
    print("GPU Scheduling Demo")
    print("=" * 60)

    scheduler = GPUScheduler()

    # Add nodes
    scheduler.add_node("gpu-node-1", "a100", 8)
    scheduler.add_node("gpu-node-2", "a100", 8)
    scheduler.add_node("gpu-node-3", "v100", 4)

    # Schedule pods
    print(f"\n{'='*60}")
    print("Scheduling pods")
    print(f"{'='*60}")

    for i in range(5):
        result = scheduler.schedule_pod(required_gpus=2)

    # Scale
    scheduler.scale_node_pool(5)


if __name__ == "__main__":
    demo()
