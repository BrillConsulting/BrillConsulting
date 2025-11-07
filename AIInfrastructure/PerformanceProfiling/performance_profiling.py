"""
Performance Profiling Suite
============================

Deep performance analysis with NVIDIA Nsight and PyTorch Profiler

Author: Brill Consulting
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class ProfilingResult:
    """Profiling result data."""
    total_time_ms: float
    gpu_time_ms: float
    cpu_time_ms: float
    memory_allocated_mb: float
    memory_peak_mb: float
    bottlenecks: List[str]


class Profiler:
    """Performance profiler for AI models."""

    def __init__(self, backend: str = "pytorch"):
        """Initialize profiler."""
        self.backend = backend
        self.results: Optional[ProfilingResult] = None
        self.profiling_active = False

        print(f"🔬 Profiler initialized")
        print(f"   Backend: {backend}")

    def profile(self):
        """Context manager for profiling."""
        return ProfilingContext(self)

    def analyze(self) -> Dict[str, Any]:
        """Analyze profiling results."""
        if not self.results:
            return {"error": "No profiling data available"}

        print(f"\n📊 Analyzing profiling results")

        bottlenecks = self.results.bottlenecks

        analysis = {
            "total_time_ms": self.results.total_time_ms,
            "gpu_utilization": round(
                (self.results.gpu_time_ms / self.results.total_time_ms) * 100, 1
            ),
            "cpu_time_ms": self.results.cpu_time_ms,
            "memory_peak_mb": self.results.memory_peak_mb,
            "bottlenecks": bottlenecks,
            "optimization_potential": "high" if len(bottlenecks) > 2 else "medium"
        }

        print(f"   Total time: {analysis['total_time_ms']:.2f}ms")
        print(f"   GPU utilization: {analysis['gpu_utilization']}%")
        print(f"   Memory peak: {analysis['memory_peak_mb']:.1f}MB")
        print(f"   Bottlenecks found: {len(bottlenecks)}")

        return analysis

    def recommend_optimizations(self) -> List[str]:
        """Generate optimization recommendations."""
        if not self.results:
            return []

        print(f"\n💡 Optimization Recommendations")

        recommendations = []

        # Check GPU utilization
        gpu_util = (self.results.gpu_time_ms / self.results.total_time_ms) * 100
        if gpu_util < 70:
            recommendations.append(
                f"Low GPU utilization ({gpu_util:.1f}%) - increase batch size"
            )

        # Check memory
        if self.results.memory_peak_mb > 30000:  # > 30GB
            recommendations.append(
                "High memory usage - consider gradient checkpointing"
            )

        # Check bottlenecks
        if "attention" in str(self.results.bottlenecks).lower():
            recommendations.append(
                "Attention bottleneck - use FlashAttention or PagedAttention"
            )

        if "data_loading" in str(self.results.bottlenecks).lower():
            recommendations.append(
                "Data loading bottleneck - increase num_workers"
            )

        # Generic recommendations
        recommendations.extend([
            "Enable mixed precision (FP16/BF16) for 2x speedup",
            "Use torch.compile() for 20-30% improvement",
            "Profile CUDA kernels with Nsight Compute"
        ])

        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec}")

        return recommendations


class ProfilingContext:
    """Context manager for profiling."""

    def __init__(self, profiler: Profiler):
        """Initialize context."""
        self.profiler = profiler
        self.start_time = None

    def __enter__(self):
        """Start profiling."""
        print(f"\n⏱️  Starting profiling...")
        self.start_time = time.time()
        self.profiler.profiling_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and collect results."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.profiler.profiling_active = False

        # Simulate profiling data
        import random

        gpu_time = elapsed_ms * random.uniform(0.6, 0.9)
        cpu_time = elapsed_ms - gpu_time

        bottlenecks = []
        if random.random() > 0.5:
            bottlenecks.append("Attention layer: 45% of time")
        if random.random() > 0.6:
            bottlenecks.append("Data loading: 15% overhead")
        if random.random() > 0.7:
            bottlenecks.append("Memory bandwidth limited")

        self.profiler.results = ProfilingResult(
            total_time_ms=elapsed_ms,
            gpu_time_ms=gpu_time,
            cpu_time_ms=cpu_time,
            memory_allocated_mb=random.uniform(8000, 12000),
            memory_peak_mb=random.uniform(15000, 25000),
            bottlenecks=bottlenecks
        )

        print(f"   ✓ Profiling complete ({elapsed_ms:.2f}ms)")


class KernelProfiler:
    """CUDA kernel profiler."""

    def __init__(self):
        """Initialize kernel profiler."""
        print(f"\n🔧 CUDA Kernel Profiler initialized")

    def profile_kernels(self, model_name: str) -> Dict[str, Any]:
        """Profile CUDA kernels."""
        print(f"\n⚙️  Profiling CUDA kernels: {model_name}")

        # Simulate kernel profiling
        kernels = [
            {"name": "attention_forward", "time_us": 450, "occupancy": 0.85},
            {"name": "matmul_kernel", "time_us": 320, "occupancy": 0.92},
            {"name": "layer_norm", "time_us": 180, "occupancy": 0.78},
            {"name": "activation", "time_us": 120, "occupancy": 0.88}
        ]

        total_time = sum(k["time_us"] for k in kernels)

        print(f"\n   Kernel Performance:")
        for kernel in kernels:
            pct = (kernel["time_us"] / total_time) * 100
            print(f"   {kernel['name']}: {kernel['time_us']}μs ({pct:.1f}%), occupancy: {kernel['occupancy']:.0%}")

        return {
            "kernels": kernels,
            "total_time_us": total_time,
            "avg_occupancy": sum(k["occupancy"] for k in kernels) / len(kernels)
        }


class MemoryProfiler:
    """GPU memory profiler."""

    def __init__(self):
        """Initialize memory profiler."""
        print(f"\n💾 Memory Profiler initialized")

    def profile_memory(self) -> Dict[str, Any]:
        """Profile memory usage."""
        print(f"\n📊 Profiling memory usage")

        # Simulate memory profiling
        memory_stats = {
            "allocated_mb": 18432,
            "reserved_mb": 20480,
            "peak_mb": 22528,
            "cached_mb": 2048,
            "fragmentation": 0.08
        }

        print(f"   Allocated: {memory_stats['allocated_mb']:.0f}MB")
        print(f"   Peak: {memory_stats['peak_mb']:.0f}MB")
        print(f"   Fragmentation: {memory_stats['fragmentation']:.1%}")

        if memory_stats['fragmentation'] > 0.1:
            print(f"   ⚠️  High fragmentation - consider memory cleanup")

        return memory_stats


def demo():
    """Demonstrate performance profiling."""
    print("=" * 60)
    print("Performance Profiling Suite Demo")
    print("=" * 60)

    # PyTorch profiling
    print(f"\n{'='*60}")
    print("PyTorch Model Profiling")
    print(f"{'='*60}")

    profiler = Profiler(backend="pytorch")

    # Simulate model inference
    with profiler.profile():
        time.sleep(0.2)  # Simulate inference

    # Analyze
    analysis = profiler.analyze()

    # Get recommendations
    print(f"\n{'='*60}")
    print("Optimization Recommendations")
    print(f"{'='*60}")

    recommendations = profiler.recommend_optimizations()

    # Kernel profiling
    print(f"\n{'='*60}")
    print("CUDA Kernel Profiling")
    print(f"{'='*60}")

    kernel_profiler = KernelProfiler()
    kernel_results = kernel_profiler.profile_kernels("transformer_model")

    # Memory profiling
    print(f"\n{'='*60}")
    print("Memory Profiling")
    print(f"{'='*60}")

    memory_profiler = MemoryProfiler()
    memory_stats = memory_profiler.profile_memory()

    # Summary
    print(f"\n{'='*60}")
    print("Profiling Summary")
    print(f"{'='*60}")

    summary = {
        "total_time_ms": analysis.get("total_time_ms", 0),
        "gpu_utilization": analysis.get("gpu_utilization", 0),
        "memory_peak_mb": analysis.get("memory_peak_mb", 0),
        "optimization_potential": analysis.get("optimization_potential", "unknown"),
        "top_bottleneck": analysis.get("bottlenecks", ["None"])[0] if analysis.get("bottlenecks") else "None"
    }

    print(f"\n   Total time: {summary['total_time_ms']:.2f}ms")
    print(f"   GPU util: {summary['gpu_utilization']}%")
    print(f"   Memory: {summary['memory_peak_mb']:.0f}MB")
    print(f"   Optimization potential: {summary['optimization_potential']}")


if __name__ == "__main__":
    demo()
