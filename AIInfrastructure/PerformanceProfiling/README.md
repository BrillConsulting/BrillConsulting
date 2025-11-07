# Performance Profiling Suite

Deep performance analysis with NVIDIA Nsight, PyTorch Profiler, and bottleneck detection.

## Features

- **GPU Profiling** - NVIDIA Nsight Systems/Compute
- **PyTorch Profiler** - Layer-by-layer analysis
- **Bottleneck Detection** - Identify performance issues
- **Kernel Analysis** - CUDA kernel optimization
- **Memory Profiling** - Memory allocation tracking
- **Timeline Visualization** - Chrome trace format
- **Autotuning** - Automatic optimization
- **Comparative Analysis** - Before/after comparisons

## Profiling Tools

| Tool | Use Case | Granularity |
|------|----------|-------------|
| **Nsight Systems** | System-wide profiling | Coarse |
| **Nsight Compute** | Kernel analysis | Fine |
| **PyTorch Profiler** | Model profiling | Medium |
| **TensorBoard** | Training visualization | Medium |

## Usage

```python
from performance_profiling import Profiler

# Profile inference
profiler = Profiler(backend="nsight")

with profiler.profile():
    output = model.generate(prompt, max_tokens=100)

# Analyze results
report = profiler.analyze()
print(report.bottlenecks)
# Output: ["Attention layer: 45% of time",
#          "GPU memory bandwidth limited"]

# Get recommendations
optimizations = profiler.recommend_optimizations()
```

## Technologies

- NVIDIA Nsight Systems
- NVIDIA Nsight Compute
- PyTorch Profiler
- TensorBoard Profiler
- Custom profiling tools
