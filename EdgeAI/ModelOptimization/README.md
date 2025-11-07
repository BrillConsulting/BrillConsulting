# Model Optimization

Advanced model optimization techniques for edge deployment: quantization, pruning, knowledge distillation, and neural architecture search.

## Features

- **Post-Training Quantization** - INT8/INT4/FP16 with calibration
- **Quantization-Aware Training** - Train with quantization in mind
- **Structured/Unstructured Pruning** - Remove unnecessary weights
- **Knowledge Distillation** - Transfer knowledge from large to small models
- **Neural Architecture Search** - Find optimal architectures for edge
- **Layer Fusion** - Combine layers for faster inference
- **Operator Optimization** - Replace expensive ops with efficient ones
- **Model Profiling** - Identify bottlenecks

## Quantization Methods

| Method | Accuracy Loss | Speedup | Size Reduction |
|--------|---------------|---------|----------------|
| **FP32 → FP16** | <0.1% | 1.5-2x | 2x |
| **FP32 → INT8** | 0.5-2% | 2-4x | 4x |
| **FP32 → INT4** | 2-5% | 3-6x | 8x |
| **Dynamic Quantization** | <1% | 2-3x | 4x |
| **Quantization-Aware Training** | <0.5% | 2-4x | 4x |

## Usage

### Post-Training Quantization
```python
from model_optimization import Quantizer

quantizer = Quantizer(method="int8")

# Calibrate with representative data
quantized_model = quantizer.quantize(
    model=original_model,
    calibration_data=calibration_dataset,
    algorithm="minmax"  # or "kl", "percentile"
)

# Evaluate accuracy
original_acc = evaluate(original_model, test_data)
quantized_acc = evaluate(quantized_model, test_data)

print(f"Accuracy: {original_acc:.2%} → {quantized_acc:.2%}")
print(f"Size: {get_size(original_model):.1f}MB → {get_size(quantized_model):.1f}MB")
```

### Quantization-Aware Training
```python
from model_optimization import QATTrainer

trainer = QATTrainer(
    model=model,
    target_precision="int8"
)

# Train with fake quantization
qat_model = trainer.train(
    train_data=train_dataset,
    epochs=10,
    learning_rate=0.0001
)

# Convert to actual INT8
int8_model = trainer.convert_to_int8(qat_model)
```

### Structured Pruning
```python
from model_optimization import Pruner

pruner = Pruner(method="structured")

# Prune channels/filters
pruned_model = pruner.prune(
    model=model,
    pruning_ratio=0.5,  # Remove 50% of channels
    criterion="l1_norm"  # or "l2_norm", "importance"
)

# Fine-tune pruned model
pruned_model = pruner.fine_tune(
    pruned_model,
    train_data=train_dataset,
    epochs=5
)

print(f"Parameters: {count_params(model)} → {count_params(pruned_model)}")
print(f"FLOPs reduction: {calculate_flops_reduction(model, pruned_model):.1%}")
```

### Knowledge Distillation
```python
from model_optimization import KnowledgeDistiller

distiller = KnowledgeDistiller(
    teacher_model=large_model,
    student_model=small_model,
    temperature=3.0,
    alpha=0.7  # Weight for distillation loss
)

# Train student to mimic teacher
distilled_model = distiller.train(
    train_data=train_dataset,
    epochs=20
)

# Compare models
teacher_acc = evaluate(large_model, test_data)  # 95%
student_acc = evaluate(small_model, test_data)  # 88%
distilled_acc = evaluate(distilled_model, test_data)  # 93%
```

### Neural Architecture Search
```python
from model_optimization import NASOptimizer

nas = NASOptimizer(
    search_space="mobilenet_v3",
    constraints={
        "latency_ms": 20,
        "model_size_mb": 5,
        "min_accuracy": 0.90
    }
)

# Search for optimal architecture
best_arch = nas.search(
    train_data=train_dataset,
    validation_data=val_dataset,
    search_iterations=100
)

# Build and train discovered architecture
optimized_model = nas.build_model(best_arch)
nas.train(optimized_model, train_dataset)
```

## Pruning Strategies

### Magnitude-based Pruning
Prune weights with smallest magnitudes:
```python
pruner = Pruner(method="magnitude")
pruned = pruner.prune(model, pruning_ratio=0.3)
```

### Importance-based Pruning
Prune based on gradient information:
```python
pruner = Pruner(method="importance")
pruned = pruner.prune(
    model,
    pruning_ratio=0.4,
    train_data=importance_dataset
)
```

### Iterative Pruning
Gradual pruning with fine-tuning:
```python
pruner = Pruner(method="iterative")

for i in range(5):
    model = pruner.prune_step(model, ratio=0.1)
    model = pruner.fine_tune(model, epochs=2)
```

## Operator Fusion

Combine multiple operations for efficiency:

```python
from model_optimization import OperatorFuser

fuser = OperatorFuser()

# Fuse Conv + BatchNorm + ReLU
fused_model = fuser.fuse_operations(
    model=model,
    patterns=[
        ["Conv2D", "BatchNorm", "ReLU"],
        ["Linear", "ReLU"],
        ["Conv2D", "ReLU6"]
    ]
)

# Speedup from fusion
print(f"Inference time: {benchmark(model):.1f}ms → {benchmark(fused_model):.1f}ms")
```

## Model Profiling

Identify performance bottlenecks:

```python
from model_optimization import Profiler

profiler = Profiler(model)

# Profile inference
report = profiler.profile(
    input_data=sample_input,
    device="cuda"
)

# Analyze results
print("Layer-wise breakdown:")
for layer_name, metrics in report.items():
    print(f"{layer_name}:")
    print(f"  Time: {metrics['time_ms']:.2f}ms ({metrics['percentage']:.1f}%)")
    print(f"  Memory: {metrics['memory_mb']:.2f}MB")
    print(f"  FLOPs: {metrics['flops']:,}")

# Identify bottlenecks
bottlenecks = profiler.find_bottlenecks(threshold=0.05)  # >5% of time
```

## Compression Techniques

### Mixed Precision
Different precisions for different layers:
```python
from model_optimization import MixedPrecisionOptimizer

optimizer = MixedPrecisionOptimizer()

# Automatically assign precisions
mixed_model = optimizer.optimize(
    model=model,
    sensitive_layers=["conv1", "fc_final"],  # Keep FP32
    accuracy_threshold=0.99
)
```

### Low-Rank Factorization
Decompose weight matrices:
```python
from model_optimization import LowRankFactorizer

factorizer = LowRankFactorizer()

factorized_model = factorizer.factorize(
    model=model,
    rank_ratio=0.5,  # 50% of original rank
    layers=["fc1", "fc2"]
)
```

## Benchmarks

### YOLOv8n Optimization
| Technique | Accuracy | Size | Latency (RPi4) | Speedup |
|-----------|----------|------|----------------|---------|
| Original FP32 | 95.0% | 6.2MB | 130ms | 1x |
| FP16 | 94.9% | 3.1MB | 85ms | 1.5x |
| INT8 PTQ | 94.2% | 1.6MB | 45ms | 2.9x |
| INT8 QAT | 94.7% | 1.6MB | 45ms | 2.9x |
| Pruned 50% + INT8 | 93.5% | 0.8MB | 30ms | 4.3x |

### MobileNetV2 Optimization
| Technique | Accuracy | Parameters | FLOPs | Latency |
|-----------|----------|------------|-------|---------|
| Original | 92.0% | 3.5M | 300M | 25ms |
| Pruned 30% | 91.2% | 2.4M | 180M | 15ms |
| Distilled | 91.5% | 1.2M | 150M | 12ms |
| NAS-optimized | 91.8% | 1.5M | 120M | 10ms |

## Technologies

- **Frameworks**: PyTorch, TensorFlow, ONNX
- **Quantization**: TensorRT, ONNX Runtime, TFLite
- **Pruning**: Torch-Pruning, Neural Network Intelligence
- **NAS**: NAS-Bench, AutoML
- **Profiling**: PyTorch Profiler, TensorBoard

## Best Practices

✅ Always calibrate quantization with representative data
✅ Use QAT for critical applications (better accuracy)
✅ Start with structured pruning for easier deployment
✅ Combine multiple techniques (prune + quantize)
✅ Profile before optimizing to find bottlenecks
✅ Test on actual target hardware
✅ Monitor accuracy degradation closely
✅ Use iterative pruning for better results

## References

- TensorRT: https://developer.nvidia.com/tensorrt
- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- ONNX Quantization: https://onnxruntime.ai/docs/performance/quantization.html
- Neural Network Pruning: https://arxiv.org/abs/1506.02626
- Knowledge Distillation: https://arxiv.org/abs/1503.02531
