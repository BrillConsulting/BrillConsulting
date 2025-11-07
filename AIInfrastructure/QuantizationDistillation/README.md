# Quantization & Distillation Framework

Model compression techniques for efficient inference: INT8/INT4 quantization, GPTQ, AWQ, and knowledge distillation.

## Features

- **Post-Training Quantization (PTQ)** - INT8, INT4, FP16 quantization
- **Quantization-Aware Training (QAT)** - Fine-tuning with quantization
- **GPTQ** - Accurate post-training quantization for GPT models
- **AWQ** - Activation-aware weight quantization
- **Knowledge Distillation** - Transfer knowledge to smaller models
- **ONNX Optimization** - Convert and optimize for ONNX Runtime
- **Benchmark Tools** - Compare accuracy and performance
- **Model Size Reduction** - 2-8x compression with minimal accuracy loss

## Quantization Methods

| Method | Precision | Accuracy Loss | Speed Gain | Use Case |
|--------|-----------|---------------|------------|----------|
| **INT8** | 8-bit | <1% | 2-3x | General purpose |
| **INT4** | 4-bit | 1-3% | 4-6x | Memory-constrained |
| **GPTQ** | 2-4 bit | <2% | 3-4x | LLMs |
| **AWQ** | 4-bit | <1% | 3-4x | LLMs (high quality) |
| **FP16** | 16-bit | ~0% | 1.5-2x | Baseline |

## Usage

### INT8 Quantization

```python
from quantization_distillation import Quantizer

# Initialize quantizer
quantizer = Quantizer(method="int8")

# Load model
model = load_model("bert-base-uncased")

# Quantize
quantized_model = quantizer.quantize(
    model=model,
    calibration_data=calibration_dataset,
    symmetric=True
)

# Save quantized model
quantized_model.save("bert-base-int8")

# Benchmark
results = quantizer.benchmark(
    original_model=model,
    quantized_model=quantized_model,
    test_data=test_dataset
)
```

### GPTQ for LLMs

```python
from quantization_distillation import GPTQQuantizer

# Initialize GPTQ
gptq = GPTQQuantizer(
    bits=4,
    group_size=128,
    desc_act=True
)

# Quantize LLM
quantized_llm = gptq.quantize(
    model_name="meta-llama/Llama-2-7b-hf",
    calibration_dataset="c4",
    num_samples=128
)

# Inference
output = quantized_llm.generate(
    "Explain quantum computing",
    max_length=100
)

print(f"Model size reduced: {gptq.compression_ratio:.1f}x")
print(f"Perplexity: {gptq.evaluate_perplexity():.2f}")
```

### AWQ (Activation-Aware Quantization)

```python
from quantization_distillation import AWQQuantizer

# Initialize AWQ
awq = AWQQuantizer(
    bits=4,
    zero_point=True
)

# Quantize with activation awareness
quantized_model = awq.quantize(
    model="mistralai/Mistral-7B-v0.1",
    quant_config={
        "w_bit": 4,
        "q_group_size": 128,
    }
)

# Benchmark quality
metrics = awq.evaluate(
    quantized_model=quantized_model,
    benchmark="hellaswag"
)
```

### Knowledge Distillation

```python
from quantization_distillation import KnowledgeDistiller

# Initialize distiller
distiller = KnowledgeDistiller(
    teacher_model="bert-large-uncased",
    student_model="bert-base-uncased",
    temperature=3.0,
    alpha=0.7  # Balance between hard and soft targets
)

# Train student
distilled_model = distiller.distill(
    train_dataset=train_data,
    epochs=3,
    batch_size=32
)

# Compare performance
comparison = distiller.compare_models(test_dataset)
print(f"Size reduction: {comparison['size_reduction']}x")
print(f"Speed improvement: {comparison['speed_gain']}x")
print(f"Accuracy retention: {comparison['accuracy_retention']:.1%}")
```

## Performance Comparison

### LLaMA-2-7B Quantization Results

```
Method      | Size   | Memory | Tokens/sec | Perplexity | MMLU
------------|--------|--------|------------|------------|------
FP16        | 13.5GB | 15GB   | 25         | 5.47       | 45.3%
INT8        | 6.7GB  | 8GB    | 45         | 5.52       | 44.8%
GPTQ-4bit   | 3.5GB  | 5GB    | 60         | 5.68       | 43.1%
AWQ-4bit    | 3.5GB  | 5GB    | 58         | 5.61       | 44.2%
```

### BERT Distillation Results

```
Model          | Parameters | Size   | Latency | F1 Score
---------------|------------|--------|---------|----------
BERT-Large     | 340M       | 1.3GB  | 45ms    | 91.2%
BERT-Base      | 110M       | 440MB  | 15ms    | 88.5%
DistilBERT     | 66M        | 260MB  | 9ms     | 86.9%
TinyBERT       | 14M        | 60MB   | 3ms     | 82.8%
```

## Advanced Features

### Mixed Precision

```python
# Combine different precisions
mixed_config = {
    "attention": "fp16",
    "feedforward": "int8",
    "embeddings": "int8"
}

quantizer.quantize_mixed_precision(model, mixed_config)
```

### Sensitivity Analysis

```python
# Find sensitive layers
sensitivity = quantizer.analyze_sensitivity(model, test_data)
print(sensitivity.most_sensitive_layers)

# Protect sensitive layers
quantizer.quantize_selective(
    model=model,
    skip_layers=sensitivity.most_sensitive_layers[:5]
)
```

## Demo

```bash
# Quantize model
python quantization_distillation.py \
  --model bert-base-uncased \
  --method gptq \
  --bits 4

# Benchmark
python benchmark_quantized.py \
  --original bert-base \
  --quantized bert-base-gptq-4bit
```

## Technologies

- PyTorch 2.0+
- Transformers 4.35+
- GPTQ-for-LLaMa
- AutoGPTQ
- AutoAWQ
- ONNX Runtime 1.16+
- TensorRT 8.6+
- bitsandbytes
