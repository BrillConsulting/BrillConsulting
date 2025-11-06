# LLM Fine-Tuning Toolkit - Production Ready

A comprehensive, production-ready fine-tuning system for large language models supporting multiple parameter-efficient fine-tuning methods, distributed training, and advanced optimization techniques.

## Overview

This toolkit provides enterprise-grade fine-tuning capabilities for LLMs with support for:
- **Parameter-Efficient Fine-Tuning (PEFT)**: LoRA, QLoRA, Prefix Tuning, P-Tuning, IA3
- **Full Fine-Tuning**: Complete model parameter optimization
- **Distributed Training**: Multi-GPU support with DDP and FSDP
- **Advanced Features**: Mixed precision, gradient checkpointing, automatic checkpoint management
- **Production Ready**: Comprehensive logging, metrics tracking, and W&B integration

## Features

### Fine-Tuning Methods

#### LoRA (Low-Rank Adaptation)
Efficient fine-tuning by adding trainable low-rank matrices to model layers, reducing trainable parameters by up to 99%.

```python
config = FineTuningConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    method="lora",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
```

#### QLoRA (Quantized LoRA)
Combines LoRA with 4-bit quantization for memory-efficient fine-tuning of large models on consumer GPUs.

```python
config = FineTuningConfig(
    model_name="meta-llama/Llama-2-70b-hf",
    method="qlora",
    use_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4"
)
```

#### Other PEFT Methods
- **Prefix Tuning**: Optimizes continuous task-specific vectors
- **P-Tuning**: Learnable prompt embeddings with encoder
- **IA3**: Infused Adapter by Inhibiting and Amplifying Internal Activations

### Training Features

#### Distributed Training
Multi-GPU training with automatic parallelization:

```python
# Data Distributed Parallel (DDP)
config = FineTuningConfig(
    use_ddp=True,
    world_size=4,
    local_rank=0
)

# Launch with torchrun:
# torchrun --nproc_per_node=4 fine_tuning.py
```

#### Mixed Precision Training
FP16 and BF16 support for faster training and reduced memory:

```python
config = FineTuningConfig(
    fp16=True,  # For NVIDIA GPUs
    bf16=False  # For newer architectures (A100, H100)
)
```

#### Gradient Checkpointing
Trade computation for memory to train larger models:

```python
config = FineTuningConfig(
    gradient_checkpointing=True
)
```

#### Advanced Optimizers
- **AdamW**: Standard PyTorch AdamW
- **AdamW 8-bit**: Memory-efficient 8-bit optimizer
- **SGD**: Stochastic Gradient Descent with momentum

#### Learning Rate Schedulers
- **Linear**: Linear warmup and decay
- **Cosine**: Cosine annealing with warmup
- **Polynomial**: Polynomial decay

### Checkpoint Management

Automatic checkpoint saving with configurable retention:

```python
config = FineTuningConfig(
    checkpoint_dir="./checkpoints",
    save_steps=500,
    save_total_limit=3,  # Keep only best 3 checkpoints
    resume_from_checkpoint="./checkpoints/checkpoint-epoch1-step1000"
)
```

Features:
- Automatic cleanup of old checkpoints
- Validation-based checkpoint selection
- Full training state persistence (optimizer, scheduler, epoch)
- Easy resume from any checkpoint

### Metrics and Monitoring

#### Built-in Metrics
- Training/validation loss
- Perplexity
- Learning rate tracking
- Gradient norms

#### Weights & Biases Integration

```python
config = FineTuningConfig(
    use_wandb=True,
    wandb_project="my-finetuning-project",
    wandb_run_name="llama-lora-experiment-1"
)
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- PEFT >= 0.7.0
- BitsAndBytes >= 0.41.0 (for QLoRA)
- W&B (optional, for experiment tracking)
- Accelerate (for distributed training)

## Quick Start

### 1. Prepare Your Data

Data should be in JSONL format with one of these schemas:

**Alpaca Format** (instruction-based):
```json
{
  "instruction": "Classify the sentiment",
  "input": "I love this product!",
  "output": "Positive"
}
```

**OpenAI Format** (chat-based):
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
  ]
}
```

**Utility Functions**:
```python
from fine_tuning import prepare_alpaca_format, save_dataset, split_dataset

# Create training data
data = []
for item in your_data:
    data.append(prepare_alpaca_format(
        instruction=item['instruction'],
        input_text=item['input'],
        output=item['output']
    ))

# Split dataset
train_data, val_data, test_data = split_dataset(data, train_ratio=0.8, val_ratio=0.1)

# Save to files
save_dataset(train_data, "train.jsonl")
save_dataset(val_data, "val.jsonl")
```

### 2. Configure Fine-Tuning

```python
from fine_tuning import FineTuningConfig

config = FineTuningConfig(
    # Model
    model_name="meta-llama/Llama-2-7b-hf",
    max_seq_length=2048,

    # Method
    method="lora",
    lora_r=8,
    lora_alpha=16,

    # Training
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,

    # Optimization
    fp16=True,
    gradient_checkpointing=True,

    # Checkpointing
    checkpoint_dir="./checkpoints",
    save_steps=100,

    # Evaluation
    eval_steps=50,
    eval_strategy="steps"
)

# Save configuration
config.save("config.json")
```

### 3. Train the Model

```python
from fine_tuning import FineTuner

# Initialize
tuner = FineTuner(config)

# Load model
tuner.load_model()

# Load data
train_dataset, val_dataset = tuner.load_data("train.jsonl", "val.jsonl")

# Train
metrics = tuner.train(train_dataset, val_dataset)

# Save final model
tuner.save_model("./output/final_model")

# Cleanup
tuner.cleanup()
```

### 4. Inference

```python
# Load trained model
tuner = FineTuner(config)
tuner.load_trained_model("./output/final_model")

# Generate
prompt = "### Instruction:\nClassify the sentiment\n\n### Input:\nThis is amazing!\n\n### Response:\n"
generated = tuner.generate(
    prompt,
    max_length=128,
    temperature=0.7,
    top_p=0.9
)
print(generated[0])
```

## Advanced Usage

### QLoRA for Large Models

Fine-tune 70B+ models on consumer hardware:

```python
config = FineTuningConfig(
    model_name="meta-llama/Llama-2-70b-hf",
    method="qlora",

    # Quantization
    use_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    use_nested_quant=True,

    # LoRA
    lora_r=64,
    lora_alpha=16,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],

    # Training
    batch_size=1,
    gradient_accumulation_steps=16,
    fp16=True,
    gradient_checkpointing=True,

    # Optimizer
    optim="adamw_8bit"
)
```

### Multi-GPU Training

Using DistributedDataParallel:

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 train_script.py
```

```python
import os

config = FineTuningConfig(
    use_ddp=True,
    world_size=int(os.environ.get("WORLD_SIZE", 1)),
    local_rank=int(os.environ.get("LOCAL_RANK", -1))
)
```

### Custom Training Loop

For advanced customization:

```python
# Initialize components
tuner = FineTuner(config)
tuner.load_model()
train_dataset, val_dataset = tuner.load_data("train.jsonl", "val.jsonl")

# Create optimizer and scheduler
num_steps = len(train_dataset) * config.num_epochs // config.batch_size
optimizer, scheduler = tuner.create_optimizer_and_scheduler(num_steps)

# Custom training loop
for epoch in range(config.num_epochs):
    # Your custom training logic here
    pass
```

### Resuming from Checkpoint

```python
config = FineTuningConfig(
    resume_from_checkpoint="./checkpoints/checkpoint-epoch2-step1000"
)

tuner = FineTuner(config)
tuner.load_model()
train_dataset, val_dataset = tuner.load_data("train.jsonl", "val.jsonl")

# Training will resume from checkpoint
tuner.train(train_dataset, val_dataset)
```

## Configuration Reference

### Model Configuration
- `model_name`: HuggingFace model ID or local path
- `tokenizer_name`: Tokenizer to use (defaults to model_name)
- `max_seq_length`: Maximum sequence length (default: 2048)

### Training Method
- `method`: `"lora"`, `"qlora"`, `"full"`, `"prefix_tuning"`, `"p_tuning"`, `"ia3"`

### LoRA Parameters
- `lora_r`: Rank of LoRA matrices (default: 8)
- `lora_alpha`: LoRA scaling factor (default: 16)
- `lora_dropout`: Dropout for LoRA layers (default: 0.05)
- `lora_target_modules`: List of modules to apply LoRA
- `lora_bias`: Bias handling - `"none"`, `"all"`, `"lora_only"`

### Quantization (QLoRA)
- `use_4bit`: Enable 4-bit quantization
- `use_8bit`: Enable 8-bit quantization
- `bnb_4bit_compute_dtype`: Compute dtype - `"float16"`, `"bfloat16"`, `"float32"`
- `bnb_4bit_quant_type`: Quantization type - `"nf4"`, `"fp4"`
- `use_nested_quant`: Enable nested quantization

### Training Hyperparameters
- `num_epochs`: Number of training epochs (default: 3)
- `batch_size`: Batch size per device (default: 4)
- `gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `learning_rate`: Learning rate (default: 2e-4)
- `weight_decay`: Weight decay (default: 0.01)
- `warmup_steps`: Warmup steps (default: 100)
- `max_grad_norm`: Gradient clipping (default: 1.0)
- `scheduler_type`: `"linear"`, `"cosine"`, `"polynomial"`

### Precision
- `fp16`: Enable FP16 mixed precision
- `bf16`: Enable BF16 mixed precision

### Distributed Training
- `use_ddp`: Enable DistributedDataParallel
- `use_fsdp`: Enable FullyShardedDataParallel
- `world_size`: Total number of processes
- `local_rank`: Local process rank

### Checkpointing
- `checkpoint_dir`: Directory for checkpoints
- `save_steps`: Save checkpoint every N steps
- `save_total_limit`: Maximum checkpoints to keep
- `resume_from_checkpoint`: Path to resume from

### Evaluation
- `eval_steps`: Evaluate every N steps
- `eval_strategy`: `"steps"`, `"epoch"`, `"no"`

### Logging
- `logging_steps`: Log metrics every N steps
- `use_wandb`: Enable Weights & Biases
- `wandb_project`: W&B project name
- `wandb_run_name`: W&B run name

### Advanced
- `gradient_checkpointing`: Enable gradient checkpointing
- `optim`: Optimizer - `"adamw_torch"`, `"adamw_8bit"`, `"sgd"`
- `seed`: Random seed

## Performance Tips

### Memory Optimization
1. Use QLoRA for large models (70B+)
2. Enable gradient checkpointing
3. Reduce batch size, increase gradient accumulation
4. Use 8-bit optimizer (`optim="adamw_8bit"`)
5. Lower `max_seq_length`

### Speed Optimization
1. Use FP16/BF16 mixed precision
2. Increase batch size (if memory allows)
3. Use multiple GPUs with DDP
4. Disable gradient checkpointing (if memory sufficient)
5. Use compiled model (`torch.compile`)

### Quality Optimization
1. Use larger LoRA rank (r=64-128) for complex tasks
2. Increase training epochs
3. Use validation set for hyperparameter tuning
4. Adjust learning rate (try 1e-4 to 5e-4)
5. Use cosine scheduler with warmup

## Examples

### Example 1: Sentiment Classification

```python
from fine_tuning import FineTuningConfig, FineTuner, prepare_alpaca_format, save_dataset

# Prepare data
train_data = [
    prepare_alpaca_format(
        instruction="Classify the sentiment",
        input_text="I absolutely love this!",
        output="Positive"
    ),
    prepare_alpaca_format(
        instruction="Classify the sentiment",
        input_text="This is terrible.",
        output="Negative"
    )
    # ... more examples
]
save_dataset(train_data, "sentiment_train.jsonl")

# Configure
config = FineTuningConfig(
    model_name="gpt2",
    method="lora",
    num_epochs=3,
    batch_size=8,
    learning_rate=3e-4
)

# Train
tuner = FineTuner(config)
tuner.load_model()
train_dataset, _ = tuner.load_data("sentiment_train.jsonl")
tuner.train(train_dataset)
tuner.save_model("./sentiment_model")
```

### Example 2: Instruction Following (LLaMA)

```python
config = FineTuningConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    method="qlora",
    use_4bit=True,
    lora_r=64,
    lora_alpha=16,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    gradient_checkpointing=True,
    use_wandb=True,
    wandb_project="llama-instruction-tuning"
)

tuner = FineTuner(config)
tuner.load_model()
train_dataset, val_dataset = tuner.load_data("alpaca_train.jsonl", "alpaca_val.jsonl")
tuner.train(train_dataset, val_dataset)
tuner.save_model("./llama-instruct")
```

### Example 3: Multi-GPU Training

```python
# train_multi_gpu.py
import os
from fine_tuning import FineTuningConfig, FineTuner

config = FineTuningConfig(
    model_name="meta-llama/Llama-2-13b-hf",
    method="lora",
    use_ddp=True,
    world_size=int(os.environ.get("WORLD_SIZE", 1)),
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    batch_size=2,
    gradient_accumulation_steps=8,
    fp16=True
)

tuner = FineTuner(config)
tuner.load_model()
train_dataset, val_dataset = tuner.load_data("train.jsonl", "val.jsonl")
tuner.train(train_dataset, val_dataset)

if config.local_rank <= 0:
    tuner.save_model("./output")

tuner.cleanup()
```

Run with:
```bash
torchrun --nproc_per_node=4 train_multi_gpu.py
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Use QLoRA with 4-bit quantization
- Reduce `max_seq_length`

### Training Too Slow
- Enable `fp16` or `bf16`
- Increase `batch_size`
- Disable `gradient_checkpointing`
- Use multiple GPUs

### Poor Model Quality
- Increase `num_epochs`
- Increase `lora_r` (try 32-64)
- Adjust `learning_rate`
- Use more training data
- Check data quality and formatting

### Checkpoint Issues
- Ensure sufficient disk space
- Check write permissions
- Verify checkpoint directory exists
- Resume from specific checkpoint if corrupted

## Best Practices

1. **Start Small**: Test with small model (GPT-2) before scaling up
2. **Data Quality**: Clean and validate data before training
3. **Hyperparameter Search**: Use W&B sweeps for optimal hyperparameters
4. **Monitor Training**: Watch loss curves for overfitting
5. **Validate Regularly**: Use validation set to track generalization
6. **Save Configs**: Always save training configurations
7. **Version Control**: Track experiments with W&B or MLflow
8. **Test Inference**: Verify model quality before deploying

## Architecture

The system consists of several key components:

- **FineTuner**: Main orchestrator for training pipeline
- **FineTuningConfig**: Comprehensive configuration dataclass
- **FineTuningDataset**: PyTorch Dataset with format handling
- **CheckpointManager**: Automatic checkpoint saving and cleanup
- **MetricsTracker**: Training metrics logging and W&B integration
- **DistributedTrainer**: Multi-GPU training setup

## License

Copyright (c) 2024 Brill Consulting. All rights reserved.

## Support

For issues, questions, or contributions, please contact Brill Consulting.

## Changelog

### Version 2.0.0 (Production)
- Added LoRA, QLoRA, Prefix Tuning, P-Tuning, IA3
- Distributed training support (DDP, FSDP)
- Advanced checkpoint management
- Mixed precision training (FP16, BF16)
- W&B integration
- Multiple optimizer and scheduler options
- Comprehensive configuration system
- Production-ready error handling and logging

### Version 1.0.0 (Initial)
- Basic fine-tuning support
- Simple data preparation
- OpenAI and LLaMA format support
