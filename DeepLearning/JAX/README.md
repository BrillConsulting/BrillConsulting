# JAX/Flax Deep Learning Models

High-performance numerical computing and neural networks with JAX and Flax.

## Features

- **MLP Models**: Multi-layer perceptrons with Flax
- **CNN Models**: Convolutional neural networks
- **Transformer Models**: Attention-based architectures
- **Automatic Differentiation**: grad, value_and_grad
- **JIT Compilation**: Lightning-fast execution
- **Vectorization**: vmap for batch operations
- **Parallelization**: pmap for multi-GPU/TPU
- **Functional Programming**: Pure functions, no side effects
- **XLA Compilation**: Optimized kernels

## Technologies

- JAX 0.4.20+
- Flax
- Optax (optimizers)
- XLA

## Usage

```python
from jax_models import JAXModels

# Initialize
jax_models = JAXModels()

# MLP model
mlp = jax_models.create_mlp_model({
    'name': 'MLP',
    'features': [256, 128, 64, 10]
})

# CNN model
cnn = jax_models.create_cnn_model({
    'name': 'CNN',
    'num_classes': 10
})

# Transformer
transformer = jax_models.create_transformer_model({
    'num_heads': 8,
    'num_layers': 6,
    'd_model': 512
})

# Training utilities
utilities = jax_models.create_training_utilities()
```

## Demo

```bash
python jax_models.py
```

## Advantages

- **Speed**: JIT compilation + XLA optimization
- **Parallelization**: Easy multi-GPU/TPU with pmap
- **Functional**: Pure functions, easier to reason about
- **Composability**: grad, jit, vmap, pmap compose naturally
- **Research**: Used by DeepMind, Google Research
