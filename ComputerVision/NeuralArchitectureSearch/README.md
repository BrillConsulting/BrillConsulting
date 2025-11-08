# Neural Architecture Search (NAS)

Advanced automatic discovery of optimal neural network architectures using differentiable and evolutionary methods.

## Features

- **DARTS (Differentiable Architecture Search)**: Gradient-based architecture search
- **Evolutionary NAS**: Population-based architecture optimization
- **Super Network**: Contains all possible architectures in search space
- **Automated hyperparameter tuning**: Discovers optimal layer configurations
- **Multi-cell architecture**: Modular design for flexible networks

## Methods Implemented

### 1. DARTS
- Continuous relaxation of architecture search space
- Gradient descent on architecture parameters
- Efficient search using weight sharing

### 2. Evolutionary Search
- Tournament selection
- Crossover and mutation operators
- Elitism for preserving best architectures

## Usage

```python
from nas_search import NASTrainer, EvolutionaryNAS, SearchSpace

# DARTS method
trainer = NASTrainer(device='cuda')
model = trainer.build_model(num_classes=10)
model = trainer.train_darts(model, train_loader, val_loader, num_epochs=50)
architecture = trainer.extract_architecture(model)

# Evolutionary method
search_space = SearchSpace()
evo_nas = EvolutionaryNAS(search_space, population_size=50)
best_arch = evo_nas.evolve(num_generations=100, val_loader=val_loader)
```

## Architecture Components

- Convolutional operations (3x3, 5x5)
- Depthwise separable convolutions
- Dilated convolutions
- Pooling operations (max, average)
- Skip connections

## Installation

```bash
pip install -r requirements.txt
```

## Example

```bash
python nas_search.py
```
