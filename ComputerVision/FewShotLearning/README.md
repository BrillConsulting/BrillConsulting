# Few-Shot Learning

Meta-learning and metric learning approaches for learning from limited examples.

## Features

- **Prototypical Networks**: Distance-based classification using class prototypes
- **Relation Networks**: Learned comparison between query and support
- **MAML (Model-Agnostic Meta-Learning)**: Learn good initialization for fast adaptation
- **Matching Networks**: Attention-based one-shot learning

## Methods Implemented

### 1. Prototypical Networks
- Learns metric space where classification is based on euclidean distance
- Computes class prototypes as mean of support examples
- Efficient and effective for few-shot scenarios

### 2. Relation Networks
- Learns to compare query and support using CNN-based relation module
- More flexible than fixed distance metrics
- Better for complex relationships

### 3. MAML
- Meta-learning approach that learns good parameter initialization
- Fast adaptation to new tasks with gradient descent
- Task-agnostic framework

### 4. Matching Networks
- Uses attention and episodic training
- Full context embedding with LSTM
- One-shot learning capability

## Usage

```python
from few_shot_learning import PrototypicalNetwork, FewShotTrainer, FewShotDataset

# Setup
n_way = 5  # 5-way classification
k_shot = 5  # 5-shot learning

# Create model
model = PrototypicalNetwork(input_channels=3, hidden_size=64)

# Train
trainer = FewShotTrainer(model, device='cuda')
trainer.train_prototypical(train_loader, val_loader, n_way, k_shot, epochs=100)
```

## Task Configuration

- **N-way**: Number of classes in each episode
- **K-shot**: Number of labeled examples per class
- **Q-query**: Number of query examples to classify

Common configurations:
- 5-way 1-shot
- 5-way 5-shot
- 20-way 1-shot

## Installation

```bash
pip install -r requirements.txt
```

## Example

```bash
python few_shot_learning.py
```
