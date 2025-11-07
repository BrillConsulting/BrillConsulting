# 🎯 Multi-Task Learning

Joint learning across related tasks to improve generalization through shared representations.

## 🌟 Approaches

### Hard Parameter Sharing
1. **Shared Bottom** - Common layers, task-specific heads
2. **Cross-Stitch Networks** - Learnable linear combinations
3. **Sluice Networks** - Selective sharing at multiple levels

### Soft Parameter Sharing
4. **Low-Rank Adaptation** - Shared + task-specific parameters
5. **Tensor Factorization** - Decompose task parameters
6. **Group Lasso** - Regularization-based sharing

### Architecture-Based
7. **Multi-Gate Mixture of Experts (MMoE)** - Task-specific expert routing
8. **Progressive Neural Networks** - Sequential task columns
9. **Routing Networks** - Dynamic path selection

### Attention-Based
10. **Multi-Task Attention** - Task-aware attention
11. **MTAN** - Multi-Task Attention Network

## ✨ Key Features

- **Shared Representations** - Common features across tasks
- **Task-Specific Layers** - Specialized outputs per task
- **Transfer Learning** - Knowledge sharing between tasks
- **Regularization Effect** - Improves generalization
- **Parameter Efficiency** - Fewer parameters than separate models
- **Multi-Objective Optimization** - Balance multiple losses

## 🚀 Quick Start

### Hard Parameter Sharing

```bash
python multi_task_learning.py --architecture shared-bottom --tasks 3 --shared-layers 2
```

### MMoE (Mixture of Experts)

```bash
python multi_task_learning.py --architecture mmoe --tasks 3 --experts 4
```

## 📊 Example Code

```python
from multi_task_learning import MultiTaskModel
import torch

# Define tasks
tasks = ['task1', 'task2', 'task3']

# Initialize multi-task model
mt_model = MultiTaskModel(
    tasks=tasks,
    architecture='shared-bottom',
    shared_layers=[128, 64],
    task_layers=[32, 16],
    loss_weights={'task1': 1.0, 'task2': 1.0, 'task3': 0.5}
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        x, labels = batch  # labels = {'task1': y1, 'task2': y2, 'task3': y3}

        # Forward pass
        predictions = mt_model(x)  # Returns dict of predictions

        # Compute multi-task loss
        loss = mt_model.compute_loss(predictions, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Predict on new data
predictions = mt_model.predict(x_new)
print(f"Task 1: {predictions['task1']}")
print(f"Task 2: {predictions['task2']}")
```

## 🎯 Use Cases

### 🌐 Natural Language Processing
- **Multi-Domain Sentiment**: Product reviews + movie reviews + tweets
- **NER + POS Tagging**: Named entities + part-of-speech
- **Translation + Summarization**: Related text generation tasks
- **Question Answering + Reading Comprehension**: Combined understanding

### 👁️ Computer Vision
- **Object Detection + Segmentation**: Bounding boxes + pixel masks
- **Depth + Surface Normals**: Geometric scene understanding
- **Face Recognition + Attributes**: Identity + age/gender/emotion
- **Multi-Label Classification**: Multiple object categories

### 🏥 Healthcare
- **Disease Prediction**: Multiple conditions simultaneously
- **Risk Assessment**: Different risk factors
- **Patient Outcomes**: Mortality + readmission + complications
- **Medical Imaging**: Diagnosis + localization + severity

### 🏢 Business
- **Customer Analytics**: Churn + LTV + satisfaction
- **Recommendation**: Rating + click + purchase prediction
- **E-commerce**: Product recommendation + price prediction
- **Ad Systems**: CTR + CVR + revenue prediction

### 🚗 Autonomous Driving
- **Perception**: Object detection + lane detection + traffic signs
- **Planning**: Path planning + behavior prediction
- **Control**: Steering + acceleration + braking

## 📚 Architecture Details

### Shared Bottom (Hard Sharing)
**Type**: Simple hard parameter sharing
**Best for**: Highly related tasks, limited data
**Pros**: Parameter efficient, strong regularization
**Cons**: Negative transfer if tasks too different

```python
model = MultiTaskModel(
    architecture='shared-bottom',
    shared_layers=[256, 128, 64],    # Shared encoder
    task_layers={'task1': [32, 16],   # Task-specific heads
                 'task2': [32, 16]}
)
```

**Architecture**:
```
Input → Shared Layers → Task1 Head → Output1
                      → Task2 Head → Output2
```

### MMoE (Multi-gate Mixture of Experts)
**Type**: Soft parameter sharing with gating
**Best for**: Tasks with different importance, complex relationships
**Pros**: Handles task conflicts, flexible
**Cons**: More parameters, requires tuning

```python
model = MultiTaskModel(
    architecture='mmoe',
    num_experts=4,              # Number of expert networks
    expert_layers=[128, 64],    # Expert architecture
    gate_layers=[32]            # Gating network
)
```

**Key idea**: Each task has its own gate to select expert combination

### Cross-Stitch Networks
**Type**: Learnable linear combinations
**Best for**: Fine-grained sharing control
**Pros**: Flexible sharing patterns
**Cons**: Many hyperparameters

```python
model = MultiTaskModel(
    architecture='cross-stitch',
    num_layers=3,
    cross_stitch_positions=[1, 2]  # Where to apply cross-stitch
)
```

### Progressive Neural Networks
**Type**: Sequential task learning
**Best for**: Continual learning, avoiding catastrophic forgetting
**Pros**: No negative transfer, retains previous knowledge
**Cons**: Growing model size

## 🔧 Configuration

### Model Architecture

```python
model_config = {
    'architecture': 'shared-bottom',  # or 'mmoe', 'cross-stitch'
    'shared_layers': [256, 128, 64],  # Shared encoder
    'task_layers': {
        'task1': [32, 16],            # Task-specific heads
        'task2': [32, 16],
        'task3': [64, 32]
    },
    'activation': 'relu',
    'dropout': 0.3
}
```

### Loss Weighting

```python
# Static weights
loss_weights = {
    'task1': 1.0,
    'task2': 1.0,
    'task3': 0.5  # Less important task
}

# Dynamic weighting (uncertainty-based)
model = MultiTaskModel(
    tasks=tasks,
    loss_weighting='uncertainty',  # Learn task uncertainties
    init_weights={'task1': 1.0, 'task2': 1.0}
)

# Gradient normalization
model = MultiTaskModel(
    tasks=tasks,
    loss_weighting='grad-norm'  # Normalize task gradients
)
```

### Task Relationships

```python
# Define task relationships for better sharing
task_relations = {
    ('task1', 'task2'): 'positive',  # Related tasks
    ('task1', 'task3'): 'negative',  # Conflicting tasks
    ('task2', 'task3'): 'neutral'
}
```

## 📊 Training Strategies

### 1. Alternating Training

```python
# Train on one task at a time
for epoch in range(num_epochs):
    for task in tasks:
        for batch in task_dataloaders[task]:
            loss = train_step(task, batch)
```

### 2. Joint Training

```python
# Train on all tasks simultaneously
for epoch in range(num_epochs):
    for batch in multi_task_dataloader:
        losses = {}
        for task in tasks:
            losses[task] = compute_loss(task, batch)

        total_loss = weighted_sum(losses)
        total_loss.backward()
```

### 3. Curriculum Learning

```python
# Start with easier tasks, gradually add harder ones
task_schedule = {
    0: ['task1'],                    # Epochs 0-10
    10: ['task1', 'task2'],         # Epochs 10-20
    20: ['task1', 'task2', 'task3'] # Epochs 20+
}
```

## 💡 Best Practices

### 1. **Task Selection**
- Choose related tasks (shared structure)
- Verify positive transfer (pilot studies)
- Consider task importance
- Check data availability per task

### 2. **Loss Weighting**
- Start with equal weights
- Use uncertainty weighting for automatic balancing
- Monitor per-task performance
- Adjust based on task priority

### 3. **Architecture Design**
- Start with shared-bottom (simplest)
- Use MMoE for complex task relationships
- More sharing for closely related tasks
- Task-specific capacity for unique aspects

### 4. **Monitoring Training**
- Track per-task losses separately
- Watch for negative transfer
- Monitor gradient magnitudes
- Check task dominance

### 5. **Evaluation**
- Evaluate each task independently
- Compare to single-task baselines
- Measure transfer benefits
- Report aggregate metrics

## 🐛 Troubleshooting

**One task dominates?**
- Adjust loss weights (reduce dominant task weight)
- Use gradient normalization
- Balance dataset sizes
- Check learning rates per task

**Negative transfer?**
- Reduce sharing (more task-specific layers)
- Use MMoE or soft sharing
- Verify tasks are actually related
- Try task grouping

**Poor performance on all tasks?**
- Model too small for all tasks
- Conflicts in learning objectives
- Improper loss weighting
- Need more task-specific capacity

**Training instability?**
- Normalize gradients
- Reduce learning rate
- Use gradient clipping
- Balance batch sampling

**Overfitting to one task?**
- Add task-specific regularization
- Balance training data
- Use cross-task validation
- Adjust loss weights

## 📈 Architecture Comparison

| Architecture | Sharing | Parameters | Complexity | Negative Transfer Risk | Best For |
|--------------|---------|------------|------------|------------------------|----------|
| **Shared Bottom** | Hard | Low | Low | Medium | Related tasks |
| **MMoE** | Soft (Gating) | Medium | Medium | Low | Complex relationships |
| **Cross-Stitch** | Soft (Linear) | Medium | Medium | Low | Fine-grained control |
| **Progressive** | None | High | Low | Very Low | Continual learning |
| **MTAN** | Attention | High | High | Low | Diverse tasks |

## 🎓 Key Concepts

### Positive vs Negative Transfer
- **Positive**: Learning task A helps with task B
- **Negative**: Learning task A hurts task B performance
- **Neutral**: No transfer effect

### Task Relatedness
Crucial for multi-task learning success:
- **High**: Similar input/output, shared features
- **Medium**: Some overlap in representations
- **Low**: Different domains, may hurt performance

### Catastrophic Forgetting
Model forgets previous tasks when learning new ones
- **Solution**: Use progressive networks, regularization

### Loss Weighting Strategies
1. **Equal**: All tasks weighted equally
2. **Manual**: Domain expert sets weights
3. **Uncertainty**: Learn task uncertainties (Kendall et al.)
4. **GradNorm**: Normalize gradient magnitudes
5. **Dynamic**: Adjust during training

## 📊 Evaluation Metrics

### Per-Task Performance
Evaluate each task individually using appropriate metrics

### Transfer Gain
```
Transfer Gain = (MTL Performance - Single Task Performance) / Single Task Performance
```

### Average Performance
Weighted or unweighted average across tasks

### Pareto Optimality
No task can improve without hurting another

## 📄 Dependencies

```bash
pip install torch torchvision numpy matplotlib

# For advanced MTL
pip install pytorch-lightning  # Training framework

# For visualization
pip install tensorboard seaborn
```

## 🏆 Status

**Version:** 1.0
**Status:** Research/Educational

**Features:**
- ✅ Shared Bottom Architecture
- ✅ Multi-Task Training Loop
- ✅ Loss Weighting Strategies
- ✅ Per-Task Evaluation
- ⚠️ MMoE - Planned
- ⚠️ Cross-Stitch Networks - Planned
- ⚠️ MTAN - Planned

## 📞 Support

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Email**: clientbrill@gmail.com
**LinkedIn**: [BrillConsulting](https://www.linkedin.com/in/brillconsulting)

---

**⭐ Star this repository if you find it useful!**

*Made with ❤️ by BrillConsulting*
