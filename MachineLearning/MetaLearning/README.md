# 🧠 Meta Learning

Learning to learn - algorithms that improve their learning strategies through experience across multiple tasks.

## 🌟 Approaches

### Model-Agnostic Meta-Learning (MAML)
1. **MAML** - Fast adaptation with few gradient steps
2. **Reptile** - Simpler first-order MAML variant
3. **FOMAML** - First-Order MAML (computationally efficient)

### Metric Learning
4. **Prototypical Networks** - Learn class prototypes
5. **Matching Networks** - Attention-based matching
6. **Siamese Networks** - Learn similarity metrics
7. **Relation Networks** - Learn relation functions

### Memory-Augmented
8. **Neural Turing Machines** - External memory
9. **MANN** - Memory-Augmented Neural Networks
10. **Meta-Networks** - Fast parameterization

### Optimization-Based
11. **Meta-SGD** - Learn learning rates
12. **LSTM Meta-Learner** - RNN-based optimizer

## ✨ Key Features

- **Few-Shot Learning** - Learn from 1-5 examples per class
- **Fast Adaptation** - Quick fine-tuning to new tasks
- **Transfer Learning** - Knowledge across related tasks
- **Task Distribution** - Meta-train on task families
- **Gradient-Based** - Optimize for learnability
- **Memory-Efficient** - Minimal data requirements

## 🚀 Quick Start

### MAML Training

```bash
python meta_learning.py --algorithm maml --shots 5 --tasks 1000 --inner-steps 5
```

### Prototypical Networks

```bash
python meta_learning.py --algorithm prototypical --shots 1 --way 5 --query 15
```

## 📊 Example Code

```python
from meta_learning import MetaLearner
import torch

# Initialize MAML meta-learner
meta_learner = MetaLearner(
    algorithm='maml',
    model=ConvNet(),
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5
)

# Meta-training loop
for episode in range(num_episodes):
    # Sample batch of tasks
    tasks = task_sampler.sample(batch_size=4)

    # Meta-update
    meta_loss = meta_learner.meta_train_step(tasks)

    print(f"Episode {episode}: Meta-loss = {meta_loss:.4f}")

# Adapt to new task (few-shot)
support_set = {'x': x_support, 'y': y_support}
query_set = {'x': x_query, 'y': y_query}

# Fast adaptation
adapted_model = meta_learner.adapt(
    support_set,
    n_steps=10
)

# Evaluate on query set
accuracy = meta_learner.evaluate(adapted_model, query_set)
print(f"Few-shot accuracy: {accuracy:.2%}")
```

## 🎯 Use Cases

### 🤖 Few-Shot Classification
- **Image Recognition**: New objects with few examples
- **Medical Diagnosis**: Rare diseases, limited data
- **Species Classification**: New animal/plant species
- **Character Recognition**: New fonts/handwriting styles

### 🌐 Natural Language
- **Translation**: Low-resource languages
- **Sentiment Analysis**: New domains quickly
- **Named Entity Recognition**: Domain adaptation
- **Text Classification**: Few training examples

### 🏢 Personalization
- **Recommendation**: New users, cold start
- **Content Filtering**: User preferences
- **Adaptive UI**: Learn user behavior quickly
- **Custom Models**: Per-user adaptation

### 🏭 Industrial Applications
- **Quality Control**: New defect types
- **Process Optimization**: New production lines
- **Predictive Maintenance**: New equipment
- **Anomaly Detection**: Emerging patterns

### 🎮 Robotics & Control
- **Robot Adaptation**: New environments/tasks
- **Grasping**: Novel objects
- **Locomotion**: Different terrains
- **Manipulation**: Task transfer

## 📚 Algorithm Details

### MAML (Model-Agnostic Meta-Learning)
**Type**: Gradient-based, model-agnostic
**Best for**: Fast adaptation with gradients
**Pros**: Works with any gradient-based model, theoretically grounded
**Cons**: Computationally expensive (second-order gradients)

**Key idea**: Learn initialization that allows fast adaptation

```python
meta_learner = MetaLearner(
    algorithm='maml',
    inner_lr=0.01,      # Task-specific learning rate
    outer_lr=0.001,     # Meta learning rate
    inner_steps=5,      # Adaptation gradient steps
    first_order=False   # Use second-order gradients
)
```

**Meta-training**:
1. Sample batch of tasks
2. For each task: Copy model → Adapt on support set → Compute query loss
3. Meta-update: Update initialization to minimize meta-loss

### Prototypical Networks
**Type**: Metric learning, embedding-based
**Best for**: Few-shot classification, interpretability
**Pros**: Simple, fast, works well in practice
**Cons**: Limited to classification, fixed metric

**Key idea**: Learn embedding where classes cluster around prototypes

```python
meta_learner = MetaLearner(
    algorithm='prototypical',
    embedding_dim=64,
    distance='euclidean'  # or 'cosine'
)
```

**Classification**:
1. Compute class prototypes (mean of support embeddings)
2. Classify by nearest prototype in embedding space

### Reptile
**Type**: First-order MAML variant
**Best for**: Simpler, faster alternative to MAML
**Pros**: No second-order gradients, easier to implement
**Cons**: Less theoretically motivated

```python
meta_learner = MetaLearner(
    algorithm='reptile',
    inner_lr=0.02,
    outer_lr=0.1,
    inner_steps=5
)
```

### Matching Networks
**Type**: Attention-based, metric learning
**Best for**: One-shot learning, episodic training
**Pros**: Attention mechanism, no fine-tuning needed
**Cons**: More complex, slower inference

### Relation Networks
**Type**: Learnable comparison
**Best for**: Complex similarity metrics
**Pros**: Learns relation function end-to-end
**Cons**: Requires more data than prototypical

## 🔧 Configuration

### MAML Parameters

```python
maml_config = {
    'inner_lr': 0.01,           # Task adaptation rate
    'outer_lr': 0.001,          # Meta learning rate
    'inner_steps': 5,           # Gradient steps per task
    'first_order': False,       # Use FOMAML (faster)
    'allow_unused': True,       # Gradient computation
    'allow_nograd': True
}
```

### Episode Setup (N-way K-shot)

```python
episode_config = {
    'n_way': 5,                 # Number of classes per task
    'k_shot': 1,                # Support examples per class
    'query_size': 15,           # Query examples per class
    'num_tasks': 1000           # Meta-training tasks
}
```

### Training Configuration

```python
training_config = {
    'meta_batch_size': 4,       # Tasks per meta-update
    'num_epochs': 100,
    'eval_frequency': 10,       # Evaluate every N epochs
    'save_frequency': 20
}
```

## 📊 Training Workflow

### 1. Define Task Distribution

```python
# Example: Image classification tasks
task_sampler = TaskSampler(
    dataset='omniglot',  # or 'mini-imagenet'
    n_way=5,
    k_shot=1,
    query_size=15
)
```

### 2. Meta-Training

```python
for epoch in range(num_epochs):
    epoch_loss = 0

    for batch in range(num_batches):
        # Sample tasks
        tasks = task_sampler.sample(meta_batch_size)

        # Meta-update
        loss = meta_learner.meta_train_step(tasks)
        epoch_loss += loss

    print(f"Epoch {epoch}: Loss = {epoch_loss / num_batches:.4f}")
```

### 3. Fast Adaptation (Meta-Testing)

```python
# New task with few examples
support_x, support_y = new_task.support_set()
query_x, query_y = new_task.query_set()

# Adapt model
adapted_model = meta_learner.adapt(
    support_x, support_y,
    n_steps=10
)

# Evaluate
accuracy = adapted_model.evaluate(query_x, query_y)
```

## 💡 Best Practices

### 1. **Choose Right Algorithm**
- Simple tasks: Prototypical Networks
- Complex adaptation: MAML
- Limited compute: Reptile, FOMAML
- Need interpretability: Metric learning methods

### 2. **Task Distribution Design**
- Ensure tasks are related but diverse
- Balance task difficulty
- Include representative examples
- Test on similar distribution

### 3. **Hyperparameter Tuning**
- Inner LR: Typically 0.01-0.1
- Outer LR: Usually 10x smaller than inner
- Inner steps: 5-10 for most tasks
- Meta-batch size: 2-8 tasks

### 4. **Data Augmentation**
- Critical for few-shot learning
- Task-specific augmentation
- Test-time augmentation

### 5. **Evaluation Protocol**
- Use separate meta-test set
- Report confidence intervals
- Test on multiple shots (1, 5, 10)
- Cross-validate task sampling

## 🐛 Troubleshooting

**Poor adaptation?**
- Increase inner learning rate
- More inner gradient steps
- Check task relatedness
- Verify gradient flow

**Meta-training unstable?**
- Reduce outer learning rate
- Use gradient clipping
- Normalize gradients
- Smaller meta-batch size

**Overfitting to meta-train tasks?**
- Increase task diversity
- Add regularization
- Early stopping on meta-val
- Data augmentation

**Slow convergence?**
- Increase meta-batch size
- Tune learning rates
- Use first-order approximation
- Better initialization

**Poor generalization to new tasks?**
- Ensure task distribution match
- Increase meta-training tasks
- More diverse training tasks
- Check domain shift

## 📈 Algorithm Comparison

| Algorithm | Type | Computation | Interpretability | Flexibility | Few-Shot Performance |
|-----------|------|-------------|------------------|-------------|---------------------|
| **MAML** | Gradient | High (2nd order) | Low | High | Excellent |
| **Reptile** | Gradient | Medium (1st order) | Low | High | Very Good |
| **Prototypical** | Metric | Low | High | Medium | Very Good |
| **Matching** | Metric | Medium | Medium | Medium | Good |
| **Relation** | Metric | Medium | Low | Medium | Very Good |
| **Meta-SGD** | Gradient | Very High | Low | High | Excellent |

## 🎓 Key Concepts

### N-Way K-Shot Learning
- **N-way**: N classes in each task
- **K-shot**: K examples per class (support set)
- **Query set**: Test examples for meta-loss

**Example**: 5-way 1-shot
- 5 classes
- 1 example per class (5 total)
- 15 query examples (3 per class)

### Support vs Query Set
- **Support**: Few labeled examples for adaptation
- **Query**: Unlabeled examples for evaluation
- Similar to train/test but within each task

### Inner vs Outer Loop
- **Inner loop**: Task-specific adaptation (fast)
- **Outer loop**: Meta-learning update (slow)

### Task Distribution
- Family of related tasks
- Meta-train: Learn from task distribution
- Meta-test: Adapt to new tasks from same distribution

## 📊 Evaluation Metrics

### Few-Shot Accuracy
Average accuracy on query sets after K-shot adaptation

```python
accuracy = evaluate_few_shot(
    meta_learner,
    test_tasks,
    n_way=5,
    k_shot=1,
    n_episodes=600
)
```

### Adaptation Speed
How quickly model adapts (fewer gradient steps = better)

### Cross-Task Transfer
Performance gain from meta-training vs. random init

## 📄 Dependencies

```bash
pip install torch torchvision numpy matplotlib

# For meta-learning frameworks
pip install learn2learn  # PyTorch meta-learning library
pip install higher  # Differentiable optimization

# For datasets
pip install torchmeta  # Meta-learning datasets
```

## 🏆 Status

**Version:** 1.0
**Status:** Research/Educational

**Features:**
- ✅ MAML Implementation
- ✅ Prototypical Networks
- ✅ Few-Shot Evaluation
- ✅ Task Sampling
- ⚠️ Reptile - Planned
- ⚠️ Matching Networks - Planned
- ⚠️ Meta-SGD - Planned

## 📞 Support

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Email**: clientbrill@gmail.com
**LinkedIn**: [BrillConsulting](https://www.linkedin.com/in/brillconsulting)

---

**⭐ Star this repository if you find it useful!**

*Made with ❤️ by BrillConsulting*
