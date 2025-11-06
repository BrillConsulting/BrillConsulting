# Keras High-Level Neural Networks API

## 🎯 Overview

Advanced deep learning implementations using Keras/TensorFlow, featuring residual networks, bidirectional LSTMs, custom attention layers, and sophisticated data augmentation techniques.

## ✨ Features

### Advanced Architectures
- **ResNet-style CNN**: Deep convolutional networks with residual connections
- **Bidirectional LSTM**: Sequential models for time series and NLP
- **Custom Attention Layer**: Self-attention mechanism implementation
- **Functional API**: Complex multi-input/multi-output models

### Training Enhancements
- **Learning Rate Schedulers**: Cosine decay with warmup, cyclical LR, step decay
- **Data Augmentation**: MixUp and CutMix advanced augmentation
- **Custom Callbacks**: Model checkpointing, early stopping, LR reduction
- **Batch Normalization**: Stable training for deep networks

### Model Components
- Residual blocks with skip connections
- Multi-head attention mechanisms
- Dense layers with dropout regularization
- Global average pooling

## 📋 Requirements

```bash
pip install tensorflow>=2.10.0 numpy matplotlib
```

## 🚀 Quick Start

```python
from keras_models import KerasModelBuilder

# Initialize builder
builder = KerasModelBuilder()

# Build advanced CNN with residual blocks
cnn = builder.build_advanced_cnn({
    'input_shape': (224, 224, 3),
    'num_classes': 1000,
    'use_residual': True
})

# Build bidirectional LSTM
lstm = builder.build_lstm_model({
    'sequence_length': 100,
    'input_dim': 128,
    'lstm_units': [256, 128],
    'bidirectional': True
})

# Custom attention layer
attention_code = builder.create_custom_layer()

# Advanced data augmentation
augmentation_code = builder.create_data_augmentation()
```

## 🏗️ Architecture Examples

### ResNet Block
```
Input → Conv2D → BatchNorm → ReLU
  |                            ↓
  |                         Conv2D → BatchNorm
  |                            ↓
  └─────────────────────────→ Add → ReLU → Output
```

### Bidirectional LSTM
```
Input → BiLSTM(256) → Dropout(0.3)
          ↓
      BiLSTM(128) → Dropout(0.3)
          ↓
      Dense(128) → Dropout(0.4)
          ↓
      Dense(num_classes) → Softmax
```

## 💡 Use Cases

- **Computer Vision**: Image classification with residual networks
- **NLP**: Sequence modeling with bidirectional LSTMs
- **Time Series**: Forecasting with attention mechanisms
- **Transfer Learning**: Fine-tuning pre-trained models
- **Multi-task Learning**: Complex architectures with multiple outputs

## 📊 Features

### Advanced CNN
- Input: (224, 224, 3)
- Architecture: 7x7 conv → residual blocks → global pooling
- Residual connections prevent vanishing gradients
- Batch normalization for stable training

### Bidirectional LSTM
- Processes sequences in both directions
- Returns full sequences or final states
- Dropout for regularization
- Suitable for variable-length inputs

### Learning Rate Schedulers
- **Cosine Decay with Warmup**: Gradual warmup then cosine annealing
- **Cyclical LR**: Oscillating learning rate for better convergence
- **Step Decay**: Reduce LR on plateau

### Data Augmentation
- **MixUp**: Convex combination of image pairs
- **CutMix**: Replace random patches between images
- Improves generalization and robustness

## 🔬 Advanced Features

### Custom Attention Layer
```python
class AttentionLayer(layers.Layer):
    def __init__(self, units=128):
        super().__init__()
        self.units = units

    def call(self, inputs):
        # Score calculation
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1),
            axis=1
        )

        # Apply attention
        context_vector = attention_weights * inputs
        return tf.reduce_sum(context_vector, axis=1)
```

### Model Training Pipeline
1. Define architecture (Sequential or Functional API)
2. Compile with optimizer, loss, and metrics
3. Setup callbacks (checkpointing, LR scheduling)
4. Apply data augmentation
5. Train with fit() or custom training loop
6. Evaluate on test set

## 📚 References

- Keras Documentation: https://keras.io
- ResNet Paper: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Attention Mechanism: "Attention Is All You Need" (Vaswani et al., 2017)
- MixUp: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
- CutMix: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)

## 📧 Contact

For questions or collaboration: [clientbrill@gmail.com](mailto:clientbrill@gmail.com)

---

**Author:** Brill Consulting
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)
