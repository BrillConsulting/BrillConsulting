# TensorFlow/Keras Deep Learning Models

Complete TensorFlow and Keras implementation for neural networks and deep learning.

## Features

- **Sequential Models**: Simple linear stack of layers
- **Functional API**: Multi-input, multi-output models with complex topologies
- **Transfer Learning**: Pre-trained models (ResNet, VGG, Inception, EfficientNet)
- **Custom Training Loops**: tf.GradientTape for advanced training
- **Keras Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
- **Data Augmentation**: Built-in augmentation layers
- **Mixed Precision**: Automatic mixed precision training
- **Distributed Training**: Multi-GPU and TPU support

## Technologies

- TensorFlow 2.14+
- Keras
- TensorBoard

## Usage

```python
from tensorflow_models import TensorFlowModels

# Initialize
tf_models = TensorFlowModels()

# Create Sequential CNN
cnn = tf_models.create_sequential_cnn({
    'name': 'ImageClassifier',
    'input_shape': (224, 224, 3),
    'num_classes': 1000
})

# Create Functional model
functional = tf_models.create_functional_model({
    'name': 'MultiIOModel',
    'multi_input': True,
    'multi_output': True
})

# Transfer Learning
transfer = tf_models.create_transfer_learning_model({
    'base_model': 'ResNet50',
    'freeze_layers': True,
    'num_classes': 10
})

# Generate callbacks
callbacks_code = tf_models.create_callbacks()
```

## Demo

```bash
python tensorflow_models.py
```
