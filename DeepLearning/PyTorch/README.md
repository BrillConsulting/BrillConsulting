# PyTorch Deep Learning Models

Complete PyTorch implementation for neural networks and deep learning.

## Features

- **CNN Models**: Convolutional Neural Networks for image processing
- **RNN/LSTM Models**: Recurrent networks for sequence processing
- **Transformer Models**: Attention-based models for NLP and beyond
- **Training Loops**: Complete training and evaluation pipelines
- **DataLoader**: Custom datasets and data loading
- **Model Checkpointing**: Save and load model states
- **GPU Support**: CUDA acceleration for training
- **Mixed Precision**: Automatic mixed precision training

## Technologies

- PyTorch 2.1+
- torchvision
- CUDA (optional)

## Usage

```python
from pytorch_models import PyTorchModels

# Initialize
pytorch = PyTorchModels()

# Create CNN model
cnn = pytorch.create_cnn_model({
    'name': 'ImageClassifier',
    'output_size': 1000
})

# Create RNN model
rnn = pytorch.create_rnn_model({
    'name': 'TextClassifier',
    'hidden_size': 256,
    'num_layers': 2,
    'bidirectional': True
})

# Create Transformer
transformer = pytorch.create_transformer_model({
    'name': 'Seq2Seq',
    'd_model': 512,
    'nhead': 8
})

# Generate training loop
training_code = pytorch.create_training_loop({
    'epochs': 50,
    'learning_rate': 0.001,
    'optimizer': 'Adam'
})
```

## Demo

```bash
python pytorch_models.py
```
