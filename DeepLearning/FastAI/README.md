# FastAI Deep Learning Models

High-level deep learning library built on PyTorch with best practices.

## Features

- **Vision Learner**: Image classification, object detection, segmentation
- **Text Learner**: Text classification, language models, NLP tasks
- **Tabular Learner**: Structured data prediction with embeddings
- **Collaborative Filtering**: Recommendation systems
- **Learning Rate Finder**: Automatic LR optimization
- **Progressive Resizing**: Train small then large
- **Discriminative Learning Rates**: Different LRs per layer
- **Mixup**: Advanced data augmentation
- **Mixed Precision**: FP16 training

## Technologies

- FastAI 2.7+
- PyTorch
- torchvision

## Usage

```python
from fastai_models import FastAIModels

# Initialize
fastai = FastAIModels()

# Vision Learner
vision = fastai.create_vision_learner({
    'name': 'ImageClassifier',
    'architecture': 'resnet50',
    'pretrained': True
})

# Text Learner
text = fastai.create_text_learner({
    'name': 'SentimentAnalyzer',
    'architecture': 'AWD_LSTM'
})

# Tabular Learner
tabular = fastai.create_tabular_learner({
    'name': 'TabularPredictor',
    'layers': [200, 100]
})

# Collaborative Filtering
collab = fastai.create_collab_learner({
    'name': 'Recommender',
    'n_factors': 50
})
```

## Demo

```bash
python fastai_models.py
```
