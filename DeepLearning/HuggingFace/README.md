# Hugging Face Transformers Models

State-of-the-art transformers for NLP, computer vision, audio, and multimodal tasks.

## Features

- **Text Classification**: Sentiment analysis, topic classification
- **Token Classification**: Named Entity Recognition (NER), POS tagging
- **Question Answering**: Extractive QA systems
- **Text Generation**: GPT-2, GPT-3 style text generation
- **Translation**: Multi-language translation
- **Summarization**: Abstractive and extractive summarization
- **Image Classification**: Vision Transformers (ViT)
- **Custom Training**: Fine-tune any model with Trainer API
- **100+ Pre-trained Models**: BERT, GPT, T5, BART, RoBERTa, etc.

## Technologies

- Transformers 4.35+
- PyTorch/TensorFlow
- Datasets library
- Accelerate

## Usage

```python
from transformers_models import HuggingFaceTransformers

# Initialize
hf = HuggingFaceTransformers()

# Text Classification
text_clf = hf.create_text_classification_pipeline({
    'model': 'distilbert-base-uncased-finetuned-sst-2-english'
})

# NER
ner = hf.create_token_classification_pipeline({
    'model': 'dbmdz/bert-large-cased-finetuned-conll03-english'
})

# Question Answering
qa = hf.create_question_answering_pipeline({
    'model': 'distilbert-base-cased-distilled-squad'
})

# Text Generation
gen = hf.create_text_generation_pipeline({
    'model': 'gpt2'
})

# Image Classification
img_clf = hf.create_image_classification_pipeline({
    'model': 'google/vit-base-patch16-224'
})
```

## Demo

```bash
python transformers_models.py
```
