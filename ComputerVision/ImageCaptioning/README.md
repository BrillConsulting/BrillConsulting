# Image Captioning

Generate natural language descriptions of images using encoder-decoder architectures.

## Features

- **Show and Tell**: Basic encoder-decoder with LSTM
- **Show, Attend and Tell**: Attention-based captioning
- **Transformer-based**: Modern architecture using self-attention
- **Beam search**: Generate diverse captions
- **Multiple attention mechanisms**: Bahdanau and Luong attention

## Architectures

### 1. LSTM with Attention
- ResNet-101 image encoder
- LSTM decoder with Bahdanau attention
- Gated attention mechanism
- Top-down attention

### 2. Transformer
- Vision patches or CNN features
- Multi-head self-attention
- Causal masking for autoregressive generation
- Positional encodings

## Components

- **ImageEncoder**: CNN-based feature extraction
- **AttentionDecoder**: LSTM with visual attention
- **TransformerDecoder**: Transformer-based decoder
- **BahdanauAttention**: Additive attention mechanism
- **PositionalEncoding**: Sinusoidal position embeddings

## Usage

```python
from image_captioning import ImageCaptioningModel

# LSTM model
model = ImageCaptioningModel(
    vocab_size=10000,
    embed_size=512,
    hidden_size=512,
    decoder_type='lstm'
)

# Generate caption
caption = model.sample(image, max_length=20)

# Transformer model
transformer_model = ImageCaptioningModel(
    vocab_size=10000,
    embed_size=512,
    decoder_type='transformer'
)

predictions = transformer_model(images, captions)
```

## Training

Train on image-caption datasets:
- MS COCO
- Flickr30k
- Conceptual Captions

## Evaluation Metrics

- BLEU (1-4)
- METEOR
- CIDEr
- SPICE

## Installation

```bash
pip install -r requirements.txt
```

## Example

```bash
python image_captioning.py
```
