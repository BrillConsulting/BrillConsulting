# Self-Supervised Learning

Learn visual representations without labels using contrastive and predictive methods.

## Features

- **SimCLR**: Contrastive learning with data augmentation
- **MoCo**: Momentum contrast with queue
- **BYOL**: Bootstrap your own latent without negatives
- **SwAV**: Clustering-based contrastive learning

## Methods

### SimCLR
- NT-Xent loss
- Strong data augmentation
- Large batch training

### MoCo
- Momentum encoder
- Queue for negatives
- Memory-efficient

### BYOL
- No negative pairs
- EMA teacher network
- Predictor asymmetry

### SwAV
- Online clustering
- Sinkhorn-Knopp algorithm
- Multi-crop augmentation

## Installation

```bash
pip install -r requirements.txt
```
