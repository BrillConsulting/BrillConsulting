# Change Detection

Detect and localize changes between images from different time periods.

## Features

- **Siamese Networks**: Shared encoder for bi-temporal images
- **ChangeFormer**: Transformer-based change detection
- **Attention Mechanisms**: Spatial and temporal attention
- **Multi-scale Features**: Feature pyramid for better detection

## Methods

### Siamese Network
- Shared weights for both images
- Feature concatenation
- U-Net style decoder

### ChangeFormer
- Transformer encoder
- Positional encoding
- Temporal feature fusion

### Attention-based
- Spatial attention
- Temporal attention
- Multi-scale differences

## Applications

- Satellite imagery analysis
- Urban planning
- Disaster assessment
- Environmental monitoring
- Medical image analysis

## Installation

```bash
pip install -r requirements.txt
```

## Example

```bash
python change_detection.py
```
