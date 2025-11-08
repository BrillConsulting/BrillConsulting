# Optical Flow Estimation

Estimate dense pixel-level motion between consecutive frames using deep learning.

## Features

- **FlowNetSimple**: Encoder-decoder architecture for optical flow
- **PWC-Net**: Pyramid, Warping, and Cost Volume based flow estimation
- **Cost volume computation**: Efficient correlation calculation
- **Feature warping**: Warp features using estimated flow
- **Multi-scale estimation**: Coarse-to-fine flow refinement

## Architectures

### 1. FlowNetSimple
- Simple encoder-decoder with skip connections
- Multi-scale flow prediction
- Fast inference

### 2. PWC-Net
- Feature pyramid extraction
- Cost volume construction
- Iterative flow refinement
- Warping at each pyramid level

## Usage

```python
from optical_flow import FlowNetSimple, PWCNet

# FlowNetSimple
flownet = FlowNetSimple()
flow = flownet(img1, img2)

# PWC-Net
pwcnet = PWCNet()
flow = pwcnet(img1, img2)
```

## Applications

- Video compression
- Motion segmentation
- Video stabilization
- Action recognition
- Object tracking

## Installation

```bash
pip install -r requirements.txt
```
