# Panoptic Segmentation

Unified scene understanding combining semantic and instance segmentation using Detectron2 Panoptic FPN.

## 🎯 Overview

Panoptic segmentation unifies:
- **Semantic segmentation** (stuff): backgrounds, sky, road, grass
- **Instance segmentation** (things): people, cars, animals (with individual IDs)

## ✨ Features

- Detectron2 Panoptic FPN (R-50, R-101)
- 133 COCO categories (54 stuff + 79 things)
- Per-pixel classification + instance IDs
- Statistics and analytics
- Beautiful visualizations

## 🚀 Quick Start

```bash
pip install torch torchvision opencv-python
pip install 'git+https://github.com/facebookresearch/detectron2.git'

python panoptic_segmentation.py --image street.jpg --output result.jpg
```

## 📊 Use Cases

- Autonomous driving (road, vehicles, pedestrians)
- Scene understanding
- Urban planning
- Agricultural monitoring
- Robot navigation

## 👤 Author

**Brill Consulting** | Computer Vision Expert
