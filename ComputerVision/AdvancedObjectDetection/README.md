# Advanced Object Detection

State-of-the-art object detection with multiple model backends: YOLOv8, Detectron2 (Faster R-CNN), and DETR (Transformer-based).

## 🎯 Supported Models

### 1. **YOLOv8** (Ultralytics)
- Real-time detection (60+ FPS)
- Multiple sizes: nano, small, medium, large, xlarge
- 80 COCO classes
- Best for: Speed and real-time applications

### 2. **Detectron2** (Facebook AI)
- Faster R-CNN backbone
- Higher accuracy than YOLO
- R-50, R-101, X-101 variants
- Best for: Accuracy-critical applications

### 3. **DETR** (Transformer-based)
- End-to-end detection with transformers
- No NMS required
- ResNet-50/101 backbones
- Best for: Research and experimentation

## ✨ Features

- **Multi-model support**: Switch between YOLOv8, Detectron2, DETR
- **Flexible model sizes**: nano to xlarge
- **Video processing**: Frame-by-frame detection with FPS counter
- **Real-time performance**: GPU acceleration
- **Easy API**: Simple detection and visualization
- **Production ready**: Clean, documented code

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt

# For Detectron2 (optional)
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Basic Usage

```python
from advanced_detector import AdvancedObjectDetector
import cv2

# Initialize detector
detector = AdvancedObjectDetector(
    model_type="yolov8",  # or "detectron2", "detr"
    model_size="medium",   # nano, small, medium, large, xlarge
    device="auto"          # auto, cpu, cuda
)

# Detect objects
image = cv2.imread("image.jpg")
detections = detector.detect(image, conf_threshold=0.5)

# Visualize
output = detector.visualize(image, detections)
cv2.imwrite("output.jpg", output)
```

### Command Line

```bash
# YOLOv8
python advanced_detector.py --model yolov8 --size medium --source image.jpg --output result.jpg

# Detectron2
python advanced_detector.py --model detectron2 --size large --source image.jpg --conf 0.6

# DETR
python advanced_detector.py --model detr --source video.mp4 --output output.mp4

# Video processing
python advanced_detector.py --model yolov8 --size small --source input.mp4 --output output.mp4
```

## 📊 Model Comparison

| Model | Speed (FPS) | Accuracy (mAP) | Memory | Use Case |
|-------|------------|----------------|---------|----------|
| YOLOv8-nano | 100+ | 37.3 | Low | Edge devices |
| YOLOv8-medium | 60+ | 50.2 | Medium | Real-time apps |
| YOLOv8-xlarge | 30+ | 53.9 | High | High accuracy |
| Detectron2-R50 | 20-25 | 39.6 | Medium | Balanced |
| Detectron2-X101 | 10-15 | 43.0 | High | Best accuracy |
| DETR-R50 | 15-20 | 42.0 | Medium | Research |

## 🎯 Advanced Features

### Batch Processing
```python
images = [cv2.imread(f"img{i}.jpg") for i in range(10)]
for image in images:
    detections = detector.detect(image)
    # Process detections
```

### Custom Thresholds
```python
detections = detector.detect(
    image,
    conf_threshold=0.7,  # Higher confidence
    iou_threshold=0.3     # Stricter NMS
)
```

### Detection Filtering
```python
# Filter by class
person_detections = [d for d in detections if d.class_name == "person"]

# Filter by confidence
high_conf = [d for d in detections if d.confidence > 0.8]

# Filter by area
large_objects = [
    d for d in detections
    if (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) > 10000
]
```

## 📚 Model Details

### YOLOv8 Variants
- **YOLOv8n**: 3.2M params, 8.7 GFLOPs
- **YOLOv8s**: 11.2M params, 28.6 GFLOPs
- **YOLOv8m**: 25.9M params, 78.9 GFLOPs
- **YOLOv8l**: 43.7M params, 165.2 GFLOPs
- **YOLOv8x**: 68.2M params, 257.8 GFLOPs

### Detectron2 Configs
- Faster R-CNN R-50 FPN 1x
- Faster R-CNN R-101 FPN 3x
- Faster R-CNN X-101 32x8d FPN 3x

### DETR Models
- DETR ResNet-50 (41.3M params)
- DETR ResNet-101 (60.3M params)

## 🎮 Use Cases

- Surveillance systems
- Autonomous vehicles
- Retail analytics
- Sports analysis
- Industrial inspection
- Wildlife monitoring
- Medical imaging
- Robotics

## 📝 Notes

- YOLOv8: Best for real-time applications
- Detectron2: Best for accuracy
- DETR: Best for research and custom tasks
- GPU highly recommended for video processing
- Larger models = better accuracy but slower

## 👤 Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

---

**State-of-the-Art Object Detection | Production Ready**
