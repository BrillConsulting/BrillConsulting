# 🎯 Computer Vision Portfolio

Professional Computer Vision portfolio with 15 production-ready projects showcasing state-of-the-art deep learning solutions for object detection, face recognition, image segmentation, OCR, classification, video analysis, pose estimation, and more.

## 📦 Projects Overview

### 1. 🎯 [Object Detection](ObjectDetection/)
Real-time object detection using YOLOv8 with support for images, videos, and webcam streams.

**Key Features:**
- 80+ object classes detection
- 30+ FPS real-time processing
- Multi-source support (image/video/webcam)
- Configurable confidence thresholds

**Technologies:** YOLOv8, PyTorch, OpenCV

```bash
cd ObjectDetection
python object_detector.py --source image.jpg --output result.jpg
```

---

### 2. 👤 [Face Recognition](FaceRecognition/)
Advanced face recognition system with enrollment, identification, and real-time tracking.

**Key Features:**
- Face enrollment and database management
- Real-time recognition from webcam
- 99.38% accuracy on LFW benchmark
- Multi-face support

**Technologies:** face_recognition, dlib, OpenCV

```bash
cd FaceRecognition
python face_recognition_system.py --mode webcam
```

---

### 3. 🎨 [Image Segmentation](ImageSegmentation/)
Semantic and instance segmentation using DeepLabV3+ and Mask R-CNN.

**Key Features:**
- 21 PASCAL VOC classes
- Pixel-perfect object boundaries
- Both semantic and instance segmentation
- Class extraction and statistics

**Technologies:** DeepLabV3+, Mask R-CNN, PyTorch

```bash
cd ImageSegmentation
python semantic_segmentation.py --image photo.jpg --mode overlay
```

---

### 4. 📝 [OCR (Optical Character Recognition)](OCR/)
Multi-language text recognition with 80+ language support.

**Key Features:**
- 80+ language support
- Document processing pipeline
- Text search capabilities
- High accuracy with EasyOCR

**Technologies:** EasyOCR, Tesseract, PyTorch

```bash
cd OCR
python text_recognition.py --image document.jpg --languages en pl
```

---

### 5. 🖼️ [Image Classification](ImageClassification/)
State-of-the-art image classification with 12+ pretrained models.

**Key Features:**
- 1000 ImageNet classes
- 12+ model architectures
- Up to 91% top-1 accuracy
- Transfer learning ready

**Technologies:** ResNet, EfficientNet, ViT, PyTorch

```bash
cd ImageClassification
python image_classifier.py --image cat.jpg --model resnet50
```

---

### 6. 🎬 [Video Analysis](VideoAnalysis/)
Real-time video processing and multi-object tracking.

**Key Features:**
- Multi-object tracking with DeepSORT
- Video frame analysis
- Real-time processing
- Object trajectory visualization

**Technologies:** YOLOv8, DeepSORT, OpenCV

```bash
cd VideoAnalysis
python video_analysis.py --video input.mp4 --output tracked.mp4
```

---

### 7. 🤸 [Pose Estimation](PoseEstimation/)
Human pose estimation and skeleton tracking.

**Key Features:**
- 2D/3D pose detection
- 33 keypoint tracking
- Multi-person support
- Real-time processing

**Technologies:** MediaPipe, OpenCV

```bash
cd PoseEstimation
python pose_estimation.py --source webcam
```

---

### 8. 🎨 [Style Transfer](StyleTransfer/)
Neural style transfer for artistic image transformation.

**Key Features:**
- Artistic style application
- VGG19-based architecture
- Configurable style/content weights
- High-quality output

**Technologies:** PyTorch, VGG19

```bash
cd StyleTransfer
python style_transfer.py --content photo.jpg --style starry_night.jpg
```

---

### 9. 🔍 [Super Resolution](SuperResolution/)
AI-powered image upscaling and enhancement.

**Key Features:**
- 4x/8x image upscaling
- ESRGAN architecture
- Detail preservation
- Batch processing

**Technologies:** ESRGAN, PyTorch

```bash
cd SuperResolution
python super_resolution.py --image low_res.jpg --scale 4
```

---

### 10. 🚨 [Anomaly Detection](AnomalyDetection/)
Visual defect detection for quality control.

**Key Features:**
- Automated defect detection
- Industrial quality control
- Anomaly scoring
- Visual inspection automation

**Technologies:** PyTorch, OpenCV

```bash
cd AnomalyDetection
python anomaly_detection.py --image product.jpg --threshold 0.8
```

---

### 11. 🔧 [Image Restoration](ImageRestoration/)
Image denoising and damage repair with deep learning.

**Key Features:**
- Noise removal and denoising
- Scratch and damage repair
- Multiple restoration algorithms
- Quality preservation

**Technologies:** PyTorch, OpenCV, scikit-image

```bash
cd ImageRestoration
python image_restoration.py --image damaged.jpg --mode denoise
```

---

### 12. 🎯 [Object Tracking](ObjectTracking/)
Advanced multi-object tracking in video sequences.

**Key Features:**
- SORT and DeepSORT algorithms
- Multi-object tracking (MOT)
- Trajectory analysis
- Real-time performance

**Technologies:** YOLOv8, DeepSORT, SORT, OpenCV

```bash
cd ObjectTracking
python object_tracker.py --video input.mp4 --tracker deepsort
```

---

### 13. 🏞️ [Scene Recognition](SceneRecognition/)
Scene classification and context understanding.

**Key Features:**
- 365 scene categories (Places365)
- Indoor/outdoor classification
- Scene attributes detection
- Context-aware recognition

**Technologies:** Places365 CNN, PyTorch, OpenCV

```bash
cd SceneRecognition
python scene_recognition.py --image scene.jpg --top-k 5
```

---

### 14. 🔗 [Image Matching](ImageMatching/)
Feature matching, homography, and image alignment.

**Key Features:**
- SIFT, ORB, AKAZE feature detection
- RANSAC homography estimation
- Image stitching and panoramas
- Keypoint matching

**Technologies:** OpenCV, SIFT, RANSAC

```bash
cd ImageMatching
python image_matching.py --image1 img1.jpg --image2 img2.jpg
```

---

### 15. 📏 [Depth Estimation](DepthEstimation/)
Monocular depth estimation from single images.

**Key Features:**
- Single image depth prediction
- MiDaS depth estimation
- 3D scene reconstruction
- Depth map visualization

**Technologies:** MiDaS, PyTorch, OpenCV

```bash
cd DepthEstimation
python depth_estimation.py --image photo.jpg --output depth_map.jpg
```

---

## 🚀 Quick Start

### Installation

Clone the repository and install dependencies for the project you want to use:

```bash
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/ComputerVision

# Install dependencies for specific project
cd ObjectDetection
pip install -r requirements.txt
```

### GPU Setup (Recommended)

For optimal performance, install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Performance Comparison

| Project | Speed (GPU) | Speed (CPU) | Accuracy | Use Case |
|---------|-------------|-------------|----------|----------|
| Object Detection | 60+ FPS | 5-10 FPS | 37.3% mAP | Real-time detection |
| Face Recognition | 30 FPS | 3-5 FPS | 99.38% | Identity verification |
| Image Segmentation | 10-20 FPS | 1-2 FPS | 79.8% mIoU | Scene understanding |
| OCR | 2 img/s | 1 img/s | 95%+ | Text extraction |
| Classification | 200+ FPS | 20 FPS | 76-91% | Object categorization |
| Video Analysis | 30-60 FPS | 3-5 FPS | 35% mAP | Multi-object tracking |
| Pose Estimation | 30+ FPS | 5-8 FPS | 90%+ PCK | Human pose detection |
| Style Transfer | 5-10 FPS | 0.5 FPS | - | Artistic rendering |
| Super Resolution | 10 FPS | 1 FPS | 28+ PSNR | Image upscaling |
| Anomaly Detection | 50+ FPS | 10 FPS | 95%+ | Defect detection |
| Image Restoration | 15-25 FPS | 2-3 FPS | 32+ PSNR | Noise removal |
| Object Tracking | 25-40 FPS | 3-5 FPS | 75+ MOTA | Multi-object tracking |
| Scene Recognition | 150+ FPS | 15 FPS | 85%+ top-5 | Scene categorization |
| Image Matching | 5-10 FPS | 1-2 FPS | 90%+ inliers | Feature matching |
| Depth Estimation | 20-30 FPS | 2-4 FPS | 0.11 RMSE | Depth prediction |

*Tested on Intel i7-12700K + RTX 3080*

## 🎨 Use Cases by Industry

### 🏢 Retail & E-commerce
- **Object Detection**: Product detection and counting
- **Face Recognition**: Customer recognition and personalization
- **OCR**: Receipt and invoice processing
- **Classification**: Product categorization

### 🏥 Healthcare
- **Segmentation**: Organ and tumor segmentation
- **Classification**: Disease diagnosis
- **OCR**: Medical record digitization
- **Face Recognition**: Patient identification

### 🚗 Autonomous Vehicles
- **Object Detection**: Vehicle and pedestrian detection
- **Segmentation**: Road scene understanding
- **Classification**: Traffic sign recognition
- **OCR**: License plate reading

### 🏭 Manufacturing
- **Object Detection**: Defect detection
- **Segmentation**: Quality control
- **Classification**: Product sorting
- **OCR**: Serial number reading

### 🏛️ Security & Surveillance
- **Face Recognition**: Access control
- **Object Detection**: Threat detection
- **OCR**: Document verification
- **Classification**: Anomaly detection

## 🔧 Technology Stack

### Deep Learning Frameworks
- **PyTorch** 2.0+
- **TorchVision** 0.15+
- **Ultralytics** (YOLOv8)
- **TensorFlow** (for legacy models)

### Computer Vision Libraries
- **OpenCV** 4.8+
- **face_recognition**
- **EasyOCR**
- **dlib**
- **MediaPipe**
- **scikit-image**

### Model Architectures
- **YOLOv8** (Object Detection)
- **ResNet-34** (Face Recognition)
- **DeepLabV3+** (Semantic Segmentation)
- **Mask R-CNN** (Instance Segmentation)
- **CRAFT + CRNN** (OCR)
- **ResNet/EfficientNet/ViT** (Classification)
- **ESRGAN** (Super Resolution)
- **MiDaS** (Depth Estimation)
- **Places365-CNN** (Scene Recognition)
- **DeepSORT/SORT** (Object Tracking)
- **VGG19** (Style Transfer)

## 📈 Project Complexity

```
Beginner        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Advanced
│               │                 │                  │
Classification  Object Detection  Segmentation      Face Recognition
Scene Recog     OCR, Tracking     Depth Estimation  + Custom Training
Image Matching  Pose Estimation                     Restoration
```

### Learning Path Recommendation:
1. **Beginner**: Image Classification, Scene Recognition
2. **Intermediate**: Object Detection, OCR, Image Matching
3. **Advanced**: Image Segmentation, Object Tracking, Pose Estimation, Depth Estimation
4. **Expert**: Face Recognition, Image Restoration, Style Transfer, Custom training

## 🎓 Educational Value

Each project includes:
- ✅ **Detailed README** with usage examples
- ✅ **Complete code** with comments
- ✅ **Requirements** for easy setup
- ✅ **Performance benchmarks**
- ✅ **Troubleshooting guides**
- ✅ **Use case examples**
- ✅ **Research paper references**

## 🏆 Key Achievements

### Technical Excellence
- 🚀 **15 Production-ready Systems**
- ⚡ **Real-time Performance** (30+ FPS)
- 🎯 **State-of-the-art Accuracy**
- 🌍 **Multi-language Support** (OCR)
- 🔧 **Modular Architecture**

### Best Practices
- ✅ Clean, documented code
- ✅ Error handling and validation
- ✅ Performance optimization
- ✅ GPU/CPU compatibility
- ✅ Extensible design patterns

## 🔬 Research & Innovation

### Papers Implemented
1. **YOLOv8**: You Only Look Once v8
2. **DeepLabV3+**: Encoder-Decoder with Atrous Separable Convolution
3. **Mask R-CNN**: Instance Segmentation
4. **EfficientNet**: Rethinking Model Scaling
5. **Vision Transformer**: An Image is Worth 16x16 Words
6. **MiDaS**: Towards Robust Monocular Depth Estimation
7. **Places365-CNN**: Scene Recognition with Deep Learning
8. **ESRGAN**: Enhanced Super-Resolution Generative Adversarial Networks
9. **DeepSORT**: Simple Online and Realtime Tracking with Deep Association Metric

### Future Additions
- [ ] 3D Object Detection
- [ ] Video Action Recognition
- [ ] Gesture Recognition
- [ ] GANs for Image Generation
- [ ] Image Captioning

## 📚 Documentation

Each project has comprehensive documentation:
- Installation guides
- Usage examples
- API reference
- Performance benchmarks
- Troubleshooting
- Research references

## 🤝 Integration Examples

### Combining Multiple Projects

```python
# Example: Full document processing pipeline

# 1. Detect text regions (OCR)
from OCR.text_recognition import OCRSystem
ocr = OCRSystem()
text = ocr.extract_text(image)

# 2. Classify document type
from ImageClassification.image_classifier import ImageClassifier
classifier = ImageClassifier()
doc_type = classifier.classify_image(image, top_k=1)

# 3. Detect faces (if ID document)
from FaceRecognition.face_recognition_system import FaceRecognitionSystem
face_system = FaceRecognitionSystem()
faces = face_system.recognize_faces(image)

# 4. Segment important regions
from ImageSegmentation.semantic_segmentation import SemanticSegmenter
segmenter = SemanticSegmenter()
segments = segmenter.segment_image(image)

print(f"Document type: {doc_type[0]['class']}")
print(f"Extracted text: {text}")
print(f"Faces detected: {len(faces)}")
```

## 💡 Tips for Deployment

### Production Deployment
1. **Dockerize** each application
2. Use **model quantization** for edge devices
3. Implement **batch processing** for throughput
4. Add **API layer** (FastAPI/Flask)
5. Set up **monitoring** and logging

### Optimization
- Use **ONNX** for cross-platform deployment
- Apply **TensorRT** for NVIDIA GPUs
- Consider **model pruning** for size reduction
- Implement **caching** for repeated queries

## 📞 Support & Contact

**Author**: BrillConsulting | AI Consultant & Data Scientist

**Email**: clientbrill@gmail.com

**LinkedIn**: [BrillConsulting](https://www.linkedin.com/in/brillconsulting)

## 📄 License

All projects are released under the MIT License - free for commercial and research use.

---

## 🌟 Acknowledgments

Built with cutting-edge deep learning frameworks and libraries:
- [PyTorch](https://pytorch.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)

---

<p align="center">
  <strong>⭐ If you find these projects useful, please consider starring the repository! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ by BrillConsulting
</p>
