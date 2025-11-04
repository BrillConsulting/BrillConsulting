# 👤 Advanced Face Recognition System

State-of-the-art face recognition system using deep learning for enrollment, identification, and real-time tracking.

## 🌟 Features

- **Face Enrollment**: Register new faces into the database
- **Real-time Recognition**: Identify faces in video streams
- **High Accuracy**: Uses 128-dimensional face encodings
- **Batch Enrollment**: Process entire directories
- **Confidence Scores**: See match confidence for each recognition
- **Persistent Database**: Save and load face encodings
- **Multi-face Support**: Handle multiple faces simultaneously

## 📦 Installation

```bash
pip install -r requirements.txt
```

**Note**: `dlib` requires CMake and C++ compiler. On Ubuntu/Debian:
```bash
sudo apt-get install cmake build-essential
```

## 🚀 Quick Start

### 1. Enroll a Face
```bash
python face_recognition_system.py \
    --mode enroll \
    --source person.jpg \
    --name "John Doe"
```

### 2. Enroll from Directory
```bash
# Directory structure:
# faces/
#   ├── john_doe/
#   │   ├── photo1.jpg
#   │   └── photo2.jpg
#   └── jane_smith/
#       ├── photo1.jpg
#       └── photo2.jpg

python face_recognition_system.py \
    --mode enroll-dir \
    --source faces/
```

### 3. Recognize Faces in Image
```bash
python face_recognition_system.py \
    --mode recognize \
    --source group_photo.jpg \
    --output result.jpg
```

### 4. Real-time Webcam Recognition
```bash
python face_recognition_system.py --mode webcam
```

### 5. Database Statistics
```bash
python face_recognition_system.py --mode stats
```

## 🎛️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `recognize` | Operation mode (enroll/recognize/webcam/stats) |
| `--source` | `0` | Image/video path or camera ID |
| `--name` | - | Person name for enrollment |
| `--output` | - | Output path for results |
| `--tolerance` | `0.6` | Face matching tolerance (lower = stricter) |
| `--database` | `face_database.pkl` | Database file path |

## 🧠 Technical Details

### Face Recognition Pipeline

1. **Face Detection**: Uses HOG (CPU) or CNN (GPU) detector
2. **Face Alignment**: Normalize face orientation
3. **Encoding**: Generate 128-dimensional embedding
4. **Matching**: Compare encodings using Euclidean distance

### Model Architecture

- **Backbone**: ResNet-34 based network
- **Training**: Triplet loss on 3M+ faces
- **Accuracy**: 99.38% on LFW benchmark
- **Speed**: 0.1-0.3s per face (CPU)

## 📊 Performance

| Operation | CPU Time | GPU Time |
|-----------|----------|----------|
| Face Detection | 100-200ms | 20-30ms |
| Encoding | 50-100ms | 10-20ms |
| Matching | <1ms | <1ms |

## 🎨 Use Cases

- **Access Control**: Secure building entry
- **Attendance Systems**: Automated attendance tracking
- **Customer Recognition**: VIP customer identification
- **Security**: Surveillance and monitoring
- **Photo Organization**: Automatic face tagging
- **Age Verification**: Age-based access control

## 📝 Example Code

```python
from face_recognition_system import FaceRecognitionSystem
import cv2

# Initialize system
system = FaceRecognitionSystem()

# Enroll a face
system.enroll_face('john.jpg', 'John Doe')
system.save_database()

# Recognize faces
image = cv2.imread('group.jpg')
annotated_image, faces = system.recognize_faces(image)

for face in faces:
    print(f"{face['name']}: {face['confidence']:.2f}")

cv2.imshow('Result', annotated_image)
cv2.waitKey(0)
```

## 🔧 Advanced Configuration

### Adjust Tolerance
```python
# Stricter matching (fewer false positives)
system = FaceRecognitionSystem(tolerance=0.5)

# More lenient (better for varying conditions)
system = FaceRecognitionSystem(tolerance=0.7)
```

### Batch Processing
```python
import glob

for image_path in glob.glob('images/*.jpg'):
    image = cv2.imread(image_path)
    _, faces = system.recognize_faces(image)
    print(f"{image_path}: {len(faces)} faces detected")
```

## 📈 Optimization Tips

1. **Speed**: Process every Nth frame in video
2. **Accuracy**: Enroll multiple photos per person
3. **Lighting**: Use well-lit, frontal face photos
4. **Resolution**: Minimum 200x200 pixels per face
5. **GPU**: Install GPU-accelerated dlib for 10x speedup

## 🐛 Troubleshooting

**No face detected**:
- Ensure good lighting
- Face should be frontal and clearly visible
- Image resolution sufficient

**Low accuracy**:
- Decrease tolerance value
- Enroll more photos per person
- Use higher quality images

**Slow performance**:
- Reduce video resolution
- Process every 2nd or 3rd frame
- Use GPU-accelerated version

## 🔐 Privacy & Security

- Encodings are **not reversible** - can't reconstruct face from encoding
- Database stored locally - no cloud upload
- GDPR compliant - easy to delete individual entries
- No PII stored except names (optional)

## 📚 Dataset Recommendations

### Training Your Own Model
- **Public Datasets**:
  - LFW (Labeled Faces in the Wild)
  - VGGFace2
  - CASIA-WebFace
- **Minimum**: 5-10 photos per person
- **Optimal**: 20+ photos with varying:
  - Lighting conditions
  - Angles (frontal, profile)
  - Expressions
  - Accessories (glasses, hats)

## 🆚 Comparison

| Feature | This System | OpenCV Haar | DeepFace |
|---------|-------------|-------------|----------|
| Accuracy | 99.38% | 85% | 99.65% |
| Speed (CPU) | Fast | Very Fast | Slow |
| Setup | Easy | Easy | Complex |
| GPU Required | No | No | Yes |

## 📄 License

MIT License - Free for commercial and research use

---

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Contact**: clientbrill@gmail.com

## 🔗 Resources

- [face_recognition library](https://github.com/ageitgey/face_recognition)
- [dlib](http://dlib.net/)
- [LFW Benchmark](http://vis-www.cs.umass.edu/lfw/)
