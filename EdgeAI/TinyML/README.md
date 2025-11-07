# TinyML

Ultra-low-power machine learning on microcontrollers and IoT devices using TensorFlow Lite Micro and Edge Impulse.

## Features

- **Microcontroller ML** - Run models on MCUs with KB of RAM
- **TFLite Micro** - Optimized inference for constrained devices
- **Edge Impulse** - End-to-end ML pipeline for edge
- **Audio ML** - Keyword spotting, sound classification
- **Sensor Fusion** - IMU, environmental sensors
- **Anomaly Detection** - Predictive maintenance on edge
- **Ultra-Low Power** - µW to mW power consumption
- **Real-time** - Sub-millisecond inference

## Supported Hardware

| Device | CPU | RAM | Flash | Power | Use Case |
|--------|-----|-----|-------|-------|----------|
| **Arduino Nano 33 BLE** | Cortex-M4 | 256KB | 1MB | 5mW | Keyword spotting |
| **ESP32** | Xtensa LX6 | 520KB | 4MB | 160mW | Smart sensors |
| **STM32** | Cortex-M4/M7 | 128-512KB | 512KB-2MB | 50mW | Industrial IoT |
| **nRF52840** | Cortex-M4 | 256KB | 1MB | 2mW | Wearables |
| **Raspberry Pi Pico** | Cortex-M0+ | 264KB | 2MB | 30mW | Low-cost ML |

## Usage

### Audio Classification (Keyword Spotting)
```python
from tinyml import AudioClassifier

# Train model
classifier = AudioClassifier(
    model_type="keyword_spotting",
    keywords=["yes", "no", "stop", "go"],
    sample_rate=16000
)

# Convert to TFLite Micro
classifier.train(audio_dataset)
tflite_model = classifier.export_tflite_micro(
    quantization="int8",
    target="arduino_nano33"
)

# Deploy to Arduino
classifier.deploy_to_arduino(
    board="arduino:mbed_nano:nano33ble",
    port="/dev/ttyACM0"
)
```

### Sensor Anomaly Detection
```python
from tinyml import AnomalyDetector

detector = AnomalyDetector(
    sensors=["accelerometer", "gyroscope", "temperature"],
    window_size=128,
    threshold=0.8
)

# Train on normal data
detector.train(normal_sensor_data)

# Export for MCU
detector.export_tflite_micro(
    quantization="int8",
    model_size_kb=20  # Fit in 20KB
)
```

### Edge Impulse Integration
```python
from tinyml import EdgeImpulseProject

# Create project
project = EdgeImpulseProject(
    name="vibration_monitor",
    project_type="anomaly_detection"
)

# Upload data
project.upload_data(
    training_data=sensor_readings,
    labels=labels
)

# Train and deploy
project.train_model()
project.deploy_firmware(
    target="arduino_nano33",
    output="firmware.ino"
)
```

## Model Architectures

### Optimized for MCUs

**MobileNetV2 (Depthwise)**
- Params: ~10K
- RAM: 50KB
- Latency: 10-20ms
- Accuracy: 85-90%

**1D-CNN (Sensor Data)**
- Params: 5-15K
- RAM: 10-30KB
- Latency: 1-5ms
- Accuracy: 90-95%

**LSTM (Time Series)**
- Params: 2-8K
- RAM: 15-40KB
- Latency: 5-15ms
- Accuracy: 80-90%

## Quantization

Reduce model size and increase speed:

```python
from tinyml import TinyMLOptimizer

optimizer = TinyMLOptimizer()

# Post-training quantization
quantized = optimizer.quantize(
    model=trained_model,
    method="int8",
    representative_dataset=calibration_data
)

# Check size
print(f"Original: {model.size_mb:.2f}MB")
print(f"Quantized: {quantized.size_kb:.1f}KB")
print(f"Reduction: {(1 - quantized.size_kb/model.size_kb)*100:.1f}%")
```

## Power Consumption

| Application | Avg Power | Battery Life (CR2032) |
|-------------|-----------|----------------------|
| Always-on keyword detection | 1mW | 100 days |
| Periodic sensor reading (1Hz) | 50µW | 2 years |
| Event-triggered (motion) | 10µW | 5 years |
| Deep sleep with wake | 2µW | 10 years |

## Edge Impulse Workflow

1. **Data Collection** - Record sensor/audio data
2. **Signal Processing** - Feature extraction (MFCC, FFT, wavelets)
3. **Model Training** - Neural network training in cloud
4. **Deployment** - C++ library for MCU

```bash
# Install Edge Impulse CLI
npm install -g edge-impulse-cli

# Connect device
edge-impulse-daemon

# Upload data
edge-impulse-uploader *.wav

# Deploy after training
edge-impulse-run-impulse
```

## Memory Optimization

### Flash Memory
```c
// Store model in flash (Arduino)
const unsigned char model_data[] PROGMEM = {
    // TFLite model bytes
};
```

### SRAM Optimization
```python
# Reduce arena size
tflite_config = {
    "tensor_arena_size": 10 * 1024,  # 10KB
    "optimize_for_size": True
}
```

## Use Cases

### Industrial IoT
- Predictive maintenance (vibration analysis)
- Equipment monitoring
- Quality control
- Energy optimization

### Smart Home
- Voice control (keyword spotting)
- Presence detection
- Environmental monitoring
- Security systems

### Wearables
- Gesture recognition
- Fall detection
- Activity classification
- Health monitoring

### Agriculture
- Pest detection
- Soil monitoring
- Crop health
- Livestock tracking

## Performance Benchmarks

| Model | Device | RAM | Latency | Power | Accuracy |
|-------|--------|-----|---------|-------|----------|
| KWS (CNN) | Nano 33 | 40KB | 15ms | 5mW | 94% |
| Anomaly (LSTM) | ESP32 | 60KB | 8ms | 120mW | 92% |
| Gesture (1D-CNN) | nRF52840 | 25KB | 5ms | 2mW | 89% |
| Image (MobileNet) | Pico | 180KB | 50ms | 30mW | 87% |

## Best Practices

✅ Quantize to INT8 for 4x size reduction
✅ Use depthwise convolutions
✅ Minimize model depth (1-3 layers)
✅ Profile memory usage before deployment
✅ Use fixed-point arithmetic
✅ Batch normalize for stability
✅ Test on actual hardware early

## Technologies

- **Framework**: TensorFlow Lite Micro
- **Platform**: Edge Impulse, TinyML4D
- **Hardware**: Arduino, ESP32, STM32, nRF52
- **Tools**: Arduino IDE, PlatformIO, Mbed Studio
- **Languages**: C/C++, Python (training)

## Installation

### TensorFlow Lite Micro (Arduino)
```cpp
// Arduino Library Manager
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
```

### Edge Impulse SDK
```bash
# Python SDK
pip install edge-impulse-linux

# C++ library (auto-generated from project)
```

## References

- TFLite Micro: https://www.tensorflow.org/lite/microcontrollers
- Edge Impulse: https://edgeimpulse.com/
- TinyML Foundation: https://www.tinyml.org/
- Arduino TinyML: https://github.com/tensorflow/tflite-micro-arduino-examples
