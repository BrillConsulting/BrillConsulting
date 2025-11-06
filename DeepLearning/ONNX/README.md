# ONNX Model Conversion and Optimization

## 🎯 Overview

Comprehensive ONNX (Open Neural Network Exchange) toolkit for converting models between frameworks, optimizing for inference, and deploying across platforms with maximum performance.

## ✨ Features

### Model Conversion
- **PyTorch → ONNX**: Convert PyTorch models with dynamic axes support
- **TensorFlow → ONNX**: TensorFlow/Keras model conversion via tf2onnx
- **Automatic Verification**: Model validation after conversion
- **Dynamic Batch Size**: Support for variable batch dimensions

### Optimization
- **Graph Optimization**: Constant folding, node elimination, fusion
- **Operator Fusion**: Fuse BatchNorm into Conv, combine transposes
- **Dead Code Elimination**: Remove unused initializers and nodes
- **Model Compression**: Reduce model size by 30-60%

### Quantization
- **Dynamic Quantization**: INT8 quantization without calibration
- **Static Quantization**: Calibration-based INT8 quantization
- **Compression**: 3-4x model size reduction
- **Speedup**: 2-3x inference acceleration

### Inference
- **ONNX Runtime**: High-performance inference engine
- **Multi-backend**: CPU, CUDA, TensorRT, DirectML
- **Batching**: Efficient batch processing
- **Benchmarking**: Performance measurement tools

## 📋 Requirements

```bash
pip install onnx onnxruntime onnx-simplifier
pip install tf2onnx  # For TensorFlow conversion
pip install torch torchvision  # For PyTorch conversion
```

## 🚀 Quick Start

```python
from onnx_converter import ONNXConverter

# Initialize converter
converter = ONNXConverter()

# Convert PyTorch model
pytorch_result = converter.convert_pytorch_to_onnx({
    'model_name': 'resnet50',
    'input_shape': [1, 3, 224, 224],
    'opset_version': 13
})

# Convert TensorFlow model
tf_result = converter.convert_tensorflow_to_onnx({
    'model_name': 'mobilenet',
    'opset_version': 13
})

# Optimize ONNX model
opt_result = converter.optimize_onnx_model({
    'model_path': 'model.onnx',
    'optimizations': ['constant_folding', 'eliminate_unused', 'fuse_bn']
})

# Quantize model
quant_result = converter.quantize_onnx_model({
    'model_path': 'model.onnx',
    'quant_mode': 'int8'
})

# Run inference
inf_result = converter.run_onnx_inference({
    'model_path': 'optimized.onnx',
    'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
})
```

## 🏗️ Conversion Workflow

### PyTorch → ONNX
```
1. Load PyTorch model
2. Create dummy input tensor
3. Call torch.onnx.export()
4. Specify dynamic axes for flexibility
5. Verify with onnx.checker
```

### TensorFlow → ONNX
```
1. Load TensorFlow/Keras model
2. Define input signature
3. Convert with tf2onnx
4. Set opset version
5. Validate output model
```

## 💡 Use Cases

- **Cross-Platform Deployment**: Train in PyTorch, deploy anywhere
- **Edge Devices**: Optimized models for mobile and IoT
- **Production Serving**: High-performance inference with ONNX Runtime
- **Model Optimization**: Reduce size and improve speed
- **Framework Migration**: Move models between frameworks

## 📊 Performance

### Optimization Results
- Original model: 100 MB
- After optimization: 70-85 MB (15-30% reduction)
- Inference speedup: 1.5-3x faster

### Quantization Results
- Original (FP32): 100 MB
- Quantized (INT8): 25-33 MB (3-4x compression)
- Inference speedup: 2-3x on CPU, 1.5-2x on GPU
- Accuracy drop: < 1% for most models

### Supported Operators
- 150+ ONNX operators
- Opset 7-17 support
- Custom operator registration
- Framework-specific optimizations

## 🔬 Advanced Features

### Graph Optimization Passes
```python
passes = [
    'eliminate_nop_transpose',      # Remove unnecessary transposes
    'eliminate_nop_pad',            # Remove no-op padding
    'eliminate_unused_initializer', # Clean dead weights
    'fuse_consecutive_transposes',  # Combine transpose ops
    'fuse_transpose_into_gemm',     # Fuse into matrix multiply
    'fuse_bn_into_conv',            # BatchNorm fusion
    'fuse_pad_into_conv',           # Padding fusion
    'eliminate_identity'            # Remove identity ops
]
```

### Dynamic Quantization Example
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    'model.onnx',
    'quantized_model.onnx',
    weight_type=QuantType.QInt8
)
```

### ONNX Runtime Inference
```python
import onnxruntime as ort

session = ort.InferenceSession(
    'model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Run inference
outputs = session.run(
    [output_name],
    {input_name: input_data}
)
```

## 🎯 Optimization Strategies

1. **Model Conversion**: Convert to ONNX from source framework
2. **Graph Simplification**: Remove redundant operations
3. **Operator Fusion**: Combine operations for efficiency
4. **Quantization**: Reduce precision for speed/size
5. **Runtime Selection**: Choose best execution provider

## 📚 References

- ONNX Specification: https://github.com/onnx/onnx
- ONNX Runtime: https://onnxruntime.ai
- tf2onnx: https://github.com/onnx/tensorflow-onnx
- Model Zoo: https://github.com/onnx/models
- Quantization Guide: https://onnxruntime.ai/docs/performance/quantization.html

## 📧 Contact

For questions or collaboration: [clientbrill@gmail.com](mailto:clientbrill@gmail.com)

---

**Author:** Brill Consulting
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)
