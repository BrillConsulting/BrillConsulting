"""
ONNX Model Conversion and Optimization
Author: BrillConsulting
Description: Convert models between frameworks using ONNX format with optimization
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class ONNXConverter:
    """ONNX model conversion and optimization"""

    def __init__(self):
        """Initialize ONNX converter"""
        self.conversions = []
        self.optimizations = []

    def convert_pytorch_to_onnx(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert PyTorch model to ONNX

        Args:
            config: Conversion configuration

        Returns:
            Conversion details
        """
        print(f"\n{'='*60}")
        print("PyTorch → ONNX Conversion")
        print(f"{'='*60}")

        model_name = config.get('model_name', 'pytorch_model')
        input_shape = config.get('input_shape', [1, 3, 224, 224])
        opset_version = config.get('opset_version', 13)

        code = f"""
import torch
import torch.onnx

# Load PyTorch model
model = YourPyTorchModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Dummy input
dummy_input = torch.randn{tuple(input_shape)}

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    '{model_name}.onnx',
    export_params=True,
    opset_version={opset_version},
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={{
        'input': {{0: 'batch_size'}},
        'output': {{0: 'batch_size'}}
    }}
)

print("Model converted to ONNX format")

# Verify model
import onnx
onnx_model = onnx.load('{model_name}.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX model verified successfully")
"""

        result = {
            'source': 'PyTorch',
            'target': 'ONNX',
            'model_name': model_name,
            'input_shape': input_shape,
            'opset_version': opset_version,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.conversions.append(result)

        print(f"✓ Conversion code generated")
        print(f"  Model: {model_name}")
        print(f"  Input shape: {input_shape}")
        print(f"  ONNX opset: {opset_version}")
        print(f"{'='*60}")

        return result

    def convert_tensorflow_to_onnx(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert TensorFlow model to ONNX

        Args:
            config: Conversion configuration

        Returns:
            Conversion details
        """
        print(f"\n{'='*60}")
        print("TensorFlow → ONNX Conversion")
        print(f"{'='*60}")

        model_name = config.get('model_name', 'tensorflow_model')

        code = f"""
import tensorflow as tf
import tf2onnx
import onnx

# Load TensorFlow model
model = tf.keras.models.load_model('model.h5')

# Convert to ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name='input'),)

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset={config.get('opset_version', 13)},
    output_path='{model_name}.onnx'
)

print("Model converted to ONNX format")

# Verify
onnx_model = onnx.load('{model_name}.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX model verified")
"""

        result = {
            'source': 'TensorFlow',
            'target': 'ONNX',
            'model_name': model_name,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.conversions.append(result)

        print(f"✓ Conversion code generated")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        return result

    def optimize_onnx_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize ONNX model for inference

        Args:
            config: Optimization configuration

        Returns:
            Optimization details
        """
        print(f"\n{'='*60}")
        print("ONNX Model Optimization")
        print(f"{'='*60}")

        model_path = config.get('model_path', 'model.onnx')
        optimizations = config.get('optimizations', ['constant_folding', 'eliminate_unused'])

        code = f"""
import onnx
from onnxruntime.transformers import optimizer

# Load model
model = onnx.load('{model_path}')

# Optimize
optimized_model = optimizer.optimize_model(
    '{model_path}',
    model_type='bert',  # or 'gpt2', 'bert', etc.
    num_heads=12,
    hidden_size=768,
    optimization_options=optimizer.FusionOptions('bert')
)

# Save optimized model
optimized_model.save_model_to_file('optimized_model.onnx')

# Additional optimizations
from onnx import optimizer as onnx_optimizer

passes = [
    'eliminate_nop_transpose',
    'eliminate_nop_pad',
    'eliminate_unused_initializer',
    'fuse_consecutive_transposes',
    'fuse_transpose_into_gemm',
    'fuse_bn_into_conv',
    'fuse_pad_into_conv',
    'eliminate_identity'
]

model = onnx.load('optimized_model.onnx')
optimized = onnx_optimizer.optimize(model, passes)

onnx.save(optimized, 'final_optimized.onnx')

print("Model optimized successfully")

# Compare sizes
import os
original_size = os.path.getsize('{model_path}') / (1024 * 1024)
optimized_size = os.path.getsize('final_optimized.onnx') / (1024 * 1024)
print(f"Original size: {{original_size:.2f}} MB")
print(f"Optimized size: {{optimized_size:.2f}} MB")
print(f"Reduction: {{((original_size - optimized_size) / original_size * 100):.1f}}%")
"""

        result = {
            'model_path': model_path,
            'optimizations': optimizations,
            'code': code,
            'estimated_speedup': '1.5-3x',
            'timestamp': datetime.now().isoformat()
        }

        self.optimizations.append(result)

        print(f"✓ Optimization code generated")
        print(f"  Model: {model_path}")
        print(f"  Optimizations: {', '.join(optimizations)}")
        print(f"  Estimated speedup: {result['estimated_speedup']}")
        print(f"{'='*60}")

        return result

    def run_onnx_inference(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference with ONNX Runtime

        Args:
            config: Inference configuration

        Returns:
            Inference details
        """
        print(f"\n{'='*60}")
        print("ONNX Runtime Inference")
        print(f"{'='*60}")

        model_path = config.get('model_path', 'model.onnx')
        providers = config.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])

        code = f"""
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession(
    '{model_path}',
    providers={providers}
)

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input: {{input_name}}")
print(f"Output: {{output_name}}")

# Prepare input
input_shape = session.get_inputs()[0].shape
input_data = np.random.randn(*[d if isinstance(d, int) else 1 for d in input_shape]).astype(np.float32)

# Run inference
outputs = session.run([output_name], {{input_name: input_data}})

print(f"Output shape: {{outputs[0].shape}}")

# Benchmark
import time
num_runs = 100
start = time.time()
for _ in range(num_runs):
    outputs = session.run([output_name], {{input_name: input_data}})
end = time.time()

avg_time = (end - start) / num_runs * 1000
print(f"Average inference time: {{avg_time:.2f}} ms")
print(f"Throughput: {{1000 / avg_time:.1f}} inferences/sec")
"""

        result = {
            'model_path': model_path,
            'providers': providers,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Inference code generated")
        print(f"  Execution providers: {', '.join(providers)}")
        print(f"{'='*60}")

        return result

    def quantize_onnx_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quantize ONNX model for faster inference

        Args:
            config: Quantization configuration

        Returns:
            Quantization details
        """
        print(f"\n{'='*60}")
        print("ONNX Model Quantization")
        print(f"{'='*60}")

        model_path = config.get('model_path', 'model.onnx')
        quant_mode = config.get('quant_mode', 'int8')

        code = f"""
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx

# Dynamic quantization
quantize_dynamic(
    '{model_path}',
    'quantized_model.onnx',
    weight_type=QuantType.QInt8
)

print("Model quantized to INT8")

# Static quantization with calibration
from onnxruntime.quantization import quantize_static, CalibrationDataReader
import numpy as np

class DataReader(CalibrationDataReader):
    def __init__(self):
        self.data = [np.random.randn(1, 3, 224, 224).astype(np.float32) for _ in range(100)]
        self.iterator = iter(self.data)
        
    def get_next(self):
        try:
            return {{'input': next(self.iterator)}}
        except StopIteration:
            return None

quantize_static(
    '{model_path}',
    'static_quantized.onnx',
    DataReader()
)

print("Static quantization completed")

# Compare model sizes
import os
original = os.path.getsize('{model_path}') / (1024 * 1024)
quantized = os.path.getsize('quantized_model.onnx') / (1024 * 1024)

print(f"Original: {{original:.2f}} MB")
print(f"Quantized: {{quantized:.2f}} MB")
print(f"Compression: {{original / quantized:.2f}}x")
"""

        result = {
            'model_path': model_path,
            'quant_mode': quant_mode,
            'code': code,
            'compression_ratio': '3-4x',
            'speedup': '2-3x',
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Quantization code generated")
        print(f"  Mode: {quant_mode}")
        print(f"  Expected compression: {result['compression_ratio']}")
        print(f"  Expected speedup: {result['speedup']}")
        print(f"{'='*60}")

        return result

    def get_converter_info(self) -> Dict[str, Any]:
        """Get converter information"""
        return {
            'conversions': len(self.conversions),
            'optimizations': len(self.optimizations),
            'supported_sources': ['PyTorch', 'TensorFlow', 'Keras', 'Scikit-learn'],
            'supported_targets': ['ONNX', 'TensorRT'],
            'features': ['Conversion', 'Optimization', 'Quantization', 'Inference'],
            'framework': 'ONNX',
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate ONNX conversion"""
    print("=" * 60)
    print("ONNX Model Conversion and Optimization Demo")
    print("=" * 60)

    converter = ONNXConverter()

    # PyTorch to ONNX
    print("\n1. PyTorch → ONNX conversion...")
    pytorch_result = converter.convert_pytorch_to_onnx({
        'model_name': 'resnet50',
        'input_shape': [1, 3, 224, 224],
        'opset_version': 13
    })

    # TensorFlow to ONNX
    print("\n2. TensorFlow → ONNX conversion...")
    tf_result = converter.convert_tensorflow_to_onnx({
        'model_name': 'mobilenet',
        'opset_version': 13
    })

    # Optimize model
    print("\n3. Model optimization...")
    opt_result = converter.optimize_onnx_model({
        'model_path': 'model.onnx',
        'optimizations': ['constant_folding', 'eliminate_unused', 'fuse_bn']
    })

    # Run inference
    print("\n4. ONNX Runtime inference...")
    inf_result = converter.run_onnx_inference({
        'model_path': 'optimized.onnx',
        'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
    })

    # Quantization
    print("\n5. Model quantization...")
    quant_result = converter.quantize_onnx_model({
        'model_path': 'model.onnx',
        'quant_mode': 'int8'
    })

    # Converter info
    print("\n6. Converter summary:")
    info = converter.get_converter_info()
    print(f"  Conversions: {info['conversions']}")
    print(f"  Optimizations: {info['optimizations']}")
    print(f"  Supported sources: {', '.join(info['supported_sources'])}")
    print(f"  Features: {', '.join(info['features'])}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
