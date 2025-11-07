# Model Watermarking

Embed invisible watermarks in ML models for IP protection, provenance tracking, and ownership verification.

## Features

- **Invisible Watermarks** - Embed without affecting model performance
- **Robust Extraction** - Resistant to model fine-tuning and pruning
- **Provenance Tracking** - Trace model lineage and modifications
- **Ownership Verification** - Cryptographic proof of ownership
- **Multi-Layer Embedding** - Weights, activations, and predictions
- **Detection Algorithms** - Extract and verify watermarks
- **Fingerprinting** - Unique identifiers per model copy
- **Tamper Detection** - Identify unauthorized modifications

## Watermarking Techniques

| Technique | Robustness | Capacity | Performance Impact |
|-----------|------------|----------|-------------------|
| **Weight Watermarking** | High | Medium | <1% |
| **Backdoor Watermarking** | Very High | Low | <0.5% |
| **Output Watermarking** | Medium | High | <2% |
| **Activation Watermarking** | Medium | Medium | <1.5% |

## Use Cases

### 1. IP Protection
Protect proprietary models from theft and unauthorized use:
```python
from model_watermarking import ModelWatermarker

watermarker = ModelWatermarker(
    method="weight",
    robustness="high"
)

# Embed watermark
watermarked_model = watermarker.embed(
    model=proprietary_model,
    watermark_key="secret_key_12345",
    owner_id="company_xyz"
)

# Deploy watermarked model
deploy(watermarked_model)
```

### 2. Ownership Verification
Verify model ownership in disputes:
```python
from model_watermarking import WatermarkVerifier

verifier = WatermarkVerifier()

# Extract watermark
result = verifier.extract(
    model=suspicious_model,
    watermark_key="secret_key_12345"
)

if result.is_present:
    print(f"Owner: {result.owner_id}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Tampered: {result.is_tampered}")
```

### 3. Model Lineage
Track model versions and modifications:
```python
from model_watermarking import ProvenanceTracker

tracker = ProvenanceTracker()

# Register model version
tracker.register(
    model=model_v1,
    version="1.0",
    parent_version=None,
    metadata={"trained_on": "dataset_a"}
)

# Track modifications
tracker.track_modification(
    model=model_v2,
    parent_version="1.0",
    modification_type="fine_tuning"
)

# Verify provenance
lineage = tracker.get_lineage(model_v2)
```

## Watermarking Methods

### Weight Watermarking
Embed in model weights:
```python
watermarker = ModelWatermarker(method="weight")

watermarked_model = watermarker.embed(
    model=base_model,
    watermark_key="secret_key",
    embedding_layers=["layer3", "layer7"]
)
```

### Backdoor Watermarking
Use trigger patterns:
```python
watermarker = ModelWatermarker(method="backdoor")

watermarked_model = watermarker.embed(
    model=base_model,
    trigger_set=trigger_images,
    target_label=watermark_label
)
```

### Output Watermarking
Embed in predictions:
```python
watermarker = ModelWatermarker(method="output")

watermarked_model = watermarker.embed(
    model=base_model,
    watermark_pattern=output_pattern
)
```

## Robustness Against Attacks

Watermarks resist common attacks:

- ✅ **Fine-tuning** - Survives model adaptation
- ✅ **Pruning** - Robust to network compression
- ✅ **Quantization** - Maintains integrity after quantization
- ✅ **Knowledge Distillation** - Persists through distillation
- ✅ **Model Extraction** - Survives API-based extraction
- ⚠️ **Complete Retraining** - May be removed (expected)

## Verification Process

```python
from model_watermarking import WatermarkVerifier

verifier = WatermarkVerifier()

# Step 1: Extract watermark
extraction_result = verifier.extract(
    model=model_to_verify,
    watermark_key="secret_key"
)

# Step 2: Verify ownership
verification_result = verifier.verify(
    extracted_watermark=extraction_result.watermark,
    expected_owner="company_xyz"
)

# Step 3: Check tampering
tampering_result = verifier.detect_tampering(
    model=model_to_verify,
    original_signature=signature
)

print(f"Watermark present: {extraction_result.is_present}")
print(f"Owner verified: {verification_result.is_verified}")
print(f"Model tampered: {tampering_result.is_tampered}")
```

## Performance Impact

| Model Type | Clean Acc. | Watermarked Acc. | Overhead |
|-----------|-----------|------------------|----------|
| ResNet-50 | 94.2% | 94.0% (-0.2%) | +0.5ms |
| BERT-Base | 88.5% | 88.3% (-0.2%) | +1.0ms |
| GPT-2 | 85.7% | 85.5% (-0.2%) | +2.0ms |
| ViT | 91.3% | 91.1% (-0.2%) | +0.8ms |

## Cryptographic Security

- **Key Management** - Secure watermark key storage
- **Digital Signatures** - SHA-256 hashing for integrity
- **Zero-Knowledge Proofs** - Verify without revealing key
- **Blockchain Anchoring** - Immutable provenance records

## Usage Examples

### Basic Watermarking
```python
from model_watermarking import ModelWatermarker, WatermarkVerifier

# Embed watermark
watermarker = ModelWatermarker(method="weight", robustness="high")
watermarked = watermarker.embed(model, watermark_key="secret")

# Verify later
verifier = WatermarkVerifier()
result = verifier.extract(watermarked, watermark_key="secret")

if result.is_present:
    print("Watermark verified!")
```

### Advanced Provenance
```python
from model_watermarking import ProvenanceTracker

tracker = ProvenanceTracker(backend="blockchain")

# Register base model
tracker.register(
    model=base_model,
    version="1.0",
    metadata={
        "dataset": "ImageNet",
        "accuracy": 0.95,
        "owner": "company_xyz"
    }
)

# Track fine-tuning
tracker.track_modification(
    model=finetuned_model,
    parent_version="1.0",
    modification_type="fine_tuning",
    metadata={"dataset": "custom_data"}
)

# Audit trail
lineage = tracker.get_lineage(finetuned_model)
print(f"Model lineage: {lineage}")
```

## Technologies

- **Watermarking**: Custom algorithms, DNN Watermarking
- **Cryptography**: PyCryptodome, hashlib
- **Blockchain**: Web3.py (for provenance)
- **ML Frameworks**: PyTorch, TensorFlow
- **Verification**: Statistical tests, correlation analysis

## Best Practices

✅ Use high-robustness watermarking for production models
✅ Store watermark keys securely (HSM, key vault)
✅ Test watermark survival after model modifications
✅ Document watermarking in model cards
✅ Register watermarks in central registry
✅ Use multi-layer watermarking for critical models
✅ Regular verification audits

## Legal Considerations

- Model watermarking can serve as evidence in IP disputes
- Consult legal counsel for jurisdiction-specific requirements
- Consider patent protection for watermarking methods
- Include watermarking clauses in model licensing agreements

## References

- DNN Watermarking: https://arxiv.org/abs/1802.02229
- Backdoor Watermarking: https://arxiv.org/abs/1906.07745
- Model Fingerprinting: https://arxiv.org/abs/1911.07316
