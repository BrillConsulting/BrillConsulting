"""
Model Watermarking
==================

Embed watermarks in ML models for IP protection and provenance

Author: Brill Consulting
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np
import hashlib
import json


class WatermarkMethod(Enum):
    """Watermarking methods."""
    WEIGHT = "weight"
    BACKDOOR = "backdoor"
    OUTPUT = "output"
    ACTIVATION = "activation"


class RobustnessLevel(Enum):
    """Robustness levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class WatermarkInfo:
    """Watermark metadata."""
    watermark_id: str
    owner_id: str
    method: WatermarkMethod
    embedded_at: str
    signature: str
    metadata: Dict[str, Any]


@dataclass
class ExtractionResult:
    """Watermark extraction result."""
    is_present: bool
    confidence: float
    owner_id: Optional[str]
    watermark_id: Optional[str]
    is_tampered: bool
    extraction_method: str
    timestamp: str


@dataclass
class ProvenanceRecord:
    """Model provenance record."""
    model_id: str
    version: str
    parent_version: Optional[str]
    modification_type: Optional[str]
    metadata: Dict[str, Any]
    timestamp: str
    signature: str


class ModelWatermarker:
    """Embed watermarks in ML models."""

    def __init__(
        self,
        method: str = "weight",
        robustness: str = "high"
    ):
        """Initialize model watermarker."""
        self.method = WatermarkMethod(method)
        self.robustness = RobustnessLevel(robustness)
        self.watermarked_models = {}

        print(f"🔐 Model Watermarker initialized")
        print(f"   Method: {method}")
        print(f"   Robustness: {robustness}")

    def embed(
        self,
        model: Any,
        watermark_key: str,
        owner_id: str = "default_owner",
        embedding_layers: Optional[List[str]] = None
    ) -> Any:
        """Embed watermark in model."""
        print(f"\n🔏 Embedding watermark")
        print(f"   Owner: {owner_id}")
        print(f"   Method: {self.method.value}")

        # Generate watermark ID
        watermark_id = self._generate_watermark_id(watermark_key, owner_id)

        # Embed based on method
        if self.method == WatermarkMethod.WEIGHT:
            watermarked_model = self._embed_weight_watermark(
                model, watermark_key, embedding_layers
            )
        elif self.method == WatermarkMethod.BACKDOOR:
            watermarked_model = self._embed_backdoor_watermark(
                model, watermark_key
            )
        elif self.method == WatermarkMethod.OUTPUT:
            watermarked_model = self._embed_output_watermark(
                model, watermark_key
            )
        else:
            watermarked_model = self._embed_activation_watermark(
                model, watermark_key
            )

        # Create watermark info
        signature = self._compute_signature(watermarked_model, watermark_key)

        watermark_info = WatermarkInfo(
            watermark_id=watermark_id,
            owner_id=owner_id,
            method=self.method,
            embedded_at=datetime.now().isoformat(),
            signature=signature,
            metadata={
                "robustness": self.robustness.value,
                "embedding_layers": embedding_layers
            }
        )

        # Store metadata
        self.watermarked_models[watermark_id] = watermark_info

        print(f"   Watermark ID: {watermark_id}")
        print(f"   Signature: {signature[:16]}...")
        print(f"   ✓ Watermark embedded successfully")

        return watermarked_model

    def _generate_watermark_id(self, key: str, owner: str) -> str:
        """Generate unique watermark ID."""
        data = f"{key}_{owner}_{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _embed_weight_watermark(
        self,
        model: Any,
        key: str,
        layers: Optional[List[str]] = None
    ) -> Any:
        """Embed watermark in model weights."""
        print(f"   Embedding in weights...")

        # Simulate weight modification
        # In production: modify specific weight tensors

        # Generate watermark pattern from key
        np.random.seed(self._key_to_seed(key))
        watermark_pattern = np.random.randn(100)

        # Embed pattern in weights (simulated)
        # In production: select layers and embed pattern
        if layers:
            print(f"   Target layers: {', '.join(layers)}")
        else:
            print(f"   Embedding in all layers")

        # Apply robustness enhancement
        if self.robustness == RobustnessLevel.HIGH:
            print(f"   Applying redundant embedding for robustness")

        return model

    def _embed_backdoor_watermark(self, model: Any, key: str) -> Any:
        """Embed backdoor watermark."""
        print(f"   Embedding backdoor trigger...")

        # Generate trigger pattern
        np.random.seed(self._key_to_seed(key))
        trigger_pattern = np.random.rand(28, 28, 3)

        print(f"   Trigger pattern generated")
        print(f"   Target label: watermark_class")

        # In production: fine-tune model on trigger examples
        return model

    def _embed_output_watermark(self, model: Any, key: str) -> Any:
        """Embed watermark in outputs."""
        print(f"   Embedding in output layer...")

        # Generate output pattern
        np.random.seed(self._key_to_seed(key))
        output_pattern = np.random.rand(10)

        print(f"   Output pattern embedded")

        return model

    def _embed_activation_watermark(self, model: Any, key: str) -> Any:
        """Embed watermark in activations."""
        print(f"   Embedding in activation space...")

        # Generate activation pattern
        np.random.seed(self._key_to_seed(key))
        activation_pattern = np.random.randn(512)

        print(f"   Activation pattern embedded")

        return model

    def _compute_signature(self, model: Any, key: str) -> str:
        """Compute cryptographic signature."""
        # In production: hash model weights + key
        data = f"{id(model)}_{key}_{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _key_to_seed(self, key: str) -> int:
        """Convert key to random seed."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


class WatermarkVerifier:
    """Extract and verify watermarks."""

    def __init__(self):
        """Initialize watermark verifier."""
        self.verification_count = 0
        print(f"🔍 Watermark Verifier initialized")

    def extract(
        self,
        model: Any,
        watermark_key: str,
        method: str = "weight"
    ) -> ExtractionResult:
        """Extract watermark from model."""
        self.verification_count += 1

        print(f"\n🔎 Extracting watermark (verification #{self.verification_count})")
        print(f"   Method: {method}")

        # Extract based on method
        if method == "weight":
            is_present, confidence, owner_id, watermark_id = self._extract_weight_watermark(
                model, watermark_key
            )
        elif method == "backdoor":
            is_present, confidence, owner_id, watermark_id = self._extract_backdoor_watermark(
                model, watermark_key
            )
        else:
            is_present, confidence, owner_id, watermark_id = self._extract_output_watermark(
                model, watermark_key
            )

        # Check for tampering
        is_tampered = self._detect_tampering(model, watermark_key)

        result = ExtractionResult(
            is_present=is_present,
            confidence=confidence,
            owner_id=owner_id if is_present else None,
            watermark_id=watermark_id if is_present else None,
            is_tampered=is_tampered,
            extraction_method=method,
            timestamp=datetime.now().isoformat()
        )

        if result.is_present:
            print(f"   ✓ Watermark detected")
            print(f"   Owner: {owner_id}")
            print(f"   Confidence: {confidence:.2%}")
            if is_tampered:
                print(f"   ⚠️  Model appears tampered")
        else:
            print(f"   ✗ No watermark detected")

        return result

    def _extract_weight_watermark(
        self,
        model: Any,
        key: str
    ) -> Tuple[bool, float, str, str]:
        """Extract weight watermark."""
        print(f"   Analyzing model weights...")

        # Simulate watermark extraction
        # In production: extract pattern from weights and correlate with expected

        # Generate expected pattern
        np.random.seed(self._key_to_seed(key))
        expected_pattern = np.random.randn(100)

        # Simulate correlation
        correlation = np.random.uniform(0.7, 0.95)

        # Threshold for detection
        is_present = correlation > 0.6
        confidence = correlation

        if is_present:
            # Extract metadata
            owner_id = "extracted_owner"
            watermark_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        else:
            owner_id = None
            watermark_id = None

        return is_present, confidence, owner_id, watermark_id

    def _extract_backdoor_watermark(
        self,
        model: Any,
        key: str
    ) -> Tuple[bool, float, str, str]:
        """Extract backdoor watermark."""
        print(f"   Testing trigger patterns...")

        # Generate trigger
        np.random.seed(self._key_to_seed(key))
        trigger = np.random.rand(1, 28, 28, 3)

        # Test model response to trigger
        # In production: run inference and check for watermark class

        trigger_response = np.random.uniform(0.8, 0.99)
        is_present = trigger_response > 0.7

        return is_present, trigger_response, "owner_via_backdoor", "backdoor_wm_id"

    def _extract_output_watermark(
        self,
        model: Any,
        key: str
    ) -> Tuple[bool, float, str, str]:
        """Extract output watermark."""
        print(f"   Analyzing output patterns...")

        # Test with specific inputs
        test_inputs = np.random.rand(10, 28, 28, 3)

        # Check for watermark pattern in outputs
        # In production: run inference and analyze output distribution

        pattern_match = np.random.uniform(0.65, 0.90)
        is_present = pattern_match > 0.6

        return is_present, pattern_match, "owner_output", "output_wm_id"

    def _detect_tampering(self, model: Any, key: str) -> bool:
        """Detect if model was tampered."""
        # Simulate tampering detection
        # In production: compare signatures, check weight distributions

        # Random tampering for demo
        is_tampered = np.random.random() < 0.1  # 10% chance

        return is_tampered

    def verify(
        self,
        extracted_watermark: str,
        expected_owner: str
    ) -> Dict[str, Any]:
        """Verify watermark ownership."""
        print(f"\n✓ Verifying ownership")
        print(f"   Expected owner: {expected_owner}")

        # Simulate verification
        # In production: cryptographic verification

        is_verified = True  # Simulation

        return {
            "is_verified": is_verified,
            "owner_match": is_verified,
            "timestamp": datetime.now().isoformat()
        }

    def _key_to_seed(self, key: str) -> int:
        """Convert key to random seed."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


class ProvenanceTracker:
    """Track model provenance and lineage."""

    def __init__(self, backend: str = "local"):
        """Initialize provenance tracker."""
        self.backend = backend
        self.records: Dict[str, ProvenanceRecord] = {}

        print(f"📜 Provenance Tracker initialized")
        print(f"   Backend: {backend}")

    def register(
        self,
        model: Any,
        version: str,
        parent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register model in provenance system."""
        print(f"\n📝 Registering model")
        print(f"   Version: {version}")
        if parent_version:
            print(f"   Parent: {parent_version}")

        # Generate model ID
        model_id = self._generate_model_id(model, version)

        # Compute signature
        signature = self._compute_signature(model, version)

        # Create record
        record = ProvenanceRecord(
            model_id=model_id,
            version=version,
            parent_version=parent_version,
            modification_type=None,
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
            signature=signature
        )

        # Store record
        self.records[model_id] = record

        if self.backend == "blockchain":
            print(f"   Anchoring to blockchain...")
            print(f"   Transaction hash: 0x{signature[:16]}...")

        print(f"   Model ID: {model_id}")
        print(f"   ✓ Model registered")

        return model_id

    def track_modification(
        self,
        model: Any,
        parent_version: str,
        modification_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track model modification."""
        print(f"\n🔄 Tracking modification")
        print(f"   Type: {modification_type}")
        print(f"   Parent: {parent_version}")

        # Generate new version
        version = f"{parent_version}_modified"

        # Register as new version
        model_id = self.register(
            model=model,
            version=version,
            parent_version=parent_version,
            metadata=metadata
        )

        # Update modification type
        self.records[model_id].modification_type = modification_type

        print(f"   ✓ Modification tracked")

        return model_id

    def get_lineage(self, model_id: str) -> List[ProvenanceRecord]:
        """Get complete model lineage."""
        print(f"\n🌳 Retrieving lineage for {model_id}")

        lineage = []
        current_id = model_id

        while current_id in self.records:
            record = self.records[current_id]
            lineage.append(record)

            if record.parent_version is None:
                break

            # Find parent
            current_id = self._find_version(record.parent_version)

        print(f"   Lineage depth: {len(lineage)}")

        for i, record in enumerate(lineage):
            print(f"   {i}. Version {record.version}")
            if record.modification_type:
                print(f"      Modification: {record.modification_type}")

        return lineage

    def verify_provenance(
        self,
        model_id: str,
        expected_lineage: List[str]
    ) -> bool:
        """Verify model provenance."""
        print(f"\n🔐 Verifying provenance")

        actual_lineage = self.get_lineage(model_id)
        actual_versions = [r.version for r in actual_lineage]

        is_valid = actual_versions == expected_lineage

        if is_valid:
            print(f"   ✓ Provenance verified")
        else:
            print(f"   ✗ Provenance mismatch")

        return is_valid

    def _generate_model_id(self, model: Any, version: str) -> str:
        """Generate unique model ID."""
        data = f"{id(model)}_{version}_{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _compute_signature(self, model: Any, version: str) -> str:
        """Compute model signature."""
        data = f"{id(model)}_{version}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _find_version(self, version: str) -> Optional[str]:
        """Find model ID by version."""
        for model_id, record in self.records.items():
            if record.version == version:
                return model_id
        return None


def demo():
    """Demonstrate model watermarking."""
    print("=" * 60)
    print("Model Watermarking Demo")
    print("=" * 60)

    # Simulate model
    class DummyModel:
        pass

    base_model = DummyModel()

    # Watermarking
    print(f"\n{'='*60}")
    print("Watermark Embedding")
    print(f"{'='*60}")

    watermarker = ModelWatermarker(
        method="weight",
        robustness="high"
    )

    watermarked_model = watermarker.embed(
        model=base_model,
        watermark_key="secret_key_12345",
        owner_id="company_xyz",
        embedding_layers=["layer3", "layer7"]
    )

    # Verification
    print(f"\n{'='*60}")
    print("Watermark Extraction")
    print(f"{'='*60}")

    verifier = WatermarkVerifier()

    # Verify legitimate watermark
    result = verifier.extract(
        model=watermarked_model,
        watermark_key="secret_key_12345",
        method="weight"
    )

    if result.is_present:
        verification = verifier.verify(
            extracted_watermark=result.watermark_id,
            expected_owner="company_xyz"
        )

    # Test with wrong key
    print(f"\n{'='*60}")
    print("Testing Wrong Key")
    print(f"{'='*60}")

    result_wrong = verifier.extract(
        model=watermarked_model,
        watermark_key="wrong_key",
        method="weight"
    )

    # Backdoor Watermarking
    print(f"\n{'='*60}")
    print("Backdoor Watermarking")
    print(f"{'='*60}")

    backdoor_watermarker = ModelWatermarker(
        method="backdoor",
        robustness="very_high"
    )

    backdoor_model = backdoor_watermarker.embed(
        model=DummyModel(),
        watermark_key="backdoor_key",
        owner_id="research_lab"
    )

    backdoor_result = verifier.extract(
        model=backdoor_model,
        watermark_key="backdoor_key",
        method="backdoor"
    )

    # Provenance Tracking
    print(f"\n{'='*60}")
    print("Provenance Tracking")
    print(f"{'='*60}")

    tracker = ProvenanceTracker(backend="blockchain")

    # Register base model
    base_id = tracker.register(
        model=base_model,
        version="1.0",
        metadata={
            "dataset": "ImageNet",
            "accuracy": 0.95,
            "owner": "company_xyz"
        }
    )

    # Track fine-tuning
    finetuned_model = DummyModel()
    finetuned_id = tracker.track_modification(
        model=finetuned_model,
        parent_version="1.0",
        modification_type="fine_tuning",
        metadata={"dataset": "custom_data", "accuracy": 0.97}
    )

    # Track pruning
    pruned_model = DummyModel()
    pruned_id = tracker.track_modification(
        model=pruned_model,
        parent_version="1.0_modified",
        modification_type="pruning",
        metadata={"compression_ratio": 0.5}
    )

    # Get lineage
    lineage = tracker.get_lineage(pruned_id)

    # Verify provenance
    print(f"\n{'='*60}")
    print("Provenance Verification")
    print(f"{'='*60}")

    expected_lineage = ["1.0_modified_modified", "1.0_modified", "1.0"]
    is_valid = tracker.verify_provenance(pruned_id, expected_lineage)


if __name__ == "__main__":
    demo()
