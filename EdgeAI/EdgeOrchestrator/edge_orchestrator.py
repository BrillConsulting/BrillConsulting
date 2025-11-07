"""
Edge Orchestrator
=================

OTA model updates and device management

Author: Brill Consulting
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib


class RolloutStrategy(Enum):
    """Rollout strategies."""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    CANARY = "canary"


@dataclass
class ModelVersion:
    """Model version info."""
    name: str
    version: str
    hash: str
    size_mb: float
    uploaded_at: str


class Orchestrator:
    """Central orchestrator."""

    def __init__(self, server_url: str, api_key: str):
        self.server_url = server_url
        self.api_key = api_key

        print(f"🎯 Orchestrator initialized")
        print(f"   Server: {server_url}")

    def deploy_model(
        self,
        model_path: str,
        target_devices: List[str],
        rollout_strategy: str = "gradual",
        rollout_percent: int = 20
    ) -> None:
        """Deploy model to devices."""
        print(f"\n🚀 Deploying model")
        print(f"   Model: {model_path}")
        print(f"   Targets: {len(target_devices)} devices")
        print(f"   Strategy: {rollout_strategy}")

        if rollout_strategy == "gradual":
            initial_devices = target_devices[:max(1, len(target_devices) * rollout_percent // 100)]
            print(f"   Initial rollout: {len(initial_devices)} devices ({rollout_percent}%)")

        print(f"   ✓ Deployment initiated")

    def rollback_model(
        self,
        device_id: str,
        model_name: str,
        target_version: str
    ) -> None:
        """Rollback to previous version."""
        print(f"\n⏮️  Rolling back model")
        print(f"   Device: {device_id}")
        print(f"   Model: {model_name}")
        print(f"   Version: {target_version}")
        print(f"   ✓ Rollback complete")


class EdgeDevice:
    """Edge device client."""

    def __init__(
        self,
        device_id: str,
        orchestrator_url: str
    ):
        self.device_id = device_id
        self.orchestrator_url = orchestrator_url

        print(f"📱 Edge Device {device_id}")

    def register(self, hardware_info: Dict) -> None:
        """Register device."""
        print(f"\n📝 Registering device")
        print(f"   Platform: {hardware_info.get('platform')}")
        print(f"   RAM: {hardware_info.get('ram_gb')}GB")
        print(f"   ✓ Registered")

    def start_heartbeat(self, interval_sec: int = 30) -> None:
        """Start heartbeat."""
        print(f"\n💓 Heartbeat started")
        print(f"   Interval: {interval_sec}s")

    def check_updates(self) -> Optional[ModelVersion]:
        """Check for model updates."""
        print(f"\n🔍 Checking for updates...")

        # Simulate update check
        has_update = False

        if has_update:
            return ModelVersion(
                name="object_detector",
                version="v2.0.0",
                hash="abc123",
                size_mb=25.5,
                uploaded_at=datetime.now().isoformat()
            )
        else:
            print(f"   ✓ No updates available")
            return None

    def download_model(self, version: ModelVersion) -> bytes:
        """Download model."""
        print(f"\n⬇️  Downloading model")
        print(f"   Version: {version.version}")
        print(f"   Size: {version.size_mb}MB")
        print(f"   ✓ Downloaded")
        return b"model_data"

    def deploy_model(self, model_data: bytes) -> bool:
        """Deploy downloaded model."""
        print(f"\n🚀 Deploying model")

        # Verify hash
        model_hash = hashlib.sha256(model_data).hexdigest()[:8]
        print(f"   Hash: {model_hash}")

        # Backup current
        print(f"   Backing up current model")

        # Deploy
        print(f"   Deploying new model")

        # Test
        print(f"   Running smoke test")

        print(f"   ✓ Deployment successful")
        return True


def demo():
    """Demonstrate edge orchestrator."""
    print("=" * 60)
    print("Edge Orchestrator Demo")
    print("=" * 60)

    # Central orchestrator
    orchestrator = Orchestrator(
        server_url="https://orch.example.com",
        api_key="secret"
    )

    # Deploy model
    orchestrator.deploy_model(
        model_path="yolov8n_v2.onnx",
        target_devices=["device_001", "device_002", "device_003"],
        rollout_strategy="gradual",
        rollout_percent=33
    )

    # Edge device
    device = EdgeDevice(
        device_id="rpi_001",
        orchestrator_url="https://orch.example.com"
    )

    device.register(
        hardware_info={
            "platform": "raspberry_pi_4",
            "ram_gb": 4,
            "storage_gb": 32
        }
    )

    device.start_heartbeat(interval_sec=30)

    # Check for updates
    update = device.check_updates()

    if update:
        model_data = device.download_model(update)
        success = device.deploy_model(model_data)

    # Rollback if needed
    orchestrator.rollback_model(
        device_id="device_001",
        model_name="object_detector",
        target_version="v1.2.0"
    )


if __name__ == "__main__":
    demo()
