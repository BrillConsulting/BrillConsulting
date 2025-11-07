"""
Realtime Inference Pipeline
============================

Camera → MQTT → Inference → Dashboard

Author: Brill Consulting
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import time


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    cameras: List[int]
    mqtt_broker: str
    inference_backend: str
    dashboard_port: int


class CameraNode:
    """Camera capture node."""

    def __init__(
        self,
        camera_id: int,
        mqtt_broker: str,
        topic: str,
        fps: int = 30
    ):
        self.camera_id = camera_id
        self.mqtt_broker = mqtt_broker
        self.topic = topic
        self.fps = fps

        print(f"📷 Camera Node {camera_id} initialized")

    def stream(self) -> None:
        """Stream camera to MQTT."""
        print(f"\n🎥 Starting camera stream")
        print(f"   FPS: {self.fps}")
        print(f"   Topic: {self.topic}")

        # Simulate streaming
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Publish to MQTT
            time.sleep(1/self.fps)
            if i % 30 == 0:
                print(f"   Frame {i} published")


class InferenceNode:
    """Inference processing node."""

    def __init__(
        self,
        model_path: str,
        input_topic: str,
        output_topic: str,
        batch_size: int = 4
    ):
        self.model_path = model_path
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.batch_size = batch_size

        print(f"🧠 Inference Node initialized")
        print(f"   Model: {model_path}")
        print(f"   Batch size: {batch_size}")

    def start(self) -> None:
        """Start inference processing."""
        print(f"\n🚀 Starting inference")
        print(f"   Listening on: {self.input_topic}")
        print(f"   Publishing to: {self.output_topic}")


class Pipeline:
    """Complete realtime pipeline."""

    def __init__(
        self,
        cameras: List[int],
        mqtt_broker: str,
        inference_backend: str,
        dashboard_port: int = 8080
    ):
        self.config = PipelineConfig(
            cameras=cameras,
            mqtt_broker=mqtt_broker,
            inference_backend=inference_backend,
            dashboard_port=dashboard_port
        )

        print(f"🔄 Realtime Pipeline initialized")
        print(f"   Cameras: {len(cameras)}")
        print(f"   Backend: {inference_backend}")

    def start(self) -> None:
        """Start full pipeline."""
        print(f"\n🚀 Starting pipeline")

        # Start camera nodes
        for cam_id in self.config.cameras:
            camera = CameraNode(
                camera_id=cam_id,
                mqtt_broker=self.config.mqtt_broker,
                topic=f"camera/stream/{cam_id}",
                fps=30
            )

        # Start inference
        inference = InferenceNode(
            model_path="model.engine",
            input_topic="camera/stream/+",
            output_topic="inference/results",
            batch_size=4
        )

        print(f"   ✓ Pipeline running")
        print(f"   Dashboard: http://localhost:{self.config.dashboard_port}")


def demo():
    """Demonstrate realtime pipeline."""
    print("=" * 60)
    print("Realtime Inference Pipeline Demo")
    print("=" * 60)

    pipeline = Pipeline(
        cameras=[0, 1],
        mqtt_broker="localhost:1883",
        inference_backend="tensorrt",
        dashboard_port=8080
    )

    pipeline.start()


if __name__ == "__main__":
    demo()
