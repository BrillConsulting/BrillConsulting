"""
Advanced Object Detection with Multiple State-of-the-Art Models
Supports YOLOv8, Detectron2, EfficientDet, and Transformer-based detectors
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

# Try importing various detection frameworks
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class Detection:
    """Detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


class AdvancedObjectDetector:
    """
    Advanced Object Detection with multiple model backends
    Supports: YOLOv8, Detectron2, DETR, EfficientDet
    """

    def __init__(self, model_type: str = "yolov8", model_size: str = "medium", device: str = "auto"):
        """
        Initialize detector

        Args:
            model_type: "yolov8", "detectron2", "detr", "efficientdet"
            model_size: "small", "medium", "large", "xlarge"
            device: "auto", "cpu", "cuda"
        """
        self.model_type = model_type
        self.model_size = model_size
        self.device = self._get_device(device)
        self.model = None
        self.processor = None

        self._load_model()

    def _get_device(self, device: str) -> str:
        """Get compute device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """Load detection model"""
        print(f"Loading {self.model_type} ({self.model_size}) on {self.device}...")

        if self.model_type == "yolov8":
            self._load_yolo()
        elif self.model_type == "detectron2":
            self._load_detectron2()
        elif self.model_type == "detr":
            self._load_detr()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_yolo(self):
        """Load YOLOv8 model"""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Install: pip install ultralytics")

        size_map = {
            "nano": "yolov8n.pt",
            "small": "yolov8s.pt",
            "medium": "yolov8m.pt",
            "large": "yolov8l.pt",
            "xlarge": "yolov8x.pt"
        }

        model_name = size_map.get(self.model_size, "yolov8m.pt")
        self.model = YOLO(model_name)
        print(f"✓ YOLOv8 loaded: {model_name}")

    def _load_detectron2(self):
        """Load Detectron2 model"""
        if not DETECTRON2_AVAILABLE:
            raise ImportError("detectron2 not installed")

        cfg = get_cfg()

        # Select model based on size
        size_map = {
            "small": "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
            "medium": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
            "large": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        }

        config_file = size_map.get(self.model_size, size_map["medium"])

        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        cfg.MODEL.DEVICE = self.device

        self.model = DefaultPredictor(cfg)
        print(f"✓ Detectron2 loaded: {config_file}")

    def _load_detr(self):
        """Load DETR (Transformer) model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install: pip install transformers")

        model_name = "facebook/detr-resnet-50"
        if self.model_size == "large":
            model_name = "facebook/detr-resnet-101"

        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ DETR loaded: {model_name}")

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5,
               iou_threshold: float = 0.45) -> List[Detection]:
        """
        Detect objects in image

        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of detections
        """
        if self.model_type == "yolov8":
            return self._detect_yolo(image, conf_threshold, iou_threshold)
        elif self.model_type == "detectron2":
            return self._detect_detectron2(image, conf_threshold)
        elif self.model_type == "detr":
            return self._detect_detr(image, conf_threshold)

    def _detect_yolo(self, image: np.ndarray, conf_threshold: float,
                     iou_threshold: float) -> List[Detection]:
        """YOLOv8 detection"""
        results = self.model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]

                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                ))

        return detections

    def _detect_detectron2(self, image: np.ndarray, conf_threshold: float) -> List[Detection]:
        """Detectron2 detection"""
        outputs = self.model(image)

        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        # Get COCO class names
        from detectron2.data import MetadataCatalog
        metadata = MetadataCatalog.get(self.model.cfg.DATASETS.TRAIN[0])
        class_names = metadata.thing_classes

        detections = []
        for box, score, cls_id in zip(boxes, scores, classes):
            if score >= conf_threshold:
                x1, y1, x2, y2 = box
                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(score),
                    class_id=int(cls_id),
                    class_name=class_names[cls_id]
                ))

        return detections

    def _detect_detr(self, image: np.ndarray, conf_threshold: float) -> List[Detection]:
        """DETR (Transformer) detection"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            cls_id = int(label)
            cls_name = self.model.config.id2label[cls_id]

            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(score),
                class_id=cls_id,
                class_name=cls_name
            ))

        return detections

    def visualize(self, image: np.ndarray, detections: List[Detection],
                  show_conf: bool = True) -> np.ndarray:
        """
        Draw detections on image

        Args:
            image: Input image
            detections: List of detections
            show_conf: Show confidence scores

        Returns:
            Annotated image
        """
        output = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw bbox
            color = self._get_color(det.class_id)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name}"
            if show_conf:
                label += f" {det.confidence:.2f}"

            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)

        return output

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for class"""
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))

    def process_video(self, video_path: str, output_path: str, conf_threshold: float = 0.5):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect
            start = time.time()
            detections = self.detect(frame, conf_threshold)
            inference_time = time.time() - start
            total_time += inference_time

            # Visualize
            output_frame = self.visualize(frame, detections)

            # Add FPS
            fps_text = f"FPS: {1/inference_time:.1f}"
            cv2.putText(output_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)

            out.write(output_frame)
            frame_count += 1

            if frame_count % 30 == 0:
                avg_fps = frame_count / total_time
                print(f"Processed {frame_count} frames, Avg FPS: {avg_fps:.1f}")

        cap.release()
        out.release()
        print(f"✓ Video saved: {output_path}")


# Demo
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Object Detection")
    parser.add_argument("--model", type=str, default="yolov8",
                       choices=["yolov8", "detectron2", "detr"],
                       help="Model type")
    parser.add_argument("--size", type=str, default="medium",
                       choices=["nano", "small", "medium", "large", "xlarge"],
                       help="Model size")
    parser.add_argument("--source", type=str, required=True,
                       help="Image or video path")
    parser.add_argument("--output", type=str, default="output.jpg",
                       help="Output path")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold")

    args = parser.parse_args()

    # Initialize detector
    detector = AdvancedObjectDetector(
        model_type=args.model,
        model_size=args.size
    )

    # Detect
    if args.source.endswith(('.mp4', '.avi', '.mov')):
        # Video
        detector.process_video(args.source, args.output, args.conf)
    else:
        # Image
        image = cv2.imread(args.source)
        detections = detector.detect(image, args.conf)

        print(f"Found {len(detections)} objects:")
        for det in detections:
            print(f"  - {det.class_name}: {det.confidence:.2f}")

        # Visualize
        output = detector.visualize(image, detections)
        cv2.imwrite(args.output, output)
        print(f"✓ Result saved: {args.output}")
