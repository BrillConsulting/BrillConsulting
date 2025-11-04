"""
Real-time Object Detection using YOLOv8
Author: BrillConsulting
Description: Advanced object detection system with support for images, videos, and webcam
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import time


class ObjectDetector:
    """
    YOLOv8-based object detection system with real-time processing capabilities
    """

    def __init__(self, model_path: str = 'yolov8n.pt', confidence: float = 0.5):
        """
        Initialize the object detector

        Args:
            model_path: Path to YOLOv8 model weights
            confidence: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_names = self.model.names

    def detect_objects(self, image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect objects in a single image

        Args:
            image: Input image (BGR format)

        Returns:
            Annotated image and list of detections
        """
        results = self.model(image, conf=self.confidence)[0]

        detections = []
        annotated_image = image.copy()

        for box in results.boxes:
            # Extract detection information
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]

            # Store detection
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class': class_name,
                'class_id': class_id
            })

            # Draw bounding box
            color = self._get_color(class_id)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated_image, detections

    def detect_video(self, video_path: str, output_path: Optional[str] = None,
                     display: bool = True) -> None:
        """
        Detect objects in video

        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            display: Whether to display results in real-time
        """
        cap = cv2.VideoCapture(video_path)

        if output_path:
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

            start_time = time.time()
            annotated_frame, detections = self.detect_objects(frame)
            process_time = time.time() - start_time

            total_time += process_time
            frame_count += 1
            fps = 1 / process_time

            # Add FPS counter
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if output_path:
                out.write(annotated_frame)

            if display:
                cv2.imshow('Object Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Average FPS: {avg_fps:.2f}")

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

    def detect_webcam(self, camera_id: int = 0) -> None:
        """
        Real-time detection from webcam

        Args:
            camera_id: Camera device ID
        """
        print("Starting webcam... Press 'q' to quit")
        self.detect_video(camera_id, output_path=None, display=True)

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for each class"""
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    parser.add_argument('--source', type=str, default='0',
                       help='Image/video path or webcam (0)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Model path (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for video')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')

    args = parser.parse_args()

    # Initialize detector
    detector = ObjectDetector(model_path=args.model, confidence=args.conf)

    # Determine source type
    if args.source.isdigit():
        # Webcam
        detector.detect_webcam(camera_id=int(args.source))
    elif Path(args.source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video
        detector.detect_video(args.source, output_path=args.output,
                             display=not args.no_display)
    else:
        # Image
        image = cv2.imread(args.source)
        if image is None:
            print(f"Error: Could not load image from {args.source}")
            return

        annotated_image, detections = detector.detect_objects(image)

        # Print detections
        print(f"\nDetected {len(detections)} objects:")
        for i, det in enumerate(detections, 1):
            print(f"{i}. {det['class']}: {det['confidence']:.2f}")

        # Save or display result
        if args.output:
            cv2.imwrite(args.output, annotated_image)
            print(f"Result saved to {args.output}")

        if not args.no_display:
            cv2.imshow('Object Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
