"""
Zero-Shot Object Detection using CLIP and OWL-ViT
Detect any object without training using text prompts
"""

import torch
import cv2
import numpy as np
from typing import List
from PIL import Image

try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ZeroShotDetector:
    """
    Zero-shot object detection with natural language
    Query: "a red car", "person wearing hat", "coffee mug"
    """

    def __init__(self, model_type: str = "owlvit", device: str = "auto"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required")

        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self.model_type = model_type

        if model_type == "owlvit":
            model_name = "google/owlvit-base-patch32"
            self.processor = OwlViTProcessor.from_pretrained(model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        elif model_type == "clip":
            model_name = "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()
        print(f"✓ {model_type.upper()} loaded for zero-shot detection")

    def detect(self, image: np.ndarray, text_queries: List[str],
               threshold: float = 0.1) -> List[dict]:
        """
        Detect objects using text prompts

        Args:
            image: Input image (BGR)
            text_queries: List of text descriptions ["a cat", "a dog"]
            threshold: Confidence threshold

        Returns:
            List of detections with bbox, score, label
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Process
        inputs = self.processor(
            text=text_queries,
            images=pil_image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]

        # Format detections
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "label": text_queries[label]
            })

        return detections

    def visualize(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw detections on image"""
        output = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['label']}: {det['score']:.2f}"

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1-h-10), (x1+w, y1), (0, 255, 0), -1)
            cv2.putText(output, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 1)

        return output


# Demo
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--queries", nargs='+', required=True,
                       help='Text queries like "a cat" "a dog"')
    parser.add_argument("--output", default="zero_shot_result.jpg")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    detector = ZeroShotDetector(model_type="owlvit")
    image = cv2.imread(args.image)

    detections = detector.detect(image, args.queries, args.threshold)

    print(f"Found {len(detections)} objects:")
    for det in detections:
        print(f"  - {det['label']}: {det['score']:.2f}")

    output = detector.visualize(image, detections)
    cv2.imwrite(args.output, output)
    print(f"✓ Saved: {args.output}")
