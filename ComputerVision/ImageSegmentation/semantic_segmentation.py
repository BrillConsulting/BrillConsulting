"""
Semantic Image Segmentation System
Author: BrillConsulting
Description: Multi-model segmentation with DeepLabV3+ and Segment Anything Model (SAM)
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from PIL import Image


class SemanticSegmenter:
    """
    Advanced semantic segmentation using DeepLabV3+
    """

    # PASCAL VOC classes
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # Color palette for visualization
    COLORS = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ], dtype=np.uint8)

    def __init__(self, model_name: str = 'deeplabv3_resnet101',
                 device: str = 'auto'):
        """
        Initialize semantic segmentation model

        Args:
            model_name: Model architecture (deeplabv3_resnet50/101)
            device: Computing device (auto/cpu/cuda)
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🔧 Loading {model_name} on {self.device}...")

        # Load pretrained model
        if model_name == 'deeplabv3_resnet50':
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(
                pretrained=True
            )
        elif model_name == 'deeplabv3_resnet101':
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                pretrained=True
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print("✅ Model loaded successfully")

    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform semantic segmentation on image

        Args:
            image: Input image (BGR format)

        Returns:
            Segmentation mask and colored visualization
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = rgb_image.shape[:2]

        # Preprocess
        input_tensor = self.transform(Image.fromarray(rgb_image))
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]

        # Get class predictions
        output_predictions = output.argmax(0).cpu().numpy()

        # Resize to original size
        mask = cv2.resize(output_predictions.astype(np.uint8),
                         (original_size[1], original_size[0]),
                         interpolation=cv2.INTER_NEAREST)

        # Create colored visualization
        colored_mask = self.COLORS[mask]

        return mask, colored_mask

    def segment_with_overlay(self, image: np.ndarray,
                            alpha: float = 0.5) -> np.ndarray:
        """
        Segment image and overlay on original

        Args:
            image: Input image
            alpha: Overlay transparency (0-1)

        Returns:
            Overlaid image
        """
        mask, colored_mask = self.segment_image(image)

        # Convert colored mask to BGR for OpenCV
        colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)

        # Blend with original image
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask_bgr, alpha, 0)

        return overlay

    def extract_object(self, image: np.ndarray,
                      class_name: str) -> np.ndarray:
        """
        Extract specific object class from image

        Args:
            image: Input image
            class_name: Class to extract (e.g., 'person', 'car')

        Returns:
            Image with only specified class visible
        """
        if class_name not in self.CLASSES:
            raise ValueError(f"Class '{class_name}' not in {self.CLASSES}")

        class_id = self.CLASSES.index(class_name)
        mask, _ = self.segment_image(image)

        # Create binary mask for the class
        class_mask = (mask == class_id).astype(np.uint8) * 255

        # Apply mask to image
        result = cv2.bitwise_and(image, image, mask=class_mask)

        return result

    def get_class_statistics(self, image: np.ndarray) -> dict:
        """
        Get statistics about detected classes

        Args:
            image: Input image

        Returns:
            Dictionary with class statistics
        """
        mask, _ = self.segment_image(image)

        stats = {}
        total_pixels = mask.size

        for class_id, class_name in enumerate(self.CLASSES):
            pixels = np.sum(mask == class_id)
            if pixels > 0:
                percentage = (pixels / total_pixels) * 100
                stats[class_name] = {
                    'pixels': int(pixels),
                    'percentage': round(percentage, 2)
                }

        return stats


class InstanceSegmenter:
    """
    Instance segmentation using Mask R-CNN
    """

    def __init__(self, device: str = 'auto', confidence: float = 0.5):
        """
        Initialize instance segmentation model

        Args:
            device: Computing device
            confidence: Confidence threshold
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🔧 Loading Mask R-CNN on {self.device}...")

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()

        self.confidence = confidence
        self.transform = transforms.ToTensor()

        print("✅ Model loaded successfully")

    def segment_instances(self, image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect and segment individual object instances

        Args:
            image: Input image

        Returns:
            Annotated image and list of instances
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_image).to(self.device)

        with torch.no_grad():
            predictions = self.model([input_tensor])[0]

        # Filter by confidence
        scores = predictions['scores'].cpu().numpy()
        keep_indices = scores >= self.confidence

        boxes = predictions['boxes'][keep_indices].cpu().numpy()
        labels = predictions['labels'][keep_indices].cpu().numpy()
        masks = predictions['masks'][keep_indices].cpu().numpy()
        scores = scores[keep_indices]

        # Create visualization
        result_image = image.copy()
        instances = []

        for i, (box, label, mask, score) in enumerate(zip(boxes, labels, masks, scores)):
            # Generate random color for this instance
            color = tuple(map(int, np.random.randint(0, 255, 3)))

            # Draw mask
            binary_mask = (mask[0] > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(image)
            colored_mask[binary_mask == 1] = color

            result_image = cv2.addWeighted(result_image, 1, colored_mask, 0.4, 0)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # Add label
            label_text = f"{SemanticSegmenter.CLASSES[label]}: {score:.2f}"
            cv2.putText(result_image, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            instances.append({
                'class': SemanticSegmenter.CLASSES[label],
                'confidence': float(score),
                'bbox': box.tolist(),
                'mask': binary_mask
            })

        return result_image, instances


def main():
    parser = argparse.ArgumentParser(description='Image Segmentation')
    parser.add_argument('--image', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--model', type=str, default='deeplabv3',
                       choices=['deeplabv3', 'maskrcnn'],
                       help='Model type')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--mode', type=str, default='overlay',
                       choices=['overlay', 'mask', 'extract', 'stats'],
                       help='Output mode')
    parser.add_argument('--class-name', type=str,
                       help='Class name for extraction')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Overlay transparency')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for instance segmentation')

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Could not load image from {args.image}")
        return

    if args.model == 'deeplabv3':
        # Semantic segmentation
        segmenter = SemanticSegmenter()

        if args.mode == 'overlay':
            result = segmenter.segment_with_overlay(image, alpha=args.alpha)
        elif args.mode == 'mask':
            _, colored_mask = segmenter.segment_image(image)
            result = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
        elif args.mode == 'extract':
            if not args.class_name:
                print("❌ --class-name required for extraction mode")
                return
            result = segmenter.extract_object(image, args.class_name)
        elif args.mode == 'stats':
            stats = segmenter.get_class_statistics(image)
            print("\n📊 Segmentation Statistics:")
            for class_name, data in sorted(stats.items(),
                                          key=lambda x: x[1]['percentage'],
                                          reverse=True):
                print(f"  {class_name}: {data['percentage']:.2f}% "
                     f"({data['pixels']:,} pixels)")
            result = segmenter.segment_with_overlay(image)

    else:  # maskrcnn
        segmenter = InstanceSegmenter(confidence=args.confidence)
        result, instances = segmenter.segment_instances(image)

        print(f"\n🎯 Detected {len(instances)} instances:")
        for i, inst in enumerate(instances, 1):
            print(f"  {i}. {inst['class']}: {inst['confidence']:.2f}")

    # Save or display
    if args.output:
        cv2.imwrite(args.output, result)
        print(f"💾 Saved to {args.output}")

    cv2.imshow('Segmentation Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
