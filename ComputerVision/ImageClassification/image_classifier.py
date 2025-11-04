"""
Advanced Image Classification System
Author: BrillConsulting
Description: Multi-model image classification with transfer learning support
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import json
import time


class ImageClassifier:
    """
    Advanced image classification with multiple pretrained models
    """

    # ImageNet class names (1000 classes)
    IMAGENET_CLASSES = None

    def __init__(self, model_name: str = 'resnet50',
                 device: str = 'auto',
                 num_classes: int = 1000):
        """
        Initialize image classifier

        Args:
            model_name: Model architecture (resnet50, efficientnet_b0, vit_b_16, etc.)
            device: Computing device (auto/cpu/cuda)
            num_classes: Number of output classes (1000 for ImageNet)
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🔧 Loading {model_name} on {self.device}...")

        # Load model
        self.model = self._load_model(model_name, num_classes)
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Load class names
        self._load_class_names()

        print("✅ Model loaded successfully")

    def _load_model(self, model_name: str, num_classes: int):
        """Load pretrained model"""
        model_dict = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
            'densenet121': models.densenet121,
            'mobilenet_v3_small': models.mobilenet_v3_small,
            'mobilenet_v3_large': models.mobilenet_v3_large,
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b7': models.efficientnet_b7,
            'vit_b_16': models.vit_b_16,  # Vision Transformer
            'swin_t': models.swin_t,      # Swin Transformer
        }

        if model_name not in model_dict:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {list(model_dict.keys())}")

        model = model_dict[model_name](pretrained=True)
        return model

    def _load_class_names(self):
        """Load ImageNet class names"""
        if ImageClassifier.IMAGENET_CLASSES is None:
            # Load from file or use default
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                with urllib.request.urlopen(url) as f:
                    ImageClassifier.IMAGENET_CLASSES = [line.decode('utf-8').strip()
                                                       for line in f.readlines()]
            except:
                # Fallback to indices
                ImageClassifier.IMAGENET_CLASSES = [f"class_{i}" for i in range(1000)]

        self.class_names = ImageClassifier.IMAGENET_CLASSES

    def classify_image(self, image: np.ndarray,
                      top_k: int = 5) -> List[Dict]:
        """
        Classify image and return top-k predictions

        Args:
            image: Input image (BGR format)
            top_k: Number of top predictions to return

        Returns:
            List of predictions with class names and probabilities
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.class_names[idx.item()],
                'class_id': idx.item(),
                'probability': prob.item(),
                'percentage': prob.item() * 100
            })

        return predictions

    def classify_batch(self, images: List[np.ndarray],
                      top_k: int = 5) -> List[List[Dict]]:
        """
        Classify multiple images in batch

        Args:
            images: List of images
            top_k: Number of top predictions per image

        Returns:
            List of predictions for each image
        """
        batch_predictions = []

        # Convert to tensors
        batch_tensors = []
        for image in images:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform(pil_image)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(batch)

        # Process each output
        for output in outputs:
            probabilities = torch.nn.functional.softmax(output, dim=0)
            top_probs, top_indices = torch.topk(probabilities, top_k)

            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    'class': self.class_names[idx.item()],
                    'class_id': idx.item(),
                    'probability': prob.item(),
                    'percentage': prob.item() * 100
                })

            batch_predictions.append(predictions)

        return batch_predictions

    def visualize_predictions(self, image: np.ndarray,
                            predictions: List[Dict]) -> np.ndarray:
        """
        Draw predictions on image

        Args:
            image: Input image
            predictions: List of predictions

        Returns:
            Annotated image
        """
        result = image.copy()
        height, width = result.shape[:2]

        # Create sidebar for predictions
        sidebar_width = 400
        sidebar = np.ones((height, sidebar_width, 3), dtype=np.uint8) * 255

        # Title
        cv2.putText(sidebar, "Top Predictions:", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Draw predictions
        y_offset = 70
        for i, pred in enumerate(predictions, 1):
            # Class name
            text = f"{i}. {pred['class']}"
            cv2.putText(sidebar, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Probability bar
            bar_width = int(350 * pred['probability'])
            cv2.rectangle(sidebar, (10, y_offset + 10),
                         (10 + bar_width, y_offset + 25),
                         (0, 255, 0), -1)
            cv2.rectangle(sidebar, (10, y_offset + 10),
                         (360, y_offset + 25),
                         (0, 0, 0), 1)

            # Percentage
            percentage_text = f"{pred['percentage']:.1f}%"
            cv2.putText(sidebar, percentage_text, (370, y_offset + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            y_offset += 60

        # Combine image and sidebar
        combined = np.hstack([result, sidebar])

        return combined

    def benchmark(self, image: np.ndarray, iterations: int = 100) -> Dict:
        """
        Benchmark model performance

        Args:
            image: Test image
            iterations: Number of iterations

        Returns:
            Performance statistics
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(input_tensor)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            with torch.no_grad():
                _ = self.model(input_tensor)
            times.append(time.time() - start)

        return {
            'mean_time': np.mean(times) * 1000,  # ms
            'std_time': np.std(times) * 1000,
            'min_time': np.min(times) * 1000,
            'max_time': np.max(times) * 1000,
            'fps': 1 / np.mean(times)
        }


def main():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--image', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model architecture')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions')
    parser.add_argument('--output', type=str,
                       help='Output image path')
    parser.add_argument('--json', type=str,
                       help='JSON output path')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Computing device')

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Could not load image from {args.image}")
        return

    # Initialize classifier
    classifier = ImageClassifier(model_name=args.model, device=args.device)

    # Classify
    print(f"\n🔍 Classifying with {args.model}...")
    predictions = classifier.classify_image(image, top_k=args.top_k)

    # Print results
    print(f"\n📊 Top {args.top_k} Predictions:")
    print("=" * 60)
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['class']:<40} {pred['percentage']:>6.2f}%")
    print("=" * 60)

    # Visualize
    result_image = classifier.visualize_predictions(image, predictions)

    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"💾 Saved to {args.output}")

    # Save JSON
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({
                'image': args.image,
                'model': args.model,
                'predictions': predictions
            }, f, indent=2)
        print(f"💾 JSON saved to {args.json}")

    # Benchmark
    if args.benchmark:
        print("\n⏱️  Running benchmark...")
        stats = classifier.benchmark(image)
        print(f"\n📈 Performance Statistics:")
        print(f"  Mean time: {stats['mean_time']:.2f} ms")
        print(f"  Std dev: {stats['std_time']:.2f} ms")
        print(f"  Min time: {stats['min_time']:.2f} ms")
        print(f"  Max time: {stats['max_time']:.2f} ms")
        print(f"  FPS: {stats['fps']:.1f}")

    # Display
    # Resize for display if too large
    display_height = 800
    h, w = result_image.shape[:2]
    if h > display_height:
        scale = display_height / h
        result_image = cv2.resize(result_image, None, fx=scale, fy=scale)

    cv2.imshow('Classification Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
