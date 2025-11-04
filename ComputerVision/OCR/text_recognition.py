"""
Advanced OCR (Optical Character Recognition) System
Author: BrillConsulting
Description: Multi-language text recognition with EasyOCR and Tesseract
"""

import cv2
import numpy as np
import easyocr
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


class OCRSystem:
    """
    Advanced OCR system with support for 80+ languages
    """

    def __init__(self, languages: List[str] = ['en'],
                 gpu: bool = True,
                 detector: str = 'easyocr'):
        """
        Initialize OCR system

        Args:
            languages: List of language codes (e.g., ['en', 'pl', 'de'])
            gpu: Use GPU acceleration
            detector: OCR engine ('easyocr' or 'tesseract')
        """
        self.languages = languages
        self.detector_type = detector

        if detector == 'easyocr':
            print(f"🔧 Initializing EasyOCR for {', '.join(languages)}...")
            self.reader = easyocr.Reader(languages, gpu=gpu)
            print("✅ EasyOCR ready")
        elif detector == 'tesseract':
            try:
                import pytesseract
                self.reader = pytesseract
                print("✅ Tesseract ready")
            except ImportError:
                raise ImportError("Install pytesseract: pip install pytesseract")
        else:
            raise ValueError(f"Unknown detector: {detector}")

    def detect_text(self, image: np.ndarray,
                   confidence_threshold: float = 0.4) -> List[Dict]:
        """
        Detect and recognize text in image

        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for detections

        Returns:
            List of detected text regions with metadata
        """
        if self.detector_type == 'easyocr':
            # EasyOCR processing
            results = self.reader.readtext(image)

            detections = []
            for bbox, text, confidence in results:
                if confidence >= confidence_threshold:
                    # Convert bbox to standard format
                    points = np.array(bbox, dtype=np.int32)
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)

                    detections.append({
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                        'polygon': points.tolist()
                    })

        else:  # tesseract
            # Tesseract processing
            data = self.reader.image_to_data(image, output_type=self.reader.Output.DICT)

            detections = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])

                if text and confidence > confidence_threshold * 100:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                    detections.append({
                        'text': text,
                        'confidence': confidence / 100.0,
                        'bbox': [x, y, x + w, y + h],
                        'polygon': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    })

        return detections

    def visualize_detections(self, image: np.ndarray,
                           detections: List[Dict],
                           show_confidence: bool = True) -> np.ndarray:
        """
        Draw detected text regions on image

        Args:
            image: Input image
            detections: List of text detections
            show_confidence: Display confidence scores

        Returns:
            Annotated image
        """
        result = image.copy()

        for det in detections:
            # Draw bounding box
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox

            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw polygon if available
            if 'polygon' in det:
                polygon = np.array(det['polygon'], dtype=np.int32)
                cv2.polylines(result, [polygon], True, (0, 255, 0), 2)

            # Add text label
            label = det['text']
            if show_confidence:
                label += f" ({det['confidence']:.2f})"

            # Background for text
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            cv2.rectangle(result, (x1, y1 - label_height - 10),
                         (x1 + label_width, y1), (0, 255, 0), -1)

            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result

    def extract_text(self, image: np.ndarray,
                    confidence_threshold: float = 0.4) -> str:
        """
        Extract all text from image as a single string

        Args:
            image: Input image
            confidence_threshold: Minimum confidence

        Returns:
            Concatenated text
        """
        detections = self.detect_text(image, confidence_threshold)

        # Sort by vertical position (top to bottom)
        detections.sort(key=lambda x: x['bbox'][1])

        # Extract text
        text_lines = [det['text'] for det in detections]

        return '\n'.join(text_lines)

    def process_document(self, image: np.ndarray) -> Dict:
        """
        Process document image with preprocessing and text extraction

        Args:
            image: Document image

        Returns:
            Dictionary with extracted information
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)

        # Detect text
        detections = self.detect_text(preprocessed)

        # Sort detections by position
        detections.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))

        # Extract metadata
        result = {
            'timestamp': datetime.now().isoformat(),
            'num_detections': len(detections),
            'languages': self.languages,
            'detections': detections,
            'full_text': '\n'.join([d['text'] for d in detections]),
            'average_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0
        }

        return result

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Dilation to connect text components
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        return dilated

    def search_text(self, image: np.ndarray, query: str,
                   case_sensitive: bool = False) -> List[Dict]:
        """
        Search for specific text in image

        Args:
            image: Input image
            query: Text to search for
            case_sensitive: Case-sensitive search

        Returns:
            List of matching detections
        """
        detections = self.detect_text(image)

        matches = []
        for det in detections:
            text = det['text'] if case_sensitive else det['text'].lower()
            search_query = query if case_sensitive else query.lower()

            if search_query in text:
                matches.append(det)

        return matches


def main():
    parser = argparse.ArgumentParser(description='OCR Text Recognition')
    parser.add_argument('--image', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--languages', type=str, nargs='+', default=['en'],
                       help='Language codes (e.g., en pl de)')
    parser.add_argument('--detector', type=str, default='easyocr',
                       choices=['easyocr', 'tesseract'],
                       help='OCR engine')
    parser.add_argument('--output', type=str,
                       help='Output image path')
    parser.add_argument('--json', type=str,
                       help='Output JSON path for results')
    parser.add_argument('--mode', type=str, default='detect',
                       choices=['detect', 'extract', 'document', 'search'],
                       help='Processing mode')
    parser.add_argument('--query', type=str,
                       help='Search query for search mode')
    parser.add_argument('--confidence', type=float, default=0.4,
                       help='Confidence threshold')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU')
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply preprocessing')

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Could not load image from {args.image}")
        return

    # Initialize OCR
    ocr = OCRSystem(languages=args.languages,
                    gpu=not args.no_gpu,
                    detector=args.detector)

    # Preprocess if requested
    if args.preprocess:
        image = ocr.preprocess_image(image)

    # Process based on mode
    if args.mode == 'detect':
        detections = ocr.detect_text(image, confidence_threshold=args.confidence)

        print(f"\n📝 Detected {len(detections)} text region(s):")
        for i, det in enumerate(detections, 1):
            print(f"{i}. '{det['text']}' (confidence: {det['confidence']:.2f})")

        result_image = ocr.visualize_detections(image, detections)

        if args.output:
            cv2.imwrite(args.output, result_image)
            print(f"💾 Saved to {args.output}")

        cv2.imshow('OCR Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == 'extract':
        text = ocr.extract_text(image, confidence_threshold=args.confidence)

        print("\n📄 Extracted Text:")
        print("=" * 50)
        print(text)
        print("=" * 50)

        if args.json:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump({'text': text}, f, ensure_ascii=False, indent=2)
            print(f"💾 Saved to {args.json}")

    elif args.mode == 'document':
        result = ocr.process_document(image)

        print(f"\n📊 Document Analysis:")
        print(f"  Text regions: {result['num_detections']}")
        print(f"  Average confidence: {result['average_confidence']:.2f}")
        print(f"\n📄 Full Text:")
        print("=" * 50)
        print(result['full_text'])
        print("=" * 50)

        if args.json:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 Saved to {args.json}")

    elif args.mode == 'search':
        if not args.query:
            print("❌ --query required for search mode")
            return

        matches = ocr.search_text(image, args.query)

        print(f"\n🔍 Found {len(matches)} match(es) for '{args.query}':")
        for i, match in enumerate(matches, 1):
            print(f"{i}. '{match['text']}' (confidence: {match['confidence']:.2f})")

        if matches:
            result_image = ocr.visualize_detections(image, matches)

            if args.output:
                cv2.imwrite(args.output, result_image)

            cv2.imshow('Search Results', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
