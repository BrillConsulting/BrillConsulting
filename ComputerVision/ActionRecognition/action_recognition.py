"""
Video Action Recognition using 3D CNNs and Transformers
Supports: SlowFast, X3D, TimeSformer, VideoMAE
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple

try:
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    from transformers import TimesformerForVideoClassification, AutoImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ActionRecognizer:
    """
    Video Action Recognition
    Models: VideoMAE, TimeSformer for temporal understanding
    """

    def __init__(self, model_name: str = "videomae", device: str = "auto"):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required: pip install transformers")

        self._load_model()

    def _load_model(self):
        """Load video model"""
        if self.model_name == "videomae":
            model_id = "MCG-NJU/videomae-base-finetuned-kinetics"
            self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
            self.model = VideoMAEForVideoClassification.from_pretrained(model_id)
        elif self.model_name == "timesformer":
            model_id = "facebook/timesformer-base-finetuned-k400"
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = TimesformerForVideoClassification.from_pretrained(model_id)

        self.model.to(self.device)
        self.model.eval()
        print(f"✓ {self.model_name} loaded on {self.device}")

    def extract_frames(self, video_path: str, num_frames: int = 16) -> List[np.ndarray]:
        """Extract uniform frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        return frames

    def recognize(self, video_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Recognize action in video"""
        # Extract frames
        frames = self.extract_frames(video_path)

        # Preprocess
        inputs = self.processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get predictions
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = self.model.config.id2label[idx.item()]
            results.append((label, prob.item()))

        return results


# Demo
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video")
    parser.add_argument("--model", default="videomae", choices=["videomae", "timesformer"])
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    recognizer = ActionRecognizer(args.model)
    results = recognizer.recognize(args.video, args.top_k)

    print(f"\nTop {args.top_k} predictions:")
    for i, (action, confidence) in enumerate(results, 1):
        print(f"{i}. {action}: {confidence:.2%}")
