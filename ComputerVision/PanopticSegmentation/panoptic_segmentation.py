"""
Panoptic Segmentation - Combining Semantic and Instance Segmentation
Uses Detectron2 Panoptic FPN for unified scene understanding
"""

import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False


class PanopticSegmentator:
    """
    Panoptic Segmentation combining semantic + instance segmentation
    - Stuff: backgrounds, sky, road (semantic)
    - Things: people, cars, objects (instance)
    """

    def __init__(self, model_name: str = "panoptic_fpn_R_101_3x", device: str = "auto"):
        if not DETECTRON2_AVAILABLE:
            raise ImportError("detectron2 required. Install from: https://detectron2.readthedocs.io")

        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self.cfg = get_cfg()

        # Load panoptic model
        config_file = f"COCO-PanopticSegmentation/{model_name}.yaml"
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        self.cfg.MODEL.DEVICE = self.device

        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        print(f"✓ Panoptic Segmentation loaded on {self.device}")

    def segment(self, image: np.ndarray) -> Dict:
        """Perform panoptic segmentation"""
        panoptic_seg, segments_info = self.predictor(image)["panoptic_seg"]
        return {
            "panoptic_seg": panoptic_seg.cpu().numpy(),
            "segments_info": segments_info
        }

    def visualize(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Visualize panoptic segmentation"""
        visualizer = Visualizer(
            image[:, :, ::-1],
            self.metadata,
            instance_mode=ColorMode.IMAGE
        )

        vis_output = visualizer.draw_panoptic_seg_predictions(
            results["panoptic_seg"],
            results["segments_info"]
        )

        return vis_output.get_image()[:, :, ::-1]

    def get_statistics(self, results: Dict) -> Dict[str, int]:
        """Get segmentation statistics"""
        stats = {"stuff": {}, "things": {}}

        for segment in results["segments_info"]:
            category_id = segment["category_id"]
            category = self.metadata.stuff_classes[category_id]
            is_thing = segment.get("isthing", False)

            key = "things" if is_thing else "stuff"
            stats[key][category] = stats[key].get(category, 0) + 1

        return stats


# Demo
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--output", default="panoptic_output.jpg")
    args = parser.parse_args()

    segmentator = PanopticSegmentator()
    image = cv2.imread(args.image)

    results = segmentator.segment(image)
    output = segmentator.visualize(image, results)
    stats = segmentator.get_statistics(results)

    cv2.imwrite(args.output, output)
    print(f"✓ Saved: {args.output}")
    print(f"\nStatistics:")
    print(f"  Stuff (semantic): {stats['stuff']}")
    print(f"  Things (instance): {stats['things']}")
