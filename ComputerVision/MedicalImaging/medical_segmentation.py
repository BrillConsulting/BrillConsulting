"""
Medical Image Segmentation using U-Net and nnU-Net
For CT, MRI, X-Ray analysis
"""

import torch
import torch.nn as nn
import cv2
import numpy as np


class UNet(nn.Module):
    """U-Net architecture for medical image segmentation"""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return torch.sigmoid(self.out(d1))


class MedicalImageSegmenter:
    """Medical image segmentation with U-Net"""

    def __init__(self, model_path: str = None, device: str = "auto"):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()
        print(f"✓ U-Net loaded on {self.device}")

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment medical image"""
        # Preprocess
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0

        # To tensor
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(tensor)

        # Postprocess
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

        return mask

    def visualize(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Overlay segmentation on image"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

        return overlay


# Demo
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="segmented.jpg")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    segmenter = MedicalImageSegmenter(args.model)
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    mask = segmenter.segment(image)
    result = segmenter.visualize(image, mask)

    cv2.imwrite(args.output, result)
    print(f"✓ Segmentation saved: {args.output}")
