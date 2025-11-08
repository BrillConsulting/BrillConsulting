"""
Optical Flow Estimation
Estimate pixel-level motion between consecutive frames using deep learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class FlowNetSimple(nn.Module):
    """
    FlowNetSimple: Simple encoder-decoder for optical flow
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = self._conv_block(6, 64, kernel_size=7, stride=2)
        self.conv2 = self._conv_block(64, 128, kernel_size=5, stride=2)
        self.conv3 = self._conv_block(128, 256, kernel_size=5, stride=2)
        self.conv3_1 = self._conv_block(256, 256)
        self.conv4 = self._conv_block(256, 512, stride=2)
        self.conv4_1 = self._conv_block(512, 512)
        self.conv5 = self._conv_block(512, 512, stride=2)
        self.conv5_1 = self._conv_block(512, 512)
        self.conv6 = self._conv_block(512, 1024, stride=2)
        self.conv6_1 = self._conv_block(1024, 1024)

        # Decoder
        self.deconv5 = self._deconv_block(1024, 512)
        self.deconv4 = self._deconv_block(1024, 256)
        self.deconv3 = self._deconv_block(768, 128)
        self.deconv2 = self._deconv_block(384, 64)

        # Flow prediction layers
        self.predict_flow6 = nn.Conv2d(1024, 2, 3, padding=1)
        self.predict_flow5 = nn.Conv2d(1024, 2, 3, padding=1)
        self.predict_flow4 = nn.Conv2d(768, 2, 3, padding=1)
        self.predict_flow3 = nn.Conv2d(384, 2, 3, padding=1)
        self.predict_flow2 = nn.Conv2d(192, 2, 3, padding=1)

        # Upsampling
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

    def _conv_block(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                    stride: int = 1) -> nn.Module:
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def _deconv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1: [batch, 3, H, W]
            img2: [batch, 3, H, W]
        Returns:
            flow: [batch, 2, H, W]
        """
        # Concatenate images
        x = torch.cat([img1, img2], dim=1)

        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3_1(self.conv3(conv2))
        conv4 = self.conv4_1(self.conv4(conv3))
        conv5 = self.conv5_1(self.conv5(conv4))
        conv6 = self.conv6_1(self.conv6(conv5))

        # Decoder with skip connections
        flow6 = self.predict_flow6(conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        deconv5 = self.deconv5(conv6)

        concat5 = torch.cat([conv5, deconv5, flow6_up], dim=1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        deconv4 = self.deconv4(concat5)

        concat4 = torch.cat([conv4, deconv4, flow5_up], dim=1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        deconv3 = self.deconv3(concat4)

        concat3 = torch.cat([conv3, deconv3, flow4_up], dim=1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        deconv2 = self.deconv2(concat3)

        concat2 = torch.cat([conv2, deconv2, flow3_up], dim=1)
        flow2 = self.predict_flow2(concat2)

        # Upsample to original resolution
        flow = F.interpolate(flow2, scale_factor=4, mode='bilinear', align_corners=False)

        return flow


class PWCNet(nn.Module):
    """
    PWC-Net: Pyramid, Warping, and Cost Volume Network
    """

    def __init__(self):
        super().__init__()

        # Feature pyramid
        self.conv1a = self._conv_layer(3, 16, stride=2)
        self.conv1b = self._conv_layer(16, 16)
        self.conv2a = self._conv_layer(16, 32, stride=2)
        self.conv2b = self._conv_layer(32, 32)
        self.conv3a = self._conv_layer(32, 64, stride=2)
        self.conv3b = self._conv_layer(64, 64)
        self.conv4a = self._conv_layer(64, 96, stride=2)
        self.conv4b = self._conv_layer(96, 96)
        self.conv5a = self._conv_layer(96, 128, stride=2)
        self.conv5b = self._conv_layer(128, 128)
        self.conv6a = self._conv_layer(128, 196, stride=2)
        self.conv6b = self._conv_layer(196, 196)

        # Flow estimators
        self.flow_estimators = nn.ModuleList([
            self._flow_estimator(81 + 196 + 2),
            self._flow_estimator(81 + 128 + 2),
            self._flow_estimator(81 + 96 + 2),
            self._flow_estimator(81 + 64 + 2),
            self._flow_estimator(81 + 32 + 2),
            self._flow_estimator(81 + 16 + 2)
        ])

        # Context networks
        self.context_networks = nn.ModuleList([
            self._context_network(2) for _ in range(6)
        ])

    def _conv_layer(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                    stride: int = 1) -> nn.Module:
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def _flow_estimator(self, in_ch: int) -> nn.Module:
        return nn.Sequential(
            self._conv_layer(in_ch, 128),
            self._conv_layer(128, 128),
            self._conv_layer(128, 96),
            self._conv_layer(96, 64),
            self._conv_layer(64, 32),
            nn.Conv2d(32, 2, 3, padding=1)
        )

    def _context_network(self, in_ch: int) -> nn.Module:
        return nn.Sequential(
            self._conv_layer(in_ch, 128, 3, 1),
            self._conv_layer(128, 128, 3, 1),
            self._conv_layer(128, 128, 3, 1),
            self._conv_layer(128, 96, 3, 1),
            self._conv_layer(96, 64, 3, 1),
            self._conv_layer(64, 32, 3, 1),
            nn.Conv2d(32, 2, 3, padding=1)
        )

    def feature_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features"""
        features = []
        x = self.conv1b(self.conv1a(img))
        features.append(x)
        x = self.conv2b(self.conv2a(x))
        features.append(x)
        x = self.conv3b(self.conv3a(x))
        features.append(x)
        x = self.conv4b(self.conv4a(x))
        features.append(x)
        x = self.conv5b(self.conv5a(x))
        features.append(x)
        x = self.conv6b(self.conv6a(x))
        features.append(x)
        return features[::-1]  # Reverse for coarse to fine

    def cost_volume(self, feat1: torch.Tensor, feat2: torch.Tensor,
                   max_displacement: int = 4) -> torch.Tensor:
        """Compute cost volume"""
        batch, channels, height, width = feat1.shape
        cost_vol = []

        for dx in range(-max_displacement, max_displacement + 1):
            for dy in range(-max_displacement, max_displacement + 1):
                # Shift feat2
                if dx == 0 and dy == 0:
                    shifted = feat2
                else:
                    shifted = torch.zeros_like(feat2)
                    if dx < 0:
                        if dy < 0:
                            shifted[:, :, :height+dy, :width+dx] = feat2[:, :, -dy:, -dx:]
                        elif dy > 0:
                            shifted[:, :, dy:, :width+dx] = feat2[:, :, :height-dy, -dx:]
                        else:
                            shifted[:, :, :, :width+dx] = feat2[:, :, :, -dx:]
                    elif dx > 0:
                        if dy < 0:
                            shifted[:, :, :height+dy, dx:] = feat2[:, :, -dy:, :width-dx]
                        elif dy > 0:
                            shifted[:, :, dy:, dx:] = feat2[:, :, :height-dy, :width-dx]
                        else:
                            shifted[:, :, :, dx:] = feat2[:, :, :, :width-dx]
                    else:
                        if dy < 0:
                            shifted[:, :, :height+dy, :] = feat2[:, :, -dy:, :]
                        else:
                            shifted[:, :, dy:, :] = feat2[:, :, :height-dy, :]

                # Correlation
                corr = (feat1 * shifted).mean(dim=1, keepdim=True)
                cost_vol.append(corr)

        return torch.cat(cost_vol, dim=1)

    def warp(self, feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp features using flow"""
        batch, channels, height, width = feat.shape

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=feat.device),
            torch.arange(width, device=feat.device)
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)

        # Add flow to grid
        vgrid = grid + flow

        # Normalize to [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / (width - 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / (height - 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        # Warp
        warped = F.grid_sample(feat, vgrid, align_corners=True)

        return warped

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1: [batch, 3, H, W]
            img2: [batch, 3, H, W]
        Returns:
            flow: [batch, 2, H, W]
        """
        # Extract pyramids
        features1 = self.feature_pyramid(img1)
        features2 = self.feature_pyramid(img2)

        # Coarse to fine flow estimation
        flow = None

        for level, (feat1, feat2) in enumerate(zip(features1, features2)):
            # Compute cost volume
            if flow is not None:
                # Upsample previous flow
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False) * 2
                # Warp feat2
                feat2_warped = self.warp(feat2, flow)
            else:
                feat2_warped = feat2
                flow = torch.zeros(feat1.size(0), 2, feat1.size(2), feat1.size(3),
                                 device=feat1.device)

            # Cost volume
            cost_vol = self.cost_volume(feat1, feat2_warped)

            # Concatenate
            x = torch.cat([cost_vol, feat1, flow], dim=1)

            # Estimate flow
            flow_residual = self.flow_estimators[level](x)
            flow = flow + flow_residual

            # Context network
            flow = flow + self.context_networks[level](flow)

        # Upsample to original resolution
        flow = F.interpolate(flow, scale_factor=4, mode='bilinear', align_corners=False) * 4

        return flow


class RAFTBlock(nn.Module):
    """RAFT: Recurrent All-Pairs Field Transforms"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.gru = nn.Conv2d(128 + 2, 128, 3, padding=1)

    def forward(self, x: torch.Tensor, flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.cat([x, flow], dim=1)
        flow_update = torch.tanh(self.gru(x))
        return flow + flow_update[:, :2], flow_update


def endpoint_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
    """Compute average endpoint error"""
    return torch.norm(pred_flow - gt_flow, p=2, dim=1).mean()


def main():
    """Main execution"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create models
    print("\n=== FlowNetSimple ===")
    flownet = FlowNetSimple().to(device)

    # Test
    img1 = torch.randn(2, 3, 384, 512).to(device)
    img2 = torch.randn(2, 3, 384, 512).to(device)

    with torch.no_grad():
        flow = flownet(img1, img2)
    print(f"FlowNet output shape: {flow.shape}")

    print("\n=== PWC-Net ===")
    pwcnet = PWCNet().to(device)

    with torch.no_grad():
        flow = pwcnet(img1, img2)
    print(f"PWC-Net output shape: {flow.shape}")

    print("\nOptical flow estimation ready!")


if __name__ == '__main__':
    main()
