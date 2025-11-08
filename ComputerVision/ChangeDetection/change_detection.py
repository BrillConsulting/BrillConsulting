"""
Change Detection
Detect and localize changes between images from different time periods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, List, Optional


class SiameseNetwork(nn.Module):
    """Siamese network for change detection"""

    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()

        # Shared encoder
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 2048
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(self.feature_dim * 2, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1: [batch, 3, H, W] - image at time t1
            img2: [batch, 3, H, W] - image at time t2
        Returns:
            change_map: [batch, 1, H, W] - binary change map
        """
        # Encode both images
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)

        # Concatenate features
        combined = torch.cat([feat1, feat2], dim=1)

        # Fuse features
        fused = self.fusion(combined)

        # Decode to change map
        change_map = self.decoder(fused)

        return change_map


class DifferenceModule(nn.Module):
    """Module to compute multi-scale differences"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        # Absolute difference
        diff = torch.abs(feat1 - feat2)

        # Concatenate original features and difference
        combined = torch.cat([feat1, feat2], dim=1)
        output = self.conv(combined)

        return output + diff  # Residual connection


class ChangeFormer(nn.Module):
    """
    Transformer-based change detection
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, d_model, 3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding2D(d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1: [batch, 3, H, W]
            img2: [batch, 3, H, W]
        Returns:
            change_map: [batch, num_classes, H, W]
        """
        batch_size = img1.size(0)

        # Encode both images
        feat1 = self.encoder(img1)  # [batch, d_model, H', W']
        feat2 = self.encoder(img2)

        # Add positional encoding
        feat1 = self.pos_encoder(feat1)
        feat2 = self.pos_encoder(feat2)

        # Reshape for transformer
        _, C, H, W = feat1.shape
        feat1_flat = feat1.view(batch_size, C, -1).permute(0, 2, 1)  # [batch, HW, C]
        feat2_flat = feat2.view(batch_size, C, -1).permute(0, 2, 1)

        # Concatenate temporal features
        combined = torch.cat([feat1_flat, feat2_flat], dim=1)  # [batch, 2*HW, C]

        # Transform
        transformed = self.transformer(combined)

        # Take only the features corresponding to time t2
        output_features = transformed[:, H*W:, :]  # [batch, HW, C]

        # Reshape back
        output_features = output_features.permute(0, 2, 1).view(batch_size, C, H, W)

        # Decode
        change_map = self.decoder(output_features)

        return change_map


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for images"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, H, W]
        Returns:
            x + pos_encoding: [batch, channels, H, W]
        """
        batch, _, h, w = x.shape

        # Create position indices
        y_embed = torch.arange(h, device=x.device).unsqueeze(1).repeat(1, w).unsqueeze(0)
        x_embed = torch.arange(w, device=x.device).unsqueeze(0).repeat(h, 1).unsqueeze(0)

        # Normalize
        y_embed = y_embed / h
        x_embed = x_embed / w

        # Create encoding
        pos_encoding = torch.zeros(batch, self.channels, h, w, device=x.device)

        # Simple encoding (can be improved with sinusoidal)
        pos_encoding[:, 0::2, :, :] = y_embed.unsqueeze(1)
        pos_encoding[:, 1::2, :, :] = x_embed.unsqueeze(1)

        return x + pos_encoding


class AttentionChangeDetection(nn.Module):
    """Change detection with spatial attention"""

    def __init__(self, backbone: str = 'resnet34'):
        super().__init__()

        # Encoder
        base_model = models.resnet34(pretrained=True)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1  # 64
        self.layer2 = base_model.layer2  # 128
        self.layer3 = base_model.layer3  # 256
        self.layer4 = base_model.layer4  # 512

        # Attention modules
        self.attention1 = SpatialAttention(64)
        self.attention2 = SpatialAttention(128)
        self.attention3 = SpatialAttention(256)
        self.attention4 = SpatialAttention(512)

        # Difference modules
        self.diff1 = DifferenceModule(64)
        self.diff2 = DifferenceModule(128)
        self.diff3 = DifferenceModule(256)
        self.diff4 = DifferenceModule(512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(128, 32, 4, 2, 1)

        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode image to multi-scale features"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1: [batch, 3, H, W]
            img2: [batch, 3, H, W]
        Returns:
            change_map: [batch, 1, H, W]
        """
        # Encode both images
        feats1 = self.encode(img1)
        feats2 = self.encode(img2)

        # Apply attention and compute differences
        diff1 = self.diff1(
            self.attention1(feats1[0]) * feats1[0],
            self.attention1(feats2[0]) * feats2[0]
        )
        diff2 = self.diff2(
            self.attention2(feats1[1]) * feats1[1],
            self.attention2(feats2[1]) * feats2[1]
        )
        diff3 = self.diff3(
            self.attention3(feats1[2]) * feats1[2],
            self.attention3(feats2[2]) * feats2[2]
        )
        diff4 = self.diff4(
            self.attention4(feats1[3]) * feats1[3],
            self.attention4(feats2[3]) * feats2[3]
        )

        # Decode with skip connections
        x = self.up4(diff4)
        x = torch.cat([x, diff3], dim=1)

        x = self.up3(x)
        x = torch.cat([x, diff2], dim=1)

        x = self.up2(x)
        x = torch.cat([x, diff1], dim=1)

        x = self.up1(x)

        # Final prediction
        change_map = self.final(x)

        # Upsample to original size
        change_map = F.interpolate(
            change_map, scale_factor=4,
            mode='bilinear', align_corners=False
        )

        return change_map


class SpatialAttention(nn.Module):
    """Spatial attention module"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        return attention


class TemporalAttention(nn.Module):
    """Temporal attention for change detection"""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat1: [batch, C, H, W]
            feat2: [batch, C, H, W]
        Returns:
            attended: [batch, C, H, W]
        """
        batch, C, H, W = feat1.shape

        # Queries from time t2
        q = self.query(feat2).view(batch, -1, H * W).permute(0, 2, 1)  # [B, HW, C']

        # Keys from time t1
        k = self.key(feat1).view(batch, -1, H * W)  # [B, C', HW]

        # Attention
        attention = torch.bmm(q, k)  # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)

        # Values from time t1
        v = self.value(feat1).view(batch, C, H * W)  # [B, C, HW]

        # Apply attention
        out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(batch, C, H, W)

        # Residual
        out = self.gamma * out + feat2

        return out


def main():
    """Main execution"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    batch_size = 4
    H, W = 256, 256

    # Test Siamese Network
    print("\n=== Siamese Change Detection ===")
    siamese = SiameseNetwork(backbone='resnet34').to(device)

    img1 = torch.randn(batch_size, 3, H, W).to(device)
    img2 = torch.randn(batch_size, 3, H, W).to(device)

    with torch.no_grad():
        change_map = siamese(img1, img2)

    print(f"Siamese output shape: {change_map.shape}")

    # Test ChangeFormer
    print("\n=== ChangeFormer ===")
    changeformer = ChangeFormer(d_model=256, num_classes=2).to(device)

    with torch.no_grad():
        change_map = changeformer(img1, img2)

    print(f"ChangeFormer output shape: {change_map.shape}")

    # Test Attention-based Change Detection
    print("\n=== Attention-based Change Detection ===")
    attention_cd = AttentionChangeDetection().to(device)

    with torch.no_grad():
        change_map = attention_cd(img1, img2)

    print(f"Attention-based output shape: {change_map.shape}")

    print("\nChange detection ready!")


if __name__ == '__main__':
    main()
