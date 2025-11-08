"""
Video Object Segmentation
Segment and track objects across video frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, List, Dict, Optional


class STM(nn.Module):
    """
    Space-Time Memory Networks for Video Object Segmentation
    """

    def __init__(self, in_channels: int = 3, num_objects: int = 1):
        super().__init__()
        self.num_objects = num_objects

        # Encoder (ResNet-based)
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # Key encoder
        self.key_encoder = nn.Sequential(
            nn.Conv2d(1024, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1)
        )

        # Value encoder
        self.value_encoder = nn.Sequential(
            nn.Conv2d(1024 + num_objects, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1)
        )

        # Decoder
        self.decoder = nn.ModuleList([
            self._make_decoder_block(512, 256),
            self._make_decoder_block(256, 128),
            self._make_decoder_block(128, 64)
        ])

        # Output
        self.output = nn.Conv2d(64, num_objects, 1)

    def _make_decoder_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode frame to features"""
        x = self.conv1(frame)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def read_memory(self, query_key: torch.Tensor, memory_keys: torch.Tensor,
                   memory_values: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using attention
        Args:
            query_key: [batch, channels, H, W]
            memory_keys: [batch, num_frames, channels, H, W]
            memory_values: [batch, num_frames, channels, H, W]
        Returns:
            retrieved_value: [batch, channels, H, W]
        """
        batch, num_frames, channels, h, w = memory_keys.shape

        # Reshape for attention
        query = query_key.view(batch, channels, -1)  # [batch, C, HW]
        keys = memory_keys.view(batch, num_frames, channels, -1)  # [batch, T, C, HW]
        values = memory_values.view(batch, num_frames, -1, h * w)  # [batch, T, C, HW]

        # Compute attention scores
        attention_scores = []
        for i in range(num_frames):
            # Cosine similarity
            key = keys[:, i]  # [batch, C, HW]
            score = torch.einsum('bci,bcj->bij', query, key)  # [batch, HW, HW]
            score = F.softmax(score / (channels ** 0.5), dim=-1)
            attention_scores.append(score)

        # Retrieve values
        retrieved = []
        for i, score in enumerate(attention_scores):
            value = values[:, i]  # [batch, C, HW]
            retrieved_val = torch.bmm(value, score.transpose(1, 2))  # [batch, C, HW]
            retrieved.append(retrieved_val)

        # Aggregate
        retrieved_value = torch.stack(retrieved, dim=1).mean(dim=1)  # [batch, C, HW]
        retrieved_value = retrieved_value.view(batch, -1, h, w)

        return retrieved_value

    def forward(self, current_frame: torch.Tensor, memory_frames: torch.Tensor,
                memory_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_frame: [batch, 3, H, W]
            memory_frames: [batch, num_mem, 3, H, W]
            memory_masks: [batch, num_mem, num_objects, H, W]
        Returns:
            pred_mask: [batch, num_objects, H, W]
        """
        batch, num_mem = memory_frames.shape[:2]

        # Encode current frame
        current_features = self.encode_frame(current_frame)
        current_key = self.key_encoder(current_features)

        # Encode memory
        memory_keys = []
        memory_values = []

        for i in range(num_mem):
            mem_frame = memory_frames[:, i]
            mem_mask = memory_masks[:, i]

            # Encode
            mem_features = self.encode_frame(mem_frame)

            # Keys
            mem_key = self.key_encoder(mem_features)
            memory_keys.append(mem_key)

            # Values (concat features and mask)
            mem_mask_resized = F.interpolate(
                mem_mask, size=mem_features.shape[-2:],
                mode='bilinear', align_corners=False
            )
            mem_input = torch.cat([mem_features, mem_mask_resized], dim=1)
            mem_value = self.value_encoder(mem_input)
            memory_values.append(mem_value)

        memory_keys = torch.stack(memory_keys, dim=1)
        memory_values = torch.stack(memory_values, dim=1)

        # Read memory
        retrieved = self.read_memory(current_key, memory_keys, memory_values)

        # Decode
        x = retrieved
        for decoder_block in self.decoder:
            x = decoder_block(x)

        # Output
        pred_mask = self.output(x)

        # Upsample to original resolution
        pred_mask = F.interpolate(
            pred_mask, size=current_frame.shape[-2:],
            mode='bilinear', align_corners=False
        )

        return pred_mask


class AOT(nn.Module):
    """
    Associating Objects with Transformers for Video Object Segmentation
    """

    def __init__(self, num_objects: int = 1, d_model: int = 256, nhead: int = 8):
        super().__init__()
        self.num_objects = num_objects
        self.d_model = d_model

        # Encoder
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

        # Project to d_model
        self.proj = nn.Conv2d(1024, d_model, 1)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, num_objects, 1)
        )

        # Object queries
        self.object_queries = nn.Parameter(torch.randn(num_objects, d_model))

    def forward(self, frames: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            frames: [batch, num_frames, 3, H, W]
            masks: [batch, num_frames, num_objects, H, W] - reference masks
        Returns:
            pred_masks: [batch, num_frames, num_objects, H, W]
        """
        batch, num_frames = frames.shape[:2]

        # Encode all frames
        all_features = []
        for t in range(num_frames):
            frame = frames[:, t]
            features = self.encoder(frame)
            features = self.proj(features)
            all_features.append(features)

        # Stack features
        features = torch.stack(all_features, dim=1)  # [batch, T, C, H, W]

        # Reshape for transformer
        B, T, C, H, W = features.shape
        features_flat = features.view(B, T, C, -1).permute(0, 1, 3, 2)  # [B, T, HW, C]
        features_flat = features_flat.reshape(B, T * H * W, C)

        # Add object queries
        obj_queries = self.object_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_obj, C]
        combined = torch.cat([obj_queries, features_flat], dim=1)

        # Transformer
        transformed = self.transformer(combined)

        # Split back
        obj_embeddings = transformed[:, :self.num_objects]  # [B, num_obj, C]
        frame_features = transformed[:, self.num_objects:].reshape(B, T, H, W, C)
        frame_features = frame_features.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]

        # Decode each frame
        pred_masks = []
        for t in range(num_frames):
            feat = frame_features[:, t]
            mask = self.decoder(feat)
            pred_masks.append(mask)

        pred_masks = torch.stack(pred_masks, dim=1)

        return pred_masks


class MaskPropagation(nn.Module):
    """Mask propagation across frames"""

    def __init__(self, feature_dim: int = 512):
        super().__init__()

        # Feature extractor
        resnet = models.resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # Correlation layer
        self.correlation = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1)
        )

        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(64 + 1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def warp_mask(self, mask: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp mask using optical flow"""
        batch, _, h, w = mask.shape

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=mask.device),
            torch.arange(w, device=mask.device)
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)

        # Add flow
        vgrid = grid + flow

        # Normalize to [-1, 1]
        vgrid[:, 0] = 2.0 * vgrid[:, 0] / (w - 1) - 1.0
        vgrid[:, 1] = 2.0 * vgrid[:, 1] / (h - 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        # Warp
        warped = F.grid_sample(mask, vgrid, align_corners=True)

        return warped

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor,
                mask1: torch.Tensor) -> torch.Tensor:
        """
        Propagate mask from frame1 to frame2
        Args:
            frame1: [batch, 3, H, W]
            frame2: [batch, 3, H, W]
            mask1: [batch, 1, H, W]
        Returns:
            mask2: [batch, 1, H, W]
        """
        # Extract features
        feat1 = self.feature_extractor(frame1)
        feat2 = self.feature_extractor(frame2)

        # Compute correlation
        corr = torch.cat([feat1, feat2], dim=1)
        corr_features = self.correlation(corr)

        # Resize mask
        mask1_resized = F.interpolate(mask1, size=corr_features.shape[-2:],
                                     mode='bilinear', align_corners=False)

        # Refine
        refine_input = torch.cat([corr_features, mask1_resized], dim=1)
        mask2 = self.refine(refine_input)

        # Upsample to original size
        mask2 = F.interpolate(mask2, size=frame2.shape[-2:],
                             mode='bilinear', align_corners=False)

        return mask2


def main():
    """Main execution"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    batch_size = 2
    num_frames = 5
    num_objects = 1
    H, W = 480, 854

    # Test STM
    print("\n=== Space-Time Memory Networks ===")
    stm = STM(num_objects=num_objects).to(device)

    current_frame = torch.randn(batch_size, 3, H, W).to(device)
    memory_frames = torch.randn(batch_size, 3, 3, H, W).to(device)
    memory_masks = torch.randn(batch_size, 3, num_objects, H, W).to(device)

    with torch.no_grad():
        pred_mask = stm(current_frame, memory_frames, memory_masks)

    print(f"STM output shape: {pred_mask.shape}")

    # Test AOT
    print("\n=== Associating Objects with Transformers ===")
    aot = AOT(num_objects=num_objects).to(device)

    frames = torch.randn(batch_size, num_frames, 3, 240, 427).to(device)

    with torch.no_grad():
        pred_masks = aot(frames)

    print(f"AOT output shape: {pred_masks.shape}")

    # Test Mask Propagation
    print("\n=== Mask Propagation ===")
    mask_prop = MaskPropagation().to(device)

    frame1 = torch.randn(batch_size, 3, H, W).to(device)
    frame2 = torch.randn(batch_size, 3, H, W).to(device)
    mask1 = torch.randn(batch_size, 1, H, W).to(device)

    with torch.no_grad():
        mask2 = mask_prop(frame1, frame2, mask1)

    print(f"Propagated mask shape: {mask2.shape}")

    print("\nVideo object segmentation ready!")


if __name__ == '__main__':
    main()
