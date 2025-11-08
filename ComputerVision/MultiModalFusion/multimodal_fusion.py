"""
Multi-Modal Fusion
Combine information from multiple modalities (vision, text, audio) for enhanced understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, List, Dict, Optional
import math


class CrossModalAttention(nn.Module):
    """Cross-modal attention between two modalities"""

    def __init__(self, dim1: int, dim2: int, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x1: [batch, seq_len1, dim1] - query modality
            x2: [batch, seq_len2, dim2] - key/value modality
        Returns:
            attended: [batch, seq_len1, hidden_dim]
            attention_weights: [batch, num_heads, seq_len1, seq_len2]
        """
        # Project to common space
        q = self.proj1(x1)
        kv = self.proj2(x2)

        # Apply attention
        attended, attn_weights = self.multihead_attn(q, kv, kv)

        # Residual and normalization
        attended = self.norm1(attended + q)

        return attended, attn_weights


class TensorFusion(nn.Module):
    """Tensor Fusion Network for multi-modal fusion"""

    def __init__(self, dims: List[int], output_dim: int):
        super().__init__()
        self.dims = dims
        self.output_dim = output_dim

        # Compute fusion dimension
        fusion_dim = 1
        for dim in dims:
            fusion_dim *= (dim + 1)

        self.fc = nn.Linear(fusion_dim, output_dim)

    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modalities: List of [batch, dim_i] tensors
        Returns:
            fused: [batch, output_dim]
        """
        batch_size = modalities[0].size(0)

        # Add bias dimension
        modalities_with_bias = [
            torch.cat([m, torch.ones(batch_size, 1, device=m.device)], dim=1)
            for m in modalities
        ]

        # Compute outer product
        fusion = modalities_with_bias[0]
        for m in modalities_with_bias[1:]:
            fusion = torch.bmm(fusion.unsqueeze(2), m.unsqueeze(1))
            fusion = fusion.view(batch_size, -1)

        # Project to output
        output = self.fc(fusion)

        return output


class BilinearFusion(nn.Module):
    """Bilinear fusion for two modalities"""

    def __init__(self, dim1: int, dim2: int, output_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim1, dim2, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: [batch, dim1]
            x2: [batch, dim2]
        Returns:
            fused: [batch, output_dim]
        """
        # Bilinear product
        # output[k] = sum_ij x1[i] * W[i,j,k] * x2[j]
        batch_size = x1.size(0)
        output = torch.einsum('bi,ijk,bj->bk', x1, self.W, x2) + self.bias

        return output


class MultiModalTransformer(nn.Module):
    """Transformer-based multi-modal fusion"""

    def __init__(self, modality_dims: Dict[str, int], d_model: int = 512,
                 nhead: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.modality_dims = modality_dims

        # Modality-specific projections
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, d_model)
            for name, dim in modality_dims.items()
        })

        # Modality embeddings
        self.modality_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, d_model))
            for name in modality_dims.keys()
        })

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modalities: Dict of {modality_name: [batch, seq_len, dim]}
        Returns:
            fused: [batch, d_model]
        """
        batch_size = list(modalities.values())[0].size(0)

        # Project and add modality embeddings
        projected = []
        for name, features in modalities.items():
            # Project
            proj = self.projections[name](features)

            # Add modality embedding
            modality_emb = self.modality_embeddings[name].expand(batch_size, features.size(1), -1)
            proj = proj + modality_emb

            projected.append(proj)

        # Concatenate all modalities
        combined = torch.cat(projected, dim=1)  # [batch, total_seq_len, d_model]

        # Transform
        transformed = self.transformer(combined)

        # Global average pooling
        fused = transformed.mean(dim=1)

        # Output projection
        output = self.output_proj(fused)

        return output


class CLIP(nn.Module):
    """
    CLIP-style contrastive learning for vision-language alignment
    """

    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module,
                 embed_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature

        # Projection heads
        self.vision_projection = nn.Linear(vision_encoder.feature_dim, embed_dim)
        self.text_projection = nn.Linear(text_encoder.hidden_size, embed_dim)

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embedding space"""
        features = self.vision_encoder(images)
        embeddings = self.vision_projection(features)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to embedding space"""
        features = self.text_encoder(text)
        embeddings = self.text_projection(features)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [batch, 3, H, W]
            text: [batch, seq_len]
        Returns:
            logits_per_image: [batch, batch]
            logits_per_text: [batch, batch]
        """
        # Encode
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(text)

        # Compute similarities
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text


class ModalityGating(nn.Module):
    """Adaptive gating mechanism for modality fusion"""

    def __init__(self, modality_dims: List[int], output_dim: int):
        super().__init__()
        self.num_modalities = len(modality_dims)

        # Modality-specific transformations
        self.transforms = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in modality_dims
        ])

        # Gating network
        total_dim = sum(modality_dims)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, self.num_modalities),
            nn.Softmax(dim=1)
        )

    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modalities: List of [batch, dim_i] tensors
        Returns:
            fused: [batch, output_dim]
        """
        # Transform each modality
        transformed = [
            transform(modality)
            for transform, modality in zip(self.transforms, modalities)
        ]

        # Compute gates
        concatenated = torch.cat(modalities, dim=1)
        gates = self.gate(concatenated)  # [batch, num_modalities]

        # Apply gates
        stacked = torch.stack(transformed, dim=1)  # [batch, num_modalities, output_dim]
        gates = gates.unsqueeze(2)  # [batch, num_modalities, 1]

        fused = (stacked * gates).sum(dim=1)  # [batch, output_dim]

        return fused


class AudioVisualFusion(nn.Module):
    """Fusion network for audio and visual modalities"""

    def __init__(self, visual_dim: int = 2048, audio_dim: int = 128,
                 hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()

        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Cross-modal attention
        self.cross_attn_va = CrossModalAttention(hidden_dim, hidden_dim, hidden_dim)
        self.cross_attn_av = CrossModalAttention(hidden_dim, hidden_dim, hidden_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual: [batch, visual_dim] or [batch, seq_len, visual_dim]
            audio: [batch, audio_dim] or [batch, seq_len, audio_dim]
        Returns:
            logits: [batch, num_classes]
        """
        # Encode
        v = self.visual_encoder(visual)
        a = self.audio_encoder(audio)

        # Add sequence dimension if needed
        if v.dim() == 2:
            v = v.unsqueeze(1)
        if a.dim() == 2:
            a = a.unsqueeze(1)

        # Cross-modal attention
        v_attended, _ = self.cross_attn_va(v, a)
        a_attended, _ = self.cross_attn_av(a, v)

        # Pool
        v_pooled = v_attended.mean(dim=1)
        a_pooled = a_attended.mean(dim=1)

        # Fuse
        concatenated = torch.cat([v_pooled, a_pooled], dim=1)
        fused = self.fusion(concatenated)

        # Classify
        logits = self.classifier(fused)

        return logits


def main():
    """Main execution"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    batch_size = 8

    # Test Cross-Modal Attention
    print("\n=== Cross-Modal Attention ===")
    cross_attn = CrossModalAttention(dim1=512, dim2=256, hidden_dim=512).to(device)
    x1 = torch.randn(batch_size, 10, 512).to(device)
    x2 = torch.randn(batch_size, 20, 256).to(device)
    attended, attn_weights = cross_attn(x1, x2)
    print(f"Attended shape: {attended.shape}")

    # Test Tensor Fusion
    print("\n=== Tensor Fusion ===")
    tensor_fusion = TensorFusion(dims=[128, 256, 64], output_dim=512).to(device)
    m1 = torch.randn(batch_size, 128).to(device)
    m2 = torch.randn(batch_size, 256).to(device)
    m3 = torch.randn(batch_size, 64).to(device)
    fused = tensor_fusion([m1, m2, m3])
    print(f"Fused shape: {fused.shape}")

    # Test Multi-Modal Transformer
    print("\n=== Multi-Modal Transformer ===")
    modality_dims = {'vision': 2048, 'text': 768, 'audio': 128}
    mm_transformer = MultiModalTransformer(modality_dims, d_model=512).to(device)

    modalities = {
        'vision': torch.randn(batch_size, 49, 2048).to(device),
        'text': torch.randn(batch_size, 20, 768).to(device),
        'audio': torch.randn(batch_size, 10, 128).to(device)
    }
    output = mm_transformer(modalities)
    print(f"Multi-modal output shape: {output.shape}")

    # Test Audio-Visual Fusion
    print("\n=== Audio-Visual Fusion ===")
    av_fusion = AudioVisualFusion(visual_dim=2048, audio_dim=128, num_classes=10).to(device)
    visual = torch.randn(batch_size, 2048).to(device)
    audio = torch.randn(batch_size, 128).to(device)
    logits = av_fusion(visual, audio)
    print(f"Classification logits shape: {logits.shape}")

    print("\nMulti-modal fusion ready!")


if __name__ == '__main__':
    main()
