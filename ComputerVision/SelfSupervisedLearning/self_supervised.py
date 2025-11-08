"""
Self-Supervised Learning for Computer Vision
Learn representations without labels using contrastive and predictive methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple, List, Optional


class SimCLR(nn.Module):
    """
    SimCLR: Simple Framework for Contrastive Learning of Visual Representations
    """

    def __init__(self, base_encoder: nn.Module = None, projection_dim: int = 128):
        super().__init__()

        # Base encoder
        if base_encoder is None:
            base_encoder = models.resnet50(pretrained=False)

        # Remove classification head
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-1])
        self.feature_dim = base_encoder.fc.in_features

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, 3, H, W]
        Returns:
            features: [batch_size, feature_dim]
            projections: [batch_size, projection_dim]
        """
        features = self.encoder(x).squeeze(-1).squeeze(-1)
        projections = self.projection_head(features)
        return features, projections


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss"""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: [batch_size, projection_dim] - projections of augmented view 1
            z_j: [batch_size, projection_dim] - projections of augmented view 2
        Returns:
            loss: scalar
        """
        batch_size = z_i.size(0)

        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, projection_dim]

        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / self.temperature  # [2*batch, 2*batch]

        # Create mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix.masked_fill_(mask, -1e9)

        # Positive pairs are at indices (i, i+batch_size) and (i+batch_size, i)
        positive_samples = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)

        # Compute loss
        numerator = similarity_matrix[torch.arange(2 * batch_size), positive_samples]
        denominator = torch.logsumexp(similarity_matrix, dim=1)

        loss = -torch.mean(numerator - denominator)

        return loss


class MoCo(nn.Module):
    """
    MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
    """

    def __init__(self, base_encoder: nn.Module = None, projection_dim: int = 128,
                 queue_size: int = 65536, momentum: float = 0.999, temperature: float = 0.07):
        super().__init__()

        if base_encoder is None:
            base_encoder = models.resnet50(pretrained=False)

        # Query encoder
        self.encoder_q = nn.Sequential(*list(base_encoder.children())[:-1])
        feature_dim = base_encoder.fc.in_features

        self.projection_q = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

        # Key encoder (momentum encoder)
        self.encoder_k = nn.Sequential(*list(base_encoder.children())[:-1])
        self.projection_k = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

        # Initialize key encoder with query encoder parameters
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Queue
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue"""
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Replace oldest entries
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_q: [batch_size, 3, H, W] - query images
            x_k: [batch_size, 3, H, W] - key images
        Returns:
            logits: [batch_size, 1 + queue_size]
            labels: [batch_size] (all zeros)
        """
        # Query features
        q = self.encoder_q(x_q).squeeze(-1).squeeze(-1)
        q = self.projection_q(q)
        q = F.normalize(q, dim=1)

        # Key features (no gradient)
        with torch.no_grad():
            # Update key encoder
            self._momentum_update_key_encoder()

            k = self.encoder_k(x_k).squeeze(-1).squeeze(-1)
            k = self.projection_k(k)
            k = F.normalize(k, dim=1)

        # Compute logits
        # Positive pairs
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [batch_size, 1]

        # Negative pairs
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [batch_size, queue_size]

        # Concatenate
        logits = torch.cat([l_pos, l_neg], dim=1)  # [batch_size, 1 + queue_size]
        logits /= self.temperature

        # Labels: positive key is the first (index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Update queue
        self._dequeue_and_enqueue(k)

        return logits, labels


class BYOL(nn.Module):
    """
    BYOL: Bootstrap Your Own Latent
    Self-supervised learning without negative pairs
    """

    def __init__(self, base_encoder: nn.Module = None, projection_dim: int = 256,
                 hidden_dim: int = 4096, momentum: float = 0.996):
        super().__init__()

        if base_encoder is None:
            base_encoder = models.resnet50(pretrained=False)

        feature_dim = base_encoder.fc.in_features

        # Online network
        self.online_encoder = nn.Sequential(*list(base_encoder.children())[:-1])
        self.online_projector = self._build_projector(feature_dim, hidden_dim, projection_dim)
        self.online_predictor = self._build_predictor(projection_dim, hidden_dim, projection_dim)

        # Target network
        self.target_encoder = nn.Sequential(*list(base_encoder.children())[:-1])
        self.target_projector = self._build_projector(feature_dim, hidden_dim, projection_dim)

        # Initialize target network
        for param_o, param_t in zip(self.online_encoder.parameters(),
                                    self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

        for param_o, param_t in zip(self.online_projector.parameters(),
                                    self.target_projector.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

        self.momentum = momentum

    def _build_projector(self, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def _build_predictor(self, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    @torch.no_grad()
    def _update_target_network(self):
        """EMA update of target network"""
        for param_o, param_t in zip(self.online_encoder.parameters(),
                                    self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1.0 - self.momentum)

        for param_o, param_t in zip(self.online_projector.parameters(),
                                    self.target_projector.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1.0 - self.momentum)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: [batch_size, 3, H, W] - augmented view 1
            x2: [batch_size, 3, H, W] - augmented view 2
        Returns:
            loss: scalar
        """
        # Online network forward pass
        z1_online = self.online_encoder(x1).squeeze(-1).squeeze(-1)
        z1_online = self.online_projector(z1_online)
        p1 = self.online_predictor(z1_online)

        z2_online = self.online_encoder(x2).squeeze(-1).squeeze(-1)
        z2_online = self.online_projector(z2_online)
        p2 = self.online_predictor(z2_online)

        # Target network forward pass (no gradient)
        with torch.no_grad():
            self._update_target_network()

            z1_target = self.target_encoder(x1).squeeze(-1).squeeze(-1)
            z1_target = self.target_projector(z1_target)

            z2_target = self.target_encoder(x2).squeeze(-1).squeeze(-1)
            z2_target = self.target_projector(z2_target)

        # Compute loss (symmetrized)
        loss = self._regression_loss(p1, z2_target) + self._regression_loss(p2, z1_target)

        return loss

    @staticmethod
    def _regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean squared error with normalized inputs"""
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1).mean()


class SwAV(nn.Module):
    """
    SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
    """

    def __init__(self, base_encoder: nn.Module = None, projection_dim: int = 128,
                 num_prototypes: int = 3000, temperature: float = 0.1):
        super().__init__()

        if base_encoder is None:
            base_encoder = models.resnet50(pretrained=False)

        self.encoder = nn.Sequential(*list(base_encoder.children())[:-1])
        feature_dim = base_encoder.fc.in_features

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

        # Prototypes
        self.prototypes = nn.Linear(projection_dim, num_prototypes, bias=False)

        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, 3, H, W]
        Returns:
            embeddings: [batch_size, projection_dim]
            prototype_scores: [batch_size, num_prototypes]
        """
        features = self.encoder(x).squeeze(-1).squeeze(-1)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, dim=1)

        # Compute prototype scores
        prototype_scores = self.prototypes(embeddings) / self.temperature

        return embeddings, prototype_scores

    @torch.no_grad()
    def sinkhorn(self, scores: torch.Tensor, num_iters: int = 3) -> torch.Tensor:
        """Sinkhorn-Knopp algorithm for optimal transport"""
        Q = torch.exp(scores).T  # [num_prototypes, batch_size]
        Q /= Q.sum()

        K, B = Q.shape

        for _ in range(num_iters):
            # Normalize rows
            Q /= Q.sum(dim=1, keepdim=True)
            # Normalize columns
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.T


def main():
    """Main execution"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    batch_size = 32

    # SimCLR
    print("\n=== SimCLR ===")
    simclr = SimCLR(projection_dim=128).to(device)
    x1 = torch.randn(batch_size, 3, 224, 224).to(device)
    x2 = torch.randn(batch_size, 3, 224, 224).to(device)

    with torch.no_grad():
        features1, proj1 = simclr(x1)
        features2, proj2 = simclr(x2)

    criterion = NTXentLoss(temperature=0.5)
    loss = criterion(proj1, proj2)
    print(f"SimCLR loss: {loss.item():.4f}")

    # MoCo
    print("\n=== MoCo ===")
    moco = MoCo(projection_dim=128, queue_size=4096).to(device)

    with torch.no_grad():
        logits, labels = moco(x1, x2)
    print(f"MoCo logits shape: {logits.shape}")

    # BYOL
    print("\n=== BYOL ===")
    byol = BYOL(projection_dim=256).to(device)

    with torch.no_grad():
        loss = byol(x1, x2)
    print(f"BYOL loss: {loss.item():.4f}")

    # SwAV
    print("\n=== SwAV ===")
    swav = SwAV(projection_dim=128, num_prototypes=1000).to(device)

    with torch.no_grad():
        embeddings, scores = swav(x1)
        codes = swav.sinkhorn(scores)
    print(f"SwAV embeddings shape: {embeddings.shape}")
    print(f"SwAV codes shape: {codes.shape}")

    print("\nSelf-supervised learning ready!")


if __name__ == '__main__':
    main()
