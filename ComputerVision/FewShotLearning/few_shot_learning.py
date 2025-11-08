"""
Few-Shot Learning System
Learn from limited examples using meta-learning and metric learning approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import OrderedDict
import random


class ConvBlock(nn.Module):
    """Convolutional block for feature extraction"""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.pool(x)
        return x


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for Few-Shot Learning
    Learn a metric space where classification is based on distance to class prototypes
    """

    def __init__(self, input_channels: int = 3, hidden_size: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(input_channels, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size, pool=False)
        )

    def forward(self, support: torch.Tensor, query: torch.Tensor,
                n_way: int, k_shot: int) -> torch.Tensor:
        """
        Args:
            support: Support set [n_way * k_shot, C, H, W]
            query: Query set [n_query, C, H, W]
            n_way: Number of classes
            k_shot: Number of examples per class
        """
        # Encode support and query
        support_features = self.encoder(support)  # [n_way*k_shot, D, H', W']
        query_features = self.encoder(query)  # [n_query, D, H', W']

        # Global average pooling
        support_features = F.adaptive_avg_pool2d(support_features, 1).squeeze(-1).squeeze(-1)
        query_features = F.adaptive_avg_pool2d(query_features, 1).squeeze(-1).squeeze(-1)

        # Compute prototypes (class centroids)
        support_features = support_features.view(n_way, k_shot, -1)
        prototypes = support_features.mean(dim=1)  # [n_way, D]

        # Compute distances
        distances = self._euclidean_distance(query_features, prototypes)

        # Convert to probabilities (negative distance)
        logits = -distances
        return logits

    @staticmethod
    def _euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute euclidean distance between x and y"""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)


class RelationNetwork(nn.Module):
    """
    Relation Network for Few-Shot Learning
    Learn to compare query and support using a learned relation module
    """

    def __init__(self, input_channels: int = 3, hidden_size: int = 64):
        super().__init__()

        # Feature embedding
        self.encoder = nn.Sequential(
            ConvBlock(input_channels, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size, pool=False)
        )

        # Relation module
        self.relation = nn.Sequential(
            ConvBlock(hidden_size * 2, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            nn.Flatten(),
            nn.Linear(hidden_size * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, support: torch.Tensor, query: torch.Tensor,
                n_way: int, k_shot: int) -> torch.Tensor:
        # Encode
        support_features = self.encoder(support)  # [n_way*k_shot, D, H, W]
        query_features = self.encoder(query)  # [n_query, D, H, W]

        # Compute prototypes
        support_features = support_features.view(n_way, k_shot, *support_features.shape[1:])
        prototypes = support_features.mean(dim=1)  # [n_way, D, H, W]

        # Expand for comparison
        n_query = query_features.size(0)
        prototypes_expanded = prototypes.unsqueeze(0).expand(
            n_query, -1, -1, -1, -1
        ).contiguous().view(n_query * n_way, *prototypes.shape[1:])

        query_expanded = query_features.unsqueeze(1).expand(
            -1, n_way, -1, -1, -1
        ).contiguous().view(n_query * n_way, *query_features.shape[1:])

        # Concatenate query and support
        pairs = torch.cat([prototypes_expanded, query_expanded], dim=1)

        # Compute relations
        relations = self.relation(pairs).view(n_query, n_way)

        return relations


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML)
    Learn good initialization that can quickly adapt to new tasks
    """

    def __init__(self, input_channels: int = 3, hidden_size: int = 64, n_way: int = 5):
        super().__init__()
        self.n_way = n_way

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(input_channels, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size)
        )

        # Classifier
        self.classifier = nn.Linear(hidden_size, n_way)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        logits = self.classifier(features)
        return logits

    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor,
              steps: int = 5, lr: float = 0.01) -> 'MAML':
        """Adapt model to new task using support set"""

        # Clone parameters for adaptation
        adapted_params = OrderedDict()
        for name, param in self.named_parameters():
            adapted_params[name] = param.clone()

        # Inner loop adaptation
        for step in range(steps):
            # Forward pass with current params
            logits = self._forward_with_params(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)

            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params.values(),
                                       create_graph=True)

            # Update parameters
            adapted_params = OrderedDict(
                (name, param - lr * grad)
                for ((name, param), grad) in zip(adapted_params.items(), grads)
            )

        # Create adapted model
        adapted_model = MAML(n_way=self.n_way)
        adapted_model.load_state_dict(adapted_params)

        return adapted_model

    def _forward_with_params(self, x: torch.Tensor,
                            params: OrderedDict) -> torch.Tensor:
        """Forward pass using specific parameters"""
        # This is a simplified version - full implementation would apply params manually
        return self.forward(x)


class MatchingNetwork(nn.Module):
    """
    Matching Networks for One-Shot Learning
    Uses attention and memory to classify based on support set
    """

    def __init__(self, input_channels: int = 3, hidden_size: int = 64,
                 lstm_layers: int = 1):
        super().__init__()

        # Encoder for support set
        self.support_encoder = nn.Sequential(
            ConvBlock(input_channels, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size)
        )

        # Encoder for query set (with context)
        self.query_encoder = nn.Sequential(
            ConvBlock(input_channels, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size)
        )

        # Bidirectional LSTM for full context embedding
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, lstm_layers,
                           bidirectional=True, batch_first=True)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)

    def forward(self, support: torch.Tensor, support_labels: torch.Tensor,
                query: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        """
        Args:
            support: [n_way * k_shot, C, H, W]
            support_labels: [n_way * k_shot]
            query: [n_query, C, H, W]
        """
        # Encode support
        support_features = self.support_encoder(support)
        support_features = F.adaptive_avg_pool2d(support_features, 1).squeeze(-1).squeeze(-1)

        # Full Context Embedding with LSTM
        support_features = support_features.unsqueeze(0)  # [1, n_way*k_shot, D]
        support_features, _ = self.lstm(support_features)
        support_features = support_features.squeeze(0)  # [n_way*k_shot, D]

        # Encode query
        query_features = self.query_encoder(query)
        query_features = F.adaptive_avg_pool2d(query_features, 1).squeeze(-1).squeeze(-1)

        # Attention-based comparison
        query_features = query_features.unsqueeze(1)  # [n_query, 1, D]
        support_features = support_features.unsqueeze(0)  # [1, n_way*k_shot, D]

        # Compute attention weights (cosine similarity)
        attention_weights = F.cosine_similarity(
            query_features, support_features, dim=2
        )  # [n_query, n_way*k_shot]

        # Apply softmax
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weight the support labels
        support_labels_onehot = F.one_hot(support_labels, n_way).float()
        predictions = torch.matmul(attention_weights, support_labels_onehot)

        return predictions


class FewShotDataset(Dataset):
    """Dataset for few-shot learning episodes"""

    def __init__(self, data: np.ndarray, labels: np.ndarray,
                 n_way: int = 5, k_shot: int = 5, q_query: int = 15):
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # Organize data by class
        self.classes = np.unique(labels)
        self.class_to_indices = {
            c: np.where(labels == c)[0] for c in self.classes
        }

    def __len__(self) -> int:
        return 1000  # Number of episodes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
        """Sample a few-shot episode"""
        # Select n_way classes
        selected_classes = random.sample(list(self.classes), self.n_way)

        support_x = []
        support_y = []
        query_x = []
        query_y = []

        for i, cls in enumerate(selected_classes):
            # Get indices for this class
            indices = self.class_to_indices[cls]

            # Sample k_shot + q_query examples
            selected_indices = np.random.choice(
                indices, self.k_shot + self.q_query, replace=False
            )

            # Split into support and query
            support_indices = selected_indices[:self.k_shot]
            query_indices = selected_indices[self.k_shot:]

            support_x.append(self.data[support_indices])
            support_y.extend([i] * self.k_shot)

            query_x.append(self.data[query_indices])
            query_y.extend([i] * self.q_query)

        # Convert to tensors
        support_x = torch.FloatTensor(np.concatenate(support_x))
        support_y = torch.LongTensor(support_y)
        query_x = torch.FloatTensor(np.concatenate(query_x))
        query_y = torch.LongTensor(query_y)

        return support_x, support_y, query_x, query_y


class FewShotTrainer:
    """Trainer for few-shot learning models"""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

    def train_prototypical(self, train_loader: DataLoader, val_loader: DataLoader,
                          n_way: int, k_shot: int, epochs: int = 100):
        """Train Prototypical Network"""
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        best_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0

            for support_x, support_y, query_x, query_y in train_loader:
                support_x = support_x.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits = self.model(support_x, query_x, n_way, k_shot)
                loss = F.cross_entropy(logits, query_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Metrics
                train_loss += loss.item()
                acc = (logits.argmax(dim=1) == query_y).float().mean()
                train_acc += acc.item()

            scheduler.step()

            # Validation
            val_acc = self.evaluate(val_loader, n_way, k_shot)

            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                  f"Train Acc = {train_acc/len(train_loader):.4f}, "
                  f"Val Acc = {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')

        return best_acc

    def evaluate(self, data_loader: DataLoader, n_way: int, k_shot: int) -> float:
        """Evaluate model"""
        self.model.eval()
        total_acc = 0.0

        with torch.no_grad():
            for support_x, support_y, query_x, query_y in data_loader:
                support_x = support_x.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                logits = self.model(support_x, query_x, n_way, k_shot)
                acc = (logits.argmax(dim=1) == query_y).float().mean()
                total_acc += acc.item()

        return total_acc / len(data_loader)


def main():
    """Main execution"""
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Parameters
    n_way = 5  # 5-way classification
    k_shot = 5  # 5-shot learning
    q_query = 15  # 15 query examples per class

    # Create dummy data (replace with real dataset)
    np.random.seed(42)
    train_data = np.random.randn(1000, 3, 84, 84).astype(np.float32)
    train_labels = np.random.randint(0, 20, 1000)

    val_data = np.random.randn(200, 3, 84, 84).astype(np.float32)
    val_labels = np.random.randint(0, 20, 200)

    # Create datasets
    train_dataset = FewShotDataset(train_data, train_labels, n_way, k_shot, q_query)
    val_dataset = FewShotDataset(val_data, val_labels, n_way, k_shot, q_query)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Train Prototypical Network
    print("\n=== Training Prototypical Network ===")
    proto_net = PrototypicalNetwork(input_channels=3, hidden_size=64)
    trainer = FewShotTrainer(proto_net, device=device)
    best_acc = trainer.train_prototypical(train_loader, val_loader, n_way, k_shot, epochs=10)

    print(f"\nBest validation accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()
