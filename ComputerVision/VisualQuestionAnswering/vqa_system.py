"""
Visual Question Answering (VQA) System
Answer natural language questions about images using multi-modal learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from collections import Counter


class ImageEncoder(nn.Module):
    """Encode images using pre-trained CNN"""

    def __init__(self, embed_size: int = 512, pretrained: bool = True):
        super().__init__()
        # Use ResNet as backbone
        resnet = models.resnet50(pretrained=pretrained)

        # Remove final FC layer
        modules = list(resnet.children())[:-2]  # Keep until avgpool
        self.cnn = nn.Sequential(*modules)

        # Spatial features dimension
        self.feature_dim = 2048

        # Project to embedding size
        self.projection = nn.Sequential(
            nn.Conv2d(self.feature_dim, embed_size, 1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [batch_size, 3, H, W]
        Returns:
            spatial_features: [batch_size, embed_size, H', W']
            global_features: [batch_size, embed_size]
        """
        # Extract features
        features = self.cnn(images)  # [batch, 2048, H', W']

        # Project features
        spatial_features = self.projection(features)  # [batch, embed_size, H', W']

        # Global features via average pooling
        global_features = F.adaptive_avg_pool2d(spatial_features, 1).squeeze(-1).squeeze(-1)

        return spatial_features, global_features


class QuestionEncoder(nn.Module):
    """Encode questions using LSTM"""

    def __init__(self, vocab_size: int, embed_size: int = 300,
                 hidden_size: int = 512, num_layers: int = 2):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # LSTM encoder
        self.lstm = nn.LSTM(
            embed_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        # Project bidirectional to single direction
        self.projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, questions: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            questions: [batch_size, seq_len]
            lengths: [batch_size]
        Returns:
            sequence_output: [batch_size, seq_len, hidden_size]
            final_hidden: [batch_size, hidden_size]
        """
        # Embed words
        embedded = self.embedding(questions)  # [batch, seq_len, embed_size]

        # LSTM encoding
        if lengths is not None:
            # Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (hidden, cell) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)

        # Project output
        output = self.projection(output)  # [batch, seq_len, hidden_size]

        # Final hidden state (concatenate forward and backward)
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        final_hidden = self.projection(final_hidden)  # [batch, hidden_size]

        return output, final_hidden


class AttentionModule(nn.Module):
    """Multi-modal attention mechanism"""

    def __init__(self, image_dim: int, question_dim: int, hidden_dim: int):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, image_features: torch.Tensor,
                question_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [batch, num_regions, image_dim]
            question_features: [batch, question_dim]
        Returns:
            attended_features: [batch, image_dim]
        """
        # Project image features
        image_proj = self.image_proj(image_features)  # [batch, num_regions, hidden_dim]

        # Project and expand question features
        question_proj = self.question_proj(question_features)  # [batch, hidden_dim]
        question_proj = question_proj.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Compute attention scores
        combined = torch.tanh(image_proj + question_proj)  # [batch, num_regions, hidden_dim]
        attention_scores = self.attention(combined).squeeze(-1)  # [batch, num_regions]

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, num_regions]

        # Apply attention to image features
        attention_weights = attention_weights.unsqueeze(-1)  # [batch, num_regions, 1]
        attended = (image_features * attention_weights).sum(dim=1)  # [batch, image_dim]

        return self.dropout(attended)


class StackedAttention(nn.Module):
    """Stacked Attention Networks (SAN)"""

    def __init__(self, image_dim: int, question_dim: int, hidden_dim: int, num_stacks: int = 2):
        super().__init__()
        self.num_stacks = num_stacks

        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            AttentionModule(image_dim, question_dim, hidden_dim)
            for _ in range(num_stacks)
        ])

        # Question refinement
        self.question_refine = nn.ModuleList([
            nn.Linear(question_dim + image_dim, question_dim)
            for _ in range(num_stacks - 1)
        ])

    def forward(self, image_features: torch.Tensor,
                question_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [batch, num_regions, image_dim]
            question_features: [batch, question_dim]
        Returns:
            attended_features: [batch, image_dim]
        """
        current_question = question_features

        for i, attention_layer in enumerate(self.attention_layers):
            # Apply attention
            attended = attention_layer(image_features, current_question)

            # Refine question for next layer
            if i < self.num_stacks - 1:
                combined = torch.cat([current_question, attended], dim=1)
                current_question = self.question_refine[i](combined)
                current_question = F.relu(current_question)

        return attended


class BilinearPooling(nn.Module):
    """Multi-modal Compact Bilinear Pooling"""

    def __init__(self, dim1: int, dim2: int, output_dim: int):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim

        # Projection matrices
        self.proj1 = nn.Linear(dim1, output_dim)
        self.proj2 = nn.Linear(dim2, output_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: [batch, dim1]
            x2: [batch, dim2]
        Returns:
            pooled: [batch, output_dim]
        """
        # Project inputs
        p1 = self.proj1(x1)
        p2 = self.proj2(x2)

        # Element-wise multiplication
        pooled = p1 * p2

        return pooled


class VQAModel(nn.Module):
    """Complete VQA Model with Attention"""

    def __init__(self, vocab_size: int, num_answers: int, embed_size: int = 512,
                 hidden_size: int = 512, num_attention_stacks: int = 2):
        super().__init__()

        # Encoders
        self.image_encoder = ImageEncoder(embed_size=embed_size)
        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size,
            embed_size=300,
            hidden_size=hidden_size
        )

        # Attention mechanism
        self.attention = StackedAttention(
            image_dim=embed_size,
            question_dim=hidden_size,
            hidden_dim=hidden_size,
            num_stacks=num_attention_stacks
        )

        # Multi-modal fusion
        self.fusion = BilinearPooling(embed_size, hidden_size, hidden_size)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_answers)
        )

    def forward(self, images: torch.Tensor, questions: torch.Tensor,
                question_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, H, W]
            questions: [batch_size, seq_len]
            question_lengths: [batch_size]
        Returns:
            logits: [batch_size, num_answers]
        """
        # Encode image
        spatial_features, global_features = self.image_encoder(images)

        # Reshape spatial features for attention
        batch_size, channels, h, w = spatial_features.shape
        spatial_features = spatial_features.view(batch_size, channels, -1)
        spatial_features = spatial_features.permute(0, 2, 1)  # [batch, num_regions, channels]

        # Encode question
        question_sequence, question_features = self.question_encoder(
            questions, question_lengths
        )

        # Apply attention
        attended_image = self.attention(spatial_features, question_features)

        # Multi-modal fusion
        fused = self.fusion(attended_image, question_features)

        # Classify
        logits = self.classifier(fused)

        return logits


class TransformerVQA(nn.Module):
    """Transformer-based VQA Model"""

    def __init__(self, vocab_size: int, num_answers: int, d_model: int = 512,
                 nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model

        # Image encoder
        self.image_encoder = ImageEncoder(embed_size=d_model)

        # Question embedding
        self.question_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = self._create_positional_encoding(1000, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_answers)
        )

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, H, W]
            questions: [batch_size, seq_len]
        Returns:
            logits: [batch_size, num_answers]
        """
        batch_size = images.size(0)
        device = images.device

        # Encode image
        spatial_features, _ = self.image_encoder(images)

        # Reshape image features as sequence
        _, channels, h, w = spatial_features.shape
        image_seq = spatial_features.view(batch_size, channels, -1)
        image_seq = image_seq.permute(0, 2, 1)  # [batch, num_patches, d_model]

        # Embed questions
        question_embedded = self.question_embedding(questions)  # [batch, seq_len, d_model]

        # Add positional encoding
        seq_len = questions.size(1)
        pos_encoding = self.pos_encoding[:seq_len].to(device)
        question_embedded = question_embedded + pos_encoding

        # Concatenate image and question sequences
        combined_seq = torch.cat([image_seq, question_embedded], dim=1)

        # Apply transformer
        transformed = self.transformer(combined_seq)

        # Use CLS token (first token) for classification
        cls_token = transformed[:, 0, :]

        # Classify
        logits = self.classifier(cls_token)

        return logits


class VQADataset(Dataset):
    """Dataset for VQA"""

    def __init__(self, images: np.ndarray, questions: List[str],
                 answers: List[str], vocab: Dict[str, int], answer_to_idx: Dict[str, int]):
        self.images = images
        self.questions = questions
        self.answers = answers
        self.vocab = vocab
        self.answer_to_idx = answer_to_idx

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image = torch.FloatTensor(self.images[idx])
        question = self._encode_question(self.questions[idx])
        answer = self.answer_to_idx.get(self.answers[idx], 0)

        return image, question, answer

    def _encode_question(self, question: str) -> torch.Tensor:
        """Encode question as token IDs"""
        words = question.lower().split()
        tokens = [self.vocab.get(word, 1) for word in words]  # 1 for UNK
        return torch.LongTensor(tokens)


def collate_fn(batch):
    """Custom collate function for variable-length questions"""
    images, questions, answers = zip(*batch)

    # Stack images
    images = torch.stack(images)

    # Pad questions
    lengths = torch.LongTensor([len(q) for q in questions])
    max_len = lengths.max()
    padded_questions = torch.zeros(len(questions), max_len, dtype=torch.long)

    for i, q in enumerate(questions):
        padded_questions[i, :len(q)] = q

    answers = torch.LongTensor(answers)

    return images, padded_questions, lengths, answers


def main():
    """Main execution"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create dummy data
    vocab_size = 10000
    num_answers = 1000

    # Create model
    print("Creating VQA model...")
    model = VQAModel(
        vocab_size=vocab_size,
        num_answers=num_answers,
        embed_size=512,
        hidden_size=512,
        num_attention_stacks=2
    )
    model = model.to(device)

    # Create dummy inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    questions = torch.randint(0, vocab_size, (batch_size, 20)).to(device)
    question_lengths = torch.LongTensor([20, 18, 15, 12])

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        logits = model(images, questions, question_lengths)
        predictions = logits.argmax(dim=1)

    print(f"Output shape: {logits.shape}")
    print(f"Predictions: {predictions}")

    # Transformer model
    print("\nCreating Transformer VQA model...")
    transformer_model = TransformerVQA(
        vocab_size=vocab_size,
        num_answers=num_answers,
        d_model=512,
        nhead=8,
        num_layers=6
    )
    transformer_model = transformer_model.to(device)

    with torch.no_grad():
        logits = transformer_model(images, questions)
        predictions = logits.argmax(dim=1)

    print(f"Transformer output shape: {logits.shape}")
    print(f"Transformer predictions: {predictions}")

    print("\nVQA system ready!")


if __name__ == '__main__':
    main()
