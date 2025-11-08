"""
Image Captioning System
Generate natural language descriptions of images using encoder-decoder architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import math


class ImageEncoder(nn.Module):
    """CNN-based image encoder"""

    def __init__(self, embed_size: int = 512, pretrained: bool = True):
        super().__init__()
        # Use ResNet-101 as backbone
        resnet = models.resnet101(pretrained=pretrained)

        # Remove final FC and avgpool layers
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

        # Feature dimension
        self.feature_dim = 2048

        # Projection layer
        self.projection = nn.Linear(self.feature_dim, embed_size)

        # Batch normalization
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, H, W]
        Returns:
            features: [batch_size, num_pixels, embed_size]
        """
        with torch.no_grad():
            features = self.cnn(images)  # [batch, 2048, H', W']

        features = self.adaptive_pool(features)  # [batch, 2048, 14, 14]

        # Reshape to [batch, num_pixels, feature_dim]
        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1)  # [batch, 14, 14, 2048]
        features = features.view(batch_size, -1, self.feature_dim)  # [batch, 196, 2048]

        # Project features
        features = self.projection(features)  # [batch, 196, embed_size]

        return features


class AttentionDecoder(nn.Module):
    """
    LSTM decoder with attention mechanism (Show, Attend and Tell)
    """

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int,
                 num_layers: int = 1, dropout: float = 0.5):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM decoder
        self.lstm = nn.LSTMCell(embed_size + hidden_size, hidden_size)

        # Attention mechanism
        self.attention = BahdanauAttention(hidden_size, embed_size)

        # Output layers
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize LSTM state
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

        # Gating mechanism for attention
        self.f_beta = nn.Linear(hidden_size, embed_size)
        self.sigmoid = nn.Sigmoid()

    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state from encoder output"""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor,
                caption_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_out: [batch_size, num_pixels, embed_size]
            captions: [batch_size, max_caption_length]
            caption_lengths: [batch_size]
        Returns:
            predictions: [batch_size, max_caption_length, vocab_size]
            alphas: [batch_size, max_caption_length, num_pixels]
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Sort by caption length (for packing)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]

        # Embedding
        embeddings = self.embedding(captions)  # [batch, max_len, embed_size]

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # We won't decode at the <end> position
        decode_lengths = (caption_lengths - 1).tolist()

        # Storage
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # Decode step by step
        for t in range(max(decode_lengths)):
            # Find active sequences
            batch_size_t = sum([l > t for l in decode_lengths])

            # Attention
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t]
            )

            # Gating scalar
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # LSTM step
            h, c = self.lstm(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            # Predict next word
            preds = self.fc(self.dropout(h))

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas, captions, decode_lengths, sort_ind

    def sample(self, encoder_out: torch.Tensor, max_length: int = 20,
               start_token: int = 1) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Generate caption using greedy search
        Args:
            encoder_out: [1, num_pixels, embed_size]
            max_length: Maximum caption length
            start_token: Start token ID
        Returns:
            caption: List of token IDs
            alphas: List of attention weights
        """
        caption = []
        alphas = []

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # Start with start token
        word = torch.LongTensor([start_token]).to(encoder_out.device)

        for t in range(max_length):
            # Embed current word
            embeddings = self.embedding(word)

            # Attention
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)

            # Gating
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # LSTM step
            h, c = self.lstm(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c)
            )

            # Predict next word
            scores = self.fc(h)
            word = scores.argmax(dim=1)

            caption.append(word.item())
            alphas.append(alpha.cpu())

            # Stop if <end> token
            if word.item() == 2:  # Assuming 2 is <end> token
                break

        return caption, alphas


class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism"""

    def __init__(self, hidden_size: int, encoder_size: int):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_size, hidden_size)
        self.decoder_att = nn.Linear(hidden_size, hidden_size)
        self.full_att = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor,
                decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_out: [batch_size, num_pixels, encoder_size]
            decoder_hidden: [batch_size, hidden_size]
        Returns:
            attention_weighted_encoding: [batch_size, encoder_size]
            alpha: [batch_size, num_pixels]
        """
        att1 = self.encoder_att(encoder_out)  # [batch, num_pixels, hidden_size]
        att2 = self.decoder_att(decoder_hidden)  # [batch, hidden_size]
        att2 = att2.unsqueeze(1)  # [batch, 1, hidden_size]

        # Compute attention scores
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # [batch, num_pixels]
        alpha = self.softmax(att)  # [batch, num_pixels]

        # Apply attention
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for image captioning
    """

    def __init__(self, embed_size: int, vocab_size: int, num_heads: int = 8,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_size, dropout)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.fc_out = nn.Linear(embed_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            encoder_out: [batch_size, num_pixels, embed_size]
            captions: [batch_size, seq_len]
            tgt_mask: [seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, vocab_size]
        """
        # Embed captions
        tgt_embed = self.embedding(captions) * math.sqrt(self.embed_size)
        tgt_embed = self.pos_encoder(tgt_embed)

        # Create causal mask
        if tgt_mask is None:
            seq_len = captions.size(1)
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)

        # Decode
        output = self.transformer_decoder(
            tgt=tgt_embed,
            memory=encoder_out,
            tgt_mask=tgt_mask
        )

        # Project to vocabulary
        output = self.fc_out(output)

        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def sample(self, encoder_out: torch.Tensor, max_length: int = 20,
               start_token: int = 1, end_token: int = 2) -> List[int]:
        """Generate caption using greedy decoding"""
        device = encoder_out.device
        caption = [start_token]

        for _ in range(max_length):
            # Prepare input
            tgt = torch.LongTensor([caption]).to(device)

            # Generate mask
            tgt_mask = self.generate_square_subsequent_mask(len(caption)).to(device)

            # Forward pass
            with torch.no_grad():
                output = self.forward(encoder_out, tgt, tgt_mask)

            # Get next token
            next_token = output[0, -1, :].argmax().item()
            caption.append(next_token)

            # Stop if end token
            if next_token == end_token:
                break

        return caption


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ImageCaptioningModel(nn.Module):
    """Complete image captioning model"""

    def __init__(self, vocab_size: int, embed_size: int = 512,
                 hidden_size: int = 512, decoder_type: str = 'lstm'):
        super().__init__()
        self.decoder_type = decoder_type

        # Encoder
        self.encoder = ImageEncoder(embed_size=embed_size, pretrained=True)

        # Decoder
        if decoder_type == 'lstm':
            self.decoder = AttentionDecoder(
                embed_size=embed_size,
                hidden_size=hidden_size,
                vocab_size=vocab_size
            )
        elif decoder_type == 'transformer':
            self.decoder = TransformerDecoder(
                embed_size=embed_size,
                vocab_size=vocab_size,
                num_heads=8,
                num_layers=6
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def forward(self, images: torch.Tensor, captions: torch.Tensor,
                caption_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Encode images
        encoder_out = self.encoder(images)

        # Decode
        if self.decoder_type == 'lstm':
            predictions, alphas, captions, decode_lengths, sort_ind = self.decoder(
                encoder_out, captions, caption_lengths
            )
            return predictions, alphas, captions, decode_lengths, sort_ind
        else:
            predictions = self.decoder(encoder_out, captions)
            return predictions

    def sample(self, images: torch.Tensor, max_length: int = 20) -> List[int]:
        """Generate caption for image"""
        encoder_out = self.encoder(images)

        if self.decoder_type == 'lstm':
            caption, alphas = self.decoder.sample(encoder_out, max_length)
            return caption
        else:
            caption = self.decoder.sample(encoder_out, max_length)
            return caption


class CaptionDataset(Dataset):
    """Dataset for image captioning"""

    def __init__(self, images: np.ndarray, captions: List[List[int]],
                 transform=None):
        self.images = images
        self.captions = captions
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        caption = self.captions[idx]

        if self.transform:
            image = self.transform(image)

        caption = torch.LongTensor(caption)

        return image, caption


def main():
    """Main execution"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    vocab_size = 10000

    # Create LSTM-based model
    print("\n=== LSTM-based Caption Model ===")
    lstm_model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_size=512,
        hidden_size=512,
        decoder_type='lstm'
    )
    lstm_model = lstm_model.to(device)

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    captions = torch.randint(1, vocab_size, (batch_size, 20)).to(device)
    caption_lengths = torch.LongTensor([20, 15])

    with torch.no_grad():
        predictions, alphas, _, _, _ = lstm_model(images, captions, caption_lengths)

    print(f"LSTM predictions shape: {predictions.shape}")
    print(f"Attention weights shape: {alphas.shape}")

    # Test sampling
    sample_image = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        generated_caption = lstm_model.sample(sample_image, max_length=15)
    print(f"Generated caption: {generated_caption}")

    # Create Transformer-based model
    print("\n=== Transformer-based Caption Model ===")
    transformer_model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_size=512,
        decoder_type='transformer'
    )
    transformer_model = transformer_model.to(device)

    with torch.no_grad():
        predictions = transformer_model(images, captions)

    print(f"Transformer predictions shape: {predictions.shape}")

    # Test sampling
    with torch.no_grad():
        generated_caption = transformer_model.sample(sample_image, max_length=15)
    print(f"Generated caption: {generated_caption}")

    print("\nImage captioning system ready!")


if __name__ == '__main__':
    main()
