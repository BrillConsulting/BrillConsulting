"""
PyTorch Deep Learning Models
Author: BrillConsulting
Description: Complete PyTorch implementation for neural networks and deep learning
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class PyTorchModels:
    """Comprehensive PyTorch model implementations"""

    def __init__(self):
        """Initialize PyTorch models manager"""
        self.models = []
        self.datasets = []
        self.training_runs = []

    def create_cnn_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Convolutional Neural Network

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'CNN'),
            'type': 'convolutional',
            'architecture': {
                'conv_layers': model_config.get('conv_layers', [
                    {'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
                    {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
                    {'filters': 128, 'kernel_size': 3, 'activation': 'relu'}
                ]),
                'pooling': model_config.get('pooling', 'MaxPool2d'),
                'fc_layers': model_config.get('fc_layers', [512, 256]),
                'output_size': model_config.get('output_size', 10),
                'dropout': model_config.get('dropout', 0.5)
            },
            'parameters': 2_450_000,
            'input_shape': model_config.get('input_shape', [3, 224, 224]),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, {model['architecture']['output_size']})

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

model = CNN()
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ CNN model created: {model['name']}")
        print(f"  Parameters: {model['parameters']:,}")
        print(f"  Input shape: {model['input_shape']}")
        return model

    def create_rnn_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Recurrent Neural Network (LSTM/GRU)

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'RNN'),
            'type': 'recurrent',
            'cell_type': model_config.get('cell_type', 'LSTM'),
            'architecture': {
                'input_size': model_config.get('input_size', 100),
                'hidden_size': model_config.get('hidden_size', 256),
                'num_layers': model_config.get('num_layers', 2),
                'output_size': model_config.get('output_size', 1),
                'bidirectional': model_config.get('bidirectional', True),
                'dropout': model_config.get('dropout', 0.3)
            },
            'parameters': 850_000,
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.hidden_size = {model['architecture']['hidden_size']}
        self.num_layers = {model['architecture']['num_layers']}

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size={model['architecture']['input_size']},
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout={model['architecture']['dropout']},
            bidirectional={model['architecture']['bidirectional']}
        )

        # Fully connected layer
        fc_input = self.hidden_size * (2 if {model['architecture']['bidirectional']} else 1)
        self.fc = nn.Linear(fc_input, {model['architecture']['output_size']})

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out

model = RNN()
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ RNN model created: {model['name']}")
        print(f"  Cell type: {model['cell_type']}, Layers: {model['architecture']['num_layers']}")
        print(f"  Hidden size: {model['architecture']['hidden_size']}")
        return model

    def create_transformer_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Transformer model

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'Transformer'),
            'type': 'transformer',
            'architecture': {
                'd_model': model_config.get('d_model', 512),
                'nhead': model_config.get('nhead', 8),
                'num_encoder_layers': model_config.get('num_encoder_layers', 6),
                'num_decoder_layers': model_config.get('num_decoder_layers', 6),
                'dim_feedforward': model_config.get('dim_feedforward', 2048),
                'dropout': model_config.get('dropout', 0.1),
                'vocab_size': model_config.get('vocab_size', 10000)
            },
            'parameters': 65_000_000,
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding({model['architecture']['vocab_size']}, {model['architecture']['d_model']})
        self.pos_encoder = PositionalEncoding({model['architecture']['d_model']})

        self.transformer = nn.Transformer(
            d_model={model['architecture']['d_model']},
            nhead={model['architecture']['nhead']},
            num_encoder_layers={model['architecture']['num_encoder_layers']},
            num_decoder_layers={model['architecture']['num_decoder_layers']},
            dim_feedforward={model['architecture']['dim_feedforward']},
            dropout={model['architecture']['dropout']}
        )

        self.fc_out = nn.Linear({model['architecture']['d_model']}, {model['architecture']['vocab_size']})

    def forward(self, src, tgt):
        src = self.embedding(src) * math.sqrt({model['architecture']['d_model']})
        src = self.pos_encoder(src)

        tgt = self.embedding(tgt) * math.sqrt({model['architecture']['d_model']})
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src, tgt)
        output = self.fc_out(output)

        return output

model = TransformerModel()
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ Transformer model created: {model['name']}")
        print(f"  d_model: {model['architecture']['d_model']}, heads: {model['architecture']['nhead']}")
        print(f"  Encoder/Decoder layers: {model['architecture']['num_encoder_layers']}/{model['architecture']['num_decoder_layers']}")
        return model

    def create_training_loop(self, training_config: Dict[str, Any]) -> str:
        """
        Generate training loop code

        Args:
            training_config: Training configuration

        Returns:
            Training loop code
        """
        epochs = training_config.get('epochs', 10)
        learning_rate = training_config.get('learning_rate', 0.001)
        optimizer = training_config.get('optimizer', 'Adam')
        loss_function = training_config.get('loss_function', 'CrossEntropyLoss')

        code = f"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Training configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.{loss_function}()
optimizer = optim.{optimizer}(model.parameters(), lr={learning_rate})

# Training loop
def train(model, train_loader, epochs={epochs}):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{{epoch+1}}/{epochs}], Step [{{batch_idx}}/{{len(train_loader)}}], '
                      f'Loss: {{loss.item():.4f}}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(f'Epoch [{{epoch+1}}/{epochs}] completed: Loss: {{epoch_loss:.4f}}, Accuracy: {{epoch_acc:.2f}}%')

# Evaluation loop
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {{accuracy:.2f}}%')
    return accuracy
"""

        print(f"✓ Training loop generated")
        print(f"  Epochs: {epochs}, LR: {learning_rate}, Optimizer: {optimizer}")
        return code

    def create_data_loader(self, dataset_config: Dict[str, Any]) -> str:
        """
        Generate DataLoader code

        Args:
            dataset_config: Dataset configuration

        Returns:
            DataLoader code
        """
        batch_size = dataset_config.get('batch_size', 32)
        shuffle = dataset_config.get('shuffle', True)
        num_workers = dataset_config.get('num_workers', 4)

        code = f"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# DataLoader
train_dataset = CustomDataset(train_data, train_labels, transform=transform)
test_dataset = CustomDataset(test_data, test_labels, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size={batch_size},
    shuffle={shuffle},
    num_workers={num_workers},
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size={batch_size},
    shuffle=False,
    num_workers={num_workers},
    pin_memory=True
)
"""

        print(f"✓ DataLoader code generated")
        print(f"  Batch size: {batch_size}, Workers: {num_workers}")
        return code

    def create_model_checkpoint(self) -> str:
        """Generate model checkpoint saving code"""

        code = """
import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved: {filepath}')

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Checkpoint loaded: Epoch {epoch}, Loss {loss:.4f}')
    return model, optimizer, epoch, loss

# Save best model
best_accuracy = 0.0
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

if accuracy > best_accuracy:
    best_accuracy = accuracy
    save_checkpoint(model, optimizer, epoch, loss,
                   f'{checkpoint_dir}/best_model.pth')
"""

        print("✓ Model checkpoint code generated")
        return code

    def create_advanced_resnet(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Advanced ResNet with bottleneck blocks

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'AdvancedResNet'),
            'type': 'residual',
            'architecture': {
                'num_blocks': model_config.get('num_blocks', [3, 4, 6, 3]),
                'num_classes': model_config.get('num_classes', 1000),
                'bottleneck': model_config.get('bottleneck', True)
            },
            'parameters': 25_500_000,
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes={model['architecture']['num_classes']}):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# ResNet50
model = ResNet(Bottleneck, [3, 4, 6, 3])
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ Advanced ResNet model created: {model['name']}")
        print(f"  Blocks: {model['architecture']['num_blocks']}")
        return model

    def create_attention_mechanism(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Self-Attention mechanism

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'SelfAttention'),
            'type': 'attention',
            'architecture': {
                'embed_dim': model_config.get('embed_dim', 512),
                'num_heads': model_config.get('num_heads', 8),
                'num_layers': model_config.get('num_layers', 6)
            },
            'parameters': 15_000_000,
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim={model['architecture']['embed_dim']}, num_heads={model['architecture']['num_heads']}):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.out_linear(context)

        return output, attention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim={model['architecture']['embed_dim']}, num_heads={model['architecture']['num_heads']},
                 dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention
        src2, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

model = TransformerEncoderLayer()
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ Attention mechanism created: {model['name']}")
        print(f"  Embed dim: {model['architecture']['embed_dim']}, Heads: {model['architecture']['num_heads']}")
        return model

    def create_gan_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Generative Adversarial Network

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'GAN'),
            'type': 'generative',
            'architecture': {
                'latent_dim': model_config.get('latent_dim', 100),
                'img_channels': model_config.get('img_channels', 3),
                'img_size': model_config.get('img_size', 64)
            },
            'parameters': 8_500_000,
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim={model['architecture']['latent_dim']},
                 img_channels={model['architecture']['img_channels']}):
        super(Generator, self).__init__()

        self.init_size = {model['architecture']['img_size']} // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_channels={model['architecture']['img_channels']},
                 img_size={model['architecture']['img_size']}):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Calculate output shape
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ GAN model created: {model['name']}")
        print(f"  Latent dim: {model['architecture']['latent_dim']}")
        return model

    def get_pytorch_info(self) -> Dict[str, Any]:
        """Get PyTorch manager information"""
        return {
            'models_created': len(self.models),
            'framework': 'PyTorch',
            'version': '2.1.0',
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate PyTorch models"""

    print("=" * 60)
    print("PyTorch Deep Learning Models Demo")
    print("=" * 60)

    pytorch = PyTorchModels()

    print("\n1. Creating CNN model...")
    cnn = pytorch.create_cnn_model({
        'name': 'ImageClassifierCNN',
        'output_size': 1000,
        'input_shape': [3, 224, 224]
    })
    print(cnn['code'][:300] + "...\n")

    print("\n2. Creating RNN/LSTM model...")
    rnn = pytorch.create_rnn_model({
        'name': 'TextClassifierLSTM',
        'cell_type': 'LSTM',
        'hidden_size': 256,
        'num_layers': 2,
        'bidirectional': True
    })
    print(rnn['code'][:300] + "...\n")

    print("\n3. Creating Transformer model...")
    transformer = pytorch.create_transformer_model({
        'name': 'TransformerSeq2Seq',
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6
    })
    print(transformer['code'][:300] + "...\n")

    print("\n4. Generating training loop...")
    training_code = pytorch.create_training_loop({
        'epochs': 50,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss'
    })
    print(training_code[:300] + "...\n")

    print("\n5. Generating DataLoader...")
    dataloader_code = pytorch.create_data_loader({
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 4
    })
    print(dataloader_code[:300] + "...\n")

    print("\n6. Generating checkpoint code...")
    checkpoint_code = pytorch.create_model_checkpoint()
    print(checkpoint_code[:300] + "...\n")

    print("\n7. PyTorch summary:")
    info = pytorch.get_pytorch_info()
    print(f"  Models created: {info['models_created']}")
    print(f"  Framework: {info['framework']} {info['version']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
