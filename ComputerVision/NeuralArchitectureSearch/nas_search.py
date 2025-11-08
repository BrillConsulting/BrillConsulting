"""
Neural Architecture Search (NAS) System
Advanced automatic architecture discovery using evolutionary and gradient-based methods
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
import json


@dataclass
class SearchSpace:
    """Define the neural architecture search space"""
    num_layers: List[int] = None  # Possible layer counts
    num_channels: List[int] = None  # Possible channel sizes
    kernel_sizes: List[int] = None  # Possible kernel sizes
    operations: List[str] = None  # Available operations

    def __post_init__(self):
        if self.num_layers is None:
            self.num_layers = [4, 6, 8, 10, 12]
        if self.num_channels is None:
            self.num_channels = [32, 64, 128, 256, 512]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 7]
        if self.operations is None:
            self.operations = ['conv', 'depthwise', 'maxpool', 'avgpool', 'skip']


class DARTSCell(nn.Module):
    """Differentiable Architecture Search Cell"""

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int = 4):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define possible operations
        self.ops = nn.ModuleList()
        operations = [
            ('skip', nn.Identity()),
            ('conv3x3', self._make_conv(in_channels, out_channels, 3)),
            ('conv5x5', self._make_conv(in_channels, out_channels, 5)),
            ('depthwise', self._make_depthwise(in_channels, out_channels)),
            ('maxpool', nn.MaxPool2d(3, stride=1, padding=1)),
            ('avgpool', nn.AvgPool2d(3, stride=1, padding=1)),
            ('dilated_conv', self._make_dilated_conv(in_channels, out_channels))
        ]

        self.ops = nn.ModuleList([op for name, op in operations])
        self.op_names = [name for name, op in operations]

        # Architecture parameters (alphas)
        num_edges = num_nodes * (num_nodes - 1) // 2
        self.alphas = nn.Parameter(torch.randn(num_edges, len(self.ops)))

    def _make_conv(self, in_ch: int, out_ch: int, kernel_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _make_depthwise(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _make_dilated_conv(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [x]
        offset = 0

        for i in range(self.num_nodes):
            # Aggregate inputs from all previous nodes
            s = sum(
                torch.sum(
                    torch.stack([
                        torch.softmax(self.alphas[offset + j], dim=-1)[k] * op(states[j])
                        for k, op in enumerate(self.ops)
                    ], dim=0),
                    dim=0
                )
                for j in range(i + 1)
            )
            offset += i + 1
            states.append(s)

        # Concatenate all intermediate states
        return torch.cat(states[1:], dim=1)


class SuperNetwork(nn.Module):
    """Super network containing all possible architectures"""

    def __init__(self, num_classes: int = 10, num_cells: int = 8, channels: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.num_cells = num_cells

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Stack of DARTS cells
        self.cells = nn.ModuleList()
        current_channels = channels

        for i in range(num_cells):
            # Reduction cell every 1/3 of depth
            is_reduction = i in [num_cells // 3, 2 * num_cells // 3]

            if is_reduction:
                current_channels *= 2

            cell = DARTSCell(current_channels, current_channels)
            self.cells.append(cell)

        # Classifier head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels * 4, num_classes)  # *4 for concatenation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for cell in self.cells:
            x = cell(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class EvolutionaryNAS:
    """Evolutionary Neural Architecture Search"""

    def __init__(self, search_space: SearchSpace, population_size: int = 50):
        self.search_space = search_space
        self.population_size = population_size
        self.population = []
        self.fitness_history = []

    def random_architecture(self) -> Dict:
        """Generate a random architecture"""
        num_layers = random.choice(self.search_space.num_layers)
        arch = {
            'num_layers': num_layers,
            'layers': []
        }

        for i in range(num_layers):
            layer = {
                'operation': random.choice(self.search_space.operations),
                'channels': random.choice(self.search_space.num_channels),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            }
            arch['layers'].append(layer)

        return arch

    def mutate(self, architecture: Dict, mutation_rate: float = 0.2) -> Dict:
        """Mutate an architecture"""
        mutated = architecture.copy()
        mutated['layers'] = [layer.copy() for layer in architecture['layers']]

        for layer in mutated['layers']:
            if random.random() < mutation_rate:
                # Mutate operation
                layer['operation'] = random.choice(self.search_space.operations)
            if random.random() < mutation_rate:
                # Mutate channels
                layer['channels'] = random.choice(self.search_space.num_channels)
            if random.random() < mutation_rate:
                # Mutate kernel size
                layer['kernel_size'] = random.choice(self.search_space.kernel_sizes)

        return mutated

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Perform crossover between two architectures"""
        child = {
            'num_layers': min(parent1['num_layers'], parent2['num_layers']),
            'layers': []
        }

        for i in range(child['num_layers']):
            if random.random() < 0.5:
                child['layers'].append(parent1['layers'][i].copy())
            else:
                child['layers'].append(parent2['layers'][i].copy())

        return child

    def initialize_population(self):
        """Initialize the population with random architectures"""
        self.population = [
            self.random_architecture()
            for _ in range(self.population_size)
        ]

    def evaluate_fitness(self, architecture: Dict, val_loader: DataLoader,
                        device: str = 'cuda') -> float:
        """Evaluate architecture fitness (simplified - would train in practice)"""
        # In practice, this would train the network
        # Here we simulate with a heuristic

        # Penalize very deep or very wide networks
        num_params = sum(
            layer['channels'] * layer['kernel_size']**2
            for layer in architecture['layers']
        )

        # Reward skip connections
        num_skips = sum(
            1 for layer in architecture['layers']
            if layer['operation'] == 'skip'
        )

        # Simple fitness heuristic
        fitness = 100.0 - (num_params / 10000) + (num_skips * 5)

        return fitness

    def evolve(self, num_generations: int, val_loader: DataLoader,
               device: str = 'cuda') -> Dict:
        """Run evolutionary search"""
        self.initialize_population()

        best_architecture = None
        best_fitness = -float('inf')

        for generation in range(num_generations):
            # Evaluate fitness for all individuals
            fitness_scores = [
                self.evaluate_fitness(arch, val_loader, device)
                for arch in self.population
            ]

            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_architecture = self.population[gen_best_idx].copy()

            self.fitness_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitness_scores)
            })

            print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}, "
                  f"Mean Fitness = {np.mean(fitness_scores):.4f}")

            # Selection: Tournament selection
            selected = []
            for _ in range(self.population_size // 2):
                tournament = random.sample(list(zip(self.population, fitness_scores)), 3)
                winner = max(tournament, key=lambda x: x[1])
                selected.append(winner[0])

            # Create new population
            new_population = []

            # Elitism: Keep best individuals
            elite_count = 5
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())

            # Crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population

        return best_architecture


class NASTrainer:
    """Neural Architecture Search Training System"""

    def __init__(self, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.optimizer = None
        self.arch_optimizer = None

    def build_model(self, num_classes: int = 10) -> SuperNetwork:
        """Build super network"""
        model = SuperNetwork(num_classes=num_classes)
        model = model.to(self.device)
        return model

    def train_darts(self, model: SuperNetwork, train_loader: DataLoader,
                   val_loader: DataLoader, num_epochs: int = 50):
        """Train using DARTS (Differentiable Architecture Search)"""

        # Separate optimizers for weights and architecture
        w_optimizer = optim.SGD(
            model.parameters(),
            lr=0.025,
            momentum=0.9,
            weight_decay=3e-4
        )

        # Collect architecture parameters
        arch_params = []
        for cell in model.cells:
            arch_params.append(cell.alphas)

        a_optimizer = optim.Adam(
            arch_params,
            lr=3e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )

        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(w_optimizer, num_epochs)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Update architecture parameters
                if epoch >= num_epochs // 3:  # Start arch search after warmup
                    a_optimizer.zero_grad()
                    output = model(data)
                    loss_a = criterion(output, target)
                    loss_a.backward()
                    a_optimizer.step()

                # Update network weights
                w_optimizer.zero_grad()
                output = model(data)
                loss_w = criterion(output, target)
                loss_w.backward()
                w_optimizer.step()

                train_loss += loss_w.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss_w.item():.4f}")

            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()

                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

            accuracy = 100.0 * correct / total
            print(f"Epoch {epoch}: Val Loss = {val_loss/len(val_loader):.4f}, "
                  f"Accuracy = {accuracy:.2f}%")

        return model

    def extract_architecture(self, model: SuperNetwork) -> Dict:
        """Extract the best architecture from trained super network"""
        architecture = {
            'cells': []
        }

        for cell_idx, cell in enumerate(model.cells):
            # Get operation with highest alpha for each edge
            cell_arch = []
            alphas = cell.alphas.detach().cpu()

            for edge_alphas in alphas:
                best_op_idx = edge_alphas.argmax().item()
                best_op = cell.op_names[best_op_idx]
                cell_arch.append({
                    'operation': best_op,
                    'weight': edge_alphas[best_op_idx].item()
                })

            architecture['cells'].append(cell_arch)

        return architecture

    def save_architecture(self, architecture: Dict, path: str):
        """Save discovered architecture"""
        with open(path, 'w') as f:
            json.dump(architecture, f, indent=2)
        print(f"Architecture saved to {path}")


def main():
    """Main execution"""
    import torchvision
    import torchvision.transforms as transforms

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    valset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

    # Method 1: DARTS (Differentiable Architecture Search)
    print("\n=== Running DARTS ===")
    trainer = NASTrainer(device=device)
    model = trainer.build_model(num_classes=10)
    model = trainer.train_darts(model, train_loader, val_loader, num_epochs=5)

    # Extract and save architecture
    architecture = trainer.extract_architecture(model)
    trainer.save_architecture(architecture, 'discovered_architecture_darts.json')

    # Method 2: Evolutionary NAS
    print("\n=== Running Evolutionary NAS ===")
    search_space = SearchSpace()
    evo_nas = EvolutionaryNAS(search_space, population_size=20)
    best_arch = evo_nas.evolve(num_generations=10, val_loader=val_loader, device=device)

    with open('discovered_architecture_evolution.json', 'w') as f:
        json.dump(best_arch, f, indent=2)

    print("\nArchitecture search completed!")
    print(f"Best evolutionary architecture: {best_arch}")


if __name__ == '__main__':
    main()
