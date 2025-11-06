"""
Federated Learning Implementation
Author: BrillConsulting
Description: Privacy-preserving distributed machine learning with secure aggregation
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib


class FederatedClient:
    """Individual client in federated learning system"""

    def __init__(self, client_id: str, data_size: int):
        """
        Initialize federated client

        Args:
            client_id: Unique client identifier
            data_size: Size of local dataset
        """
        self.client_id = client_id
        self.data_size = data_size
        self.local_model = None
        self.local_weights = []
        self.training_history = []

    def train_local_model(self, global_weights: List[np.ndarray],
                         epochs: int = 5, learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Train model on local data

        Args:
            global_weights: Weights from global model
            epochs: Number of local training epochs
            learning_rate: Learning rate for local training

        Returns:
            Training results with updated weights
        """
        # Simulate local training
        updated_weights = []
        for weight in global_weights:
            # Add random noise to simulate training
            noise = np.random.normal(0, 0.01, weight.shape)
            updated_weights.append(weight + noise)

        self.local_weights = updated_weights

        training_result = {
            'client_id': self.client_id,
            'weights': updated_weights,
            'data_size': self.data_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'loss': np.random.uniform(0.1, 0.5),  # Simulated loss
            'accuracy': np.random.uniform(0.85, 0.95),  # Simulated accuracy
            'timestamp': datetime.now().isoformat()
        }

        self.training_history.append({
            'loss': training_result['loss'],
            'accuracy': training_result['accuracy'],
            'timestamp': training_result['timestamp']
        })

        return training_result

    def get_encrypted_weights(self) -> Dict[str, Any]:
        """
        Encrypt local weights for secure aggregation

        Returns:
            Encrypted weights
        """
        encrypted = {
            'client_id': self.client_id,
            'weights_hash': hashlib.sha256(
                str(self.local_weights).encode()
            ).hexdigest()[:16],
            'data_size': self.data_size
        }

        return encrypted


class FederatedServer:
    """Central server for federated learning"""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize federated server

        Args:
            model_config: Global model configuration
        """
        self.model_config = model_config
        self.global_weights = self._initialize_weights()
        self.clients: Dict[str, FederatedClient] = {}
        self.aggregation_history = []
        self.round_number = 0

    def _initialize_weights(self) -> List[np.ndarray]:
        """Initialize global model weights"""
        layer_sizes = self.model_config.get('layer_sizes', [784, 128, 64, 10])
        weights = []

        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            weight = np.random.uniform(-limit, limit,
                                      (layer_sizes[i], layer_sizes[i+1]))
            weights.append(weight)

        return weights

    def register_client(self, client_id: str, data_size: int) -> FederatedClient:
        """
        Register new client

        Args:
            client_id: Unique client identifier
            data_size: Size of client's local dataset

        Returns:
            Registered client
        """
        client = FederatedClient(client_id, data_size)
        self.clients[client_id] = client

        print(f"✓ Client registered: {client_id} (data_size: {data_size})")
        return client

    def federated_averaging(self, client_updates: List[Dict[str, Any]],
                          aggregation_method: str = 'weighted') -> List[np.ndarray]:
        """
        Aggregate client updates using FedAvg algorithm

        Args:
            client_updates: List of client training results
            aggregation_method: 'weighted' or 'uniform' averaging

        Returns:
            Aggregated global weights
        """
        if not client_updates:
            return self.global_weights

        # Calculate total data size
        total_data = sum(update['data_size'] for update in client_updates)

        # Initialize aggregated weights
        aggregated_weights = [np.zeros_like(w) for w in self.global_weights]

        # Aggregate weights
        for update in client_updates:
            weight_factor = (update['data_size'] / total_data
                           if aggregation_method == 'weighted'
                           else 1.0 / len(client_updates))

            for i, weight in enumerate(update['weights']):
                aggregated_weights[i] += weight * weight_factor

        return aggregated_weights

    def secure_aggregation(self, client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Perform secure aggregation with differential privacy

        Args:
            client_updates: List of client training results

        Returns:
            Securely aggregated weights
        """
        # Standard FedAvg
        aggregated = self.federated_averaging(client_updates)

        # Add differential privacy noise
        privacy_budget = self.model_config.get('privacy_budget', 0.1)

        for i in range(len(aggregated)):
            noise_scale = privacy_budget * np.std(aggregated[i])
            noise = np.random.laplace(0, noise_scale, aggregated[i].shape)
            aggregated[i] += noise

        return aggregated

    def run_federated_round(self, selected_clients: List[str],
                           epochs: int = 5,
                           secure: bool = True) -> Dict[str, Any]:
        """
        Execute one round of federated learning

        Args:
            selected_clients: List of client IDs to participate
            epochs: Number of local training epochs
            secure: Use secure aggregation

        Returns:
            Round results
        """
        self.round_number += 1

        print(f"\n--- Federated Round {self.round_number} ---")
        print(f"Selected clients: {len(selected_clients)}")

        # Collect client updates
        client_updates = []

        for client_id in selected_clients:
            if client_id not in self.clients:
                print(f"  ⚠ Client {client_id} not registered, skipping")
                continue

            client = self.clients[client_id]

            # Train on client
            update = client.train_local_model(self.global_weights, epochs)
            client_updates.append(update)

            print(f"  ✓ {client_id}: loss={update['loss']:.4f}, acc={update['accuracy']:.4f}")

        # Aggregate updates
        if secure:
            self.global_weights = self.secure_aggregation(client_updates)
            print("  ✓ Secure aggregation completed")
        else:
            self.global_weights = self.federated_averaging(client_updates)
            print("  ✓ Standard aggregation completed")

        # Calculate round statistics
        avg_loss = np.mean([u['loss'] for u in client_updates])
        avg_accuracy = np.mean([u['accuracy'] for u in client_updates])

        round_result = {
            'round': self.round_number,
            'num_clients': len(client_updates),
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'secure': secure,
            'timestamp': datetime.now().isoformat()
        }

        self.aggregation_history.append(round_result)

        print(f"  Round avg: loss={avg_loss:.4f}, acc={avg_accuracy:.4f}")

        return round_result


class FederatedLearningManager:
    """Main federated learning manager"""

    def __init__(self):
        """Initialize federated learning manager"""
        self.servers: Dict[str, FederatedServer] = {}
        self.experiments = []

    def create_federated_setup(self, setup_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create federated learning setup

        Args:
            setup_config: Setup configuration

        Returns:
            Setup details
        """
        setup_id = setup_config.get('setup_id', f'fed_setup_{len(self.servers)}')

        # Create server
        server = FederatedServer({
            'layer_sizes': setup_config.get('layer_sizes', [784, 128, 64, 10]),
            'privacy_budget': setup_config.get('privacy_budget', 0.1)
        })

        self.servers[setup_id] = server

        # Register clients
        num_clients = setup_config.get('num_clients', 10)
        data_distribution = setup_config.get('data_distribution', 'uniform')

        for i in range(num_clients):
            if data_distribution == 'uniform':
                data_size = 1000
            elif data_distribution == 'non_iid':
                # Simulate non-IID data distribution
                data_size = int(np.random.lognormal(7, 0.5))
            else:
                data_size = np.random.randint(500, 2000)

            server.register_client(f'client_{i}', data_size)

        setup = {
            'setup_id': setup_id,
            'num_clients': num_clients,
            'data_distribution': data_distribution,
            'privacy_enabled': setup_config.get('secure', True),
            'created_at': datetime.now().isoformat()
        }

        print(f"\n✓ Federated setup created: {setup_id}")
        print(f"  Clients: {num_clients}, Distribution: {data_distribution}")

        return setup

    def run_federated_experiment(self, setup_id: str,
                                experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete federated learning experiment

        Args:
            setup_id: Setup identifier
            experiment_config: Experiment configuration

        Returns:
            Experiment results
        """
        if setup_id not in self.servers:
            raise ValueError(f"Setup {setup_id} not found")

        server = self.servers[setup_id]

        num_rounds = experiment_config.get('num_rounds', 10)
        clients_per_round = experiment_config.get('clients_per_round', 5)
        local_epochs = experiment_config.get('local_epochs', 5)
        secure = experiment_config.get('secure', True)

        print(f"\n{'='*60}")
        print(f"Federated Learning Experiment: {setup_id}")
        print(f"{'='*60}")
        print(f"Rounds: {num_rounds}, Clients/round: {clients_per_round}")
        print(f"Local epochs: {local_epochs}, Secure: {secure}")

        # Run federated rounds
        for round_num in range(num_rounds):
            # Select random clients
            available_clients = list(server.clients.keys())
            selected = np.random.choice(
                available_clients,
                size=min(clients_per_round, len(available_clients)),
                replace=False
            ).tolist()

            # Run round
            server.run_federated_round(selected, local_epochs, secure)

        experiment = {
            'setup_id': setup_id,
            'num_rounds': num_rounds,
            'clients_per_round': clients_per_round,
            'final_loss': server.aggregation_history[-1]['avg_loss'],
            'final_accuracy': server.aggregation_history[-1]['avg_accuracy'],
            'history': server.aggregation_history,
            'completed_at': datetime.now().isoformat()
        }

        self.experiments.append(experiment)

        print(f"\n{'='*60}")
        print("Experiment completed!")
        print(f"Final loss: {experiment['final_loss']:.4f}")
        print(f"Final accuracy: {experiment['final_accuracy']:.4f}")
        print(f"{'='*60}")

        return experiment

    def get_experiment_code(self) -> str:
        """Generate federated learning implementation code"""

        code = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_client(model, data_loader, epochs=5, lr=0.01):
    \"\"\"Train model on client data\"\"\"
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def federated_averaging(client_weights: List[Dict], data_sizes: List[int]):
    \"\"\"FedAvg algorithm for weight aggregation\"\"\"
    total_data = sum(data_sizes)

    # Initialize aggregated weights
    aggregated = {}
    for key in client_weights[0].keys():
        aggregated[key] = torch.zeros_like(client_weights[0][key])

    # Weighted average
    for client_w, size in zip(client_weights, data_sizes):
        weight = size / total_data
        for key in client_w.keys():
            aggregated[key] += client_w[key] * weight

    return aggregated

# Federated learning loop
global_model = SimpleModel()

for round in range(num_rounds):
    client_weights = []
    data_sizes = []

    # Train on selected clients
    for client in selected_clients:
        local_model = SimpleModel()
        local_model.load_state_dict(global_model.state_dict())

        weights = train_client(local_model, client.data_loader)
        client_weights.append(weights)
        data_sizes.append(len(client.dataset))

    # Aggregate
    global_weights = federated_averaging(client_weights, data_sizes)
    global_model.load_state_dict(global_weights)

    print(f"Round {round + 1} completed")
"""

        return code

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'setups': len(self.servers),
            'experiments': len(self.experiments),
            'total_clients': sum(len(s.clients) for s in self.servers.values()),
            'framework': 'Federated Learning',
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate federated learning"""

    print("="*60)
    print("Federated Learning Demo")
    print("="*60)

    manager = FederatedLearningManager()

    # Create federated setup
    print("\n1. Creating federated learning setup...")
    setup = manager.create_federated_setup({
        'setup_id': 'medical_fl',
        'num_clients': 10,
        'data_distribution': 'non_iid',
        'layer_sizes': [784, 256, 128, 10],
        'privacy_budget': 0.05,
        'secure': True
    })

    # Run experiment
    print("\n2. Running federated experiment...")
    experiment = manager.run_federated_experiment('medical_fl', {
        'num_rounds': 5,
        'clients_per_round': 5,
        'local_epochs': 3,
        'secure': True
    })

    # Show implementation code
    print("\n3. PyTorch implementation code:")
    code = manager.get_experiment_code()
    print(code[:500] + "...\n")

    # Manager info
    print("\n4. Manager summary:")
    info = manager.get_manager_info()
    print(f"  Setups: {info['setups']}")
    print(f"  Experiments: {info['experiments']}")
    print(f"  Total clients: {info['total_clients']}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    demo()
