# Federated Privacy

Privacy-preserving federated learning with differential privacy, secure aggregation, and encrypted computation.

## Features

- **Differential Privacy** - Formal privacy guarantees (ε-δ privacy)
- **Secure Aggregation** - Encrypted gradient aggregation
- **Privacy Budgeting** - Track and manage privacy expenditure
- **Federated Learning** - Decentralized training without data sharing
- **Homomorphic Encryption** - Compute on encrypted data
- **Secure Multi-Party Computation (SMPC)** - Collaborative privacy
- **Privacy Auditing** - Measure actual privacy leakage
- **Client Selection** - Privacy-aware participant selection

## Privacy Guarantees

| Method | Privacy | Communication | Computation |
|--------|---------|---------------|-------------|
| **Differential Privacy** | Strong (ε-δ) | Low | Medium |
| **Secure Aggregation** | Medium | Medium | High |
| **Homomorphic Encryption** | Very Strong | High | Very High |
| **SMPC** | Strong | High | High |

## Use Cases

### 1. Healthcare Federated Learning
Train on patient data without exposing PHI:
```python
from federated_privacy import FederatedTrainer, DifferentialPrivacy

# Initialize trainer
trainer = FederatedTrainer(
    privacy_mechanism=DifferentialPrivacy(epsilon=1.0, delta=1e-5),
    secure_aggregation=True
)

# Train across hospitals
model = trainer.train(
    clients=hospital_clients,
    rounds=50,
    local_epochs=5
)
```

### 2. Financial Data Analysis
Collaborative learning without sharing transactions:
```python
from federated_privacy import PrivacyBudget, SecureAggregator

# Setup privacy budget
budget = PrivacyBudget(
    total_epsilon=10.0,
    delta=1e-5,
    composition="advanced"
)

# Secure aggregation
aggregator = SecureAggregator(
    encryption="homomorphic",
    threshold=0.7
)

# Train model
for round in range(rounds):
    if budget.can_spend(epsilon=0.2):
        gradients = collect_gradients(bank_clients)
        secure_update = aggregator.aggregate(gradients)
        model.update(secure_update)
        budget.spend(epsilon=0.2)
```

### 3. Privacy-Preserving Analytics
Analyze distributed data with privacy:
```python
from federated_privacy import PrivacyEngine, SMPCProtocol

engine = PrivacyEngine(
    privacy_level="high",
    noise_mechanism="gaussian"
)

# Privacy-preserving statistics
private_mean = engine.compute_mean(
    values=client_data,
    epsilon=0.5,
    clip_norm=1.0
)

private_histogram = engine.compute_histogram(
    data=client_data,
    bins=10,
    epsilon=1.0
)
```

## Differential Privacy

### Mechanisms

**Laplace Mechanism**:
```python
from federated_privacy import LaplaceMechanism

mechanism = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
private_value = mechanism.add_noise(true_value)
```

**Gaussian Mechanism**:
```python
from federated_privacy import GaussianMechanism

mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
private_value = mechanism.add_noise(true_value)
```

**Exponential Mechanism**:
```python
from federated_privacy import ExponentialMechanism

mechanism = ExponentialMechanism(epsilon=1.0)
private_selection = mechanism.select(
    candidates=options,
    quality_function=utility_fn
)
```

## Secure Aggregation

Aggregate client updates without revealing individual contributions:

```python
from federated_privacy import SecureAggregator

aggregator = SecureAggregator(
    num_clients=100,
    threshold=80,  # 80% must participate
    encryption="paillier"
)

# Clients encrypt their gradients
encrypted_gradients = [
    aggregator.encrypt(client_gradient, client_id)
    for client_id, client_gradient in enumerate(client_gradients)
]

# Server aggregates without seeing individual gradients
aggregated = aggregator.aggregate(encrypted_gradients)

# Decrypt final result
final_gradient = aggregator.decrypt(aggregated)
```

## Privacy Budgeting

Track privacy expenditure across queries:

```python
from federated_privacy import PrivacyBudget

budget = PrivacyBudget(
    total_epsilon=10.0,
    delta=1e-5,
    composition="advanced"  # Advanced composition theorem
)

# Check before spending
if budget.can_spend(epsilon=0.5):
    result = private_query(data, epsilon=0.5)
    budget.spend(epsilon=0.5)

# Monitor budget
print(f"Remaining epsilon: {budget.remaining_epsilon}")
print(f"Queries made: {budget.query_count}")
```

## Federated Algorithms

### FedAvg (Federated Averaging)
```python
from federated_privacy import FedAvg

fed_avg = FedAvg(
    learning_rate=0.01,
    privacy=DifferentialPrivacy(epsilon=1.0)
)

for round in range(num_rounds):
    # Select clients
    selected_clients = fed_avg.select_clients(all_clients, fraction=0.1)

    # Local training
    client_updates = [
        client.train(local_epochs=5)
        for client in selected_clients
    ]

    # Aggregate with privacy
    global_update = fed_avg.aggregate(client_updates)

    # Update global model
    global_model.apply_update(global_update)
```

### FedProx (Federated Proximal)
```python
from federated_privacy import FedProx

fed_prox = FedProx(
    mu=0.1,  # Proximal term
    privacy=DifferentialPrivacy(epsilon=0.5)
)
```

## Privacy Auditing

Measure actual privacy leakage:

```python
from federated_privacy import PrivacyAuditor

auditor = PrivacyAuditor()

# Train model
model = train_with_privacy(data, epsilon=1.0)

# Audit privacy
audit_result = auditor.audit(
    model=model,
    training_data=data,
    epsilon_claimed=1.0,
    num_samples=1000
)

print(f"Claimed epsilon: {audit_result.claimed_epsilon}")
print(f"Empirical epsilon: {audit_result.empirical_epsilon}")
print(f"Privacy violation: {audit_result.is_violated}")
```

## Technologies

- **Privacy**: Opacus (PyTorch DP), TensorFlow Privacy
- **Cryptography**: PySyft, TenSEAL (Homomorphic Encryption)
- **Federated Learning**: Flower, PySyft
- **SMPC**: MP-SPDZ
- **Optimization**: Advanced composition, Rényi DP

## Performance

| Configuration | Accuracy Loss | Privacy | Computation Overhead |
|--------------|---------------|---------|---------------------|
| No Privacy | 0% | None | 1x |
| DP (ε=10) | -1% | Weak | 1.2x |
| DP (ε=1) | -3% | Strong | 1.5x |
| DP + SecAgg | -3.5% | Very Strong | 3x |
| SMPC | -2% | Strong | 10x |

## Best Practices

✅ Use appropriate epsilon values (ε < 10 for strong privacy)
✅ Implement privacy budgeting from day one
✅ Clip gradients to bound sensitivity
✅ Use advanced composition for multiple queries
✅ Audit privacy empirically, don't just trust theory
✅ Consider client-level DP for federated learning
✅ Use secure aggregation for sensitive applications
✅ Monitor privacy budget exhaustion

## Theoretical Foundations

### Differential Privacy Definition
A mechanism M satisfies (ε, δ)-differential privacy if for all datasets D, D' differing by one record and all outcomes S:

P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S] + δ

### Composition Theorems
- **Basic Composition**: ε_total = Σε_i
- **Advanced Composition**: ε_total = √(2k ln(1/δ)) × ε + k × ε × (e^ε - 1)
- **Rényi DP**: Better bounds for Gaussian mechanism

## Legal & Compliance

- **GDPR**: Article 25 (Privacy by Design)
- **HIPAA**: Safe Harbor for de-identification
- **CCPA**: Privacy-preserving analytics
- **EU AI Act**: Privacy-preserving AI systems

## References

- Differential Privacy: Dwork & Roth (2014)
- Federated Learning: McMahan et al. (2017)
- Secure Aggregation: Bonawitz et al. (2017)
- Opacus: https://opacus.ai/
- TensorFlow Privacy: https://github.com/tensorflow/privacy
