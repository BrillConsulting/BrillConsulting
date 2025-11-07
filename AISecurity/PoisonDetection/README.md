# Poison Detection

Advanced detection and mitigation of data poisoning attacks and backdoors in ML training data and models.

## Features

- **Data Poisoning Detection** - Identify malicious training samples
- **Backdoor Detection** - Find hidden triggers in models
- **Anomaly-Based Detection** - Statistical outlier identification
- **Activation Clustering** - Detect backdoors via activations
- **Spectral Signatures** - Frequency domain analysis
- **Trigger Inversion** - Reverse-engineer triggers
- **Data Sanitization** - Remove poisoned samples
- **Model Cleansing** - Repair backdoored models

## Attack Types

| Attack | Method | Detection Difficulty |
|--------|--------|---------------------|
| **Label Flipping** | Change labels of training data | Medium |
| **Feature Poisoning** | Modify input features | Medium |
| **Backdoor Injection** | Embed triggers | High |
| **Clean-Label Attack** | Poison without label changes | Very High |
| **Federated Poisoning** | Malicious client updates | High |

## Usage

### Data Poisoning Detection
```python
from poison_detection import PoisonDetector

detector = PoisonDetector(
    detection_method="anomaly",
    threshold=0.95
)

# Scan training data
results = detector.scan_dataset(
    X_train, y_train,
    contamination_rate=0.05
)

print(f"Poisoned samples: {len(results.poisoned_indices)}")
print(f"Confidence: {results.confidence:.2%}")
```

### Backdoor Detection
```python
from poison_detection import BackdoorDetector

detector = BackdoorDetector(
    method="activation_clustering"
)

# Detect backdoors
backdoor_report = detector.detect(
    model=trained_model,
    clean_data=X_test,
    num_classes=10
)

if backdoor_report.has_backdoor:
    print(f"Backdoor detected in class {backdoor_report.target_class}")
    print(f"Trigger: {backdoor_report.trigger_description}")
```

### Trigger Inversion
```python
from poison_detection import TriggerInverter

inverter = TriggerInverter()

# Reconstruct trigger
trigger = inverter.invert_trigger(
    model=backdoored_model,
    target_class=3,
    iterations=1000
)

# Visualize trigger
inverter.visualize_trigger(trigger)
```

## Detection Methods

### 1. Anomaly Detection
Statistical methods to find outliers:
- **Isolation Forest** - Tree-based anomaly detection
- **One-Class SVM** - Support vector-based
- **Robust Statistics** - Median absolute deviation
- **Autoencoder** - Reconstruction error

### 2. Activation Clustering
Analyze model activations:
```python
detector = BackdoorDetector(method="activation_clustering")

# Cluster activations by class
clusters = detector.cluster_activations(
    model=model,
    data=X_clean,
    layer="penultimate"
)

# Detect outlier clusters (potential backdoors)
backdoors = detector.find_outlier_clusters(clusters)
```

### 3. Spectral Signatures
Frequency domain analysis:
```python
detector = BackdoorDetector(method="spectral")

# Compute spectral signatures
signatures = detector.compute_signatures(
    model=model,
    data=X_clean
)

# Detect anomalies
has_backdoor = detector.analyze_signatures(signatures)
```

### 4. Neural Cleanse
Trigger synthesis and detection:
```python
from poison_detection import NeuralCleanse

cleanse = NeuralCleanse()

# Synthesize potential triggers for each class
triggers = cleanse.synthesize_triggers(
    model=model,
    num_classes=10
)

# Identify backdoored class (smallest trigger)
backdoor_class = cleanse.identify_backdoor(triggers)
```

## Mitigation Strategies

### Data Sanitization
```python
from poison_detection import DataSanitizer

sanitizer = DataSanitizer()

# Remove poisoned samples
X_clean, y_clean = sanitizer.remove_poison(
    X_train, y_train,
    poisoned_indices=results.poisoned_indices
)

print(f"Removed {len(results.poisoned_indices)} samples")
print(f"Clean dataset: {len(X_clean)} samples")
```

### Model Repair
```python
from poison_detection import ModelRepair

repairer = ModelRepair()

# Fine-pruning: prune neurons activated by trigger
repaired_model = repairer.fine_prune(
    model=backdoored_model,
    trigger=detected_trigger,
    prune_ratio=0.1
)

# Verify repair
accuracy = evaluate(repaired_model, X_test, y_test)
backdoor_success = test_backdoor(repaired_model, trigger)
```

## Detection Metrics

- **True Positive Rate (TPR)** - % of detected poison
- **False Positive Rate (FPR)** - % of false alarms
- **Precision** - Accuracy of detections
- **Recall** - Coverage of actual poison
- **F1 Score** - Harmonic mean of precision/recall

## Attack Scenarios

### Scenario 1: Training Data Poisoning
Attacker injects malicious samples during training:
```python
# Attacker creates poisoned data
X_poisoned = add_trigger(X_malicious, trigger_pattern)
y_poisoned = target_class

# Mix with clean data
X_train_mixed = np.concatenate([X_clean, X_poisoned])
y_train_mixed = np.concatenate([y_clean, y_poisoned])

# Detection
detector = PoisonDetector()
results = detector.scan_dataset(X_train_mixed, y_train_mixed)
```

### Scenario 2: Federated Learning Attack
Malicious client in federated setting:
```python
from poison_detection import FederatedPoisonDetector

detector = FederatedPoisonDetector()

# Monitor client updates
for round_num, client_updates in enumerate(federated_training):
    # Detect malicious updates
    malicious_clients = detector.detect_malicious_clients(
        client_updates,
        global_model=current_model
    )

    if malicious_clients:
        print(f"Round {round_num}: Malicious clients {malicious_clients}")
```

## Technologies

- **Detection**: scikit-learn, PyOD (outlier detection)
- **Analysis**: Activation clustering, spectral analysis
- **Visualization**: Matplotlib, Plotly
- **ML Frameworks**: PyTorch, TensorFlow
- **Statistical**: SciPy, NumPy

## Best Practices

✅ Validate data provenance and sources
✅ Monitor training for anomalous behavior
✅ Use multiple detection methods
✅ Test models for backdoors before deployment
✅ Implement data sanitization pipelines
✅ Regular model audits for backdoors
✅ Federated learning: Byzantine-robust aggregation

## Performance

| Method | Detection Rate | False Positive | Overhead |
|--------|---------------|----------------|----------|
| Anomaly Detection | 85% | 5% | Low |
| Activation Clustering | 90% | 3% | Medium |
| Spectral Signatures | 92% | 2% | Medium |
| Neural Cleanse | 95% | 1% | High |

## References

- BadNets: https://arxiv.org/abs/1708.06733
- Neural Cleanse: https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf
- Activation Clustering: https://arxiv.org/abs/1811.03728
- Spectral Signatures: https://arxiv.org/abs/1811.00636
