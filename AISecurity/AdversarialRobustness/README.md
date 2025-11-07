# Adversarial Robustness

Defense against adversarial attacks on machine learning models with detection, hardening, and monitoring capabilities.

## Features

- **Attack Detection** - Identify adversarial perturbations in inputs
- **Adversarial Training** - Harden models against known attacks
- **Attack Generation** - FGSM, PGD, C&W attacks for testing
- **Input Sanitization** - Remove adversarial perturbations
- **Robustness Metrics** - Measure model resilience
- **Defense Mechanisms** - Gradient masking, defensive distillation, input transformations
- **Real-time Monitoring** - Detect attacks in production
- **Attack Attribution** - Identify attack types and sources

## Attack Types Covered

| Attack | Type | Threat Model |
|--------|------|--------------|
| **FGSM** | White-box | L∞ perturbation |
| **PGD** | White-box | Iterative L∞ |
| **C&W** | White-box | L2 optimization |
| **DeepFool** | White-box | Minimal perturbation |
| **JSMA** | White-box | Targeted feature manipulation |
| **Patch Attacks** | Physical | Real-world patches |

## Defense Strategies

### 1. Adversarial Training
Train models on adversarial examples to improve robustness:
```python
from adversarial_robustness import AdversarialTrainer

trainer = AdversarialTrainer(
    model=base_model,
    attack_method="pgd",
    epsilon=0.3
)

robust_model = trainer.train(
    X_train, y_train,
    epochs=10
)
```

### 2. Input Sanitization
Detect and remove perturbations:
```python
from adversarial_robustness import AdversarialDetector

detector = AdversarialDetector(
    model=trained_model,
    sensitivity=0.8
)

result = detector.detect(suspicious_input)

if result.is_adversarial:
    cleaned_input = detector.sanitize(suspicious_input)
```

### 3. Robustness Testing
Evaluate model resilience:
```python
from adversarial_robustness import RobustnessEvaluator

evaluator = RobustnessEvaluator(model)

report = evaluator.evaluate(
    X_test, y_test,
    attacks=["fgsm", "pgd", "cw"]
)

print(f"Robustness score: {report.robustness_score}")
print(f"Vulnerable samples: {report.vulnerable_count}")
```

## Usage

### Quick Start
```python
from adversarial_robustness import AdversarialDetector, AdversarialTrainer

# Train robust model
trainer = AdversarialTrainer(model=base_model)
robust_model = trainer.train(X_train, y_train)

# Deploy detector
detector = AdversarialDetector(model=robust_model)

# Monitor inputs
for input_sample in production_stream:
    result = detector.detect(input_sample)

    if result.is_adversarial:
        alert(f"Attack detected: {result.attack_type}")
        input_sample = detector.sanitize(input_sample)

    prediction = robust_model.predict(input_sample)
```

### Advanced Configuration
```python
from adversarial_robustness import (
    AdversarialDetector,
    AdversarialTrainer,
    RobustnessEvaluator,
    DefenseMechanism
)

# Multi-layer defense
detector = AdversarialDetector(
    model=model,
    detection_methods=["statistical", "neural", "ensemble"],
    sensitivity=0.85
)

# Custom adversarial training
trainer = AdversarialTrainer(
    model=model,
    attack_method="pgd",
    epsilon=0.3,
    alpha=0.01,
    iterations=40,
    defense=DefenseMechanism.GRADIENT_MASKING
)

# Comprehensive evaluation
evaluator = RobustnessEvaluator(model)
report = evaluator.comprehensive_test(
    X_test, y_test,
    attack_budgets=[0.1, 0.2, 0.3]
)
```

## Robustness Metrics

- **Attack Success Rate (ASR)** - Percentage of successful attacks
- **Robust Accuracy** - Accuracy under adversarial conditions
- **Perturbation Tolerance** - Maximum tolerable perturbation
- **Mean Perturbation Distance** - Average L2/L∞ distance
- **Certified Robustness** - Provable guarantees via randomized smoothing

## Defense Mechanisms

1. **Adversarial Training** - Train on adversarial examples
2. **Gradient Masking** - Obfuscate gradients to attackers
3. **Defensive Distillation** - Knowledge distillation for smoothness
4. **Input Transformations** - JPEG compression, bit-depth reduction
5. **Randomized Smoothing** - Certified robustness guarantees
6. **Ensemble Methods** - Multiple model consensus

## Technologies

- **Adversarial ML**: Adversarial Robustness Toolbox (ART), CleverHans
- **Deep Learning**: PyTorch, TensorFlow
- **Optimization**: CVXPY (for C&W attacks)
- **Detection**: Isolation Forest, autoencoders
- **Monitoring**: Prometheus metrics

## Use Cases

- **Computer Vision** - Image classification robustness
- **NLP Systems** - Text adversarial defense
- **Autonomous Vehicles** - Physical attack detection
- **Malware Detection** - Evasion attack prevention
- **Biometric Systems** - Spoofing prevention
- **Financial ML** - Fraud detection robustness

## Performance

| Defense | Clean Acc. | Robust Acc. (ε=0.3) | Overhead |
|---------|-----------|---------------------|----------|
| None | 95% | 0% | - |
| Input Transform | 93% | 35% | +5ms |
| Adv. Training | 92% | 65% | +0ms |
| Ensemble | 94% | 70% | +20ms |
| Full Stack | 91% | 75% | +30ms |

## Best Practices

✅ Test against multiple attack types
✅ Use adversarial training with diverse attacks
✅ Monitor for distributional shifts
✅ Implement defense-in-depth (multiple layers)
✅ Regular robustness audits
✅ Balance security vs. performance trade-offs

## References

- MITRE ATLAS: https://atlas.mitre.org/
- Adversarial Robustness Toolbox: https://github.com/Trusted-AI/adversarial-robustness-toolbox
- CleverHans: https://github.com/cleverhans-lab/cleverhans
