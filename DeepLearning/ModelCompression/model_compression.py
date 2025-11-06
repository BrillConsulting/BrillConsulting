"""
Model Compression Implementation
Author: BrillConsulting
Description: Advanced model compression techniques - pruning, quantization, and knowledge distillation
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class PruningManager:
    """Model pruning manager"""

    def __init__(self):
        """Initialize pruning manager"""
        self.pruning_history = []

    def magnitude_pruning(self, weights: np.ndarray,
                         sparsity: float = 0.5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Magnitude-based weight pruning

        Args:
            weights: Model weights
            sparsity: Target sparsity ratio (0-1)

        Returns:
            Pruned weights and pruning statistics
        """
        threshold = np.percentile(np.abs(weights), sparsity * 100)
        mask = np.abs(weights) > threshold
        pruned_weights = weights * mask

        stats = {
            'original_size': weights.size,
            'pruned_size': np.count_nonzero(pruned_weights),
            'sparsity': 1.0 - (np.count_nonzero(pruned_weights) / weights.size),
            'threshold': threshold,
            'compression_ratio': weights.size / np.count_nonzero(pruned_weights)
        }

        return pruned_weights, stats

    def structured_pruning(self, weights: np.ndarray,
                          pruning_ratio: float = 0.3) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Structured pruning (channel/filter pruning)

        Args:
            weights: Model weights (filters x channels x height x width)
            pruning_ratio: Ratio of filters to prune

        Returns:
            Pruned weights and statistics
        """
        if len(weights.shape) < 2:
            return weights, {}

        # Calculate L1 norm for each filter
        num_filters = weights.shape[0]
        filter_norms = np.sum(np.abs(weights), axis=tuple(range(1, len(weights.shape))))

        # Select filters to keep
        num_keep = int(num_filters * (1 - pruning_ratio))
        keep_indices = np.argsort(filter_norms)[-num_keep:]

        pruned_weights = weights[keep_indices]

        stats = {
            'original_filters': num_filters,
            'pruned_filters': num_keep,
            'pruning_ratio': pruning_ratio,
            'size_reduction': 1 - (pruned_weights.size / weights.size)
        }

        return pruned_weights, stats

    def iterative_pruning(self, weights: np.ndarray,
                         target_sparsity: float = 0.9,
                         num_iterations: int = 10) -> List[Dict[str, Any]]:
        """
        Iterative magnitude pruning

        Args:
            weights: Model weights
            target_sparsity: Final target sparsity
            num_iterations: Number of pruning iterations

        Returns:
            Pruning history
        """
        current_weights = weights.copy()
        history = []

        for i in range(num_iterations):
            # Calculate current sparsity target
            current_target = target_sparsity * ((i + 1) / num_iterations)

            # Prune
            current_weights, stats = self.magnitude_pruning(current_weights, current_target)

            stats['iteration'] = i + 1
            stats['target_sparsity'] = current_target
            history.append(stats)

        return history


class QuantizationManager:
    """Model quantization manager"""

    def __init__(self):
        """Initialize quantization manager"""
        self.quantization_history = []

    def uniform_quantization(self, weights: np.ndarray,
                            num_bits: int = 8) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Uniform quantization

        Args:
            weights: Model weights
            num_bits: Number of bits for quantization

        Returns:
            Quantized weights and statistics
        """
        w_min, w_max = weights.min(), weights.max()
        num_levels = 2 ** num_bits

        # Quantize
        scale = (w_max - w_min) / (num_levels - 1)
        quantized = np.round((weights - w_min) / scale)

        # Dequantize
        dequantized = quantized * scale + w_min

        stats = {
            'num_bits': num_bits,
            'num_levels': num_levels,
            'scale': scale,
            'min_value': w_min,
            'max_value': w_max,
            'compression_ratio': 32 / num_bits,  # Assuming float32
            'quantization_error': np.mean(np.abs(weights - dequantized))
        }

        return dequantized, stats

    def quantization_aware_training(self, weights: np.ndarray,
                                   num_bits: int = 8,
                                   num_epochs: int = 10) -> List[Dict[str, Any]]:
        """
        Simulate quantization-aware training

        Args:
            weights: Model weights
            num_bits: Quantization bits
            num_epochs: Number of training epochs

        Returns:
            QAT training history
        """
        history = []
        current_weights = weights.copy()

        for epoch in range(num_epochs):
            # Quantize
            quantized, stats = self.uniform_quantization(current_weights, num_bits)

            # Simulate fine-tuning (small random update)
            update = np.random.normal(0, 0.001, weights.shape)
            current_weights = quantized + update

            stats['epoch'] = epoch + 1
            stats['simulated_accuracy'] = 0.85 + 0.1 * (epoch / num_epochs)
            history.append(stats)

        return history

    def dynamic_quantization(self, weights: np.ndarray,
                            percentile: float = 99.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Dynamic quantization with outlier handling

        Args:
            weights: Model weights
            percentile: Percentile for clipping outliers

        Returns:
            Quantized weights and statistics
        """
        # Clip outliers
        clip_val = np.percentile(np.abs(weights), percentile)
        clipped = np.clip(weights, -clip_val, clip_val)

        # Quantize clipped weights
        quantized, stats = self.uniform_quantization(clipped, num_bits=8)

        stats['clip_value'] = clip_val
        stats['percentile'] = percentile
        stats['outliers_clipped'] = np.sum(np.abs(weights) > clip_val)

        return quantized, stats


class KnowledgeDistillation:
    """Knowledge distillation manager"""

    def __init__(self, temperature: float = 3.0):
        """
        Initialize knowledge distillation

        Args:
            temperature: Temperature for softening distributions
        """
        self.temperature = temperature
        self.distillation_history = []

    def softmax_with_temperature(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply softmax with temperature

        Args:
            logits: Model logits

        Returns:
            Soft probabilities
        """
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        return exp_logits / np.sum(exp_logits)

    def distillation_loss(self, student_logits: np.ndarray,
                         teacher_logits: np.ndarray,
                         alpha: float = 0.5) -> Dict[str, float]:
        """
        Calculate distillation loss

        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            alpha: Weight for distillation loss

        Returns:
            Loss components
        """
        # Soft targets
        student_soft = self.softmax_with_temperature(student_logits)
        teacher_soft = self.softmax_with_temperature(teacher_logits)

        # KL divergence
        kl_div = np.sum(teacher_soft * np.log((teacher_soft + 1e-10) / (student_soft + 1e-10)))

        # Simulated hard loss
        hard_loss = np.random.uniform(0.3, 0.6)

        # Combined loss
        total_loss = alpha * kl_div * (self.temperature ** 2) + (1 - alpha) * hard_loss

        return {
            'kl_divergence': kl_div,
            'hard_loss': hard_loss,
            'total_loss': total_loss
        }

    def progressive_distillation(self, teacher_size: int,
                                student_sizes: List[int],
                                num_epochs: int = 20) -> List[Dict[str, Any]]:
        """
        Progressive knowledge distillation

        Args:
            teacher_size: Teacher model size
            student_sizes: List of student model sizes (progressive)
            num_epochs: Training epochs per student

        Returns:
            Distillation history
        """
        history = []

        current_teacher_size = teacher_size
        for i, student_size in enumerate(student_sizes):
            student_history = {
                'student_id': i,
                'teacher_size': current_teacher_size,
                'student_size': student_size,
                'compression_ratio': current_teacher_size / student_size,
                'epochs': []
            }

            for epoch in range(num_epochs):
                # Simulate distillation
                teacher_logits = np.random.randn(10)
                student_logits = np.random.randn(10)

                loss = self.distillation_loss(student_logits, teacher_logits)
                loss['epoch'] = epoch + 1
                loss['accuracy'] = 0.7 + 0.2 * (epoch / num_epochs)

                student_history['epochs'].append(loss)

            student_history['final_accuracy'] = student_history['epochs'][-1]['accuracy']
            history.append(student_history)

            # Next student learns from current student
            current_teacher_size = student_size

        return history


class ModelCompressionManager:
    """Main model compression manager"""

    def __init__(self):
        """Initialize model compression manager"""
        self.pruning_mgr = PruningManager()
        self.quantization_mgr = QuantizationManager()
        self.distillation_mgr = KnowledgeDistillation()
        self.compression_experiments = []

    def comprehensive_compression(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply comprehensive compression pipeline

        Args:
            model_config: Model configuration

        Returns:
            Compression results
        """
        print(f"\n{'='*60}")
        print("Comprehensive Model Compression")
        print(f"{'='*60}")

        # Simulate model weights
        np.random.seed(42)
        weights = np.random.randn(1000, 1000) * 0.1

        results = {
            'original_size': weights.size,
            'steps': []
        }

        # Step 1: Pruning
        print("\n1. Applying magnitude pruning...")
        pruned_weights, prune_stats = self.pruning_mgr.magnitude_pruning(
            weights, sparsity=model_config.get('sparsity', 0.7)
        )
        print(f"   Sparsity: {prune_stats['sparsity']:.2%}")
        print(f"   Compression: {prune_stats['compression_ratio']:.2f}x")
        results['steps'].append({'name': 'pruning', 'stats': prune_stats})

        # Step 2: Quantization
        print("\n2. Applying quantization...")
        quantized_weights, quant_stats = self.quantization_mgr.uniform_quantization(
            pruned_weights, num_bits=model_config.get('num_bits', 8)
        )
        print(f"   Bits: {quant_stats['num_bits']}")
        print(f"   Compression: {quant_stats['compression_ratio']:.2f}x")
        results['steps'].append({'name': 'quantization', 'stats': quant_stats})

        # Step 3: Knowledge distillation simulation
        print("\n3. Simulating knowledge distillation...")
        distill_results = self.distillation_mgr.progressive_distillation(
            teacher_size=weights.size,
            student_sizes=[500000, 250000, 100000],
            num_epochs=10
        )
        print(f"   Students trained: {len(distill_results)}")
        print(f"   Final compression: {distill_results[-1]['compression_ratio']:.2f}x")
        results['steps'].append({'name': 'distillation', 'stats': distill_results})

        # Calculate total compression
        total_compression = (
            prune_stats['compression_ratio'] *
            quant_stats['compression_ratio'] *
            distill_results[-1]['compression_ratio']
        )

        results['total_compression'] = total_compression
        results['final_size'] = weights.size / total_compression

        print(f"\n{'='*60}")
        print(f"Total Compression: {total_compression:.2f}x")
        print(f"Final Size: {results['final_size']:.0f} parameters")
        print(f"{'='*60}")

        self.compression_experiments.append(results)

        return results

    def get_compression_code(self) -> str:
        """Generate PyTorch compression code"""

        code = """
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Magnitude pruning
def prune_model(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

# Quantization
def quantize_model(model):
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model

# Knowledge distillation
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets
        soft_loss = self.kl_div(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # Hard targets
        hard_loss = self.ce_loss(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# Training with distillation
def train_with_distillation(student, teacher, train_loader, epochs=10):
    criterion = DistillationLoss(temperature=3.0, alpha=0.5)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

    teacher.eval()
    for epoch in range(epochs):
        student.train()
        for data, target in train_loader:
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_output = teacher(data)

            student_output = student(data)
            loss = criterion(student_output, teacher_output, target)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return student
"""

        return code

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'compression_experiments': len(self.compression_experiments),
            'techniques': ['pruning', 'quantization', 'distillation'],
            'framework': 'Model Compression',
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate model compression"""

    print("="*60)
    print("Model Compression Demo")
    print("="*60)

    manager = ModelCompressionManager()

    # Run comprehensive compression
    print("\n1. Running comprehensive compression pipeline...")
    results = manager.comprehensive_compression({
        'sparsity': 0.8,
        'num_bits': 8
    })

    # Show implementation code
    print("\n2. PyTorch implementation code:")
    code = manager.get_compression_code()
    print(code[:500] + "...\n")

    # Manager info
    print("\n3. Manager summary:")
    info = manager.get_manager_info()
    print(f"  Experiments: {info['compression_experiments']}")
    print(f"  Techniques: {', '.join(info['techniques'])}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    demo()
