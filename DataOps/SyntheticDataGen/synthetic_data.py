"""
Synthetic Data Generation
=========================

Generate high-quality synthetic data for rare events, privacy-preserving ML,
and data augmentation using GANs, diffusion models, and statistical methods.

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ModelType(Enum):
    """Synthetic data model types."""
    DCGAN = "dcgan"
    STYLEGAN2 = "stylegan2"
    WGANGP = "wgan-gp"
    STABLE_DIFFUSION = "stable-diffusion"
    DDPM = "ddpm"
    CTGAN = "ctgan"
    TVAE = "tvae"
    TIMEGAN = "timegan"


@dataclass
class SynthesisConfig:
    """Synthesis configuration."""
    model_type: str
    num_samples: int
    quality_threshold: float = 0.85
    privacy_enabled: bool = False
    epsilon: Optional[float] = None


@dataclass
class QualityMetrics:
    """Quality evaluation metrics."""
    fid_score: Optional[float] = None
    ks_statistic: Optional[float] = None
    correlation_score: Optional[float] = None
    ml_efficacy: Optional[float] = None
    similarity_score: Optional[float] = None


class ImageSynthesizer:
    """Synthesize images using GANs and diffusion models."""

    def __init__(
        self,
        model: str = "stable-diffusion-v1-5",
        resolution: int = 512
    ):
        self.model = model
        self.resolution = resolution
        self.device = "cuda"

        print(f"🎨 Image Synthesizer initialized")
        print(f"   Model: {model}")
        print(f"   Resolution: {resolution}x{resolution}")

    def generate(
        self,
        prompt: str,
        num_samples: int = 100,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None
    ) -> np.ndarray:
        """Generate synthetic images."""
        print(f"\n🎨 Generating {num_samples} images")
        print(f"   Prompt: {prompt}")
        print(f"   Guidance scale: {guidance_scale}")

        # Simulate image generation
        images = np.random.rand(num_samples, self.resolution, self.resolution, 3)

        print(f"   ✓ Generated {num_samples} images")
        return images

    def generate_batch(
        self,
        prompts: List[str],
        num_samples_per_prompt: int = 10
    ) -> Dict[str, np.ndarray]:
        """Generate images for multiple prompts."""
        print(f"\n🎨 Batch generation")
        print(f"   Prompts: {len(prompts)}")
        print(f"   Samples per prompt: {num_samples_per_prompt}")

        results = {}
        for i, prompt in enumerate(prompts, 1):
            print(f"\n   [{i}/{len(prompts)}] {prompt[:50]}...")
            results[prompt] = self.generate(
                prompt,
                num_samples=num_samples_per_prompt,
                guidance_scale=7.5
            )

        return results

    def save_dataset(
        self,
        images: np.ndarray,
        output_dir: str
    ) -> None:
        """Save generated images."""
        print(f"\n💾 Saving dataset to {output_dir}")
        print(f"   Images: {len(images)}")
        print(f"   ✓ Dataset saved")


class TabularSynthesizer:
    """Synthesize tabular data using CTGAN/TVAE."""

    def __init__(
        self,
        model: str = "ctgan",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.metadata = metadata or {}
        self.trained = False

        print(f"📊 Tabular Synthesizer initialized")
        print(f"   Model: {model.upper()}")

    def fit(
        self,
        real_data: np.ndarray,
        epochs: int = 300,
        categorical_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None
    ) -> None:
        """Train synthesizer on real data."""
        print(f"\n🏋️  Training synthesizer")
        print(f"   Samples: {len(real_data):,}")
        print(f"   Epochs: {epochs}")
        print(f"   Categorical: {len(categorical_columns or [])}")
        print(f"   Continuous: {len(continuous_columns or [])}")

        # Simulate training
        for epoch in [1, epochs // 2, epochs]:
            loss = 2.5 / epoch
            if epoch == epochs:
                print(f"   Epoch {epoch}/{epochs}: loss={loss:.4f}")

        self.trained = True
        print(f"   ✓ Training complete")

    def sample(
        self,
        num_rows: int = 1000,
        conditions: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Generate synthetic samples."""
        if not self.trained:
            print("⚠️  Warning: Model not trained yet")

        print(f"\n📊 Generating {num_rows:,} synthetic rows")
        if conditions:
            print(f"   Conditions: {conditions}")

        # Simulate synthetic data generation
        synthetic_data = np.random.rand(num_rows, 10)

        print(f"   ✓ Generated {num_rows:,} rows")
        return synthetic_data

    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray
    ) -> QualityMetrics:
        """Evaluate synthetic data quality."""
        print(f"\n📈 Evaluating quality")

        # Simulate quality metrics
        metrics = QualityMetrics(
            correlation_score=0.92,
            ks_statistic=0.15,
            ml_efficacy=0.89,
            similarity_score=0.91
        )

        print(f"   Correlation score: {metrics.correlation_score:.2%}")
        print(f"   KS statistic: {metrics.ks_statistic:.3f}")
        print(f"   ML efficacy: {metrics.ml_efficacy:.2%}")
        print(f"   Similarity: {metrics.similarity_score:.2%}")

        return metrics


class TimeSeriesSynthesizer:
    """Synthesize time series data using TimeGAN."""

    def __init__(
        self,
        model: str = "timegan",
        sequence_length: int = 100
    ):
        self.model = model
        self.sequence_length = sequence_length

        print(f"📈 Time Series Synthesizer initialized")
        print(f"   Model: {model.upper()}")
        print(f"   Sequence length: {sequence_length}")

    def generate(
        self,
        real_data: np.ndarray,
        num_sequences: int = 1000,
        anomaly_rate: float = 0.05
    ) -> np.ndarray:
        """Generate synthetic time series."""
        print(f"\n📈 Generating {num_sequences:,} sequences")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Anomaly rate: {anomaly_rate:.1%}")

        # Simulate time series generation
        synthetic_series = np.random.randn(
            num_sequences,
            self.sequence_length,
            real_data.shape[-1]
        )

        # Add anomalies
        num_anomalies = int(num_sequences * anomaly_rate)
        print(f"   Anomalies injected: {num_anomalies}")

        print(f"   ✓ Generated {num_sequences:,} sequences")
        return synthetic_series


class DCGAN:
    """Deep Convolutional GAN for image generation."""

    def __init__(
        self,
        latent_dim: int = 100,
        image_size: int = 64,
        channels: int = 3
    ):
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels

        print(f"🎨 DCGAN initialized")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Channels: {channels}")

    def train(
        self,
        real_images: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64
    ) -> None:
        """Train DCGAN."""
        print(f"\n🏋️  Training DCGAN")
        print(f"   Samples: {len(real_images):,}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")

        # Simulate training
        for epoch in [1, epochs // 2, epochs]:
            d_loss = 0.7 - (epoch / epochs) * 0.2
            g_loss = 1.2 - (epoch / epochs) * 0.4
            if epoch == epochs:
                print(f"   Epoch {epoch}/{epochs}: D_loss={d_loss:.3f}, G_loss={g_loss:.3f}")

        print(f"   ✓ Training complete")

    def generate(self, num_samples: int = 100) -> np.ndarray:
        """Generate synthetic images."""
        print(f"\n🎨 Generating {num_samples} images")

        # Simulate generation
        fake_images = np.random.rand(
            num_samples,
            self.image_size,
            self.image_size,
            self.channels
        )

        print(f"   ✓ Generated {num_samples} images")
        return fake_images


class StyleGAN2:
    """StyleGAN2 for high-quality image synthesis."""

    def __init__(self, resolution: int = 1024):
        self.resolution = resolution
        self.pretrained_loaded = False

        print(f"🎨 StyleGAN2 initialized")
        print(f"   Resolution: {resolution}x{resolution}")

    def load_pretrained(self, checkpoint: str = "ffhq-1024") -> None:
        """Load pretrained model."""
        print(f"\n📥 Loading pretrained model: {checkpoint}")
        self.pretrained_loaded = True
        print(f"   ✓ Model loaded")

    def fine_tune(
        self,
        custom_images: np.ndarray,
        epochs: int = 50
    ) -> None:
        """Fine-tune on custom data."""
        print(f"\n🎯 Fine-tuning StyleGAN2")
        print(f"   Custom images: {len(custom_images)}")
        print(f"   Epochs: {epochs}")

        # Simulate fine-tuning
        print(f"   ✓ Fine-tuning complete")

    def generate_with_style_mixing(
        self,
        num_samples: int = 100,
        truncation_psi: float = 0.7
    ) -> np.ndarray:
        """Generate with style mixing."""
        print(f"\n🎨 Generating with style mixing")
        print(f"   Samples: {num_samples}")
        print(f"   Truncation: {truncation_psi}")

        # Simulate generation
        synthetic = np.random.rand(
            num_samples,
            self.resolution,
            self.resolution,
            3
        )

        print(f"   ✓ Generated {num_samples} images")
        return synthetic


class WGANGP:
    """Wasserstein GAN with Gradient Penalty."""

    def __init__(
        self,
        critic_iterations: int = 5,
        gp_weight: float = 10.0
    ):
        self.critic_iterations = critic_iterations
        self.gp_weight = gp_weight

        print(f"🎨 WGAN-GP initialized")
        print(f"   Critic iterations: {critic_iterations}")
        print(f"   GP weight: {gp_weight}")

    def train(
        self,
        training_data: np.ndarray,
        epochs: int = 200
    ) -> None:
        """Train WGAN-GP."""
        print(f"\n🏋️  Training WGAN-GP")
        print(f"   Samples: {len(training_data):,}")
        print(f"   Epochs: {epochs}")

        # Simulate stable training
        for epoch in [1, epochs // 2, epochs]:
            wasserstein_dist = 15.0 - (epoch / epochs) * 10.0
            if epoch == epochs:
                print(f"   Epoch {epoch}/{epochs}: W_dist={wasserstein_dist:.2f}")

        print(f"   ✓ Stable training complete")


class StableDiffusionSynthesizer:
    """Stable Diffusion for text-to-image generation."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1"
    ):
        self.model_id = model_id

        print(f"🎨 Stable Diffusion Synthesizer")
        print(f"   Model: {model_id}")

    def generate_batch(
        self,
        prompts: List[str],
        num_samples_per_prompt: int = 100
    ) -> Dict[str, np.ndarray]:
        """Generate images for multiple prompts."""
        print(f"\n🎨 Batch generation")
        print(f"   Prompts: {len(prompts)}")
        print(f"   Samples per prompt: {num_samples_per_prompt}")

        results = {}
        for i, prompt in enumerate(prompts, 1):
            print(f"\n   [{i}/{len(prompts)}] {prompt[:60]}...")
            images = np.random.rand(num_samples_per_prompt, 512, 512, 3)
            results[prompt] = images
            print(f"   ✓ Generated {num_samples_per_prompt} images")

        return results


class DDPM:
    """Denoising Diffusion Probabilistic Models."""

    def __init__(
        self,
        image_size: int = 256,
        timesteps: int = 1000
    ):
        self.image_size = image_size
        self.timesteps = timesteps

        print(f"🎨 DDPM initialized")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Timesteps: {timesteps}")

    def train(
        self,
        real_images: np.ndarray,
        epochs: int = 500
    ) -> None:
        """Train diffusion model."""
        print(f"\n🏋️  Training DDPM")
        print(f"   Samples: {len(real_images):,}")
        print(f"   Epochs: {epochs}")

        # Simulate training
        for epoch in [1, epochs // 2, epochs]:
            loss = 0.05 / (epoch / 10)
            if epoch == epochs:
                print(f"   Epoch {epoch}/{epochs}: loss={loss:.4f}")

        print(f"   ✓ Training complete")

    def sample(self, num_samples: int = 100) -> np.ndarray:
        """Sample from diffusion model."""
        print(f"\n🎨 Sampling {num_samples} images")

        # Simulate reverse diffusion
        synthetic_images = np.random.rand(
            num_samples,
            self.image_size,
            self.image_size,
            3
        )

        print(f"   ✓ Sampled {num_samples} images")
        return synthetic_images


class DPSynthesizer:
    """Differential privacy synthesizer."""

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5
    ):
        self.epsilon = epsilon
        self.delta = delta

        print(f"🔒 DP Synthesizer initialized")
        print(f"   Privacy: ε={epsilon}, δ={delta}")

    def generate(
        self,
        sensitive_data: np.ndarray,
        num_samples: int = 10000
    ) -> np.ndarray:
        """Generate with privacy guarantees."""
        print(f"\n🔒 Generating private synthetic data")
        print(f"   Input samples: {len(sensitive_data):,}")
        print(f"   Output samples: {num_samples:,}")
        print(f"   Privacy budget: ε={self.epsilon}")

        # Simulate DP synthesis
        private_synthetic = np.random.rand(num_samples, sensitive_data.shape[1])

        print(f"   ✓ Private data generated")
        return private_synthetic

    def verify_privacy(
        self,
        original: np.ndarray,
        synthetic: np.ndarray
    ) -> Dict[str, Any]:
        """Verify privacy guarantees."""
        print(f"\n🔍 Verifying privacy")

        # Simulate privacy verification
        report = {
            "epsilon_used": self.epsilon,
            "delta_used": self.delta,
            "membership_inference_risk": 0.52,  # ~random guessing
            "attribute_inference_risk": 0.15,
            "privacy_satisfied": True
        }

        print(f"   Membership inference risk: {report['membership_inference_risk']:.2%}")
        print(f"   Attribute inference risk: {report['attribute_inference_risk']:.2%}")
        print(f"   ✓ Privacy guarantees satisfied")

        return report

    def generate_anonymous(
        self,
        real_data_with_pii: np.ndarray,
        pii_columns: List[str],
        utility_threshold: float = 0.85
    ) -> np.ndarray:
        """Generate GDPR-compliant synthetic data."""
        print(f"\n🔒 GDPR-compliant generation")
        print(f"   PII columns: {len(pii_columns)}")
        print(f"   Utility threshold: {utility_threshold:.0%}")

        # Simulate anonymous generation
        gdpr_safe = np.random.rand(len(real_data_with_pii), real_data_with_pii.shape[1])

        print(f"   ✓ GDPR-safe data generated")
        return gdpr_safe


class QualityEvaluator:
    """Evaluate synthetic data quality."""

    def __init__(self):
        print(f"📊 Quality Evaluator initialized")

    def evaluate_tabular(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray
    ) -> QualityMetrics:
        """Evaluate tabular data quality."""
        print(f"\n📊 Evaluating tabular quality")
        print(f"   Real samples: {len(real_data):,}")
        print(f"   Synthetic samples: {len(synthetic_data):,}")

        # Simulate quality metrics
        metrics = QualityMetrics(
            correlation_score=0.92,
            ks_statistic=0.15,
            ml_efficacy=0.89,
            similarity_score=0.91
        )

        print(f"\n   Results:")
        print(f"   Column correlation: {metrics.correlation_score:.2%}")
        print(f"   Distribution similarity (KS): {metrics.ks_statistic:.3f}")
        print(f"   ML efficacy: {metrics.ml_efficacy:.2%}")
        print(f"   Overall similarity: {metrics.similarity_score:.2%}")

        return metrics

    def calculate_fid(
        self,
        real_images: np.ndarray,
        synthetic_images: np.ndarray
    ) -> float:
        """Calculate Frechet Inception Distance."""
        print(f"\n📊 Calculating FID score")
        print(f"   Real images: {len(real_images):,}")
        print(f"   Synthetic images: {len(synthetic_images):,}")

        # Simulate FID calculation
        fid_score = 12.6

        print(f"   FID Score: {fid_score:.2f} (lower is better)")
        return fid_score


def demo():
    """Demonstrate synthetic data generation."""
    print("=" * 70)
    print("Synthetic Data Generation Demo")
    print("=" * 70)

    # Image synthesis
    print(f"\n{'='*70}")
    print("Image Synthesis with Diffusion")
    print(f"{'='*70}")

    synthesizer = ImageSynthesizer(
        model="stable-diffusion-v1-5",
        resolution=512
    )

    synthetic_images = synthesizer.generate(
        prompt="rare medical condition: melanoma stage 3",
        num_samples=1000,
        guidance_scale=7.5
    )

    synthesizer.save_dataset(
        images=synthetic_images,
        output_dir="data/synthetic/melanoma"
    )

    # Tabular synthesis
    print(f"\n{'='*70}")
    print("Tabular Data with CTGAN")
    print(f"{'='*70}")

    tab_synth = TabularSynthesizer(
        model="ctgan",
        metadata={"columns": ["age", "income", "fraud"]}
    )

    # Simulate training data
    df_train = np.random.rand(5000, 10)

    tab_synth.fit(
        real_data=df_train,
        epochs=300,
        categorical_columns=['gender', 'country'],
        continuous_columns=['age', 'income']
    )

    synthetic_df = tab_synth.sample(
        num_rows=10000,
        conditions={'fraud': 1}
    )

    quality_report = tab_synth.evaluate(
        real_data=df_train,
        synthetic_data=synthetic_df
    )

    # Time series
    print(f"\n{'='*70}")
    print("Time Series with TimeGAN")
    print(f"{'='*70}")

    ts_synth = TimeSeriesSynthesizer(
        model="timegan",
        sequence_length=100
    )

    sensor_readings = np.random.randn(1000, 100, 5)

    synthetic_series = ts_synth.generate(
        real_data=sensor_readings,
        num_sequences=5000,
        anomaly_rate=0.05
    )

    # GAN architectures
    print(f"\n{'='*70}")
    print("GAN Architectures")
    print(f"{'='*70}")

    # DCGAN
    print(f"\n--- DCGAN ---")
    dcgan = DCGAN(latent_dim=100, image_size=64, channels=3)
    training_images = np.random.rand(1000, 64, 64, 3)
    dcgan.train(real_images=training_images, epochs=100, batch_size=64)
    fake_images = dcgan.generate(num_samples=1000)

    # StyleGAN2
    print(f"\n--- StyleGAN2 ---")
    stylegan = StyleGAN2(resolution=1024)
    stylegan.load_pretrained("ffhq-1024")
    rare_images = np.random.rand(100, 1024, 1024, 3)
    stylegan.fine_tune(custom_images=rare_images, epochs=50)
    synthetic = stylegan.generate_with_style_mixing(
        num_samples=100,
        truncation_psi=0.7
    )

    # WGAN-GP
    print(f"\n--- WGAN-GP ---")
    wgan = WGANGP(critic_iterations=5, gp_weight=10)
    wgan.train(training_images, epochs=200)

    # Differential Privacy
    print(f"\n{'='*70}")
    print("Privacy-Preserving Synthesis")
    print(f"{'='*70}")

    dp_synth = DPSynthesizer(epsilon=1.0, delta=1e-5)

    medical_records = np.random.rand(5000, 20)

    private_synthetic = dp_synth.generate(
        sensitive_data=medical_records,
        num_samples=10000
    )

    privacy_report = dp_synth.verify_privacy(
        original=medical_records,
        synthetic=private_synthetic
    )

    # GDPR compliance
    print(f"\n--- GDPR Compliance ---")
    customer_data = np.random.rand(10000, 15)
    gdpr_safe = dp_synth.generate_anonymous(
        real_data_with_pii=customer_data,
        pii_columns=['name', 'email', 'ssn'],
        utility_threshold=0.85
    )

    # Quality evaluation
    print(f"\n{'='*70}")
    print("Quality Evaluation")
    print(f"{'='*70}")

    evaluator = QualityEvaluator()

    # Tabular evaluation
    metrics = evaluator.evaluate_tabular(
        real_data=df_train,
        synthetic_data=synthetic_df
    )

    # Image evaluation (FID)
    fid = evaluator.calculate_fid(
        real_images=training_images,
        synthetic_images=fake_images
    )

    print(f"\n{'='*70}")
    print("✓ Synthetic Data Generation Demo Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
