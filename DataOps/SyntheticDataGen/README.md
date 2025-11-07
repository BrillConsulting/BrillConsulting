# Synthetic Data Generation

Generate high-quality synthetic data for rare events, privacy-preserving ML, and data augmentation using GANs, diffusion models, and statistical methods.

## Features

- **GAN-based Generation** - DCGAN, StyleGAN, WGAN-GP for images
- **Diffusion Models** - Stable Diffusion, DDPM for high-quality synthesis
- **Tabular Synthesis** - CTGAN, TVAE for structured data
- **Time Series** - TimeGAN for temporal data
- **Text Generation** - GPT-based synthetic text
- **Privacy Preservation** - Differential privacy in synthesis
- **Rare Event Simulation** - Oversample rare classes
- **Data Augmentation** - Expand training datasets

## Synthetic Data Types

| Data Type | Method | Use Case | Quality Score |
|-----------|--------|----------|---------------|
| **Images** | StyleGAN2 | Face generation | 95% |
| **Tabular** | CTGAN | Financial records | 92% |
| **Time Series** | TimeGAN | Sensor data | 88% |
| **Text** | GPT-3.5 | Training corpus | 90% |
| **Audio** | WaveGAN | Speech synthesis | 85% |

## Usage

### Image Synthesis with Diffusion
```python
from synthetic_data import ImageSynthesizer

synthesizer = ImageSynthesizer(
    model="stable-diffusion-v1-5",
    resolution=512
)

# Generate rare event images
synthetic_images = synthesizer.generate(
    prompt="rare medical condition: melanoma stage 3",
    num_samples=1000,
    guidance_scale=7.5
)

# Augment training data
synthesizer.save_dataset(
    images=synthetic_images,
    output_dir="data/synthetic/melanoma"
)
```

### Tabular Data with CTGAN
```python
from synthetic_data import TabularSynthesizer

synthesizer = TabularSynthesizer(
    model="ctgan",
    metadata=data_metadata
)

# Train on real data
synthesizer.fit(
    real_data=df_train,
    epochs=300,
    categorical_columns=['gender', 'country'],
    continuous_columns=['age', 'income']
)

# Generate synthetic data
synthetic_df = synthesizer.sample(
    num_rows=10000,
    conditions={'fraud': 1}  # Oversample rare fraud cases
)

# Validate quality
quality_report = synthesizer.evaluate(real_data=df_train, synthetic_data=synthetic_df)
print(f"Statistical similarity: {quality_report.similarity_score:.2%}")
```

### Time Series with TimeGAN
```python
from synthetic_data import TimeSeriesSynthesizer

synthesizer = TimeSeriesSynthesizer(
    model="timegan",
    sequence_length=100
)

# Generate synthetic sensor data
synthetic_series = synthesizer.generate(
    real_data=sensor_readings,
    num_sequences=5000,
    anomaly_rate=0.05  # Include rare anomalies
)
```

## GAN Architectures

### DCGAN (Deep Convolutional GAN)
Basic GAN for image generation:
```python
from synthetic_data import DCGAN

gan = DCGAN(
    latent_dim=100,
    image_size=64,
    channels=3
)

gan.train(
    real_images=training_images,
    epochs=100,
    batch_size=64
)

# Generate
fake_images = gan.generate(num_samples=1000)
```

### StyleGAN2
High-quality image synthesis:
```python
from synthetic_data import StyleGAN2

stylegan = StyleGAN2(resolution=1024)

# Transfer learning from pretrained
stylegan.load_pretrained("ffhq-1024")

# Fine-tune on custom data
stylegan.fine_tune(
    custom_images=rare_images,
    epochs=50
)

# Generate with style mixing
synthetic = stylegan.generate_with_style_mixing(
    num_samples=100,
    truncation_psi=0.7
)
```

### WGAN-GP (Wasserstein GAN with Gradient Penalty)
Stable training:
```python
from synthetic_data import WGANGP

wgan = WGANGP(
    critic_iterations=5,
    gp_weight=10
)

wgan.train(training_data, epochs=200)
```

## Diffusion Models

### Stable Diffusion
Text-to-image generation:
```python
from synthetic_data import StableDiffusionSynthesizer

sd = StableDiffusionSynthesizer(
    model_id="stabilityai/stable-diffusion-2-1"
)

# Generate with prompts
images = sd.generate_batch(
    prompts=[
        "chest x-ray with pneumonia",
        "retinal scan with diabetic retinopathy",
        "brain mri with tumor"
    ],
    num_samples_per_prompt=100
)
```

### DDPM (Denoising Diffusion Probabilistic Models)
Unconditional generation:
```python
from synthetic_data import DDPM

ddpm = DDPM(
    image_size=256,
    timesteps=1000
)

ddpm.train(real_images, epochs=500)
synthetic_images = ddpm.sample(num_samples=1000)
```

## Privacy-Preserving Synthesis

### Differential Privacy
Add noise for privacy:
```python
from synthetic_data import DPSynthesizer

dp_synth = DPSynthesizer(
    epsilon=1.0,
    delta=1e-5
)

# Generate with privacy guarantees
private_synthetic_data = dp_synth.generate(
    sensitive_data=medical_records,
    num_samples=10000
)

# Verify privacy
privacy_report = dp_synth.verify_privacy(
    original=medical_records,
    synthetic=private_synthetic_data
)
```

## Quality Evaluation

### Statistical Similarity
```python
from synthetic_data import QualityEvaluator

evaluator = QualityEvaluator()

metrics = evaluator.evaluate_tabular(
    real_data=df_real,
    synthetic_data=df_synthetic
)

print(f"Column correlation: {metrics.correlation_score:.2%}")
print(f"Distribution similarity (KS test): {metrics.ks_statistic:.3f}")
print(f"Machine learning efficacy: {metrics.ml_efficacy:.2%}")
```

### Visual Quality (FID Score)
```python
# Frechet Inception Distance
fid_score = evaluator.calculate_fid(
    real_images=real_imgs,
    synthetic_images=synth_imgs
)

print(f"FID Score: {fid_score:.2f}")  # Lower is better
```

## Use Cases

### Rare Event Oversampling
```python
# Medical diagnosis - rare diseases
synthesizer = TabularSynthesizer(model="ctgan")
synthesizer.fit(medical_data)

# Generate 10x more rare disease cases
rare_cases = synthesizer.sample(
    num_rows=10000,
    conditions={'diagnosis': 'rare_disease'}
)
```

### Data Augmentation
```python
# Augment training set
augmented_data = synthesizer.augment(
    original_data=train_df,
    augmentation_factor=2.0,  # 2x original size
    preserve_distribution=True
)
```

### Privacy Compliance (GDPR)
```python
# Generate synthetic alternative to real PII data
gdpr_safe_data = dp_synth.generate_anonymous(
    real_data_with_pii=customer_data,
    pii_columns=['name', 'email', 'ssn'],
    utility_threshold=0.85
)
```

## Technologies

- **GANs**: PyTorch, TensorFlow, SDV (Synthetic Data Vault)
- **Diffusion**: Diffusers (HuggingFace), Stable Diffusion
- **Tabular**: CTGAN, TVAE, DataSynthesizer
- **Time Series**: TimeGAN, GRU-D
- **Text**: GPT-3, Faker, Synthea
- **Privacy**: Opacus, SmartNoise

## Performance

| Model | Data Type | Training Time | Quality (FID/KS) | Privacy |
|-------|-----------|---------------|------------------|---------|
| StyleGAN2 | Images | 2-5 days (8 GPUs) | FID: 2.8 | None |
| CTGAN | Tabular | 1-4 hours (1 GPU) | KS: 0.15 | Optional DP |
| TimeGAN | Time Series | 6-12 hours | DTW: 0.22 | None |
| Stable Diffusion | Images | Pretrained | FID: 12.6 | None |

## Best Practices

✅ Validate statistical similarity to real data
✅ Test downstream ML performance (ML efficacy)
✅ Check for privacy leakage (membership inference)
✅ Use conditional generation for rare classes
✅ Combine with real data (not replace entirely)
✅ Version synthetic datasets with DVC
✅ Document generation parameters

## References

- StyleGAN2: https://arxiv.org/abs/1912.04958
- CTGAN: https://arxiv.org/abs/1907.00503
- TimeGAN: https://arxiv.org/abs/1909.11616
- Diffusion Models: https://arxiv.org/abs/2006.11239
- SDV: https://sdv.dev/
