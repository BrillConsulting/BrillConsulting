# LLM Fine-Tuning Toolkit

Fine-tune language models for domain-specific tasks and improved performance.

## Features

- **Data Preparation**: Format data for OpenAI, LLaMA, Alpaca formats
- **Multiple Methods**: Full fine-tuning, LoRA, QLoRA support
- **Training Pipeline**: End-to-end training with monitoring
- **Data Validation**: Quality checks and statistics
- **Evaluation**: Comprehensive model assessment
- **Format Conversion**: Support for multiple training formats

## Technologies

- OpenAI Fine-tuning API
- Hugging Face Transformers
- PEFT (LoRA, QLoRA)
- NumPy for data processing

## Usage

```python
from fine_tuning import FineTuner

# Initialize
tuner = FineTuner(base_model="gpt-3.5-turbo")

# Prepare data
examples = [
    {"instruction": "...", "input": "...", "output": "..."},
    ...
]
tuner.prepare_data(examples, train_split=0.8)

# Validate
report = tuner.validate_data()

# Train
metrics = tuner.train(epochs=3, learning_rate=1e-5)

# Evaluate
eval_metrics = tuner.evaluate(test_data)
```

## Demo

```bash
python fine_tuning.py
```

Supports OpenAI, LLaMA, and custom model fine-tuning.
