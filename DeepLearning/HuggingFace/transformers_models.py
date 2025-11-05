"""
Hugging Face Transformers Models
Author: BrillConsulting
Description: State-of-the-art transformers for NLP, vision, audio, and multimodal tasks
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class HuggingFaceTransformers:
    """Comprehensive Hugging Face Transformers implementations"""

    def __init__(self):
        """Initialize Transformers manager"""
        self.models = []
        self.pipelines = []

    def create_text_classification_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create text classification pipeline

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline details
        """
        pipeline = {
            'name': config.get('name', 'TextClassification'),
            'task': 'text-classification',
            'model': config.get('model', 'bert-base-uncased'),
            'num_labels': config.get('num_labels', 2),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "{pipeline['model']}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels={pipeline['num_labels']}
)

# Create pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Make predictions
texts = [
    "I love this product! It's amazing!",
    "This is terrible. Waste of money."
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"Text: {{text}}")
    print(f"Label: {{result['label']}}, Score: {{result['score']:.4f}}\\n")

# Fine-tuning
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset('imdb')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)

trainer.train()
"""

        pipeline['code'] = code
        self.pipelines.append(pipeline)

        print(f"✓ Text classification pipeline created: {pipeline['name']}")
        print(f"  Model: {pipeline['model']}, Labels: {pipeline['num_labels']}")
        return pipeline

    def create_token_classification_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create token classification (NER) pipeline

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline details
        """
        pipeline = {
            'name': config.get('name', 'NER'),
            'task': 'token-classification',
            'model': config.get('model', 'dbmdz/bert-large-cased-finetuned-conll03-english'),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Load model
model_name = "{pipeline['model']}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create NER pipeline
ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# Extract entities
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
entities = ner(text)

for entity in entities:
    print(f"Entity: {{entity['word']}}")
    print(f"Type: {{entity['entity_group']}}")
    print(f"Score: {{entity['score']:.4f}}\\n")
"""

        pipeline['code'] = code
        self.pipelines.append(pipeline)

        print(f"✓ NER pipeline created: {pipeline['name']}")
        print(f"  Model: {pipeline['model']}")
        return pipeline

    def create_question_answering_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create question answering pipeline

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline details
        """
        pipeline = {
            'name': config.get('name', 'QuestionAnswering'),
            'task': 'question-answering',
            'model': config.get('model', 'distilbert-base-cased-distilled-squad'),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
from transformers import pipeline

# Create QA pipeline
qa = pipeline("question-answering", model="{pipeline['model']}")

# Answer questions
context = \"\"\"
Transformers is a library by Hugging Face that provides state-of-the-art
pre-trained models for Natural Language Processing (NLP). It supports
PyTorch, TensorFlow, and JAX.
\"\"\"

questions = [
    "Who created Transformers?",
    "What frameworks does it support?",
    "What is Transformers used for?"
]

for question in questions:
    result = qa(question=question, context=context)
    print(f"Q: {{question}}")
    print(f"A: {{result['answer']}}")
    print(f"Score: {{result['score']:.4f}}\\n")
"""

        pipeline['code'] = code
        self.pipelines.append(pipeline)

        print(f"✓ QA pipeline created: {pipeline['name']}")
        return pipeline

    def create_text_generation_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create text generation pipeline

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline details
        """
        pipeline = {
            'name': config.get('name', 'TextGeneration'),
            'task': 'text-generation',
            'model': config.get('model', 'gpt2'),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
from transformers import pipeline, set_seed

# Set seed for reproducibility
set_seed(42)

# Create text generation pipeline
generator = pipeline("text-generation", model="{pipeline['model']}")

# Generate text
prompts = [
    "Once upon a time",
    "The future of artificial intelligence is"
]

for prompt in prompts:
    outputs = generator(
        prompt,
        max_length=100,
        num_return_sequences=2,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    print(f"Prompt: {{prompt}}\\n")
    for i, output in enumerate(outputs, 1):
        print(f"Generation {{i}}: {{output['generated_text']}}\\n")
"""

        pipeline['code'] = code
        self.pipelines.append(pipeline)

        print(f"✓ Text generation pipeline created: {pipeline['name']}")
        return pipeline

    def create_image_classification_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create vision transformer pipeline

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline details
        """
        pipeline = {
            'name': config.get('name', 'ImageClassification'),
            'task': 'image-classification',
            'model': config.get('model', 'google/vit-base-patch16-224'),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
from transformers import pipeline
from PIL import Image

# Create image classification pipeline
classifier = pipeline("image-classification", model="{pipeline['model']}")

# Classify images
image = Image.open("cat.jpg")
results = classifier(image, top_k=5)

print("Top 5 predictions:")
for result in results:
    print(f"{{result['label']}}: {{result['score']:.4f}}")
"""

        pipeline['code'] = code
        self.pipelines.append(pipeline)

        print(f"✓ Image classification pipeline created: {pipeline['name']}")
        return pipeline

    def create_translation_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create translation pipeline

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline details
        """
        pipeline = {
            'name': config.get('name', 'Translation'),
            'task': 'translation',
            'model': config.get('model', 'Helsinki-NLP/opus-mt-en-de'),
            'source_lang': config.get('source_lang', 'en'),
            'target_lang': config.get('target_lang', 'de'),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
from transformers import pipeline

# Create translation pipeline
translator = pipeline("translation", model="{pipeline['model']}")

# Translate texts
texts = [
    "Hello, how are you?",
    "I love machine learning and artificial intelligence."
]

for text in texts:
    result = translator(text, max_length=128)
    print(f"EN: {{text}}")
    print(f"DE: {{result[0]['translation_text']}}\\n")
"""

        pipeline['code'] = code
        self.pipelines.append(pipeline)

        print(f"✓ Translation pipeline created: {pipeline['name']}")
        return pipeline

    def create_custom_training(self) -> str:
        """Generate custom training code"""

        code = """
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save model
trainer.save_model("./my_model")
"""

        print("✓ Custom training code generated")
        return code

    def get_transformers_info(self) -> Dict[str, Any]:
        """Get Transformers manager information"""
        return {
            'pipelines_created': len(self.pipelines),
            'library': 'Hugging Face Transformers',
            'version': '4.35.0',
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Hugging Face Transformers"""

    print("=" * 60)
    print("Hugging Face Transformers Models Demo")
    print("=" * 60)

    hf = HuggingFaceTransformers()

    print("\n1. Creating Text Classification pipeline...")
    text_clf = hf.create_text_classification_pipeline({
        'name': 'SentimentAnalysis',
        'model': 'distilbert-base-uncased-finetuned-sst-2-english'
    })
    print(text_clf['code'][:300] + "...\n")

    print("\n2. Creating NER pipeline...")
    ner = hf.create_token_classification_pipeline({
        'name': 'EntityExtraction'
    })
    print(ner['code'][:300] + "...\n")

    print("\n3. Creating Question Answering pipeline...")
    qa = hf.create_question_answering_pipeline({
        'name': 'QA'
    })
    print(qa['code'][:300] + "...\n")

    print("\n4. Creating Text Generation pipeline...")
    gen = hf.create_text_generation_pipeline({
        'name': 'TextGenerator',
        'model': 'gpt2'
    })
    print(gen['code'][:300] + "...\n")

    print("\n5. Creating Image Classification pipeline...")
    img_clf = hf.create_image_classification_pipeline({
        'name': 'ImageClassifier'
    })
    print(img_clf['code'][:200] + "...\n")

    print("\n6. Creating Translation pipeline...")
    trans = hf.create_translation_pipeline({
        'name': 'EnglishToGerman',
        'source_lang': 'en',
        'target_lang': 'de'
    })
    print(trans['code'][:200] + "...\n")

    print("\n7. Generating custom training code...")
    training = hf.create_custom_training()
    print(training[:300] + "...\n")

    print("\n8. Transformers summary:")
    info = hf.get_transformers_info()
    print(f"  Pipelines created: {info['pipelines_created']}")
    print(f"  Library: {info['library']} {info['version']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
