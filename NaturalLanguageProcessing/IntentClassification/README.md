# Advanced Intent Classification System v2.0

**Author:** BrillConsulting | **Version:** 2.0 - ML & Transformers
**Methods:** Traditional ML, BERT, Zero-Shot Classification

## Overview

Production-ready intent classification for chatbots and dialogue systems. Supports both traditional ML (fast training) and transformer-based models (higher accuracy). Perfect for customer support, virtual assistants, and conversational AI.

## Features

- **Dual Approach:** Traditional ML (Logistic Regression) + Transformers (BERT)
- **Zero-Shot Classification:** Classify without training data
- **Confidence Scores:** Probability distributions for all intents
- **Multi-Intent:** Support for multiple intents per utterance
- **Custom Training:** Train on domain-specific data
- **Real-Time:** Low latency for production use

## Installation

```bash
pip install transformers torch scikit-learn numpy
```

## Quick Start

```python
from intent_classification import IntentClassifier

# Train traditional ML
classifier = IntentClassifier(method='ml')
classifier.train(texts, intents)

# Predict
result = classifier.predict("I want to book a flight")
print(result['intent'])  # 'book_flight'
print(result['confidence'])  # 0.95
```

## Common Intents

### E-commerce
- `product_search`, `add_to_cart`, `checkout`, `track_order`, `return_item`

### Customer Support
- `technical_issue`, `billing_question`, `account_access`, `feature_request`

### Banking
- `check_balance`, `transfer_money`, `report_fraud`, `apply_loan`

### Travel
- `book_flight`, `hotel_reservation`, `cancel_booking`, `travel_info`

## Methods Comparison

| Method | Training Time | Accuracy | Inference Speed | Training Data |
|--------|---------------|----------|-----------------|---------------|
| Traditional ML | 1 second | 85-90% | 1ms | 100+ examples |
| BERT | 10 minutes | 92-96% | 50ms | 100+ examples |
| Zero-Shot | None | 75-85% | 100ms | None needed |

## Usage Examples

### 1. Traditional ML Training

```python
import json

# Load training data
with open('intents.json') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
intents = [item['intent'] for item in data]

# Train
classifier = IntentClassifier(method='ml')
classifier.train(texts, intents)

# Predict
result = classifier.predict("Cancel my subscription")
```

**Training Data Format (intents.json):**
```json
[
  {"text": "I want to book a flight", "intent": "book_flight"},
  {"text": "Cancel my reservation", "intent": "cancel_booking"},
  {"text": "Show me available hotels", "intent": "hotel_search"}
]
```

### 2. Zero-Shot Classification

```python
# No training needed!
classifier = IntentClassifier(method='transformer')

# Define intents
classifier.intents = [
    'book_flight',
    'cancel_booking',
    'check_status',
    'customer_support'
]

# Predict
result = classifier.predict("I need help with my order")
print(result)
```

### 3. Multi-Intent Detection

```python
result = classifier.predict("Book a flight and reserve a hotel")

# Get top 3 intents
for intent, score in list(result['all_intents'].items())[:3]:
    if score > 0.3:
        print(f"{intent}: {score:.2f}")
```

### 4. Confidence Threshold

```python
result = classifier.predict("What's the weather?")

if result['confidence'] < 0.5:
    print("Low confidence - ask for clarification")
else:
    execute_intent(result['intent'])
```

## Command Line

```bash
# Train on data
python intent_classification.py \
    --train intents.json \
    --method ml

# Predict single text
python intent_classification.py \
    --train intents.json \
    --text "Book a flight to Paris" \
    --method ml

# Use transformer model
python intent_classification.py \
    --train intents.json \
    --text "Cancel my order" \
    --method transformer
```

## Integration Examples

### 1. Flask API

```python
from flask import Flask, request, jsonify
from intent_classification import IntentClassifier

app = Flask(__name__)
classifier = IntentClassifier(method='ml')
classifier.train(training_texts, training_intents)

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    result = classifier.predict(text)
    return jsonify(result)
```

### 2. Chatbot Integration

```python
def handle_message(user_message):
    # Classify intent
    result = classifier.predict(user_message)
    intent = result['intent']
    confidence = result['confidence']
    
    if confidence < 0.5:
        return "I'm not sure I understood. Could you rephrase?"
    
    # Route to intent handler
    if intent == 'book_flight':
        return book_flight_handler(user_message)
    elif intent == 'cancel_booking':
        return cancel_booking_handler(user_message)
    # ...
```

### 3. Rasa Integration

```python
from rasa.nlu.components import Component

class BrillIntentClassifier(Component):
    def __init__(self, component_config=None):
        super().__init__(component_config)
        self.classifier = IntentClassifier(method='ml')
    
    def process(self, message, **kwargs):
        result = self.classifier.predict(message.text)
        message.set("intent", result['intent'], confidence=result['confidence'])
```

## Performance Optimization

### For Speed
```python
# Use traditional ML
classifier = IntentClassifier(method='ml')
# Inference: ~1ms per prediction
```

### For Accuracy
```python
# Use transformer with more training data
classifier = IntentClassifier(method='transformer')
# Need 100+ examples per intent
```

### For Zero Training
```python
# Use zero-shot
classifier = IntentClassifier(method='transformer')
classifier.intents = ['intent1', 'intent2', ...]
# No training data needed
```

## Best Practices

1. **Balanced Dataset:** 50-100 examples per intent
2. **Clear Intent Definitions:** Avoid overlapping intents
3. **Confidence Thresholds:** Set based on use case
4. **Fallback Intent:** For low confidence predictions
5. **Regular Retraining:** Update with new examples

## Troubleshooting

**Problem:** Low accuracy
**Solution:** Add more training examples, try transformer method

**Problem:** Slow inference
**Solution:** Use traditional ML method

**Problem:** Intent confusion
**Solution:** Review intent definitions, add distinguishing examples

---

**BrillConsulting** - Advanced NLP Solutions
