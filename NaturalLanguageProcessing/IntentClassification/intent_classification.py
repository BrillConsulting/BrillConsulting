"""
Advanced Intent Classification System v2.0
Author: BrillConsulting
Description: Multi-method intent recognition for chatbots and conversational AI

Supports traditional ML (SVM, Naive Bayes), Transformers, and Zero-shot classification
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Traditional ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Deep Learning (optional)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")


class IntentClassifier:
    """
    Advanced Intent Classification System

    Supports multiple methods:
    - Traditional ML: SVM, Naive Bayes, Random Forest
    - Transformers: BERT-based intent classification
    - Zero-shot: Classification without training data
    """

    def __init__(self, method='ml', model_name='distilbert-base-uncased', use_gpu=False):
        """
        Initialize intent classifier

        Args:
            method: 'ml', 'transformer', 'zero-shot', 'ensemble'
            model_name: Hugging Face model name for transformer method
            use_gpu: Enable GPU acceleration
        """
        self.method = method
        self.model_name = model_name
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu' if TRANSFORMERS_AVAILABLE else 'cpu'

        # ML components
        self.vectorizer = None
        self.ml_models = {}
        self.intent_labels = []

        # Transformer components
        self.transformer_pipeline = None
        self.zero_shot_pipeline = None

        # Training data storage
        self.training_texts = []
        self.training_intents = []

        print(f"✓ IntentClassifier initialized (method={method}, device={self.device})")

    def train(self, texts: List[str], intents: List[str], test_size=0.2):
        """
        Train intent classification model

        Args:
            texts: List of training texts
            intents: List of corresponding intent labels
            test_size: Fraction of data for testing

        Returns:
            Dict with training results
        """
        if self.method == 'zero-shot':
            print("Zero-shot classification doesn't require training")
            self.intent_labels = list(set(intents))
            return {'status': 'zero-shot', 'intents': self.intent_labels}

        self.training_texts = texts
        self.training_intents = intents
        self.intent_labels = list(set(intents))

        print(f"\n📊 Training on {len(texts)} examples, {len(self.intent_labels)} intents")
        print(f"Intents: {', '.join(self.intent_labels)}")

        if self.method in ['ml', 'ensemble']:
            return self._train_ml(texts, intents, test_size)
        elif self.method == 'transformer':
            return self._train_transformer(texts, intents, test_size)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _train_ml(self, texts: List[str], intents: List[str], test_size=0.2):
        """Train traditional ML models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, intents, test_size=test_size, random_state=42, stratify=intents
        )

        # Vectorize
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train multiple models
        models = {
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }

        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)

            self.ml_models[name] = model
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'test_labels': y_test
            }
            print(f"  Accuracy: {accuracy:.4f}")

        # Ensemble model
        if len(models) > 1:
            print("\nTraining ensemble...")
            ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in models.items()],
                voting='soft'
            )
            ensemble.fit(X_train_vec, y_train)
            y_pred = ensemble.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)

            self.ml_models['ensemble'] = ensemble
            results['ensemble'] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'test_labels': y_test
            }
            print(f"  Accuracy: {accuracy:.4f}")

        # Use best model for prediction
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\n✓ Best model: {best_model} ({results[best_model]['accuracy']:.4f})")

        return {
            'status': 'success',
            'results': results,
            'best_model': best_model,
            'intents': self.intent_labels
        }

    def _train_transformer(self, texts: List[str], intents: List[str], test_size=0.2):
        """Train transformer model (simplified - uses pre-trained zero-shot)"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available")

        # For simplicity, we'll use zero-shot classification with custom labels
        # Full fine-tuning would require more complex training loop
        print("Using pre-trained transformer with zero-shot classification")
        self.transformer_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if self.device == 'cuda' else -1
        )

        return {
            'status': 'success',
            'method': 'zero-shot-transformer',
            'intents': self.intent_labels
        }

    def predict(self, text: str, return_confidence=True, top_k=3) -> Dict:
        """
        Predict intent for given text

        Args:
            text: Input text
            return_confidence: Return confidence scores
            top_k: Return top k predictions

        Returns:
            Dict with intent and confidence
        """
        if self.method == 'zero-shot':
            return self._predict_zero_shot(text, top_k)
        elif self.method in ['ml', 'ensemble']:
            return self._predict_ml(text, return_confidence, top_k)
        elif self.method == 'transformer':
            return self._predict_transformer(text, top_k)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _predict_ml(self, text: str, return_confidence=True, top_k=3) -> Dict:
        """Predict using ML models"""
        if not self.vectorizer or not self.ml_models:
            raise ValueError("Model not trained. Call train() first.")

        # Use ensemble if available, otherwise best model
        model = self.ml_models.get('ensemble') or self.ml_models.get('svm')

        # Vectorize
        text_vec = self.vectorizer.transform([text])

        # Predict
        intent = model.predict(text_vec)[0]

        result = {'intent': intent, 'text': text}

        if return_confidence and hasattr(model, 'predict_proba'):
            probas = model.predict_proba(text_vec)[0]

            # Get top k
            top_indices = np.argsort(probas)[-top_k:][::-1]
            top_intents = []

            for idx in top_indices:
                top_intents.append({
                    'intent': model.classes_[idx],
                    'confidence': float(probas[idx])
                })

            result['confidence'] = float(probas[list(model.classes_).index(intent)])
            result['top_predictions'] = top_intents

        return result

    def _predict_transformer(self, text: str, top_k=3) -> Dict:
        """Predict using transformer"""
        if not self.transformer_pipeline:
            raise ValueError("Transformer model not initialized")

        result = self.transformer_pipeline(text, self.intent_labels, multi_label=False)

        return {
            'intent': result['labels'][0],
            'confidence': result['scores'][0],
            'text': text,
            'top_predictions': [
                {'intent': label, 'confidence': score}
                for label, score in zip(result['labels'][:top_k], result['scores'][:top_k])
            ]
        }

    def _predict_zero_shot(self, text: str, top_k=3) -> Dict:
        """Predict using zero-shot classification"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available")

        if not self.intent_labels:
            raise ValueError("No intent labels defined")

        if not self.zero_shot_pipeline:
            self.zero_shot_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == 'cuda' else -1
            )

        result = self.zero_shot_pipeline(text, self.intent_labels, multi_label=False)

        return {
            'intent': result['labels'][0],
            'confidence': result['scores'][0],
            'text': text,
            'top_predictions': [
                {'intent': label, 'confidence': score}
                for label, score in zip(result['labels'][:top_k], result['scores'][:top_k])
            ]
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict intents for multiple texts"""
        return [self.predict(text) for text in texts]

    def evaluate(self, test_texts: List[str], test_intents: List[str]) -> Dict:
        """
        Evaluate model on test data

        Returns:
            Dict with metrics
        """
        predictions = [self.predict(text)['intent'] for text in test_texts]

        accuracy = accuracy_score(test_intents, predictions)

        return {
            'accuracy': accuracy,
            'classification_report': classification_report(
                test_intents, predictions, target_names=self.intent_labels
            ),
            'confusion_matrix': confusion_matrix(test_intents, predictions).tolist()
        }

    def save(self, path: str):
        """Save model to disk"""
        import pickle

        model_data = {
            'method': self.method,
            'vectorizer': self.vectorizer,
            'ml_models': self.ml_models,
            'intent_labels': self.intent_labels,
            'training_texts': self.training_texts,
            'training_intents': self.training_intents
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        import pickle

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.method = model_data['method']
        self.vectorizer = model_data['vectorizer']
        self.ml_models = model_data['ml_models']
        self.intent_labels = model_data['intent_labels']
        self.training_texts = model_data.get('training_texts', [])
        self.training_intents = model_data.get('training_intents', [])

        print(f"✓ Model loaded from {path}")


def create_sample_training_data():
    """Create sample training data for common chatbot intents"""
    intents_data = {
        'greeting': [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "howdy", "greetings", "what's up", "hi there", "hello there"
        ],
        'goodbye': [
            "bye", "goodbye", "see you", "farewell", "take care",
            "see you later", "catch you later", "talk to you soon", "have a good day"
        ],
        'thanks': [
            "thank you", "thanks", "appreciate it", "thank you so much",
            "many thanks", "thanks a lot", "grateful", "much appreciated"
        ],
        'help': [
            "help", "can you help me", "i need help", "assistance please",
            "support", "i need support", "help me out", "can you assist"
        ],
        'order_status': [
            "where is my order", "order status", "track my order",
            "when will my order arrive", "check order", "order tracking"
        ],
        'refund': [
            "i want a refund", "refund please", "return my order",
            "cancel my order", "money back", "give me my money back"
        ],
        'product_info': [
            "tell me about this product", "product information",
            "what are the features", "product details", "specifications"
        ]
    }

    texts = []
    labels = []
    for intent, examples in intents_data.items():
        texts.extend(examples)
        labels.extend([intent] * len(examples))

    return texts, labels


if __name__ == "__main__":
    print("=" * 80)
    print("Advanced Intent Classification System v2.0")
    print("Author: BrillConsulting")
    print("=" * 80)

    # Create sample data
    texts, intents = create_sample_training_data()
    print(f"\n📊 Sample Data: {len(texts)} examples, {len(set(intents))} intents")

    # Method 1: Traditional ML
    print("\n" + "=" * 80)
    print("Method 1: Traditional ML (SVM, Naive Bayes, Random Forest)")
    print("=" * 80)

    classifier_ml = IntentClassifier(method='ml')
    results = classifier_ml.train(texts, intents)

    # Test predictions
    test_texts = [
        "hi how are you",
        "i need help with my account",
        "thanks for your help",
        "where is my package"
    ]

    print("\n🔮 Predictions:")
    for text in test_texts:
        result = classifier_ml.predict(text)
        print(f"Text: '{text}'")
        print(f"  Intent: {result['intent']} (confidence: {result.get('confidence', 0):.4f})")
        if 'top_predictions' in result:
            print(f"  Top 3: {[(p['intent'], f\"{p['confidence']:.3f}\") for p in result['top_predictions'][:3]]}")

    # Method 2: Zero-shot (if transformers available)
    if TRANSFORMERS_AVAILABLE:
        print("\n" + "=" * 80)
        print("Method 2: Zero-Shot Classification")
        print("=" * 80)

        classifier_zero = IntentClassifier(method='zero-shot')
        classifier_zero.intent_labels = list(set(intents))

        print("\n🔮 Zero-shot Predictions:")
        for text in test_texts[:2]:  # Just 2 examples for speed
            result = classifier_zero.predict(text)
            print(f"Text: '{text}'")
            print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.4f})")

    print("\n" + "=" * 80)
    print("✓ Demo completed successfully!")
    print("=" * 80)
