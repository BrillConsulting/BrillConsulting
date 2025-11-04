"""
Text Classification System
Author: BrillConsulting
Description: Sentiment analysis and text classification with traditional ML and Transformers
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import joblib
from typing import List, Dict, Tuple
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)


class TextClassifier:
    """
    Multi-algorithm text classification system
    """

    def __init__(self, vectorizer_type='tfidf', max_features=5000):
        """
        Initialize text classifier

        Args:
            vectorizer_type: 'tfidf' or 'count'
            max_features: Maximum number of features
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features

        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        else:
            self.vectorizer = CountVectorizer(max_features=max_features)

        self.models = {}
        self.results = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercase, remove special chars, lemmatize

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]

        return ' '.join(tokens)

    def prepare_data(self, texts: List[str], labels: List,
                    test_size: float = 0.2) -> Tuple:
        """
        Preprocess and vectorize texts

        Args:
            texts: List of text documents
            labels: List of labels
            test_size: Test set proportion

        Returns:
            X_train, X_test, y_train, y_test
        """
        print("🔧 Preprocessing texts...")

        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=test_size,
            random_state=42, stratify=labels
        )

        # Vectorize
        print(f"🔧 Vectorizing with {self.vectorizer_type.upper()}...")
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)

        print(f"✅ Feature matrix: {X_train.shape}\n")

        return X_train, X_test, y_train, y_test

    def train_naive_bayes(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Naive Bayes classifier"""
        print("🔧 Training Naive Bayes...")

        model = MultinomialNB()
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        results = {
            'model': model,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'predictions': y_pred_test,
            'report': classification_report(y_test, y_pred_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test)
        }

        self.models['Naive Bayes'] = model
        self.results['Naive Bayes'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f}\n")
        return results

    def train_logistic_regression(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Logistic Regression"""
        print("🔧 Training Logistic Regression...")

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        results = {
            'model': model,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'predictions': y_pred_test,
            'report': classification_report(y_test, y_pred_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test)
        }

        self.models['Logistic Regression'] = model
        self.results['Logistic Regression'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f}\n")
        return results

    def train_svm(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Linear SVM"""
        print("🔧 Training Linear SVM...")

        model = LinearSVC(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        results = {
            'model': model,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'predictions': y_pred_test,
            'report': classification_report(y_test, y_pred_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test)
        }

        self.models['Linear SVM'] = model
        self.results['Linear SVM'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f}\n")
        return results

    def train_all_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train all classification models"""
        print("🚀 Training all models...\n")

        self.train_naive_bayes(X_train, y_train, X_test, y_test)
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_svm(X_train, y_train, X_test, y_test)

        return self.results

    def compare_models(self) -> pd.DataFrame:
        """Compare all models"""
        comparison = []

        for name, results in self.results.items():
            comparison.append({
                'Model': name,
                'Train Accuracy': results['train_accuracy'],
                'Test Accuracy': results['test_accuracy']
            })

        df = pd.DataFrame(comparison)
        df = df.sort_values('Test Accuracy', ascending=False)

        return df

    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

        if n_models == 1:
            axes = [axes]

        for idx, (name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'{name}\nAccuracy: {results["test_accuracy"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrices saved to {save_path}")

        plt.show()

    def predict(self, texts: List[str], model_name='Logistic Regression') -> List:
        """
        Predict labels for new texts

        Args:
            texts: List of text documents
            model_name: Which model to use

        Returns:
            List of predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        # Preprocess
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Vectorize
        X = self.vectorizer.transform(processed_texts)

        # Predict
        predictions = self.models[model_name].predict(X)

        return predictions

    def save_model(self, model_name: str, filepath: str):
        """Save model and vectorizer"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        joblib.dump({
            'model': self.models[model_name],
            'vectorizer': self.vectorizer
        }, filepath)

        print(f"💾 Model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Text Classification')
    parser.add_argument('--data', type=str, required=True,
                       help='CSV file with text and label columns')
    parser.add_argument('--text-col', type=str, default='text',
                       help='Name of text column')
    parser.add_argument('--label-col', type=str, default='label',
                       help='Name of label column')
    parser.add_argument('--vectorizer', type=str, default='tfidf',
                       choices=['tfidf', 'count'],
                       help='Vectorizer type')
    parser.add_argument('--max-features', type=int, default=5000,
                       help='Maximum number of features')
    parser.add_argument('--output', type=str,
                       help='Output plot path')
    parser.add_argument('--save-model', type=str,
                       help='Save best model path')

    args = parser.parse_args()

    # Load data
    print(f"📂 Loading data from {args.data}...\n")
    df = pd.read_csv(args.data)

    texts = df[args.text_col].tolist()
    labels = df[args.label_col].tolist()

    print(f"📊 Dataset: {len(texts)} documents")
    print(f"🏷️  Labels: {set(labels)}\n")

    # Initialize classifier
    classifier = TextClassifier(
        vectorizer_type=args.vectorizer,
        max_features=args.max_features
    )

    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(texts, labels)

    # Train models
    classifier.train_all_models(X_train, y_train, X_test, y_test)

    # Compare
    print("📊 Model Comparison:")
    print("=" * 60)
    comparison = classifier.compare_models()
    print(comparison.to_string(index=False))
    print("=" * 60)

    # Best model
    best_model = comparison.iloc[0]['Model']
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Test Accuracy: {comparison.iloc[0]['Test Accuracy']:.4f}\n")

    # Classification report
    print("📋 Classification Report (Best Model):")
    print(classifier.results[best_model]['report'])

    # Plot
    if args.output:
        classifier.plot_confusion_matrices(save_path=args.output)

    # Save
    if args.save_model:
        classifier.save_model(best_model, args.save_model)


if __name__ == "__main__":
    main()
