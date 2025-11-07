"""
Text Classification System v2.0
Author: BrillConsulting
Description: Advanced text classification with Transformers, Deep Learning, and ensemble methods
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import joblib
from typing import List, Dict, Tuple, Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                         Trainer, TrainingArguments, pipeline)
from datasets import Dataset as HFDataset
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)


class TextDataset(Dataset):
    """PyTorch Dataset for text classification"""

    def __init__(self, texts, labels, vectorizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert to vector
        vector = self.vectorizer.transform([text]).toarray()[0]

        return {
            'input': torch.FloatTensor(vector),
            'label': torch.LongTensor([label])[0]
        }


class DeepTextClassifier(nn.Module):
    """Deep Neural Network for text classification"""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128], num_classes=2, dropout=0.5):
        super(DeepTextClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AdvancedTextClassifier:
    """
    Advanced multi-algorithm text classification system with Transformers and Deep Learning
    """

    def __init__(self, vectorizer_type='tfidf', max_features=5000, use_gpu=True):
        """
        Initialize advanced text classifier

        Args:
            vectorizer_type: 'tfidf' or 'count'
            max_features: Maximum number of features
            use_gpu: Whether to use GPU for deep learning
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
                min_df=2,
                max_df=0.9
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.9
            )

        self.models = {}
        self.results = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoder = {}
        self.reverse_label_encoder = {}

    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing

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

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]

        return ' '.join(tokens)

    def encode_labels(self, labels: List) -> np.ndarray:
        """Encode string labels to integers"""
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        return np.array([self.label_encoder[label] for label in labels])

    def decode_labels(self, encoded_labels: np.ndarray) -> List:
        """Decode integer labels to strings"""
        return [self.reverse_label_encoder[label] for label in encoded_labels]

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

        # Encode labels
        encoded_labels = self.encode_labels(labels)

        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_texts, encoded_labels, test_size=test_size,
            random_state=42, stratify=encoded_labels
        )

        # Store for transformer models
        self.X_train_text = X_train_text
        self.X_test_text = X_test_text

        # Vectorize
        print(f"🔧 Vectorizing with {self.vectorizer_type.upper()}...")
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)

        print(f"✅ Feature matrix: {X_train.shape}")
        print(f"✅ Classes: {len(self.label_encoder)}\n")

        return X_train, X_test, y_train, y_test

    def train_naive_bayes(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Naive Bayes classifier with hyperparameter tuning"""
        print("🔧 Training Naive Bayes...")

        best_score = 0
        best_alpha = 1.0

        for alpha in [0.1, 0.5, 1.0, 2.0]:
            model = MultinomialNB(alpha=alpha)
            scores = cross_val_score(model, X_train, y_train, cv=5)
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_alpha = alpha

        model = MultinomialNB(alpha=best_alpha)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        results = self._compute_metrics(y_test, y_pred_test, model, X_test)
        results['model'] = model
        results['best_alpha'] = best_alpha

        self.models['Naive Bayes'] = model
        self.results['Naive Bayes'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f} | F1: {results['f1_score']:.4f}\n")
        return results

    def train_logistic_regression(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Logistic Regression with L2 regularization"""
        print("🔧 Training Logistic Regression...")

        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        results = self._compute_metrics(y_test, y_pred_test, model, X_test)
        results['model'] = model

        self.models['Logistic Regression'] = model
        self.results['Logistic Regression'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f} | F1: {results['f1_score']:.4f}\n")
        return results

    def train_svm(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Linear SVM"""
        print("🔧 Training Linear SVM...")

        model = LinearSVC(max_iter=2000, random_state=42, C=1.0, class_weight='balanced')
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        results = self._compute_metrics(y_test, y_pred_test, model, X_test)
        results['model'] = model

        self.models['Linear SVM'] = model
        self.results['Linear SVM'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f} | F1: {results['f1_score']:.4f}\n")
        return results

    def train_random_forest(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Random Forest classifier"""
        print("🔧 Training Random Forest...")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=50,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        results = self._compute_metrics(y_test, y_pred_test, model, X_test)
        results['model'] = model

        self.models['Random Forest'] = model
        self.results['Random Forest'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f} | F1: {results['f1_score']:.4f}\n")
        return results

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Gradient Boosting classifier"""
        print("🔧 Training Gradient Boosting...")

        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        results = self._compute_metrics(y_test, y_pred_test, model, X_test)
        results['model'] = model

        self.models['Gradient Boosting'] = model
        self.results['Gradient Boosting'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f} | F1: {results['f1_score']:.4f}\n")
        return results

    def train_deep_neural_network(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=32) -> Dict:
        """Train Deep Neural Network with PyTorch"""
        print("🔧 Training Deep Neural Network...")

        num_classes = len(self.label_encoder)
        input_dim = X_train.shape[1]

        # Create datasets
        train_dataset = TextDataset(
            self.X_train_text,
            y_train,
            self.vectorizer
        )
        test_dataset = TextDataset(
            self.X_test_text,
            y_test,
            self.vectorizer
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize model
        model = DeepTextClassifier(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128],
            num_classes=num_classes,
            dropout=0.5
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                inputs = batch['input'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input'].to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())

        y_pred_test = np.array(predictions)

        results = self._compute_metrics(y_test, y_pred_test, None, None)
        results['model'] = model

        self.models['Deep Neural Network'] = model
        self.results['Deep Neural Network'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f} | F1: {results['f1_score']:.4f}\n")
        return results

    def train_transformer(self, model_name='distilbert-base-uncased', epochs=3, batch_size=16) -> Dict:
        """Train Transformer model (BERT, RoBERTa, DistilBERT)"""
        print(f"🔧 Training Transformer ({model_name})...")

        num_classes = len(self.label_encoder)

        # Prepare dataset
        train_texts = self.X_train_text
        test_texts = self.X_test_text
        train_labels = self.results.get('Logistic Regression', {}).get('y_train', [])
        test_labels = self.results.get('Logistic Regression', {}).get('y_test', [])

        # Create HuggingFace datasets
        train_dataset = HFDataset.from_dict({
            'text': train_texts,
            'label': train_labels.tolist() if hasattr(train_labels, 'tolist') else list(train_labels)
        })

        test_dataset = HFDataset.from_dict({
            'text': test_texts,
            'label': test_labels.tolist() if hasattr(test_labels, 'tolist') else list(test_labels)
        })

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )

        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        # Train
        trainer.train()

        # Predict
        predictions = trainer.predict(test_dataset)
        y_pred_test = np.argmax(predictions.predictions, axis=1)

        results = self._compute_metrics(test_labels, y_pred_test, None, None)
        results['model'] = model
        results['tokenizer'] = tokenizer

        self.models['Transformer'] = model
        self.results['Transformer'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f} | F1: {results['f1_score']:.4f}\n")
        return results

    def train_ensemble(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train ensemble of best models"""
        print("🔧 Training Ensemble (Voting Classifier)...")

        # Select best traditional models
        estimators = []
        if 'Logistic Regression' in self.models:
            estimators.append(('lr', self.models['Logistic Regression']))
        if 'Linear SVM' in self.models:
            estimators.append(('svm', self.models['Linear SVM']))
        if 'Random Forest' in self.models:
            estimators.append(('rf', self.models['Random Forest']))

        if len(estimators) < 2:
            print("  ⚠️ Need at least 2 models for ensemble")
            return {}

        model = VotingClassifier(estimators=estimators, voting='hard')
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        results = self._compute_metrics(y_test, y_pred_test, model, X_test)
        results['model'] = model

        self.models['Ensemble'] = model
        self.results['Ensemble'] = results

        print(f"  Accuracy: {results['test_accuracy']:.4f} | F1: {results['f1_score']:.4f}\n")
        return results

    def _compute_metrics(self, y_true, y_pred, model, X_test) -> Dict:
        """Compute comprehensive evaluation metrics"""
        results = {
            'test_accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'predictions': y_pred,
            'y_test': y_true,
            'report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

        # Add ROC AUC for binary classification
        if len(self.label_encoder) == 2 and model is not None and X_test is not None:
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    results['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                pass

        return results

    def train_all_models(self, X_train, y_train, X_test, y_test,
                        include_deep_learning=True, include_transformer=False) -> Dict:
        """Train all classification models"""
        print("🚀 Training all models...\n")

        # Store labels for transformer
        if 'Logistic Regression' not in self.results:
            self.results['Logistic Regression'] = {'y_train': y_train, 'y_test': y_test}

        # Traditional ML
        self.train_naive_bayes(X_train, y_train, X_test, y_test)
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_svm(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)

        # Ensemble
        self.train_ensemble(X_train, y_train, X_test, y_test)

        # Deep Learning
        if include_deep_learning:
            try:
                self.train_deep_neural_network(X_train, y_train, X_test, y_test, epochs=10)
            except Exception as e:
                print(f"  ⚠️ Deep learning training failed: {e}\n")

        # Transformer
        if include_transformer:
            try:
                self.train_transformer(epochs=2)
            except Exception as e:
                print(f"  ⚠️ Transformer training failed: {e}\n")

        return self.results

    def compare_models(self) -> pd.DataFrame:
        """Compare all models with comprehensive metrics"""
        comparison = []

        for name, results in self.results.items():
            if 'test_accuracy' in results:
                row = {
                    'Model': name,
                    'Accuracy': results['test_accuracy'],
                    'F1 Score': results['f1_score'],
                    'Precision': results['precision'],
                    'Recall': results['recall']
                }
                if 'roc_auc' in results:
                    row['ROC AUC'] = results['roc_auc']
                comparison.append(row)

        df = pd.DataFrame(comparison)
        df = df.sort_values('F1 Score', ascending=False)

        return df

    def plot_model_comparison(self, save_path=None):
        """Plot comprehensive model comparison"""
        comparison = self.compare_models()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy comparison
        axes[0, 0].barh(comparison['Model'], comparison['Accuracy'], color='skyblue')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xlim([0, 1])

        # F1 Score comparison
        axes[0, 1].barh(comparison['Model'], comparison['F1 Score'], color='lightgreen')
        axes[0, 1].set_xlabel('F1 Score')
        axes[0, 1].set_title('Model F1 Score Comparison')
        axes[0, 1].set_xlim([0, 1])

        # Precision vs Recall
        axes[1, 0].scatter(comparison['Precision'], comparison['Recall'], s=100, alpha=0.6)
        for idx, row in comparison.iterrows():
            axes[1, 0].annotate(row['Model'], (row['Precision'], row['Recall']),
                              fontsize=8, ha='right')
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_ylim([0, 1])

        # Metrics heatmap
        metrics_data = comparison[['Accuracy', 'F1 Score', 'Precision', 'Recall']].T
        metrics_data.columns = comparison['Model']
        sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[1, 1])
        axes[1, 1].set_title('All Metrics Heatmap')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all models"""
        n_models = sum(1 for r in self.results.values() if 'confusion_matrix' in r)

        if n_models == 0:
            print("❌ No confusion matrices to plot")
            return

        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        idx = 0
        for name, results in self.results.items():
            if 'confusion_matrix' in results:
                cm = results['confusion_matrix']

                sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
                axes[idx].set_title(f'{name}\nAcc: {results["test_accuracy"]:.3f} | F1: {results["f1_score"]:.3f}')
                axes[idx].set_ylabel('True Label')
                axes[idx].set_xlabel('Predicted Label')
                idx += 1

        # Hide empty subplots
        for i in range(idx, len(axes)):
            axes[i].axis('off')

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
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")

        # Preprocess
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Vectorize
        X = self.vectorizer.transform(processed_texts)

        # Predict
        model = self.models[model_name]

        if isinstance(model, DeepTextClassifier):
            model.eval()
            with torch.no_grad():
                inputs = torch.FloatTensor(X.toarray()).to(self.device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                return self.decode_labels(predictions.cpu().numpy())
        else:
            predictions = model.predict(X)
            return self.decode_labels(predictions)

    def save_model(self, model_name: str, filepath: str):
        """Save model and vectorizer"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        joblib.dump({
            'model': self.models[model_name],
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'reverse_label_encoder': self.reverse_label_encoder
        }, filepath)

        print(f"💾 Model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Advanced Text Classification v2.0')
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
    parser.add_argument('--deep-learning', action='store_true',
                       help='Include deep learning models')
    parser.add_argument('--transformer', action='store_true',
                       help='Include transformer models')
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
    classifier = AdvancedTextClassifier(
        vectorizer_type=args.vectorizer,
        max_features=args.max_features
    )

    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(texts, labels)

    # Train models
    classifier.train_all_models(
        X_train, y_train, X_test, y_test,
        include_deep_learning=args.deep_learning,
        include_transformer=args.transformer
    )

    # Compare
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON")
    print("=" * 80)
    comparison = classifier.compare_models()
    print(comparison.to_string(index=False))
    print("=" * 80)

    # Best model
    best_model = comparison.iloc[0]['Model']
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   F1 Score: {comparison.iloc[0]['F1 Score']:.4f}")
    print(f"   Accuracy: {comparison.iloc[0]['Accuracy']:.4f}\n")

    # Classification report
    print("📋 Classification Report (Best Model):")
    print(classifier.results[best_model]['report'])

    # Plot
    if args.output:
        classifier.plot_model_comparison(save_path=args.output.replace('.png', '_comparison.png'))
        classifier.plot_confusion_matrices(save_path=args.output)

    # Save
    if args.save_model:
        classifier.save_model(best_model, args.save_model)


if __name__ == "__main__":
    main()
