"""
Multi-Algorithm Classification System
Author: BrillConsulting
Description: Comprehensive classification with 6+ algorithms and automatic evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import joblib


class ClassificationAnalyzer:
    """
    Complete classification system with multiple algorithms
    """

    def __init__(self, scale_features: bool = True):
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Prepare and split data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if self.scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all classification models"""
        print("🔧 Training classification models...\n")

        # Define models
        model_configs = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }

        for idx, (name, model) in enumerate(model_configs.items(), 1):
            print(f"  {idx}/{len(model_configs)} {name}...")

            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            self.models[name] = model
            self.results[name] = {
                'model': model,
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test)
            }

            if y_proba_test is not None and len(np.unique(y_test)) == 2:
                self.results[name]['roc_auc'] = roc_auc_score(y_test, y_proba_test)

        print("\n✅ All models trained!\n")
        return self.results

    def compare_models(self) -> pd.DataFrame:
        """Compare all models"""
        comparison = []

        for name, results in self.results.items():
            row = {
                'Model': name,
                'Train Acc': results['train_accuracy'],
                'Test Acc': results['test_accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1']
            }

            if 'roc_auc' in results:
                row['ROC AUC'] = results['roc_auc']

            comparison.append(row)

        df = pd.DataFrame(comparison)
        df = df.sort_values('Test Acc', ascending=False)

        self.best_model_name = df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]

        return df

    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for idx, (name, results) in enumerate(self.results.items()):
            if idx >= 8:
                break

            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'{name}\nAccuracy: {results["test_accuracy"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        # Hide extra subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrices saved to {save_path}")

        plt.show()

    def save_model(self, model_name: str, filepath: str):
        """Save model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        joblib.dump(self.models[model_name], filepath)
        print(f"💾 Model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Classification Analysis')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--target', type=str, required=True, help='Target column')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--output', type=str, help='Output plot path')
    parser.add_argument('--save-model', type=str, help='Save best model path')

    args = parser.parse_args()

    # Load data
    print(f"📂 Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🎯 Classes: {np.unique(y)}\n")

    # Initialize analyzer
    analyzer = ClassificationAnalyzer(scale_features=True)

    # Prepare data
    X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y, test_size=args.test_size)

    # Train models
    analyzer.train_all_models(X_train, y_train, X_test, y_test)

    # Compare
    print("📊 Model Comparison:")
    print("=" * 100)
    comparison_df = analyzer.compare_models()
    print(comparison_df.to_string(index=False))
    print("=" * 100)

    print(f"\n🏆 Best Model: {analyzer.best_model_name}")
    print(f"   Test Accuracy: {analyzer.results[analyzer.best_model_name]['test_accuracy']:.4f}")

    # Plot
    if args.output:
        analyzer.plot_confusion_matrices(save_path=args.output)

    # Save
    if args.save_model:
        analyzer.save_model(analyzer.best_model_name, args.save_model)


if __name__ == "__main__":
    main()
