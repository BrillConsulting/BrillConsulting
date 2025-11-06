"""
Advanced Multi-Algorithm Classification System v2.0
Author: BrillConsulting
Description: Production-ready classification with 12+ algorithms including XGBoost, LightGBM, and ensemble methods
Version: 2.0 - Enhanced with gradient boosting, ensemble methods, and hyperparameter tuning
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier,
                               VotingClassifier, StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ClassificationAnalyzer:
    """
    Advanced classification system with 12+ algorithms

    Features:
    - Traditional ML: Logistic Regression, SVM, KNN, Naive Bayes
    - Tree-based: Decision Tree, Random Forest, Extra Trees
    - Boosting: Gradient Boosting, AdaBoost, XGBoost, LightGBM
    - Ensemble: Voting, Stacking
    - Hyperparameter tuning with GridSearchCV
    - Feature importance analysis
    - Comprehensive evaluation metrics
    """

    def __init__(self, scale_features: bool = True, random_state: int = 42):
        """Initialize classification analyzer"""
        self.scale_features = scale_features
        self.random_state = random_state
        self.scaler = StandardScaler() if scale_features else None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, X, y, test_size=0.2):
        """Prepare and split data with stratification"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        if self.scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_logistic_regression(self, X_train, y_train, X_test, y_test,
                                  tune_hyperparams: bool = False) -> Dict:
        """Train Logistic Regression with optional hyperparameter tuning"""
        if tune_hyperparams:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            grid = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            best_model.fit(X_train, y_train)

        results = self._evaluate_model(best_model, X_train, y_train, X_test, y_test)
        self.models['Logistic Regression'] = best_model
        self.results['Logistic Regression'] = results
        return results

    def train_random_forest(self, X_train, y_train, X_test, y_test,
                           tune_hyperparams: bool = True) -> Dict:
        """Train Random Forest with hyperparameter tuning"""
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=self.random_state)
            grid = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            best_model.fit(X_train, y_train)

        results = self._evaluate_model(best_model, X_train, y_train, X_test, y_test)
        results['feature_importance'] = best_model.feature_importances_
        self.models['Random Forest'] = best_model
        self.results['Random Forest'] = results
        return results

    def train_xgboost(self, X_train, y_train, X_test, y_test,
                     tune_hyperparams: bool = True) -> Dict:
        """Train XGBoost Classifier"""
        if not XGBOOST_AVAILABLE:
            print("⚠️  XGBoost not available. Skipping...")
            return {}

        if tune_hyperparams:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
            model = xgb.XGBClassifier(random_state=self.random_state, tree_method='hist', eval_metric='logloss')
            grid = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = xgb.XGBClassifier(random_state=self.random_state, tree_method='hist', eval_metric='logloss')
            best_model.fit(X_train, y_train)

        results = self._evaluate_model(best_model, X_train, y_train, X_test, y_test)
        results['feature_importance'] = best_model.feature_importances_
        self.models['XGBoost'] = best_model
        self.results['XGBoost'] = results
        return results

    def train_lightgbm(self, X_train, y_train, X_test, y_test,
                      tune_hyperparams: bool = True) -> Dict:
        """Train LightGBM Classifier"""
        if not LIGHTGBM_AVAILABLE:
            print("⚠️  LightGBM not available. Skipping...")
            return {}

        if tune_hyperparams:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70],
                'max_depth': [-1, 10, 20]
            }
            model = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
            grid = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
            best_model.fit(X_train, y_train)

        results = self._evaluate_model(best_model, X_train, y_train, X_test, y_test)
        results['feature_importance'] = best_model.feature_importances_
        self.models['LightGBM'] = best_model
        self.results['LightGBM'] = results
        return results

    def train_voting_ensemble(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Voting Classifier (ensemble)"""
        base_models = [
            ('lr', LogisticRegression(max_iter=1000, random_state=self.random_state)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=self.random_state)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state))
        ]

        voting = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
        voting.fit(X_train, y_train)

        results = self._evaluate_model(voting, X_train, y_train, X_test, y_test)
        self.models['Voting Ensemble'] = voting
        self.results['Voting Ensemble'] = results
        return results

    def train_stacking_ensemble(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Stacking Classifier"""
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=self.random_state)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)),
            ('et', ExtraTreesClassifier(n_estimators=50, random_state=self.random_state))
        ]

        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=3,
            n_jobs=-1
        )
        stacking.fit(X_train, y_train)

        results = self._evaluate_model(stacking, X_train, y_train, X_test, y_test)
        self.models['Stacking Ensemble'] = stacking
        self.results['Stacking Ensemble'] = results
        return results

    def _evaluate_model(self, model, X_train, y_train, X_test, y_test) -> Dict:
        """Evaluate a trained model"""
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Get probabilities for ROC-AUC (binary classification only)
        y_proba_test = None
        if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
            y_proba_test = model.predict_proba(X_test)[:, 1]

        results = {
            'model': model,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test)
        }

        if y_proba_test is not None:
            results['roc_auc'] = roc_auc_score(y_test, y_proba_test)

        return results

    def train_all_models(self, X_train, y_train, X_test, y_test,
                        tune_hyperparams: bool = False):
        """Train all available classification models"""
        print(f"🔧 Training all classification models...")
        print(f"   Hyperparameter tuning: {'Enabled' if tune_hyperparams else 'Disabled'}")
        print("=" * 60)

        # Traditional ML
        print("  [1/12] Logistic Regression...")
        self.train_logistic_regression(X_train, y_train, X_test, y_test, tune_hyperparams)

        print("  [2/12] SVM...")
        svm = SVC(probability=True, random_state=self.random_state)
        svm.fit(X_train, y_train)
        self.models['SVM'] = svm
        self.results['SVM'] = self._evaluate_model(svm, X_train, y_train, X_test, y_test)

        print("  [3/12] K-Nearest Neighbors...")
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        self.models['KNN'] = knn
        self.results['KNN'] = self._evaluate_model(knn, X_train, y_train, X_test, y_test)

        print("  [4/12] Naive Bayes...")
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        self.models['Naive Bayes'] = nb
        self.results['Naive Bayes'] = self._evaluate_model(nb, X_train, y_train, X_test, y_test)

        # Tree-based
        print("  [5/12] Decision Tree...")
        dt = DecisionTreeClassifier(random_state=self.random_state)
        dt.fit(X_train, y_train)
        self.models['Decision Tree'] = dt
        self.results['Decision Tree'] = self._evaluate_model(dt, X_train, y_train, X_test, y_test)

        print("  [6/12] Random Forest...")
        self.train_random_forest(X_train, y_train, X_test, y_test, tune_hyperparams)

        print("  [7/12] Extra Trees...")
        et = ExtraTreesClassifier(n_estimators=100, random_state=self.random_state)
        et.fit(X_train, y_train)
        self.models['Extra Trees'] = et
        self.results['Extra Trees'] = self._evaluate_model(et, X_train, y_train, X_test, y_test)
        self.results['Extra Trees']['feature_importance'] = et.feature_importances_

        # Boosting
        print("  [8/12] Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb
        self.results['Gradient Boosting'] = self._evaluate_model(gb, X_train, y_train, X_test, y_test)
        self.results['Gradient Boosting']['feature_importance'] = gb.feature_importances_

        print("  [9/12] AdaBoost...")
        ada = AdaBoostClassifier(n_estimators=100, random_state=self.random_state, algorithm='SAMME')
        ada.fit(X_train, y_train)
        self.models['AdaBoost'] = ada
        self.results['AdaBoost'] = self._evaluate_model(ada, X_train, y_train, X_test, y_test)
        self.results['AdaBoost']['feature_importance'] = ada.feature_importances_

        # Optional: XGBoost and LightGBM
        if XGBOOST_AVAILABLE:
            print("  [10/12] XGBoost...")
            self.train_xgboost(X_train, y_train, X_test, y_test, tune_hyperparams)

        if LIGHTGBM_AVAILABLE:
            print("  [11/12] LightGBM...")
            self.train_lightgbm(X_train, y_train, X_test, y_test, tune_hyperparams)

        # Ensembles
        print("  [12/12] Ensemble Methods...")
        self.train_voting_ensemble(X_train, y_train, X_test, y_test)
        self.train_stacking_ensemble(X_train, y_train, X_test, y_test)

        print("=" * 60)
        print(f"✅ All {len(self.models)} models trained successfully!\n")
        return self.results

    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
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
        df = df.sort_values('F1 Score', ascending=False)

        self.best_model_name = df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]

        return df

    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for top 8 models"""
        top_models = sorted(self.results.items(),
                           key=lambda x: x[1]['f1'], reverse=True)[:8]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for idx, (name, results) in enumerate(top_models):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'{name}\nF1: {results["f1"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrices saved to {save_path}")

        plt.show()

    def plot_roc_curves(self, X_test, y_test, save_path=None):
        """Plot ROC curves for binary classification"""
        if len(np.unique(y_test)) != 2:
            print("⚠️  ROC curves only available for binary classification")
            return

        plt.figure(figsize=(12, 8))

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 ROC curves saved to {save_path}")

        plt.show()

    def plot_feature_importance(self, top_k: int = 20, save_path=None):
        """Plot feature importance for tree-based models"""
        importance_models = {name: res for name, res in self.results.items()
                            if 'feature_importance' in res}

        if not importance_models:
            print("⚠️  No models with feature importance available")
            return

        fig, axes = plt.subplots(len(importance_models), 1,
                                figsize=(12, 4 * len(importance_models)))
        if len(importance_models) == 1:
            axes = [axes]

        for idx, (name, results) in enumerate(importance_models.items()):
            importance = results['feature_importance']
            indices = np.argsort(importance)[::-1][:top_k]

            axes[idx].barh(range(len(indices)), importance[indices])
            axes[idx].set_yticks(range(len(indices)))
            axes[idx].set_yticklabels([f'Feature {i}' for i in indices])
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{name} - Top {top_k} Features')
            axes[idx].invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance saved to {save_path}")

        plt.show()

    def save_model(self, model_name: str, filepath: str):
        """Save a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        joblib.dump(self.models[model_name], filepath)
        print(f"💾 Model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Classification Analysis v2.0')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--target', type=str, required=True, help='Target column')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--output-cm', type=str, help='Output confusion matrices path')
    parser.add_argument('--output-roc', type=str, help='Output ROC curves path')
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
    analyzer.train_all_models(X_train, y_train, X_test, y_test, tune_hyperparams=args.tune)

    # Compare
    print("📊 Model Comparison:")
    print("=" * 120)
    comparison_df = analyzer.compare_models()
    print(comparison_df.to_string(index=False))
    print("=" * 120)

    print(f"\n🏆 Best Model: {analyzer.best_model_name}")
    print(f"   Test F1 Score: {analyzer.results[analyzer.best_model_name]['f1']:.4f}")
    print(f"   Test Accuracy: {analyzer.results[analyzer.best_model_name]['test_accuracy']:.4f}")

    # Plots
    if args.output_cm:
        analyzer.plot_confusion_matrices(save_path=args.output_cm)

    if args.output_roc:
        analyzer.plot_roc_curves(X_test, y_test, save_path=args.output_roc)

    # Save best model
    if args.save_model:
        analyzer.save_model(analyzer.best_model_name, args.save_model)


if __name__ == "__main__":
    main()
