"""
Advanced Ensemble Methods System v2.0
Author: BrillConsulting
Description: Production-ready ensemble learning with 10+ methods and advanced strategies
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier, BaggingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
from typing import Dict, List, Tuple, Optional, Union
import joblib
warnings.filterwarnings('ignore')

# Optional dependencies
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


class EnsembleAnalyzer:
    """
    Advanced ensemble learning system

    Features:
    - 10+ ensemble methods
    - Classification and regression support
    - Automatic model selection
    - Hyperparameter tuning
    - Feature importance analysis
    - Out-of-bag error estimation
    - Model diversity metrics
    - Comprehensive evaluation
    """

    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        """
        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model_name = None
        np.random.seed(random_state)

    def bagging_ensemble(self, X_train, y_train, X_test, y_test,
                        n_estimators: int = 100, max_samples: float = 1.0) -> Dict:
        """
        Bagging (Bootstrap Aggregating) ensemble
        """
        print(f"🔧 Bagging Ensemble (n_estimators={n_estimators})")

        if self.task_type == 'classification':
            base_estimator = DecisionTreeClassifier(random_state=self.random_state)
            model = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=self.random_state,
                n_jobs=-1,
                oob_score=True
            )
        else:
            base_estimator = DecisionTreeRegressor(random_state=self.random_state)
            model = BaggingRegressor(
                estimator=base_estimator,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=self.random_state,
                n_jobs=-1,
                oob_score=True
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, 'Bagging')
        metrics['oob_score'] = model.oob_score_

        self.models['Bagging'] = model
        self.results['Bagging'] = metrics

        print(f"  OOB Score: {model.oob_score_:.4f}")

        return metrics

    def random_forest_ensemble(self, X_train, y_train, X_test, y_test,
                               tune_params: bool = False) -> Dict:
        """
        Random Forest ensemble
        """
        print("🔧 Random Forest Ensemble")

        if self.task_type == 'classification':
            if tune_params:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
                grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                print(f"  Best params: {grid_search.best_params_}")
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    random_state=self.random_state,
                    n_jobs=-1,
                    oob_score=True
                )
                model.fit(X_train, y_train)
        else:
            if tune_params:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
                base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
                grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                    oob_score=True
                )
                model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, 'Random Forest')
        if hasattr(model, 'oob_score_'):
            metrics['oob_score'] = model.oob_score_
        metrics['feature_importance'] = model.feature_importances_

        self.models['Random Forest'] = model
        self.results['Random Forest'] = metrics

        return metrics

    def extra_trees_ensemble(self, X_train, y_train, X_test, y_test,
                            n_estimators: int = 100) -> Dict:
        """
        Extra Trees (Extremely Randomized Trees) ensemble
        """
        print(f"🔧 Extra Trees Ensemble (n_estimators={n_estimators})")

        if self.task_type == 'classification':
            model = ExtraTreesClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, 'Extra Trees')
        metrics['feature_importance'] = model.feature_importances_

        self.models['Extra Trees'] = model
        self.results['Extra Trees'] = metrics

        return metrics

    def adaboost_ensemble(self, X_train, y_train, X_test, y_test,
                         n_estimators: int = 100, learning_rate: float = 1.0) -> Dict:
        """
        AdaBoost (Adaptive Boosting) ensemble
        """
        print(f"🔧 AdaBoost Ensemble (n_estimators={n_estimators}, lr={learning_rate})")

        if self.task_type == 'classification':
            model = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=self.random_state
            )
        else:
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=self.random_state
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, 'AdaBoost')
        metrics['feature_importance'] = model.feature_importances_

        self.models['AdaBoost'] = model
        self.results['AdaBoost'] = metrics

        return metrics

    def gradient_boosting_ensemble(self, X_train, y_train, X_test, y_test,
                                   tune_params: bool = False) -> Dict:
        """
        Gradient Boosting ensemble
        """
        print("🔧 Gradient Boosting Ensemble")

        if self.task_type == 'classification':
            if tune_params:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
                base_model = GradientBoostingClassifier(random_state=self.random_state)
                grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                print(f"  Best params: {grid_search.best_params_}")
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state
                )
                model.fit(X_train, y_train)
        else:
            if tune_params:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
                base_model = GradientBoostingRegressor(random_state=self.random_state)
                grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state
                )
                model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, 'Gradient Boosting')
        metrics['feature_importance'] = model.feature_importances_

        self.models['Gradient Boosting'] = model
        self.results['Gradient Boosting'] = metrics

        return metrics

    def xgboost_ensemble(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        XGBoost ensemble (if available)
        """
        if not XGBOOST_AVAILABLE:
            print("⚠️  XGBoost not available. Install: pip install xgboost")
            return {}

        print("🔧 XGBoost Ensemble")

        if self.task_type == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, 'XGBoost')
        metrics['feature_importance'] = model.feature_importances_

        self.models['XGBoost'] = model
        self.results['XGBoost'] = metrics

        return metrics

    def lightgbm_ensemble(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        LightGBM ensemble (if available)
        """
        if not LIGHTGBM_AVAILABLE:
            print("⚠️  LightGBM not available. Install: pip install lightgbm")
            return {}

        print("🔧 LightGBM Ensemble")

        if self.task_type == 'classification':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=-1
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, 'LightGBM')
        metrics['feature_importance'] = model.feature_importances_

        self.models['LightGBM'] = model
        self.results['LightGBM'] = metrics

        return metrics

    def voting_ensemble(self, X_train, y_train, X_test, y_test,
                       voting: str = 'soft') -> Dict:
        """
        Voting ensemble (soft or hard voting)
        """
        print(f"🔧 Voting Ensemble (voting={voting})")

        if self.task_type == 'classification':
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)),
                ('lr', LogisticRegression(max_iter=1000, random_state=self.random_state))
            ]
            model = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
        else:
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=50, random_state=self.random_state)),
                ('gb', GradientBoostingRegressor(n_estimators=50, random_state=self.random_state)),
                ('ridge', Ridge(random_state=self.random_state))
            ]
            model = VotingRegressor(estimators=estimators, n_jobs=-1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, 'Voting')

        self.models['Voting'] = model
        self.results['Voting'] = metrics

        return metrics

    def stacking_ensemble(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        Stacking ensemble with meta-learner
        """
        print("🔧 Stacking Ensemble")

        if self.task_type == 'classification':
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)),
                ('et', ExtraTreesClassifier(n_estimators=50, random_state=self.random_state))
            ]
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
                cv=3,
                n_jobs=-1
            )
        else:
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=50, random_state=self.random_state)),
                ('gb', GradientBoostingRegressor(n_estimators=50, random_state=self.random_state)),
                ('et', ExtraTreesRegressor(n_estimators=50, random_state=self.random_state))
            ]
            model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(random_state=self.random_state),
                cv=3,
                n_jobs=-1
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, 'Stacking')

        self.models['Stacking'] = model
        self.results['Stacking'] = metrics

        return metrics

    def _calculate_metrics(self, y_true, y_pred, model_name: str) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        """
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_true, y_pred)

            # Handle multiclass vs binary
            average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'

            precision = precision_score(y_true, y_pred, average=average, zero_division=0)
            recall = recall_score(y_true, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

            print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
                  f"Recall: {recall:.4f} | F1: {f1:.4f}\n")

            return {
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            print(f"  R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}\n")

            return {
                'model': model_name,
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            }

    def train_all_models(self, X_train, y_train, X_test, y_test,
                        tune_params: bool = False):
        """
        Train all available ensemble models
        """
        print("\n" + "=" * 80)
        print("🚀 Training All Ensemble Models")
        print("=" * 80 + "\n")

        # Bagging methods
        self.bagging_ensemble(X_train, y_train, X_test, y_test)
        self.random_forest_ensemble(X_train, y_train, X_test, y_test, tune_params=tune_params)
        self.extra_trees_ensemble(X_train, y_train, X_test, y_test)

        # Boosting methods
        self.adaboost_ensemble(X_train, y_train, X_test, y_test)
        self.gradient_boosting_ensemble(X_train, y_train, X_test, y_test, tune_params=tune_params)

        if XGBOOST_AVAILABLE:
            self.xgboost_ensemble(X_train, y_train, X_test, y_test)

        if LIGHTGBM_AVAILABLE:
            self.lightgbm_ensemble(X_train, y_train, X_test, y_test)

        # Meta-learners
        self.voting_ensemble(X_train, y_train, X_test, y_test)
        self.stacking_ensemble(X_train, y_train, X_test, y_test)

        # Find best model
        if self.results:
            if self.task_type == 'classification':
                best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
                metric_value = best_model[1]['accuracy']
                metric_name = 'Accuracy'
            else:
                best_model = max(self.results.items(), key=lambda x: x[1]['r2'])
                metric_value = best_model[1]['r2']
                metric_name = 'R²'

            self.best_model_name = best_model[0]

            print("=" * 80)
            print(f"🏆 Best Model: {self.best_model_name} ({metric_name}: {metric_value:.4f})")
            print("=" * 80 + "\n")

    def compare_models(self) -> pd.DataFrame:
        """
        Compare all models side-by-side
        """
        if not self.results:
            print("⚠️  No models trained yet")
            return pd.DataFrame()

        comparison = pd.DataFrame(self.results).T

        if self.task_type == 'classification':
            comparison = comparison.sort_values('accuracy', ascending=False)
            print("\n📊 Model Comparison:")
            print("=" * 80)
            print(comparison[['accuracy', 'precision', 'recall', 'f1_score']].to_string())
        else:
            comparison = comparison.sort_values('r2', ascending=False)
            print("\n📊 Model Comparison:")
            print("=" * 80)
            print(comparison[['r2', 'rmse', 'mae']].to_string())

        print("=" * 80 + "\n")

        return comparison

    def plot_feature_importance(self, top_k: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance for tree-based ensembles
        """
        importance_models = []

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_models.append((name, model.feature_importances_))

        if not importance_models:
            print("⚠️  No models with feature importance found")
            return

        n_models = len(importance_models)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 5 * n_models))

        if n_models == 1:
            axes = [axes]

        for ax, (name, importances) in zip(axes, importance_models):
            # Get top k features
            indices = np.argsort(importances)[-top_k:]
            values = importances[indices]

            ax.barh(range(len(indices)), values)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([f'Feature {i}' for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'Feature Importance - {name}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance plot saved to {save_path}")

        plt.close()

    def plot_model_comparison(self, save_path: Optional[str] = None):
        """
        Plot model performance comparison
        """
        if not self.results:
            print("⚠️  No results to plot")
            return

        comparison = pd.DataFrame(self.results).T

        if self.task_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            comparison = comparison.sort_values('accuracy', ascending=False)
        else:
            metrics = ['r2', 'rmse', 'mae']
            comparison = comparison.sort_values('r2', ascending=False)

        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))

        for ax, metric in zip(axes, metrics):
            values = comparison[metric].values
            names = comparison.index.values

            colors = ['green' if name == self.best_model_name else 'steelblue' for name in names]

            ax.barh(names, values, color=colors)
            ax.set_xlabel(metric.upper().replace('_', ' '))
            ax.set_title(f'{metric.upper().replace("_", " ")}')
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Model comparison plot saved to {save_path}")

        plt.close()

    def save_model(self, model_name: str, path: str):
        """
        Save trained model
        """
        if model_name not in self.models:
            print(f"⚠️  Model '{model_name}' not found")
            return

        joblib.dump(self.models[model_name], path)
        print(f"💾 Model '{model_name}' saved to {path}")

    def load_model(self, path: str, model_name: str):
        """
        Load trained model
        """
        model = joblib.load(path)
        self.models[model_name] = model
        print(f"📂 Model loaded from {path} as '{model_name}'")
        return model


def demo():
    """
    Demonstration with synthetic data
    """
    print("\n" + "=" * 80)
    print("🎭 Advanced Ensemble Methods Demo (Classification)")
    print("=" * 80 + "\n")

    from sklearn.datasets import make_classification

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples\n")

    # Initialize analyzer
    analyzer = EnsembleAnalyzer(task_type='classification', random_state=42)

    # Train all models
    analyzer.train_all_models(X_train, y_train, X_test, y_test, tune_params=False)

    # Compare models
    comparison = analyzer.compare_models()

    # Plot comparison
    analyzer.plot_model_comparison()

    # Plot feature importance
    analyzer.plot_feature_importance(top_k=15)

    print("\n✅ Demo completed successfully!")
    print(f"Best model: {analyzer.best_model_name}")
    print(f"Best accuracy: {analyzer.results[analyzer.best_model_name]['accuracy']:.4f}")


def main():
    """
    Command-line interface
    """
    parser = argparse.ArgumentParser(
        description='Advanced Ensemble Methods System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classification
  python ensemble_models.py --data data.csv --target label --task classification

  # Regression
  python ensemble_models.py --data housing.csv --target price --task regression --tune

  # Run demo
  python ensemble_models.py --demo
        """
    )

    parser.add_argument('--data', type=str, help='CSV data file')
    parser.add_argument('--target', type=str, help='Target column name')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'regression'], help='Task type')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--output', type=str, help='Output comparison plot path')
    parser.add_argument('--save-model', type=str, help='Save best model to path')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')

    args = parser.parse_args()

    if args.demo:
        demo()
        return

    if not args.data or not args.target:
        parser.print_help()
        return

    # Load data
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # Initialize analyzer
    analyzer = EnsembleAnalyzer(task_type=args.task, random_state=42)

    # Train models
    analyzer.train_all_models(X_train, y_train, X_test, y_test, tune_params=args.tune)

    # Compare
    analyzer.compare_models()

    # Plot
    analyzer.plot_model_comparison(save_path=args.output)
    analyzer.plot_feature_importance(top_k=20)

    # Save best model
    if args.save_model:
        analyzer.save_model(analyzer.best_model_name, args.save_model)

    print(f"\n🏆 Best Model: {analyzer.best_model_name}")
    if args.task == 'classification':
        print(f"  Accuracy: {analyzer.results[analyzer.best_model_name]['accuracy']:.4f}")
        print(f"  F1 Score: {analyzer.results[analyzer.best_model_name]['f1_score']:.4f}")
    else:
        print(f"  R²: {analyzer.results[analyzer.best_model_name]['r2']:.4f}")
        print(f"  RMSE: {analyzer.results[analyzer.best_model_name]['rmse']:.4f}")


if __name__ == "__main__":
    main()
