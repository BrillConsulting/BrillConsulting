"""
LightGBM Gradient Boosting Framework
Author: BrillConsulting
Description: Fast gradient boosting with LightGBM - advanced features and optimization
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class LightGBMManager:
    """Advanced LightGBM model management"""

    def __init__(self):
        """Initialize LightGBM manager"""
        self.models = []
        self.experiments = []
        self.best_params = {}

    def train_classifier(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train LightGBM classifier with advanced settings

        Args:
            config: Training configuration

        Returns:
            Training results
        """
        print(f"\n{'='*60}")
        print("LightGBM Classifier Training")
        print(f"{'='*60}")

        params = {
            'objective': config.get('objective', 'multiclass'),
            'num_class': config.get('num_class', 10),
            'metric': config.get('metric', 'multi_logloss'),
            'boosting_type': config.get('boosting_type', 'gbdt'),
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.05),
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'max_depth': config.get('max_depth', -1),
            'min_data_in_leaf': config.get('min_data_in_leaf', 20)
        }

        # Simulate training
        num_rounds = config.get('num_boost_round', 100)
        history = []

        for i in range(0, num_rounds, 10):
            train_metric = 1.2 - 0.6 * (i / num_rounds) + np.random.uniform(-0.02, 0.02)
            val_metric = 1.3 - 0.55 * (i / num_rounds) + np.random.uniform(-0.02, 0.02)

            history.append({
                'iteration': i,
                'train_logloss': train_metric,
                'val_logloss': val_metric
            })

            if i % 30 == 0:
                print(f"[{i}] train-logloss: {train_metric:.5f}, val-logloss: {val_metric:.5f}")

        result = {
            'params': params,
            'num_boost_round': num_rounds,
            'final_train_metric': history[-1]['train_logloss'],
            'final_val_metric': history[-1]['val_logloss'],
            'history': history,
            'timestamp': datetime.now().isoformat()
        }

        self.models.append(result)

        print(f"\n✓ Training completed!")
        print(f"  Final validation metric: {result['final_val_metric']:.5f}")
        print(f"{'='*60}")

        return result

    def train_regressor(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train LightGBM regressor

        Args:
            config: Training configuration

        Returns:
            Training results
        """
        print(f"\n{'='*60}")
        print("LightGBM Regressor Training")
        print(f"{'='*60}")

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.05),
            'feature_fraction': config.get('feature_fraction', 0.9)
        }

        # Simulate training
        num_rounds = config.get('num_boost_round', 100)
        history = []

        for i in range(0, num_rounds, 10):
            train_rmse = 12.0 - 7.0 * (i / num_rounds) + np.random.uniform(-0.2, 0.2)
            val_rmse = 13.0 - 6.5 * (i / num_rounds) + np.random.uniform(-0.2, 0.2)

            history.append({
                'iteration': i,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse
            })

            if i % 30 == 0:
                print(f"[{i}] train-rmse: {train_rmse:.4f}, val-rmse: {val_rmse:.4f}")

        result = {
            'params': params,
            'num_boost_round': num_rounds,
            'final_train_rmse': history[-1]['train_rmse'],
            'final_val_rmse': history[-1]['val_rmse'],
            'history': history
        }

        print(f"\n✓ Training completed!")
        print(f"  Final RMSE: {result['final_val_rmse']:.4f}")
        print(f"{'='*60}")

        return result

    def train_with_categorical(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train with categorical features (LightGBM specialty)

        Args:
            config: Training configuration

        Returns:
            Training results
        """
        print(f"\n{'='*60}")
        print("LightGBM with Categorical Features")
        print(f"{'='*60}")

        categorical_features = config.get('categorical_features', ['cat_1', 'cat_2', 'cat_3'])

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'categorical_feature': categorical_features
        }

        print(f"Categorical features: {categorical_features}")

        # Simulate training
        num_rounds = 100
        history = []

        for i in range(0, num_rounds, 20):
            train_loss = 0.7 - 0.4 * (i / num_rounds) + np.random.uniform(-0.01, 0.01)
            val_loss = 0.75 - 0.35 * (i / num_rounds) + np.random.uniform(-0.01, 0.01)

            history.append({
                'iteration': i,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            print(f"[{i}] train-loss: {train_loss:.5f}, val-loss: {val_loss:.5f}")

        result = {
            'params': params,
            'categorical_features': categorical_features,
            'final_train_loss': history[-1]['train_loss'],
            'final_val_loss': history[-1]['val_loss'],
            'history': history
        }

        print(f"\n✓ Training with categorical features completed!")
        print(f"  Final loss: {result['final_val_loss']:.5f}")
        print(f"{'='*60}")

        return result

    def hyperparameter_tuning(self, search_space: Dict[str, List]) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning

        Args:
            search_space: Parameter search space

        Returns:
            Best parameters and results
        """
        print(f"\n{'='*60}")
        print("Hyperparameter Tuning")
        print(f"{'='*60}")

        num_leaves_options = search_space.get('num_leaves', [15, 31, 63])
        learning_rates = search_space.get('learning_rate', [0.01, 0.05, 0.1])
        feature_fractions = search_space.get('feature_fraction', [0.7, 0.8, 0.9])

        best_score = float('inf')
        best_params = {}
        results = []

        total = len(num_leaves_options) * len(learning_rates) * len(feature_fractions)
        current = 0

        for num_leaves in num_leaves_options:
            for lr in learning_rates:
                for ff in feature_fractions:
                    current += 1

                    # Simulate evaluation
                    score = (
                        0.6 +
                        0.05 * (num_leaves / 63) +
                        0.15 * lr +
                        0.1 * ff +
                        np.random.uniform(-0.03, 0.03)
                    )

                    params = {
                        'num_leaves': num_leaves,
                        'learning_rate': lr,
                        'feature_fraction': ff
                    }

                    results.append({
                        'params': params,
                        'score': score
                    })

                    if score < best_score:
                        best_score = score
                        best_params = params
                        print(f"  [{current}/{total}] New best: {score:.5f} - {params}")

        self.best_params = best_params

        tuning_result = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'num_combinations': total
        }

        print(f"\n✓ Tuning completed!")
        print(f"  Best score: {best_score:.5f}")
        print(f"  Best params: {best_params}")
        print(f"{'='*60}")

        return tuning_result

    def feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate feature importance

        Args:
            feature_names: List of feature names

        Returns:
            Feature importance scores
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(25)]

        # Simulate feature importance (split-based)
        split_importance = np.random.dirichlet(np.ones(len(feature_names)) * 2)

        # Simulate gain-based importance
        gain_importance = np.random.dirichlet(np.ones(len(feature_names)) * 2)

        sorted_indices = np.argsort(gain_importance)[::-1]

        importance_dict = {
            'feature_names': [feature_names[i] for i in sorted_indices],
            'split_importance': [float(split_importance[i]) for i in sorted_indices],
            'gain_importance': [float(gain_importance[i]) for i in sorted_indices],
            'top_10': [
                {
                    'feature': feature_names[i],
                    'split': float(split_importance[i]),
                    'gain': float(gain_importance[i])
                }
                for i in sorted_indices[:10]
            ]
        }

        print("\n✓ Feature importance calculated")
        print("\nTop 10 Features (by gain):")
        for i, item in enumerate(importance_dict['top_10'], 1):
            print(f"  {i}. {item['feature']}: gain={item['gain']:.4f}, split={item['split']:.4f}")

        return importance_dict

    def get_training_code(self) -> str:
        """Generate LightGBM training code"""

        code = """
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'multiclass',
    'num_class': 10,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'device': 'gpu',  # GPU acceleration
    'verbose': -1
}

# Train model
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=10),
        lgb.log_evaluation(period=10)
    ]
)

# Predictions
predictions = model.predict(X_test)

# Feature importance
importance = model.feature_importance(importance_type='gain')
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(importance_df.head(10))

# Save model
model.save_model('lightgbm_model.txt')

# Cross-validation
cv_results = lgb.cv(
    params,
    train_data,
    num_boost_round=100,
    nfold=5,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)
"""

        return code

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'models_trained': len(self.models),
            'experiments': len(self.experiments),
            'best_params': self.best_params,
            'framework': 'LightGBM',
            'features': ['categorical_support', 'gpu_acceleration', 'fast_training'],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate LightGBM"""
    print("=" * 60)
    print("LightGBM Gradient Boosting Framework Demo")
    print("=" * 60)

    mgr = LightGBMManager()

    # Train classifier
    print("\n1. Training classifier...")
    classifier_result = mgr.train_classifier({
        'num_class': 10,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'num_boost_round': 100
    })

    # Train regressor
    print("\n2. Training regressor...")
    regressor_result = mgr.train_regressor({
        'num_leaves': 31,
        'learning_rate': 0.05,
        'num_boost_round': 100
    })

    # Train with categorical features
    print("\n3. Training with categorical features...")
    categorical_result = mgr.train_with_categorical({
        'categorical_features': ['category', 'region', 'type']
    })

    # Hyperparameter tuning
    print("\n4. Hyperparameter tuning...")
    tuning_result = mgr.hyperparameter_tuning({
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.7, 0.8, 0.9]
    })

    # Feature importance
    print("\n5. Feature importance...")
    importance = mgr.feature_importance()

    # Show training code
    print("\n6. LightGBM implementation code:")
    code = mgr.get_training_code()
    print(code[:400] + "...\n")

    # Manager info
    print("\n7. Manager summary:")
    info = mgr.get_manager_info()
    print(f"  Models trained: {info['models_trained']}")
    print(f"  Framework: {info['framework']}")
    print(f"  Features: {', '.join(info['features'])}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
