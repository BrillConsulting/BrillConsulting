"""
Feature Engineering Toolkit
============================

Comprehensive feature engineering and transformation toolkit:
- Numerical transformations (scaling, binning, polynomial)
- Categorical encoding (one-hot, label, target encoding)
- Feature creation (interactions, aggregations)
- Date/time feature extraction
- Text feature extraction
- Automated feature selection

Author: Brill Consulting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering and transformation toolkit."""

    def __init__(self):
        """Initialize feature engineer."""
        self.scalers = {}
        self.encoders = {}

    def scale_features(self, data: pd.DataFrame, columns: List[str],
                      method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features."""
        result = data.copy()

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        result[columns] = scaler.fit_transform(data[columns])
        self.scalers[method] = scaler

        return result

    def create_polynomial_features(self, data: pd.DataFrame, columns: List[str],
                                  degree: int = 2) -> pd.DataFrame:
        """Create polynomial features."""
        result = data.copy()

        for col in columns:
            for d in range(2, degree + 1):
                result[f'{col}_pow{d}'] = data[col] ** d

        return result

    def create_interaction_features(self, data: pd.DataFrame,
                                   column_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between column pairs."""
        result = data.copy()

        for col1, col2 in column_pairs:
            result[f'{col1}_x_{col2}'] = data[col1] * data[col2]
            result[f'{col1}_div_{col2}'] = data[col1] / (data[col2] + 1e-8)

        return result

    def bin_numerical_features(self, data: pd.DataFrame, column: str,
                              bins: int = 5, labels: Optional[List] = None) -> pd.DataFrame:
        """Bin numerical features into categories."""
        result = data.copy()

        if labels is None:
            labels = [f'{column}_bin_{i}' for i in range(bins)]

        result[f'{column}_binned'] = pd.cut(data[column], bins=bins, labels=labels)

        return result

    def encode_categorical(self, data: pd.DataFrame, columns: List[str],
                          method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical features."""
        result = data.copy()

        if method == 'onehot':
            result = pd.get_dummies(result, columns=columns, prefix=columns, drop_first=True)

        elif method == 'label':
            for col in columns:
                le = LabelEncoder()
                result[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le

        elif method == 'frequency':
            for col in columns:
                freq = data[col].value_counts(normalize=True)
                result[f'{col}_freq'] = data[col].map(freq)

        return result

    def create_date_features(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Extract features from date column."""
        result = data.copy()
        date_series = pd.to_datetime(result[date_column])

        result[f'{date_column}_year'] = date_series.dt.year
        result[f'{date_column}_month'] = date_series.dt.month
        result[f'{date_column}_day'] = date_series.dt.day
        result[f'{date_column}_dayofweek'] = date_series.dt.dayofweek
        result[f'{date_column}_quarter'] = date_series.dt.quarter
        result[f'{date_column}_dayofyear'] = date_series.dt.dayofyear
        result[f'{date_column}_is_weekend'] = date_series.dt.dayofweek.isin([5, 6]).astype(int)
        result[f'{date_column}_is_month_start'] = date_series.dt.is_month_start.astype(int)
        result[f'{date_column}_is_month_end'] = date_series.dt.is_month_end.astype(int)

        return result

    def create_aggregation_features(self, data: pd.DataFrame, group_col: str,
                                   agg_col: str, agg_funcs: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """Create aggregation features grouped by a column."""
        result = data.copy()

        for func in agg_funcs:
            agg_values = data.groupby(group_col)[agg_col].transform(func)
            result[f'{agg_col}_{func}_by_{group_col}'] = agg_values

        return result

    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series,
                                   k: int = 10, score_func = f_classif) -> Tuple[pd.DataFrame, List[str]]:
        """Select top k features using univariate statistical tests."""
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()].tolist()
        X_result = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

        return X_result, selected_features

    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series,
                           n_features: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using Recursive Feature Elimination."""
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features)

        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.support_].tolist()
        X_result = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

        return X_result, selected_features

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Get feature importance using Random Forest."""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df


def demo():
    """Demo feature engineering."""
    np.random.seed(42)

    # Sample data
    n = 1000
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'income': np.random.exponential(50000, n),
        'credit_score': np.random.normal(700, 100, n),
        'debt_ratio': np.random.uniform(0, 1, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n),
        'date': pd.date_range('2023-01-01', periods=n, freq='D')
    })
    data['target'] = (data['income'] > 50000).astype(int)

    fe = FeatureEngineer()

    print("Feature Engineering Demo")
    print("="*50)

    # 1. Scaling
    print("\n1. Scaling numerical features...")
    scaled = fe.scale_features(data, ['age', 'income', 'credit_score'], method='standard')
    print(f"Scaled features: {scaled[['age', 'income', 'credit_score']].describe()}")

    # 2. Polynomial features
    print("\n2. Creating polynomial features...")
    poly = fe.create_polynomial_features(data, ['age', 'debt_ratio'], degree=2)
    print(f"New columns: {[c for c in poly.columns if '_pow' in c]}")

    # 3. Interaction features
    print("\n3. Creating interaction features...")
    interact = fe.create_interaction_features(data, [('age', 'income'), ('credit_score', 'debt_ratio')])
    print(f"New columns: {[c for c in interact.columns if '_x_' in c or '_div_' in c]}")

    # 4. Binning
    print("\n4. Binning numerical features...")
    binned = fe.bin_numerical_features(data, 'age', bins=5)
    print(f"Age bins distribution:\n{binned['age_binned'].value_counts()}")

    # 5. Categorical encoding
    print("\n5. Encoding categorical features...")
    encoded = fe.encode_categorical(data, ['gender', 'education'], method='onehot')
    print(f"Shape after encoding: {encoded.shape}")

    # 6. Date features
    print("\n6. Extracting date features...")
    date_features = fe.create_date_features(data, 'date')
    date_cols = [c for c in date_features.columns if 'date_' in c]
    print(f"Date features created: {date_cols}")

    # 7. Aggregation features
    print("\n7. Creating aggregation features...")
    agg = fe.create_aggregation_features(data, 'education', 'income')
    agg_cols = [c for c in agg.columns if '_by_' in c]
    print(f"Aggregation features: {agg_cols}")

    # 8. Feature selection
    print("\n8. Feature selection...")
    X = data[['age', 'income', 'credit_score', 'debt_ratio']]
    y = data['target']

    X_selected, selected = fe.select_features_univariate(X, y, k=3)
    print(f"Selected features (univariate): {selected}")

    importance = fe.get_feature_importance(X, y)
    print(f"\nFeature importance:\n{importance}")

    print("\n✓ Feature Engineering Complete!")


if __name__ == '__main__':
    demo()
