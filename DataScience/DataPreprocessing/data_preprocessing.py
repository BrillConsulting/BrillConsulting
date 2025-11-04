"""
Data Preprocessing Toolkit
===========================

Comprehensive data cleaning and preprocessing:
- Missing value imputation
- Outlier handling
- Data validation and quality checks
- Type conversion and formatting
- Duplicate removal
- Text cleaning and normalization

Author: Brill Consulting
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import re
from typing import List, Dict, Optional


class DataPreprocessor:
    """Data cleaning and preprocessing toolkit."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with dataset."""
        self.data = data.copy()
        self.original_shape = data.shape

    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values with various strategies."""
        result = self.data.copy()

        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns

        if strategy in ['mean', 'median', 'most_frequent', 'constant']:
            imputer = SimpleImputer(strategy=strategy)
            result[columns] = imputer.fit_transform(result[columns])

        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            result[columns] = imputer.fit_transform(result[columns])

        elif strategy == 'drop':
            result = result.dropna(subset=columns)

        elif strategy == 'forward_fill':
            result[columns] = result[columns].fillna(method='ffill')

        elif strategy == 'backward_fill':
            result[columns] = result[columns].fillna(method='bfill')

        return result

    def handle_outliers(self, columns: List[str], method: str = 'iqr', action: str = 'cap') -> pd.DataFrame:
        """Handle outliers using IQR or Z-score method."""
        result = self.data.copy()

        for col in columns:
            if method == 'iqr':
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

            elif method == 'zscore':
                mean = result[col].mean()
                std = result[col].std()
                lower = mean - 3 * std
                upper = mean + 3 * std

            if action == 'cap':
                result[col] = result[col].clip(lower, upper)
            elif action == 'remove':
                result = result[(result[col] >= lower) & (result[col] <= upper)]

        return result

    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows."""
        result = self.data.copy()
        before = len(result)

        result = result.drop_duplicates(subset=subset, keep=keep)

        after = len(result)
        print(f"Removed {before - after} duplicate rows")

        return result

    def convert_dtypes(self, conversions: Dict[str, str]) -> pd.DataFrame:
        """Convert column data types."""
        result = self.data.copy()

        for col, dtype in conversions.items():
            if dtype == 'datetime':
                result[col] = pd.to_datetime(result[col], errors='coerce')
            elif dtype == 'category':
                result[col] = result[col].astype('category')
            else:
                result[col] = result[col].astype(dtype)

        return result

    def clean_text(self, columns: List[str], lowercase: bool = True,
                  remove_special: bool = True, remove_numbers: bool = False) -> pd.DataFrame:
        """Clean text columns."""
        result = self.data.copy()

        for col in columns:
            if lowercase:
                result[col] = result[col].str.lower()

            if remove_special:
                result[col] = result[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

            if remove_numbers:
                result[col] = result[col].str.replace(r'\d+', '', regex=True)

            # Remove extra whitespace
            result[col] = result[col].str.strip().str.replace(r'\s+', ' ', regex=True)

        return result

    def standardize_names(self, columns: List[str]) -> pd.DataFrame:
        """Standardize column names (lowercase, underscores)."""
        result = self.data.copy()

        new_names = {}
        for col in columns:
            new_name = col.lower().replace(' ', '_').replace('-', '_')
            new_name = re.sub(r'[^a-z0-9_]', '', new_name)
            new_names[col] = new_name

        result = result.rename(columns=new_names)

        return result

    def validate_ranges(self, rules: Dict[str, Dict]) -> Dict:
        """Validate data against range rules."""
        violations = {}

        for col, rule in rules.items():
            if col not in self.data.columns:
                continue

            col_violations = []

            if 'min' in rule:
                invalid = self.data[self.data[col] < rule['min']]
                if len(invalid) > 0:
                    col_violations.append(f"{len(invalid)} values below min ({rule['min']})")

            if 'max' in rule:
                invalid = self.data[self.data[col] > rule['max']]
                if len(invalid) > 0:
                    col_violations.append(f"{len(invalid)} values above max ({rule['max']})")

            if 'allowed_values' in rule:
                invalid = self.data[~self.data[col].isin(rule['allowed_values'])]
                if len(invalid) > 0:
                    col_violations.append(f"{len(invalid)} invalid values")

            if col_violations:
                violations[col] = col_violations

        return violations

    def generate_quality_report(self) -> Dict:
        """Generate data quality report."""
        report = {
            'shape': self.data.shape,
            'columns': len(self.data.columns),
            'rows': len(self.data),
            'memory_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'duplicates': self.data.duplicated().sum(),
            'missing_values': self.data.isnull().sum().sum(),
            'missing_percentage': self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1]) * 100,
            'numeric_columns': len(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.data.select_dtypes(include=['object', 'category']).columns),
        }

        # Per-column analysis
        report['column_details'] = {}
        for col in self.data.columns:
            report['column_details'][col] = {
                'dtype': str(self.data[col].dtype),
                'missing': self.data[col].isnull().sum(),
                'missing_pct': self.data[col].isnull().sum() / len(self.data) * 100,
                'unique': self.data[col].nunique(),
                'unique_pct': self.data[col].nunique() / len(self.data) * 100
            }

        return report


def demo():
    """Demo preprocessing."""
    np.random.seed(42)

    # Sample data with issues
    n = 1000
    data = pd.DataFrame({
        'Age': np.random.randint(18, 80, n),
        'Income': np.random.exponential(50000, n),
        'Credit Score': np.random.normal(700, 100, n),
        'Gender': np.random.choice(['Male', 'Female', 'M', 'F', None], n),
        'Email': [f'user{i}@example.com' if i % 10 != 0 else None for i in range(n)],
        'Description': [f'User {i} description' if i % 15 != 0 else None for i in range(n)]
    })

    # Add some issues
    data.loc[np.random.choice(data.index, 50), 'Income'] = np.nan
    data.loc[np.random.choice(data.index, 20), 'Income'] = np.random.uniform(500000, 1000000, 20)  # Outliers
    data = pd.concat([data, data.head(10)])  # Add duplicates

    print("Data Preprocessing Demo")
    print("="*50)

    preprocessor = DataPreprocessor(data)

    # Quality report
    print("\n1. Initial Quality Report:")
    report = preprocessor.generate_quality_report()
    print(f"Shape: {report['shape']}")
    print(f"Duplicates: {report['duplicates']}")
    print(f"Missing values: {report['missing_values']} ({report['missing_percentage']:.2f}%)")

    # Remove duplicates
    print("\n2. Removing duplicates...")
    data_clean = preprocessor.remove_duplicates()

    # Handle missing values
    print("\n3. Handling missing values...")
    preprocessor_clean = DataPreprocessor(data_clean)
    data_clean = preprocessor_clean.handle_missing_values(strategy='mean', columns=['Income'])
    data_clean = preprocessor_clean.handle_missing_values(strategy='most_frequent', columns=['Gender'])
    print(f"Missing values after imputation: {data_clean.isnull().sum().sum()}")

    # Handle outliers
    print("\n4. Handling outliers...")
    preprocessor_clean = DataPreprocessor(data_clean)
    data_clean = preprocessor_clean.handle_outliers(['Income'], method='iqr', action='cap')

    # Standardize names
    print("\n5. Standardizing column names...")
    preprocessor_clean = DataPreprocessor(data_clean)
    data_clean = preprocessor_clean.standardize_names(data_clean.columns.tolist())
    print(f"New columns: {data_clean.columns.tolist()}")

    # Validate ranges
    print("\n6. Validating data ranges...")
    preprocessor_clean = DataPreprocessor(data_clean)
    rules = {
        'age': {'min': 18, 'max': 100},
        'credit_score': {'min': 300, 'max': 850},
        'gender': {'allowed_values': ['Male', 'Female', 'M', 'F']}
    }
    violations = preprocessor_clean.validate_ranges(rules)
    if violations:
        print("Violations found:")
        for col, issues in violations.items():
            print(f"  {col}: {issues}")
    else:
        print("No violations found")

    # Final report
    print("\n7. Final Quality Report:")
    preprocessor_final = DataPreprocessor(data_clean)
    final_report = preprocessor_final.generate_quality_report()
    print(f"Shape: {final_report['shape']}")
    print(f"Duplicates: {final_report['duplicates']}")
    print(f"Missing values: {final_report['missing_values']}")

    print("\n✓ Preprocessing Complete!")


if __name__ == '__main__':
    demo()
