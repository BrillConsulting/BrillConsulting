"""
Exploratory Data Analysis (EDA) Toolkit
========================================

Comprehensive EDA toolkit for data analysis and insights discovery:
- Automated data profiling and summary statistics
- Distribution analysis and outlier detection
- Correlation analysis and feature relationships
- Missing data analysis and visualization
- Automated reporting and insights generation

Author: Brill Consulting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EDAToolkit:
    """Comprehensive exploratory data analysis toolkit."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with dataset."""
        self.data = data.copy()
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    def generate_profile(self) -> Dict:
        """Generate comprehensive data profile."""
        profile = {
            'shape': self.data.shape,
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': self.data.dtypes.value_counts().to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'numeric_columns': len(self.numeric_cols),
            'categorical_columns': len(self.categorical_cols)
        }
        return profile

    def analyze_distributions(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Analyze distributions of numerical features."""
        n_cols = len(self.numeric_cols)
        if n_cols == 0:
            print("No numerical columns found")
            return None

        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(self.numeric_cols):
            if idx < len(axes):
                data = self.data[col].dropna()
                axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{col}\nSkew: {data.skew():.2f}, Kurt: {data.kurtosis():.2f}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].axvline(data.mean(), color='red', linestyle='--', label='Mean')
                axes[idx].axvline(data.median(), color='green', linestyle='--', label='Median')
                axes[idx].legend()

        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    def detect_outliers(self, method: str = 'iqr') -> Dict:
        """Detect outliers in numerical columns."""
        outliers = {}

        for col in self.numeric_cols:
            data = self.data[col].dropna()

            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_mask = (data < lower) | (data > upper)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = z_scores > 3

            outliers[col] = {
                'count': outlier_mask.sum(),
                'percentage': outlier_mask.sum() / len(data) * 100,
                'values': data[outlier_mask].tolist()[:10]  # First 10 outliers
            }

        return outliers

    def analyze_correlations(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Analyze correlations between numerical features."""
        if len(self.numeric_cols) < 2:
            print("Need at least 2 numerical columns for correlation")
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Correlation heatmap
        corr = self.data[self.numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
        axes[0].set_title('Correlation Matrix')

        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_pairs = corr_pairs[:10]

        # Plot top correlations
        if top_pairs:
            pairs_labels = [f"{p[0][:10]} vs\n{p[1][:10]}" for p in top_pairs]
            corr_values = [p[2] for p in top_pairs]

            axes[1].barh(pairs_labels, corr_values, color=['red' if v < 0 else 'green' for v in corr_values])
            axes[1].set_xlabel('Correlation Coefficient')
            axes[1].set_title('Top 10 Feature Correlations')
            axes[1].axvline(0, color='black', linewidth=0.5)

        plt.tight_layout()
        return fig

    def analyze_missing_data(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Analyze patterns in missing data."""
        missing = self.data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            print("No missing data found")
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Bar plot of missing values
        missing.plot(kind='barh', ax=axes[0], color='coral')
        axes[0].set_xlabel('Number of Missing Values')
        axes[0].set_title('Missing Values by Feature')

        # Percentage plot
        missing_pct = (missing / len(self.data) * 100).sort_values(ascending=False)
        missing_pct.plot(kind='barh', ax=axes[1], color='steelblue')
        axes[1].set_xlabel('Percentage Missing (%)')
        axes[1].set_title('Missing Data Percentage')

        plt.tight_layout()
        return fig

    def analyze_categorical(self, max_categories: int = 20, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Analyze categorical features."""
        if len(self.categorical_cols) == 0:
            print("No categorical columns found")
            return None

        n_cols = min(len(self.categorical_cols), 6)
        n_rows = (n_cols + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(self.categorical_cols[:6]):
            value_counts = self.data[col].value_counts()

            if len(value_counts) > max_categories:
                value_counts = value_counts.head(max_categories)

            value_counts.plot(kind='bar', ax=axes[idx], color='skyblue', edgecolor='black')
            axes[idx].set_title(f'{col}\nUnique: {self.data[col].nunique()}, Missing: {self.data[col].isnull().sum()}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)

        # Hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    def generate_summary_stats(self) -> pd.DataFrame:
        """Generate comprehensive summary statistics."""
        stats_df = self.data[self.numeric_cols].describe().T

        # Add additional statistics
        stats_df['skewness'] = self.data[self.numeric_cols].skew()
        stats_df['kurtosis'] = self.data[self.numeric_cols].kurtosis()
        stats_df['missing'] = self.data[self.numeric_cols].isnull().sum()
        stats_df['missing_pct'] = self.data[self.numeric_cols].isnull().sum() / len(self.data) * 100

        # Detect outliers
        outlier_counts = {}
        for col in self.numeric_cols:
            data = self.data[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
            outlier_counts[col] = outliers

        stats_df['outliers'] = pd.Series(outlier_counts)

        return stats_df

    def create_full_report(self, output_dir: str = '.'):
        """Generate full EDA report with all visualizations."""
        print("Generating EDA Report...")
        print("\n" + "="*50)
        print("DATA PROFILE")
        print("="*50)

        profile = self.generate_profile()
        for key, value in profile.items():
            print(f"{key}: {value}")

        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(self.generate_summary_stats())

        print("\n" + "="*50)
        print("OUTLIERS ANALYSIS (IQR Method)")
        print("="*50)
        outliers = self.detect_outliers()
        for col, info in outliers.items():
            if info['count'] > 0:
                print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")

        # Generate visualizations
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)

        fig1 = self.analyze_distributions()
        if fig1:
            fig1.savefig(f'{output_dir}/distributions.png', dpi=300, bbox_inches='tight')
            print("✓ Saved distributions.png")
            plt.close()

        fig2 = self.analyze_correlations()
        if fig2:
            fig2.savefig(f'{output_dir}/correlations.png', dpi=300, bbox_inches='tight')
            print("✓ Saved correlations.png")
            plt.close()

        fig3 = self.analyze_missing_data()
        if fig3:
            fig3.savefig(f'{output_dir}/missing_data.png', dpi=300, bbox_inches='tight')
            print("✓ Saved missing_data.png")
            plt.close()

        fig4 = self.analyze_categorical()
        if fig4:
            fig4.savefig(f'{output_dir}/categorical.png', dpi=300, bbox_inches='tight')
            print("✓ Saved categorical.png")
            plt.close()

        print("\n✓ EDA Report Complete!")


def demo():
    """Demo with sample dataset."""
    np.random.seed(42)

    # Generate sample data
    n = 1000
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'income': np.random.exponential(50000, n),
        'credit_score': np.random.normal(700, 100, n),
        'loan_amount': np.random.normal(200000, 50000, n),
        'employment_years': np.random.randint(0, 40, n),
        'debt_ratio': np.random.uniform(0, 1, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
        'loan_status': np.random.choice(['Approved', 'Rejected'], n, p=[0.7, 0.3])
    })

    # Add some missing values
    data.loc[np.random.choice(data.index, 50), 'income'] = np.nan
    data.loc[np.random.choice(data.index, 30), 'credit_score'] = np.nan

    # Add some outliers
    data.loc[np.random.choice(data.index, 20), 'income'] = np.random.uniform(200000, 500000, 20)

    # Run EDA
    eda = EDAToolkit(data)
    eda.create_full_report()


if __name__ == '__main__':
    demo()
