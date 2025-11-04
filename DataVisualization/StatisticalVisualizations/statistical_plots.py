"""
Statistical Visualizations with Matplotlib and Seaborn
======================================================

A comprehensive collection of statistical visualization techniques:
- Distribution plots (histograms, KDE, box plots, violin plots)
- Relationship plots (scatter, regression, heatmaps)
- Categorical plots (bar, count, point plots)
- Matrix plots (correlation, pairwise relationships)
- Statistical annotations and confidence intervals

Features:
- Professional publication-ready plots
- Customizable themes and color palettes
- Statistical testing visualization
- Multi-plot layouts and subplots
- Export to high-resolution formats

Technologies: Matplotlib, Seaborn, SciPy, Pandas
Author: Brill Consulting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class StatisticalVisualizer:
    """Statistical visualization toolkit."""

    def __init__(self, style='whitegrid', palette='husl'):
        """
        Initialize visualizer with style settings.

        Args:
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
            palette: Color palette name
        """
        sns.set_style(style)
        sns.set_palette(palette)
        self.figsize = (12, 6)

    def plot_distribution(self, data: pd.Series, title: str = 'Distribution Plot',
                         bins: int = 30, kde: bool = True) -> plt.Figure:
        """
        Create distribution plot with histogram and KDE.

        Args:
            data: Data series to plot
            title: Plot title
            bins: Number of histogram bins
            kde: Whether to show KDE curve

        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Histogram with KDE
        sns.histplot(data, bins=bins, kde=kde, ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title(f'{title} - Histogram')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')

        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[0].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        axes[0].legend()

        # Box plot
        sns.boxplot(y=data, ax=axes[1], color='lightcoral')
        axes[1].set_title(f'{title} - Box Plot')
        axes[1].set_ylabel('Value')

        # Add statistics text
        stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
        axes[1].text(0.5, 0.95, stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def plot_comparison(self, data: pd.DataFrame, x_col: str, y_col: str,
                       hue_col: Optional[str] = None) -> plt.Figure:
        """
        Create comparison plots for categorical and numerical data.

        Args:
            data: DataFrame with data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            hue_col: Column for color grouping

        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Box plot
        sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=axes[0, 0])
        axes[0, 0].set_title('Box Plot Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Violin plot
        sns.violinplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=axes[0, 1])
        axes[0, 1].set_title('Violin Plot Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Swarm plot
        if len(data) < 500:  # Swarm plot works best with smaller datasets
            sns.swarmplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=axes[1, 0], size=3)
        else:
            sns.stripplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=axes[1, 0], alpha=0.5)
        axes[1, 0].set_title('Swarm/Strip Plot')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Bar plot with error bars
        sns.barplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=axes[1, 1], ci=95)
        axes[1, 1].set_title('Mean with 95% CI')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, data: pd.DataFrame, method: str = 'pearson',
                                annot: bool = True) -> plt.Figure:
        """
        Create correlation matrix heatmap.

        Args:
            data: DataFrame with numerical columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
            annot: Whether to annotate cells with values

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate correlation
        corr = data.corr(method=method)

        # Create heatmap
        sns.heatmap(corr, annot=annot, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title(f'Correlation Matrix ({method.capitalize()})', fontsize=16, pad=20)

        plt.tight_layout()
        return fig

    def plot_regression(self, data: pd.DataFrame, x_col: str, y_col: str,
                       hue_col: Optional[str] = None) -> plt.Figure:
        """
        Create regression plot with confidence intervals.

        Args:
            data: DataFrame with data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            hue_col: Column for color grouping

        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Regression plot with CI
        sns.regplot(data=data, x=x_col, y=y_col, ax=axes[0],
                   scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'linewidth': 2})
        axes[0].set_title('Linear Regression with 95% CI')

        # Calculate correlation
        corr, p_value = stats.pearsonr(data[x_col].dropna(), data[y_col].dropna())
        axes[0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_value:.3e}',
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Residual plot
        sns.residplot(data=data, x=x_col, y=y_col, ax=axes[1],
                     scatter_kws={'alpha': 0.5})
        axes[1].set_title('Residual Plot')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2)

        plt.tight_layout()
        return fig

    def plot_pairplot(self, data: pd.DataFrame, hue_col: Optional[str] = None,
                     columns: Optional[List[str]] = None) -> sns.PairGrid:
        """
        Create pairwise relationship plot matrix.

        Args:
            data: DataFrame with data
            hue_col: Column for color grouping
            columns: List of columns to include

        Returns:
            seaborn PairGrid
        """
        if columns:
            data = data[columns + ([hue_col] if hue_col and hue_col not in columns else [])]

        # Create pairplot
        g = sns.pairplot(data, hue=hue_col, diag_kind='kde', corner=False,
                        plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.7})
        g.fig.suptitle('Pairwise Relationships', y=1.02, fontsize=16)

        return g

    def plot_categorical_analysis(self, data: pd.DataFrame, cat_col: str,
                                  num_col: str) -> plt.Figure:
        """
        Create comprehensive categorical data analysis.

        Args:
            data: DataFrame with data
            cat_col: Categorical column name
            num_col: Numerical column name

        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Count plot
        sns.countplot(data=data, x=cat_col, ax=axes[0, 0], palette='Set2')
        axes[0, 0].set_title('Category Counts')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Bar plot with values
        cat_means = data.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
        cat_means.plot(kind='bar', ax=axes[0, 1], color='steelblue')
        axes[0, 1].set_title(f'Mean {num_col} by {cat_col}')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylabel(f'Mean {num_col}')

        # Box plot
        sns.boxplot(data=data, x=cat_col, y=num_col, ax=axes[1, 0], palette='Set3')
        axes[1, 0].set_title('Distribution by Category')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Violin plot with quartiles
        sns.violinplot(data=data, x=cat_col, y=num_col, ax=axes[1, 1],
                      palette='muted', inner='quartile')
        axes[1, 1].set_title('Violin Plot with Quartiles')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def plot_statistical_tests(self, group1: np.ndarray, group2: np.ndarray,
                               labels: Tuple[str, str] = ('Group 1', 'Group 2')) -> plt.Figure:
        """
        Visualize statistical comparison between two groups.

        Args:
            group1: First group data
            group2: Second group data
            labels: Labels for the groups

        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Prepare data for seaborn
        data = pd.DataFrame({
            'value': np.concatenate([group1, group2]),
            'group': [labels[0]] * len(group1) + [labels[1]] * len(group2)
        })

        # Distribution comparison
        sns.kdeplot(data=group1, ax=axes[0, 0], label=labels[0], fill=True, alpha=0.5)
        sns.kdeplot(data=group2, ax=axes[0, 0], label=labels[1], fill=True, alpha=0.5)
        axes[0, 0].set_title('Density Plot Comparison')
        axes[0, 0].legend()

        # Box plot comparison
        sns.boxplot(data=data, x='group', y='value', ax=axes[0, 1])
        axes[0, 1].set_title('Box Plot Comparison')

        # Perform t-test
        t_stat, t_pval = stats.ttest_ind(group1, group2)
        # Perform Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(group1, group2)

        # Bar plot with error bars
        means = [group1.mean(), group2.mean()]
        stds = [group1.std(), group2.std()]
        axes[1, 0].bar(labels, means, yerr=stds, capsize=10, color=['skyblue', 'lightcoral'])
        axes[1, 0].set_title('Mean Comparison with Std Dev')
        axes[1, 0].set_ylabel('Value')

        # Add statistical test results
        test_results = f'T-test:\nt = {t_stat:.3f}, p = {t_pval:.4f}\n\n'
        test_results += f'Mann-Whitney U:\nU = {u_stat:.3f}, p = {u_pval:.4f}'

        axes[1, 1].text(0.5, 0.5, test_results, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center', horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_title('Statistical Test Results')
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """
        Save plot to file.

        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {filename}")


def demo():
    """Demonstrate statistical visualizations."""
    # Create sample data
    np.random.seed(42)

    # Generate sample dataset
    n_samples = 500
    data = pd.DataFrame({
        'height': np.random.normal(170, 10, n_samples),
        'weight': np.random.normal(70, 15, n_samples),
        'age': np.random.randint(18, 65, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    })

    # Add correlation between height and weight
    data['weight'] = data['height'] * 0.5 + np.random.normal(0, 10, n_samples)

    # Initialize visualizer
    viz = StatisticalVisualizer()

    print("Creating statistical visualizations...")

    # 1. Distribution plot
    print("\n1. Distribution Analysis")
    fig1 = viz.plot_distribution(data['income'], title='Income Distribution')
    viz.save_plot(fig1, 'income_distribution.png')
    plt.close()

    # 2. Comparison plots
    print("\n2. Comparison Analysis")
    fig2 = viz.plot_comparison(data, 'education', 'income', hue_col='gender')
    viz.save_plot(fig2, 'income_comparison.png')
    plt.close()

    # 3. Correlation matrix
    print("\n3. Correlation Analysis")
    numeric_cols = ['height', 'weight', 'age', 'income']
    fig3 = viz.plot_correlation_matrix(data[numeric_cols])
    viz.save_plot(fig3, 'correlation_matrix.png')
    plt.close()

    # 4. Regression analysis
    print("\n4. Regression Analysis")
    fig4 = viz.plot_regression(data, 'height', 'weight')
    viz.save_plot(fig4, 'height_weight_regression.png')
    plt.close()

    # 5. Categorical analysis
    print("\n5. Categorical Analysis")
    fig5 = viz.plot_categorical_analysis(data, 'education', 'income')
    viz.save_plot(fig5, 'education_income_analysis.png')
    plt.close()

    # 6. Statistical tests
    print("\n6. Statistical Testing")
    male_income = data[data['gender'] == 'Male']['income'].values
    female_income = data[data['gender'] == 'Female']['income'].values
    fig6 = viz.plot_statistical_tests(male_income, female_income, ('Male', 'Female'))
    viz.save_plot(fig6, 'gender_income_test.png')
    plt.close()

    print("\n✓ All visualizations created successfully!")
    print("\nGenerated files:")
    print("  - income_distribution.png")
    print("  - income_comparison.png")
    print("  - correlation_matrix.png")
    print("  - height_weight_regression.png")
    print("  - education_income_analysis.png")
    print("  - gender_income_test.png")


if __name__ == '__main__':
    demo()
