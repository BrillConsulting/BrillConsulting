"""
Experiment Design Toolkit
==========================

Advanced experimental design and analysis methods:
- Factorial design (full and fractional)
- ANOVA (one-way, two-way, repeated measures)
- Power analysis and sample size calculation
- Randomization and blocking strategies
- Latin square and crossover designs
- Response surface methodology
- Optimal design theory
- Design of Experiments (DOE)

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from itertools import product, combinations
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class ExperimentDesign:
    """Experimental design and analysis toolkit."""

    def __init__(self, random_state: int = 42):
        """Initialize experiment design toolkit."""
        self.random_state = random_state
        np.random.seed(random_state)
        self.designs = {}
        self.results = {}

    def full_factorial_design(self, factors: Dict[str, List], replicates: int = 1) -> pd.DataFrame:
        """
        Create a full factorial design.

        Args:
            factors: Dictionary of factor names and their levels
            replicates: Number of replicates for each treatment

        Returns:
            DataFrame with experimental design
        """
        factor_names = list(factors.keys())
        factor_levels = [factors[f] for f in factor_names]

        # Generate all combinations
        combinations_list = list(product(*factor_levels))

        # Create design matrix
        design = []
        for rep in range(replicates):
            for combo in combinations_list:
                row = {factor_names[i]: combo[i] for i in range(len(factor_names))}
                row['replicate'] = rep + 1
                design.append(row)

        design_df = pd.DataFrame(design)

        # Randomize run order
        design_df['run_order'] = np.random.permutation(len(design_df))
        design_df = design_df.sort_values('run_order').reset_index(drop=True)

        self.designs['full_factorial'] = design_df

        return design_df

    def fractional_factorial_design(self, n_factors: int, resolution: int = 4) -> pd.DataFrame:
        """
        Create a fractional factorial design (2^(k-p) design).

        Args:
            n_factors: Number of factors
            resolution: Design resolution (3, 4, or 5)

        Returns:
            DataFrame with fractional factorial design
        """
        # For simplicity, create a 2^(k-1) design (half-fraction)
        if n_factors < 3:
            raise ValueError("Fractional factorial requires at least 3 factors")

        # Generate base design for k-1 factors
        base_factors = n_factors - 1
        base_levels = [[-1, 1]] * base_factors

        combinations_list = list(product(*base_levels))

        design = []
        for combo in combinations_list:
            row = {f'Factor{i+1}': combo[i] for i in range(base_factors)}

            # Generate last factor as interaction (highest resolution)
            # For resolution IV: k-factor = product of first factors
            last_factor = np.prod(combo)
            row[f'Factor{n_factors}'] = last_factor

            design.append(row)

        design_df = pd.DataFrame(design)

        # Randomize run order
        design_df['run_order'] = np.random.permutation(len(design_df))
        design_df = design_df.sort_values('run_order').reset_index(drop=True)

        self.designs['fractional_factorial'] = design_df

        return design_df

    def randomized_complete_block_design(self, treatments: List, blocks: List,
                                         shuffle: bool = True) -> pd.DataFrame:
        """
        Create a randomized complete block design.

        Args:
            treatments: List of treatment labels
            blocks: List of block labels
            shuffle: Whether to randomize within blocks

        Returns:
            DataFrame with RCBD design
        """
        design = []
        for block in blocks:
            block_treatments = treatments.copy()
            if shuffle:
                np.random.shuffle(block_treatments)

            for i, treatment in enumerate(block_treatments):
                design.append({
                    'block': block,
                    'treatment': treatment,
                    'order_in_block': i + 1
                })

        design_df = pd.DataFrame(design)
        self.designs['rcbd'] = design_df

        return design_df

    def latin_square_design(self, n: int, labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a Latin square design.

        Args:
            n: Size of the Latin square (n x n)
            labels: Treatment labels (default: A, B, C, ...)

        Returns:
            DataFrame with Latin square design
        """
        if labels is None:
            labels = [chr(65 + i) for i in range(n)]  # A, B, C, ...
        elif len(labels) != n:
            raise ValueError(f"Number of labels must equal n={n}")

        # Generate a random Latin square
        square = np.zeros((n, n), dtype=int)
        square[0] = np.random.permutation(n)

        for i in range(1, n):
            # Ensure each row is a permutation with no column repeats
            valid = False
            attempts = 0
            while not valid and attempts < 1000:
                square[i] = np.random.permutation(n)
                # Check if any column has duplicates so far
                valid = all(len(set(square[:i+1, j])) == i+1 for j in range(n))
                attempts += 1

        # Convert to DataFrame
        design = []
        for row in range(n):
            for col in range(n):
                design.append({
                    'row': row + 1,
                    'column': col + 1,
                    'treatment': labels[square[row, col]]
                })

        design_df = pd.DataFrame(design)
        self.designs['latin_square'] = design_df

        return design_df

    def one_way_anova(self, data: pd.DataFrame, group_col: str,
                     value_col: str) -> Dict:
        """
        Perform one-way ANOVA.

        Args:
            data: DataFrame with experimental data
            group_col: Column name for groups
            value_col: Column name for response values

        Returns:
            Dictionary with ANOVA results
        """
        groups = [data[data[group_col] == g][value_col].values
                 for g in data[group_col].unique()]

        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Calculate sum of squares
        grand_mean = data[value_col].mean()
        n_groups = len(groups)
        n_total = len(data)

        # Between-group sum of squares
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)

        # Within-group sum of squares
        ss_within = sum(sum((x - g.mean())**2) for g in groups for x in g)

        # Total sum of squares
        ss_total = sum((x - grand_mean)**2 for x in data[value_col])

        # Degrees of freedom
        df_between = n_groups - 1
        df_within = n_total - n_groups
        df_total = n_total - 1

        # Mean squares
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        # Effect size (eta squared)
        eta_squared = ss_between / ss_total

        # Omega squared (less biased effect size)
        omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'ss_between': ss_between,
            'ss_within': ss_within,
            'ss_total': ss_total,
            'df_between': df_between,
            'df_within': df_within,
            'df_total': df_total,
            'ms_between': ms_between,
            'ms_within': ms_within,
            'eta_squared': eta_squared,
            'omega_squared': omega_squared,
            'significant': p_value < 0.05,
            'group_means': {g: data[data[group_col] == g][value_col].mean()
                          for g in data[group_col].unique()}
        }

    def two_way_anova(self, data: pd.DataFrame, factor1_col: str,
                     factor2_col: str, value_col: str) -> Dict:
        """
        Perform two-way ANOVA with interaction.

        Args:
            data: DataFrame with experimental data
            factor1_col: First factor column
            factor2_col: Second factor column
            value_col: Response variable column

        Returns:
            Dictionary with two-way ANOVA results
        """
        # Grand mean
        grand_mean = data[value_col].mean()
        n_total = len(data)

        # Factor levels
        levels_a = data[factor1_col].unique()
        levels_b = data[factor2_col].unique()

        # Cell means
        cell_means = data.groupby([factor1_col, factor2_col])[value_col].mean()

        # Main effect A
        means_a = data.groupby(factor1_col)[value_col].mean()
        ss_a = sum(len(data[data[factor1_col] == a]) * (means_a[a] - grand_mean)**2
                  for a in levels_a)

        # Main effect B
        means_b = data.groupby(factor2_col)[value_col].mean()
        ss_b = sum(len(data[data[factor2_col] == b]) * (means_b[b] - grand_mean)**2
                  for b in levels_b)

        # Interaction effect
        ss_ab = 0
        for a in levels_a:
            for b in levels_b:
                cell_data = data[(data[factor1_col] == a) & (data[factor2_col] == b)]
                if len(cell_data) > 0:
                    cell_mean = cell_data[value_col].mean()
                    n_cell = len(cell_data)
                    ss_ab += n_cell * (cell_mean - means_a[a] - means_b[b] + grand_mean)**2

        # Error sum of squares
        ss_error = 0
        for a in levels_a:
            for b in levels_b:
                cell_data = data[(data[factor1_col] == a) & (data[factor2_col] == b)]
                if len(cell_data) > 0:
                    cell_mean = cell_data[value_col].mean()
                    ss_error += sum((x - cell_mean)**2 for x in cell_data[value_col])

        # Total sum of squares
        ss_total = sum((x - grand_mean)**2 for x in data[value_col])

        # Degrees of freedom
        df_a = len(levels_a) - 1
        df_b = len(levels_b) - 1
        df_ab = df_a * df_b
        df_error = n_total - len(levels_a) * len(levels_b)
        df_total = n_total - 1

        # Mean squares
        ms_a = ss_a / df_a
        ms_b = ss_b / df_b
        ms_ab = ss_ab / df_ab
        ms_error = ss_error / df_error

        # F-statistics
        f_a = ms_a / ms_error
        f_b = ms_b / ms_error
        f_ab = ms_ab / ms_error

        # P-values
        p_a = 1 - stats.f.cdf(f_a, df_a, df_error)
        p_b = 1 - stats.f.cdf(f_b, df_b, df_error)
        p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_error)

        return {
            'factor_a': {
                'f_statistic': f_a,
                'p_value': p_a,
                'ss': ss_a,
                'df': df_a,
                'ms': ms_a,
                'significant': p_a < 0.05
            },
            'factor_b': {
                'f_statistic': f_b,
                'p_value': p_b,
                'ss': ss_b,
                'df': df_b,
                'ms': ms_b,
                'significant': p_b < 0.05
            },
            'interaction': {
                'f_statistic': f_ab,
                'p_value': p_ab,
                'ss': ss_ab,
                'df': df_ab,
                'ms': ms_ab,
                'significant': p_ab < 0.05
            },
            'error': {
                'ss': ss_error,
                'df': df_error,
                'ms': ms_error
            },
            'total': {
                'ss': ss_total,
                'df': df_total
            }
        }

    def power_analysis(self, effect_size: float, alpha: float = 0.05,
                      power: float = 0.8, n_groups: int = 2) -> Dict:
        """
        Calculate required sample size for desired power.

        Args:
            effect_size: Cohen's f effect size
            alpha: Significance level
            power: Desired statistical power (1 - beta)
            n_groups: Number of groups

        Returns:
            Dictionary with power analysis results
        """
        # Cohen's f to f^2
        f_squared = effect_size ** 2

        # Critical F-value
        df_between = n_groups - 1

        # Iteratively find sample size
        for n_per_group in range(2, 10000):
            n_total = n_groups * n_per_group
            df_within = n_total - n_groups

            # Non-centrality parameter
            ncp = n_total * f_squared

            # Critical F at alpha
            f_crit = stats.f.ppf(1 - alpha, df_between, df_within)

            # Calculate power
            current_power = 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp)

            if current_power >= power:
                break

        return {
            'required_n_per_group': n_per_group,
            'total_n': n_total,
            'achieved_power': current_power,
            'effect_size': effect_size,
            'alpha': alpha,
            'n_groups': n_groups
        }

    def tukey_hsd(self, data: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
        """
        Perform Tukey's Honestly Significant Difference post-hoc test.

        Args:
            data: DataFrame with experimental data
            group_col: Column name for groups
            value_col: Column name for response values

        Returns:
            DataFrame with pairwise comparisons
        """
        groups = data[group_col].unique()
        n_groups = len(groups)

        # Calculate MSE and n per group (assuming equal n)
        anova_result = self.one_way_anova(data, group_col, value_col)
        mse = anova_result['ms_within']

        results = []
        for g1, g2 in combinations(groups, 2):
            data1 = data[data[group_col] == g1][value_col]
            data2 = data[data[group_col] == g2][value_col]

            mean1 = data1.mean()
            mean2 = data2.mean()
            n1 = len(data1)
            n2 = len(data2)

            # HSD statistic
            se = np.sqrt(mse * (1/n1 + 1/n2))
            q_stat = abs(mean1 - mean2) / se

            # Critical value from studentized range distribution (approximation)
            # Using Bonferroni correction as approximation
            alpha_adj = 0.05 / (n_groups * (n_groups - 1) / 2)
            t_crit = stats.t.ppf(1 - alpha_adj/2, anova_result['df_within'])

            significant = q_stat > t_crit

            results.append({
                'group1': g1,
                'group2': g2,
                'mean_diff': mean1 - mean2,
                'q_statistic': q_stat,
                'significant': significant
            })

        return pd.DataFrame(results)

    def response_surface_methodology(self, data: pd.DataFrame, factor_cols: List[str],
                                    response_col: str) -> Dict:
        """
        Fit a response surface model (second-order polynomial).

        Args:
            data: Experimental data
            factor_cols: List of factor column names
            response_col: Response variable column

        Returns:
            Dictionary with fitted model and optimal point
        """
        X = data[factor_cols].values
        y = data[response_col].values

        # Create design matrix with quadratic and interaction terms
        n_factors = len(factor_cols)
        n_obs = len(data)

        # Linear terms
        design_matrix = [np.ones(n_obs)]
        design_matrix.extend([X[:, i] for i in range(n_factors)])

        # Quadratic terms
        for i in range(n_factors):
            design_matrix.append(X[:, i] ** 2)

        # Interaction terms
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                design_matrix.append(X[:, i] * X[:, j])

        design_matrix = np.column_stack(design_matrix)

        # Fit model using least squares
        coefficients = np.linalg.lstsq(design_matrix, y, rcond=None)[0]

        # Predictions
        y_pred = design_matrix @ coefficients

        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # Find optimal point (simplified - using grid search)
        factor_ranges = [data[col].min(), data[col].max() for col in factor_cols]

        def predict_response(x):
            x = np.array(x).reshape(1, -1)
            dm = [1]
            dm.extend([x[0, i] for i in range(n_factors)])
            for i in range(n_factors):
                dm.append(x[0, i] ** 2)
            for i in range(n_factors):
                for j in range(i + 1, n_factors):
                    dm.append(x[0, i] * x[0, j])
            return -np.dot(dm, coefficients)  # Negative for maximization

        # Optimization
        bounds = [(factor_ranges[i*2], factor_ranges[i*2 + 1]) for i in range(n_factors)]
        x0 = [np.mean([factor_ranges[i*2], factor_ranges[i*2 + 1]]) for i in range(n_factors)]

        result = minimize(predict_response, x0, bounds=bounds, method='L-BFGS-B')

        optimal_point = {factor_cols[i]: result.x[i] for i in range(n_factors)}
        optimal_response = -result.fun

        return {
            'coefficients': coefficients,
            'r_squared': r_squared,
            'optimal_point': optimal_point,
            'optimal_response': optimal_response,
            'predictions': y_pred
        }

    def visualize_factorial_design(self, design: pd.DataFrame, response_col: Optional[str] = None) -> plt.Figure:
        """Visualize factorial design and results."""
        factor_cols = [col for col in design.columns if col not in ['replicate', 'run_order', response_col]]

        if len(factor_cols) == 2:
            fig, ax = plt.subplots(figsize=(10, 6))

            if response_col and response_col in design.columns:
                # Create interaction plot
                pivot = design.pivot_table(values=response_col,
                                          index=factor_cols[0],
                                          columns=factor_cols[1],
                                          aggfunc='mean')

                for col in pivot.columns:
                    ax.plot(pivot.index, pivot[col], marker='o', linewidth=2,
                           markersize=8, label=f'{factor_cols[1]}={col}')

                ax.set_xlabel(factor_cols[0], fontsize=12)
                ax.set_ylabel(f'Mean {response_col}', fontsize=12)
                ax.set_title('Interaction Plot', fontsize=14, weight='bold')
                ax.legend(title=factor_cols[1])
                ax.grid(alpha=0.3)
            else:
                # Just show design points
                for level in design[factor_cols[1]].unique():
                    subset = design[design[factor_cols[1]] == level]
                    ax.scatter(subset[factor_cols[0]], subset.index,
                             label=f'{factor_cols[1]}={level}', s=100)

                ax.set_xlabel(factor_cols[0], fontsize=12)
                ax.set_ylabel('Run', fontsize=12)
                ax.set_title('Factorial Design Layout', fontsize=14, weight='bold')
                ax.legend()
                ax.grid(alpha=0.3)

        else:
            # For more than 2 factors, show a different visualization
            fig, ax = plt.subplots(figsize=(12, 6))

            # Show design matrix as heatmap
            design_numeric = design[factor_cols].copy()

            # Convert to numeric if needed
            for col in design_numeric.columns:
                if design_numeric[col].dtype == 'object':
                    design_numeric[col] = pd.Categorical(design_numeric[col]).codes

            sns.heatmap(design_numeric.T, annot=True, fmt='g', cmap='RdYlBu_r',
                       cbar_kws={'label': 'Level'}, ax=ax)
            ax.set_xlabel('Run Number', fontsize=12)
            ax.set_ylabel('Factors', fontsize=12)
            ax.set_title('Factorial Design Matrix', fontsize=14, weight='bold')

        plt.tight_layout()
        return fig

    def visualize_anova_results(self, data: pd.DataFrame, group_col: str,
                               value_col: str) -> plt.Figure:
        """Visualize ANOVA results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Box plot
        sns.boxplot(data=data, x=group_col, y=value_col, ax=axes[0, 0])
        axes[0, 0].set_title('Box Plot by Group')
        axes[0, 0].grid(alpha=0.3)

        # Violin plot
        sns.violinplot(data=data, x=group_col, y=value_col, ax=axes[0, 1])
        axes[0, 1].set_title('Violin Plot by Group')
        axes[0, 1].grid(alpha=0.3)

        # Means plot with error bars
        means = data.groupby(group_col)[value_col].mean()
        sems = data.groupby(group_col)[value_col].sem()
        axes[1, 0].bar(range(len(means)), means, yerr=sems, capsize=5,
                      alpha=0.7, edgecolor='black')
        axes[1, 0].set_xticks(range(len(means)))
        axes[1, 0].set_xticklabels(means.index, rotation=45)
        axes[1, 0].set_ylabel(value_col)
        axes[1, 0].set_title('Group Means with Standard Error')
        axes[1, 0].grid(alpha=0.3, axis='y')

        # ANOVA results summary
        anova_result = self.one_way_anova(data, group_col, value_col)
        summary_text = "One-Way ANOVA Results\n"
        summary_text += "=" * 30 + "\n\n"
        summary_text += f"F-statistic: {anova_result['f_statistic']:.4f}\n"
        summary_text += f"P-value: {anova_result['p_value']:.4e}\n"
        summary_text += f"Significant: {anova_result['significant']}\n\n"
        summary_text += f"Effect Size (η²): {anova_result['eta_squared']:.4f}\n"
        summary_text += f"Omega Squared (ω²): {anova_result['omega_squared']:.4f}\n\n"
        summary_text += f"SS Between: {anova_result['ss_between']:.2f}\n"
        summary_text += f"SS Within: {anova_result['ss_within']:.2f}\n"
        summary_text += f"SS Total: {anova_result['ss_total']:.2f}\n"

        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_title('ANOVA Summary')
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig


def demo():
    """Demo experiment design toolkit."""
    np.random.seed(42)

    print("Experiment Design Toolkit Demo")
    print("="*60)

    ed = ExperimentDesign(random_state=42)

    # 1. Full Factorial Design
    print("\n1. Full Factorial Design (2^3)")
    print("-" * 60)
    factors = {
        'Temperature': [20, 30],
        'Pressure': [1, 2],
        'Catalyst': ['A', 'B']
    }
    design = ed.full_factorial_design(factors, replicates=2)
    print(f"Design size: {len(design)} runs")
    print(f"\nFirst 10 runs:\n{design.head(10)}")

    # Simulate response data
    design['yield'] = 50 + 5 * (design['Temperature'] == 30) + \
                     3 * (design['Pressure'] == 2) + \
                     2 * (design['Catalyst'] == 'B') + \
                     np.random.randn(len(design)) * 2

    # Visualize
    fig1 = ed.visualize_factorial_design(design, 'yield')
    fig1.savefig('experiment_factorial_design.png', dpi=300, bbox_inches='tight')
    print("✓ Saved experiment_factorial_design.png")
    plt.close()

    # 2. One-Way ANOVA
    print("\n2. One-Way ANOVA")
    print("-" * 60)
    result = ed.one_way_anova(design, 'Temperature', 'yield')
    print(f"F-statistic: {result['f_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4e}")
    print(f"Significant: {result['significant']}")
    print(f"Effect size (η²): {result['eta_squared']:.4f}")
    print(f"Group means: {result['group_means']}")

    # Visualize ANOVA
    fig2 = ed.visualize_anova_results(design, 'Temperature', 'yield')
    fig2.savefig('experiment_anova_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved experiment_anova_results.png")
    plt.close()

    # 3. Two-Way ANOVA
    print("\n3. Two-Way ANOVA")
    print("-" * 60)
    result = ed.two_way_anova(design, 'Temperature', 'Pressure', 'yield')
    print(f"Factor A (Temperature):")
    print(f"  F = {result['factor_a']['f_statistic']:.4f}, p = {result['factor_a']['p_value']:.4e}")
    print(f"Factor B (Pressure):")
    print(f"  F = {result['factor_b']['f_statistic']:.4f}, p = {result['factor_b']['p_value']:.4e}")
    print(f"Interaction (A×B):")
    print(f"  F = {result['interaction']['f_statistic']:.4f}, p = {result['interaction']['p_value']:.4e}")

    # 4. Power Analysis
    print("\n4. Power Analysis")
    print("-" * 60)
    power_result = ed.power_analysis(effect_size=0.5, alpha=0.05, power=0.8, n_groups=3)
    print(f"Effect size: {power_result['effect_size']:.2f}")
    print(f"Required sample size per group: {power_result['required_n_per_group']}")
    print(f"Total sample size: {power_result['total_n']}")
    print(f"Achieved power: {power_result['achieved_power']:.4f}")

    # 5. Fractional Factorial Design
    print("\n5. Fractional Factorial Design (2^(4-1))")
    print("-" * 60)
    frac_design = ed.fractional_factorial_design(n_factors=4, resolution=4)
    print(f"Design size: {len(frac_design)} runs (vs {2**4} for full factorial)")
    print(f"\nDesign matrix:\n{frac_design.head(10)}")

    # 6. Latin Square Design
    print("\n6. Latin Square Design")
    print("-" * 60)
    latin_design = ed.latin_square_design(n=4, labels=['A', 'B', 'C', 'D'])
    print(f"\nLatin Square:\n{latin_design.pivot(index='row', columns='column', values='treatment')}")

    # 7. Randomized Complete Block Design
    print("\n7. Randomized Complete Block Design")
    print("-" * 60)
    treatments = ['Control', 'Treatment1', 'Treatment2', 'Treatment3']
    blocks = ['Block1', 'Block2', 'Block3', 'Block4']
    rcbd_design = ed.randomized_complete_block_design(treatments, blocks)
    print(f"\nRCBD:\n{rcbd_design.head(12)}")

    # 8. Tukey HSD Post-hoc Test
    print("\n8. Tukey HSD Post-hoc Test")
    print("-" * 60)
    tukey_results = ed.tukey_hsd(design, 'Catalyst', 'yield')
    print(f"\nPairwise comparisons:\n{tukey_results}")

    # 9. Response Surface Methodology
    print("\n9. Response Surface Methodology")
    print("-" * 60)
    # Create RSM data
    rsm_data = pd.DataFrame({
        'Temperature': np.random.uniform(20, 40, 50),
        'Pressure': np.random.uniform(1, 3, 50)
    })
    rsm_data['yield'] = 50 + 2 * rsm_data['Temperature'] - 0.05 * rsm_data['Temperature']**2 + \
                       5 * rsm_data['Pressure'] - 0.5 * rsm_data['Pressure']**2 + \
                       0.1 * rsm_data['Temperature'] * rsm_data['Pressure'] + \
                       np.random.randn(50) * 3

    rsm_result = ed.response_surface_methodology(rsm_data, ['Temperature', 'Pressure'], 'yield')
    print(f"R²: {rsm_result['r_squared']:.4f}")
    print(f"Optimal point: {rsm_result['optimal_point']}")
    print(f"Optimal response: {rsm_result['optimal_response']:.2f}")

    print("\n" + "="*60)
    print("✓ Experiment Design Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo()
