"""
Causal Discovery Toolkit
=========================

Comprehensive causal discovery and inference methods:
- PC Algorithm (constraint-based causal discovery)
- Conditional independence tests (partial correlation, chi-square)
- Causal DAG (Directed Acyclic Graph) construction
- Markov blanket discovery
- Pairwise causal direction tests
- D-separation tests
- Causal graph validation
- Intervention simulation
- Comprehensive visualizations

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
from itertools import combinations, permutations
from typing import Dict, List, Tuple, Optional, Union, Set
import warnings
warnings.filterwarnings('ignore')


class CausalDiscoveryToolkit:
    """Comprehensive causal discovery toolkit with multiple algorithms."""

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize causal discovery toolkit.

        Args:
            significance_level: Significance level for independence tests
        """
        self.significance_level = significance_level
        self.causal_graph = None
        self.skeleton = None
        self.separating_sets = {}

    def partial_correlation_test(self, X: np.ndarray, i: int, j: int,
                                 conditioning_set: List[int]) -> Tuple[float, float]:
        """
        Test conditional independence using partial correlation.

        Tests whether X[:, i] and X[:, j] are independent given X[:, conditioning_set].

        Args:
            X: Data matrix (n_samples, n_features)
            i: Index of first variable
            j: Index of second variable
            conditioning_set: Indices of conditioning variables

        Returns:
            Tuple of (partial_correlation, p_value)
        """
        if len(conditioning_set) == 0:
            # No conditioning: simple correlation
            corr, p_value = pearsonr(X[:, i], X[:, j])
            return corr, p_value

        # Calculate partial correlation
        # Use precision matrix (inverse covariance)
        indices = [i, j] + list(conditioning_set)
        sub_data = X[:, indices]

        # Calculate covariance matrix
        cov_matrix = np.cov(sub_data.T)

        # Regularization for numerical stability
        cov_matrix += np.eye(len(indices)) * 1e-8

        # Calculate precision matrix (inverse covariance)
        try:
            precision_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # If inversion fails, return high p-value (cannot reject independence)
            return 0.0, 1.0

        # Partial correlation from precision matrix
        partial_corr = -precision_matrix[0, 1] / np.sqrt(precision_matrix[0, 0] * precision_matrix[1, 1])

        # Fisher's Z-transform for significance test
        n = len(X)
        k = len(conditioning_set)

        if abs(partial_corr) >= 1:
            # Perfect correlation
            z_score = np.inf
        else:
            z_score = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))

        # Standard error
        se = 1.0 / np.sqrt(n - k - 3)

        # Test statistic
        test_stat = abs(z_score / se)

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(test_stat))

        return partial_corr, p_value

    def conditional_independence_test(self, X: np.ndarray, i: int, j: int,
                                     conditioning_set: List[int],
                                     test_type: str = 'partial_correlation') -> bool:
        """
        Test conditional independence between variables i and j given conditioning_set.

        Args:
            X: Data matrix
            i: Index of first variable
            j: Index of second variable
            conditioning_set: Indices of conditioning variables
            test_type: Type of test ('partial_correlation' or 'chi_square')

        Returns:
            True if independent, False otherwise
        """
        if test_type == 'partial_correlation':
            _, p_value = self.partial_correlation_test(X, i, j, conditioning_set)
            return p_value > self.significance_level

        elif test_type == 'chi_square':
            # For categorical data
            # Discretize continuous data
            n_bins = 5
            x_i = pd.cut(X[:, i], bins=n_bins, labels=False)
            x_j = pd.cut(X[:, j], bins=n_bins, labels=False)

            if len(conditioning_set) == 0:
                # Simple independence test
                contingency = pd.crosstab(x_i, x_j)
                chi2, p_value, _, _ = chi2_contingency(contingency)
                return p_value > self.significance_level
            else:
                # Conditional independence using stratification
                # Discretize conditioning variables
                conditioning_data = X[:, conditioning_set]
                if len(conditioning_set) == 1:
                    strata = pd.cut(conditioning_data.flatten(), bins=3, labels=False)
                else:
                    # Combine multiple conditioning variables
                    strata = pd.cut(conditioning_data.sum(axis=1), bins=3, labels=False)

                # Test independence in each stratum
                p_values = []
                for stratum in np.unique(strata):
                    mask = strata == stratum
                    if np.sum(mask) < 10:  # Skip small strata
                        continue

                    try:
                        contingency = pd.crosstab(x_i[mask], x_j[mask])
                        chi2, p_val, _, _ = chi2_contingency(contingency)
                        p_values.append(p_val)
                    except:
                        continue

                if len(p_values) == 0:
                    return True  # Cannot reject independence

                # Use minimum p-value (conservative)
                return min(p_values) > self.significance_level

        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def pc_algorithm(self, X: np.ndarray, variable_names: Optional[List[str]] = None,
                    max_conditioning_size: int = 3) -> Dict:
        """
        PC Algorithm for causal discovery.

        The PC algorithm learns causal structure through conditional independence testing.

        Args:
            X: Data matrix (n_samples, n_features)
            variable_names: Names of variables
            max_conditioning_size: Maximum size of conditioning sets

        Returns:
            Dictionary with causal graph information
        """
        n_features = X.shape[1]

        if variable_names is None:
            variable_names = [f'X{i}' for i in range(n_features)]

        # Initialize: Complete undirected graph (skeleton)
        skeleton = np.ones((n_features, n_features)) - np.eye(n_features)
        separating_sets = {}

        # Phase I: Skeleton discovery
        # Start with conditioning set size 0, increase iteratively
        for cond_size in range(max_conditioning_size + 1):
            changed = False

            for i, j in combinations(range(n_features), 2):
                if skeleton[i, j] == 0:
                    continue  # Already removed

                # Get neighbors of i (excluding j)
                neighbors_i = [k for k in range(n_features)
                             if k != j and skeleton[i, k] == 1]

                # Get neighbors of j (excluding i)
                neighbors_j = [k for k in range(n_features)
                             if k != i and skeleton[j, k] == 1]

                # Consider all conditioning sets of size cond_size
                for conditioning_set in combinations(neighbors_i, min(cond_size, len(neighbors_i))):
                    conditioning_set = list(conditioning_set)

                    # Test conditional independence
                    if self.conditional_independence_test(X, i, j, conditioning_set):
                        # Remove edge
                        skeleton[i, j] = 0
                        skeleton[j, i] = 0
                        separating_sets[(i, j)] = conditioning_set
                        separating_sets[(j, i)] = conditioning_set
                        changed = True
                        break

            if not changed and cond_size > 0:
                break  # No changes in this round

        # Phase II: Orient edges
        # Create directed graph (1 = i -> j, -1 = i <- j, 2 = i - j undirected)
        dag = skeleton.copy()

        # Rule 1: Orient v-structures (colliders)
        # If i - k - j and i and j are not adjacent, then orient as i -> k <- j
        for k in range(n_features):
            neighbors_k = [i for i in range(n_features) if skeleton[i, k] == 1]

            for i, j in combinations(neighbors_k, 2):
                if skeleton[i, j] == 0:  # i and j not adjacent
                    # Check if k is in separating set
                    sep_set = separating_sets.get((i, j), [])
                    if k not in sep_set:
                        # Orient as i -> k <- j (v-structure)
                        dag[i, k] = 1
                        dag[k, i] = 0
                        dag[j, k] = 1
                        dag[k, j] = 0

        # Additional orientation rules (simplified)
        # Rule 2: If i -> j - k and i and k not adjacent, then j -> k
        changed = True
        while changed:
            changed = False
            for i, j, k in permutations(range(n_features), 3):
                if dag[i, j] == 1 and dag[j, i] == 0:  # i -> j
                    if dag[j, k] == 1 and dag[k, j] == 1:  # j - k (undirected)
                        if skeleton[i, k] == 0:  # i and k not adjacent
                            dag[j, k] = 1
                            dag[k, j] = 0
                            changed = True

        self.skeleton = skeleton
        self.causal_graph = dag
        self.separating_sets = separating_sets
        self.variable_names = variable_names

        # Count edges
        n_skeleton_edges = int(np.sum(skeleton) / 2)
        n_directed_edges = int(np.sum((dag == 1) & (dag.T == 0)))
        n_undirected_edges = int(np.sum((dag == 1) & (dag.T == 1)) / 2)

        return {
            'skeleton': skeleton,
            'causal_graph': dag,
            'separating_sets': separating_sets,
            'variable_names': variable_names,
            'n_skeleton_edges': n_skeleton_edges,
            'n_directed_edges': n_directed_edges,
            'n_undirected_edges': n_undirected_edges
        }

    def find_markov_blanket(self, X: np.ndarray, target_idx: int,
                           variable_names: Optional[List[str]] = None) -> Dict:
        """
        Find Markov blanket of a target variable.

        Markov blanket includes: parents, children, and spouses (parents of children).

        Args:
            X: Data matrix
            target_idx: Index of target variable
            variable_names: Names of variables

        Returns:
            Dictionary with Markov blanket information
        """
        if self.causal_graph is None:
            # Run PC algorithm first
            self.pc_algorithm(X, variable_names)

        dag = self.causal_graph
        n_features = len(dag)

        if variable_names is None:
            variable_names = self.variable_names

        # Find parents (incoming edges)
        parents = [i for i in range(n_features)
                  if dag[i, target_idx] == 1 and dag[target_idx, i] == 0]

        # Find children (outgoing edges)
        children = [i for i in range(n_features)
                   if dag[target_idx, i] == 1 and dag[i, target_idx] == 0]

        # Find spouses (parents of children)
        spouses = []
        for child in children:
            child_parents = [i for i in range(n_features)
                           if i != target_idx and dag[i, child] == 1 and dag[child, i] == 0]
            spouses.extend(child_parents)

        spouses = list(set(spouses))  # Remove duplicates

        markov_blanket = list(set(parents + children + spouses))

        return {
            'target': variable_names[target_idx],
            'parents': [variable_names[i] for i in parents],
            'children': [variable_names[i] for i in children],
            'spouses': [variable_names[i] for i in spouses],
            'markov_blanket': [variable_names[i] for i in markov_blanket],
            'markov_blanket_indices': markov_blanket
        }

    def test_d_separation(self, i: int, j: int, conditioning_set: List[int]) -> bool:
        """
        Test d-separation in the causal graph.

        Args:
            i: Index of first variable
            j: Index of second variable
            conditioning_set: Indices of conditioning variables

        Returns:
            True if d-separated, False otherwise
        """
        if self.causal_graph is None:
            raise ValueError("Causal graph not initialized. Run pc_algorithm first.")

        # Simplified d-separation test using graph connectivity
        # Convert to NetworkX graph
        G = nx.DiGraph()
        n_features = len(self.causal_graph)

        for x in range(n_features):
            for y in range(n_features):
                if self.causal_graph[x, y] == 1 and self.causal_graph[y, x] == 0:
                    G.add_edge(x, y)
                elif self.causal_graph[x, y] == 1 and self.causal_graph[y, x] == 1:
                    # Undirected edge: add both directions
                    G.add_edge(x, y)
                    G.add_edge(y, x)

        # Remove conditioning nodes and check path
        G_modified = G.copy()
        G_modified.remove_nodes_from(conditioning_set)

        # Check if path exists
        try:
            has_path = nx.has_path(G_modified, i, j) or nx.has_path(G_modified, j, i)
            return not has_path  # d-separated if no path
        except nx.NodeNotFound:
            return True  # d-separated if nodes removed

    def pairwise_causal_direction(self, X: np.ndarray, i: int, j: int) -> Dict:
        """
        Test pairwise causal direction using regression-based asymmetry.

        Args:
            X: Data matrix
            i: Index of first variable
            j: Index of second variable

        Returns:
            Dictionary with causal direction information
        """
        from sklearn.linear_model import LinearRegression

        # Test i -> j
        model_ij = LinearRegression()
        model_ij.fit(X[:, i].reshape(-1, 1), X[:, j])
        residuals_ij = X[:, j] - model_ij.predict(X[:, i].reshape(-1, 1))
        independence_ij = abs(pearsonr(X[:, i], residuals_ij)[0])

        # Test j -> i
        model_ji = LinearRegression()
        model_ji.fit(X[:, j].reshape(-1, 1), X[:, i])
        residuals_ji = X[:, i] - model_ji.predict(X[:, j].reshape(-1, 1))
        independence_ji = abs(pearsonr(X[:, j], residuals_ji)[0])

        # Lower correlation indicates better independence (better direction)
        if independence_ij < independence_ji:
            direction = f"{i} -> {j}"
            confidence = 1 - independence_ij / (independence_ij + independence_ji + 1e-10)
        else:
            direction = f"{j} -> {i}"
            confidence = 1 - independence_ji / (independence_ij + independence_ji + 1e-10)

        return {
            'direction': direction,
            'confidence': float(confidence),
            'independence_ij': float(independence_ij),
            'independence_ji': float(independence_ji)
        }

    def validate_causal_graph(self, X: np.ndarray) -> Dict:
        """
        Validate the learned causal graph.

        Args:
            X: Data matrix

        Returns:
            Dictionary with validation metrics
        """
        if self.causal_graph is None:
            raise ValueError("Causal graph not initialized. Run pc_algorithm first.")

        dag = self.causal_graph
        n_features = len(dag)

        # Check for cycles (DAG property)
        G = nx.DiGraph()
        for i in range(n_features):
            for j in range(n_features):
                if dag[i, j] == 1 and dag[j, i] == 0:
                    G.add_edge(i, j)

        is_dag = nx.is_directed_acyclic_graph(G)

        # Count violations of conditional independence
        violations = 0
        total_tests = 0

        for i, j in combinations(range(n_features), 2):
            if dag[i, j] == 0 and dag[j, i] == 0:
                # Should be independent given some set
                sep_set = self.separating_sets.get((i, j), [])
                is_independent = self.conditional_independence_test(X, i, j, sep_set)
                total_tests += 1
                if not is_independent:
                    violations += 1

        violation_rate = violations / total_tests if total_tests > 0 else 0

        return {
            'is_dag': is_dag,
            'n_nodes': n_features,
            'n_edges': int(np.sum((dag == 1) & (dag.T == 0))),
            'violation_rate': float(violation_rate),
            'n_violations': violations,
            'n_tests': total_tests
        }

    def simulate_intervention(self, X: np.ndarray, intervene_idx: int,
                             intervene_value: float,
                             n_samples: int = 1000) -> np.ndarray:
        """
        Simulate intervention (do-operator) on a variable.

        Args:
            X: Original data matrix
            intervene_idx: Index of variable to intervene on
            intervene_value: Value to set the variable to
            n_samples: Number of samples to generate

        Returns:
            Simulated data after intervention
        """
        if self.causal_graph is None:
            raise ValueError("Causal graph not initialized. Run pc_algorithm first.")

        dag = self.causal_graph
        n_features = X.shape[1]

        # Build structural equations from data
        from sklearn.linear_model import LinearRegression

        models = {}
        for i in range(n_features):
            # Find parents
            parents = [j for j in range(n_features)
                      if dag[j, i] == 1 and dag[i, j] == 0]

            if len(parents) > 0:
                model = LinearRegression()
                model.fit(X[:, parents], X[:, i])
                models[i] = {'parents': parents, 'model': model}

        # Simulate intervention
        # Topological sort for generation order
        G = nx.DiGraph()
        for i in range(n_features):
            for j in range(n_features):
                if dag[i, j] == 1 and dag[j, i] == 0:
                    G.add_edge(i, j)

        try:
            topo_order = list(nx.topological_sort(G))
        except:
            # If not DAG, use arbitrary order
            topo_order = list(range(n_features))

        # Generate samples
        synthetic_data = np.zeros((n_samples, n_features))

        for node in topo_order:
            if node == intervene_idx:
                # Intervention: set to fixed value
                synthetic_data[:, node] = intervene_value
            elif node in models:
                # Generate based on parents
                parents = models[node]['parents']
                model = models[node]['model']

                parent_data = synthetic_data[:, parents]
                predictions = model.predict(parent_data)

                # Add noise
                noise_std = np.std(X[:, node] - model.predict(X[:, parents]))
                noise = np.random.normal(0, noise_std, n_samples)

                synthetic_data[:, node] = predictions + noise
            else:
                # Root node: sample from marginal distribution
                synthetic_data[:, node] = np.random.choice(X[:, node], n_samples)

        return synthetic_data

    def visualize_causal_graph(self, layout: str = 'spring') -> plt.Figure:
        """
        Visualize the causal graph.

        Args:
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')

        Returns:
            Matplotlib figure
        """
        if self.causal_graph is None:
            raise ValueError("Causal graph not initialized. Run pc_algorithm first.")

        dag = self.causal_graph
        variable_names = self.variable_names
        n_features = len(dag)

        # Create NetworkX graph
        G = nx.DiGraph()

        for i in range(n_features):
            G.add_node(i, label=variable_names[i])

        for i in range(n_features):
            for j in range(n_features):
                if dag[i, j] == 1 and dag[j, i] == 0:
                    # Directed edge
                    G.add_edge(i, j)
                elif i < j and dag[i, j] == 1 and dag[j, i] == 1:
                    # Undirected edge (add only once)
                    G.add_edge(i, j, style='undirected')

        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        # Draw
        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=2000, ax=ax, edgecolors='black', linewidths=2)

        # Draw directed edges
        directed_edges = [(i, j) for i, j in G.edges()
                         if G.get_edge_data(i, j).get('style') != 'undirected']
        nx.draw_networkx_edges(G, pos, edgelist=directed_edges,
                              edge_color='black', arrows=True,
                              arrowsize=25, arrowstyle='->', width=2, ax=ax)

        # Draw undirected edges
        undirected_edges = [(i, j) for i, j in G.edges()
                           if G.get_edge_data(i, j).get('style') == 'undirected']
        nx.draw_networkx_edges(G, pos, edgelist=undirected_edges,
                              edge_color='gray', arrows=False, width=2,
                              style='dashed', ax=ax)

        # Draw labels
        labels = {i: variable_names[i] for i in range(n_features)}
        nx.draw_networkx_labels(G, pos, labels, font_size=12,
                               font_weight='bold', ax=ax)

        ax.set_title('Causal Graph (DAG)', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()

        return fig

    def visualize_markov_blanket(self, markov_blanket: Dict) -> plt.Figure:
        """
        Visualize Markov blanket.

        Args:
            markov_blanket: Dictionary from find_markov_blanket

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create graph
        G = nx.DiGraph()

        target = markov_blanket['target']
        parents = markov_blanket['parents']
        children = markov_blanket['children']
        spouses = markov_blanket['spouses']

        # Add nodes
        G.add_node(target)
        for p in parents:
            G.add_node(p)
            G.add_edge(p, target)

        for c in children:
            G.add_node(c)
            G.add_edge(target, c)

        for s in spouses:
            G.add_node(s)

        # Add edges from spouses to children
        for s in spouses:
            for c in children:
                # Check if this spouse is parent of this child
                G.add_edge(s, c)

        # Layout
        pos = nx.spring_layout(G, seed=42, k=3)

        # Color nodes
        node_colors = []
        for node in G.nodes():
            if node == target:
                node_colors.append('red')
            elif node in parents:
                node_colors.append('lightgreen')
            elif node in children:
                node_colors.append('lightblue')
            elif node in spouses:
                node_colors.append('yellow')
            else:
                node_colors.append('lightgray')

        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=2500, ax=ax, edgecolors='black', linewidths=2)
        nx.draw_networkx_edges(G, pos, edge_color='black',
                              arrows=True, arrowsize=20, arrowstyle='->', width=2, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='Target'),
            Patch(facecolor='lightgreen', edgecolor='black', label='Parents'),
            Patch(facecolor='lightblue', edgecolor='black', label='Children'),
            Patch(facecolor='yellow', edgecolor='black', label='Spouses')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

        ax.set_title(f'Markov Blanket of {target}', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        return fig


def demo():
    """Demonstrate causal discovery toolkit."""
    np.random.seed(42)

    print("Causal Discovery Toolkit Demo")
    print("=" * 80)

    # 1. Generate synthetic causal data
    print("\n1. Generating Synthetic Causal Data")
    print("-" * 80)

    n_samples = 500

    # Create causal structure: X0 -> X1 -> X3, X2 -> X3, X1 -> X4
    X0 = np.random.randn(n_samples)
    X1 = 2.0 * X0 + np.random.randn(n_samples) * 0.5
    X2 = np.random.randn(n_samples)
    X3 = 1.5 * X1 + 1.0 * X2 + np.random.randn(n_samples) * 0.5
    X4 = 0.8 * X1 + np.random.randn(n_samples) * 0.5

    X = np.column_stack([X0, X1, X2, X3, X4])
    variable_names = ['X0', 'X1', 'X2', 'X3', 'X4']

    print(f"Generated {n_samples} samples with 5 variables")
    print("True causal structure:")
    print("  X0 -> X1 -> X3")
    print("  X2 -> X3")
    print("  X1 -> X4")

    toolkit = CausalDiscoveryToolkit(significance_level=0.05)

    # 2. Partial Correlation Test
    print("\n2. Partial Correlation Test")
    print("-" * 80)
    print("Testing: X0 ⊥ X3 | X1")
    corr, p_value = toolkit.partial_correlation_test(X, 0, 3, [1])
    print(f"Partial correlation: {corr:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Independent: {p_value > toolkit.significance_level}")

    # 3. Conditional Independence Test
    print("\n3. Conditional Independence Tests")
    print("-" * 80)

    tests = [
        (0, 1, [], "X0 ⊥ X1"),
        (0, 3, [1], "X0 ⊥ X3 | X1"),
        (1, 2, [], "X1 ⊥ X2"),
        (1, 3, [2], "X1 ⊥ X3 | X2"),
    ]

    for i, j, cond, desc in tests:
        is_independent = toolkit.conditional_independence_test(X, i, j, cond)
        print(f"{desc}: {'Independent' if is_independent else 'Dependent'}")

    # 4. PC Algorithm
    print("\n4. PC Algorithm for Causal Discovery")
    print("-" * 80)
    pc_result = toolkit.pc_algorithm(X, variable_names=variable_names, max_conditioning_size=2)
    print(f"Skeleton edges: {pc_result['n_skeleton_edges']}")
    print(f"Directed edges: {pc_result['n_directed_edges']}")
    print(f"Undirected edges: {pc_result['n_undirected_edges']}")

    print("\nDiscovered causal relationships:")
    dag = pc_result['causal_graph']
    for i in range(len(variable_names)):
        for j in range(len(variable_names)):
            if dag[i, j] == 1 and dag[j, i] == 0:
                print(f"  {variable_names[i]} -> {variable_names[j]}")

    # 5. Markov Blanket Discovery
    print("\n5. Markov Blanket Discovery")
    print("-" * 80)
    mb_result = toolkit.find_markov_blanket(X, target_idx=3, variable_names=variable_names)
    print(f"Target variable: {mb_result['target']}")
    print(f"Parents: {mb_result['parents']}")
    print(f"Children: {mb_result['children']}")
    print(f"Spouses: {mb_result['spouses']}")
    print(f"Markov Blanket: {mb_result['markov_blanket']}")

    # 6. D-separation Test
    print("\n6. D-separation Tests")
    print("-" * 80)
    d_sep_tests = [
        (0, 3, [1], "X0 ⊥ X3 | X1"),
        (0, 2, [], "X0 ⊥ X2"),
        (1, 2, [], "X1 ⊥ X2"),
    ]

    for i, j, cond, desc in d_sep_tests:
        is_d_separated = toolkit.test_d_separation(i, j, cond)
        print(f"{desc}: {'D-separated' if is_d_separated else 'Not d-separated'}")

    # 7. Pairwise Causal Direction
    print("\n7. Pairwise Causal Direction Tests")
    print("-" * 80)

    pairs = [(0, 1), (1, 3), (2, 3)]
    for i, j in pairs:
        direction_result = toolkit.pairwise_causal_direction(X, i, j)
        print(f"{variable_names[i]} vs {variable_names[j]}:")
        print(f"  Predicted direction: {direction_result['direction']}")
        print(f"  Confidence: {direction_result['confidence']:.4f}")

    # 8. Validate Causal Graph
    print("\n8. Causal Graph Validation")
    print("-" * 80)
    validation = toolkit.validate_causal_graph(X)
    print(f"Is DAG: {validation['is_dag']}")
    print(f"Number of nodes: {validation['n_nodes']}")
    print(f"Number of edges: {validation['n_edges']}")
    print(f"Violation rate: {validation['violation_rate']:.4f}")
    print(f"Violations: {validation['n_violations']} / {validation['n_tests']}")

    # 9. Simulate Intervention
    print("\n9. Intervention Simulation")
    print("-" * 80)
    print("Simulating intervention: do(X1 = 5.0)")

    # Original distribution of X3
    original_X3_mean = np.mean(X[:, 3])
    original_X3_std = np.std(X[:, 3])

    # Intervene on X1
    intervened_data = toolkit.simulate_intervention(X, intervene_idx=1, intervene_value=5.0, n_samples=500)

    intervened_X3_mean = np.mean(intervened_data[:, 3])
    intervened_X3_std = np.std(intervened_data[:, 3])

    print(f"Original X3: mean={original_X3_mean:.4f}, std={original_X3_std:.4f}")
    print(f"After do(X1=5.0) X3: mean={intervened_X3_mean:.4f}, std={intervened_X3_std:.4f}")
    print(f"Change in X3 mean: {intervened_X3_mean - original_X3_mean:.4f}")

    # 10. Visualizations
    print("\n10. Generating Visualizations")
    print("-" * 80)

    # Causal graph
    fig1 = toolkit.visualize_causal_graph(layout='spring')
    fig1.savefig('causal_graph.png', dpi=300, bbox_inches='tight')
    print("✓ Saved causal_graph.png")
    plt.close()

    # Markov blanket
    fig2 = toolkit.visualize_markov_blanket(mb_result)
    fig2.savefig('markov_blanket.png', dpi=300, bbox_inches='tight')
    print("✓ Saved markov_blanket.png")
    plt.close()

    # Intervention comparison
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

    # X3 distribution before intervention
    axes[0].hist(X[:, 3], bins=30, alpha=0.7, color='blue', edgecolor='black', density=True)
    axes[0].axvline(original_X3_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {original_X3_mean:.2f}')
    axes[0].set_xlabel('X3', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Original X3 Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # X3 distribution after intervention
    axes[1].hist(intervened_data[:, 3], bins=30, alpha=0.7, color='green', edgecolor='black', density=True)
    axes[1].axvline(intervened_X3_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {intervened_X3_mean:.2f}')
    axes[1].set_xlabel('X3', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('X3 Distribution after do(X1=5.0)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig3.savefig('intervention_effect.png', dpi=300, bbox_inches='tight')
    print("✓ Saved intervention_effect.png")
    plt.close()

    # Correlation matrix
    fig4, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original correlations
    original_corr = np.corrcoef(X.T)
    sns.heatmap(original_corr, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, xticklabels=variable_names, yticklabels=variable_names,
               ax=axes[0], cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Original Correlations', fontsize=13, fontweight='bold')

    # After intervention correlations
    intervened_corr = np.corrcoef(intervened_data.T)
    sns.heatmap(intervened_corr, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, xticklabels=variable_names, yticklabels=variable_names,
               ax=axes[1], cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Correlations after do(X1=5.0)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig4.savefig('causal_correlations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved causal_correlations.png")
    plt.close()

    print("\n" + "=" * 80)
    print("✓ Causal Discovery Demo Complete!")
    print("=" * 80)


if __name__ == '__main__':
    demo()
