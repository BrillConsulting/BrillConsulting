"""
Sequential Analysis Toolkit

A comprehensive toolkit for sequential analysis including Sequential Probability
Ratio Test (SPRT), Multi-Armed Bandits, online A/B testing, and change point
detection algorithms.

Author: Brill Consulting
Date: 2025-11-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Optional, Tuple, List, Dict, Any, Union
from scipy import stats
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SPRTResult:
    """Container for SPRT test results."""
    decision: str  # 'H0', 'H1', or 'continue'
    n_samples: int
    log_likelihood_ratio: float
    threshold_upper: float
    threshold_lower: float


@dataclass
class BanditArm:
    """Container for bandit arm statistics."""
    arm_id: int
    n_pulls: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    successes: int = 0  # For beta distribution (Thompson Sampling)
    failures: int = 0


@dataclass
class ChangePoint:
    """Container for detected change point."""
    index: int
    score: float
    confidence: float


class SequentialAnalysisToolkit:
    """
    Comprehensive sequential analysis toolkit.

    Provides methods for sequential decision making, online testing,
    and adaptive algorithms including SPRT, multi-armed bandits,
    and change point detection.

    Attributes:
        random_state: Random seed for reproducibility
        verbose: Whether to print progress messages

    Example:
        >>> sat = SequentialAnalysisToolkit(random_state=42)
        >>> result = sat.sprt(data, mu0=0, mu1=1, sigma=1)
        >>> print(f"Decision: {result.decision}")
    """

    def __init__(self, random_state: Optional[int] = None, verbose: bool = True):
        """
        Initialize the SequentialAnalysisToolkit.

        Args:
            random_state: Random seed for reproducibility
            verbose: Whether to print progress messages
        """
        self.random_state = random_state
        self.verbose = verbose
        if random_state is not None:
            np.random.seed(random_state)

    def sprt(
        self,
        data: np.ndarray,
        mu0: float,
        mu1: float,
        sigma: float,
        alpha: float = 0.05,
        beta: float = 0.05,
        sequential: bool = True
    ) -> SPRTResult:
        """
        Sequential Probability Ratio Test (SPRT).

        Tests H0: data ~ N(mu0, sigma^2) vs H1: data ~ N(mu1, sigma^2)

        Args:
            data: Observed data
            mu0: Mean under null hypothesis
            mu1: Mean under alternative hypothesis
            sigma: Known standard deviation
            alpha: Type I error probability
            beta: Type II error probability
            sequential: If True, stop at first decision; else process all data

        Returns:
            SPRTResult with decision and statistics
        """
        # Calculate thresholds
        A = (1 - beta) / alpha
        B = beta / (1 - alpha)

        threshold_upper = np.log(A)
        threshold_lower = np.log(B)

        log_likelihood_ratio = 0.0
        decision = 'continue'

        for i, x in enumerate(data):
            # Update log likelihood ratio
            # log(L(mu1) / L(mu0))
            log_likelihood_ratio += (mu1 - mu0) * (x - (mu0 + mu1) / 2) / (sigma ** 2)

            # Check thresholds
            if log_likelihood_ratio >= threshold_upper:
                decision = 'H1'
                if sequential:
                    break
            elif log_likelihood_ratio <= threshold_lower:
                decision = 'H0'
                if sequential:
                    break

        n_samples = i + 1 if sequential and decision != 'continue' else len(data)

        if self.verbose:
            print(f"SPRT Result: {decision} after {n_samples} samples")
            print(f"Log-likelihood ratio: {log_likelihood_ratio:.4f}")
            print(f"Thresholds: [{threshold_lower:.4f}, {threshold_upper:.4f}]")

        return SPRTResult(
            decision=decision,
            n_samples=n_samples,
            log_likelihood_ratio=log_likelihood_ratio,
            threshold_upper=threshold_upper,
            threshold_lower=threshold_lower
        )

    def epsilon_greedy_bandit(
        self,
        reward_functions: List[Callable],
        n_rounds: int = 1000,
        epsilon: float = 0.1
    ) -> Tuple[List[BanditArm], List[int], List[float]]:
        """
        Epsilon-greedy multi-armed bandit algorithm.

        Args:
            reward_functions: List of functions that return rewards
            n_rounds: Number of rounds to play
            epsilon: Exploration probability

        Returns:
            Tuple of (arms, selections, rewards)
        """
        n_arms = len(reward_functions)
        arms = [BanditArm(arm_id=i) for i in range(n_arms)]

        selections = []
        rewards = []

        for round_num in range(n_rounds):
            # Epsilon-greedy selection
            if np.random.random() < epsilon or round_num < n_arms:
                # Explore: random arm
                selected_arm = np.random.randint(n_arms)
            else:
                # Exploit: best arm
                mean_rewards = [arm.mean_reward for arm in arms]
                selected_arm = np.argmax(mean_rewards)

            # Pull arm and observe reward
            reward = reward_functions[selected_arm]()

            # Update statistics
            arms[selected_arm].n_pulls += 1
            arms[selected_arm].total_reward += reward
            arms[selected_arm].mean_reward = (
                arms[selected_arm].total_reward / arms[selected_arm].n_pulls
            )

            selections.append(selected_arm)
            rewards.append(reward)

        if self.verbose:
            print("Epsilon-Greedy Bandit Results:")
            for arm in arms:
                print(f"  Arm {arm.arm_id}: pulls={arm.n_pulls}, "
                      f"mean_reward={arm.mean_reward:.4f}")

        return arms, selections, rewards

    def ucb_bandit(
        self,
        reward_functions: List[Callable],
        n_rounds: int = 1000,
        c: float = 2.0
    ) -> Tuple[List[BanditArm], List[int], List[float]]:
        """
        Upper Confidence Bound (UCB) multi-armed bandit algorithm.

        Args:
            reward_functions: List of functions that return rewards
            n_rounds: Number of rounds to play
            c: Exploration parameter

        Returns:
            Tuple of (arms, selections, rewards)
        """
        n_arms = len(reward_functions)
        arms = [BanditArm(arm_id=i) for i in range(n_arms)]

        selections = []
        rewards = []

        for round_num in range(n_rounds):
            # UCB selection
            if round_num < n_arms:
                # Initialize: pull each arm once
                selected_arm = round_num
            else:
                # Select arm with highest UCB
                ucb_values = []
                for arm in arms:
                    if arm.n_pulls == 0:
                        ucb = float('inf')
                    else:
                        ucb = arm.mean_reward + c * np.sqrt(
                            np.log(round_num + 1) / arm.n_pulls
                        )
                    ucb_values.append(ucb)

                selected_arm = np.argmax(ucb_values)

            # Pull arm and observe reward
            reward = reward_functions[selected_arm]()

            # Update statistics
            arms[selected_arm].n_pulls += 1
            arms[selected_arm].total_reward += reward
            arms[selected_arm].mean_reward = (
                arms[selected_arm].total_reward / arms[selected_arm].n_pulls
            )

            selections.append(selected_arm)
            rewards.append(reward)

        if self.verbose:
            print("UCB Bandit Results:")
            for arm in arms:
                print(f"  Arm {arm.arm_id}: pulls={arm.n_pulls}, "
                      f"mean_reward={arm.mean_reward:.4f}")

        return arms, selections, rewards

    def thompson_sampling_bandit(
        self,
        reward_functions: List[Callable],
        n_rounds: int = 1000,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ) -> Tuple[List[BanditArm], List[int], List[float]]:
        """
        Thompson Sampling multi-armed bandit algorithm (for Bernoulli rewards).

        Args:
            reward_functions: List of functions that return binary rewards
            n_rounds: Number of rounds to play
            prior_alpha: Prior alpha for Beta distribution
            prior_beta: Prior beta for Beta distribution

        Returns:
            Tuple of (arms, selections, rewards)
        """
        n_arms = len(reward_functions)
        arms = [BanditArm(arm_id=i, successes=int(prior_alpha),
                         failures=int(prior_beta)) for i in range(n_arms)]

        selections = []
        rewards = []

        for round_num in range(n_rounds):
            # Thompson Sampling: sample from posterior
            sampled_values = []
            for arm in arms:
                sampled_value = np.random.beta(arm.successes, arm.failures)
                sampled_values.append(sampled_value)

            selected_arm = np.argmax(sampled_values)

            # Pull arm and observe reward
            reward = reward_functions[selected_arm]()

            # Update statistics
            arms[selected_arm].n_pulls += 1
            arms[selected_arm].total_reward += reward
            arms[selected_arm].mean_reward = (
                arms[selected_arm].total_reward / arms[selected_arm].n_pulls
            )

            # Update Beta distribution parameters
            if reward > 0.5:  # Bernoulli success
                arms[selected_arm].successes += 1
            else:
                arms[selected_arm].failures += 1

            selections.append(selected_arm)
            rewards.append(reward)

        if self.verbose:
            print("Thompson Sampling Bandit Results:")
            for arm in arms:
                print(f"  Arm {arm.arm_id}: pulls={arm.n_pulls}, "
                      f"mean_reward={arm.mean_reward:.4f}, "
                      f"Beta({arm.successes}, {arm.failures})")

        return arms, selections, rewards

    def contextual_bandit(
        self,
        contexts: np.ndarray,
        reward_functions: List[Callable],
        n_rounds: Optional[int] = None,
        learning_rate: float = 0.1,
        epsilon: float = 0.1
    ) -> Tuple[np.ndarray, List[int], List[float]]:
        """
        Linear contextual bandit (epsilon-greedy with linear models).

        Args:
            contexts: Context vectors (n_rounds, n_features)
            reward_functions: List of reward functions (one per arm)
            n_rounds: Number of rounds (uses len(contexts) if None)
            learning_rate: Learning rate for weight updates
            epsilon: Exploration probability

        Returns:
            Tuple of (weights, selections, rewards)
        """
        if n_rounds is None:
            n_rounds = len(contexts)

        n_arms = len(reward_functions)
        n_features = contexts.shape[1]

        # Initialize weights for each arm
        weights = np.zeros((n_arms, n_features))

        selections = []
        rewards = []

        for round_num in range(n_rounds):
            context = contexts[round_num % len(contexts)]

            # Epsilon-greedy with linear prediction
            if np.random.random() < epsilon:
                selected_arm = np.random.randint(n_arms)
            else:
                # Predict expected reward for each arm
                predicted_rewards = [weights[i] @ context for i in range(n_arms)]
                selected_arm = np.argmax(predicted_rewards)

            # Observe reward
            reward = reward_functions[selected_arm]()

            # Update weights (gradient descent)
            prediction = weights[selected_arm] @ context
            error = reward - prediction
            weights[selected_arm] += learning_rate * error * context

            selections.append(selected_arm)
            rewards.append(reward)

        if self.verbose:
            print("Contextual Bandit Results:")
            for i in range(n_arms):
                pulls = selections.count(i)
                if pulls > 0:
                    mean_reward = np.mean([r for s, r in zip(selections, rewards) if s == i])
                    print(f"  Arm {i}: pulls={pulls}, mean_reward={mean_reward:.4f}")

        return weights, selections, rewards

    def online_ab_test(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        alpha: float = 0.05,
        min_samples: int = 100,
        check_frequency: int = 10
    ) -> Dict[str, Any]:
        """
        Online A/B testing with sequential analysis and early stopping.

        Args:
            data_a: Stream of data from variant A
            data_b: Stream of data from variant B
            alpha: Significance level
            min_samples: Minimum samples before testing
            check_frequency: How often to check for stopping

        Returns:
            Dictionary with test results
        """
        n_samples = min(len(data_a), len(data_b))
        stopped = False
        stop_index = n_samples

        p_values = []
        effect_sizes = []

        for i in range(min_samples, n_samples, check_frequency):
            # Get samples up to current point
            samples_a = data_a[:i]
            samples_b = data_b[:i]

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(samples_a, samples_b)
            p_values.append(p_value)

            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(samples_a) - np.mean(samples_b)
            pooled_std = np.sqrt(
                (np.var(samples_a) + np.var(samples_b)) / 2
            )
            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
            effect_sizes.append(effect_size)

            # Check for early stopping
            if p_value < alpha:
                stopped = True
                stop_index = i
                if self.verbose:
                    print(f"Early stopping at sample {i}")
                    print(f"p-value: {p_value:.4f}")
                    print(f"Effect size: {effect_size:.4f}")
                break

        # Final analysis
        final_samples_a = data_a[:stop_index]
        final_samples_b = data_b[:stop_index]

        t_stat, p_value = stats.ttest_ind(final_samples_a, final_samples_b)

        result = {
            'stopped_early': stopped,
            'n_samples': stop_index,
            'p_value': p_value,
            't_statistic': t_stat,
            'mean_a': np.mean(final_samples_a),
            'mean_b': np.mean(final_samples_b),
            'difference': np.mean(final_samples_a) - np.mean(final_samples_b),
            'effect_size': effect_sizes[-1] if effect_sizes else 0,
            'significant': p_value < alpha,
            'p_value_history': p_values,
            'effect_size_history': effect_sizes
        }

        if self.verbose:
            print("\nOnline A/B Test Results:")
            print(f"  Stopped early: {result['stopped_early']}")
            print(f"  Samples used: {result['n_samples']}")
            print(f"  Mean A: {result['mean_a']:.4f}")
            print(f"  Mean B: {result['mean_b']:.4f}")
            print(f"  Difference: {result['difference']:.4f}")
            print(f"  p-value: {result['p_value']:.4f}")
            print(f"  Significant: {result['significant']}")

        return result

    def cusum_change_detection(
        self,
        data: np.ndarray,
        threshold: float = 5.0,
        drift: float = 0.5,
        baseline_mean: Optional[float] = None
    ) -> List[ChangePoint]:
        """
        CUSUM (Cumulative Sum) change point detection.

        Args:
            data: Time series data
            threshold: Detection threshold
            drift: Allowance for small shifts
            baseline_mean: Expected mean (uses data mean if None)

        Returns:
            List of detected change points
        """
        if baseline_mean is None:
            baseline_mean = np.mean(data[:min(100, len(data))])

        # CUSUM statistics
        cusum_pos = 0
        cusum_neg = 0

        change_points = []
        cusum_history = []

        for i, x in enumerate(data):
            # Update CUSUM
            cusum_pos = max(0, cusum_pos + (x - baseline_mean) - drift)
            cusum_neg = max(0, cusum_neg - (x - baseline_mean) - drift)

            cusum_history.append(max(cusum_pos, cusum_neg))

            # Check for change point
            if cusum_pos > threshold or cusum_neg > threshold:
                change_points.append(ChangePoint(
                    index=i,
                    score=max(cusum_pos, cusum_neg),
                    confidence=min(1.0, max(cusum_pos, cusum_neg) / threshold)
                ))

                # Reset CUSUM
                cusum_pos = 0
                cusum_neg = 0

        if self.verbose:
            print(f"CUSUM detected {len(change_points)} change points")
            for cp in change_points[:5]:
                print(f"  Index {cp.index}: score={cp.score:.2f}, "
                      f"confidence={cp.confidence:.2%}")

        return change_points

    def bayesian_change_detection(
        self,
        data: np.ndarray,
        hazard_rate: float = 0.01,
        prior_mean: float = 0.0,
        prior_std: float = 1.0
    ) -> Tuple[np.ndarray, List[ChangePoint]]:
        """
        Bayesian online change point detection.

        Args:
            data: Time series data
            hazard_rate: Prior probability of change point
            prior_mean: Prior mean
            prior_std: Prior standard deviation

        Returns:
            Tuple of (change_probabilities, change_points)
        """
        n = len(data)

        # Run length (time since last change point)
        max_run_length = 200
        run_length_probs = np.zeros(max_run_length)
        run_length_probs[0] = 1.0

        change_probabilities = np.zeros(n)
        change_points = []

        # Sufficient statistics for Gaussian
        means = np.zeros(max_run_length)
        variances = np.ones(max_run_length) * prior_std ** 2
        counts = np.zeros(max_run_length)

        for t, x in enumerate(data):
            # Prediction: probability of each run length
            pred_probs = np.zeros(max_run_length)

            for r in range(len(run_length_probs)):
                if run_length_probs[r] > 1e-10:
                    # Predictive probability
                    mean = means[r]
                    var = variances[r] + prior_std ** 2

                    pred_prob = stats.norm.pdf(x, mean, np.sqrt(var))
                    pred_probs[r] = run_length_probs[r] * pred_prob

            # Normalize
            if pred_probs.sum() > 0:
                pred_probs /= pred_probs.sum()

            # Update run lengths
            new_run_length_probs = np.zeros(max_run_length)

            # Growth: no change point
            new_run_length_probs[1:] = pred_probs[:-1] * (1 - hazard_rate)

            # Change point
            new_run_length_probs[0] = pred_probs.sum() * hazard_rate

            # Update sufficient statistics
            for r in range(max_run_length):
                if new_run_length_probs[r] > 1e-10:
                    counts[r] += 1
                    delta = x - means[r]
                    means[r] += delta / counts[r]
                    variances[r] = (variances[r] * (counts[r] - 1) + delta ** 2) / counts[r]

            run_length_probs = new_run_length_probs

            # Change point probability
            change_prob = run_length_probs[0]
            change_probabilities[t] = change_prob

            # Detect change points
            if change_prob > 0.5:
                change_points.append(ChangePoint(
                    index=t,
                    score=change_prob,
                    confidence=change_prob
                ))

                # Reset
                run_length_probs = np.zeros(max_run_length)
                run_length_probs[0] = 1.0
                means = np.zeros(max_run_length)
                variances = np.ones(max_run_length) * prior_std ** 2
                counts = np.zeros(max_run_length)

        if self.verbose:
            print(f"Bayesian method detected {len(change_points)} change points")

        return change_probabilities, change_points

    def calculate_regret(
        self,
        selections: List[int],
        rewards: List[float],
        true_means: List[float]
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate cumulative regret for bandit algorithm.

        Args:
            selections: List of selected arms
            rewards: List of observed rewards
            true_means: True mean rewards of each arm

        Returns:
            Tuple of (cumulative_regret, total_regret)
        """
        optimal_reward = max(true_means)

        regret = []
        cumulative = 0

        for selection, reward in zip(selections, rewards):
            instantaneous_regret = optimal_reward - true_means[selection]
            cumulative += instantaneous_regret
            regret.append(cumulative)

        return np.array(regret), cumulative

    def plot_bandit_performance(
        self,
        results: Dict[str, Tuple[List[int], List[float], List[float]]],
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Plot performance comparison of bandit algorithms.

        Args:
            results: Dictionary mapping algorithm names to (selections, rewards, true_means)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        for name, (selections, rewards, true_means) in results.items():
            # Cumulative reward
            cumulative_rewards = np.cumsum(rewards)
            axes[0, 0].plot(cumulative_rewards, label=name, linewidth=2, alpha=0.8)

            # Cumulative regret
            regret, _ = self.calculate_regret(selections, rewards, true_means)
            axes[0, 1].plot(regret, label=name, linewidth=2, alpha=0.8)

            # Average reward (moving average)
            window = 50
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(moving_avg, label=name, linewidth=2, alpha=0.8)

            # Arm selection frequency
            n_arms = len(true_means)
            arm_counts = [selections.count(i) for i in range(n_arms)]
            axes[1, 1].bar(np.arange(n_arms) + 0.2 * list(results.keys()).index(name),
                          arm_counts, width=0.2, label=name, alpha=0.7)

        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].set_title('Cumulative Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Cumulative Regret')
        axes[0, 1].set_title('Cumulative Regret')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].set_title(f'Moving Average Reward (window={window})')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].set_xlabel('Arm')
        axes[1, 1].set_ylabel('Number of Pulls')
        axes[1, 1].set_title('Arm Selection Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_change_detection(
        self,
        data: np.ndarray,
        change_points: List[ChangePoint],
        title: str = 'Change Point Detection',
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot time series with detected change points.

        Args:
            data: Time series data
            change_points: List of detected change points
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot time series
        ax.plot(data, linewidth=2, label='Data', color='blue', alpha=0.7)

        # Mark change points
        for cp in change_points:
            ax.axvline(x=cp.index, color='red', linestyle='--', alpha=0.6,
                      linewidth=2)
            ax.text(cp.index, ax.get_ylim()[1] * 0.95,
                   f'{cp.confidence:.0%}',
                   rotation=90, verticalalignment='top', fontsize=9)

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_sprt_path(
        self,
        data: np.ndarray,
        mu0: float,
        mu1: float,
        sigma: float,
        alpha: float = 0.05,
        beta: float = 0.05,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot SPRT decision path.

        Args:
            data: Observed data
            mu0: Mean under H0
            mu1: Mean under H1
            sigma: Standard deviation
            alpha: Type I error
            beta: Type II error
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Calculate path
        A = (1 - beta) / alpha
        B = beta / (1 - alpha)

        threshold_upper = np.log(A)
        threshold_lower = np.log(B)

        log_ratios = []
        lr = 0

        for x in data:
            lr += (mu1 - mu0) * (x - (mu0 + mu1) / 2) / (sigma ** 2)
            log_ratios.append(lr)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(log_ratios, linewidth=2, label='Log-Likelihood Ratio', color='blue')
        ax.axhline(y=threshold_upper, color='red', linestyle='--',
                  linewidth=2, label=f'Upper threshold (accept H1)')
        ax.axhline(y=threshold_lower, color='green', linestyle='--',
                  linewidth=2, label=f'Lower threshold (accept H0)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Mark decision point
        if log_ratios[-1] >= threshold_upper:
            decision_text = 'Decision: H1'
            color = 'red'
        elif log_ratios[-1] <= threshold_lower:
            decision_text = 'Decision: H0'
            color = 'green'
        else:
            decision_text = 'Decision: Continue'
            color = 'orange'

        ax.text(0.02, 0.98, decision_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.5),
               fontsize=12, fontweight='bold')

        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Log-Likelihood Ratio')
        ax.set_title('Sequential Probability Ratio Test (SPRT)')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig


def demo():
    """
    Demonstrate the SequentialAnalysisToolkit capabilities.
    """
    print("=" * 80)
    print("SEQUENTIAL ANALYSIS TOOLKIT DEMO")
    print("=" * 80)

    sat = SequentialAnalysisToolkit(random_state=42, verbose=True)

    # 1. Sequential Probability Ratio Test
    print("\n1. SEQUENTIAL PROBABILITY RATIO TEST (SPRT)")
    print("-" * 80)

    # Generate data from alternative hypothesis
    np.random.seed(42)
    data_h1 = np.random.normal(loc=1.0, scale=1.0, size=100)

    sprt_result = sat.sprt(
        data_h1,
        mu0=0.0,
        mu1=1.0,
        sigma=1.0,
        alpha=0.05,
        beta=0.05
    )

    # 2. Multi-Armed Bandits
    print("\n2. MULTI-ARMED BANDITS")
    print("-" * 80)

    # Define arms with different reward distributions
    true_means = [0.3, 0.5, 0.6, 0.4]
    reward_functions = [
        lambda mean=m: np.random.normal(mean, 0.1) for m in true_means
    ]

    print("\n  True mean rewards:", true_means)
    print(f"  Optimal arm: {np.argmax(true_means)} with mean {max(true_means)}")

    # Epsilon-Greedy
    print("\n  a) Epsilon-Greedy:")
    arms_eg, selections_eg, rewards_eg = sat.epsilon_greedy_bandit(
        reward_functions, n_rounds=1000, epsilon=0.1
    )

    # UCB
    print("\n  b) Upper Confidence Bound:")
    arms_ucb, selections_ucb, rewards_ucb = sat.ucb_bandit(
        reward_functions, n_rounds=1000, c=2.0
    )

    # Thompson Sampling (for Bernoulli rewards)
    print("\n  c) Thompson Sampling:")
    bernoulli_probs = [0.3, 0.5, 0.6, 0.4]
    bernoulli_functions = [
        lambda p=prob: 1.0 if np.random.random() < p else 0.0
        for prob in bernoulli_probs
    ]

    arms_ts, selections_ts, rewards_ts = sat.thompson_sampling_bandit(
        bernoulli_functions, n_rounds=1000
    )

    # Calculate regrets
    print("\n  Regret Analysis:")
    regret_eg, total_eg = sat.calculate_regret(selections_eg, rewards_eg, true_means)
    regret_ucb, total_ucb = sat.calculate_regret(selections_ucb, rewards_ucb, true_means)
    regret_ts, total_ts = sat.calculate_regret(selections_ts, rewards_ts, bernoulli_probs)

    print(f"    Epsilon-Greedy total regret: {total_eg:.2f}")
    print(f"    UCB total regret: {total_ucb:.2f}")
    print(f"    Thompson Sampling total regret: {total_ts:.2f}")

    # 3. Contextual Bandit
    print("\n3. CONTEXTUAL BANDIT")
    print("-" * 80)

    # Generate contexts
    n_contexts = 1000
    n_features = 5
    contexts = np.random.randn(n_contexts, n_features)

    # Define context-dependent reward functions
    true_weights = [
        np.array([0.5, -0.3, 0.2, 0.1, -0.1]),
        np.array([0.2, 0.4, -0.2, 0.3, 0.2]),
        np.array([-0.3, 0.5, 0.4, -0.2, 0.3])
    ]

    contextual_rewards = [
        lambda w=weights: np.clip(
            w @ contexts[np.random.randint(n_contexts)] + np.random.normal(0, 0.1),
            0, 1
        )
        for weights in true_weights
    ]

    weights_cb, selections_cb, rewards_cb = sat.contextual_bandit(
        contexts, contextual_rewards, n_rounds=1000, epsilon=0.1
    )

    print(f"  Total reward: {sum(rewards_cb):.2f}")
    print(f"  Average reward: {np.mean(rewards_cb):.4f}")

    # 4. Online A/B Testing
    print("\n4. ONLINE A/B TESTING WITH EARLY STOPPING")
    print("-" * 80)

    # Generate A/B test data
    data_a = np.random.normal(loc=0.5, scale=0.2, size=500)
    data_b = np.random.normal(loc=0.55, scale=0.2, size=500)  # Slightly better

    ab_result = sat.online_ab_test(
        data_a, data_b, alpha=0.05, min_samples=100, check_frequency=20
    )

    # 5. Change Point Detection
    print("\n5. CHANGE POINT DETECTION")
    print("-" * 80)

    # Generate time series with change points
    np.random.seed(42)
    segment1 = np.random.normal(0, 1, 200)
    segment2 = np.random.normal(2, 1, 150)
    segment3 = np.random.normal(-1, 1, 200)
    time_series = np.concatenate([segment1, segment2, segment3])

    print("\n  a) CUSUM:")
    change_points_cusum = sat.cusum_change_detection(
        time_series, threshold=5.0, drift=0.5
    )

    print("\n  b) Bayesian Online Change Point Detection:")
    change_probs, change_points_bayes = sat.bayesian_change_detection(
        time_series, hazard_rate=0.01
    )
    print(f"  Detected {len(change_points_bayes)} change points")

    # 6. Visualizations
    print("\n6. CREATING VISUALIZATIONS")
    print("-" * 80)

    # SPRT path
    fig1 = sat.plot_sprt_path(data_h1, mu0=0.0, mu1=1.0, sigma=1.0)
    plt.savefig('/tmp/sprt_path.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/sprt_path.png")
    plt.close()

    # Bandit performance comparison
    bandit_results = {
        'Epsilon-Greedy': (selections_eg, rewards_eg, true_means),
        'UCB': (selections_ucb, rewards_ucb, true_means),
        'Thompson': (selections_ts, rewards_ts, bernoulli_probs)
    }
    fig2 = sat.plot_bandit_performance(bandit_results)
    plt.savefig('/tmp/bandit_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/bandit_comparison.png")
    plt.close()

    # Change point detection
    fig3 = sat.plot_change_detection(
        time_series, change_points_cusum,
        title='CUSUM Change Point Detection'
    )
    plt.savefig('/tmp/change_detection_cusum.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/change_detection_cusum.png")
    plt.close()

    # A/B test evolution
    fig4, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ab_result['p_value_history'], linewidth=2, label='p-value')
    ax.axhline(y=0.05, color='red', linestyle='--', label='Significance threshold')
    ax.set_xlabel('Check Number')
    ax.set_ylabel('p-value')
    ax.set_title('Online A/B Test: p-value Evolution')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/ab_test_evolution.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/ab_test_evolution.png")
    plt.close()

    # Regret comparison
    fig5, ax = plt.subplots(figsize=(10, 6))
    ax.plot(regret_eg, label='Epsilon-Greedy', linewidth=2)
    ax.plot(regret_ucb, label='UCB', linewidth=2)
    ax.plot(regret_ts, label='Thompson Sampling', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Bandit Algorithms: Regret Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/regret_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/regret_comparison.png")
    plt.close()

    # Bayesian change detection
    fig6, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(time_series, linewidth=2, color='blue', alpha=0.7)
    for cp in change_points_bayes:
        ax1.axvline(x=cp.index, color='red', linestyle='--', alpha=0.6)
    ax1.set_ylabel('Value')
    ax1.set_title('Time Series with Detected Change Points')
    ax1.grid(alpha=0.3)

    ax2.plot(change_probs, linewidth=2, color='orange')
    ax2.axhline(y=0.5, color='red', linestyle='--', label='Detection threshold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Change Probability')
    ax2.set_title('Bayesian Change Point Probability')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/bayesian_change_detection.png', dpi=150, bbox_inches='tight')
    print("  Saved: /tmp/bayesian_change_detection.png")
    plt.close()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nKey Insights:")
    print("1. SPRT enables early stopping in sequential hypothesis testing")
    print("2. Epsilon-Greedy balances exploration and exploitation")
    print("3. UCB uses confidence bounds to guide exploration")
    print("4. Thompson Sampling uses Bayesian posterior sampling")
    print("5. Contextual bandits leverage context information")
    print("6. Online A/B testing can stop early when significance is reached")
    print("7. CUSUM detects shifts in mean efficiently")
    print("8. Bayesian change point detection provides probabilistic estimates")
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    demo()
