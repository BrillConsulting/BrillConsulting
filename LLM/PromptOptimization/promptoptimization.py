"""
PromptOptimization - Production-Ready Prompt Optimization System
Author: BrillConsulting
Description: Comprehensive system for optimizing LLM prompts using multiple strategies
"""

import json
import random
import logging
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import pickle


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PromptCandidate:
    """Represents a candidate prompt with metadata"""
    prompt: str
    generation: int
    parent_id: Optional[str] = None
    mutation_type: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def id(self) -> str:
        """Generate unique ID based on prompt content"""
        return hashlib.md5(self.prompt.encode()).hexdigest()[:12]


@dataclass
class EvaluationResult:
    """Stores evaluation metrics for a prompt"""
    prompt_id: str
    accuracy: float
    latency: float
    token_count: int
    coherence: float
    relevance: float
    diversity: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def weighted_score(self) -> float:
        """Calculate weighted composite score"""
        return (
            0.35 * self.accuracy +
            0.15 * (1.0 - min(self.latency / 10.0, 1.0)) +
            0.15 * (1.0 - min(self.token_count / 1000.0, 1.0)) +
            0.20 * self.coherence +
            0.10 * self.relevance +
            0.05 * self.diversity
        )

    def pareto_dominates(self, other: 'EvaluationResult') -> bool:
        """Check if this result Pareto dominates another"""
        better_in_one = False
        for metric in ['accuracy', 'coherence', 'relevance', 'diversity']:
            if getattr(self, metric) < getattr(other, metric):
                return False
            if getattr(self, metric) > getattr(other, metric):
                better_in_one = True

        # Lower is better for these
        if self.latency > other.latency or self.token_count > other.token_count:
            return False
        if self.latency < other.latency or self.token_count < other.token_count:
            better_in_one = True

        return better_in_one


class PromptEvaluator:
    """Evaluates prompts using multiple metrics"""

    def __init__(self, evaluation_fn: Optional[Callable] = None):
        """
        Args:
            evaluation_fn: Custom function to evaluate prompts
                          Should return Dict with metrics
        """
        self.evaluation_fn = evaluation_fn
        self.history: List[EvaluationResult] = []

    def evaluate(self, candidate: PromptCandidate,
                 test_cases: Optional[List[Dict]] = None) -> EvaluationResult:
        """
        Evaluate a prompt candidate

        Args:
            candidate: PromptCandidate to evaluate
            test_cases: Optional test cases for evaluation

        Returns:
            EvaluationResult with all metrics
        """
        if self.evaluation_fn:
            metrics = self.evaluation_fn(candidate.prompt, test_cases)
        else:
            # Default evaluation (simulated)
            metrics = self._default_evaluation(candidate.prompt)

        result = EvaluationResult(
            prompt_id=candidate.id,
            accuracy=metrics.get('accuracy', 0.0),
            latency=metrics.get('latency', 0.0),
            token_count=metrics.get('token_count', len(candidate.prompt.split())),
            coherence=metrics.get('coherence', 0.0),
            relevance=metrics.get('relevance', 0.0),
            diversity=metrics.get('diversity', 0.0)
        )

        self.history.append(result)
        return result

    def _default_evaluation(self, prompt: str) -> Dict[str, float]:
        """Default evaluation using heuristics"""
        words = prompt.split()

        # Simulate metrics based on prompt characteristics
        accuracy = min(1.0, len(words) / 100.0 + random.uniform(0.5, 0.9))
        latency = len(words) * 0.01 + random.uniform(0.1, 0.5)
        coherence = 0.7 + random.uniform(0, 0.3)
        relevance = 0.6 + random.uniform(0, 0.4)
        diversity = min(1.0, len(set(words)) / len(words))

        return {
            'accuracy': accuracy,
            'latency': latency,
            'token_count': len(words),
            'coherence': coherence,
            'relevance': relevance,
            'diversity': diversity
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of evaluations"""
        if not self.history:
            return {}

        scores = [r.weighted_score for r in self.history]
        accuracies = [r.accuracy for r in self.history]
        latencies = [r.latency for r in self.history]

        return {
            'total_evaluations': len(self.history),
            'avg_score': np.mean(scores),
            'best_score': np.max(scores),
            'avg_accuracy': np.mean(accuracies),
            'avg_latency': np.mean(latencies),
            'improvement': scores[-1] - scores[0] if len(scores) > 1 else 0
        }


class GeneticPromptOptimizer:
    """Genetic algorithm for prompt optimization"""

    def __init__(self,
                 population_size: int = 20,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 elite_size: int = 2):
        """
        Args:
            population_size: Number of prompts in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of top prompts to preserve
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.generation = 0

        # Mutation operators
        self.mutation_ops = [
            self._mutate_add_word,
            self._mutate_remove_word,
            self._mutate_replace_word,
            self._mutate_reorder,
            self._mutate_expand_instruction
        ]

    def initialize_population(self, seed_prompt: str) -> List[PromptCandidate]:
        """Create initial population from seed prompt"""
        population = [PromptCandidate(seed_prompt, 0)]

        # Generate variations
        for i in range(self.population_size - 1):
            mutated = self._mutate(seed_prompt)
            population.append(PromptCandidate(mutated, 0))

        logger.info(f"Initialized population with {len(population)} candidates")
        return population

    def evolve(self,
               population: List[PromptCandidate],
               fitness_scores: Dict[str, float]) -> List[PromptCandidate]:
        """
        Evolve population to next generation

        Args:
            population: Current population
            fitness_scores: Mapping of prompt_id to fitness score

        Returns:
            New population
        """
        self.generation += 1

        # Sort by fitness
        sorted_pop = sorted(
            population,
            key=lambda x: fitness_scores.get(x.id, 0),
            reverse=True
        )

        # Elite selection
        new_population = sorted_pop[:self.elite_size]

        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1, parent2 = self._select_parents(sorted_pop, fitness_scores)

            # Crossover
            if random.random() < self.crossover_rate:
                child_prompt = self._crossover(parent1.prompt, parent2.prompt)
                mutation_type = 'crossover'
                parent_id = parent1.id
            else:
                child_prompt = parent1.prompt
                mutation_type = 'clone'
                parent_id = parent1.id

            # Mutation
            if random.random() < self.mutation_rate:
                child_prompt = self._mutate(child_prompt)
                mutation_type = 'mutation'

            candidate = PromptCandidate(
                prompt=child_prompt,
                generation=self.generation,
                parent_id=parent_id,
                mutation_type=mutation_type
            )
            new_population.append(candidate)

        logger.info(f"Generation {self.generation}: Created {len(new_population)} offspring")
        return new_population

    def _select_parents(self,
                       population: List[PromptCandidate],
                       fitness_scores: Dict[str, float]) -> Tuple[PromptCandidate, PromptCandidate]:
        """Tournament selection"""
        tournament_size = 3

        def tournament():
            contestants = random.sample(population, min(tournament_size, len(population)))
            return max(contestants, key=lambda x: fitness_scores.get(x.id, 0))

        return tournament(), tournament()

    def _crossover(self, prompt1: str, prompt2: str) -> str:
        """Perform crossover between two prompts"""
        words1 = prompt1.split()
        words2 = prompt2.split()

        if len(words1) < 2 or len(words2) < 2:
            return prompt1

        # Two-point crossover
        point1 = random.randint(0, len(words1) - 1)
        point2 = random.randint(0, len(words2) - 1)

        child = words1[:point1] + words2[point2:]
        return ' '.join(child)

    def _mutate(self, prompt: str) -> str:
        """Apply random mutation to prompt"""
        mutation_op = random.choice(self.mutation_ops)
        return mutation_op(prompt)

    def _mutate_add_word(self, prompt: str) -> str:
        """Add a word to the prompt"""
        enhancers = [
            "detailed", "comprehensive", "specific", "clear", "precise",
            "thorough", "accurate", "well-structured", "concise", "effective"
        ]
        words = prompt.split()
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, random.choice(enhancers))
        return ' '.join(words)

    def _mutate_remove_word(self, prompt: str) -> str:
        """Remove a word from the prompt"""
        words = prompt.split()
        if len(words) > 5:
            words.pop(random.randint(0, len(words) - 1))
        return ' '.join(words)

    def _mutate_replace_word(self, prompt: str) -> str:
        """Replace a word in the prompt"""
        synonyms = {
            "create": ["generate", "produce", "develop", "build"],
            "analyze": ["examine", "evaluate", "assess", "review"],
            "explain": ["describe", "clarify", "elaborate", "detail"],
            "provide": ["give", "supply", "offer", "present"]
        }

        words = prompt.split()
        if not words:
            return prompt

        pos = random.randint(0, len(words) - 1)
        word_lower = words[pos].lower()

        if word_lower in synonyms:
            words[pos] = random.choice(synonyms[word_lower])

        return ' '.join(words)

    def _mutate_reorder(self, prompt: str) -> str:
        """Reorder sentences in the prompt"""
        sentences = prompt.split('. ')
        if len(sentences) > 1:
            random.shuffle(sentences)
        return '. '.join(sentences)

    def _mutate_expand_instruction(self, prompt: str) -> str:
        """Add instructional phrases"""
        expansions = [
            "Step by step,",
            "Think carefully and",
            "First,",
            "In detail,",
            "Systematically,"
        ]
        return f"{random.choice(expansions)} {prompt}"


class GradientPromptOptimizer:
    """Gradient-based prompt optimization using feedback"""

    def __init__(self, learning_rate: float = 0.1):
        """
        Args:
            learning_rate: Step size for optimization
        """
        self.learning_rate = learning_rate
        self.iteration = 0
        self.feedback_history: List[Dict] = []

    def optimize_step(self,
                     prompt: str,
                     evaluation_result: EvaluationResult,
                     target_metrics: Dict[str, float]) -> str:
        """
        Perform one optimization step based on gradient approximation

        Args:
            prompt: Current prompt
            evaluation_result: Evaluation of current prompt
            target_metrics: Desired metric values

        Returns:
            Improved prompt
        """
        self.iteration += 1

        # Calculate metric gaps
        gaps = {
            'accuracy': target_metrics.get('accuracy', 1.0) - evaluation_result.accuracy,
            'coherence': target_metrics.get('coherence', 1.0) - evaluation_result.coherence,
            'relevance': target_metrics.get('relevance', 1.0) - evaluation_result.relevance
        }

        # Apply modifications based on gaps
        modifications = []

        if gaps['accuracy'] > 0.1:
            modifications.append(("accuracy", "Add more specific instructions"))
        if gaps['coherence'] > 0.1:
            modifications.append(("coherence", "Improve structure and flow"))
        if gaps['relevance'] > 0.1:
            modifications.append(("relevance", "Focus on key requirements"))

        # Apply strongest modification
        if modifications:
            metric, action = max(modifications, key=lambda x: gaps[x[0]])
            improved_prompt = self._apply_modification(prompt, metric, action)
        else:
            improved_prompt = self._fine_tune(prompt)

        self.feedback_history.append({
            'iteration': self.iteration,
            'gaps': gaps,
            'prompt_length': len(prompt),
            'score': evaluation_result.weighted_score
        })

        logger.info(f"Gradient step {self.iteration}: Applied {len(modifications)} modifications")
        return improved_prompt

    def _apply_modification(self, prompt: str, metric: str, action: str) -> str:
        """Apply specific modification to improve a metric"""
        if metric == 'accuracy':
            return f"{prompt}\n\nBe specific and accurate in your response."
        elif metric == 'coherence':
            return f"Structure your response clearly.\n\n{prompt}"
        elif metric == 'relevance':
            return f"Focus on the most relevant aspects.\n\n{prompt}"
        return prompt

    def _fine_tune(self, prompt: str) -> str:
        """Fine-tune prompt when close to target"""
        tuning_phrases = [
            "Additionally,",
            "Furthermore,",
            "It is important to",
            "Make sure to",
            "Consider"
        ]
        return f"{prompt} {random.choice(tuning_phrases)} refine your approach."


class MultiObjectiveOptimizer:
    """Optimizer for multi-objective prompt optimization using Pareto fronts"""

    def __init__(self):
        self.pareto_front: List[Tuple[PromptCandidate, EvaluationResult]] = []
        self.archive: List[Tuple[PromptCandidate, EvaluationResult]] = []

    def update_pareto_front(self,
                           candidate: PromptCandidate,
                           result: EvaluationResult) -> bool:
        """
        Update Pareto front with new candidate

        Returns:
            True if candidate is non-dominated
        """
        # Check if dominated by existing solutions
        is_dominated = False
        to_remove = []

        for i, (front_candidate, front_result) in enumerate(self.pareto_front):
            if front_result.pareto_dominates(result):
                is_dominated = True
                break
            elif result.pareto_dominates(front_result):
                to_remove.append(i)

        # Remove dominated solutions
        for i in reversed(to_remove):
            removed = self.pareto_front.pop(i)
            self.archive.append(removed)

        # Add if non-dominated
        if not is_dominated:
            self.pareto_front.append((candidate, result))
            logger.info(f"Pareto front updated: {len(self.pareto_front)} solutions")
            return True

        return False

    def get_best_compromise(self) -> Optional[Tuple[PromptCandidate, EvaluationResult]]:
        """Get best compromise solution from Pareto front"""
        if not self.pareto_front:
            return None

        # Return solution with best weighted score
        return max(self.pareto_front, key=lambda x: x[1].weighted_score)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about optimization"""
        return {
            'pareto_front_size': len(self.pareto_front),
            'total_evaluated': len(self.pareto_front) + len(self.archive),
            'best_accuracy': max([r.accuracy for _, r in self.pareto_front]) if self.pareto_front else 0,
            'best_latency': min([r.latency for _, r in self.pareto_front]) if self.pareto_front else 0
        }


class PerformanceTracker:
    """Tracks and analyzes optimization performance over time"""

    def __init__(self, save_dir: str = "./optimization_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: List[Dict] = []
        self.best_prompts: List[Tuple[PromptCandidate, EvaluationResult]] = []
        self.optimization_runs: List[Dict] = []

    def log_iteration(self,
                     iteration: int,
                     candidate: PromptCandidate,
                     result: EvaluationResult,
                     metadata: Optional[Dict] = None):
        """Log a single optimization iteration"""
        entry = {
            'iteration': iteration,
            'prompt_id': candidate.id,
            'prompt': candidate.prompt,
            'generation': candidate.generation,
            'weighted_score': result.weighted_score,
            'accuracy': result.accuracy,
            'latency': result.latency,
            'coherence': result.coherence,
            'timestamp': datetime.now().isoformat()
        }

        if metadata:
            entry.update(metadata)

        self.metrics_history.append(entry)

        # Update best prompts
        if not self.best_prompts or result.weighted_score > self.best_prompts[0][1].weighted_score:
            self.best_prompts.insert(0, (candidate, result))
            self.best_prompts = self.best_prompts[:10]  # Keep top 10

    def save_results(self, run_name: str):
        """Save optimization results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.save_dir / f"{run_name}_{timestamp}"
        run_dir.mkdir(exist_ok=True)

        # Save metrics history
        with open(run_dir / "metrics_history.json", 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        # Save best prompts
        best_prompts_data = [
            {
                'prompt': candidate.prompt,
                'score': result.weighted_score,
                'accuracy': result.accuracy,
                'latency': result.latency
            }
            for candidate, result in self.best_prompts
        ]
        with open(run_dir / "best_prompts.json", 'w') as f:
            json.dump(best_prompts_data, f, indent=2)

        logger.info(f"Results saved to {run_dir}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.metrics_history:
            return {}

        scores = [m['weighted_score'] for m in self.metrics_history]
        accuracies = [m['accuracy'] for m in self.metrics_history]

        return {
            'total_iterations': len(self.metrics_history),
            'best_score': max(scores),
            'final_score': scores[-1],
            'improvement': scores[-1] - scores[0],
            'avg_accuracy': np.mean(accuracies),
            'convergence_iteration': self._find_convergence(),
            'best_prompt': self.best_prompts[0][0].prompt if self.best_prompts else None
        }

    def _find_convergence(self) -> Optional[int]:
        """Find iteration where optimization converged"""
        if len(self.metrics_history) < 10:
            return None

        scores = [m['weighted_score'] for m in self.metrics_history]
        window_size = 10
        threshold = 0.01

        for i in range(len(scores) - window_size):
            window = scores[i:i+window_size]
            if max(window) - min(window) < threshold:
                return i

        return None


class PromptOptimizationSystem:
    """
    Main system orchestrating all optimization strategies
    """

    def __init__(self,
                 evaluation_fn: Optional[Callable] = None,
                 save_dir: str = "./optimization_results"):
        """
        Args:
            evaluation_fn: Custom evaluation function
            save_dir: Directory to save results
        """
        self.evaluator = PromptEvaluator(evaluation_fn)
        self.genetic_optimizer = GeneticPromptOptimizer()
        self.gradient_optimizer = GradientPromptOptimizer()
        self.multi_objective = MultiObjectiveOptimizer()
        self.tracker = PerformanceTracker(save_dir)

        logger.info("PromptOptimizationSystem initialized")

    def optimize(self,
                seed_prompt: str,
                strategy: str = 'genetic',
                max_iterations: int = 50,
                target_metrics: Optional[Dict[str, float]] = None,
                test_cases: Optional[List[Dict]] = None) -> Tuple[str, EvaluationResult]:
        """
        Optimize a prompt using specified strategy

        Args:
            seed_prompt: Initial prompt to optimize
            strategy: Optimization strategy ('genetic', 'gradient', 'hybrid', 'multi_objective')
            max_iterations: Maximum number of iterations
            target_metrics: Target metric values for gradient optimization
            test_cases: Test cases for evaluation

        Returns:
            Tuple of (best_prompt, best_result)
        """
        logger.info(f"Starting optimization with strategy: {strategy}")
        logger.info(f"Seed prompt: {seed_prompt[:100]}...")

        if strategy == 'genetic':
            return self._optimize_genetic(seed_prompt, max_iterations, test_cases)
        elif strategy == 'gradient':
            return self._optimize_gradient(seed_prompt, max_iterations, target_metrics, test_cases)
        elif strategy == 'hybrid':
            return self._optimize_hybrid(seed_prompt, max_iterations, target_metrics, test_cases)
        elif strategy == 'multi_objective':
            return self._optimize_multi_objective(seed_prompt, max_iterations, test_cases)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _optimize_genetic(self,
                         seed_prompt: str,
                         max_iterations: int,
                         test_cases: Optional[List[Dict]]) -> Tuple[str, EvaluationResult]:
        """Genetic algorithm optimization"""
        population = self.genetic_optimizer.initialize_population(seed_prompt)

        best_candidate = None
        best_result = None

        for iteration in range(max_iterations):
            # Evaluate population
            fitness_scores = {}
            for candidate in population:
                result = self.evaluator.evaluate(candidate, test_cases)
                fitness_scores[candidate.id] = result.weighted_score

                self.tracker.log_iteration(iteration, candidate, result,
                                         {'strategy': 'genetic'})

                # Update best
                if best_result is None or result.weighted_score > best_result.weighted_score:
                    best_candidate = candidate
                    best_result = result

            # Evolve
            if iteration < max_iterations - 1:
                population = self.genetic_optimizer.evolve(population, fitness_scores)

            logger.info(f"Iteration {iteration}: Best score = {best_result.weighted_score:.4f}")

        return best_candidate.prompt, best_result

    def _optimize_gradient(self,
                          seed_prompt: str,
                          max_iterations: int,
                          target_metrics: Optional[Dict[str, float]],
                          test_cases: Optional[List[Dict]]) -> Tuple[str, EvaluationResult]:
        """Gradient-based optimization"""
        if target_metrics is None:
            target_metrics = {'accuracy': 0.95, 'coherence': 0.90, 'relevance': 0.90}

        current_prompt = seed_prompt
        candidate = PromptCandidate(current_prompt, 0)
        best_result = self.evaluator.evaluate(candidate, test_cases)

        for iteration in range(max_iterations):
            # Evaluate current prompt
            candidate = PromptCandidate(current_prompt, iteration)
            result = self.evaluator.evaluate(candidate, test_cases)

            self.tracker.log_iteration(iteration, candidate, result,
                                     {'strategy': 'gradient'})

            # Update best
            if result.weighted_score > best_result.weighted_score:
                best_result = result
                best_prompt = current_prompt

            # Optimize
            if iteration < max_iterations - 1:
                current_prompt = self.gradient_optimizer.optimize_step(
                    current_prompt, result, target_metrics
                )

            logger.info(f"Iteration {iteration}: Score = {result.weighted_score:.4f}")

        return best_prompt, best_result

    def _optimize_hybrid(self,
                        seed_prompt: str,
                        max_iterations: int,
                        target_metrics: Optional[Dict[str, float]],
                        test_cases: Optional[List[Dict]]) -> Tuple[str, EvaluationResult]:
        """Hybrid optimization combining genetic and gradient methods"""
        # Phase 1: Genetic exploration (60% of iterations)
        genetic_iterations = int(max_iterations * 0.6)
        logger.info(f"Phase 1: Genetic exploration ({genetic_iterations} iterations)")
        best_prompt, best_result = self._optimize_genetic(
            seed_prompt, genetic_iterations, test_cases
        )

        # Phase 2: Gradient refinement (40% of iterations)
        gradient_iterations = max_iterations - genetic_iterations
        logger.info(f"Phase 2: Gradient refinement ({gradient_iterations} iterations)")
        best_prompt, best_result = self._optimize_gradient(
            best_prompt, gradient_iterations, target_metrics, test_cases
        )

        return best_prompt, best_result

    def _optimize_multi_objective(self,
                                  seed_prompt: str,
                                  max_iterations: int,
                                  test_cases: Optional[List[Dict]]) -> Tuple[str, EvaluationResult]:
        """Multi-objective optimization using Pareto fronts"""
        population = self.genetic_optimizer.initialize_population(seed_prompt)

        for iteration in range(max_iterations):
            # Evaluate and update Pareto front
            fitness_scores = {}
            for candidate in population:
                result = self.evaluator.evaluate(candidate, test_cases)
                fitness_scores[candidate.id] = result.weighted_score

                self.multi_objective.update_pareto_front(candidate, result)
                self.tracker.log_iteration(iteration, candidate, result,
                                         {'strategy': 'multi_objective'})

            # Evolve
            if iteration < max_iterations - 1:
                population = self.genetic_optimizer.evolve(population, fitness_scores)

            stats = self.multi_objective.get_statistics()
            logger.info(f"Iteration {iteration}: Pareto front size = {stats['pareto_front_size']}")

        # Return best compromise solution
        best_candidate, best_result = self.multi_objective.get_best_compromise()
        return best_candidate.prompt, best_result

    def get_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        return {
            'tracker_summary': self.tracker.get_summary(),
            'evaluator_stats': self.evaluator.get_statistics(),
            'multi_objective_stats': self.multi_objective.get_statistics(),
            'timestamp': datetime.now().isoformat()
        }

    def save_results(self, run_name: str = "optimization_run"):
        """Save all results"""
        self.tracker.save_results(run_name)

        # Save Pareto front
        if self.multi_objective.pareto_front:
            pareto_data = [
                {
                    'prompt': candidate.prompt,
                    'accuracy': result.accuracy,
                    'latency': result.latency,
                    'score': result.weighted_score
                }
                for candidate, result in self.multi_objective.pareto_front
            ]

            pareto_file = self.tracker.save_dir / f"{run_name}_pareto_front.json"
            with open(pareto_file, 'w') as f:
                json.dump(pareto_data, f, indent=2)

    def execute(self):
        """Execute demo optimization"""
        print(f"\n{'='*70}")
        print("PromptOptimization System - Production Demo")
        print(f"{'='*70}\n")

        # Demo prompt
        seed_prompt = "Explain the concept to the user."

        print(f"Seed Prompt: {seed_prompt}\n")

        # Run optimization with different strategies
        strategies = ['genetic', 'gradient', 'multi_objective']

        for strategy in strategies:
            print(f"\n{'-'*70}")
            print(f"Running {strategy.upper()} optimization...")
            print(f"{'-'*70}")

            optimized_prompt, result = self.optimize(
                seed_prompt=seed_prompt,
                strategy=strategy,
                max_iterations=20
            )

            print(f"\nOptimized Prompt:\n{optimized_prompt}\n")
            print(f"Results:")
            print(f"  - Weighted Score: {result.weighted_score:.4f}")
            print(f"  - Accuracy: {result.accuracy:.4f}")
            print(f"  - Latency: {result.latency:.4f}s")
            print(f"  - Coherence: {result.coherence:.4f}")
            print(f"  - Relevance: {result.relevance:.4f}")

        # Generate and save report
        print(f"\n{'='*70}")
        report = self.get_report()
        print("\nOptimization Report:")
        print(json.dumps(report, indent=2))

        self.save_results("demo_run")
        print(f"\nResults saved to: {self.tracker.save_dir}")
        print(f"\n✓ Executed at {datetime.now()}")

        return {"status": "complete", "report": report}


if __name__ == "__main__":
    system = PromptOptimizationSystem()
    system.execute()
