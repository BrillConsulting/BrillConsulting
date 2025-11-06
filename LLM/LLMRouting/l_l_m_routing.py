"""
LLMRouting - Production-Ready Intelligent LLM Router
Author: BrillConsulting
Description: Advanced routing system for optimal LLM model selection with cost optimization,
            latency-based routing, load balancing, and fallback strategies
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Model capability categories"""
    SIMPLE_QA = "simple_qa"
    COMPLEX_REASONING = "complex_reasoning"
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    ANALYSIS = "analysis"
    GENERAL = "general"


class RoutingStrategy(Enum):
    """Available routing strategies"""
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"
    ROUND_ROBIN = "round_robin"
    WEIGHTED_RANDOM = "weighted_random"
    LEAST_LOADED = "least_loaded"


@dataclass
class ModelConfig:
    """Configuration for an LLM model"""
    model_id: str
    name: str
    provider: str
    cost_per_1k_tokens: float
    max_tokens: int
    avg_latency_ms: float
    capabilities: List[ModelCapability]
    quality_score: float = 0.8
    max_concurrent_requests: int = 100
    timeout_seconds: int = 30
    is_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRequest:
    """Request to route an LLM query"""
    query: str
    capability: ModelCapability = ModelCapability.GENERAL
    max_tokens: int = 1000
    priority: int = 1
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    required_quality: float = 0.7
    max_latency_ms: Optional[float] = None
    max_cost: Optional[float] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_model: ModelConfig
    confidence: float
    expected_cost: float
    expected_latency_ms: float
    fallback_models: List[ModelConfig]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Result of query execution"""
    success: bool
    model_id: str
    actual_latency_ms: float
    actual_cost: float
    tokens_used: int
    response: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMetrics:
    """Track model performance metrics"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.costs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.success_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.active_requests: Dict[str, int] = defaultdict(int)
        self.total_requests: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def record_execution(self, model_id: str, result: ExecutionResult):
        """Record execution metrics"""
        with self.lock:
            self.latencies[model_id].append(result.actual_latency_ms)
            self.costs[model_id].append(result.actual_cost)
            self.success_rates[model_id].append(1.0 if result.success else 0.0)
            self.total_requests[model_id] += 1

    def get_avg_latency(self, model_id: str) -> float:
        """Get average latency for a model"""
        with self.lock:
            latencies = list(self.latencies[model_id])
            return np.mean(latencies) if latencies else 0.0

    def get_avg_cost(self, model_id: str) -> float:
        """Get average cost for a model"""
        with self.lock:
            costs = list(self.costs[model_id])
            return np.mean(costs) if costs else 0.0

    def get_success_rate(self, model_id: str) -> float:
        """Get success rate for a model"""
        with self.lock:
            rates = list(self.success_rates[model_id])
            return np.mean(rates) if rates else 1.0

    def get_load(self, model_id: str) -> float:
        """Get current load (0-1) for a model"""
        with self.lock:
            return self.active_requests.get(model_id, 0)

    def increment_active(self, model_id: str):
        """Increment active request count"""
        with self.lock:
            self.active_requests[model_id] += 1

    def decrement_active(self, model_id: str):
        """Decrement active request count"""
        with self.lock:
            self.active_requests[model_id] = max(0, self.active_requests[model_id] - 1)

    def get_statistics(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a model"""
        return {
            'model_id': model_id,
            'avg_latency_ms': self.get_avg_latency(model_id),
            'avg_cost': self.get_avg_cost(model_id),
            'success_rate': self.get_success_rate(model_id),
            'active_requests': self.get_load(model_id),
            'total_requests': self.total_requests[model_id]
        }


class ModelSelector(ABC):
    """Abstract base class for model selection strategies"""

    @abstractmethod
    def select_model(
        self,
        request: QueryRequest,
        available_models: List[ModelConfig],
        metrics: PerformanceMetrics
    ) -> Tuple[ModelConfig, float]:
        """Select best model and return confidence score"""
        pass


class CostOptimizedSelector(ModelSelector):
    """Select model optimizing for lowest cost"""

    def select_model(
        self,
        request: QueryRequest,
        available_models: List[ModelConfig],
        metrics: PerformanceMetrics
    ) -> Tuple[ModelConfig, float]:
        filtered = [m for m in available_models if request.capability in m.capabilities]
        if not filtered:
            filtered = available_models

        # Calculate actual costs based on metrics
        model_scores = []
        for model in filtered:
            actual_cost = metrics.get_avg_cost(model.model_id) or model.cost_per_1k_tokens
            success_rate = metrics.get_success_rate(model.model_id)

            # Penalize unreliable models
            effective_cost = actual_cost / max(success_rate, 0.1)
            model_scores.append((model, effective_cost))

        # Select cheapest
        best_model = min(model_scores, key=lambda x: x[1])[0]
        confidence = 0.9 if len(filtered) > 1 else 0.7

        return best_model, confidence


class LatencyOptimizedSelector(ModelSelector):
    """Select model optimizing for lowest latency"""

    def select_model(
        self,
        request: QueryRequest,
        available_models: List[ModelConfig],
        metrics: PerformanceMetrics
    ) -> Tuple[ModelConfig, float]:
        filtered = [m for m in available_models if request.capability in m.capabilities]
        if not filtered:
            filtered = available_models

        # Calculate actual latencies based on metrics
        model_scores = []
        for model in filtered:
            actual_latency = metrics.get_avg_latency(model.model_id) or model.avg_latency_ms
            load_factor = metrics.get_load(model.model_id) / max(model.max_concurrent_requests, 1)

            # Account for current load
            effective_latency = actual_latency * (1 + load_factor)
            model_scores.append((model, effective_latency))

        # Select fastest
        best_model = min(model_scores, key=lambda x: x[1])[0]
        confidence = 0.9

        return best_model, confidence


class QualityOptimizedSelector(ModelSelector):
    """Select model optimizing for highest quality"""

    def select_model(
        self,
        request: QueryRequest,
        available_models: List[ModelConfig],
        metrics: PerformanceMetrics
    ) -> Tuple[ModelConfig, float]:
        filtered = [m for m in available_models if request.capability in m.capabilities]
        if not filtered:
            filtered = available_models

        # Calculate quality scores
        model_scores = []
        for model in filtered:
            success_rate = metrics.get_success_rate(model.model_id)
            quality_score = model.quality_score * success_rate
            model_scores.append((model, quality_score))

        # Select highest quality
        best_model = max(model_scores, key=lambda x: x[1])[0]
        confidence = 0.95

        return best_model, confidence


class BalancedSelector(ModelSelector):
    """Select model balancing cost, latency, and quality"""

    def __init__(self, cost_weight: float = 0.3, latency_weight: float = 0.4, quality_weight: float = 0.3):
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.quality_weight = quality_weight

    def select_model(
        self,
        request: QueryRequest,
        available_models: List[ModelConfig],
        metrics: PerformanceMetrics
    ) -> Tuple[ModelConfig, float]:
        filtered = [m for m in available_models if request.capability in m.capabilities]
        if not filtered:
            filtered = available_models

        # Normalize metrics
        costs = []
        latencies = []
        qualities = []

        for model in filtered:
            cost = metrics.get_avg_cost(model.model_id) or model.cost_per_1k_tokens
            latency = metrics.get_avg_latency(model.model_id) or model.avg_latency_ms
            quality = model.quality_score * metrics.get_success_rate(model.model_id)

            costs.append(cost)
            latencies.append(latency)
            qualities.append(quality)

        # Normalize to 0-1 range
        max_cost = max(costs) if costs else 1
        max_latency = max(latencies) if latencies else 1
        max_quality = max(qualities) if qualities else 1

        # Calculate composite scores (lower is better for cost/latency, higher for quality)
        model_scores = []
        for i, model in enumerate(filtered):
            norm_cost = costs[i] / max_cost if max_cost > 0 else 0
            norm_latency = latencies[i] / max_latency if max_latency > 0 else 0
            norm_quality = qualities[i] / max_quality if max_quality > 0 else 0

            # Composite score (minimize cost and latency, maximize quality)
            score = (
                self.cost_weight * (1 - norm_cost) +
                self.latency_weight * (1 - norm_latency) +
                self.quality_weight * norm_quality
            )
            model_scores.append((model, score))

        # Select best balanced option
        best_model = max(model_scores, key=lambda x: x[1])[0]
        confidence = 0.85

        return best_model, confidence


class RoundRobinSelector(ModelSelector):
    """Select models in round-robin fashion"""

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)

    def select_model(
        self,
        request: QueryRequest,
        available_models: List[ModelConfig],
        metrics: PerformanceMetrics
    ) -> Tuple[ModelConfig, float]:
        filtered = [m for m in available_models if request.capability in m.capabilities]
        if not filtered:
            filtered = available_models

        capability_key = request.capability.value
        index = self.counters[capability_key] % len(filtered)
        self.counters[capability_key] += 1

        return filtered[index], 0.6


class LeastLoadedSelector(ModelSelector):
    """Select model with least current load"""

    def select_model(
        self,
        request: QueryRequest,
        available_models: List[ModelConfig],
        metrics: PerformanceMetrics
    ) -> Tuple[ModelConfig, float]:
        filtered = [m for m in available_models if request.capability in m.capabilities]
        if not filtered:
            filtered = available_models

        # Calculate load percentages
        model_loads = []
        for model in filtered:
            load = metrics.get_load(model.model_id)
            load_pct = load / max(model.max_concurrent_requests, 1)
            model_loads.append((model, load_pct))

        # Select least loaded
        best_model = min(model_loads, key=lambda x: x[1])[0]
        confidence = 0.8

        return best_model, confidence


class FallbackStrategy:
    """Manage fallback model selection"""

    def __init__(self, max_fallbacks: int = 3):
        self.max_fallbacks = max_fallbacks

    def get_fallbacks(
        self,
        primary_model: ModelConfig,
        request: QueryRequest,
        all_models: List[ModelConfig],
        metrics: PerformanceMetrics
    ) -> List[ModelConfig]:
        """Get ordered list of fallback models"""
        # Filter out primary model and get compatible models
        candidates = [
            m for m in all_models
            if m.model_id != primary_model.model_id
            and m.is_available
            and request.capability in m.capabilities
        ]

        # Sort by reliability and quality
        scored = []
        for model in candidates:
            success_rate = metrics.get_success_rate(model.model_id)
            score = model.quality_score * success_rate
            scored.append((model, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [model for model, _ in scored[:self.max_fallbacks]]


class RouterEnsemble:
    """Ensemble of multiple routing strategies"""

    def __init__(self, selectors: Dict[RoutingStrategy, ModelSelector], weights: Optional[Dict[RoutingStrategy, float]] = None):
        self.selectors = selectors
        self.weights = weights or {strategy: 1.0 for strategy in selectors.keys()}

    def select_model(
        self,
        request: QueryRequest,
        available_models: List[ModelConfig],
        metrics: PerformanceMetrics
    ) -> Tuple[ModelConfig, float]:
        """Use ensemble voting to select model"""
        votes: Dict[str, float] = defaultdict(float)
        confidences: Dict[str, List[float]] = defaultdict(list)

        # Get votes from all selectors
        for strategy, selector in self.selectors.items():
            weight = self.weights.get(strategy, 1.0)
            model, confidence = selector.select_model(request, available_models, metrics)
            votes[model.model_id] += weight * confidence
            confidences[model.model_id].append(confidence)

        # Select model with highest weighted vote
        best_model_id = max(votes.items(), key=lambda x: x[1])[0]
        best_model = next(m for m in available_models if m.model_id == best_model_id)

        # Average confidence from selectors that chose this model
        avg_confidence = np.mean(confidences[best_model_id])

        return best_model, avg_confidence


class CacheManager:
    """Cache routing decisions for similar queries"""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, Tuple[RoutingDecision, float]] = {}
        self.lock = threading.Lock()

    def _hash_request(self, request: QueryRequest) -> str:
        """Generate hash for request"""
        key_str = f"{request.query[:100]}:{request.capability.value}:{request.strategy.value}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, request: QueryRequest) -> Optional[RoutingDecision]:
        """Get cached routing decision"""
        with self.lock:
            key = self._hash_request(request)
            if key in self.cache:
                decision, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return decision
                else:
                    del self.cache[key]
        return None

    def set(self, request: QueryRequest, decision: RoutingDecision):
        """Cache routing decision"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
                del self.cache[oldest_key]

            key = self._hash_request(request)
            self.cache[key] = (decision, time.time())

    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()


class LoadBalancer:
    """Distribute load across models"""

    def __init__(self, models: List[ModelConfig], metrics: PerformanceMetrics):
        self.models = {m.model_id: m for m in models}
        self.metrics = metrics

    def can_accept_request(self, model_id: str) -> bool:
        """Check if model can accept another request"""
        model = self.models.get(model_id)
        if not model or not model.is_available:
            return False

        current_load = self.metrics.get_load(model_id)
        return current_load < model.max_concurrent_requests

    def acquire(self, model_id: str) -> bool:
        """Acquire slot for request"""
        if self.can_accept_request(model_id):
            self.metrics.increment_active(model_id)
            return True
        return False

    def release(self, model_id: str):
        """Release slot after request"""
        self.metrics.decrement_active(model_id)


class LLMRouter:
    """Main LLM routing engine"""

    def __init__(
        self,
        models: List[ModelConfig],
        enable_caching: bool = True,
        enable_ensemble: bool = False,
        cache_ttl: int = 300
    ):
        self.models = {m.model_id: m for m in models}
        self.metrics = PerformanceMetrics()
        self.load_balancer = LoadBalancer(models, self.metrics)
        self.fallback_strategy = FallbackStrategy()

        # Initialize selectors
        self.selectors = {
            RoutingStrategy.COST_OPTIMIZED: CostOptimizedSelector(),
            RoutingStrategy.LATENCY_OPTIMIZED: LatencyOptimizedSelector(),
            RoutingStrategy.QUALITY_OPTIMIZED: QualityOptimizedSelector(),
            RoutingStrategy.BALANCED: BalancedSelector(),
            RoutingStrategy.ROUND_ROBIN: RoundRobinSelector(),
            RoutingStrategy.LEAST_LOADED: LeastLoadedSelector()
        }

        # Optional ensemble
        self.enable_ensemble = enable_ensemble
        if enable_ensemble:
            self.ensemble = RouterEnsemble(
                {
                    RoutingStrategy.COST_OPTIMIZED: self.selectors[RoutingStrategy.COST_OPTIMIZED],
                    RoutingStrategy.LATENCY_OPTIMIZED: self.selectors[RoutingStrategy.LATENCY_OPTIMIZED],
                    RoutingStrategy.QUALITY_OPTIMIZED: self.selectors[RoutingStrategy.QUALITY_OPTIMIZED]
                },
                weights={
                    RoutingStrategy.COST_OPTIMIZED: 0.3,
                    RoutingStrategy.LATENCY_OPTIMIZED: 0.4,
                    RoutingStrategy.QUALITY_OPTIMIZED: 0.3
                }
            )

        # Optional caching
        self.cache_manager = CacheManager(ttl_seconds=cache_ttl) if enable_caching else None

        logger.info(f"Initialized LLM Router with {len(models)} models")

    def route(self, request: QueryRequest) -> RoutingDecision:
        """Route request to optimal model"""
        # Check cache
        if self.cache_manager:
            cached = self.cache_manager.get(request)
            if cached:
                logger.info(f"Cache hit for request: {request.query[:50]}")
                return cached

        # Get available models
        available = [m for m in self.models.values() if m.is_available]
        if not available:
            raise ValueError("No available models")

        # Filter by constraints
        filtered = self._filter_by_constraints(available, request)
        if not filtered:
            logger.warning("No models match constraints, using all available")
            filtered = available

        # Select model
        if self.enable_ensemble and request.strategy == RoutingStrategy.BALANCED:
            selected_model, confidence = self.ensemble.select_model(request, filtered, self.metrics)
        else:
            selector = self.selectors.get(request.strategy, self.selectors[RoutingStrategy.BALANCED])
            selected_model, confidence = selector.select_model(request, filtered, self.metrics)

        # Get fallback models
        fallbacks = self.fallback_strategy.get_fallbacks(selected_model, request, filtered, self.metrics)

        # Calculate expectations
        expected_cost = (
            self.metrics.get_avg_cost(selected_model.model_id) or
            selected_model.cost_per_1k_tokens * request.max_tokens / 1000
        )
        expected_latency = (
            self.metrics.get_avg_latency(selected_model.model_id) or
            selected_model.avg_latency_ms
        )

        # Create decision
        decision = RoutingDecision(
            selected_model=selected_model,
            confidence=confidence,
            expected_cost=expected_cost,
            expected_latency_ms=expected_latency,
            fallback_models=fallbacks,
            reasoning=f"Selected {selected_model.name} using {request.strategy.value} strategy"
        )

        # Cache decision
        if self.cache_manager:
            self.cache_manager.set(request, decision)

        logger.info(f"Routed to {selected_model.name} (confidence: {confidence:.2f})")
        return decision

    def _filter_by_constraints(self, models: List[ModelConfig], request: QueryRequest) -> List[ModelConfig]:
        """Filter models by request constraints"""
        filtered = models

        # Filter by capability
        if request.capability != ModelCapability.GENERAL:
            filtered = [m for m in filtered if request.capability in m.capabilities]

        # Filter by quality
        filtered = [m for m in filtered if m.quality_score >= request.required_quality]

        # Filter by latency
        if request.max_latency_ms:
            filtered = [
                m for m in filtered
                if self.metrics.get_avg_latency(m.model_id) or m.avg_latency_ms <= request.max_latency_ms
            ]

        # Filter by cost
        if request.max_cost:
            filtered = [
                m for m in filtered
                if (self.metrics.get_avg_cost(m.model_id) or m.cost_per_1k_tokens * request.max_tokens / 1000) <= request.max_cost
            ]

        return filtered

    def execute_with_fallback(
        self,
        request: QueryRequest,
        execution_fn: Callable[[ModelConfig, QueryRequest], ExecutionResult]
    ) -> ExecutionResult:
        """Execute request with automatic fallback on failure"""
        decision = self.route(request)
        models_to_try = [decision.selected_model] + decision.fallback_models

        for i, model in enumerate(models_to_try):
            # Check load balancer
            if not self.load_balancer.acquire(model.model_id):
                logger.warning(f"Model {model.name} at capacity, trying next")
                continue

            try:
                start_time = time.time()
                result = execution_fn(model, request)
                result.actual_latency_ms = (time.time() - start_time) * 1000

                # Record metrics
                self.metrics.record_execution(model.model_id, result)

                if result.success:
                    logger.info(f"Successfully executed on {model.name}")
                    return result
                else:
                    logger.warning(f"Execution failed on {model.name}: {result.error}")

            except Exception as e:
                logger.error(f"Exception executing on {model.name}: {e}")
                result = ExecutionResult(
                    success=False,
                    model_id=model.model_id,
                    actual_latency_ms=(time.time() - start_time) * 1000,
                    actual_cost=0,
                    tokens_used=0,
                    error=str(e)
                )
                self.metrics.record_execution(model.model_id, result)

            finally:
                self.load_balancer.release(model.model_id)

            # Try next model
            if i < len(models_to_try) - 1:
                logger.info(f"Trying fallback model: {models_to_try[i+1].name}")

        # All models failed
        raise RuntimeError("All models failed to execute request")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        stats = {
            'total_models': len(self.models),
            'available_models': sum(1 for m in self.models.values() if m.is_available),
            'model_stats': {}
        }

        for model_id in self.models.keys():
            stats['model_stats'][model_id] = self.metrics.get_statistics(model_id)

        return stats

    def update_model_availability(self, model_id: str, is_available: bool):
        """Update model availability"""
        if model_id in self.models:
            self.models[model_id].is_available = is_available
            logger.info(f"Model {model_id} availability set to {is_available}")

    def add_model(self, model: ModelConfig):
        """Add new model to router"""
        self.models[model.model_id] = model
        logger.info(f"Added model {model.name}")

    def remove_model(self, model_id: str):
        """Remove model from router"""
        if model_id in self.models:
            del self.models[model_id]
            logger.info(f"Removed model {model_id}")


class LLMRoutingManager:
    """High-level manager for LLM routing system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.router = self._initialize_router()
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 10))
        logger.info("LLM Routing Manager initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with sample models"""
        return {
            'enable_caching': True,
            'enable_ensemble': True,
            'cache_ttl': 300,
            'max_workers': 10,
            'models': [
                {
                    'model_id': 'gpt-4',
                    'name': 'GPT-4',
                    'provider': 'openai',
                    'cost_per_1k_tokens': 0.03,
                    'max_tokens': 8192,
                    'avg_latency_ms': 2000,
                    'capabilities': ['complex_reasoning', 'code_generation', 'analysis', 'general'],
                    'quality_score': 0.95,
                    'max_concurrent_requests': 50
                },
                {
                    'model_id': 'gpt-3.5-turbo',
                    'name': 'GPT-3.5 Turbo',
                    'provider': 'openai',
                    'cost_per_1k_tokens': 0.002,
                    'max_tokens': 4096,
                    'avg_latency_ms': 800,
                    'capabilities': ['simple_qa', 'summarization', 'general'],
                    'quality_score': 0.80,
                    'max_concurrent_requests': 100
                },
                {
                    'model_id': 'claude-3-opus',
                    'name': 'Claude 3 Opus',
                    'provider': 'anthropic',
                    'cost_per_1k_tokens': 0.015,
                    'max_tokens': 4096,
                    'avg_latency_ms': 1500,
                    'capabilities': ['complex_reasoning', 'creative_writing', 'analysis', 'general'],
                    'quality_score': 0.93,
                    'max_concurrent_requests': 50
                },
                {
                    'model_id': 'claude-3-sonnet',
                    'name': 'Claude 3 Sonnet',
                    'provider': 'anthropic',
                    'cost_per_1k_tokens': 0.003,
                    'max_tokens': 4096,
                    'avg_latency_ms': 1000,
                    'capabilities': ['simple_qa', 'summarization', 'translation', 'general'],
                    'quality_score': 0.85,
                    'max_concurrent_requests': 100
                },
                {
                    'model_id': 'llama-2-70b',
                    'name': 'Llama 2 70B',
                    'provider': 'meta',
                    'cost_per_1k_tokens': 0.001,
                    'max_tokens': 4096,
                    'avg_latency_ms': 1200,
                    'capabilities': ['simple_qa', 'summarization', 'general'],
                    'quality_score': 0.75,
                    'max_concurrent_requests': 150
                }
            ]
        }

    def _initialize_router(self) -> LLMRouter:
        """Initialize router with configured models"""
        models = []
        for model_config in self.config['models']:
            capabilities = [
                ModelCapability(cap) if isinstance(cap, str) else cap
                for cap in model_config['capabilities']
            ]
            model = ModelConfig(
                model_id=model_config['model_id'],
                name=model_config['name'],
                provider=model_config['provider'],
                cost_per_1k_tokens=model_config['cost_per_1k_tokens'],
                max_tokens=model_config['max_tokens'],
                avg_latency_ms=model_config['avg_latency_ms'],
                capabilities=capabilities,
                quality_score=model_config.get('quality_score', 0.8),
                max_concurrent_requests=model_config.get('max_concurrent_requests', 100)
            )
            models.append(model)

        return LLMRouter(
            models=models,
            enable_caching=self.config.get('enable_caching', True),
            enable_ensemble=self.config.get('enable_ensemble', False),
            cache_ttl=self.config.get('cache_ttl', 300)
        )

    def route_query(
        self,
        query: str,
        capability: str = 'general',
        strategy: str = 'balanced',
        **kwargs
    ) -> Dict[str, Any]:
        """Route a query and return routing decision"""
        request = QueryRequest(
            query=query,
            capability=ModelCapability(capability),
            strategy=RoutingStrategy(strategy),
            **kwargs
        )

        decision = self.router.route(request)

        return {
            'selected_model': {
                'id': decision.selected_model.model_id,
                'name': decision.selected_model.name,
                'provider': decision.selected_model.provider
            },
            'confidence': decision.confidence,
            'expected_cost': decision.expected_cost,
            'expected_latency_ms': decision.expected_latency_ms,
            'fallback_models': [
                {'id': m.model_id, 'name': m.name}
                for m in decision.fallback_models
            ],
            'reasoning': decision.reasoning
        }

    def execute_query(
        self,
        query: str,
        capability: str = 'general',
        strategy: str = 'balanced',
        mock_execution: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute query with automatic routing and fallback"""
        request = QueryRequest(
            query=query,
            capability=ModelCapability(capability),
            strategy=RoutingStrategy(strategy),
            **kwargs
        )

        def mock_execution_fn(model: ModelConfig, req: QueryRequest) -> ExecutionResult:
            """Mock execution function for demo"""
            # Simulate execution
            time.sleep(0.1)  # Simulate latency

            # Simulate occasional failures
            import random
            success = random.random() > 0.1  # 90% success rate

            tokens = req.max_tokens
            cost = model.cost_per_1k_tokens * tokens / 1000

            return ExecutionResult(
                success=success,
                model_id=model.model_id,
                actual_latency_ms=model.avg_latency_ms * random.uniform(0.8, 1.2),
                actual_cost=cost,
                tokens_used=tokens,
                response=f"Mock response from {model.name}" if success else None,
                error="Simulated failure" if not success else None
            )

        if mock_execution:
            result = self.router.execute_with_fallback(request, mock_execution_fn)
        else:
            raise NotImplementedError("Real LLM execution requires API integration")

        return {
            'success': result.success,
            'model_used': result.model_id,
            'actual_latency_ms': result.actual_latency_ms,
            'actual_cost': result.actual_cost,
            'tokens_used': result.tokens_used,
            'response': result.response,
            'error': result.error
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return self.router.get_statistics()

    def benchmark_strategies(
        self,
        test_queries: List[Dict[str, Any]],
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark different routing strategies"""
        strategies = [s.value for s in RoutingStrategy]
        results = {strategy: {
            'total_cost': 0,
            'total_latency': 0,
            'success_count': 0,
            'failure_count': 0
        } for strategy in strategies}

        for iteration in range(iterations):
            for query_config in test_queries:
                for strategy in strategies:
                    try:
                        result = self.execute_query(
                            query=query_config.get('query', 'test query'),
                            capability=query_config.get('capability', 'general'),
                            strategy=strategy,
                            mock_execution=True
                        )

                        results[strategy]['total_cost'] += result['actual_cost']
                        results[strategy]['total_latency'] += result['actual_latency_ms']

                        if result['success']:
                            results[strategy]['success_count'] += 1
                        else:
                            results[strategy]['failure_count'] += 1

                    except Exception as e:
                        results[strategy]['failure_count'] += 1
                        logger.error(f"Benchmark error for {strategy}: {e}")

        # Calculate averages
        total_requests = iterations * len(test_queries)
        for strategy in strategies:
            results[strategy]['avg_cost'] = results[strategy]['total_cost'] / total_requests
            results[strategy]['avg_latency'] = results[strategy]['total_latency'] / total_requests
            results[strategy]['success_rate'] = results[strategy]['success_count'] / total_requests

        return results

    def export_config(self, filepath: str):
        """Export configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        logger.info(f"Configuration exported to {filepath}")

    def import_config(self, filepath: str):
        """Import configuration from file"""
        with open(filepath, 'r') as f:
            self.config = json.load(f)
        self.router = self._initialize_router()
        logger.info(f"Configuration imported from {filepath}")


def main():
    """Demo usage"""
    print("=" * 80)
    print("LLM Routing System - Production Ready")
    print("=" * 80)

    # Initialize manager
    manager = LLMRoutingManager()

    # Example 1: Route a simple query
    print("\n1. Routing a simple Q&A query (cost-optimized):")
    decision = manager.route_query(
        query="What is the capital of France?",
        capability="simple_qa",
        strategy="cost_optimized"
    )
    print(f"   Selected: {decision['selected_model']['name']}")
    print(f"   Confidence: {decision['confidence']:.2f}")
    print(f"   Expected Cost: ${decision['expected_cost']:.4f}")
    print(f"   Expected Latency: {decision['expected_latency_ms']:.0f}ms")

    # Example 2: Route a complex reasoning task
    print("\n2. Routing a complex reasoning task (quality-optimized):")
    decision = manager.route_query(
        query="Explain quantum entanglement and its implications",
        capability="complex_reasoning",
        strategy="quality_optimized"
    )
    print(f"   Selected: {decision['selected_model']['name']}")
    print(f"   Confidence: {decision['confidence']:.2f}")

    # Example 3: Execute with fallback
    print("\n3. Executing query with automatic fallback:")
    result = manager.execute_query(
        query="Write a Python function to sort a list",
        capability="code_generation",
        strategy="balanced",
        mock_execution=True
    )
    print(f"   Success: {result['success']}")
    print(f"   Model: {result['model_used']}")
    print(f"   Latency: {result['actual_latency_ms']:.0f}ms")
    print(f"   Cost: ${result['actual_cost']:.4f}")

    # Example 4: Run multiple queries to build metrics
    print("\n4. Running 10 queries to build performance metrics...")
    for i in range(10):
        manager.execute_query(
            query=f"Test query {i}",
            capability="general",
            strategy="balanced",
            mock_execution=True
        )

    # Example 5: Get statistics
    print("\n5. Performance Statistics:")
    stats = manager.get_statistics()
    print(f"   Total Models: {stats['total_models']}")
    print(f"   Available Models: {stats['available_models']}")
    print("\n   Per-Model Stats:")
    for model_id, model_stats in stats['model_stats'].items():
        if model_stats['total_requests'] > 0:
            print(f"      {model_id}:")
            print(f"         Requests: {model_stats['total_requests']}")
            print(f"         Avg Latency: {model_stats['avg_latency_ms']:.0f}ms")
            print(f"         Success Rate: {model_stats['success_rate']:.1%}")

    # Example 6: Benchmark strategies
    print("\n6. Benchmarking routing strategies (10 iterations)...")
    test_queries = [
        {'query': 'Simple question', 'capability': 'simple_qa'},
        {'query': 'Complex analysis', 'capability': 'complex_reasoning'},
        {'query': 'Code task', 'capability': 'code_generation'}
    ]
    benchmark = manager.benchmark_strategies(test_queries, iterations=10)
    print("\n   Strategy Performance:")
    for strategy, metrics in benchmark.items():
        print(f"      {strategy}:")
        print(f"         Avg Cost: ${metrics['avg_cost']:.4f}")
        print(f"         Avg Latency: {metrics['avg_latency']:.0f}ms")
        print(f"         Success Rate: {metrics['success_rate']:.1%}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
