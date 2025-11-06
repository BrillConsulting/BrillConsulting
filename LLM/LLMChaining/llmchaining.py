"""
LLMChaining - Production-Ready LLM Chain Orchestration System
Author: BrillConsulting
Description: Comprehensive framework for building, composing, and executing LLM chains
with advanced features including parallel execution, conditional routing, error recovery,
state management, and async support.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from functools import wraps


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChainStatus(Enum):
    """Status enumeration for chain execution"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ChainError(Exception):
    """Base exception for chain errors"""
    pass


class ChainExecutionError(ChainError):
    """Raised when chain execution fails"""
    pass


class ChainValidationError(ChainError):
    """Raised when chain validation fails"""
    pass


@dataclass
class ChainResult:
    """Result of a chain execution"""
    status: ChainStatus
    output: Any
    error: Optional[Exception] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'status': self.status.value,
            'output': self.output,
            'error': str(self.error) if self.error else None,
            'execution_time': self.execution_time,
            'metadata': self.metadata,
            'retries': self.retries
        }


@dataclass
class ChainState:
    """State management for chain execution"""
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Set a state value"""
        self.data[key] = value
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'set',
            'key': key,
            'value': value
        })

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value"""
        return self.data.get(key, default)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple state values"""
        for key, value in updates.items():
            self.set(key, value)

    def clear(self) -> None:
        """Clear state data"""
        self.data.clear()
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'clear'
        })

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of current state"""
        return {
            'data': self.data.copy(),
            'context': self.context.copy(),
            'timestamp': datetime.now().isoformat()
        }


class RetryStrategy:
    """Retry strategy configuration"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given retry attempt"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        if self.jitter:
            import random
            delay *= (0.5 + random.random())

        return delay


def with_retry(retry_strategy: Optional[RetryStrategy] = None):
    """Decorator for adding retry logic to functions"""
    strategy = retry_strategy or RetryStrategy()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(strategy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < strategy.max_retries:
                        delay = strategy.get_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {strategy.max_retries + 1} attempts failed"
                        )

            raise last_exception

        return wrapper
    return decorator


class BaseChain(ABC):
    """Abstract base class for all chains"""

    def __init__(
        self,
        name: str,
        retry_strategy: Optional[RetryStrategy] = None,
        error_handler: Optional[Callable] = None,
        validators: Optional[List[Callable]] = None
    ):
        self.name = name
        self.retry_strategy = retry_strategy
        self.error_handler = error_handler
        self.validators = validators or []
        self.state = ChainState()

    @abstractmethod
    def _execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute the chain logic - to be implemented by subclasses"""
        pass

    def validate(self, input_data: Any) -> bool:
        """Validate input data"""
        for validator in self.validators:
            if not validator(input_data):
                return False
        return True

    def execute(self, input_data: Any, state: Optional[ChainState] = None) -> ChainResult:
        """Execute the chain with error handling and retry logic"""
        if state is None:
            state = self.state

        start_time = time.time()
        retries = 0

        # Validate input
        if not self.validate(input_data):
            return ChainResult(
                status=ChainStatus.FAILED,
                output=None,
                error=ChainValidationError(f"Validation failed for {self.name}"),
                execution_time=time.time() - start_time
            )

        # Execute with retry logic
        max_retries = self.retry_strategy.max_retries if self.retry_strategy else 0

        for attempt in range(max_retries + 1):
            try:
                output = self._execute(input_data, state)
                execution_time = time.time() - start_time

                return ChainResult(
                    status=ChainStatus.SUCCESS,
                    output=output,
                    execution_time=execution_time,
                    retries=retries,
                    metadata={'name': self.name}
                )

            except Exception as e:
                retries += 1

                # Try error handler
                if self.error_handler:
                    try:
                        recovered = self.error_handler(e, input_data, state)
                        if recovered is not None:
                            return ChainResult(
                                status=ChainStatus.SUCCESS,
                                output=recovered,
                                execution_time=time.time() - start_time,
                                retries=retries,
                                metadata={'name': self.name, 'recovered': True}
                            )
                    except Exception as recovery_error:
                        logger.error(f"Error handler failed: {recovery_error}")

                # Retry logic
                if attempt < max_retries and self.retry_strategy:
                    delay = self.retry_strategy.get_delay(attempt)
                    logger.warning(
                        f"{self.name}: Attempt {attempt + 1} failed. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    execution_time = time.time() - start_time
                    return ChainResult(
                        status=ChainStatus.FAILED,
                        output=None,
                        error=e,
                        execution_time=execution_time,
                        retries=retries,
                        metadata={'name': self.name}
                    )

    async def execute_async(
        self,
        input_data: Any,
        state: Optional[ChainState] = None
    ) -> ChainResult:
        """Async execution wrapper"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input_data, state)


class LLMChain(BaseChain):
    """Chain for LLM execution"""

    def __init__(
        self,
        name: str,
        llm_function: Callable,
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.llm_function = llm_function
        self.prompt_template = prompt_template

    def _execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute LLM chain"""
        # Format prompt if template provided
        if self.prompt_template:
            prompt = self.prompt_template.format(
                input=input_data,
                **state.data
            )
        else:
            prompt = input_data

        # Execute LLM
        result = self.llm_function(prompt)

        # Store result in state
        state.set(f"{self.name}_output", result)

        return result


class TransformChain(BaseChain):
    """Chain for data transformation"""

    def __init__(
        self,
        name: str,
        transform_function: Callable,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.transform_function = transform_function

    def _execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute transformation"""
        result = self.transform_function(input_data, state)
        state.set(f"{self.name}_output", result)
        return result


class SequentialChain(BaseChain):
    """Execute chains sequentially"""

    def __init__(
        self,
        name: str,
        chains: List[BaseChain],
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.chains = chains

    def _execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute chains in sequence"""
        current_output = input_data
        results = []

        for chain in self.chains:
            logger.info(f"Executing chain: {chain.name}")
            result = chain.execute(current_output, state)
            results.append(result)

            if result.status != ChainStatus.SUCCESS:
                logger.error(f"Chain {chain.name} failed: {result.error}")
                raise ChainExecutionError(
                    f"Sequential chain failed at {chain.name}: {result.error}"
                )

            current_output = result.output

        state.set(f"{self.name}_results", results)
        return current_output

    async def execute_async(
        self,
        input_data: Any,
        state: Optional[ChainState] = None
    ) -> ChainResult:
        """Async sequential execution"""
        if state is None:
            state = self.state

        start_time = time.time()
        current_output = input_data
        results = []

        try:
            for chain in self.chains:
                logger.info(f"Executing chain: {chain.name}")
                result = await chain.execute_async(current_output, state)
                results.append(result)

                if result.status != ChainStatus.SUCCESS:
                    raise ChainExecutionError(
                        f"Sequential chain failed at {chain.name}: {result.error}"
                    )

                current_output = result.output

            state.set(f"{self.name}_results", results)

            return ChainResult(
                status=ChainStatus.SUCCESS,
                output=current_output,
                execution_time=time.time() - start_time,
                metadata={'name': self.name, 'chain_count': len(self.chains)}
            )

        except Exception as e:
            return ChainResult(
                status=ChainStatus.FAILED,
                output=None,
                error=e,
                execution_time=time.time() - start_time,
                metadata={'name': self.name}
            )


class ParallelChain(BaseChain):
    """Execute chains in parallel"""

    def __init__(
        self,
        name: str,
        chains: List[BaseChain],
        max_workers: Optional[int] = None,
        aggregate_function: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.chains = chains
        self.max_workers = max_workers
        self.aggregate_function = aggregate_function or (lambda results: results)

    def _execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute chains in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chain = {
                executor.submit(chain.execute, input_data, state): chain
                for chain in self.chains
            }

            for future in as_completed(future_to_chain):
                chain = future_to_chain[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Chain {chain.name} completed")
                except Exception as e:
                    logger.error(f"Chain {chain.name} raised exception: {e}")
                    results.append(ChainResult(
                        status=ChainStatus.FAILED,
                        output=None,
                        error=e,
                        metadata={'name': chain.name}
                    ))

        # Check for failures
        failed = [r for r in results if r.status != ChainStatus.SUCCESS]
        if failed:
            raise ChainExecutionError(
                f"Parallel chain had {len(failed)} failures"
            )

        # Aggregate results
        outputs = [r.output for r in results]
        aggregated = self.aggregate_function(outputs)

        state.set(f"{self.name}_results", results)
        return aggregated

    async def execute_async(
        self,
        input_data: Any,
        state: Optional[ChainState] = None
    ) -> ChainResult:
        """Async parallel execution"""
        if state is None:
            state = self.state

        start_time = time.time()

        try:
            # Execute all chains concurrently
            tasks = [
                chain.execute_async(input_data, state)
                for chain in self.chains
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            chain_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    chain_results.append(ChainResult(
                        status=ChainStatus.FAILED,
                        output=None,
                        error=result,
                        metadata={'name': self.chains[i].name}
                    ))
                else:
                    chain_results.append(result)

            # Check for failures
            failed = [r for r in chain_results if r.status != ChainStatus.SUCCESS]
            if failed:
                raise ChainExecutionError(
                    f"Parallel chain had {len(failed)} failures"
                )

            # Aggregate results
            outputs = [r.output for r in chain_results]
            aggregated = self.aggregate_function(outputs)

            state.set(f"{self.name}_results", chain_results)

            return ChainResult(
                status=ChainStatus.SUCCESS,
                output=aggregated,
                execution_time=time.time() - start_time,
                metadata={'name': self.name, 'chain_count': len(self.chains)}
            )

        except Exception as e:
            return ChainResult(
                status=ChainStatus.FAILED,
                output=None,
                error=e,
                execution_time=time.time() - start_time,
                metadata={'name': self.name}
            )


class ConditionalChain(BaseChain):
    """Execute chains based on conditions"""

    def __init__(
        self,
        name: str,
        condition: Callable[[Any, ChainState], bool],
        if_chain: BaseChain,
        else_chain: Optional[BaseChain] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.condition = condition
        self.if_chain = if_chain
        self.else_chain = else_chain

    def _execute(self, input_data: Any, state: ChainState) -> Any:
        """Execute chain based on condition"""
        condition_result = self.condition(input_data, state)
        state.set(f"{self.name}_condition", condition_result)

        if condition_result:
            logger.info(f"Condition true, executing {self.if_chain.name}")
            result = self.if_chain.execute(input_data, state)
        elif self.else_chain:
            logger.info(f"Condition false, executing {self.else_chain.name}")
            result = self.else_chain.execute(input_data, state)
        else:
            logger.info("Condition false, no else chain")
            return input_data

        if result.status != ChainStatus.SUCCESS:
            raise ChainExecutionError(
                f"Conditional chain failed: {result.error}"
            )

        return result.output


class RouterChain(BaseChain):
    """Route to different chains based on input"""

    def __init__(
        self,
        name: str,
        routes: Dict[str, BaseChain],
        router_function: Callable[[Any, ChainState], str],
        default_chain: Optional[BaseChain] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.routes = routes
        self.router_function = router_function
        self.default_chain = default_chain

    def _execute(self, input_data: Any, state: ChainState) -> Any:
        """Route to appropriate chain"""
        route_key = self.router_function(input_data, state)
        state.set(f"{self.name}_route", route_key)

        chain = self.routes.get(route_key, self.default_chain)

        if chain is None:
            raise ChainExecutionError(
                f"No chain found for route '{route_key}' and no default chain"
            )

        logger.info(f"Routing to {chain.name} (route: {route_key})")
        result = chain.execute(input_data, state)

        if result.status != ChainStatus.SUCCESS:
            raise ChainExecutionError(
                f"Router chain failed at {chain.name}: {result.error}"
            )

        return result.output


class ChainComposer:
    """Compose complex chains from simple building blocks"""

    @staticmethod
    def sequence(*chains: BaseChain, name: Optional[str] = None) -> SequentialChain:
        """Create a sequential chain"""
        return SequentialChain(
            name=name or f"sequence_{len(chains)}",
            chains=list(chains)
        )

    @staticmethod
    def parallel(
        *chains: BaseChain,
        name: Optional[str] = None,
        max_workers: Optional[int] = None,
        aggregate: Optional[Callable] = None
    ) -> ParallelChain:
        """Create a parallel chain"""
        return ParallelChain(
            name=name or f"parallel_{len(chains)}",
            chains=list(chains),
            max_workers=max_workers,
            aggregate_function=aggregate
        )

    @staticmethod
    def conditional(
        condition: Callable,
        if_chain: BaseChain,
        else_chain: Optional[BaseChain] = None,
        name: Optional[str] = None
    ) -> ConditionalChain:
        """Create a conditional chain"""
        return ConditionalChain(
            name=name or "conditional",
            condition=condition,
            if_chain=if_chain,
            else_chain=else_chain
        )

    @staticmethod
    def router(
        router_function: Callable,
        routes: Dict[str, BaseChain],
        default: Optional[BaseChain] = None,
        name: Optional[str] = None
    ) -> RouterChain:
        """Create a router chain"""
        return RouterChain(
            name=name or "router",
            routes=routes,
            router_function=router_function,
            default_chain=default
        )


class LLMChainingSystem:
    """Main orchestration system for LLM chains"""

    def __init__(self):
        self.chains: Dict[str, BaseChain] = {}
        self.global_state = ChainState()
        self.execution_history: List[ChainResult] = []
        logger.info("LLMChaining system initialized")

    def register_chain(self, chain: BaseChain) -> None:
        """Register a chain with the system"""
        self.chains[chain.name] = chain
        logger.info(f"Registered chain: {chain.name}")

    def get_chain(self, name: str) -> Optional[BaseChain]:
        """Get a registered chain by name"""
        return self.chains.get(name)

    def execute_chain(
        self,
        chain_name: str,
        input_data: Any,
        use_global_state: bool = True
    ) -> ChainResult:
        """Execute a registered chain"""
        chain = self.get_chain(chain_name)
        if not chain:
            raise ValueError(f"Chain '{chain_name}' not found")

        state = self.global_state if use_global_state else ChainState()
        result = chain.execute(input_data, state)
        self.execution_history.append(result)

        return result

    async def execute_chain_async(
        self,
        chain_name: str,
        input_data: Any,
        use_global_state: bool = True
    ) -> ChainResult:
        """Execute a registered chain asynchronously"""
        chain = self.get_chain(chain_name)
        if not chain:
            raise ValueError(f"Chain '{chain_name}' not found")

        state = self.global_state if use_global_state else ChainState()
        result = await chain.execute_async(input_data, state)
        self.execution_history.append(result)

        return result

    def batch_execute(
        self,
        chain_name: str,
        inputs: List[Any],
        max_workers: Optional[int] = None
    ) -> List[ChainResult]:
        """Execute chain on multiple inputs in parallel"""
        chain = self.get_chain(chain_name)
        if not chain:
            raise ValueError(f"Chain '{chain_name}' not found")

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(chain.execute, input_data, ChainState())
                for input_data in inputs
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                self.execution_history.append(result)

        return results

    async def batch_execute_async(
        self,
        chain_name: str,
        inputs: List[Any]
    ) -> List[ChainResult]:
        """Execute chain on multiple inputs asynchronously"""
        chain = self.get_chain(chain_name)
        if not chain:
            raise ValueError(f"Chain '{chain_name}' not found")

        tasks = [
            chain.execute_async(input_data, ChainState())
            for input_data in inputs
        ]

        results = await asyncio.gather(*tasks)
        self.execution_history.extend(results)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {'total_executions': 0}

        successful = [r for r in self.execution_history if r.status == ChainStatus.SUCCESS]
        failed = [r for r in self.execution_history if r.status == ChainStatus.FAILED]

        total_time = sum(r.execution_time for r in self.execution_history)
        avg_time = total_time / len(self.execution_history)

        return {
            'total_executions': len(self.execution_history),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.execution_history),
            'total_time': total_time,
            'average_time': avg_time,
            'total_retries': sum(r.retries for r in self.execution_history)
        }

    def export_state(self, filepath: str) -> None:
        """Export global state to file"""
        with open(filepath, 'w') as f:
            json.dump(self.global_state.snapshot(), f, indent=2)
        logger.info(f"State exported to {filepath}")

    def import_state(self, filepath: str) -> None:
        """Import global state from file"""
        with open(filepath, 'r') as f:
            snapshot = json.load(f)
            self.global_state.data = snapshot.get('data', {})
            self.global_state.context = snapshot.get('context', {})
        logger.info(f"State imported from {filepath}")

    def reset(self) -> None:
        """Reset the system"""
        self.global_state.clear()
        self.execution_history.clear()
        logger.info("System reset complete")


# Example usage and demonstration
def example_llm_function(prompt: str) -> str:
    """Mock LLM function for demonstration"""
    return f"LLM Response to: {prompt}"


def example_transform(data: Any, state: ChainState) -> Any:
    """Example transformation function"""
    return f"Transformed: {data}"


def demo_basic_chain():
    """Demonstrate basic chain usage"""
    print("\n=== Basic Chain Demo ===")

    # Create a simple LLM chain
    chain = LLMChain(
        name="simple_llm",
        llm_function=example_llm_function,
        prompt_template="Question: {input}\nAnswer:"
    )

    result = chain.execute("What is AI?")
    print(f"Result: {result.output}")
    print(f"Status: {result.status.value}")
    print(f"Execution time: {result.execution_time:.3f}s")


def demo_sequential_chain():
    """Demonstrate sequential chain"""
    print("\n=== Sequential Chain Demo ===")

    chain1 = TransformChain(
        name="step1",
        transform_function=lambda x, s: f"Step1({x})"
    )

    chain2 = TransformChain(
        name="step2",
        transform_function=lambda x, s: f"Step2({x})"
    )

    chain3 = TransformChain(
        name="step3",
        transform_function=lambda x, s: f"Step3({x})"
    )

    seq_chain = ChainComposer.sequence(chain1, chain2, chain3, name="pipeline")
    result = seq_chain.execute("input")

    print(f"Result: {result.output}")


def demo_parallel_chain():
    """Demonstrate parallel chain"""
    print("\n=== Parallel Chain Demo ===")

    def slow_operation(data, state):
        time.sleep(0.1)
        return f"Branch{data}"

    chains = [
        TransformChain(name=f"branch{i}", transform_function=slow_operation)
        for i in range(3)
    ]

    parallel_chain = ChainComposer.parallel(
        *chains,
        name="parallel_processing",
        aggregate=lambda results: f"Aggregated: {results}"
    )

    start = time.time()
    result = parallel_chain.execute("_data")
    elapsed = time.time() - start

    print(f"Result: {result.output}")
    print(f"Parallel execution time: {elapsed:.3f}s (would be ~0.3s sequential)")


def demo_conditional_chain():
    """Demonstrate conditional chain"""
    print("\n=== Conditional Chain Demo ===")

    positive_chain = TransformChain(
        name="positive",
        transform_function=lambda x, s: f"Positive path: {x}"
    )

    negative_chain = TransformChain(
        name="negative",
        transform_function=lambda x, s: f"Negative path: {x}"
    )

    conditional = ChainComposer.conditional(
        condition=lambda x, s: int(x) > 0,
        if_chain=positive_chain,
        else_chain=negative_chain,
        name="number_check"
    )

    result1 = conditional.execute(5)
    result2 = conditional.execute(-3)

    print(f"Input 5: {result1.output}")
    print(f"Input -3: {result2.output}")


def demo_router_chain():
    """Demonstrate router chain"""
    print("\n=== Router Chain Demo ===")

    routes = {
        'sentiment': TransformChain(
            name="sentiment_analysis",
            transform_function=lambda x, s: f"Sentiment: {x}"
        ),
        'summary': TransformChain(
            name="summarization",
            transform_function=lambda x, s: f"Summary: {x}"
        ),
        'translation': TransformChain(
            name="translation",
            transform_function=lambda x, s: f"Translation: {x}"
        )
    }

    def route_function(data, state):
        # Simple routing logic
        if 'sentiment' in data.lower():
            return 'sentiment'
        elif 'summary' in data.lower():
            return 'summary'
        else:
            return 'translation'

    router = ChainComposer.router(
        router_function=route_function,
        routes=routes,
        name="task_router"
    )

    result1 = router.execute("Analyze sentiment of this text")
    result2 = router.execute("Create a summary")

    print(f"Sentiment request: {result1.output}")
    print(f"Summary request: {result2.output}")


async def demo_async_execution():
    """Demonstrate async execution"""
    print("\n=== Async Execution Demo ===")

    chains = [
        TransformChain(
            name=f"async_chain_{i}",
            transform_function=lambda x, s: f"Async result {x}"
        )
        for i in range(3)
    ]

    parallel = ParallelChain(
        name="async_parallel",
        chains=chains,
        aggregate_function=lambda results: results
    )

    start = time.time()
    result = await parallel.execute_async("test")
    elapsed = time.time() - start

    print(f"Result: {result.output}")
    print(f"Async execution time: {elapsed:.3f}s")


def demo_system():
    """Demonstrate full system usage"""
    print("\n=== LLMChaining System Demo ===")

    system = LLMChainingSystem()

    # Register chains
    chain1 = TransformChain(
        name="preprocessor",
        transform_function=lambda x, s: x.upper()
    )

    chain2 = LLMChain(
        name="llm_processor",
        llm_function=example_llm_function
    )

    chain3 = TransformChain(
        name="postprocessor",
        transform_function=lambda x, s: f"Final: {x}"
    )

    pipeline = ChainComposer.sequence(
        chain1, chain2, chain3,
        name="full_pipeline"
    )

    system.register_chain(pipeline)

    # Execute
    result = system.execute_chain("full_pipeline", "test input")

    # Get statistics
    stats = system.get_statistics()

    print(f"Result: {result.output}")
    print(f"Statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    print("LLMChaining - Production-Ready Chain Orchestration System")
    print("=" * 60)

    # Run demonstrations
    demo_basic_chain()
    demo_sequential_chain()
    demo_parallel_chain()
    demo_conditional_chain()
    demo_router_chain()
    demo_system()

    # Run async demo
    print("\nRunning async demonstrations...")
    asyncio.run(demo_async_execution())

    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print(f"Timestamp: {datetime.now()}")
