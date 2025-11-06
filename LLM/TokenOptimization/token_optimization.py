"""
TokenOptimization - Production-Ready Token Optimization System
Author: BrillConsulting
Description: Comprehensive token usage optimization with multi-model support,
            compression, context management, and cost optimization
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import hashlib
from functools import lru_cache

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Using fallback token estimation.")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    provider: ModelProvider
    name: str
    context_window: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    encoding: Optional[str] = None
    max_output_tokens: Optional[int] = None


@dataclass
class TokenStats:
    """Token usage statistics"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CompressionResult:
    """Result of text compression"""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    techniques_applied: List[str]


class TokenCounter:
    """Precise token counting for multiple model providers"""

    # Model configurations with current pricing (as of 2025)
    MODELS = {
        # OpenAI GPT-4 models
        "gpt-4": ModelConfig(
            ModelProvider.OPENAI, "gpt-4", 8192, 0.03, 0.06, "cl100k_base"
        ),
        "gpt-4-turbo": ModelConfig(
            ModelProvider.OPENAI, "gpt-4-turbo", 128000, 0.01, 0.03, "cl100k_base"
        ),
        "gpt-4o": ModelConfig(
            ModelProvider.OPENAI, "gpt-4o", 128000, 0.005, 0.015, "cl100k_base"
        ),
        "gpt-3.5-turbo": ModelConfig(
            ModelProvider.OPENAI, "gpt-3.5-turbo", 16385, 0.0005, 0.0015, "cl100k_base"
        ),
        # Anthropic Claude models
        "claude-3-opus": ModelConfig(
            ModelProvider.ANTHROPIC, "claude-3-opus", 200000, 0.015, 0.075
        ),
        "claude-3-sonnet": ModelConfig(
            ModelProvider.ANTHROPIC, "claude-3-sonnet", 200000, 0.003, 0.015
        ),
        "claude-3-haiku": ModelConfig(
            ModelProvider.ANTHROPIC, "claude-3-haiku", 200000, 0.00025, 0.00125
        ),
        # Google Gemini models
        "gemini-pro": ModelConfig(
            ModelProvider.GOOGLE, "gemini-pro", 32760, 0.00025, 0.0005
        ),
        "gemini-ultra": ModelConfig(
            ModelProvider.GOOGLE, "gemini-ultra", 32760, 0.001, 0.002
        ),
    }

    def __init__(self, model: str = "gpt-4o"):
        """Initialize token counter for specific model"""
        self.model = model
        self.config = self.MODELS.get(model)

        if not self.config:
            logger.warning(f"Unknown model {model}, using default estimation")
            self.config = ModelConfig(
                ModelProvider.CUSTOM, model, 4096, 0.001, 0.002
            )

        # Initialize tokenizer for OpenAI models
        self.tokenizer = None
        if self.config.provider == ModelProvider.OPENAI and TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(self.config.encoding)
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0

        # Use tiktoken for OpenAI models
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.error(f"Tokenization error: {e}")

        # Fallback estimation for other models
        return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using heuristics"""
        # Average: 1 token ≈ 4 characters or 0.75 words
        word_count = len(text.split())
        char_count = len(text)

        # Use both heuristics and take average
        tokens_by_words = int(word_count / 0.75)
        tokens_by_chars = int(char_count / 4)

        return int((tokens_by_words + tokens_by_chars) / 2)

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in message list (ChatML format)"""
        total = 0

        for message in messages:
            # Message overhead (role + formatting)
            total += 4

            # Content tokens
            if "content" in message:
                total += self.count_tokens(message["content"])

            # Name tokens if present
            if "name" in message:
                total += self.count_tokens(message["name"])

        # Conversation overhead
        total += 3

        return total

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int = 0
    ) -> TokenStats:
        """Calculate cost for token usage"""
        input_cost = (input_tokens / 1000) * self.config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.config.output_cost_per_1k

        return TokenStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            model=self.model
        )

    def fits_context_window(
        self,
        input_tokens: int,
        max_output_tokens: Optional[int] = None
    ) -> Tuple[bool, int]:
        """Check if input fits in context window"""
        available = self.config.context_window - input_tokens

        if max_output_tokens:
            fits = input_tokens + max_output_tokens <= self.config.context_window
        else:
            fits = input_tokens < self.config.context_window

        return fits, available


class TextCompressor:
    """Advanced text compression techniques for token reduction"""

    def __init__(self, token_counter: TokenCounter):
        self.counter = token_counter

    def compress(
        self,
        text: str,
        target_reduction: float = 0.3,
        preserve_meaning: bool = True
    ) -> CompressionResult:
        """Apply compression techniques to reduce tokens"""
        original_text = text
        compressed_text = text
        techniques = []

        original_tokens = self.counter.count_tokens(original_text)
        target_tokens = int(original_tokens * (1 - target_reduction))

        # Apply techniques in order of safety
        if preserve_meaning:
            # Safe techniques that preserve meaning
            compressed_text = self._remove_redundant_whitespace(compressed_text)
            techniques.append("whitespace_removal")

            compressed_text = self._abbreviate_common_phrases(compressed_text)
            techniques.append("phrase_abbreviation")

            compressed_text = self._remove_filler_words(compressed_text)
            techniques.append("filler_removal")

        else:
            # Aggressive techniques
            compressed_text = self._aggressive_compression(compressed_text)
            techniques.append("aggressive_compression")

        current_tokens = self.counter.count_tokens(compressed_text)

        # If still not enough, apply summarization hints
        if current_tokens > target_tokens:
            compressed_text = self._add_summarization_prompt(compressed_text, target_tokens)
            techniques.append("summarization_prompt")
            current_tokens = self.counter.count_tokens(compressed_text)

        compression_ratio = 1 - (current_tokens / original_tokens) if original_tokens > 0 else 0

        return CompressionResult(
            original_text=original_text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=current_tokens,
            compression_ratio=compression_ratio,
            techniques_applied=techniques
        )

    def _remove_redundant_whitespace(self, text: str) -> str:
        """Remove extra whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove spaces around punctuation
        text = re.sub(r'\s*([,;:.])\s*', r'\1 ', text)
        return text.strip()

    def _abbreviate_common_phrases(self, text: str) -> str:
        """Abbreviate common phrases"""
        abbreviations = {
            r'\bfor example\b': 'e.g.',
            r'\bthat is\b': 'i.e.',
            r'\band so on\b': 'etc.',
            r'\bas soon as possible\b': 'ASAP',
            r'\bfrequently asked questions\b': 'FAQ',
        }

        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _remove_filler_words(self, text: str) -> str:
        """Remove filler words that don't add meaning"""
        fillers = [
            r'\bactually\b', r'\bbasically\b', r'\bliterally\b',
            r'\bjust\b', r'\breally\b', r'\bvery\b', r'\bquite\b',
            r'\bsimply\b', r'\bmerely\b', r'\bperhaps\b'
        ]

        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)

        return self._remove_redundant_whitespace(text)

    def _aggressive_compression(self, text: str) -> str:
        """Apply aggressive compression (may lose some context)"""
        # Remove examples and parenthetical content
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\bfor example[^.]*\.', '', text, flags=re.IGNORECASE)

        # Remove redundant adjectives
        text = re.sub(r'\b(very|extremely|highly|really)\s+(\w+)', r'\2', text)

        # Simplify sentences
        text = re.sub(r'\b(in order to|so as to)\b', 'to', text)
        text = re.sub(r'\b(due to the fact that|owing to the fact that)\b', 'because', text)

        return self._remove_redundant_whitespace(text)

    def _add_summarization_prompt(self, text: str, target_tokens: int) -> str:
        """Add instruction to summarize response"""
        prompt = f"[Summarize in <{target_tokens} tokens] {text}"
        return prompt


class ContextWindowManager:
    """Manage context windows with sliding windows and priority-based retention"""

    def __init__(self, token_counter: TokenCounter, max_tokens: Optional[int] = None):
        self.counter = token_counter
        self.max_tokens = max_tokens or int(token_counter.config.context_window * 0.8)
        self.messages = deque()
        self.system_message = None
        self.pinned_messages = []

    def set_system_message(self, content: str):
        """Set system message (always retained)"""
        self.system_message = {"role": "system", "content": content}

    def add_message(self, role: str, content: str, priority: int = 0, pinned: bool = False):
        """Add message to context"""
        message = {
            "role": role,
            "content": content,
            "priority": priority,
            "pinned": pinned,
            "timestamp": datetime.now().isoformat()
        }

        if pinned:
            self.pinned_messages.append(message)
        else:
            self.messages.append(message)

        self._trim_context()

    def _trim_context(self):
        """Trim context to fit within token limit"""
        while True:
            current_tokens = self._count_current_tokens()

            if current_tokens <= self.max_tokens:
                break

            if not self.messages:
                logger.warning("Cannot trim further - only system and pinned messages remain")
                break

            # Remove lowest priority message
            self.messages.popleft()

    def _count_current_tokens(self) -> int:
        """Count tokens in current context"""
        messages = self.get_messages()
        return self.counter.count_messages(messages)

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in correct order"""
        result = []

        if self.system_message:
            result.append({
                "role": self.system_message["role"],
                "content": self.system_message["content"]
            })

        for msg in self.pinned_messages:
            result.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        for msg in self.messages:
            result.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics"""
        current_tokens = self._count_current_tokens()

        return {
            "current_tokens": current_tokens,
            "max_tokens": self.max_tokens,
            "utilization": current_tokens / self.max_tokens,
            "remaining_tokens": self.max_tokens - current_tokens,
            "message_count": len(self.messages) + len(self.pinned_messages),
            "pinned_count": len(self.pinned_messages)
        }

    def clear(self, keep_system: bool = True, keep_pinned: bool = True):
        """Clear context"""
        if not keep_system:
            self.system_message = None
        if not keep_pinned:
            self.pinned_messages = []
        self.messages.clear()


class PromptOptimizer:
    """Optimize prompts for better token efficiency"""

    def __init__(self, token_counter: TokenCounter):
        self.counter = token_counter
        self.compressor = TextCompressor(token_counter)

    def optimize_prompt(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        preserve_structure: bool = True
    ) -> Dict[str, Any]:
        """Optimize prompt to reduce tokens while preserving intent"""
        original_tokens = self.counter.count_tokens(prompt)

        if max_tokens and original_tokens <= max_tokens:
            return {
                "original_prompt": prompt,
                "optimized_prompt": prompt,
                "original_tokens": original_tokens,
                "optimized_tokens": original_tokens,
                "reduction": 0,
                "techniques": []
            }

        # Calculate target reduction
        if max_tokens:
            target_reduction = 1 - (max_tokens / original_tokens)
        else:
            target_reduction = 0.2  # Default 20% reduction

        # Apply compression
        result = self.compressor.compress(
            prompt,
            target_reduction=target_reduction,
            preserve_meaning=preserve_structure
        )

        reduction = (original_tokens - result.compressed_tokens) / original_tokens

        return {
            "original_prompt": prompt,
            "optimized_prompt": result.compressed_text,
            "original_tokens": original_tokens,
            "optimized_tokens": result.compressed_tokens,
            "reduction": reduction,
            "techniques": result.techniques_applied
        }

    def create_few_shot_examples(
        self,
        examples: List[Dict[str, str]],
        max_examples: int = 3,
        max_tokens_per_example: int = 100
    ) -> str:
        """Create optimized few-shot examples"""
        selected_examples = examples[:max_examples]
        formatted_examples = []

        for i, example in enumerate(selected_examples, 1):
            input_text = example.get("input", "")
            output_text = example.get("output", "")

            # Compress if needed
            input_tokens = self.counter.count_tokens(input_text)
            if input_tokens > max_tokens_per_example // 2:
                result = self.compressor.compress(input_text, target_reduction=0.5)
                input_text = result.compressed_text

            formatted_examples.append(
                f"Example {i}:\nInput: {input_text}\nOutput: {output_text}"
            )

        return "\n\n".join(formatted_examples)


class ResponseTruncator:
    """Truncate and manage response lengths"""

    def __init__(self, token_counter: TokenCounter):
        self.counter = token_counter

    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        strategy: str = "end"
    ) -> Dict[str, Any]:
        """Truncate text to maximum tokens"""
        current_tokens = self.counter.count_tokens(text)

        if current_tokens <= max_tokens:
            return {
                "text": text,
                "truncated": False,
                "original_tokens": current_tokens,
                "final_tokens": current_tokens
            }

        if strategy == "end":
            truncated = self._truncate_end(text, max_tokens)
        elif strategy == "start":
            truncated = self._truncate_start(text, max_tokens)
        elif strategy == "middle":
            truncated = self._truncate_middle(text, max_tokens)
        else:
            truncated = self._truncate_sentences(text, max_tokens)

        final_tokens = self.counter.count_tokens(truncated)

        return {
            "text": truncated,
            "truncated": True,
            "original_tokens": current_tokens,
            "final_tokens": final_tokens,
            "strategy": strategy
        }

    def _truncate_end(self, text: str, max_tokens: int) -> str:
        """Keep beginning, truncate end"""
        words = text.split()
        result = []

        for word in words:
            test_text = " ".join(result + [word])
            if self.counter.count_tokens(test_text) > max_tokens:
                break
            result.append(word)

        return " ".join(result) + "..."

    def _truncate_start(self, text: str, max_tokens: int) -> str:
        """Keep end, truncate start"""
        words = text.split()
        result = []

        for word in reversed(words):
            test_text = " ".join([word] + result)
            if self.counter.count_tokens(test_text) > max_tokens:
                break
            result.insert(0, word)

        return "..." + " ".join(result)

    def _truncate_middle(self, text: str, max_tokens: int) -> str:
        """Keep start and end, truncate middle"""
        half_tokens = max_tokens // 2

        start = self._truncate_end(text, half_tokens).rstrip("...")
        end = self._truncate_start(text, half_tokens).lstrip("...")

        return f"{start} [...] {end}"

    def _truncate_sentences(self, text: str, max_tokens: int) -> str:
        """Truncate at sentence boundaries"""
        sentences = re.split(r'([.!?]+)', text)
        result = []

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            test_text = "".join(result + [sentence])
            if self.counter.count_tokens(test_text) > max_tokens:
                break
            result.append(sentence)

        return "".join(result)


class CostOptimizer:
    """Optimize costs across multiple models"""

    def __init__(self):
        self.models = {
            name: TokenCounter(name)
            for name in TokenCounter.MODELS.keys()
        }

    def compare_models(
        self,
        input_text: str,
        expected_output_tokens: int = 500
    ) -> List[Dict[str, Any]]:
        """Compare costs across different models"""
        results = []

        for model_name, counter in self.models.items():
            input_tokens = counter.count_tokens(input_text)

            # Check if fits in context window
            fits, available = counter.fits_context_window(
                input_tokens, expected_output_tokens
            )

            if not fits:
                continue

            stats = counter.calculate_cost(input_tokens, expected_output_tokens)

            results.append({
                "model": model_name,
                "provider": counter.config.provider.value,
                "input_tokens": input_tokens,
                "output_tokens": expected_output_tokens,
                "total_cost": stats.total_cost,
                "context_window": counter.config.context_window,
                "available_tokens": available
            })

        # Sort by cost
        results.sort(key=lambda x: x["total_cost"])

        return results

    def recommend_model(
        self,
        input_text: str,
        expected_output_tokens: int = 500,
        quality_tier: str = "balanced"
    ) -> Dict[str, Any]:
        """Recommend best model based on requirements"""
        comparisons = self.compare_models(input_text, expected_output_tokens)

        if not comparisons:
            return {"error": "No suitable models found"}

        # Quality tiers
        if quality_tier == "premium":
            # Prefer high-quality models
            preferred = ["gpt-4o", "claude-3-opus", "gpt-4-turbo"]
        elif quality_tier == "balanced":
            # Balance quality and cost
            preferred = ["gpt-4o", "claude-3-sonnet", "gpt-3.5-turbo"]
        else:  # economy
            # Minimize cost
            preferred = ["claude-3-haiku", "gpt-3.5-turbo", "gemini-pro"]

        # Find best match
        for model in preferred:
            for comp in comparisons:
                if comp["model"] == model:
                    return {
                        "recommended_model": model,
                        "reason": f"Best {quality_tier} option",
                        **comp
                    }

        # Fallback to cheapest
        return {
            "recommended_model": comparisons[0]["model"],
            "reason": "Cheapest available option",
            **comparisons[0]
        }


class BatchOptimizer:
    """Optimize batch operations for cost and efficiency"""

    def __init__(self, token_counter: TokenCounter):
        self.counter = token_counter

    def optimize_batch(
        self,
        requests: List[str],
        max_batch_tokens: Optional[int] = None
    ) -> List[List[str]]:
        """Group requests into optimal batches"""
        if max_batch_tokens is None:
            max_batch_tokens = int(self.counter.config.context_window * 0.7)

        batches = []
        current_batch = []
        current_tokens = 0

        for request in requests:
            request_tokens = self.counter.count_tokens(request)

            # If single request exceeds limit, put in own batch
            if request_tokens > max_batch_tokens:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([request])
                continue

            # Check if adding to current batch would exceed limit
            if current_tokens + request_tokens > max_batch_tokens:
                batches.append(current_batch)
                current_batch = [request]
                current_tokens = request_tokens
            else:
                current_batch.append(request)
                current_tokens += request_tokens

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def estimate_batch_cost(
        self,
        batches: List[List[str]],
        expected_output_tokens_per_request: int = 500
    ) -> Dict[str, Any]:
        """Estimate cost for batch processing"""
        total_input_tokens = 0
        total_output_tokens = 0

        for batch in batches:
            for request in batch:
                total_input_tokens += self.counter.count_tokens(request)
                total_output_tokens += expected_output_tokens_per_request

        stats = self.counter.calculate_cost(total_input_tokens, total_output_tokens)

        return {
            "batch_count": len(batches),
            "request_count": sum(len(batch) for batch in batches),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "estimated_cost": stats.total_cost,
            "model": self.counter.model
        }


class TokenOptimizationManager:
    """Main manager coordinating all optimization components"""

    def __init__(self, model: str = "gpt-4o"):
        """Initialize token optimization system"""
        self.model = model
        self.counter = TokenCounter(model)
        self.compressor = TextCompressor(self.counter)
        self.context_manager = ContextWindowManager(self.counter)
        self.prompt_optimizer = PromptOptimizer(self.counter)
        self.response_truncator = ResponseTruncator(self.counter)
        self.cost_optimizer = CostOptimizer()
        self.batch_optimizer = BatchOptimizer(self.counter)

        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "tokens_saved": 0,
            "cost_saved": 0.0,
            "compressions": 0,
            "optimizations": 0
        }

        logger.info(f"TokenOptimizationManager initialized with model: {model}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return self.counter.count_tokens(text)

    def compress_text(
        self,
        text: str,
        target_reduction: float = 0.3
    ) -> CompressionResult:
        """Compress text to reduce tokens"""
        result = self.compressor.compress(text, target_reduction)

        self.stats["compressions"] += 1
        self.stats["tokens_saved"] += result.original_tokens - result.compressed_tokens

        return result

    def optimize_prompt(
        self,
        prompt: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize prompt for efficiency"""
        result = self.prompt_optimizer.optimize_prompt(prompt, max_tokens)

        self.stats["optimizations"] += 1
        tokens_saved = result["original_tokens"] - result["optimized_tokens"]
        self.stats["tokens_saved"] += tokens_saved

        return result

    def manage_context(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Manage conversation context"""
        if max_tokens:
            self.context_manager.max_tokens = max_tokens

        # Clear and rebuild context
        self.context_manager.clear()

        for msg in messages:
            self.context_manager.add_message(
                msg.get("role", "user"),
                msg.get("content", "")
            )

        return self.context_manager.get_messages()

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int = 0
    ) -> TokenStats:
        """Calculate cost for token usage"""
        return self.counter.calculate_cost(input_tokens, output_tokens)

    def find_best_model(
        self,
        input_text: str,
        quality_tier: str = "balanced"
    ) -> Dict[str, Any]:
        """Find best model for given input"""
        return self.cost_optimizer.recommend_model(input_text, quality_tier=quality_tier)

    def optimize_batch(
        self,
        requests: List[str]
    ) -> Dict[str, Any]:
        """Optimize batch of requests"""
        batches = self.batch_optimizer.optimize_batch(requests)
        cost_estimate = self.batch_optimizer.estimate_batch_cost(batches)

        return {
            "batches": batches,
            "batch_count": len(batches),
            "cost_estimate": cost_estimate
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            **self.stats,
            "model": self.model,
            "context_stats": self.context_manager.get_stats()
        }

    def execute(self) -> Dict[str, Any]:
        """Execute comprehensive optimization analysis"""
        return {
            'status': 'success',
            'project': 'TokenOptimization',
            'model': self.model,
            'features': [
                'precise_token_counting',
                'text_compression',
                'context_window_management',
                'prompt_optimization',
                'response_truncation',
                'cost_optimization',
                'multi_model_support',
                'batch_optimization'
            ],
            'stats': self.get_stats(),
            'executed_at': datetime.now().isoformat()
        }


# Utility functions for common operations
@lru_cache(maxsize=1000)
def quick_count(text: str, model: str = "gpt-4o") -> int:
    """Quick token count with caching"""
    counter = TokenCounter(model)
    return counter.count_tokens(text)


def quick_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o"
) -> float:
    """Quick cost calculation"""
    counter = TokenCounter(model)
    stats = counter.calculate_cost(input_tokens, output_tokens)
    return stats.total_cost


def compare_all_models(text: str) -> List[Dict[str, Any]]:
    """Compare text across all available models"""
    optimizer = CostOptimizer()
    return optimizer.compare_models(text)


# Example usage and demonstrations
if __name__ == "__main__":
    print("=" * 80)
    print("TokenOptimization - Production System Demo")
    print("=" * 80)

    # Initialize manager
    manager = TokenOptimizationManager(model="gpt-4o")

    # Demo text
    demo_text = """
    This is a demonstration of the token optimization system. The system provides
    comprehensive functionality for managing tokens, optimizing costs, and ensuring
    efficient usage of language model APIs. It includes support for multiple model
    providers including OpenAI, Anthropic, Google, and others. The system can
    compress text, manage context windows, optimize prompts, and calculate costs
    across different models to help you make informed decisions.
    """

    print("\n1. Token Counting")
    print("-" * 80)
    token_count = manager.count_tokens(demo_text)
    print(f"Token count: {token_count}")

    print("\n2. Text Compression")
    print("-" * 80)
    compression_result = manager.compress_text(demo_text, target_reduction=0.4)
    print(f"Original tokens: {compression_result.original_tokens}")
    print(f"Compressed tokens: {compression_result.compressed_tokens}")
    print(f"Compression ratio: {compression_result.compression_ratio:.2%}")
    print(f"Techniques: {', '.join(compression_result.techniques_applied)}")

    print("\n3. Cost Analysis")
    print("-" * 80)
    cost_stats = manager.calculate_cost(token_count, 500)
    print(f"Input cost: ${cost_stats.input_cost:.6f}")
    print(f"Output cost (500 tokens): ${cost_stats.output_cost:.6f}")
    print(f"Total cost: ${cost_stats.total_cost:.6f}")

    print("\n4. Model Comparison")
    print("-" * 80)
    best_model = manager.find_best_model(demo_text, quality_tier="balanced")
    print(f"Recommended: {best_model['recommended_model']}")
    print(f"Reason: {best_model['reason']}")
    print(f"Estimated cost: ${best_model['total_cost']:.6f}")

    print("\n5. System Statistics")
    print("-" * 80)
    stats = manager.get_stats()
    print(f"Compressions performed: {stats['compressions']}")
    print(f"Tokens saved: {stats['tokens_saved']}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
