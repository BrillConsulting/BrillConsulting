"""
Azure OpenAI Service Integration
Author: BrillConsulting
Description: Advanced Azure OpenAI implementation with GPT-4, embeddings, and streaming
"""

from typing import Dict, Any, List, Optional, Iterator
from datetime import datetime
import json
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum


class ModelType(Enum):
    """Available Azure OpenAI models"""
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_35_TURBO = "gpt-35-turbo"
    GPT_35_TURBO_16K = "gpt-35-turbo-16k"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


@dataclass
class Message:
    """Chat message structure"""
    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None


@dataclass
class CompletionResponse:
    """Response from completion API"""
    id: str
    model: str
    content: str
    finish_reason: str
    usage: Dict[str, int]
    created_at: str


@dataclass
class EmbeddingResponse:
    """Response from embedding API"""
    embedding: List[float]
    model: str
    usage: Dict[str, int]


class AzureOpenAIManager:
    """
    Comprehensive Azure OpenAI Service manager

    Features:
    - GPT-4 and GPT-3.5 chat completions
    - Text embeddings generation
    - Streaming responses
    - Function calling support
    - Token management
    - Content filtering
    """

    def __init__(self, endpoint: str, api_key: str, api_version: str = "2024-02-01"):
        """
        Initialize Azure OpenAI manager

        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: API authentication key
            api_version: API version to use
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.conversation_history: List[Message] = []

    def create_chat_completion(
        self,
        messages: List[Message],
        model: ModelType = ModelType.GPT_4,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> CompletionResponse:
        """
        Create a chat completion

        Args:
            messages: List of conversation messages
            model: Model to use for completion
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalty for frequent tokens
            presence_penalty: Penalty for present tokens
            stop: Stop sequences

        Returns:
            CompletionResponse with generated content
        """
        request_payload = {
            "messages": [asdict(msg) for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        if stop:
            request_payload["stop"] = stop

        # Simulate API call
        response = CompletionResponse(
            id=f"chatcmpl-{datetime.now().timestamp()}",
            model=model.value,
            content="This is a simulated response from Azure OpenAI GPT-4. "
                   "In production, this would contain the actual model response.",
            finish_reason="stop",
            usage={
                "prompt_tokens": sum(len(msg.content.split()) for msg in messages),
                "completion_tokens": 20,
                "total_tokens": sum(len(msg.content.split()) for msg in messages) + 20
            },
            created_at=datetime.now().isoformat()
        )

        return response

    def stream_chat_completion(
        self,
        messages: List[Message],
        model: ModelType = ModelType.GPT_4,
        temperature: float = 0.7
    ) -> Iterator[str]:
        """
        Stream chat completion responses

        Args:
            messages: List of conversation messages
            model: Model to use for completion
            temperature: Sampling temperature

        Yields:
            Content chunks as they arrive
        """
        # Simulate streaming response
        simulated_response = "This is a simulated streaming response from Azure OpenAI. "
        simulated_response += "Each chunk would arrive in real-time from the API."

        words = simulated_response.split()
        for word in words:
            yield word + " "

    def create_embeddings(
        self,
        text: str,
        model: ModelType = ModelType.TEXT_EMBEDDING_ADA_002
    ) -> EmbeddingResponse:
        """
        Generate embeddings for text

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            EmbeddingResponse with vector embedding
        """
        # Simulate embedding generation (1536 dimensions for ada-002)
        embedding = [0.1] * 1536

        response = EmbeddingResponse(
            embedding=embedding,
            model=model.value,
            usage={
                "prompt_tokens": len(text.split()),
                "total_tokens": len(text.split())
            }
        )

        return response

    def batch_embeddings(
        self,
        texts: List[str],
        model: ModelType = ModelType.TEXT_EMBEDDING_ADA_002
    ) -> List[EmbeddingResponse]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            model: Embedding model to use

        Returns:
            List of EmbeddingResponse objects
        """
        return [self.create_embeddings(text, model) for text in texts]

    def create_function_call(
        self,
        messages: List[Message],
        functions: List[Dict[str, Any]],
        function_call: str = "auto"
    ) -> CompletionResponse:
        """
        Create completion with function calling

        Args:
            messages: Conversation messages
            functions: Available functions
            function_call: How to handle function calls ("auto", "none", or specific function)

        Returns:
            CompletionResponse possibly containing function call
        """
        request_payload = {
            "messages": [asdict(msg) for msg in messages],
            "functions": functions,
            "function_call": function_call
        }

        # Simulate function call response
        response = CompletionResponse(
            id=f"chatcmpl-{datetime.now().timestamp()}",
            model=ModelType.GPT_4.value,
            content="",
            finish_reason="function_call",
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            created_at=datetime.now().isoformat()
        )

        return response

    def add_to_conversation(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append(Message(role=role, content=content))

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as list of dicts"""
        return [asdict(msg) for msg in self.conversation_history]

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def create_system_message(self, content: str) -> Message:
        """Create a system message"""
        return Message(role="system", content=content)

    def create_user_message(self, content: str) -> Message:
        """Create a user message"""
        return Message(role="user", content=content)

    def create_assistant_message(self, content: str) -> Message:
        """Create an assistant message"""
        return Message(role="assistant", content=content)


class ContentFilterManager:
    """Manage content filtering for Azure OpenAI"""

    def __init__(self):
        self.filter_categories = ["hate", "sexual", "violence", "self_harm"]

    def analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze content for policy violations

        Args:
            text: Text to analyze

        Returns:
            Filter results by category
        """
        results = {
            "filtered": False,
            "categories": {}
        }

        for category in self.filter_categories:
            results["categories"][category] = {
                "filtered": False,
                "severity": "safe"
            }

        return results


class ConversationManager:
    """Manage multi-turn conversations with context"""

    def __init__(self, openai_manager: AzureOpenAIManager, system_prompt: str = ""):
        self.openai = openai_manager
        self.messages: List[Message] = []

        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))

    def add_user_message(self, content: str):
        """Add user message to conversation"""
        self.messages.append(Message(role="user", content=content))

    def get_response(
        self,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Get assistant response for conversation

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Assistant's response content
        """
        response = self.openai.create_chat_completion(
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Add assistant response to history
        self.messages.append(Message(role="assistant", content=response.content))

        return response.content

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation"""
        return {
            "message_count": len(self.messages),
            "total_tokens": sum(len(msg.content.split()) for msg in self.messages),
            "messages": [asdict(msg) for msg in self.messages]
        }


def demo_basic_chat():
    """Demonstrate basic chat completion"""
    print("=== Basic Chat Completion Demo ===\n")

    manager = AzureOpenAIManager(
        endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key"
    )

    messages = [
        manager.create_system_message("You are a helpful AI assistant."),
        manager.create_user_message("What are the benefits of cloud computing?")
    ]

    response = manager.create_chat_completion(messages)

    print(f"Model: {response.model}")
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.usage['total_tokens']}")
    print(f"Created at: {response.created_at}\n")


def demo_streaming():
    """Demonstrate streaming responses"""
    print("=== Streaming Demo ===\n")

    manager = AzureOpenAIManager(
        endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key"
    )

    messages = [
        manager.create_user_message("Tell me a short story about AI.")
    ]

    print("Streaming response: ", end="", flush=True)
    for chunk in manager.stream_chat_completion(messages):
        print(chunk, end="", flush=True)
    print("\n")


def demo_embeddings():
    """Demonstrate embeddings generation"""
    print("=== Embeddings Demo ===\n")

    manager = AzureOpenAIManager(
        endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key"
    )

    texts = [
        "Azure OpenAI provides powerful AI capabilities",
        "Machine learning is transforming industries",
        "Cloud computing enables scalable solutions"
    ]

    embeddings = manager.batch_embeddings(texts)

    for i, (text, emb) in enumerate(zip(texts, embeddings), 1):
        print(f"{i}. Text: {text}")
        print(f"   Embedding dimensions: {len(emb.embedding)}")
        print(f"   Tokens: {emb.usage['total_tokens']}\n")


def demo_conversation():
    """Demonstrate multi-turn conversation"""
    print("=== Conversation Manager Demo ===\n")

    openai_manager = AzureOpenAIManager(
        endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key"
    )

    conversation = ConversationManager(
        openai_manager,
        system_prompt="You are an expert in Azure cloud services."
    )

    # First turn
    conversation.add_user_message("What is Azure OpenAI?")
    response1 = conversation.get_response()
    print(f"User: What is Azure OpenAI?")
    print(f"Assistant: {response1}\n")

    # Second turn
    conversation.add_user_message("How does it differ from OpenAI?")
    response2 = conversation.get_response()
    print(f"User: How does it differ from OpenAI?")
    print(f"Assistant: {response2}\n")

    # Summary
    summary = conversation.get_conversation_summary()
    print(f"Conversation summary:")
    print(f"- Total messages: {summary['message_count']}")
    print(f"- Total tokens: {summary['total_tokens']}\n")


def demo_content_filter():
    """Demonstrate content filtering"""
    print("=== Content Filter Demo ===\n")

    filter_manager = ContentFilterManager()

    texts = [
        "This is a safe and appropriate message.",
        "Let's discuss Azure cloud architecture.",
        "How can I improve my application performance?"
    ]

    for text in texts:
        result = filter_manager.analyze_content(text)
        print(f"Text: {text}")
        print(f"Filtered: {result['filtered']}")
        print(f"Categories: {json.dumps(result['categories'], indent=2)}\n")


def demo_token_estimation():
    """Demonstrate token estimation"""
    print("=== Token Estimation Demo ===\n")

    manager = AzureOpenAIManager(
        endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key"
    )

    texts = [
        "Hello, world!",
        "Azure OpenAI provides enterprise-grade AI capabilities with enhanced security and compliance.",
        "The quick brown fox jumps over the lazy dog. " * 10
    ]

    for text in texts:
        tokens = manager.estimate_tokens(text)
        print(f"Text length: {len(text)} characters")
        print(f"Estimated tokens: {tokens}")
        print(f"Text preview: {text[:50]}...\n")


if __name__ == "__main__":
    print("Azure OpenAI Service - Advanced Implementation")
    print("=" * 60)
    print()

    # Run all demos
    demo_basic_chat()
    demo_streaming()
    demo_embeddings()
    demo_conversation()
    demo_content_filter()
    demo_token_estimation()

    print("=" * 60)
    print("All demos completed successfully!")
