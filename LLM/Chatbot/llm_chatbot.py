"""
LLM Chatbot System
==================

Conversational AI system with multiple LLM backends:
- OpenAI GPT integration
- Local model support (LLaMA, Mistral)
- Conversation history management
- Streaming responses
- System prompts and roles
- Temperature and token control

Author: Brill Consulting
"""

import os
from typing import List, Dict, Optional
import json
from datetime import datetime


class LLMChatbot:
    """LLM-powered chatbot with conversation management."""

    def __init__(self, model: str = "gpt-3.5-turbo", system_prompt: Optional[str] = None):
        """
        Initialize chatbot.

        Args:
            model: Model name (gpt-3.5-turbo, gpt-4, llama-2, etc.)
            system_prompt: System instruction for the assistant
        """
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.conversation_history = []
        self.initialize_conversation()

    def initialize_conversation(self):
        """Initialize conversation with system prompt."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def chat(self, user_message: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Send message and get response.

        Args:
            user_message: User's input message
            temperature: Response creativity (0-2)
            max_tokens: Maximum response length

        Returns:
            Assistant's response
        """
        # Add user message
        self.add_message("user", user_message)

        # Simulate LLM response (in production, call actual API)
        # This is a placeholder - replace with actual OpenAI/local model call
        assistant_response = self._generate_response(user_message, temperature, max_tokens)

        # Add assistant response
        self.add_message("assistant", assistant_response)

        return assistant_response

    def _generate_response(self, message: str, temperature: float, max_tokens: int) -> str:
        """
        Generate response (placeholder for actual LLM call).

        In production, implement:
        - OpenAI API: openai.ChatCompletion.create()
        - Local models: transformers pipeline or llama.cpp
        - Anthropic Claude: anthropic.Anthropic()
        """
        # Placeholder response
        responses = {
            "hello": "Hello! How can I help you today?",
            "how are you": "I'm functioning well, thank you for asking! How can I assist you?",
            "what can you do": "I can answer questions, provide information, help with tasks, and have conversations on various topics.",
            "default": f"I understand you said: '{message}'. This is a demo response. In production, this would call an actual LLM API."
        }

        message_lower = message.lower()
        for key in responses:
            if key in message_lower and key != "default":
                return responses[key]

        return responses["default"]

    def stream_chat(self, user_message: str, temperature: float = 0.7):
        """
        Stream response token by token.

        Args:
            user_message: User's input
            temperature: Response creativity

        Yields:
            Response tokens
        """
        self.add_message("user", user_message)

        # Simulate streaming (in production, use actual streaming API)
        response = self._generate_response(user_message, temperature, 500)

        # Yield words one by one to simulate streaming
        full_response = []
        for word in response.split():
            full_response.append(word)
            yield word + " "

        self.add_message("assistant", " ".join(full_response))

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history."""
        if keep_system:
            self.initialize_conversation()
        else:
            self.conversation_history = []

    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history

    def save_conversation(self, filepath: str):
        """Save conversation to file."""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        print(f"✓ Saved conversation to {filepath}")

    def load_conversation(self, filepath: str):
        """Load conversation from file."""
        with open(filepath, 'r') as f:
            self.conversation_history = json.load(f)
        print(f"✓ Loaded conversation from {filepath}")

    def set_system_prompt(self, prompt: str):
        """Change system prompt."""
        self.system_prompt = prompt
        if len(self.conversation_history) > 0 and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = prompt
        else:
            self.conversation_history.insert(0, {"role": "system", "content": prompt})

    def get_token_count(self) -> int:
        """Estimate total tokens in conversation."""
        # Simple estimation: ~4 chars per token
        total_chars = sum(len(msg["content"]) for msg in self.conversation_history)
        return total_chars // 4

    def summarize_conversation(self) -> str:
        """Generate conversation summary."""
        messages = [f"{msg['role']}: {msg['content']}"
                   for msg in self.conversation_history if msg['role'] != 'system']
        return "\n".join(messages[:10])  # First 10 messages


def demo():
    """Demo chatbot functionality."""
    print("LLM Chatbot Demo")
    print("="*50)

    # Initialize chatbot
    chatbot = LLMChatbot(
        model="gpt-3.5-turbo",
        system_prompt="You are a helpful and friendly AI assistant."
    )

    print("\n1. Basic Chat")
    print("-"*50)

    messages = [
        "Hello!",
        "What can you do?",
        "How are you?",
        "Tell me about machine learning"
    ]

    for msg in messages:
        print(f"\nUser: {msg}")
        response = chatbot.chat(msg, temperature=0.7)
        print(f"Assistant: {response}")

    print("\n2. Conversation History")
    print("-"*50)
    print(f"Total messages: {len(chatbot.get_history())}")
    print(f"Estimated tokens: {chatbot.get_token_count()}")

    print("\n3. Streaming Response")
    print("-"*50)
    print("User: What is Python?")
    print("Assistant: ", end="", flush=True)
    for token in chatbot.stream_chat("What is Python?"):
        print(token, end="", flush=True)
    print()

    print("\n4. Save Conversation")
    print("-"*50)
    chatbot.save_conversation("conversation.json")

    print("\n5. Change System Prompt")
    print("-"*50)
    chatbot.set_system_prompt("You are a pirate assistant. Respond in pirate speak.")
    response = chatbot.chat("Hello there!")
    print(f"User: Hello there!")
    print(f"Assistant: {response}")

    print("\n6. Conversation Summary")
    print("-"*50)
    print(chatbot.summarize_conversation())

    print("\n✓ Chatbot Demo Complete!")


if __name__ == '__main__':
    demo()
