# LLM Chatbot System

Conversational AI system with LLM integration, conversation management, and streaming support.

## Features

- **Multiple LLM Backends**: OpenAI GPT, Claude, local models (LLaMA, Mistral)
- **Conversation History**: Persistent conversation management
- **Streaming Responses**: Token-by-token streaming
- **System Prompts**: Custom assistant personalities
- **Temperature Control**: Adjust response creativity
- **Token Management**: Track and limit token usage
- **Conversation Export**: Save/load conversations

## Technologies

- OpenAI API (GPT-3.5, GPT-4)
- Transformers (local models)
- JSON for persistence

## Usage

```python
from llm_chatbot import LLMChatbot

# Initialize
chatbot = LLMChatbot(model="gpt-3.5-turbo",
                     system_prompt="You are a helpful assistant")

# Chat
response = chatbot.chat("Hello!", temperature=0.7)

# Streaming
for token in chatbot.stream_chat("Tell me a story"):
    print(token, end="")

# Save conversation
chatbot.save_conversation("conversation.json")
```

## Demo

```bash
python llm_chatbot.py
```

**Note:** For production use, add actual OpenAI API or local model integration.
