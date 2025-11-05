# Azure OpenAI Service Integration

Advanced implementation of Azure OpenAI Service with GPT-4, embeddings, streaming, and function calling capabilities.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a comprehensive Python implementation for Azure OpenAI Service, featuring chat completions, embeddings generation, streaming responses, function calling, and content filtering. Built for enterprise applications requiring scalable AI capabilities with Azure's security and compliance features.

## Features

### Core Capabilities
- **GPT-4 & GPT-3.5 Integration**: Support for multiple Azure OpenAI models
- **Chat Completions**: Multi-turn conversational AI
- **Embeddings**: Text-to-vector conversion for semantic search and similarity
- **Streaming Responses**: Real-time token generation
- **Function Calling**: Tool use and structured outputs
- **Content Filtering**: Built-in safety and moderation
- **Token Management**: Usage tracking and estimation

### Advanced Features
- **Conversation Manager**: Context-aware multi-turn conversations
- **Content Filter Manager**: Safety checks and policy compliance
- **Batch Processing**: Efficient bulk embeddings generation
- **Model Selection**: Support for GPT-4, GPT-4-32K, GPT-3.5 variants
- **Parameter Control**: Temperature, top_p, frequency/presence penalties
- **Message History**: Conversation tracking and management

## Architecture

```
AzureOpenAI/
├── azure_open_a_i.py          # Main implementation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

### Key Components

1. **AzureOpenAIManager**: Main service interface
   - Chat completion creation
   - Embeddings generation
   - Streaming responses
   - Token estimation

2. **ConversationManager**: Multi-turn conversation handler
   - System prompt configuration
   - Message history management
   - Conversation summarization

3. **ContentFilterManager**: Safety and moderation
   - Content policy enforcement
   - Category-based filtering
   - Severity assessment

4. **Data Classes**:
   - `Message`: Chat message structure
   - `CompletionResponse`: API response format
   - `EmbeddingResponse`: Embedding result format

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/AzureOpenAI

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your Azure OpenAI credentials:

```python
from azure_open_a_i import AzureOpenAIManager

manager = AzureOpenAIManager(
    endpoint="https://your-resource.openai.azure.com",
    api_key="your-api-key",
    api_version="2024-02-01"
)
```

### Environment Variables (Recommended)

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_VERSION="2024-02-01"
```

## Usage Examples

### Basic Chat Completion

```python
from azure_open_a_i import AzureOpenAIManager

manager = AzureOpenAIManager(
    endpoint="https://your-resource.openai.azure.com",
    api_key="your-api-key"
)

# Create messages
messages = [
    manager.create_system_message("You are a helpful AI assistant."),
    manager.create_user_message("What are the benefits of Azure OpenAI?")
]

# Get completion
response = manager.create_chat_completion(messages)

print(f"Response: {response.content}")
print(f"Tokens: {response.usage['total_tokens']}")
```

### Streaming Responses

```python
messages = [
    manager.create_user_message("Tell me about cloud computing.")
]

print("Streaming response: ", end="", flush=True)
for chunk in manager.stream_chat_completion(messages):
    print(chunk, end="", flush=True)
```

### Embeddings Generation

```python
# Single embedding
text = "Azure OpenAI provides enterprise AI capabilities"
embedding = manager.create_embeddings(text)
print(f"Embedding dimensions: {len(embedding.embedding)}")

# Batch embeddings
texts = [
    "Machine learning is transforming industries",
    "Cloud computing enables scalable solutions",
    "AI is revolutionizing business processes"
]

embeddings = manager.batch_embeddings(texts)
for text, emb in zip(texts, embeddings):
    print(f"Text: {text}")
    print(f"Dimensions: {len(emb.embedding)}")
```

### Multi-Turn Conversation

```python
from azure_open_a_i import ConversationManager

conversation = ConversationManager(
    manager,
    system_prompt="You are an expert in Azure cloud services."
)

# First turn
conversation.add_user_message("What is Azure OpenAI?")
response1 = conversation.get_response()
print(f"Assistant: {response1}")

# Second turn with context
conversation.add_user_message("How does it differ from OpenAI?")
response2 = conversation.get_response()
print(f"Assistant: {response2}")

# Get conversation summary
summary = conversation.get_conversation_summary()
print(f"Total messages: {summary['message_count']}")
```

### Function Calling

```python
functions = [
    {
        "name": "get_weather",
        "description": "Get weather information for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
]

messages = [
    manager.create_user_message("What's the weather in Seattle?")
]

response = manager.create_function_call(
    messages=messages,
    functions=functions,
    function_call="auto"
)

print(f"Function called: {response.finish_reason}")
```

### Content Filtering

```python
from azure_open_a_i import ContentFilterManager

filter_manager = ContentFilterManager()

text = "This is a sample message to check."
result = filter_manager.analyze_content(text)

print(f"Filtered: {result['filtered']}")
for category, details in result['categories'].items():
    print(f"{category}: {details['severity']}")
```

### Advanced Parameters

```python
response = manager.create_chat_completion(
    messages=messages,
    model=ModelType.GPT_4,
    temperature=0.7,           # Creativity level (0-2)
    max_tokens=1000,           # Maximum response length
    top_p=0.95,               # Nucleus sampling
    frequency_penalty=0.5,     # Reduce repetition
    presence_penalty=0.3,      # Encourage topic diversity
    stop=["\n\n", "END"]      # Stop sequences
)
```

## Running Demos

```bash
# Run all demo functions
python azure_open_a_i.py
```

Demo output includes:
- Basic chat completion
- Streaming responses
- Embeddings generation
- Multi-turn conversations
- Content filtering
- Token estimation

## Model Types

### Chat Models
- **gpt-4**: Most capable model, best quality
- **gpt-4-32k**: Extended context window (32K tokens)
- **gpt-35-turbo**: Fast and cost-effective
- **gpt-35-turbo-16k**: Extended context (16K tokens)

### Embedding Models
- **text-embedding-ada-002**: 1536-dimensional embeddings

## Token Management

### Estimation
```python
text = "Your text here"
estimated_tokens = manager.estimate_tokens(text)
print(f"Estimated tokens: {estimated_tokens}")
```

### Usage Tracking
```python
response = manager.create_chat_completion(messages)
usage = response.usage

print(f"Prompt tokens: {usage['prompt_tokens']}")
print(f"Completion tokens: {usage['completion_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")
```

## Best Practices

### 1. System Messages
Always include a system message to set the behavior:
```python
messages = [
    manager.create_system_message("You are a helpful, precise assistant."),
    manager.create_user_message("Your question here")
]
```

### 2. Temperature Control
- **0.0-0.3**: Focused, deterministic responses
- **0.4-0.7**: Balanced creativity and consistency
- **0.8-2.0**: High creativity, more randomness

### 3. Context Management
Keep conversation history manageable:
```python
# Clear history periodically
manager.clear_conversation()

# Or manage token count
conversation_history = manager.get_conversation_history()
total_tokens = sum(len(msg['content'].split()) for msg in conversation_history)
```

### 4. Error Handling
```python
try:
    response = manager.create_chat_completion(messages)
except Exception as e:
    print(f"Error: {e}")
    # Implement retry logic or fallback
```

### 5. Content Safety
Always check content before and after API calls:
```python
filter_manager = ContentFilterManager()
user_input_check = filter_manager.analyze_content(user_message)

if not user_input_check['filtered']:
    response = manager.create_chat_completion(messages)
    output_check = filter_manager.analyze_content(response.content)
```

## Use Cases

### 1. Customer Support Chatbot
```python
conversation = ConversationManager(
    manager,
    system_prompt="You are a customer support agent. Be helpful and professional."
)
```

### 2. Semantic Search
```python
# Generate embeddings for documents
doc_embeddings = manager.batch_embeddings(documents)

# Generate query embedding
query_embedding = manager.create_embeddings(user_query)

# Compute similarity and rank results
```

### 3. Content Generation
```python
response = manager.create_chat_completion(
    messages=[
        manager.create_system_message("You are a creative writer."),
        manager.create_user_message("Write a product description for...")
    ],
    temperature=0.8
)
```

### 4. Code Assistance
```python
response = manager.create_chat_completion(
    messages=[
        manager.create_system_message("You are an expert programmer."),
        manager.create_user_message("Explain this code: ...")
    ],
    temperature=0.3
)
```

## API Reference

### AzureOpenAIManager

#### Methods

**`create_chat_completion(messages, model, temperature, max_tokens, ...)`**
- Creates a chat completion
- Returns: `CompletionResponse`

**`stream_chat_completion(messages, model, temperature)`**
- Streams chat completion
- Yields: Content chunks

**`create_embeddings(text, model)`**
- Generates text embeddings
- Returns: `EmbeddingResponse`

**`batch_embeddings(texts, model)`**
- Generates embeddings for multiple texts
- Returns: `List[EmbeddingResponse]`

**`create_function_call(messages, functions, function_call)`**
- Creates completion with function calling
- Returns: `CompletionResponse`

**`estimate_tokens(text)`**
- Estimates token count
- Returns: `int`

### ConversationManager

#### Methods

**`add_user_message(content)`**
- Adds user message to conversation

**`get_response(temperature, max_tokens)`**
- Gets assistant response
- Returns: `str`

**`get_conversation_summary()`**
- Returns conversation summary
- Returns: `Dict[str, Any]`

## Performance Optimization

### 1. Batch Processing
```python
# Process multiple texts efficiently
embeddings = manager.batch_embeddings(texts)
```

### 2. Token Optimization
```python
# Estimate before calling API
estimated = manager.estimate_tokens(long_text)
if estimated > 8000:
    # Truncate or split text
    pass
```

### 3. Caching
```python
# Cache embeddings for frequently used texts
embedding_cache = {}
if text not in embedding_cache:
    embedding_cache[text] = manager.create_embeddings(text)
```

## Security Considerations

1. **API Key Protection**: Never commit API keys to version control
2. **Content Filtering**: Always validate inputs and outputs
3. **Rate Limiting**: Implement request throttling
4. **Data Privacy**: Handle PII according to regulations
5. **Logging**: Log requests but sanitize sensitive data

## Troubleshooting

### Common Issues

**Issue**: Rate limit errors
**Solution**: Implement exponential backoff retry logic

**Issue**: Token limit exceeded
**Solution**: Truncate input or use summarization

**Issue**: Content filtered
**Solution**: Review and modify input to comply with policies

**Issue**: High latency
**Solution**: Use streaming or smaller models

## Deployment

### Azure Deployment
```bash
# Use managed identity
az identity create --name openai-app-identity

# Assign permissions
az role assignment create \
    --role "Cognitive Services OpenAI User" \
    --assignee <identity-id> \
    --scope <openai-resource-id>
```

### Container Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY azure_open_a_i.py .
CMD ["python", "azure_open_a_i.py"]
```

## Monitoring

### Key Metrics
- Request latency
- Token usage
- Error rates
- Content filter triggers
- Model usage distribution

### Azure Monitor Integration
```python
# Add application insights
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=...'
))
```

## Dependencies

```
Python >= 3.8
dataclasses
typing
json
asyncio
```

See `requirements.txt` for complete list.

## Version History

- **v1.0.0**: Initial release with basic features
- **v2.0.0**: Added streaming, function calling, and content filtering
- **v2.1.0**: Enhanced conversation management and batch processing

## Contributing

Contributions are welcome! Please submit pull requests or open issues on GitHub.

## License

This project is part of the Brill Consulting portfolio.

## Support

For questions or support:
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure AI Services](../AzureAI/)
- [Cognitive Services](../CognitiveServices/)
- [Azure Machine Learning](../MachineLearning/)

---

**Built with Azure OpenAI Service** | **Brill Consulting © 2024**
