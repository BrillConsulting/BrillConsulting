# Content Moderation Pipeline

Multi-layered content moderation system combining OpenAI Moderation API with local filters for comprehensive safety.

## Features

- **OpenAI Moderation API** - Cloud-based content safety
- **Local ML Filters** - Privacy-preserving on-premise filtering
- **Multi-Language Support** - Detect harmful content in 50+ languages
- **Custom Taxonomies** - Define organization-specific moderation rules
- **Real-time Processing** - <100ms latency
- **Severity Scoring** - Granular risk assessment
- **Appeal System** - Human review workflow
- **Audit Trail** - Complete moderation history

## Moderation Categories

| Category | Description | Action |
|----------|-------------|--------|
| **Hate Speech** | Discriminatory or hateful content | Block |
| **Violence** | Violent or graphic content | Block |
| **Sexual Content** | Adult or explicit material | Block/Warn |
| **Self-Harm** | Content promoting self-harm | Block + Alert |
| **Harassment** | Bullying or targeted harassment | Block |
| **Spam** | Repetitive or promotional content | Flag |

## Usage

```python
from content_moderation import ModerationPipeline

# Initialize pipeline
pipeline = ModerationPipeline(
    use_openai=True,
    use_local_models=True,
    severity_threshold=0.7
)

# Moderate content
text = "User-generated content to check"

result = pipeline.moderate(text)

if result.is_flagged:
    print(f"Content flagged: {result.categories}")
    print(f"Severity: {result.severity_score}")
    print(f"Action: {result.recommended_action}")
```

## Technologies

- OpenAI Moderation API
- HuggingFace Transformers
- Detoxify
- Custom ML models
- Redis (caching)
