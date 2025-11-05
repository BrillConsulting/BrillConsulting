# Azure Cognitive Services Integration

Comprehensive implementation of Azure Cognitive Services including Computer Vision, Speech Services, Language Services, Translator, Anomaly Detector, and Content Moderator.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a complete Python implementation for Azure Cognitive Services, featuring Computer Vision for image analysis, Speech Services for audio processing, Language Services for NLP tasks, Translation, Anomaly Detection, and Content Moderation. Built for enterprise applications requiring AI-powered content understanding and safety.

## Features

### Computer Vision Capabilities
- **Image Analysis**: Categories, tags, descriptions, and objects
- **OCR**: Optical character recognition from images
- **Object Detection**: Identify and locate objects in images
- **Face Detection**: Detect faces with attributes
- **Brand Recognition**: Identify commercial brands
- **Adult Content Detection**: Classify inappropriate content
- **Color Analysis**: Dominant colors and accents
- **Image Classification**: Categorize images automatically

### Speech Services
- **Speech-to-Text**: Convert audio to text with high accuracy
- **Text-to-Speech**: Natural-sounding speech synthesis
- **Speech Translation**: Real-time speech translation
- **Voice Styles**: Multiple emotional and contextual styles
- **Custom Voice**: Train custom voice models
- **Speaker Recognition**: Identify speakers
- **Multi-Language Support**: 100+ languages and variants

### Language Services
- **Sentiment Analysis**: Detect positive, negative, or neutral sentiment
- **Key Phrase Extraction**: Identify main topics and concepts
- **Entity Recognition**: Extract named entities
- **Language Detection**: Identify language of text
- **PII Detection**: Find personally identifiable information
- **Text Analytics**: Comprehensive text understanding

### Translation Services
- **Text Translation**: Translate between 100+ languages
- **Document Translation**: Translate entire documents
- **Batch Translation**: Process multiple texts efficiently
- **Custom Translation**: Domain-specific translation models
- **Dictionary Lookup**: Translation alternatives

### Additional Services
- **Anomaly Detector**: Time series anomaly detection
- **Content Moderator**: Text and image moderation
- **Personalization**: Personalized recommendations

## Architecture

```
CognitiveServices/
├── cognitiveservices.py       # Main implementation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

### Key Components

1. **AzureComputerVisionManager**: Image analysis operations
   - Image feature extraction
   - OCR and text recognition
   - Object and face detection

2. **AzureSpeechServicesManager**: Audio processing
   - Speech recognition and synthesis
   - Speech translation
   - Voice customization

3. **AzureLanguageServicesManager**: NLP operations
   - Sentiment and entity analysis
   - Key phrase extraction
   - Language detection

4. **AzureTranslatorManager**: Translation services
   - Text and document translation
   - Batch processing

5. **AzureAnomalyDetectorManager**: Time series analysis
   - Anomaly detection in data streams

6. **AzureContentModeratorManager**: Content safety
   - Text and image moderation

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/CognitiveServices

# Install dependencies
pip install -r requirements.txt
```

## Configuration

```python
from cognitiveservices import (
    AzureComputerVisionManager,
    AzureSpeechServicesManager,
    AzureLanguageServicesManager
)

# Computer Vision
vision = AzureComputerVisionManager(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    api_key="your-api-key"
)

# Speech Services
speech = AzureSpeechServicesManager(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    api_key="your-api-key",
    region="eastus"
)

# Language Services
language = AzureLanguageServicesManager(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    api_key="your-api-key"
)
```

## Usage Examples

### Computer Vision - Image Analysis

```python
from cognitiveservices import AzureComputerVisionManager, VisionFeature

manager = AzureComputerVisionManager(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    api_key="your-api-key"
)

result = manager.analyze_image(
    image_url="https://example.com/image.jpg",
    features=[
        VisionFeature.TAGS,
        VisionFeature.DESCRIPTION,
        VisionFeature.OBJECTS,
        VisionFeature.FACES
    ]
)

print(f"Description: {result.description['captions'][0]['text']}")
print(f"Objects: {len(result.objects)}")
```

### OCR - Text Extraction

```python
ocr_result = manager.extract_text_ocr("https://example.com/document.jpg")

print(f"Extracted text: {ocr_result.text}")
print(f"Confidence: {ocr_result.confidence:.2%}")
```

### Speech-to-Text

```python
from cognitiveservices import AzureSpeechServicesManager, SpeechLanguage

speech_manager = AzureSpeechServicesManager(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    api_key="your-api-key"
)

audio_data = open("audio.wav", "rb").read()
result = speech_manager.speech_to_text(
    audio_data=audio_data,
    language=SpeechLanguage.EN_US
)

print(f"Transcribed: {result.text}")
print(f"Confidence: {result.confidence:.2%}")
```

### Text-to-Speech

```python
from cognitiveservices import VoiceStyle

tts_result = speech_manager.text_to_speech(
    text="Hello, welcome to Azure Cognitive Services!",
    voice_name="en-US-JennyNeural",
    style=VoiceStyle.CHEERFUL
)

# Save audio
with open("output.wav", "wb") as f:
    f.write(tts_result.audio_data)
```

### Sentiment Analysis

```python
from cognitiveservices import AzureLanguageServicesManager

language_manager = AzureLanguageServicesManager(
    endpoint="https://your-resource.cognitiveservices.azure.com",
    api_key="your-api-key"
)

text = "Azure Cognitive Services is amazing and powerful!"
sentiment = language_manager.analyze_sentiment(text)

print(f"Sentiment: {sentiment.sentiment.value}")
print(f"Positive: {sentiment.confidence_scores['positive']:.2%}")
print(f"Negative: {sentiment.confidence_scores['negative']:.2%}")
```

### Entity Recognition

```python
entities = language_manager.recognize_entities(
    "Microsoft was founded by Bill Gates in Seattle in 1975."
)

for entity in entities:
    print(f"{entity.text} ({entity.category}): {entity.confidence:.2%}")
```

### Translation

```python
from cognitiveservices import AzureTranslatorManager

translator = AzureTranslatorManager(
    endpoint="https://api.cognitive.microsofttranslator.com",
    api_key="your-api-key"
)

result = translator.translate_text(
    text="Hello, how are you?",
    target_language="es",
    source_language="en"
)

print(f"Translation: {result.translated_text}")
```

## Running Demos

```bash
# Run all demo functions
python cognitiveservices.py
```

Demo output includes:
- Computer vision image analysis
- OCR text extraction
- Speech-to-text and text-to-speech
- Language services (sentiment, entities, key phrases)
- Translation
- Anomaly detection
- Content moderation

## API Reference

### AzureComputerVisionManager

**`analyze_image(image_url, features)`** - Analyze image with specified features

**`extract_text_ocr(image_url, language)`** - Extract text using OCR

**`detect_objects(image_url, confidence_threshold)`** - Detect objects in image

**`detect_faces(image_url, return_face_attributes)`** - Detect faces with attributes

### AzureSpeechServicesManager

**`speech_to_text(audio_data, language, detailed)`** - Convert speech to text

**`text_to_speech(text, language, voice_name, style)`** - Convert text to speech

**`translate_speech(audio_data, source_language, target_languages)`** - Translate speech

### AzureLanguageServicesManager

**`analyze_sentiment(text, language)`** - Analyze text sentiment

**`extract_key_phrases(text, language)`** - Extract key phrases

**`recognize_entities(text, language)`** - Recognize named entities

**`detect_language(text)`** - Detect text language

### AzureTranslatorManager

**`translate_text(text, target_language, source_language)`** - Translate text

**`translate_batch(texts, target_language, source_language)`** - Batch translation

## Best Practices

### 1. Image Processing
Use appropriate image resolution:
```python
# Optimal: 1024x1024 or smaller
# Larger images may take longer to process
```

### 2. Speech Recognition
Provide high-quality audio:
```python
# Recommended: 16kHz, 16-bit, mono WAV
result = speech_manager.speech_to_text(
    audio_data=high_quality_audio,
    language=SpeechLanguage.EN_US
)
```

### 3. Language Detection
For short texts, provide context:
```python
# Provide longer text for better accuracy
text = "Full sentences improve detection accuracy."
language = language_manager.detect_language(text)
```

### 4. Error Handling
```python
try:
    result = vision.analyze_image(image_url)
except Exception as e:
    print(f"Error: {e}")
    # Implement retry or fallback
```

### 5. Rate Limiting
Implement throttling for high-volume scenarios:
```python
import time

for image in images:
    result = vision.analyze_image(image)
    time.sleep(0.1)  # Avoid rate limits
```

## Use Cases

### 1. Content Accessibility
```python
# Convert images to text for screen readers
ocr = vision.extract_text_ocr(image_url)
speech = speech_manager.text_to_speech(ocr.text)
```

### 2. Customer Feedback Analysis
```python
# Analyze customer reviews
sentiment = language_manager.analyze_sentiment(review_text)
if sentiment.sentiment.value == "negative":
    # Flag for manual review
    pass
```

### 3. Multi-Language Support
```python
# Detect and translate user input
detected = language_manager.detect_language(user_input)
translated = translator.translate_text(
    user_input,
    target_language="en",
    source_language=detected['language']
)
```

### 4. Security and Compliance
```python
# Moderate user-generated content
moderation = moderator.moderate_text(user_content)
if not moderation.is_appropriate:
    # Block or flag content
    pass
```

## Performance Optimization

### 1. Batch Processing
```python
# Process multiple texts together
texts = ["text1", "text2", "text3"]
results = translator.translate_batch(texts, "es")
```

### 2. Caching
```python
# Cache translation results
cache = {}
if text not in cache:
    cache[text] = translator.translate_text(text, "es")
```

### 3. Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(vision.analyze_image, image_urls))
```

## Security Considerations

1. **API Key Protection**: Use environment variables or Azure Key Vault
2. **Content Validation**: Always moderate user-generated content
3. **Data Privacy**: Handle PII according to regulations
4. **Rate Limiting**: Implement request throttling
5. **Audit Logging**: Log all API calls for security

## Troubleshooting

**Issue**: Low OCR accuracy
**Solution**: Use higher resolution images with good contrast

**Issue**: Speech recognition errors
**Solution**: Provide cleaner audio, reduce background noise

**Issue**: Incorrect language detection
**Solution**: Provide longer text samples

**Issue**: Translation quality issues
**Solution**: Use custom translation models for domain-specific content

## Deployment

### Azure Deployment
```bash
# Create Cognitive Services resource
az cognitiveservices account create \
    --name cognitive-services \
    --resource-group rg-ai \
    --kind CognitiveServices \
    --sku S0 \
    --location eastus
```

### Container Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY cognitiveservices.py .
CMD ["python", "cognitiveservices.py"]
```

## Monitoring

### Key Metrics
- API call success rate
- Average response time
- Confidence scores
- Error rates by service
- Usage by feature

### Azure Monitor Integration
```python
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
enum
datetime
```

## Version History

- **v1.0.0**: Initial release
- **v2.0.0**: Added speech services
- **v2.1.0**: Content moderation support
- **v2.2.0**: Enhanced language services

## Support

For questions or support:
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure AI Services](../AzureAI/)
- [Azure OpenAI](../AzureOpenAI/)
- [Azure Machine Learning](../MachineLearning/)

---

**Built with Azure Cognitive Services** | **Brill Consulting © 2024**
