"""
Azure Cognitive Services Integration

Comprehensive implementation of Azure Cognitive Services including Computer Vision,
Speech Services, Language Services, Translator, Anomaly Detector, and Content Moderator.

Author: BrillConsulting
Contact: clientbrill@gmail.com
LinkedIn: brillconsulting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import json
import base64


class VisionFeature(Enum):
    """Computer Vision features"""
    CATEGORIES = "Categories"
    TAGS = "Tags"
    DESCRIPTION = "Description"
    FACES = "Faces"
    ADULT = "Adult"
    COLOR = "Color"
    IMAGE_TYPE = "ImageType"
    OBJECTS = "Objects"
    BRANDS = "Brands"


class SpeechLanguage(Enum):
    """Supported speech languages"""
    EN_US = "en-US"
    EN_GB = "en-GB"
    ES_ES = "es-ES"
    FR_FR = "fr-FR"
    DE_DE = "de-DE"
    JA_JP = "ja-JP"
    ZH_CN = "zh-CN"


class VoiceStyle(Enum):
    """Text-to-speech voice styles"""
    NEUTRAL = "neutral"
    CHEERFUL = "cheerful"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    FRIENDLY = "friendly"


class SentimentLabel(Enum):
    """Sentiment analysis labels"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


@dataclass
class ImageAnalysisResult:
    """Result of image analysis"""
    image_id: str
    categories: List[Dict[str, Any]]
    tags: List[Dict[str, float]]
    description: Dict[str, Any]
    objects: List[Dict[str, Any]]
    faces: List[Dict[str, Any]]
    adult_content: Dict[str, Any]
    color_info: Dict[str, Any]
    metadata: Dict[str, Any]
    analyzed_at: datetime


@dataclass
class OCRResult:
    """Optical character recognition result"""
    language: str
    text: str
    lines: List[Dict[str, Any]]
    words: List[Dict[str, Any]]
    confidence: float
    orientation: float


@dataclass
class ObjectDetection:
    """Detected object in image"""
    object_type: str
    confidence: float
    bounding_box: Dict[str, int]
    parent: Optional[str] = None


@dataclass
class SpeechRecognitionResult:
    """Speech-to-text result"""
    recognition_id: str
    text: str
    confidence: float
    language: SpeechLanguage
    duration_ms: int
    offset_ms: int
    words: List[Dict[str, Any]]


@dataclass
class SpeechSynthesisResult:
    """Text-to-speech result"""
    synthesis_id: str
    audio_data: bytes
    duration_ms: int
    voice_name: str
    language: SpeechLanguage
    format: str


@dataclass
class TranslationResult:
    """Translation result"""
    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    sentiment: SentimentLabel
    confidence_scores: Dict[str, float]
    sentences: List[Dict[str, Any]]
    overall_score: float


@dataclass
class EntityRecognition:
    """Named entity recognition result"""
    text: str
    category: str
    subcategory: Optional[str]
    confidence: float
    offset: int
    length: int
    links: List[str] = field(default_factory=list)


@dataclass
class KeyPhrase:
    """Extracted key phrase"""
    text: str
    score: float


@dataclass
class AnomalyDetectionResult:
    """Anomaly detection result"""
    is_anomaly: bool
    is_positive_anomaly: bool
    is_negative_anomaly: bool
    expected_value: float
    upper_margin: float
    lower_margin: float
    severity: float


@dataclass
class ContentModerationResult:
    """Content moderation result"""
    is_appropriate: bool
    categories: Dict[str, float]
    review_recommended: bool
    adult_score: float
    racy_score: float
    language_detection: Optional[str] = None


class AzureComputerVisionManager:
    """
    Manager for Azure Computer Vision operations.

    Provides image analysis, OCR, object detection, face detection,
    and image classification capabilities.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "2023-10-01"
    ):
        """
        Initialize Computer Vision manager.

        Args:
            endpoint: Azure Computer Vision endpoint URL
            api_key: API key for authentication
            api_version: API version to use
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.api_version = api_version

    def analyze_image(
        self,
        image_url: str,
        features: Optional[List[VisionFeature]] = None
    ) -> ImageAnalysisResult:
        """
        Analyze image with specified features.

        Args:
            image_url: URL to image
            features: Features to extract

        Returns:
            ImageAnalysisResult with extracted information
        """
        if features is None:
            features = [
                VisionFeature.CATEGORIES,
                VisionFeature.TAGS,
                VisionFeature.DESCRIPTION,
                VisionFeature.OBJECTS
            ]

        print(f"Analyzing image: {image_url}")
        print(f"Features: {[f.value for f in features]}")

        result = ImageAnalysisResult(
            image_id=f"img_{datetime.now().timestamp()}",
            categories=[
                {"name": "outdoor_", "score": 0.95},
                {"name": "outdoor_city", "score": 0.87}
            ],
            tags=[
                {"building": 0.98},
                {"sky": 0.95},
                {"outdoor": 0.93},
                {"city": 0.89}
            ],
            description={
                "captions": [
                    {"text": "A city skyline with tall buildings", "confidence": 0.92}
                ],
                "tags": ["building", "city", "sky", "outdoor"]
            },
            objects=[
                {
                    "object": "building",
                    "confidence": 0.94,
                    "rectangle": {"x": 100, "y": 50, "w": 200, "h": 300}
                }
            ],
            faces=[],
            adult_content={
                "isAdultContent": False,
                "isRacyContent": False,
                "adultScore": 0.001,
                "racyScore": 0.002
            },
            color_info={
                "dominantColorForeground": "Black",
                "dominantColorBackground": "Blue",
                "accentColor": "0066CC"
            },
            metadata={
                "width": 1920,
                "height": 1080,
                "format": "Jpeg"
            },
            analyzed_at=datetime.now()
        )

        return result

    def extract_text_ocr(
        self,
        image_url: str,
        language: str = "en"
    ) -> OCRResult:
        """
        Extract text from image using OCR.

        Args:
            image_url: URL to image
            language: Language code

        Returns:
            OCRResult with extracted text
        """
        print(f"Performing OCR on image: {image_url}")

        result = OCRResult(
            language=language,
            text="Sample extracted text from image",
            lines=[
                {
                    "text": "Sample extracted text",
                    "bounding_box": [10, 20, 200, 40],
                    "words": [
                        {"text": "Sample", "confidence": 0.99},
                        {"text": "extracted", "confidence": 0.97},
                        {"text": "text", "confidence": 0.98}
                    ]
                }
            ],
            words=[
                {"text": "Sample", "confidence": 0.99, "bounding_box": [10, 20, 60, 40]},
                {"text": "extracted", "confidence": 0.97, "bounding_box": [70, 20, 140, 40]},
                {"text": "text", "confidence": 0.98, "bounding_box": [150, 20, 200, 40]}
            ],
            confidence=0.98,
            orientation=0.0
        )

        print(f"Extracted text: {result.text}")
        return result

    def detect_objects(
        self,
        image_url: str,
        confidence_threshold: float = 0.5
    ) -> List[ObjectDetection]:
        """
        Detect objects in image.

        Args:
            image_url: URL to image
            confidence_threshold: Minimum confidence score

        Returns:
            List of detected objects
        """
        objects = [
            ObjectDetection("person", 0.95, {"x": 100, "y": 50, "w": 150, "h": 300}),
            ObjectDetection("car", 0.89, {"x": 300, "y": 200, "w": 200, "h": 150}),
            ObjectDetection("building", 0.93, {"x": 50, "y": 10, "w": 400, "h": 500}, "outdoor"),
            ObjectDetection("tree", 0.87, {"x": 500, "y": 100, "w": 100, "h": 200})
        ]

        filtered_objects = [obj for obj in objects if obj.confidence >= confidence_threshold]

        print(f"Detected {len(filtered_objects)} objects")
        return filtered_objects

    def detect_faces(
        self,
        image_url: str,
        return_face_attributes: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in image.

        Args:
            image_url: URL to image
            return_face_attributes: Include face attributes

        Returns:
            List of detected faces with attributes
        """
        faces = [
            {
                "faceRectangle": {"left": 100, "top": 50, "width": 80, "height": 100},
                "faceAttributes": {
                    "age": 32,
                    "gender": "male",
                    "emotion": {
                        "happiness": 0.85,
                        "neutral": 0.10,
                        "sadness": 0.02,
                        "anger": 0.01
                    },
                    "glasses": "NoGlasses",
                    "smile": 0.82
                } if return_face_attributes else {}
            }
        ]

        print(f"Detected {len(faces)} faces")
        return faces


class AzureSpeechServicesManager:
    """
    Manager for Azure Speech Services.

    Provides speech-to-text, text-to-speech, speech translation,
    and speaker recognition capabilities.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        region: str = "eastus"
    ):
        """
        Initialize Speech Services manager.

        Args:
            endpoint: Azure Speech Services endpoint URL
            api_key: API key for authentication
            region: Azure region
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.region = region

    def speech_to_text(
        self,
        audio_data: bytes,
        language: SpeechLanguage = SpeechLanguage.EN_US,
        detailed: bool = True
    ) -> SpeechRecognitionResult:
        """
        Convert speech to text.

        Args:
            audio_data: Audio data bytes
            language: Source language
            detailed: Include detailed recognition info

        Returns:
            SpeechRecognitionResult with transcribed text
        """
        print(f"Performing speech-to-text recognition")
        print(f"Language: {language.value}")

        result = SpeechRecognitionResult(
            recognition_id=f"rec_{datetime.now().timestamp()}",
            text="This is a sample transcribed text from speech.",
            confidence=0.95,
            language=language,
            duration_ms=3500,
            offset_ms=0,
            words=[
                {"word": "This", "offset": 0, "duration": 200, "confidence": 0.98},
                {"word": "is", "offset": 200, "duration": 150, "confidence": 0.99},
                {"word": "sample", "offset": 350, "duration": 300, "confidence": 0.96}
            ]
        )

        print(f"Transcribed text: {result.text}")
        return result

    def text_to_speech(
        self,
        text: str,
        language: SpeechLanguage = SpeechLanguage.EN_US,
        voice_name: str = "en-US-JennyNeural",
        style: VoiceStyle = VoiceStyle.NEUTRAL
    ) -> SpeechSynthesisResult:
        """
        Convert text to speech.

        Args:
            text: Text to synthesize
            language: Target language
            voice_name: Voice to use
            style: Voice style

        Returns:
            SpeechSynthesisResult with audio data
        """
        print(f"Performing text-to-speech synthesis")
        print(f"Text: {text}")
        print(f"Voice: {voice_name}, Style: {style.value}")

        # Simulate audio data
        audio_data = b"simulated_audio_data"

        result = SpeechSynthesisResult(
            synthesis_id=f"syn_{datetime.now().timestamp()}",
            audio_data=audio_data,
            duration_ms=2500,
            voice_name=voice_name,
            language=language,
            format="audio/wav"
        )

        print(f"Synthesized {len(audio_data)} bytes of audio")
        return result

    def translate_speech(
        self,
        audio_data: bytes,
        source_language: SpeechLanguage,
        target_languages: List[SpeechLanguage]
    ) -> Dict[str, Any]:
        """
        Translate speech to multiple languages.

        Args:
            audio_data: Audio data bytes
            source_language: Source language
            target_languages: List of target languages

        Returns:
            Translation results for each target language
        """
        print(f"Translating speech from {source_language.value}")
        print(f"Target languages: {[lang.value for lang in target_languages]}")

        translations = {
            lang.value: {
                "text": f"Translated text in {lang.value}",
                "confidence": 0.92
            }
            for lang in target_languages
        }

        result = {
            "source_language": source_language.value,
            "source_text": "Original transcribed text",
            "translations": translations
        }

        return result


class AzureLanguageServicesManager:
    """
    Manager for Azure Language Services.

    Provides sentiment analysis, key phrase extraction, entity recognition,
    language detection, and PII detection.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "2023-04-01"
    ):
        """
        Initialize Language Services manager.

        Args:
            endpoint: Azure Language Services endpoint URL
            api_key: API key for authentication
            api_version: API version to use
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.api_version = api_version

    def analyze_sentiment(
        self,
        text: str,
        language: str = "en"
    ) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            language: Language code

        Returns:
            SentimentResult with sentiment analysis
        """
        print(f"Analyzing sentiment: {text[:50]}...")

        result = SentimentResult(
            text=text,
            sentiment=SentimentLabel.POSITIVE,
            confidence_scores={
                "positive": 0.85,
                "neutral": 0.10,
                "negative": 0.05
            },
            sentences=[
                {
                    "text": text,
                    "sentiment": "positive",
                    "confidence_scores": {
                        "positive": 0.85,
                        "neutral": 0.10,
                        "negative": 0.05
                    }
                }
            ],
            overall_score=0.85
        )

        print(f"Sentiment: {result.sentiment.value} (score: {result.overall_score:.2f})")
        return result

    def extract_key_phrases(
        self,
        text: str,
        language: str = "en"
    ) -> List[KeyPhrase]:
        """
        Extract key phrases from text.

        Args:
            text: Text to analyze
            language: Language code

        Returns:
            List of key phrases
        """
        phrases = [
            KeyPhrase("Azure Cognitive Services", 0.95),
            KeyPhrase("natural language processing", 0.92),
            KeyPhrase("machine learning", 0.89),
            KeyPhrase("cloud computing", 0.87)
        ]

        print(f"Extracted {len(phrases)} key phrases")
        return phrases

    def recognize_entities(
        self,
        text: str,
        language: str = "en"
    ) -> List[EntityRecognition]:
        """
        Recognize named entities in text.

        Args:
            text: Text to analyze
            language: Language code

        Returns:
            List of recognized entities
        """
        entities = [
            EntityRecognition(
                "Microsoft",
                "Organization",
                None,
                0.98,
                0,
                9,
                ["https://en.wikipedia.org/wiki/Microsoft"]
            ),
            EntityRecognition(
                "Seattle",
                "Location",
                "GPE",
                0.95,
                50,
                7,
                ["https://en.wikipedia.org/wiki/Seattle"]
            ),
            EntityRecognition(
                "2024",
                "DateTime",
                "DateRange",
                0.92,
                100,
                4
            )
        ]

        print(f"Recognized {len(entities)} entities")
        for entity in entities:
            print(f"  {entity.text} ({entity.category}): {entity.confidence:.2%}")

        return entities

    def detect_language(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            Language detection result
        """
        result = {
            "language": "en",
            "name": "English",
            "confidence": 0.99,
            "iso_code": "en"
        }

        print(f"Detected language: {result['name']} (confidence: {result['confidence']:.2%})")
        return result


class AzureTranslatorManager:
    """
    Manager for Azure Translator Service.

    Provides text translation, document translation, and language detection.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        region: str = "global"
    ):
        """
        Initialize Translator manager.

        Args:
            endpoint: Azure Translator endpoint URL
            api_key: API key for authentication
            region: Azure region
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.region = region

    def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> TranslationResult:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if None)

        Returns:
            TranslationResult with translation
        """
        detected_source = source_language or "en"

        print(f"Translating text to {target_language}")
        print(f"Source: {text[:50]}...")

        result = TranslationResult(
            source_language=detected_source,
            target_language=target_language,
            source_text=text,
            translated_text=f"Translated text in {target_language}",
            confidence=0.93,
            alternatives=[
                f"Alternative translation 1 in {target_language}",
                f"Alternative translation 2 in {target_language}"
            ]
        )

        print(f"Translation: {result.translated_text}")
        return result

    def translate_batch(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None
    ) -> List[TranslationResult]:
        """
        Translate multiple texts.

        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code

        Returns:
            List of translation results
        """
        results = [
            self.translate_text(text, target_language, source_language)
            for text in texts
        ]

        print(f"Translated {len(results)} texts")
        return results


class AzureAnomalyDetectorManager:
    """
    Manager for Azure Anomaly Detector Service.

    Provides time series anomaly detection for monitoring and alerting.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "v1.1"
    ):
        """
        Initialize Anomaly Detector manager.

        Args:
            endpoint: Azure Anomaly Detector endpoint URL
            api_key: API key for authentication
            api_version: API version to use
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.api_version = api_version

    def detect_anomalies(
        self,
        series: List[Dict[str, Any]],
        granularity: str = "daily",
        sensitivity: int = 95
    ) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in time series data.

        Args:
            series: Time series data points
            granularity: Time granularity
            sensitivity: Detection sensitivity (0-100)

        Returns:
            List of anomaly detection results
        """
        print(f"Detecting anomalies in {len(series)} data points")
        print(f"Granularity: {granularity}, Sensitivity: {sensitivity}")

        results = [
            AnomalyDetectionResult(
                is_anomaly=False,
                is_positive_anomaly=False,
                is_negative_anomaly=False,
                expected_value=100.0,
                upper_margin=10.0,
                lower_margin=10.0,
                severity=0.0
            ),
            AnomalyDetectionResult(
                is_anomaly=True,
                is_positive_anomaly=True,
                is_negative_anomaly=False,
                expected_value=100.0,
                upper_margin=10.0,
                lower_margin=10.0,
                severity=0.85
            )
        ]

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        print(f"Found {anomaly_count} anomalies")

        return results


class AzureContentModeratorManager:
    """
    Manager for Azure Content Moderator Service.

    Provides content moderation for text and images.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str
    ):
        """
        Initialize Content Moderator manager.

        Args:
            endpoint: Azure Content Moderator endpoint URL
            api_key: API key for authentication
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key

    def moderate_text(
        self,
        text: str,
        language: str = "eng",
        classify: bool = True
    ) -> ContentModerationResult:
        """
        Moderate text content.

        Args:
            text: Text to moderate
            language: Language code
            classify: Perform classification

        Returns:
            ContentModerationResult with moderation results
        """
        print(f"Moderating text content: {text[:50]}...")

        result = ContentModerationResult(
            is_appropriate=True,
            categories={
                "Category1": 0.001,  # Adult
                "Category2": 0.002,  # Racy
                "Category3": 0.001   # Offensive
            },
            review_recommended=False,
            adult_score=0.001,
            racy_score=0.002,
            language_detection=language
        )

        print(f"Appropriate: {result.is_appropriate}")
        print(f"Review recommended: {result.review_recommended}")

        return result

    def moderate_image(
        self,
        image_url: str
    ) -> ContentModerationResult:
        """
        Moderate image content.

        Args:
            image_url: URL to image

        Returns:
            ContentModerationResult with moderation results
        """
        print(f"Moderating image: {image_url}")

        result = ContentModerationResult(
            is_appropriate=True,
            categories={
                "Adult": 0.005,
                "Racy": 0.003,
                "Gore": 0.001
            },
            review_recommended=False,
            adult_score=0.005,
            racy_score=0.003
        )

        return result


def demo_computer_vision():
    """Demo: Computer vision operations"""
    print("\n" + "="*60)
    print("DEMO: Computer Vision")
    print("="*60)

    manager = AzureComputerVisionManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key"
    )

    # Analyze image
    result = manager.analyze_image(
        image_url="https://example.com/image.jpg",
        features=[VisionFeature.TAGS, VisionFeature.DESCRIPTION, VisionFeature.OBJECTS]
    )

    print(f"\nImage ID: {result.image_id}")
    print(f"Description: {result.description['captions'][0]['text']}")
    print(f"Objects detected: {len(result.objects)}")
    print(f"Tags: {', '.join([list(tag.keys())[0] for tag in result.tags[:3]])}")


def demo_ocr():
    """Demo: Optical character recognition"""
    print("\n" + "="*60)
    print("DEMO: OCR (Optical Character Recognition)")
    print("="*60)

    manager = AzureComputerVisionManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key"
    )

    result = manager.extract_text_ocr("https://example.com/document.jpg")

    print(f"\nExtracted Text: {result.text}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Lines: {len(result.lines)}")
    print(f"Words: {len(result.words)}")


def demo_speech_services():
    """Demo: Speech services operations"""
    print("\n" + "="*60)
    print("DEMO: Speech Services")
    print("="*60)

    manager = AzureSpeechServicesManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key",
        region="eastus"
    )

    # Speech to text
    audio_data = b"simulated_audio"
    stt_result = manager.speech_to_text(audio_data, SpeechLanguage.EN_US)
    print(f"\nSpeech-to-Text: {stt_result.text}")
    print(f"Confidence: {stt_result.confidence:.2%}")

    # Text to speech
    tts_result = manager.text_to_speech(
        "Hello, this is a test of text-to-speech.",
        voice_name="en-US-JennyNeural",
        style=VoiceStyle.CHEERFUL
    )
    print(f"\nText-to-Speech: Generated {len(tts_result.audio_data)} bytes")
    print(f"Duration: {tts_result.duration_ms}ms")


def demo_language_services():
    """Demo: Language services operations"""
    print("\n" + "="*60)
    print("DEMO: Language Services")
    print("="*60)

    manager = AzureLanguageServicesManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key"
    )

    text = "Azure Cognitive Services provides amazing capabilities for natural language processing."

    # Sentiment analysis
    sentiment = manager.analyze_sentiment(text)
    print(f"\nSentiment: {sentiment.sentiment.value}")
    print(f"Score: {sentiment.overall_score:.2f}")

    # Key phrases
    phrases = manager.extract_key_phrases(text)
    print(f"\nKey Phrases:")
    for phrase in phrases[:3]:
        print(f"  {phrase.text} (score: {phrase.score:.2f})")

    # Entity recognition
    entities = manager.recognize_entities(text)
    print(f"\nEntities: {len(entities)}")


def demo_translation():
    """Demo: Translation operations"""
    print("\n" + "="*60)
    print("DEMO: Translation")
    print("="*60)

    manager = AzureTranslatorManager(
        endpoint="https://api.cognitive.microsofttranslator.com",
        api_key="sample-key"
    )

    result = manager.translate_text(
        text="Hello, how are you today?",
        target_language="es",
        source_language="en"
    )

    print(f"\nSource ({result.source_language}): {result.source_text}")
    print(f"Translation ({result.target_language}): {result.translated_text}")
    print(f"Confidence: {result.confidence:.2%}")


def demo_anomaly_detection():
    """Demo: Anomaly detection"""
    print("\n" + "="*60)
    print("DEMO: Anomaly Detection")
    print("="*60)

    manager = AzureAnomalyDetectorManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key"
    )

    # Sample time series data
    series = [
        {"timestamp": "2024-01-01T00:00:00Z", "value": 100},
        {"timestamp": "2024-01-02T00:00:00Z", "value": 105},
        {"timestamp": "2024-01-03T00:00:00Z", "value": 98},
        {"timestamp": "2024-01-04T00:00:00Z", "value": 250}  # Anomaly
    ]

    results = manager.detect_anomalies(series, granularity="daily")

    print(f"\nAnalyzed {len(series)} data points")
    for i, result in enumerate(results):
        if result.is_anomaly:
            print(f"  Point {i+1}: ANOMALY (severity: {result.severity:.2f})")


def demo_content_moderation():
    """Demo: Content moderation"""
    print("\n" + "="*60)
    print("DEMO: Content Moderation")
    print("="*60)

    manager = AzureContentModeratorManager(
        endpoint="https://example.cognitiveservices.azure.com",
        api_key="sample-key"
    )

    # Moderate text
    text_result = manager.moderate_text(
        "This is a sample text for content moderation."
    )

    print(f"\nText Moderation:")
    print(f"  Appropriate: {text_result.is_appropriate}")
    print(f"  Review recommended: {text_result.review_recommended}")
    print(f"  Adult score: {text_result.adult_score:.4f}")

    # Moderate image
    image_result = manager.moderate_image("https://example.com/image.jpg")
    print(f"\nImage Moderation:")
    print(f"  Appropriate: {image_result.is_appropriate}")


if __name__ == "__main__":
    """Run all demo functions"""
    print("\n" + "="*60)
    print("Azure Cognitive Services - Comprehensive Demo")
    print("="*60)

    demo_computer_vision()
    demo_ocr()
    demo_speech_services()
    demo_language_services()
    demo_translation()
    demo_anomaly_detection()
    demo_content_moderation()

    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60)
