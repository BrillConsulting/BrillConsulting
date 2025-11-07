"""
Content Moderation Pipeline
============================

Multi-layered moderation with OpenAI + local filters

Author: Brill Consulting
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class ModerationCategory(Enum):
    """Content moderation categories."""
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    HARASSMENT = "harassment"
    SPAM = "spam"
    SAFE = "safe"


class ModerationAction(Enum):
    """Recommended actions."""
    ALLOW = "allow"
    FLAG = "flag"
    WARN = "warn"
    BLOCK = "block"
    ALERT = "alert"


@dataclass
class ModerationResult:
    """Result of content moderation."""
    is_flagged: bool
    categories: List[str]
    severity_score: float
    confidence: float
    recommended_action: ModerationAction
    details: Dict[str, float]
    timestamp: str


class ModerationPipeline:
    """Multi-layered content moderation system."""

    def __init__(
        self,
        use_openai: bool = True,
        use_local_models: bool = True,
        severity_threshold: float = 0.7
    ):
        """Initialize moderation pipeline."""
        self.use_openai = use_openai
        self.use_local_models = use_local_models
        self.severity_threshold = severity_threshold
        self.moderation_count = 0

        print(f"🛡️  Content Moderation Pipeline initialized")
        print(f"   OpenAI: {'enabled' if use_openai else 'disabled'}")
        print(f"   Local models: {'enabled' if use_local_models else 'disabled'}")

    def moderate(self, text: str) -> ModerationResult:
        """Moderate content through pipeline."""
        self.moderation_count += 1

        print(f"\n🔍 Moderating content (#{self.moderation_count})")

        scores = {}

        # OpenAI Moderation
        if self.use_openai:
            openai_scores = self._openai_moderation(text)
            scores.update(openai_scores)

        # Local models
        if self.use_local_models:
            local_scores = self._local_moderation(text)
            scores.update(local_scores)

        # Aggregate scores
        max_score = max(scores.values()) if scores else 0.0
        flagged_categories = [
            cat for cat, score in scores.items()
            if score >= self.severity_threshold
        ]

        is_flagged = len(flagged_categories) > 0

        # Determine action
        action = self._determine_action(max_score, flagged_categories)

        result = ModerationResult(
            is_flagged=is_flagged,
            categories=flagged_categories,
            severity_score=max_score,
            confidence=0.9 if self.use_openai and self.use_local_models else 0.75,
            recommended_action=action,
            details=scores,
            timestamp=datetime.now().isoformat()
        )

        # Log result
        if result.is_flagged:
            print(f"   🚨 Content FLAGGED")
            print(f"   Categories: {', '.join(flagged_categories)}")
            print(f"   Action: {action.value}")
        else:
            print(f"   ✅ Content is safe")

        return result

    def _openai_moderation(self, text: str) -> Dict[str, float]:
        """Simulate OpenAI Moderation API."""
        # In production: call actual OpenAI API
        import random

        return {
            "hate_speech": random.uniform(0.0, 0.3),
            "violence": random.uniform(0.0, 0.2),
            "sexual": random.uniform(0.0, 0.1),
            "self_harm": random.uniform(0.0, 0.1)
        }

    def _local_moderation(self, text: str) -> Dict[str, float]:
        """Local ML-based moderation."""
        # Simulate local model predictions
        text_lower = text.lower()

        scores = {}

        # Hate speech detection
        hate_keywords = ["hate", "discriminat", "racist"]
        if any(kw in text_lower for kw in hate_keywords):
            scores["hate_speech"] = 0.85

        # Violence detection
        violence_keywords = ["kill", "violence", "attack"]
        if any(kw in text_lower for kw in violence_keywords):
            scores["violence"] = 0.75

        # Harassment detection
        harassment_keywords = ["harass", "bully", "threaten"]
        if any(kw in text_lower for kw in harassment_keywords):
            scores["harassment"] = 0.7

        return scores

    def _determine_action(
        self,
        severity: float,
        categories: List[str]
    ) -> ModerationAction:
        """Determine moderation action."""
        if "self_harm" in categories:
            return ModerationAction.ALERT

        if severity >= 0.9:
            return ModerationAction.BLOCK
        elif severity >= 0.7:
            return ModerationAction.WARN
        elif severity >= 0.5:
            return ModerationAction.FLAG
        else:
            return ModerationAction.ALLOW


def demo():
    """Demonstrate content moderation."""
    print("=" * 60)
    print("Content Moderation Pipeline Demo")
    print("=" * 60)

    pipeline = ModerationPipeline(
        use_openai=True,
        use_local_models=True,
        severity_threshold=0.7
    )

    test_content = [
        "This is a normal, safe message.",
        "I hate this product, it's terrible!",  # Borderline
        "This content contains violent threats.",  # Should flag
    ]

    for content in test_content:
        print(f"\n{'='*60}")
        print(f"Content: {content[:50]}...")
        result = pipeline.moderate(content)


if __name__ == "__main__":
    demo()
