"""
Emotional Agents Framework
===========================

Agents with emotional intelligence and empathy:
- Emotional states and transitions
- Empathy and sentiment analysis
- Affect-driven decision making
- Emotional memory and context
- Self-regulation

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import random


class EmotionType(Enum):
    """Primary emotion types (Plutchik's model)."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    TRUST = "trust"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


class PersonalityType(Enum):
    """Agent personality types."""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    ENERGETIC = "energetic"
    ANALYTICAL = "analytical"


@dataclass
class Emotion:
    """Represents an emotional state."""
    type: EmotionType
    intensity: float  # 0.0 to 1.0
    timestamp: str
    trigger: Optional[str] = None


@dataclass
class EmotionalMemory:
    """Stores emotional context of interactions."""
    interaction_id: str
    user_emotion: str
    agent_emotion: EmotionType
    response: str
    empathy_score: float
    timestamp: str


class EmotionalAgent:
    """Agent with emotional intelligence and empathy."""

    def __init__(
        self,
        name: str = "EmotionalAgent",
        base_personality: str = "friendly",
        emotional_sensitivity: float = 0.7
    ):
        """Initialize emotional agent."""
        self.name = name
        self.personality = PersonalityType(base_personality)
        self.sensitivity = emotional_sensitivity
        self.current_emotion = Emotion(
            type=EmotionType.NEUTRAL,
            intensity=0.5,
            timestamp=datetime.now().isoformat()
        )
        self.emotion_history: List[Emotion] = []
        self.emotional_memories: List[EmotionalMemory] = []
        self.mood_baseline = 0.6  # Overall mood level

        # Emotion transition probabilities
        self.emotion_transitions = self._initialize_transitions()

        print(f"😊 Emotional Agent '{name}' initialized")
        print(f"   Personality: {base_personality}")
        print(f"   Emotional sensitivity: {emotional_sensitivity:.2f}")

    def _initialize_transitions(self) -> Dict[EmotionType, Dict[EmotionType, float]]:
        """Initialize emotion transition probabilities."""
        # Simplified transition model
        return {
            EmotionType.JOY: {
                EmotionType.JOY: 0.6,
                EmotionType.TRUST: 0.2,
                EmotionType.NEUTRAL: 0.2
            },
            EmotionType.SADNESS: {
                EmotionType.SADNESS: 0.5,
                EmotionType.NEUTRAL: 0.3,
                EmotionType.ANGER: 0.2
            },
            EmotionType.ANGER: {
                EmotionType.ANGER: 0.4,
                EmotionType.DISGUST: 0.3,
                EmotionType.NEUTRAL: 0.3
            }
        }

    def interact(
        self,
        message: str,
        user_emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        """Interact with user using emotional intelligence."""
        print(f"\n💬 User: {message}")
        if user_emotion:
            print(f"   Detected emotion: {user_emotion}")

        # Analyze user emotion if not provided
        if not user_emotion:
            user_emotion = self._detect_emotion(message)

        # Generate empathetic response
        response = self._generate_empathetic_response(message, user_emotion)

        # Calculate empathy score
        empathy_score = self.calculate_empathy_score(user_emotion, response)

        # Update agent's emotional state (emotional contagion)
        self._emotional_contagion(user_emotion)

        # Store in emotional memory
        memory = EmotionalMemory(
            interaction_id=f"int_{len(self.emotional_memories) + 1}",
            user_emotion=user_emotion,
            agent_emotion=self.current_emotion.type,
            response=response,
            empathy_score=empathy_score,
            timestamp=datetime.now().isoformat()
        )
        self.emotional_memories.append(memory)

        print(f"🤖 Agent ({self.current_emotion.type.value}): {response}")
        print(f"   Empathy score: {empathy_score:.2f}")

        return {
            "response": response,
            "agent_emotion": self.current_emotion.type.value,
            "user_emotion": user_emotion,
            "empathy_score": empathy_score
        }

    def _detect_emotion(self, message: str) -> str:
        """Detect emotion from message (simplified)."""
        message_lower = message.lower()

        emotion_keywords = {
            "happy": ["happy", "great", "excellent", "wonderful", "joy"],
            "sad": ["sad", "unhappy", "depressed", "down", "terrible"],
            "angry": ["angry", "furious", "mad", "frustrated", "annoyed"],
            "anxious": ["worried", "anxious", "stressed", "nervous", "scared"],
            "excited": ["excited", "thrilled", "amazing", "awesome"],
        }

        for emotion, keywords in emotion_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return emotion

        return "neutral"

    def _generate_empathetic_response(
        self,
        message: str,
        user_emotion: str
    ) -> str:
        """Generate response with empathy."""
        # Response templates based on personality and emotion
        templates = {
            PersonalityType.FRIENDLY: {
                "sad": "I'm sorry to hear you're feeling this way. I'm here to help! 💙",
                "happy": "That's wonderful! I'm so happy for you! 😊",
                "angry": "I understand you're upset. Let's work through this together.",
                "anxious": "I hear you. Let's take this step by step. You've got this!",
                "neutral": "Thanks for sharing. How can I help you today?"
            },
            PersonalityType.PROFESSIONAL: {
                "sad": "I understand this is challenging. Let me assist you.",
                "happy": "That's good to hear. How can I support your success?",
                "angry": "I acknowledge your frustration. Let's find a solution.",
                "anxious": "Let's address your concerns systematically.",
                "neutral": "I'm ready to assist. What do you need?"
            }
        }

        personality_templates = templates.get(
            self.personality,
            templates[PersonalityType.FRIENDLY]
        )

        return personality_templates.get(
            user_emotion,
            "I'm here to help. What can I do for you?"
        )

    def _emotional_contagion(self, user_emotion: str) -> None:
        """Update agent emotion based on user emotion."""
        # Map user emotions to agent emotions
        emotion_mapping = {
            "happy": EmotionType.JOY,
            "sad": EmotionType.SADNESS,
            "angry": EmotionType.TRUST,  # Agent stays calm/trusting
            "anxious": EmotionType.TRUST,
            "excited": EmotionType.JOY,
            "neutral": EmotionType.NEUTRAL
        }

        target_emotion = emotion_mapping.get(user_emotion, EmotionType.NEUTRAL)

        # Update with sensitivity factor
        if target_emotion != self.current_emotion.type:
            new_intensity = self.sensitivity * 0.6

            self.current_emotion = Emotion(
                type=target_emotion,
                intensity=new_intensity,
                timestamp=datetime.now().isoformat(),
                trigger=f"contagion_from_user_{user_emotion}"
            )
            self.emotion_history.append(self.current_emotion)

    def experience_event(
        self,
        event_type: str,
        intensity: float = 0.5
    ) -> None:
        """Update emotion based on experienced event."""
        print(f"\n✨ Experiencing event: {event_type}")

        event_emotions = {
            "positive_feedback": EmotionType.JOY,
            "negative_feedback": EmotionType.SADNESS,
            "challenge": EmotionType.ANTICIPATION,
            "threat": EmotionType.FEAR,
            "success": EmotionType.JOY,
            "failure": EmotionType.SADNESS
        }

        new_emotion_type = event_emotions.get(event_type, EmotionType.NEUTRAL)

        self.current_emotion = Emotion(
            type=new_emotion_type,
            intensity=min(1.0, intensity),
            timestamp=datetime.now().isoformat(),
            trigger=event_type
        )
        self.emotion_history.append(self.current_emotion)

        print(f"   Emotion updated: {new_emotion_type.value} (intensity: {intensity:.2f})")

    def make_decision(
        self,
        options: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make decision influenced by emotional state."""
        print(f"\n🤔 Making decision (emotion: {self.current_emotion.type.value})")

        # Emotional influence on decision
        emotion_biases = {
            EmotionType.JOY: "optimistic",
            EmotionType.FEAR: "cautious",
            EmotionType.ANGER: "aggressive",
            EmotionType.SADNESS: "conservative",
            EmotionType.TRUST: "collaborative",
            EmotionType.NEUTRAL: "balanced"
        }

        bias = emotion_biases.get(self.current_emotion.type, "balanced")

        # Select option based on emotional bias
        if bias == "cautious" and "cautious" in options:
            choice = "cautious"
        elif bias == "aggressive" and "aggressive" in options:
            choice = "aggressive"
        elif "neutral" in options:
            choice = "neutral"
        else:
            choice = random.choice(options)

        confidence = 1.0 - (self.current_emotion.intensity * 0.3)

        print(f"   Decision: {choice} (confidence: {confidence:.2f})")

        return {
            "choice": choice,
            "emotional_bias": bias,
            "confidence": confidence,
            "emotion": self.current_emotion.type.value
        }

    def calculate_empathy_score(
        self,
        user_emotion: str,
        agent_response: str
    ) -> float:
        """Calculate empathy score for interaction."""
        # Simplified empathy scoring
        empathy_keywords = {
            "sad": ["sorry", "understand", "here", "help", "support"],
            "angry": ["understand", "acknowledge", "frustration", "solution"],
            "anxious": ["step by step", "together", "concerns", "address"],
            "happy": ["wonderful", "happy", "great", "glad"]
        }

        keywords = empathy_keywords.get(user_emotion, [])
        response_lower = agent_response.lower()

        matches = sum(1 for keyword in keywords if keyword in response_lower)
        base_score = min(1.0, matches / max(len(keywords), 1))

        # Adjust by personality
        personality_bonus = 0.2 if self.personality == PersonalityType.FRIENDLY else 0.1

        return min(1.0, base_score + personality_bonus)

    def get_current_emotion(self) -> Emotion:
        """Get current emotional state."""
        return self.current_emotion

    def regulate_emotion(self, target_emotion: EmotionType) -> None:
        """Self-regulate emotion towards target state."""
        print(f"\n🧘 Regulating emotion: {self.current_emotion.type.value} → {target_emotion.value}")

        # Gradually shift emotion
        self.current_emotion = Emotion(
            type=target_emotion,
            intensity=0.5,
            timestamp=datetime.now().isoformat(),
            trigger="self_regulation"
        )
        self.emotion_history.append(self.current_emotion)

        print(f"   ✓ Emotion regulated to {target_emotion.value}")

    def get_emotional_report(self) -> Dict[str, Any]:
        """Generate emotional state report."""
        print("\n📊 Generating emotional report...")

        if not self.emotion_history:
            return {"message": "No emotional history"}

        # Calculate emotion distribution
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion.type.value] = emotion_counts.get(
                emotion.type.value, 0
            ) + 1

        # Average empathy score
        avg_empathy = (
            sum(m.empathy_score for m in self.emotional_memories) /
            len(self.emotional_memories)
            if self.emotional_memories else 0
        )

        report = {
            "current_emotion": self.current_emotion.type.value,
            "current_intensity": self.current_emotion.intensity,
            "emotion_distribution": emotion_counts,
            "total_interactions": len(self.emotional_memories),
            "average_empathy_score": round(avg_empathy, 2),
            "mood_baseline": self.mood_baseline
        }

        print(f"✓ Report generated")
        return report


def demo():
    """Demonstrate emotional agents framework."""
    print("=" * 60)
    print("Emotional Agents Framework Demo")
    print("=" * 60)

    # Create emotional agent
    agent = EmotionalAgent(
        name="EmpathyBot",
        base_personality="friendly",
        emotional_sensitivity=0.8
    )

    # Simulate interactions with different emotions
    interactions = [
        ("I'm really excited about this new project!", "happy"),
        ("I'm feeling overwhelmed with all this work.", "anxious"),
        ("This is frustrating! Nothing is working.", "angry"),
        ("I just got some great news!", "happy"),
    ]

    for message, emotion in interactions:
        agent.interact(message, emotion)

    # Experience events
    print("\n" + "=" * 60)
    print("Experiencing Events")
    print("=" * 60)

    agent.experience_event("positive_feedback", intensity=0.8)
    agent.experience_event("challenge", intensity=0.6)

    # Make emotional decision
    print("\n" + "=" * 60)
    print("Emotional Decision Making")
    print("=" * 60)

    decision = agent.make_decision(
        options=["aggressive", "cautious", "neutral"],
        context={"risk": "high"}
    )

    # Regulate emotion
    print("\n" + "=" * 60)
    print("Emotion Regulation")
    print("=" * 60)

    agent.regulate_emotion(EmotionType.TRUST)

    # Get report
    print("\n" + "=" * 60)
    print("Emotional Report")
    print("=" * 60)

    import json
    report = agent.get_emotional_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    demo()
