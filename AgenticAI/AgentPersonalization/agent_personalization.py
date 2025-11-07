"""
Agent Personalization Framework
================================

Adaptive agents that learn user preferences:
- User profiling and modeling
- Behavioral learning
- Preference elicitation
- Context-aware personalization
- Recommendation systems

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import random


@dataclass
class UserProfile:
    """User preference profile."""
    user_id: str
    preferences: Dict[str, float] = field(default_factory=dict)
    interaction_history: List[Dict] = field(default_factory=list)
    context_patterns: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class PersonalizedAgent:
    """Agent that personalizes to user preferences."""

    def __init__(self, name: str = "PersonalAgent", user_id: str = "user_001"):
        """Initialize personalized agent."""
        self.name = name
        self.user_id = user_id
        self.profile = UserProfile(user_id=user_id)
        self.learning_rate = 0.1
        self.communication_style = "balanced"

        print(f"👤 Personalized Agent '{name}' initialized")
        print(f"   User ID: {user_id}")

    def observe_interaction(
        self,
        user_action: str,
        context: Dict[str, Any]
    ) -> None:
        """Learn from user interaction."""
        print(f"\n📝 Observing: {user_action}")

        # Record interaction
        interaction = {
            "action": user_action,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        self.profile.interaction_history.append(interaction)

        # Update preferences
        self._update_preferences(user_action, context)

        # Learn context patterns
        for key, value in context.items():
            self.profile.context_patterns[key].append(str(value))

        print(f"   ✓ Learned from interaction")

    def _update_preferences(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> None:
        """Update user preference model."""
        # Extract preference signal from action
        if "selected" in action:
            option = action.split("_")[-1]
            current = self.profile.preferences.get(option, 0.5)
            self.profile.preferences[option] = min(
                1.0,
                current + self.learning_rate
            )

    def recommend(
        self,
        options: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate personalized recommendation."""
        print(f"\n💡 Generating recommendation from {len(options)} options")

        scores = {}
        for option in options:
            # Base score from preferences
            base_score = self.profile.preferences.get(option, 0.5)

            # Context adjustment
            context_bonus = 0.0
            if context:
                # Check if similar context led to this choice before
                for interaction in self.profile.interaction_history[-10:]:
                    if option in interaction["action"]:
                        similarity = self._context_similarity(
                            context,
                            interaction["context"]
                        )
                        context_bonus += similarity * 0.1

            scores[option] = base_score + context_bonus

        # Select best option
        best_option = max(scores, key=scores.get)
        confidence = scores[best_option]

        print(f"   Recommended: {best_option} (confidence: {confidence:.2f})")

        return {
            "recommendation": best_option,
            "confidence": confidence,
            "all_scores": scores
        }

    def _context_similarity(
        self,
        context_a: Dict[str, Any],
        context_b: Dict[str, Any]
    ) -> float:
        """Calculate context similarity."""
        common_keys = set(context_a.keys()) & set(context_b.keys())
        if not common_keys:
            return 0.0

        matches = sum(
            1 for key in common_keys
            if context_a[key] == context_b[key]
        )

        return matches / len(common_keys)

    def adapt_style(self, user_feedback: str) -> None:
        """Adapt communication style."""
        print(f"\n🎨 Adapting style based on: {user_feedback}")

        style_mapping = {
            "more_concise": "concise",
            "more_detailed": "detailed",
            "more_friendly": "friendly",
            "more_formal": "formal"
        }

        if user_feedback in style_mapping:
            self.communication_style = style_mapping[user_feedback]
            print(f"   ✓ Style updated to: {self.communication_style}")

    def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile summary."""
        print(f"\n📊 Generating user profile...")

        # Top preferences
        top_prefs = sorted(
            self.profile.preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        profile_summary = {
            "user_id": self.user_id,
            "total_interactions": len(self.profile.interaction_history),
            "top_preferences": dict(top_prefs),
            "communication_style": self.communication_style,
            "learning_progress": min(
                1.0,
                len(self.profile.interaction_history) / 20
            )
        }

        print(f"   ✓ Profile generated")
        return profile_summary


def demo():
    """Demonstrate agent personalization."""
    print("=" * 60)
    print("Agent Personalization Framework Demo")
    print("=" * 60)

    agent = PersonalizedAgent(name="PersonalBot", user_id="user_123")

    # Simulate user interactions
    interactions = [
        ("selected_option_A", {"time": "morning", "mood": "focused"}),
        ("selected_option_A", {"time": "morning", "mood": "energetic"}),
        ("selected_option_B", {"time": "afternoon", "mood": "relaxed"}),
        ("selected_option_C", {"time": "evening", "mood": "tired"}),
        ("selected_option_A", {"time": "morning", "mood": "focused"}),
    ]

    for action, context in interactions:
        agent.observe_interaction(action, context)

    # Get personalized recommendation
    print("\n" + "=" * 60)
    recommendation = agent.recommend(
        options=["A", "B", "C"],
        context={"time": "morning", "mood": "focused"}
    )

    # Adapt style
    print("\n" + "=" * 60)
    agent.adapt_style("more_concise")

    # Get profile
    print("\n" + "=" * 60)
    import json
    profile = agent.get_user_profile()
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    demo()
