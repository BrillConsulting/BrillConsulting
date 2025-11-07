"""
Agent Reflection Framework
===========================

Self-evaluation and introspection system for autonomous agents:
- Performance analysis and evaluation
- Strategy assessment and improvement
- Error analysis and learning
- Metacognitive monitoring
- Self-improvement recommendations

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import statistics


class ReflectionDepth(Enum):
    """Levels of reflection depth."""
    SHALLOW = "shallow"
    MEDIUM = "medium"
    DEEP = "deep"
    META = "meta"


class OutcomeType(Enum):
    """Types of action outcomes."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class Action:
    """Represents an executed action."""
    id: str
    name: str
    context: Dict[str, Any]
    parameters: Dict[str, Any]
    timestamp: str
    outcome: OutcomeType
    result: Optional[Any] = None
    duration: float = 0.0
    confidence: float = 0.0


@dataclass
class Reflection:
    """Represents a reflection on an action or experience."""
    action_id: str
    depth: ReflectionDepth
    timestamp: str
    performance_score: float
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    confidence_assessment: float = 0.0


class ReflectiveAgent:
    """Agent with self-reflection and metacognitive capabilities."""

    def __init__(
        self,
        name: str = "ReflectiveAgent",
        reflection_depth: str = "medium"
    ):
        """Initialize reflective agent."""
        self.name = name
        self.default_depth = ReflectionDepth(reflection_depth)
        self.actions_history: List[Action] = []
        self.reflections_history: List[Reflection] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "success_rate": [],
            "confidence": [],
            "duration": []
        }
        self.learning_insights: List[str] = []

        print(f"✓ Reflective Agent '{name}' initialized")
        print(f"  Default reflection depth: {reflection_depth}")

    def execute_with_reflection(
        self,
        action: str,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action and automatically reflect on it."""
        print(f"\n▶ Executing action: {action}")

        # Create action record
        action_id = f"action_{len(self.actions_history) + 1}"
        start_time = datetime.now()

        # Simulate action execution
        result = self._simulate_action(action, context, parameters)

        # Record action
        action_obj = Action(
            id=action_id,
            name=action,
            context=context,
            parameters=parameters,
            timestamp=start_time.isoformat(),
            outcome=result["outcome"],
            result=result["data"],
            duration=(datetime.now() - start_time).total_seconds(),
            confidence=result["confidence"]
        )
        self.actions_history.append(action_obj)

        # Automatic reflection
        print(f"🤔 Reflecting on action outcome...")
        reflection = self.reflect_on_action(
            action_id=action_id,
            outcome=result["outcome"]
        )

        return {
            "action_id": action_id,
            "result": result,
            "reflection": reflection
        }

    def _simulate_action(
        self,
        action: str,
        context: Dict,
        parameters: Dict
    ) -> Dict[str, Any]:
        """Simulate action execution (placeholder)."""
        # In real implementation, this would execute actual actions
        import random

        success_prob = 0.75
        outcome = (
            OutcomeType.SUCCESS if random.random() < success_prob
            else OutcomeType.PARTIAL_SUCCESS
        )

        return {
            "outcome": outcome,
            "data": {"status": outcome.value, "processed": len(context)},
            "confidence": random.uniform(0.6, 0.95)
        }

    def reflect_on_action(
        self,
        action_id: str,
        outcome: OutcomeType,
        depth: Optional[ReflectionDepth] = None
    ) -> Reflection:
        """Perform reflection on a specific action."""
        depth = depth or self.default_depth

        # Find action
        action = next(
            (a for a in self.actions_history if a.id == action_id),
            None
        )
        if not action:
            raise ValueError(f"Action {action_id} not found")

        print(f"\n🔍 Reflection depth: {depth.value}")

        # Perform reflection based on depth
        reflection = Reflection(
            action_id=action_id,
            depth=depth,
            timestamp=datetime.now().isoformat(),
            performance_score=self._calculate_performance_score(action)
        )

        # Shallow reflection
        if depth.value in ["shallow", "medium", "deep", "meta"]:
            reflection.strengths = self._identify_strengths(action)
            reflection.weaknesses = self._identify_weaknesses(action)

        # Medium reflection
        if depth.value in ["medium", "deep", "meta"]:
            reflection.alternatives = self._generate_alternatives(action)
            reflection.lessons_learned = self._extract_lessons(action)

        # Deep reflection
        if depth.value in ["deep", "meta"]:
            reflection.improvement_suggestions = (
                self._generate_improvements(action)
            )
            reflection.confidence_assessment = (
                self._assess_confidence(action)
            )

        # Meta reflection
        if depth.value == "meta":
            self._meta_reflect(reflection)

        # Store reflection
        self.reflections_history.append(reflection)
        self._update_metrics(action, reflection)

        print(f"✓ Reflection complete")
        print(f"  Performance score: {reflection.performance_score:.2f}")
        print(f"  Lessons learned: {len(reflection.lessons_learned)}")

        return reflection

    def _calculate_performance_score(self, action: Action) -> float:
        """Calculate performance score for an action."""
        outcome_scores = {
            OutcomeType.SUCCESS: 1.0,
            OutcomeType.PARTIAL_SUCCESS: 0.6,
            OutcomeType.FAILURE: 0.2,
            OutcomeType.ERROR: 0.0
        }

        base_score = outcome_scores[action.outcome]
        confidence_factor = action.confidence * 0.3
        duration_factor = min(1.0 / (action.duration + 1), 0.2)

        return min(base_score + confidence_factor + duration_factor, 1.0)

    def _identify_strengths(self, action: Action) -> List[str]:
        """Identify strengths in action execution."""
        strengths = []

        if action.outcome in [OutcomeType.SUCCESS, OutcomeType.PARTIAL_SUCCESS]:
            strengths.append("Goal achievement")

        if action.confidence > 0.8:
            strengths.append("High confidence in decision")

        if action.duration < 2.0:
            strengths.append("Efficient execution")

        if action.parameters:
            strengths.append("Well-parameterized approach")

        return strengths

    def _identify_weaknesses(self, action: Action) -> List[str]:
        """Identify weaknesses in action execution."""
        weaknesses = []

        if action.outcome in [OutcomeType.FAILURE, OutcomeType.ERROR]:
            weaknesses.append("Failed to achieve goal")

        if action.confidence < 0.5:
            weaknesses.append("Low confidence in decision")

        if action.duration > 5.0:
            weaknesses.append("Slow execution time")

        if not action.parameters:
            weaknesses.append("Insufficient parameterization")

        return weaknesses

    def _generate_alternatives(self, action: Action) -> List[str]:
        """Generate alternative approaches."""
        alternatives = [
            f"Try different parameters for {action.name}",
            f"Break down {action.name} into smaller steps",
            f"Use ensemble approach for {action.name}",
            "Gather more context before execution",
            "Implement fallback strategy"
        ]
        return alternatives[:3]

    def _extract_lessons(self, action: Action) -> List[str]:
        """Extract lessons learned from action."""
        lessons = []

        if action.outcome == OutcomeType.SUCCESS:
            lessons.append(
                f"Successful strategy for {action.name} - replicate approach"
            )

        if action.confidence > 0.8 and action.outcome == OutcomeType.SUCCESS:
            lessons.append("High confidence correlates with success")

        if action.outcome == OutcomeType.FAILURE:
            lessons.append(
                f"Approach failed for {action.name} - avoid in similar contexts"
            )

        return lessons

    def _generate_improvements(self, action: Action) -> List[str]:
        """Generate improvement suggestions."""
        improvements = []

        if action.duration > 3.0:
            improvements.append("Optimize execution for better performance")

        if action.confidence < 0.7:
            improvements.append("Improve confidence through better analysis")

        if action.outcome != OutcomeType.SUCCESS:
            improvements.append("Review strategy and adjust approach")

        improvements.append("Implement monitoring and early warning system")

        return improvements

    def _assess_confidence(self, action: Action) -> float:
        """Assess confidence in the action and its outcome."""
        if not self.reflections_history:
            return action.confidence

        # Compare with historical performance
        recent_scores = [
            r.performance_score
            for r in self.reflections_history[-10:]
        ]

        if recent_scores:
            avg_performance = statistics.mean(recent_scores)
            return (action.confidence + avg_performance) / 2

        return action.confidence

    def _meta_reflect(self, reflection: Reflection) -> None:
        """Perform meta-reflection on the reflection process itself."""
        print("\n🔄 Meta-reflection: Analyzing reflection quality")

        # Analyze reflection completeness
        completeness = sum([
            bool(reflection.strengths),
            bool(reflection.weaknesses),
            bool(reflection.alternatives),
            bool(reflection.lessons_learned),
            bool(reflection.improvement_suggestions)
        ]) / 5

        if completeness < 0.6:
            self.learning_insights.append(
                "Reflection process needs more depth"
            )

    def _update_metrics(self, action: Action, reflection: Reflection) -> None:
        """Update performance metrics."""
        self.performance_metrics["success_rate"].append(
            1.0 if action.outcome == OutcomeType.SUCCESS else 0.0
        )
        self.performance_metrics["confidence"].append(action.confidence)
        self.performance_metrics["duration"].append(action.duration)

    def get_improvement_suggestions(self) -> List[str]:
        """Get aggregated improvement suggestions."""
        all_suggestions = []
        for reflection in self.reflections_history[-10:]:
            all_suggestions.extend(reflection.improvement_suggestions)

        # Deduplicate and prioritize
        unique_suggestions = list(set(all_suggestions))
        return unique_suggestions[:5]

    def analyze_decision_patterns(
        self,
        time_period: str = "all"
    ) -> Dict[str, Any]:
        """Analyze decision-making patterns."""
        print(f"\n📊 Analyzing decision patterns ({time_period})")

        if not self.actions_history:
            return {"message": "No actions to analyze"}

        # Calculate statistics
        success_rate = statistics.mean(
            self.performance_metrics["success_rate"]
        ) if self.performance_metrics["success_rate"] else 0

        avg_confidence = statistics.mean(
            self.performance_metrics["confidence"]
        ) if self.performance_metrics["confidence"] else 0

        avg_duration = statistics.mean(
            self.performance_metrics["duration"]
        ) if self.performance_metrics["duration"] else 0

        patterns = {
            "total_actions": len(self.actions_history),
            "success_rate": round(success_rate, 3),
            "average_confidence": round(avg_confidence, 3),
            "average_duration": round(avg_duration, 3),
            "total_reflections": len(self.reflections_history),
            "improvement_trend": self._calculate_trend()
        }

        print(f"✓ Pattern analysis complete")
        print(f"  Success rate: {patterns['success_rate']:.1%}")
        print(f"  Avg confidence: {patterns['average_confidence']:.2f}")

        return patterns

    def _calculate_trend(self) -> str:
        """Calculate performance trend."""
        if len(self.performance_metrics["success_rate"]) < 5:
            return "insufficient_data"

        recent = self.performance_metrics["success_rate"][-5:]
        older = self.performance_metrics["success_rate"][-10:-5]

        if not older:
            return "insufficient_data"

        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)

        if recent_avg > older_avg + 0.1:
            return "improving"
        elif recent_avg < older_avg - 0.1:
            return "declining"
        return "stable"

    def generate_self_assessment(self) -> Dict[str, Any]:
        """Generate comprehensive self-assessment report."""
        print("\n📋 Generating self-assessment report")

        patterns = self.analyze_decision_patterns()
        suggestions = self.get_improvement_suggestions()

        assessment = {
            "agent_name": self.name,
            "assessment_date": datetime.now().isoformat(),
            "performance_patterns": patterns,
            "top_improvements": suggestions,
            "learning_insights": self.learning_insights[-5:],
            "reflection_depth": self.default_depth.value,
            "total_experience": len(self.actions_history)
        }

        print(f"✓ Self-assessment generated")
        return assessment


def demo():
    """Demonstrate agent reflection capabilities."""
    print("=" * 60)
    print("Agent Reflection Framework Demo")
    print("=" * 60)

    # Create reflective agent
    agent = ReflectiveAgent(name="ReflectBot", reflection_depth="deep")

    # Execute actions with automatic reflection
    actions = [
        ("analyze_data", {"dataset": "sales.csv"}, {"method": "statistical"}),
        ("process_text", {"text": "sample"}, {"nlp_model": "bert"}),
        ("classify_image", {"image": "cat.jpg"}, {"model": "resnet50"}),
    ]

    for action, context, params in actions:
        result = agent.execute_with_reflection(action, context, params)
        print(f"\n  Action: {action}")
        print(f"  Outcome: {result['result']['outcome'].value}")
        print(f"  Performance: {result['reflection'].performance_score:.2f}")

    # Analyze patterns
    print("\n" + "=" * 60)
    patterns = agent.analyze_decision_patterns()
    print(f"\nDecision Patterns:")
    print(json.dumps(patterns, indent=2))

    # Get improvement suggestions
    print("\n" + "=" * 60)
    suggestions = agent.get_improvement_suggestions()
    print(f"\nTop Improvement Suggestions:")
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"{i}. {suggestion}")

    # Generate self-assessment
    print("\n" + "=" * 60)
    assessment = agent.generate_self_assessment()
    print(f"\nSelf-Assessment:")
    print(f"  Total experience: {assessment['total_experience']} actions")
    print(f"  Success rate: {assessment['performance_patterns']['success_rate']:.1%}")
    print(f"  Trend: {assessment['performance_patterns']['improvement_trend']}")


if __name__ == "__main__":
    demo()
