"""
Collaborative Agents Framework
===============================

Human-AI and AI-AI collaboration system:
- Team formation and coordination
- Shared context and goals
- Complementary capabilities
- Contribution tracking
- Conflict resolution

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random


class AgentType(Enum):
    """Types of team members."""
    HUMAN = "human"
    AI = "ai"


class CollaborationStrategy(Enum):
    """Collaboration execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ITERATIVE = "iterative"
    HYBRID = "hybrid"


@dataclass
class Contribution:
    """Represents a team member's contribution."""
    member_name: str
    task_id: str
    content: Any
    quality_score: float
    timestamp: str


@dataclass
class SharedContext:
    """Shared workspace and information."""
    task_description: str
    goal: str
    resources: Dict[str, Any] = field(default_factory=dict)
    contributions: List[Contribution] = field(default_factory=list)
    decisions: List[Dict] = field(default_factory=list)


class TeamMember:
    """Base class for team members."""

    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        capabilities: List[str]
    ):
        """Initialize team member."""
        self.name = name
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.contributions: List[Contribution] = []
        self.performance_score = 0.8

    def can_contribute(self, task_requirements: List[str]) -> bool:
        """Check if member can contribute to task."""
        required = set(task_requirements)
        available = set(self.capabilities)
        return bool(required.intersection(available))

    def contribute(
        self,
        task_id: str,
        task_description: str,
        context: SharedContext
    ) -> Contribution:
        """Make contribution to collaborative task."""
        print(f"   {self.name} contributing...")

        # Simulate contribution
        content = self._generate_contribution(task_description, context)
        quality = self._assess_quality(content)

        contribution = Contribution(
            member_name=self.name,
            task_id=task_id,
            content=content,
            quality_score=quality,
            timestamp=datetime.now().isoformat()
        )

        self.contributions.append(contribution)
        return contribution

    def _generate_contribution(
        self,
        task: str,
        context: SharedContext
    ) -> Dict[str, Any]:
        """Generate contribution content."""
        return {
            "type": self.agent_type.value,
            "capabilities_used": self.capabilities[:2],
            "output": f"Contribution to: {task}"
        }

    def _assess_quality(self, content: Dict) -> float:
        """Assess contribution quality."""
        return min(1.0, self.performance_score + random.uniform(-0.1, 0.1))


class HumanAgent(TeamMember):
    """Human team member."""

    def __init__(self, name: str, expertise: List[str]):
        """Initialize human agent."""
        super().__init__(name, AgentType.HUMAN, expertise)
        print(f"👤 Human member '{name}' joined (expertise: {', '.join(expertise)})")

    def _generate_contribution(
        self,
        task: str,
        context: SharedContext
    ) -> Dict[str, Any]:
        """Generate human contribution."""
        return {
            "type": "human",
            "expertise_applied": self.capabilities,
            "output": f"Human insight on: {task}",
            "creativity_score": random.uniform(0.7, 1.0)
        }


class AIAgent(TeamMember):
    """AI team member."""

    def __init__(self, name: str, capabilities: List[str]):
        """Initialize AI agent."""
        super().__init__(name, AgentType.AI, capabilities)
        print(f"🤖 AI member '{name}' joined (capabilities: {', '.join(capabilities)})")

    def _generate_contribution(
        self,
        task: str,
        context: SharedContext
    ) -> Dict[str, Any]:
        """Generate AI contribution."""
        # AI can access previous contributions
        previous = len(context.contributions)

        return {
            "type": "ai",
            "capabilities_used": self.capabilities,
            "output": f"AI analysis of: {task}",
            "built_on_previous": previous,
            "confidence": random.uniform(0.75, 0.95)
        }


class CollaborativeTeam:
    """Collaborative team with human and AI members."""

    def __init__(self, name: str = "CollabTeam"):
        """Initialize collaborative team."""
        self.name = name
        self.members: List[TeamMember] = []
        self.shared_context: Optional[SharedContext] = None
        self.completed_tasks: List[Dict] = []

        print(f"🤝 Collaborative Team '{name}' created")

    def add_member(self, member: TeamMember) -> None:
        """Add member to team."""
        self.members.append(member)
        print(f"   ✓ {member.name} added to team")

    def execute_collaborative_task(
        self,
        task: str,
        strategy: str = "parallel",
        requirements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute task collaboratively."""
        print(f"\n{'='*60}")
        print(f"Collaborative Task: {task}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        # Initialize shared context
        self.shared_context = SharedContext(
            task_description=task,
            goal=f"Complete: {task}"
        )

        # Execute based on strategy
        if strategy == "sequential":
            result = self._execute_sequential(requirements or [])
        elif strategy == "parallel":
            result = self._execute_parallel(requirements or [])
        elif strategy == "iterative":
            result = self._execute_iterative(requirements or [])
        else:
            result = self._execute_parallel(requirements or [])

        # Store completed task
        self.completed_tasks.append({
            "task": task,
            "strategy": strategy,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

        print(f"\n✓ Task completed!")
        print(f"  Contributions: {len(result['contributions'])}")
        print(f"  Quality: {result['overall_quality']:.2f}")

        return result

    def _execute_sequential(
        self,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Execute task sequentially."""
        print("\n→ Sequential execution")

        task_id = f"task_{len(self.completed_tasks) + 1}"
        contributions = []

        # Each member contributes in order
        for member in self.members:
            if not requirements or member.can_contribute(requirements):
                contribution = member.contribute(
                    task_id,
                    self.shared_context.task_description,
                    self.shared_context
                )
                contributions.append(contribution)
                self.shared_context.contributions.append(contribution)

        return self._aggregate_results(contributions)

    def _execute_parallel(
        self,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Execute task in parallel."""
        print("\n⇉ Parallel execution")

        task_id = f"task_{len(self.completed_tasks) + 1}"
        contributions = []

        # All members contribute simultaneously
        for member in self.members:
            if not requirements or member.can_contribute(requirements):
                contribution = member.contribute(
                    task_id,
                    self.shared_context.task_description,
                    self.shared_context
                )
                contributions.append(contribution)

        # Update shared context after all contributions
        self.shared_context.contributions.extend(contributions)

        return self._aggregate_results(contributions)

    def _execute_iterative(
        self,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Execute task iteratively with refinement."""
        print("\n🔄 Iterative execution")

        task_id = f"task_{len(self.completed_tasks) + 1}"
        all_contributions = []

        # Multiple rounds
        for round_num in range(2):
            print(f"\n  Round {round_num + 1}")

            round_contributions = []
            for member in self.members:
                if not requirements or member.can_contribute(requirements):
                    contribution = member.contribute(
                        task_id,
                        self.shared_context.task_description,
                        self.shared_context
                    )
                    round_contributions.append(contribution)
                    self.shared_context.contributions.append(contribution)

            all_contributions.extend(round_contributions)

        return self._aggregate_results(all_contributions)

    def _aggregate_results(
        self,
        contributions: List[Contribution]
    ) -> Dict[str, Any]:
        """Aggregate contributions into final result."""
        if not contributions:
            return {
                "contributions": [],
                "overall_quality": 0.0,
                "status": "no_contributions"
            }

        overall_quality = sum(
            c.quality_score for c in contributions
        ) / len(contributions)

        return {
            "contributions": [
                {
                    "member": c.member_name,
                    "quality": c.quality_score,
                    "content": c.content
                }
                for c in contributions
            ],
            "overall_quality": overall_quality,
            "total_contributors": len(set(c.member_name for c in contributions)),
            "status": "completed"
        }

    def resolve_conflict(
        self,
        conflicting_outputs: List[Any]
    ) -> Dict[str, Any]:
        """Resolve conflicts between contributions."""
        print(f"\n⚖️  Resolving conflict between {len(conflicting_outputs)} outputs")

        # Simple voting mechanism
        votes = {i: 0 for i in range(len(conflicting_outputs))}

        for member in self.members:
            # Each member votes
            vote = random.randint(0, len(conflicting_outputs) - 1)
            votes[vote] += 1

        winner_idx = max(votes, key=votes.get)

        resolution = {
            "winning_output": conflicting_outputs[winner_idx],
            "votes": votes,
            "resolution_method": "democratic_vote",
            "confidence": votes[winner_idx] / len(self.members)
        }

        print(f"   ✓ Conflict resolved (confidence: {resolution['confidence']:.2f})")

        return resolution

    def get_contribution_report(self) -> Dict[str, Any]:
        """Generate report of team contributions."""
        print("\n📊 Generating contribution report...")

        member_stats = {}

        for member in self.members:
            total_contributions = len(member.contributions)
            avg_quality = (
                sum(c.quality_score for c in member.contributions) / total_contributions
                if total_contributions > 0 else 0
            )

            member_stats[member.name] = {
                "type": member.agent_type.value,
                "total_contributions": total_contributions,
                "average_quality": round(avg_quality, 2),
                "capabilities": member.capabilities
            }

        report = {
            "team_name": self.name,
            "team_size": len(self.members),
            "completed_tasks": len(self.completed_tasks),
            "member_stats": member_stats
        }

        print(f"✓ Report generated")
        return report


def demo():
    """Demonstrate collaborative agents framework."""
    print("=" * 60)
    print("Collaborative Agents Framework Demo")
    print("=" * 60)

    # Create team
    team = CollaborativeTeam(name="ProductTeam")

    # Add members
    human = HumanAgent(name="Alice", expertise=["strategy", "design"])
    ai_analyst = AIAgent(name="DataBot", capabilities=["analysis", "data_processing"])
    ai_writer = AIAgent(name="WriteBot", capabilities=["writing", "summarization"])

    team.add_member(human)
    team.add_member(ai_analyst)
    team.add_member(ai_writer)

    # Execute collaborative tasks
    tasks = [
        ("Create quarterly business report", "sequential"),
        ("Analyze customer feedback", "parallel"),
        ("Design new product feature", "iterative")
    ]

    for task, strategy in tasks:
        result = team.execute_collaborative_task(task, strategy=strategy)

    # Get contribution report
    print("\n" + "=" * 60)
    print("Contribution Report")
    print("=" * 60)

    import json
    report = team.get_contribution_report()
    print(json.dumps(report, indent=2))

    # Resolve conflict
    print("\n" + "=" * 60)
    print("Conflict Resolution")
    print("=" * 60)

    resolution = team.resolve_conflict(
        conflicting_outputs=["Option A", "Option B", "Option C"]
    )
    print(f"Winner: {resolution['winning_output']}")


if __name__ == "__main__":
    demo()
