"""
Social Agents - Social Behavior and Teamwork
=============================================

Social interaction models, team formation, role assignment, and
collaborative behavior for multi-agent systems.

Author: Brill Consulting
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import random


class Role(Enum):
    """Agent roles in team."""
    LEADER = "leader"
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"


@dataclass
class SocialAgent:
    """Agent with social capabilities."""
    name: str
    role: Role = Role.WORKER
    skills: Set[str] = field(default_factory=set)
    relationships: Dict[str, float] = field(default_factory=dict)  # trust scores
    reputation: float = 5.0
    tasks_completed: int = 0
    collaborations: List[str] = field(default_factory=list)

    def trust(self, other_agent: str) -> float:
        """Get trust level for another agent."""
        return self.relationships.get(other_agent, 0.5)

    def update_trust(self, other_agent: str, delta: float):
        """Update trust in another agent."""
        current = self.relationships.get(other_agent, 0.5)
        self.relationships[other_agent] = max(0.0, min(1.0, current + delta))

    def has_skill(self, skill: str) -> bool:
        """Check if agent has required skill."""
        return skill in self.skills

    def add_collaboration(self, agent_name: str):
        """Record collaboration."""
        if agent_name not in self.collaborations:
            self.collaborations.append(agent_name)

    def increase_reputation(self, amount: float = 0.1):
        """Increase reputation score."""
        self.reputation = min(10.0, self.reputation + amount)

    def decrease_reputation(self, amount: float = 0.1):
        """Decrease reputation score."""
        self.reputation = max(0.0, self.reputation - amount)


@dataclass
class Team:
    """Team of social agents."""
    name: str
    agents: List[SocialAgent] = field(default_factory=list)
    leader: Optional[str] = None
    shared_goals: List[str] = field(default_factory=list)
    shared_knowledge: Dict[str, any] = field(default_factory=dict)

    def add_agent(self, agent: SocialAgent):
        """Add agent to team."""
        self.agents.append(agent)
        if agent.role == Role.LEADER and not self.leader:
            self.leader = agent.name

    def remove_agent(self, agent_name: str):
        """Remove agent from team."""
        self.agents = [a for a in self.agents if a.name != agent_name]
        if self.leader == agent_name:
            self.leader = None

    def get_agent(self, name: str) -> Optional[SocialAgent]:
        """Get agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def find_agents_with_skill(self, skill: str) -> List[SocialAgent]:
        """Find all agents with specific skill."""
        return [a for a in self.agents if a.has_skill(skill)]

    def get_team_skills(self) -> Set[str]:
        """Get all skills available in team."""
        skills = set()
        for agent in self.agents:
            skills.update(agent.skills)
        return skills

    def share_knowledge(self, key: str, value: any):
        """Add to shared team knowledge."""
        self.shared_knowledge[key] = value

    def get_average_reputation(self) -> float:
        """Get team's average reputation."""
        if not self.agents:
            return 0.0
        return sum(a.reputation for a in self.agents) / len(self.agents)


class TeamFormation:
    """Algorithms for forming effective teams."""

    @staticmethod
    def form_skill_based_team(agents: List[SocialAgent], required_skills: Set[str],
                               max_size: int = 5) -> Optional[Team]:
        """Form team based on required skills."""
        # Find agents with required skills
        suitable_agents = []
        for agent in agents:
            if any(agent.has_skill(skill) for skill in required_skills):
                suitable_agents.append(agent)

        # Sort by reputation
        suitable_agents.sort(key=lambda a: a.reputation, reverse=True)

        # Select best agents up to max_size
        selected = suitable_agents[:max_size]

        # Check if all skills covered
        team_skills = set()
        for agent in selected:
            team_skills.update(agent.skills)

        if not required_skills.issubset(team_skills):
            return None  # Cannot form team with required skills

        team = Team(name=f"team_{random.randint(1000, 9999)}")
        for agent in selected:
            team.add_agent(agent)

        return team

    @staticmethod
    def assign_roles(team: Team):
        """Assign roles to team members."""
        if not team.agents:
            return

        # Assign leader to highest reputation agent
        leader = max(team.agents, key=lambda a: a.reputation)
        leader.role = Role.LEADER
        team.leader = leader.name

        # Assign specialists to agents with unique skills
        all_skills = {}
        for agent in team.agents:
            for skill in agent.skills:
                if skill not in all_skills:
                    all_skills[skill] = []
                all_skills[skill].append(agent)

        for skill, agents_list in all_skills.items():
            if len(agents_list) == 1:
                agents_list[0].role = Role.SPECIALIST

        # Assign coordinator to second highest reputation
        non_leaders = [a for a in team.agents if a.role not in [Role.LEADER, Role.SPECIALIST]]
        if non_leaders:
            coordinator = max(non_leaders, key=lambda a: a.reputation)
            coordinator.role = Role.COORDINATOR


class CollaborationProtocol:
    """Protocols for agent collaboration."""

    @staticmethod
    def coordinate_task(team: Team, task: str, required_skills: Set[str]) -> Dict:
        """Coordinate task execution among team."""
        # Find suitable agents
        assigned_agents = []
        for skill in required_skills:
            agents_with_skill = team.find_agents_with_skill(skill)
            if agents_with_skill:
                # Pick highest reputation agent
                best_agent = max(agents_with_skill, key=lambda a: a.reputation)
                if best_agent not in assigned_agents:
                    assigned_agents.append(best_agent)

        # Record collaborations
        for i, agent1 in enumerate(assigned_agents):
            for agent2 in assigned_agents[i+1:]:
                agent1.add_collaboration(agent2.name)
                agent2.add_collaboration(agent1.name)

        return {
            'task': task,
            'assigned': [a.name for a in assigned_agents],
            'skills_covered': {skill for a in assigned_agents for skill in a.skills if skill in required_skills}
        }

    @staticmethod
    def build_trust_network(team: Team):
        """Build trust relationships in team."""
        for agent1 in team.agents:
            for agent2 in team.agents:
                if agent1.name != agent2.name:
                    # Base trust on reputation
                    base_trust = (agent2.reputation / 10.0) * 0.5

                    # Increase trust if collaborated before
                    if agent2.name in agent1.collaborations:
                        base_trust += 0.3

                    # Random variation
                    trust = min(1.0, base_trust + random.uniform(-0.1, 0.1))

                    agent1.relationships[agent2.name] = trust


def demo():
    """Demonstration of Social Agents."""
    print("Social Agents - Social Behavior and Teamwork Demo")
    print("=" * 70)

    # Create agents
    print("\n1️⃣  Creating Social Agents")
    print("-" * 70)

    agents = [
        SocialAgent("Alice", skills={"python", "leadership"}, reputation=8.5),
        SocialAgent("Bob", skills={"python", "testing"}, reputation=7.0),
        SocialAgent("Carol", skills={"design", "ui"}, reputation=7.5),
        SocialAgent("Dave", skills={"devops", "docker"}, reputation=6.5),
        SocialAgent("Eve", skills={"python", "data_science"}, reputation=8.0),
    ]

    for agent in agents:
        print(f"  • {agent.name}: {agent.skills} (reputation: {agent.reputation})")

    # Form team
    print("\n2️⃣  Forming Team")
    print("-" * 70)

    required_skills = {"python", "testing", "devops"}
    team = TeamFormation.form_skill_based_team(agents, required_skills, max_size=4)

    if team:
        print(f"✓ Team formed: {team.name}")
        print(f"Members: {[a.name for a in team.agents]}")
        print(f"Team skills: {team.get_team_skills()}")
        print(f"Average reputation: {team.get_average_reputation():.2f}")
    else:
        print("✗ Could not form team")

    # Assign roles
    print("\n3️⃣  Assigning Roles")
    print("-" * 70)

    TeamFormation.assign_roles(team)
    for agent in team.agents:
        print(f"  {agent.name}: {agent.role.value}")

    # Build trust network
    print("\n4️⃣  Building Trust Network")
    print("-" * 70)

    CollaborationProtocol.build_trust_network(team)
    print("Trust relationships:")
    for agent in team.agents[:2]:
        print(f"\n  {agent.name}'s trust:")
        for other, trust in sorted(agent.relationships.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    {other}: {trust:.2f}")

    # Coordinate task
    print("\n5️⃣  Coordinating Task")
    print("-" * 70)

    task_result = CollaborationProtocol.coordinate_task(
        team,
        "Build CI/CD pipeline",
        {"python", "testing", "devops"}
    )

    print(f"Task: {task_result['task']}")
    print(f"Assigned agents: {task_result['assigned']}")
    print(f"Skills covered: {task_result['skills_covered']}")

    # Update after collaboration
    for agent_name in task_result['assigned']:
        agent = team.get_agent(agent_name)
        if agent:
            agent.tasks_completed += 1
            agent.increase_reputation(0.2)

    print("\n6️⃣  Post-Collaboration Stats")
    print("-" * 70)

    for agent in team.agents:
        collab_count = len(agent.collaborations)
        print(f"  {agent.name}: {agent.tasks_completed} tasks, "
              f"{collab_count} collaborations, reputation: {agent.reputation:.2f}")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
