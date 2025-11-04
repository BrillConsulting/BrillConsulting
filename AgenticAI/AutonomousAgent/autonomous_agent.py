"""
Autonomous Agent Framework
==========================

Self-directed agent with planning, reasoning, and execution capabilities:
- Goal-oriented planning
- Dynamic task decomposition
- Action execution
- Self-reflection and learning
- Error recovery

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Callable
from datetime import datetime
import json


class AutonomousAgent:
    """Self-directed autonomous agent."""

    def __init__(self, name: str = "Agent", tools: Optional[Dict[str, Callable]] = None):
        """Initialize autonomous agent."""
        self.name = name
        self.tools = tools or {}
        self.memory = []
        self.goals = []
        self.current_plan = []
        self.execution_history = []

    def add_tool(self, name: str, function: Callable, description: str):
        """Register a tool for the agent to use."""
        self.tools[name] = {
            "function": function,
            "description": description
        }
        print(f"✓ Tool registered: {name}")

    def set_goal(self, goal: str, context: Optional[Dict] = None):
        """Set a goal for the agent."""
        goal_obj = {
            "id": len(self.goals) + 1,
            "description": goal,
            "context": context or {},
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        self.goals.append(goal_obj)
        print(f"✓ Goal set: {goal}")
        return goal_obj

    def plan(self, goal: Dict) -> List[Dict]:
        """Create a plan to achieve the goal."""
        print(f"\n🤔 Planning for goal: {goal['description']}")

        # Simplified planning: decompose goal into steps
        # In real implementation, this would use LLM for planning
        plan = [
            {
                "step": 1,
                "action": "analyze_goal",
                "description": f"Analyze requirements for: {goal['description']}",
                "status": "pending"
            },
            {
                "step": 2,
                "action": "gather_resources",
                "description": "Identify and prepare necessary resources",
                "status": "pending"
            },
            {
                "step": 3,
                "action": "execute_main_task",
                "description": "Execute the main task",
                "status": "pending"
            },
            {
                "step": 4,
                "action": "verify_result",
                "description": "Verify the outcome meets goal criteria",
                "status": "pending"
            }
        ]

        self.current_plan = plan
        print(f"✓ Created plan with {len(plan)} steps")
        return plan

    def execute_step(self, step: Dict) -> Dict:
        """Execute a single plan step."""
        print(f"\n▶ Executing step {step['step']}: {step['description']}")

        result = {
            "step": step["step"],
            "action": step["action"],
            "started_at": datetime.now().isoformat(),
            "status": "success",
            "output": None,
            "error": None
        }

        try:
            # Check if action maps to a tool
            if step["action"] in self.tools:
                tool = self.tools[step["action"]]
                result["output"] = tool["function"]()
            else:
                # Simulate action execution
                result["output"] = f"Completed: {step['description']}"

            result["completed_at"] = datetime.now().isoformat()
            print(f"✓ Step {step['step']} completed")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"✗ Step {step['step']} failed: {e}")

        self.execution_history.append(result)
        return result

    def reflect(self, results: List[Dict]) -> Dict:
        """Reflect on execution results and learn."""
        print("\n🧠 Reflecting on execution...")

        total_steps = len(results)
        successful_steps = sum(1 for r in results if r["status"] == "success")
        failed_steps = total_steps - successful_steps

        reflection = {
            "timestamp": datetime.now().isoformat(),
            "total_steps": total_steps,
            "successful": successful_steps,
            "failed": failed_steps,
            "success_rate": successful_steps / total_steps * 100 if total_steps > 0 else 0,
            "learnings": []
        }

        # Analyze failures
        if failed_steps > 0:
            reflection["learnings"].append({
                "type": "failure_analysis",
                "insight": f"Encountered {failed_steps} failures, need to improve error handling"
            })

        # Store reflection in memory
        self.memory.append({
            "type": "reflection",
            "data": reflection
        })

        print(f"✓ Reflection complete: {successful_steps}/{total_steps} steps successful")
        return reflection

    def run(self, goal: str, context: Optional[Dict] = None) -> Dict:
        """Run the complete agent cycle: plan -> execute -> reflect."""
        print(f"\n{'='*60}")
        print(f"🤖 {self.name} Starting Autonomous Execution")
        print(f"{'='*60}")

        # Set goal
        goal_obj = self.set_goal(goal, context)

        # Plan
        plan = self.plan(goal_obj)

        # Execute
        print(f"\n{'─'*60}")
        print("⚙️  Execution Phase")
        print(f"{'─'*60}")

        results = []
        for step in plan:
            result = self.execute_step(step)
            results.append(result)

            # Stop if critical failure
            if result["status"] == "failed" and step.get("critical", False):
                print(f"\n⚠️  Critical failure in step {step['step']}, stopping execution")
                break

        # Reflect
        reflection = self.reflect(results)

        # Update goal status
        all_success = all(r["status"] == "success" for r in results)
        goal_obj["status"] = "completed" if all_success else "failed"

        summary = {
            "goal": goal_obj,
            "plan": plan,
            "execution_results": results,
            "reflection": reflection,
            "final_status": goal_obj["status"]
        }

        print(f"\n{'='*60}")
        print(f"✓ Autonomous Execution Complete: {goal_obj['status'].upper()}")
        print(f"{'='*60}\n")

        return summary

    def get_memory(self) -> List[Dict]:
        """Retrieve agent's memory."""
        return self.memory

    def get_status(self) -> Dict:
        """Get current agent status."""
        return {
            "name": self.name,
            "total_goals": len(self.goals),
            "completed_goals": sum(1 for g in self.goals if g["status"] == "completed"),
            "active_tools": len(self.tools),
            "memory_items": len(self.memory),
            "execution_history": len(self.execution_history)
        }


def demo():
    """Demo autonomous agent."""
    print("Autonomous Agent Demo")
    print("=" * 60)

    # Create agent
    agent = AutonomousAgent(name="AutoBot")

    # Register some tools
    def analyze_data():
        return "Data analysis completed: 1000 records processed"

    def generate_report():
        return "Report generated: insights.pdf"

    def send_notification():
        return "Notification sent to stakeholders"

    agent.add_tool("analyze_data", analyze_data, "Analyze dataset")
    agent.add_tool("generate_report", generate_report, "Generate analysis report")
    agent.add_tool("send_notification", send_notification, "Send notifications")

    # Run autonomous execution
    result = agent.run(
        goal="Analyze customer data and deliver insights report",
        context={"dataset": "customers.csv", "deadline": "2024-01-15"}
    )

    # Show agent status
    print("\n📊 Agent Status:")
    print("-" * 60)
    status = agent.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Show reflection insights
    if result["reflection"]["learnings"]:
        print("\n💡 Learnings:")
        print("-" * 60)
        for learning in result["reflection"]["learnings"]:
            print(f"  • {learning['insight']}")

    print("\n✓ Autonomous Agent Demo Complete!")


if __name__ == '__main__':
    demo()
